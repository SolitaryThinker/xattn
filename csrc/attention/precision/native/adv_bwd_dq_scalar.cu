// Native SM103 advanced (level 2-4) reader backward: row-owned dQ and row sinks,
// correctness-first schedule (no tensor cores).
//
// One CTA owns one query row, walks its admitted keys in ascending order in FP32
// and writes dq and the four statistics planes from that single owner, so no
// atomics are needed anywhere and ordered mode repeats bit-exactly.  This is the
// schedule the tiled kernel (adv_bwd_dq.cu) is measured against and the contract
// instrument that lets the native advanced path exist before the tiled kernels
// do; it is deliberately simple rather than fast.
//
// The key-owned dK/dV half is a separate translation unit and this file
// produces no dK/dV, so nothing here can be mistaken for a complete backward.
//
// NOT ON THE ROUTE.  `Launch()` (adv_bwd.cu) chains LaunchMainDq (the tiled leaf)
// and LaunchDkdv only; nothing in this tree calls LaunchMainDqScalar, and this
// file is on no setup.py source list.  It is a bench-only reduced-algebra arm:
// it writes the same four statistics planes as the tiled leaf, behind the same
// level/ordered gate, but it exists to be measured AGAINST that leaf, not to be
// shipped in its place.  A caller wanting dQ for the route takes the tiled leaf.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "attention/precision/native/adv_bwd_math.cuh"
#include "attention/precision/native/adv_bwd_native.h"

namespace xattn {
namespace ops {
namespace native_adv {
namespace {

namespace bm = ::xattn::ops::native_adv::bwdmath;

constexpr int kThreads = 128;
constexpr int kMaxSlots = 2;  // D <= 256, so a thread owns at most two columns

__device__ __forceinline__ float ld_elem(const void* base, int64_t off, Elem e) {
  if (e == Elem::kFp16) return __half2float(*((const __half*)base + off));
  return __bfloat162float(*((const __nv_bfloat16*)base + off));
}

__device__ __forceinline__ void st_elem(void* base, int64_t off, float x, Elem e) {
  if (e == Elem::kFp16) {
    *((__half*)base + off) = __float2half_rn(x);
  } else {
    *((__nv_bfloat16*)base + off) = __float2bfloat16_rn(x);
  }
}

// Tree reduce that BROADCASTS: every thread must see the row total, because each
// thread forms its own dS and dQ columns from it (a reduce that leaves the total
// on lane 0 only would make the sink right and every dQ column wrong).
__device__ __forceinline__ float block_reduce(float v, float* scratch) {
#pragma unroll
  for (int off = kThreads / 2; off > 0; off >>= 1) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x < off) v += scratch[threadIdx.x + off];
    __syncthreads();
  }
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  return scratch[0];
}

__global__ void __launch_bounds__(kThreads) dq_row_kernel(const Params p) {
  __shared__ float scratch[kThreads];
  __shared__ float s_m, s_lz2, s_anchor_dp, s_delta, s_two_e;
  __shared__ int s_live;

  // Planes are [B, Hq, Nq] with row stride 1 and head stride rowstat_head_stride,
  // and every tensor stride in Params is in ELEMENTS (the host fills them from
  // torch stride()), so all addressing below is element arithmetic.
  const int64_t head_stride = p.rowstat_head_stride;
  const int64_t count = p.B * p.Hq * head_stride;
  const int64_t blk = blockIdx.x;
  if (blk >= p.B * p.Hq * p.Nq) return;
  const int64_t i = blk % p.Nq;
  const int64_t bh = blk / p.Nq;
  const int64_t hq = bh % p.Hq;
  const int64_t b = bh / p.Hq;
  const int64_t ci = (b * p.Hq + hq) * head_stride + i;
  const int64_t hkv = hq / p.qk_ratio;

  const int64_t q_off = b * p.q_batch_stride + i * p.q_row_stride + hq * p.q_head_stride;
  const int64_t dy_off = b * p.dy_batch_stride + i * p.dy_row_stride + hq * p.dy_head_stride;
  const int64_t kbase = b * p.k_batch_stride + hkv * p.k_head_stride;
  const int64_t vbase = b * p.v_batch_stride + hkv * p.v_head_stride;
  const int64_t dq_off = b * p.dq_batch_stride + i * p.dq_row_stride + hq * p.dq_head_stride;
  // dQ is FP32 by contract (adv_bwd_native.h: the caller narrows), so it is
  // stored as float here; only K/V operands are element-typed.
  float* __restrict__ dq_row = reinterpret_cast<float*>(p.dq) + dq_off;

  const int64_t D = p.D, Dv = p.Dv;
  const float anchor_index = p.meta[3 * count + ci];
  if (threadIdx.x == 0) {
    const float lz = p.logz[ci];
    // Liveness is decided from saved state before any exp.
    s_live = bm::row_is_live(anchor_index, lz) ? 1 : 0;
    s_m = p.meta[0 * count + ci];
    s_lz2 = bm::lz2_from_logz(lz);
    s_anchor_dp = p.meta[1 * count + ci];
    s_delta = p.delta[ci];
    s_two_e = bm::two_e_of(p.meta[2 * count + ci], p.level);
  }
  __syncthreads();

  if (!s_live) {
    for (int64_t d = threadIdx.x; d < D; d += kThreads) dq_row[d] = 0.0f;
    if (threadIdx.x == 0) {
      bm::SinkRow sink;
      const int e = p.rho_exponent ? p.rho_exponent[ci] : 0;
      sink.init(p.level, s_two_e, p.two_component_dv, p.ordered, e, p.sink_status);
      sink.flush_tile();  // zero partial -> zero totals
      sink.store((double*)p.stats, ci, count);
    }
    return;
  }

  const int64_t jmax = i + p.offset - (p.strict ? 1 : 0);
  const bool seg = p.q_seg != nullptr;
  const int64_t qseg = seg ? p.q_seg[b * p.Nq + i] : 0;
  const int64_t anchor_col = (int64_t)anchor_index;

  float acc[kMaxSlots];
#pragma unroll
  for (int s = 0; s < kMaxSlots; ++s) acc[s] = 0.0f;

  bm::SinkRow sink;
  if (threadIdx.x == 0) {
    const int e = p.rho_exponent ? p.rho_exponent[ci] : 0;
    sink.init(p.level, s_two_e, p.two_component_dv, p.ordered, e, p.sink_status);
  }

  for (int64_t j = 0; j < p.Nt && j <= jmax; ++j) {
    if (seg && p.k_seg[b * p.Nt + j] != qseg) {
      // A physical key tile ends here whether or not this key is admitted; the
      // ordered grid is defined per physical tile (:1982-2005), so skipping the
      // flush would fold two tiles into one rounded partial.
      if (threadIdx.x == 0 && ((j + 1) & (bm::kSinkTile - 1)) == 0) sink.flush_tile();
      continue;
    }
    const int64_t k_off = kbase + j * p.k_row_stride;
    const int64_t v_off = vbase + j * p.v_row_stride;

    float s_part = 0.0f, dp_part = 0.0f;
    for (int64_t d = threadIdx.x; d < D; d += kThreads)
      s_part += ld_elem(p.q, q_off + d, p.elem) * ld_elem(p.k, k_off + d, p.elem);
    for (int64_t d = threadIdx.x; d < Dv; d += kThreads)
      dp_part += ld_elem(p.dy, dy_off + d, p.elem) * ld_elem(p.v, v_off + d, p.elem);
    const float s = block_reduce(s_part, scratch);
    const float dp = block_reduce(dp_part, scratch);

    const float ph = bm::p_half(s, s_m, s_lz2, p.softmax_scale_log2);
    const float ds = bm::ds_from(ph, dp, s_two_e, s_anchor_dp, s_delta);
    const float pop = bm::dv_operand(ph, s_two_e);  // P*2^e, the pinned mass unit
    // Same convention as the tiled leaf, so the two agree plane for plane: dS
    // is split into the element-rounded head plus its residual; below level 4
    // both the dQ contraction and the sink take the head alone (the pin's sink
    // sums h = float(hirow) with no lorow, :1978-1989), and at level 4 both take
    // hi + lo, where the pin adds lorow.
    float ds_hi, ds_lo, p_hi, p_lo;
    bm::elem_split(ds, p.elem, ds_hi, ds_lo);
    if (p.level != 4) ds_lo = 0.0f;
    bm::elem_split(pop, p.elem, p_hi, p_lo);
    if (!p.two_component_dv) p_lo = 0.0f;
    {
      const float dq_hi = ds_hi, dq_lo = ds_lo;
      if (threadIdx.x == 0)
        sink.add(dq_hi, p.level == 4 ? dq_lo : 0.0f, p_hi, p_lo, j == anchor_col);
      int slot = 0;
      for (int64_t d = threadIdx.x; d < D; d += kThreads, ++slot) {
        const float kc = ld_elem(p.k, k_off + d, p.elem);
        acc[slot] += dq_hi * kc;
        if (p.level == 4) acc[slot] += dq_lo * kc;
      }
    }
    // Ordered grids are defined per key tile, not per row: the pin rounds each
    // key-block partial onto the grid (:1982-2005) and so does the tiled leaf,
    // so this arm must round at the same boundaries or a row crossing one gets
    // a different fixed-point value for the same float partial.
    if (threadIdx.x == 0 && ((j + 1) & (bm::kSinkTile - 1)) == 0) sink.flush_tile();
  }

  {
    int slot = 0;
    for (int64_t d = threadIdx.x; d < D; d += kThreads, ++slot)
      dq_row[d] = p.softmax_scale * acc[slot];
  }

  if (threadIdx.x == 0) {
    sink.flush_tile();
    sink.store((double*)p.stats, ci, count);
  }
}

}  // namespace

// Row-owned entry: writes dq and the statistics planes, nothing else.
void LaunchMainDqScalar(const Params& p, cudaStream_t stream) {
  const int64_t rows = p.B * p.Hq * p.Nq;
  if (rows == 0) return;
  dq_row_kernel<<<(unsigned)rows, kThreads, 0, stream>>>(p);
}

}  // namespace native_adv
}  // namespace ops
}  // namespace xattn
