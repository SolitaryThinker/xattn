// Native SM103 advanced (level 2-4) reader backward: row-owned dQ leaf.
//
// One CTA per (b, hq, query tile of kQueryRows = 32 rows).  One warp owns one
// query row, walks the admitted keys in ascending order, and produces
//   dQ_ic = softmax_scale * sum_j dS_ij K_jc
// plus the row's four statistics sinks.  Nothing is atomic: a query row and its
// sinks are written by exactly one warp, once, so Main and ordered mode are both
// deterministic by construction (ordered mode only changes how the sink
// partials are rounded and stored).
//
// dP is recomputed here: dP_ij = sum_c dY_ic V_jc, then
// dS_ij = P_half * (P_half * ((dP_ij * 2^e - a_i) - delta_i)).  dY arrives
// already prescaled by 2^-e (adv_bwd_native.h item 3), which is what makes the
// *2^e restore dP and what keeps dV = P_op dY in unscaled dY units, so this
// leaf never rescales dY itself.  At levels 2/3 the plane is absent and
// 2^e = 2^-e = 1 exactly.
//
// Algebra, masks, plane order and grids: adv_bwd_native.h items 1-7 (with the
// pinned Hopper line numbers) and adv_bwd_math.cuh, which both leaves include so
// they cannot disagree.  Full-causal
// visibility walks keys 0..i+offset-strict; sliding-chunk visibility walks the
// row statistics' [first, last] admitted keys. Window visibility is rejected by
// the host half before reaching this leaf.
//
// This translation unit defines no Launch(): the aggregate entry chains this
// kernel with the key-owned dK/dV leaf.
#include "attention/precision/native/adv_bwd_native.h"
#include "attention/precision/native/adv_bwd_math.cuh"
// One definition of the level algebra, the row sink grid and the admitted-key
// walk for every family that runs these levels; this leaf is its first user.
#include "flash_sca/native/sca_precision.cuh"

#include <c10/cuda/CUDAException.h>  // C10_CUDA_KERNEL_LAUNCH_CHECK()
#include <c10/util/Exception.h>      // TORCH_CHECK

namespace xattn {
namespace ops {
namespace native_adv {
namespace {

namespace prec = ::xattn::ops::sca_prec;

// 32 query rows per CTA, one warp per row, so the admission test and every
// warp reduction are uniform across a row's 32 lanes.
constexpr int kQueryRows = 32;
constexpr int kThreads = 32 * kQueryRows;
// D and Dv are runtime widths, at most 256 (host check): 8 lane strides of 32.
constexpr int kChunks = 8;
static_assert(bwdmath::kSinkTile == 32, "the sink tile flush uses a power-of-two mask");

__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

// Full butterfly: every lane ends with the same bits, so everything derived
// from the score and dP sums is identical in all 32 lanes of a row.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <typename T>
__global__ void __launch_bounds__(kThreads) DqKernel(Params p) {
  const int lane = threadIdx.x & 31;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kQueryRows + (threadIdx.x >> 5);
  if (i >= p.Nq) return;  // no __syncthreads() in this kernel: early exit is safe
  const int64_t b = blockIdx.z;
  const int64_t hq = blockIdx.y;
  const int64_t hk = hq / p.qk_ratio;
  const int D = static_cast<int>(p.D);
  const int Dv = static_cast<int>(p.Dv);
  const int nch = (D + 31) >> 5;
  const int nchv = (Dv + 31) >> 5;

  // Planes are [B, Hq, Nq] with row stride 1 and head stride Nq.
  const int64_t count = p.B * p.Hq * p.rowstat_head_stride;
  const int64_t row = (b * p.Hq + hq) * p.rowstat_head_stride + i;

  const float scale_log2 = p.softmax_scale_log2;
  const float row_max = p.meta[row];
  const float anchor_dp = p.meta[count + row];
  const float two_e = bwdmath::two_e_of(p.meta[2 * count + row], static_cast<int>(p.level));
  const float anchor_col = p.meta[3 * count + row];  // -1 iff the admitted set is empty
  const float logz_natural = p.logz[row];
  const float lz2 = bwdmath::lz2_from_logz(logz_natural);
  const float delta = p.delta[row];

  const bool level4 = p.level == 4;

  const T* __restrict__ q_row = reinterpret_cast<const T*>(p.q) + b * p.q_batch_stride +
                                i * p.q_row_stride + hq * p.q_head_stride;
  const T* __restrict__ dy_row = reinterpret_cast<const T*>(p.dy) + b * p.dy_batch_stride +
                                 i * p.dy_row_stride + hq * p.dy_head_stride;
  float* __restrict__ dq_row =
      reinterpret_cast<float*>(p.dq) + b * p.dq_batch_stride + i * p.dq_row_stride +
      hq * p.dq_head_stride;

  // Q and dY rows stay in registers across the whole key walk: they are
  // loop-invariant and the key walk is the only loop.
  float q_reg[kChunks];
  float dy_reg[kChunks];
  float acc[kChunks];
#pragma unroll
  for (int t = 0; t < kChunks; ++t) {
    const int c = lane + 32 * t;
    q_reg[t] = (t < nch && c < D) ? to_float(q_row[c]) : 0.0f;
    dy_reg[t] = (t < nchv && c < Dv) ? to_float(dy_row[c]) : 0.0f;
    acc[t] = 0.0f;
  }

  prec::Sink sink;
  prec::SinkInit(sink, p, row, count);

  // Empty rows: meta[3] == -1 and a non-finite logz.  Decided before any exp;
  // such a row (and a reader-2 row whose strict prefix is empty) contributes
  // exactly zero dQ and zero sinks and writes no NaN.  The interval and its
  // start tile are the family's shared definition (sca_precision.cuh:WalkOf),
  // not a local copy, so the level-aware SCA kernels cannot walk differently.
  const prec::ChunkWalk walk = prec::WalkOf(p, i, row, count);
  const int64_t jfirst = walk.jfirst, jmax = walk.jlast, jstart = walk.jstart;
  if (walk.live) {
    const T* __restrict__ k_base =
        reinterpret_cast<const T*>(p.k) + b * p.k_batch_stride + hk * p.k_head_stride;
    const T* __restrict__ v_base =
        reinterpret_cast<const T*>(p.v) + b * p.v_batch_stride + hk * p.v_head_stride;
    for (int64_t j = jstart; j <= jmax; ++j) {
      if (j < jfirst || !prec::SegmentOk(p, b, i, j)) {
        // The ordered fixed-point is defined per PHYSICAL key tile (:1982-2005),
        // so the tile boundary is crossed whether or not this key is admitted:
        // a `continue` that skips the flush would fold two physical tiles into
        // one rounded partial, and the rounding points would then depend on the
        // segment pattern.  Determinism never depended on it (the mask is the
        // same on every repeat), the pinned value does.
        if (((j + 1) & (bwdmath::kSinkTile - 1)) == 0) sink.flush_tile();
        continue;
      }
      const T* __restrict__ k_row = k_base + j * p.k_row_stride;
      const T* __restrict__ v_row = v_base + j * p.v_row_stride;
      float s = 0.0f;   // raw score s_ij = sum_c q_ic k_jc
      float dp = 0.0f;  // dP_ij = sum_c dY_ic v_jc
#pragma unroll
      for (int t = 0; t < kChunks; ++t) {
        const int c = lane + 32 * t;
        const float kc = (t < nch && c < D) ? to_float(k_row[c]) : 0.0f;
        s = fmaf(q_reg[t], kc, s);
        const float vc = (t < nchv && c < Dv) ? to_float(v_row[c]) : 0.0f;
        dp = fmaf(dy_reg[t], vc, dp);
      }
      s = warp_sum(s);
      dp = warp_sum(dp);

      // Steps 1-3 of the note, element for element (:1846-1857, :1900-1930).
      const float p_half = bwdmath::p_half(s, row_max, lz2, scale_log2);
      const float ds = bwdmath::ds_from(p_half, dp, two_e, anchor_dp, delta);
      // Two components are level 4 only (:1930-1937).  One operand serves both
      // the dQ contraction and the sink, because the pin's sink is element typed
      // exactly as its GEMM operands are: the sink constructs rdS as Element
      // (:1936-1937) and at level 3 sums h = float(hirow) with no lorow
      // (:1978-1989), adding lorow only at level 4.  So below level 4 the sink
      // and dQ both take the element head, and at level 4 both take hi + lo.
      // This matters beyond tidiness: zero_sum_repair defines rho as the sum of
      // the dS the contraction actually used (zero_sum_repair.h:6), so a sink
      // fed a different operand makes the repair inconsistent with dQ.
      float dq_hi, dq_lo;
      bwdmath::elem_split(ds, p.elem, dq_hi, dq_lo);
      if (!level4) dq_lo = 0.0f;
      const float p_op = bwdmath::dv_operand(p_half, two_e);
      float p_hi, p_lo;
      bwdmath::elem_split(p_op, p.elem, p_hi, p_lo);
      if (!p.two_component_dv) p_lo = 0.0f;

      // Step 5: this row's rho / mass / variance / anchor probability.
      sink.add(dq_hi, level4 ? dq_lo : 0.0f, p_hi, p_lo,
               static_cast<float>(j) == anchor_col);

      // dQ += dS * K.  Level 4 runs the two element components as the two
      // accumulations the Hopper dQ GEMM runs (:1930-1943).  K is re-read from
      // the lines just touched instead of held in registers: at 1024 threads a
      // per-lane K tile does not fit the 64-register budget.
#pragma unroll
      for (int t = 0; t < kChunks; ++t) {
        const int c = lane + 32 * t;
        const float kc = (t < nch && c < D) ? to_float(k_row[c]) : 0.0f;
        acc[t] = fmaf(dq_hi, kc, acc[t]);
      }
      if (level4) {
#pragma unroll
        for (int t = 0; t < kChunks; ++t) {
          const int c = lane + 32 * t;
          const float kc = (t < nch && c < D) ? to_float(k_row[c]) : 0.0f;
          acc[t] = fmaf(dq_lo, kc, acc[t]);
        }
      }
      if (((j + 1) & (bwdmath::kSinkTile - 1)) == 0) sink.flush_tile();
    }
  }
  sink.flush_tile();  // leftover partial of the last (short) key tile

  // The natural softmax scale multiplies dQ once, after the reduction, in FP32
  // (adv_bwd_native.h item 5; Hopper dK postfactor :2293, dQ :270/:274).
  const float dq_scale = static_cast<float>(p.softmax_scale);
#pragma unroll
  for (int t = 0; t < kChunks; ++t) {
    const int c = lane + 32 * t;
    if (t < nch && c < D) dq_row[c] = acc[t] * dq_scale;
  }
  // Exactly once per row, every plane written (exact zeros for an empty row and
  // for the planes this level does not define).
  if (lane == 0) prec::SinkStore(sink, p.stats, row, count);
}

}  // namespace

// Row-owned entry: dQ plus the four row sinks.  Full-causal reader mask only,
// one reader per call (p.strict selects reader 2): the aggregate Launch calls
// it once per reader, with that reader's q/dY/dQ/statistics head offsets and
// head strides (so the row index uses the launch's own p.Hq).
void LaunchMainDq(const ::xattn::ops::native_adv::Params& p, cudaStream_t stream) {
  TORCH_CHECK(p.visibility == 0 || p.visibility == 2,
              "native advanced dQ leaf implements full-causal and sliding-chunk visibility");
  TORCH_CHECK(p.level >= 2 && p.level <= 4, "native advanced dQ leaf implements levels 2-4");
  TORCH_CHECK(p.D > 0 && p.D <= 256 && p.Dv > 0 && p.Dv <= 256,
              "native advanced dQ leaf requires 0 < D, Dv <= 256");
  TORCH_CHECK(p.qk_ratio > 0 && p.Hq % p.qk_ratio == 0,
              "native advanced dQ leaf requires Hq a multiple of Hkv");
  if (p.B == 0 || p.Hq == 0 || p.Nq == 0) return;
  const dim3 grid(static_cast<unsigned>((p.Nq + kQueryRows - 1) / kQueryRows),
                  static_cast<unsigned>(p.Hq), static_cast<unsigned>(p.B));
  if (p.elem == Elem::kBf16) {
    DqKernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(p);
  } else {
    DqKernel<__half><<<grid, kThreads, 0, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native_adv
}  // namespace ops
}  // namespace xattn
