// Native SM103 advanced (precision level 2-4) reader backward: key-owned dK/dV
// leaf.
//
// Ownership and determinism: one CTA owns one key tile of one K/V head of one
// batch, so every dK/dV element has exactly one writer and no atomic gradient
// participates.  Inside the CTA the query heads of that K/V head are visited in
// ascending order and the query rows in ascending order, and each thread folds
// its own dimensions in one fixed order, so a second launch of the same input
// produces bit-identical dK/dV for both the synchronous and the ordered entry.
// The ordered entry changes only how the statistics sinks round (owned by the
// row producer), not this loop.
//
// Algebra: every per-element expression comes from adv_bwd_math.cuh,
// which transcribes the pinned Hopper mainloop.  Nothing here
// restates it.  This leaf covers level 2, 3 and 4 at full-causal and sliding-chunk
// visibility; window visibility and the statistics sinks are out of its scope
// and the launcher refuses them instead of computing something else.
//
// Row workspace, as the host half stages it:
//   delta/logz/meta planes are [B, Hq, Nq] with row stride 1 and head stride
//   p.rowstat_head_stride; meta plane k starts at k*B*Hq*rowstat_stride.
//   meta[3] is the anchor column and is -1 iff the row admits no key; it is
//   converted here once, and anything that is not >= 0 becomes -1 (int(NaN) is
//   0, a valid column, so an unbounded conversion would resurrect empty rows --
//   measured on the constant-key guard).
#include "attention/precision/native/adv_bwd_dkdv.h"

#include "attention/precision/native/adv_bwd_math.cuh"
// One definition of the segment admission for every family that runs these
// levels; this leaf is its second user.
#include "flash_sca/native/sca_precision.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <c10/cuda/CUDAException.h>


namespace xattn {
namespace ops {
namespace native_adv {
namespace {

namespace prec = ::xattn::ops::sca_prec;

// The shared algebra header is the single source of the per-element
// expressions (bwdmath, the canonical file); nothing here restates it.
using bwdmath::admitted;
using bwdmath::ds_from;
using bwdmath::dv_operand;
using bwdmath::elem_round;
using bwdmath::elem_split;
using bwdmath::lz2_from_logz;
using bwdmath::p_half;
using bwdmath::row_is_live;
using bwdmath::two_e_of;

// Element-type zero that does not go through the conversion operators.  Torch's
// extension builds define __CUDA_NO_HALF_CONVERSIONS__ / __CUDA_NO_BFLOAT16_CONVERSIONS__
// (this package's own setup.py undoes them again at :501-504), so a
// "static_cast<T>(0)" compiles on one build path and fails on the other with
// "no suitable constructor exists to convert from int".  The intrinsic gives
// the same value on both paths.
template <class T>
__device__ __forceinline__ T elem_zero();
template <>
__device__ __forceinline__ __half elem_zero<__half>() {
  return __float2half_rn(0.0f);
}
template <>
__device__ __forceinline__ __nv_bfloat16 elem_zero<__nv_bfloat16>() {
  return __float2bfloat16_rn(0.0f);
}

constexpr int kThreads = 256;
constexpr int kBN = 32;                      // keys per CTA
constexpr int kLanes = kThreads / kBN;       // dimension lanes per key
constexpr int kMaxPerLane = 32;              // D and Dv must be <= kLanes*kMaxPerLane
static_assert(kBN * kLanes == kThreads, "kBN must divide kThreads");

__device__ __forceinline__ float AsFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float AsFloat(__nv_bfloat16 v) { return __bfloat162float(v); }

template <class T>
__global__ __launch_bounds__(kThreads) void AdvBwdDkdvKernel(const Params p) {
  const int j0 = blockIdx.x * kBN;   // first key of this tile
  const int hk = blockIdx.y;         // K/V head
  const int b = blockIdx.z;          // batch

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* kS = reinterpret_cast<T*>(smem_raw);          // [kBN][D]  staged once
  T* vS = kS + (size_t)kBN * p.D;                  // [kBN][Dv] staged once
  float* qS = reinterpret_cast<float*>(vS + (size_t)kBN * p.Dv);  // [D]  per row
  float* gS = qS + p.D;                                          // [Dv] per row

  const int tid = threadIdx.x;
  const int jl = tid / kLanes;       // key slot inside the tile
  const int lane = tid % kLanes;     // this thread's dimension lane
  const int jj = j0 + jl;            // global key index

  // ABI: natural scale and its log2 form are both carried in Params.
  const float scale_log2 = p.softmax_scale_log2;
  const float scale = scale_log2 * 0.6931471805599453f;  // ln 2: log2 units -> natural

  const T* __restrict__ kG = reinterpret_cast<const T*>(p.k);
  const T* __restrict__ vG = reinterpret_cast<const T*>(p.v);
  const T* __restrict__ qG = reinterpret_cast<const T*>(p.q);
  const T* __restrict__ gG = reinterpret_cast<const T*>(p.dy);
  float* __restrict__ dkG = reinterpret_cast<float*>(p.dk);
  float* __restrict__ dvG = reinterpret_cast<float*>(p.dv);

  const int64_t kBase = static_cast<int64_t>(b) * p.k_batch_stride +
                        static_cast<int64_t>(hk) * p.k_head_stride;
  const int64_t vBase = static_cast<int64_t>(b) * p.v_batch_stride +
                        static_cast<int64_t>(hk) * p.v_head_stride;

  // Stage this tile's K and V once: the rows below only ever read them.
  for (int idx = tid; idx < kBN * static_cast<int>(p.D); idx += kThreads) {
    const int jl2 = idx / static_cast<int>(p.D);
    const int d = idx - jl2 * static_cast<int>(p.D);
    const int j = j0 + jl2;
    kS[idx] = (j < p.Nk)
                  ? kG[kBase + static_cast<int64_t>(j) * p.k_row_stride + d]
                  : elem_zero<T>();
  }
  for (int idx = tid; idx < kBN * static_cast<int>(p.Dv); idx += kThreads) {
    const int jl2 = idx / static_cast<int>(p.Dv);
    const int c = idx - jl2 * static_cast<int>(p.Dv);
    const int j = j0 + jl2;
    vS[idx] = (j < p.Nk)
                  ? vG[vBase + static_cast<int64_t>(j) * p.v_row_stride + c]
                  : elem_zero<T>();
  }

  // This thread's slice of dK and dV: key jl, dimensions lane, lane+kLanes, ...
  float accK[kMaxPerLane];
  float accV[kMaxPerLane];
#pragma unroll
  for (int n = 0; n < kMaxPerLane; ++n) {
    accK[n] = 0.0f;
    accV[n] = 0.0f;
  }

  const int level = static_cast<int>(p.level);
  const int64_t plane = static_cast<int64_t>(p.B) * p.Hq * p.rowstat_head_stride;
  // Schedule bound, not algebra: query row i admits keys j <= i + offset - strict
  // only, so no row before this one can contribute to this tile.
  // Sliding chunk without reset additionally bounds the last row: row i
  // admits nothing left of (floor(pos/C) - 1) * C. Reset positions bound
  // neither end, so every row is visited and decided by its own key range.
  const bool chunk = p.visibility == 2;
  int iStart = max(0, j0 - static_cast<int>(p.offset) + (p.strict ? 1 : 0));
  int iEnd = static_cast<int>(p.Nq);
  if (chunk && p.reset) {
    iStart = 0;
  } else if (chunk) {
    const int64_t last_key = static_cast<int64_t>(j0) + kBN - 1;
    const int64_t last_row = (last_key / p.span + 2) * p.span - 1 - p.offset;
    if (last_row + 1 < iEnd) iEnd = last_row + 1 > 0 ? static_cast<int>(last_row + 1) : 0;
  }

  for (int r = 0; r < static_cast<int>(p.qk_ratio); ++r) {
    const int hq = hk * static_cast<int>(p.qk_ratio) + r;
    const int64_t statRow =
        (static_cast<int64_t>(b) * p.Hq + hq) * p.rowstat_head_stride;
    const int64_t qHead = static_cast<int64_t>(b) * p.q_batch_stride +
                          static_cast<int64_t>(hq) * p.q_head_stride;
    const int64_t gHead = static_cast<int64_t>(b) * p.dy_batch_stride +
                          static_cast<int64_t>(hq) * p.dy_head_stride;

    for (int i = iStart; i < iEnd; ++i) {
      const int64_t rs = statRow + i;
      // Row workspace comes from the caller; logz is the NATURAL log of the row
      // normalizer and meta[3] is the anchor column (-1, or non-finite logz, iff
      // the row admits no key at all).
      const float logz_nat = p.logz[rs];
      const float anchor_col = p.meta[3 * plane + rs];

      // Uniform across the block, so the barriers below stay convergent.
      if (!row_is_live(anchor_col, logz_nat)) continue;
      const int first = chunk ? static_cast<int>(p.meta[4 * plane + rs]) : 0;
      const int a = chunk ? static_cast<int>(p.meta[5 * plane + rs])
                          : i + static_cast<int>(p.offset) - (p.strict ? 1 : 0);
      const int mrow = min(kBN, a - j0 + 1);
      if (mrow <= 0 || first > j0 + kBN - 1) continue;
      const float row_max = p.meta[0 * plane + rs];
      const float anchor_dp = p.meta[1 * plane + rs];
      const float two_e = two_e_of(p.meta[2 * plane + rs], level);
      const float lz2 = lz2_from_logz(logz_nat);
      const float delta_row = p.delta[rs];

      const int64_t qRow = qHead + static_cast<int64_t>(i) * p.q_row_stride;
      const int64_t gRow = gHead + static_cast<int64_t>(i) * p.dy_row_stride;

      __syncthreads();  // qS/gS of the previous row are done being read
      for (int d = tid; d < static_cast<int>(p.D); d += kThreads)
        qS[d] = AsFloat(qG[qRow + d]);
      for (int c = tid; c < static_cast<int>(p.Dv); c += kThreads)
        gS[c] = AsFloat(gG[gRow + c]);
      __syncthreads();

      // Fixed, fully unrolled accumulation order: constant indices keep the
      // accumulators and these sums in registers (measured: 0 spill, no local
      // frame) and make the per-thread fold order bit-reproducible.
      const int D = static_cast<int>(p.D);
      const int Dv = static_cast<int>(p.Dv);
      float s = 0.0f;
      float dp = 0.0f;
#pragma unroll
      for (int n = 0; n < kMaxPerLane; ++n) {
        const int d = lane + n * kLanes;
        if (d < D) s += qS[d] * AsFloat(kS[jl * D + d]);
      }
#pragma unroll
      for (int n = 0; n < kMaxPerLane; ++n) {
        const int c = lane + n * kLanes;
        if (c < Dv) dp += gS[c] * AsFloat(vS[jl * Dv + c]);
      }
#pragma unroll
      for (int m = 1; m < kLanes; m <<= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, m);
        dp += __shfl_xor_sync(0xffffffffu, dp, m);
      }

      if (jj < static_cast<int>(p.Nk) && jl < mrow) {
        const bool admit =
            chunk ? (jj >= first && jj <= a && prec::SegmentOk(p, b, i, jj))
                  : admitted(i, jj, p, b);
        if (admit) {
          const float ph = p_half(s, row_max, lz2, scale_log2);
          float ds = ds_from(ph, dp, two_e, anchor_dp, delta_row);
          float pop = dv_operand(ph, two_e);
          // Both operand types are element typed in the pin's mainloop: the dK
          // contraction multiplies this dS and the dV contraction multiplies
          // this P*2^e, so each is rounded to the reader element type before it
          // is used.  One component below level 4 -> a single element rounding.
          // Two components (level 4, and whenever the dV operand is split) ->
          // hi + lo, which carries back the residual the first rounding dropped.
          if (level >= 4) {
            float hi, lo;
            elem_split(ds, p.elem, hi, lo);
            ds = hi + lo;
          } else {
            ds = elem_round(ds, p.elem);
          }
          if (p.two_component_dv) {
            float hi, lo;
            elem_split(pop, p.elem, hi, lo);
            pop = hi + lo;
          } else {
            pop = elem_round(pop, p.elem);
          }
          const float dsc = scale * ds;
#pragma unroll
          for (int n = 0; n < kMaxPerLane; ++n) {
            const int d = lane + n * kLanes;
            if (d < D) accK[n] += dsc * qS[d];
          }
#pragma unroll
          for (int n = 0; n < kMaxPerLane; ++n) {
            const int c = lane + n * kLanes;
            if (c < Dv) accV[n] += pop * gS[c];
          }
        }
      }
    }
  }

  // One writer per element, once.
  if (jj < static_cast<int>(p.Nk)) {
    float* dkRow = dkG + static_cast<int64_t>(b) * p.dk_batch_stride +
                   static_cast<int64_t>(hk) * p.dk_head_stride +
                   static_cast<int64_t>(jj) * p.dk_row_stride;
    float* dvRow = dvG + static_cast<int64_t>(b) * p.dv_batch_stride +
                   static_cast<int64_t>(hk) * p.dv_head_stride +
                   static_cast<int64_t>(jj) * p.dv_row_stride;
#pragma unroll
    for (int m = 0; m < kMaxPerLane; ++m) {
      const int d = lane + m * kLanes;
      if (d < static_cast<int>(p.D)) dkRow[d] = accK[m];
    }
#pragma unroll
    for (int m = 0; m < kMaxPerLane; ++m) {
      const int c = lane + m * kLanes;
      if (c < static_cast<int>(p.Dv)) dvRow[c] = accV[m];
    }
  }
}

template <class T>
void Run(const Params& p, cudaStream_t stream) {
  const size_t smem = static_cast<size_t>(kBN) * (p.D + p.Dv) * sizeof(T) +
                      static_cast<size_t>(p.D + p.Dv) * sizeof(float);
  XATTN_ADV_CHECK(smem <= 48u * 1024u,
              "native advanced dK/dV leaf: dynamic shared memory ", smem,
              " B exceeds the 48 KiB no-attribute limit");
  const dim3 grid(static_cast<unsigned>((p.Nk + kBN - 1) / kBN),
                  static_cast<unsigned>(p.Hkv), static_cast<unsigned>(p.B));
  AdvBwdDkdvKernel<T><<<grid, kThreads, smem, stream>>>(p);
  // The kDk/kDv phase rows are written only after this launcher returns, and a
  // row claims the launch returned without an immediate launch error. That is
  // only true if the launch is checked here, as adv_bwd_dq.cu's row-dQ launch
  // does: without it a kRan row could follow a launch the driver refused.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void LaunchDkdv(const Params& p, cudaStream_t stream) {
  XATTN_ADV_CHECK(p.visibility == 0 || p.visibility == 2,
              "native advanced dK/dV leaf implements full-causal and sliding-chunk visibility (got ",
              p.visibility, ")");
  XATTN_ADV_CHECK(p.level >= 2 && p.level <= 4,
              "native advanced dK/dV leaf serves precision levels 2-4 (got ", p.level, ")");
  XATTN_ADV_CHECK(p.D > 0 && p.D <= kLanes * kMaxPerLane, "D out of range: ", p.D);
  XATTN_ADV_CHECK(p.Dv > 0 && p.Dv <= kLanes * kMaxPerLane, "Dv out of range: ", p.Dv);
  XATTN_ADV_CHECK(p.q_seg == nullptr || p.k_seg != nullptr,
              "segments must be supplied as a pair");
  XATTN_ADV_CHECK(p.k_seg == nullptr || p.q_seg != nullptr,
              "segments must be supplied as a pair");
  // Params::Elem is exactly the operand element type (the current ABI: the unused
  // kFp16Fp32/kBf16Fp32 enumerators are gone), so one case per type.
  switch (p.elem) {
    case Elem::kFp16:
      Run<__half>(p, stream);
      break;
    case Elem::kBf16:
      Run<__nv_bfloat16>(p, stream);
      break;
    default:
      XATTN_ADV_CHECK(false, "native advanced dK/dV leaf: unsupported element type");
  }
}

}  // namespace native_adv
}  // namespace ops
}  // namespace xattn
