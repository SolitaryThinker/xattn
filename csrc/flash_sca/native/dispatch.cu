// Bucket dispatch of the native Blackwell FlashSCA kernels.
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flash_sca/native/backward_stage_mask.h"
#include "flash_sca/native/umma_tile_mask.cuh"
#include "flash_sca/native/visibility.cuh"

namespace xattn::ops::flash_sca_native {

template <typename E, int kD, int kDv>
void ForwardCase(const Params& p, cudaStream_t stream);
template <typename E, int kD, int kDv>
void BackwardCase(const Params& p, cudaStream_t stream);
template <typename E, int kD, int kDv>
void BackwardStages(const Params& p, cudaStream_t stream, unsigned stages);

namespace detail {

// One 64-row tile per CTA (64 threads).
__global__ void TileBoundsKernel(const Params p) {
  __shared__ int reduce[4];
  const int64_t tile = blockIdx.x;
  const int64_t b = blockIdx.y;
  if (threadIdx.x == 0) {
    reduce[0] = INT_MAX;
    reduce[1] = -1;
    reduce[2] = INT_MIN;
    reduce[3] = INT_MAX;
  }
  __syncthreads();
  const Bounds r = RowBounds(p, b, tile * kTileRows + threadIdx.x);
  const bool live = r.lo <= r.hi;
  if (live) {
    atomicMin(&reduce[0], r.lo);
    atomicMax(&reduce[1], r.hi);
  }
  atomicMax(&reduce[2], r.lo);
  atomicMin(&reduce[3], r.hi);
  __syncthreads();
  if (threadIdx.x < 4) p.tile_bounds[(b * p.tile_count + tile) * 4 + threadIdx.x] = reduce[threadIdx.x];
}

// One 64-key tile per CTA: the query tiles whose bounds intersect it.
__global__ void KeyTileRangeKernel(const Params p) {
  __shared__ int reduce[2];
  const int64_t n0 = int64_t(blockIdx.x) * kTileRows;
  const int64_t n1 = n0 + kTileRows - 1;
  const int64_t b = blockIdx.y;
  if (threadIdx.x == 0) {
    reduce[0] = INT_MAX;
    reduce[1] = -1;
  }
  __syncthreads();
  int first = INT_MAX, last = -1;
  for (int64_t t = threadIdx.x; t < p.tile_count; t += blockDim.x) {
    const int32_t* e = p.tile_bounds + (b * p.tile_count + t) * 4;
    if (e[0] <= n1 && e[1] >= n0) {
      first = min(first, static_cast<int>(t));
      last = max(last, static_cast<int>(t));
    }
  }
  atomicMin(&reduce[0], first);
  atomicMax(&reduce[1], last);
  __syncthreads();
  if (threadIdx.x < 2) p.key_tile_range[(b * p.key_tile_count + blockIdx.x) * 2 + threadIdx.x] = reduce[threadIdx.x];
}

}  // namespace detail

void LaunchTileBounds(Params& p, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(p.tile_count), static_cast<unsigned>(p.batch));
  detail::TileBoundsKernel<<<grid, detail::kTileRows, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchKeyTileRange(Params& p, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(p.key_tile_count), static_cast<unsigned>(p.batch));
  detail::KeyTileRangeKernel<<<grid, 256, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

template <typename E, bool kForward>
void DispatchBuckets(const Params& p, cudaStream_t stream) {
  const int d = QkBucket(p.qk_dim), dv = ValueBucket(p.value_dim);
#define XATTN_SCA_NATIVE_CASE(D, DV)                    \
  if (d == D && dv == DV) {                              \
    if constexpr (kForward) {                            \
      ForwardCase<E, D, DV>(p, stream);                  \
    } else {                                             \
      BackwardCase<E, D, DV>(p, stream);                 \
    }                                                    \
    return;                                              \
  }
  XATTN_SCA_NATIVE_CASE(64, 64)
  XATTN_SCA_NATIVE_CASE(64, 128)
  XATTN_SCA_NATIVE_CASE(64, 256)
  XATTN_SCA_NATIVE_CASE(128, 64)
  XATTN_SCA_NATIVE_CASE(128, 128)
  XATTN_SCA_NATIVE_CASE(128, 256)
  XATTN_SCA_NATIVE_CASE(256, 64)
  XATTN_SCA_NATIVE_CASE(256, 128)
  XATTN_SCA_NATIVE_CASE(256, 256)
#undef XATTN_SCA_NATIVE_CASE
  TORCH_CHECK(false, "native FlashSCA has no kernel for D=", p.qk_dim, ", Dv=", p.value_dim);
}

// Backward buckets with a stage mask: the same instantiations, but only the
// stages still set in `stages` run. Used by LaunchBackwardStages so another body
// can serve one phase while this one runs the rest.
template <typename E>
void DispatchBackwardBuckets(const Params& p, cudaStream_t stream, unsigned stages) {
  const int d = QkBucket(p.qk_dim), dv = ValueBucket(p.value_dim);
#define XATTN_SCA_NATIVE_CASE(D, DV)                     \
  if (d == D && dv == DV) {                              \
    BackwardStages<E, D, DV>(p, stream, stages);         \
    return;                                              \
  }
  XATTN_SCA_NATIVE_CASE(64, 64)
  XATTN_SCA_NATIVE_CASE(64, 128)
  XATTN_SCA_NATIVE_CASE(64, 256)
  XATTN_SCA_NATIVE_CASE(128, 64)
  XATTN_SCA_NATIVE_CASE(128, 128)
  XATTN_SCA_NATIVE_CASE(128, 256)
  XATTN_SCA_NATIVE_CASE(256, 64)
  XATTN_SCA_NATIVE_CASE(256, 128)
  XATTN_SCA_NATIVE_CASE(256, 256)
#undef XATTN_SCA_NATIVE_CASE
  TORCH_CHECK(false, "native FlashSCA has no kernel for D=", p.qk_dim, ", Dv=", p.value_dim);
}

}  // namespace

void LaunchForward(const Params& p, cudaStream_t stream) {
  if (p.element == Element::kHalf) {
    DispatchBuckets<__half, true>(p, stream);
  } else {
    DispatchBuckets<__nv_bfloat16, true>(p, stream);
  }
}

void LaunchBackwardStages(const Params& p, cudaStream_t stream, unsigned stages) {
  TORCH_CHECK((stages & ~kBackwardAllStages) == 0,
              "LaunchBackwardStages: unknown stage bits in mask ", stages, " (known: ",
              static_cast<unsigned>(kBackwardAllStages), ")");
  // dQ is written by two different bodies: DqKernel when dq_accum is null, and
  // the fused dK+dV pass otherwise (its atomics fill the accumulator the host
  // folds). A mask that runs the dQ stage while the fused pass is cleared and an
  // accumulator exists would leave dq = dq_accum * scale = exactly zero with no
  // error, and a zero dQ is invisible to the shaped gates. This is the one place
  // that sees both values, so it refuses.
  TORCH_CHECK(!((stages & kStageDq) != 0 && p.dq_accum != nullptr &&
                (stages & kStageDkdv) == 0),
              "LaunchBackwardStages: dq_accum is set (deterministic=False) but the fused dK+dV "
              "stage is cleared; dq would be exactly zero. Keep kStageDkdv, or clear kStageDq "
              "and have the serving body accumulate into dq_accum -- on the atomics route the "
              "host fold is the only writer of dq, so a body writing dq directly would be "
              "overwritten.");
  // The same mistake in the form the check above cannot see: an EMPTY mask with an
  // accumulator launches nothing, so the fold would turn an accumulator no body filled into
  // a dq of exact zeros. No caller passes it: LaunchBackwardPhases calls this function only
  // with kStageDelta (followed by the products) and with a non-empty product mask. So it is
  // refused here rather than trusted. An empty mask without an accumulator stays legal.
  TORCH_CHECK(!(stages == 0 && p.dq_accum != nullptr),
              "LaunchBackwardStages: an empty stage mask with dq_accum set (deterministic=False) "
              "launches nothing, so nothing here fills the accumulator and the host fold would "
              "write dq as exact zeros. Pass the stages that fill it (kStageDkdv), or no "
              "accumulator.");
  if (p.element == Element::kHalf) {
    DispatchBackwardBuckets<__half>(p, stream, stages);
  } else {
    DispatchBackwardBuckets<__nv_bfloat16>(p, stream, stages);
  }
}

void LaunchBackward(const Params& p, cudaStream_t stream) {
  LaunchBackwardStages(p, stream, kBackwardAllStages);
}

}  // namespace xattn::ops::flash_sca_native
