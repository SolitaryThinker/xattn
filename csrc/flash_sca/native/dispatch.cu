// Bucket dispatch of the native Blackwell FlashSCA kernels.
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flash_sca/native/backward_stage_mask.h"
#include "flash_sca/native/row_bounds.h"
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

// The metadata boundary's fused visible-bounds reduction (row_bounds.h): one thread per query
// row, a binary search per row over the key ids. It replaces the Python chain's cumsum-side
// searchsorted/maximum/minimum/cast launches with this one launch; the ids themselves are
// still the caller's (the BOS mask's cumsum), so the two routes cannot disagree on them.
__global__ void RowBoundsKernel(const RowBoundsArgs a) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t b = blockIdx.y;
  if (i >= a.q_length) return;
  const int64_t* const row = a.ids + b * a.total;
  const int64_t p = i + (a.total - a.q_length);
  const int64_t id = row[p];
  // lower_bound(row, id): first index whose id is >= id. torch.searchsorted(..., right=False).
  int64_t low = 0, high = a.total;
  while (low < high) {
    const int64_t mid = low + (high - low) / 2;
    if (row[mid] < id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  const int64_t first = low;
  // upper_bound(row, id): first index whose id is > id. searchsorted(..., right=True) - 1 is
  // the last key of the query's own segment run.
  low = 0;
  high = a.total;
  while (low < high) {
    const int64_t mid = low + (high - low) / 2;
    if (row[mid] <= id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  const int64_t last = low - 1;
  int64_t lo = (p / a.chunk - 1) * a.chunk;  // p >= 0, chunk > 0: floor division
  if (lo < 0) lo = 0;
  int64_t hi = p - a.strict;
  if (first > lo) lo = first;
  if (last < hi) hi = last;
  a.row_lo[b * a.q_length + i] = static_cast<int32_t>(lo);
  a.row_hi[b * a.q_length + i] = static_cast<int32_t>(hi);
}

void LaunchTileBounds(const Params& p, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(p.tile_count), static_cast<unsigned>(p.batch));
  detail::TileBoundsKernel<<<grid, detail::kTileRows, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchKeyTileRange(Params& p, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(p.key_tile_count), static_cast<unsigned>(p.batch));
  detail::KeyTileRangeKernel<<<grid, 256, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchRowBounds(const RowBoundsArgs& args, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const dim3 grid(static_cast<unsigned>((args.q_length + kThreads - 1) / kThreads),
                  static_cast<unsigned>(args.batch));
  RowBoundsKernel<<<grid, kThreads, 0, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

// Key positions per entry of the mask kernel's start table: one 32-bit word of the mask, so a
// start's chunk is its position >> 5 and phase 2 answers from shared memory alone.
constexpr int64_t kMaskChunk = 32;

}  // namespace

// The mask-side bounds of the metadata boundary (row_bounds.h): the BOS mask in, the row bounds
// out, with no cumsum and no ids plane built at all. Phase 1 reads the row once per CTA in
// 16-byte vector loads and ORs each set position into its chunk's 32-bit word (an OR of the same
// bits from any order is the same table, so the atomics are order-free and two identical calls
// cannot differ); phase 2 then answers each query out of shared memory - the highest start at or
// before p and the lowest start after it, walking the chunk-occupancy words outward instead of
// rescanning the mask.
__global__ void RowBoundsMaskKernel(const RowBoundsMaskArgs a) {
  extern __shared__ int32_t smem[];
  const int64_t nchunk = (a.total + kMaskChunk - 1) / kMaskChunk;
  const int64_t nwords = (nchunk + 31) / 32;
  uint32_t* const starts = reinterpret_cast<uint32_t*>(smem);              // [nchunk]
  uint32_t* const occupied = reinterpret_cast<uint32_t*>(smem + nchunk);   // [nwords]
  const int64_t b = blockIdx.y;
  const uint8_t* const row = a.mask + b * a.total;
  for (int64_t c = threadIdx.x; c < nchunk; c += blockDim.x) starts[c] = 0u;
  for (int64_t w = threadIdx.x; w < nwords; w += blockDim.x) occupied[w] = 0u;
  __syncthreads();
  // Phase 1: the row's start bits. The head is scalar because the row pointer need not be
  // 16-byte aligned, and only a 16-byte group that holds a start is looked at byte by byte.
  {
    const int64_t head =
        static_cast<int64_t>((16u - (reinterpret_cast<uintptr_t>(row) & 15u)) & 15u);
    const int64_t head_end = head < a.total ? head : a.total;
    for (int64_t x = threadIdx.x; x < head_end; x += blockDim.x) {
      if (row[x] != 0) {
        const int64_t c = x >> 5;
        atomicOr(&starts[c], uint32_t(1) << (static_cast<unsigned>(x) & 31u));
        atomicOr(&occupied[c >> 5], uint32_t(1) << (static_cast<unsigned>(c) & 31u));
      }
    }
    const int64_t groups = (a.total - head_end) / 16;
    const uint4* const vec = reinterpret_cast<const uint4*>(row + head_end);
    for (int64_t g = threadIdx.x; g < groups; g += blockDim.x) {
      const uint4 v = vec[g];
      if (v.x == 0u && v.y == 0u && v.z == 0u && v.w == 0u) continue;
      const int64_t x = head_end + 16 * g;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        if (row[x + k] == 0) continue;
        const int64_t y = x + k;
        const int64_t c = y >> 5;
        atomicOr(&starts[c], uint32_t(1) << (static_cast<unsigned>(y) & 31u));
        atomicOr(&occupied[c >> 5], uint32_t(1) << (static_cast<unsigned>(c) & 31u));
      }
    }
    for (int64_t x = head_end + 16 * groups + threadIdx.x; x < a.total; x += blockDim.x) {
      if (row[x] != 0) {
        const int64_t c = x >> 5;
        atomicOr(&starts[c], uint32_t(1) << (static_cast<unsigned>(x) & 31u));
        atomicOr(&occupied[c >> 5], uint32_t(1) << (static_cast<unsigned>(c) & 31u));
      }
    }
  }
  __syncthreads();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= a.q_length) return;
  const int64_t p = i + (a.total - a.q_length);
  const int64_t c = p >> 5;
  const unsigned bit = static_cast<unsigned>(p) & 31u;
  // first: the last start at or before p (0 when the row has no start at all), which is the
  // ids route's lower_bound(row, row[p]) over the cumsum.
  int64_t first = 0;
  {
    const uint32_t m =
        starts[c] & (bit == 31u ? 0xffffffffu : ((uint32_t(1) << (bit + 1u)) - 1u));
    if (m != 0u) {
      first = (c << 5) + (31 - __clz(m));
    } else if (c > 0) {
      int64_t w = (c - 1) >> 5;  // the highest occupied chunk below c
      uint32_t keep = (c & 31) != 0 ? (uint32_t(1) << (static_cast<unsigned>(c) & 31u)) - 1u
                                    : 0xffffffffu;
      while (true) {
        const uint32_t mw = occupied[w] & keep;
        if (mw != 0u) {
          const int64_t k = (w << 5) + (31 - __clz(mw));
          first = (k << 5) + (31 - __clz(starts[k]));
          break;
        }
        if (w == 0) break;
        --w;
        keep = 0xffffffffu;
      }
    }
  }
  // last: the next start after p, minus one (total - 1 when the row has no start after p), which
  // is the ids route's upper_bound(row, row[p]) - 1.
  int64_t last = a.total - 1;
  {
    const uint32_t m = bit == 31u ? 0u : (starts[c] & (0xffffffffu << (bit + 1u)));
    if (m != 0u) {
      last = (c << 5) + (__ffs(m) - 1) - 1;
    } else if (c + 1 < nchunk) {
      int64_t w = (c + 1) >> 5;  // the lowest occupied chunk above c
      uint32_t keep = ((c + 1) & 31) != 0
                          ? 0xffffffffu << (static_cast<unsigned>(c + 1) & 31u)
                          : 0xffffffffu;
      while (true) {
        const uint32_t mw = (w < nwords ? occupied[w] : 0u) & keep;
        if (mw != 0u) {
          const int64_t k = (w << 5) + (__ffs(mw) - 1);
          last = (k << 5) + (__ffs(starts[k]) - 1) - 1;
          break;
        }
        ++w;
        if (w >= nwords) break;
        keep = 0xffffffffu;
      }
    }
  }
  int64_t lo = (p / a.chunk - 1) * a.chunk;  // p >= 0, chunk > 0: floor division
  if (lo < 0) lo = 0;
  int64_t hi = p - a.strict;
  if (first > lo) lo = first;
  if (last < hi) hi = last;
  a.row_lo[b * a.q_length + i] = static_cast<int32_t>(lo);
  a.row_hi[b * a.q_length + i] = static_cast<int32_t>(hi);
}

void LaunchRowBoundsMask(const RowBoundsMaskArgs& args, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const int64_t nchunk = (args.total + kMaskChunk - 1) / kMaskChunk;
  const int64_t nwords = (nchunk + 31) / 32;
  const size_t bytes = static_cast<size_t>(nchunk + nwords) * sizeof(int32_t);
  const dim3 grid(static_cast<unsigned>((args.q_length + kThreads - 1) / kThreads),
                  static_cast<unsigned>(args.batch));
  RowBoundsMaskKernel<<<grid, kThreads, bytes, stream>>>(args);
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
