// One-launch reduction of a packed-sequence BOS mask to the kernels' row bounds.
// See row_bounds.h for what is computed and why.
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cstdint>

#include "flash_sca/native/row_bounds.h"

namespace xattn::ops::flash_sca_native {
namespace {

constexpr int kRowsPerBlock = 256;  // query rows, one per key position, per CTA
constexpr int kWarps = kRowsPerBlock / 32;

// Query row at index i of batch b sits at key position p = i + shift. Its key
// interval is [max(chunk base, segment start), p - strict]:
//
//  - the segment start is the last BOS at or before p (key position 0 when the
//    mask has none there, which is the segment the reference calls id 0);
//  - the segment END never binds: the last key of p's segment is at or after p
//    by definition, so min(p - strict, segment end) is p - strict, and only the
//    start has to be found.
//
// Block (tile, b) covers the kRowsPerBlock consecutive key positions its rows sit
// at, so one pass over the mask serves the tile.
__global__ void RowBoundsKernel(const unsigned char* __restrict__ bos,
                                int32_t* __restrict__ lo_out, int32_t* __restrict__ hi_out,
                                int64_t total, int64_t q_length, int64_t shift,
                                int32_t chunk, int strict, int reset) {
  __shared__ int s_wlast[kWarps];  // last BOS in this CTA's warp, -1 if none
  __shared__ int s_start;          // last BOS before the tile, -1 if none

  const int64_t b = blockIdx.y;
  const int64_t q0 = int64_t(blockIdx.x) * kRowsPerBlock;
  const int64_t k0 = q0 + shift;
  const int64_t k1 = min(k0 + kRowsPerBlock, total) - 1;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (tid == 0) s_start = -1;
  __syncthreads();

  const unsigned char* row = bos + b * total;

  // The last BOS before the tile, four key positions per thread per round. Only
  // the part of it that can still be the row's bound is read: with chunk
  // positions measured from the sequence start, a row whose segment started
  // before its own chunk base takes the chunk base, so a start at or below the
  // tile's first chunk base is one value (0) as far as the tile is concerned.
  // A reset call measures positions from the segment start, so it needs the
  // exact start and reads the whole prefix.
  {
    const int64_t chunk_base = ((k0 / chunk) - 1) * chunk;
    const int64_t from = reset != 0 ? 0 : (chunk_base > 0 ? chunk_base : 0);
    for (int64_t p0 = from; p0 < k0; p0 += 4 * blockDim.x) {
      const int64_t base = p0 + 4 * tid;
      int mine = -1;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int64_t p = base + k;
        if (p < k0 && row[p] != 0) mine = static_cast<int>(p);
      }
      mine = __reduce_max_sync(0xffffffffu, mine);
      if (mine >= 0 && lane == 0) atomicMax(&s_start, mine);
    }
  }

  const bool live = (k0 + tid) <= k1;
  const bool is_bos = live && row[k0 + tid] != 0;
  const unsigned mask = __ballot_sync(0xffffffffu, is_bos);
  const int32_t warp_base = static_cast<int32_t>(k0) + warp * 32;
  if (lane == 0) s_wlast[warp] = mask ? warp_base + 31 - __clz(mask) : -1;
  __syncthreads();
  if (!live) return;

  // The last BOS at or before this row: the warp's own mask answers the part
  // inside the warp, and otherwise the nearest earlier warp of the tile, and
  // otherwise the tile's prefix scan.
  const unsigned toward = lane == 31 ? 0xffffffffu : ((1u << (lane + 1)) - 1u);
  const unsigned upto = mask & toward;
  int seg_start;
  if (upto) {
    seg_start = warp_base + 31 - __clz(upto);
  } else {
    seg_start = s_start < 0 ? 0 : s_start;
    for (int w = warp - 1; w >= 0; --w) {
      if (s_wlast[w] >= 0) {
        seg_start = s_wlast[w];
        break;
      }
    }
  }

  const int64_t p = k0 + tid;
  int64_t lo;
  if (reset) {
    // Chunk positions restart in every segment, so the chunk base is measured
    // from the segment's own first key.
    int64_t lower = ((p - seg_start) / chunk - 1) * chunk;
    if (lower < 0) lower = 0;
    lo = seg_start + lower;
  } else {
    int64_t base = (p / chunk - 1) * chunk;
    if (base < 0) base = 0;
    lo = base > seg_start ? base : seg_start;
  }
  const int64_t at = b * q_length + q0 + tid;
  lo_out[at] = static_cast<int32_t>(lo);
  hi_out[at] = static_cast<int32_t>(p - strict);
}

}  // namespace

void LaunchRowBoundsFromBos(const unsigned char* bos, int32_t* lo, int32_t* hi, int64_t batch,
                            int64_t total, int64_t q_length, int64_t shift, int32_t chunk_size,
                            bool strict_past, bool reset, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((q_length + kRowsPerBlock - 1) / kRowsPerBlock),
                  static_cast<unsigned>(batch));
  RowBoundsKernel<<<grid, kRowsPerBlock, 0, stream>>>(bos, lo, hi, total, q_length, shift,
                                                      chunk_size, strict_past ? 1 : 0,
                                                      reset ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xattn::ops::flash_sca_native