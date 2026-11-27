// One-launch reduction of a packed-sequence BOS mask to the kernels' row bounds.
// See row_bounds.h for what is computed and why.
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

#include "flash_sca/native/row_bounds.h"

namespace xattn::ops::flash_sca_native {
namespace {

constexpr int kRowsPerBlock = 512;  // 16 warps; 128 CTAs on B=2, L=32768 fits in 1 wave on B300
constexpr int kWarps = kRowsPerBlock / 32;

__device__ __forceinline__ int LastNonZeroByteInU32(uint32_t w) {
  if (w & 0xff000000u) return 3;
  if (w & 0x00ff0000u) return 2;
  if (w & 0x0000ff00u) return 1;
  return 0;
}

__device__ __forceinline__ int64_t ChunkBase(int64_t pos, int32_t chunk, bool pow2_chunk) {
  int64_t base = pow2_chunk ? ((pos & ~int64_t(chunk - 1)) - chunk)
                            : ((pos / chunk - 1) * int64_t(chunk));
  return base > 0 ? base : 0;
}

// Query row at index i of batch b sits at key position p = i + shift. Its key
// interval is [max(chunk base, segment start), p - strict].
// Uses 512 rows/CTA (1 single wave on 148 SMs for B=2, L=32768), backward vectorized
// 16-byte (uint4) prefix scans (8192 bytes/iter = 1 iteration for chunk=4096) with early exit,
// and power-of-2 bitmask chunk arithmetic (0 integer divisions for chunk=4096/2048).
// Optionally fuses per-64-row (kTileMode=1) or per-128-row (kTileMode=2) tile_bounds reduction
// directly from registers (zero extra HBM reads).
template <int kTileMode>
__global__ void RowBoundsKernel(const unsigned char* __restrict__ bos,
                                int32_t* __restrict__ lo_out, int32_t* __restrict__ hi_out,
                                int32_t* __restrict__ tile_bounds_out,
                                int64_t total, int64_t q_length, int64_t shift,
                                int32_t chunk, int strict, int reset) {
  __shared__ int s_wlast[kWarps];   // last BOS in this CTA's warp, -1 if none
  __shared__ int s_start;           // last BOS before the tile, -1 if none
  __shared__ int4 s_wtile[(kTileMode != 0) ? kWarps : 1];

  const int64_t b = blockIdx.y;
  const int64_t q0 = int64_t(blockIdx.x) * kRowsPerBlock;
  const int64_t k0 = q0 + shift;
  const int64_t k1 = min(k0 + kRowsPerBlock, total) - 1;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool pow2_chunk = (chunk > 0) && ((chunk & (chunk - 1)) == 0);
  if (tid == 0) s_start = -1;
  __syncthreads();

  const unsigned char* row = bos + b * total;

  const int64_t from = reset != 0 ? 0 : ChunkBase(k0, chunk, pow2_chunk);
  if (from < k0) {
    const bool aligned16 = ((reinterpret_cast<uintptr_t>(row) & 15u) == 0u);
    if (aligned16) {
      const int64_t g_lo = from >> 4;
      const int64_t g_hi = (k0 - 1) >> 4;
      const int64_t full_groups = total >> 4;
      const uint4* const vrow = reinterpret_cast<const uint4*>(row);
      for (int64_t g_end = g_hi; g_end >= g_lo; g_end -= kRowsPerBlock) {
        const int64_t g = g_end - tid;
        int mine = -1;
        if (g >= g_lo) {
          const int64_t base = g << 4;
          if (g < full_groups) {
            const uint4 v = vrow[g];
            if ((v.x | v.y | v.z | v.w) != 0u) {
              if (base >= from && base + 16 <= k0) {
                uint32_t w;
                int off;
                if (v.w) {
                  w = v.w;
                  off = 12;
                } else if (v.z) {
                  w = v.z;
                  off = 8;
                } else if (v.y) {
                  w = v.y;
                  off = 4;
                } else {
                  w = v.x;
                  off = 0;
                }
                mine = static_cast<int>(base + off + LastNonZeroByteInU32(w));
              } else {
                for (int k = 15; k >= 0; --k) {
                  const int64_t p = base + k;
                  if (p >= from && p < k0 && row[p] != 0) {
                    mine = static_cast<int>(p);
                    break;
                  }
                }
              }
            }
          } else {
            for (int k = 15; k >= 0; --k) {
              const int64_t p = base + k;
              if (p >= from && p < k0 && row[p] != 0) {
                mine = static_cast<int>(p);
                break;
              }
            }
          }
        }
        mine = __reduce_max_sync(0xffffffffu, mine);
        if (mine >= 0 && lane == 0) atomicMax(&s_start, mine);
        if (g_end - kRowsPerBlock >= g_lo) {
          __syncthreads();
          if (s_start >= 0) break;
        }
      }
    } else {
      for (int64_t p_end = k0; p_end > from; p_end -= 4 * kRowsPerBlock) {
        const int64_t p_base = p_end - 4 * (tid + 1);
        int mine = -1;
#pragma unroll
        for (int k = 3; k >= 0; --k) {
          const int64_t p = p_base + k;
          if (p >= from && p < k0 && row[p] != 0) {
            mine = static_cast<int>(p);
            break;
          }
        }
        mine = __reduce_max_sync(0xffffffffu, mine);
        if (mine >= 0 && lane == 0) atomicMax(&s_start, mine);
        if (p_end - 4 * kRowsPerBlock > from) {
          __syncthreads();
          if (s_start >= 0) break;
        }
      }
    }
  }

  const bool live = (k0 + tid) <= k1;
  const bool is_bos = live && row[k0 + tid] != 0;
  const unsigned mask = __ballot_sync(0xffffffffu, is_bos);
  const int32_t warp_base = static_cast<int32_t>(k0) + warp * 32;
  if (lane == 0) s_wlast[warp] = mask ? warp_base + 31 - __clz(mask) : -1;
  __syncthreads();

  int v_min_lo = INT_MAX, v_max_hi = -1, v_max_lo = INT_MAX;
  int v_min_hi = (kTileMode == 2) ? INT_MIN : -1;
  if (live) {
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
      lo = seg_start + ChunkBase(p - seg_start, chunk, pow2_chunk);
    } else {
      const int64_t base = ChunkBase(p, chunk, pow2_chunk);
      lo = base > seg_start ? base : seg_start;
    }
    const int32_t lo32 = static_cast<int32_t>(lo);
    const int32_t hi32 = static_cast<int32_t>(p - strict);
    const int64_t at = b * q_length + q0 + tid;
    lo_out[at] = lo32;
    hi_out[at] = hi32;
    if constexpr (kTileMode != 0) {
      const int max_k = static_cast<int>(total - 1);
      const int cl_lo = lo32 < 0 ? 0 : lo32;
      const int cl_hi = hi32 > max_k ? max_k : hi32;
      if (cl_lo <= cl_hi) {
        v_min_lo = cl_lo;
        v_max_hi = cl_hi;
        v_max_lo = cl_lo;
        v_min_hi = cl_hi;
      }
    }
  }

  if constexpr (kTileMode != 0) {
    const int w_min_lo = __reduce_min_sync(0xffffffffu, v_min_lo);
    const int w_max_hi = __reduce_max_sync(0xffffffffu, v_max_hi);
    const int w_max_lo = __reduce_max_sync(0xffffffffu, v_max_lo);
    const int w_min_hi = __reduce_min_sync(0xffffffffu, v_min_hi);
    if (lane == 0) {
      s_wtile[warp] = make_int4(w_min_lo, w_max_hi, w_max_lo, w_min_hi);
    }
    __syncthreads();
    if constexpr (kTileMode == 1) {
      constexpr int kTilesPerCta = kRowsPerBlock / 64;  // 8
      if (tid < kTilesPerCta) {
        const int64_t tile_count = (q_length + 63) / 64;
        const int64_t t = int64_t(blockIdx.x) * kTilesPerCta + tid;
        if (t < tile_count) {
          const int4 a = s_wtile[2 * tid];
          const int4 c = s_wtile[2 * tid + 1];
          const int4 merged =
              make_int4(min(a.x, c.x), max(a.y, c.y), max(a.z, c.z), min(a.w, c.w));
          reinterpret_cast<int4*>(tile_bounds_out + (b * tile_count + t) * 4)[0] = merged;
        }
      }
    } else {
      constexpr int kTilesPerCta = kRowsPerBlock / 128;  // 4
      if (tid < kTilesPerCta) {
        const int64_t nqt = (q_length + 127) / 128;
        const int64_t qt = int64_t(blockIdx.x) * kTilesPerCta + tid;
        if (qt < nqt) {
          const int4 a0 = s_wtile[4 * tid + 0];
          const int4 a1 = s_wtile[4 * tid + 1];
          const int4 a2 = s_wtile[4 * tid + 2];
          const int4 a3 = s_wtile[4 * tid + 3];
          const int4 merged = make_int4(
              min(min(a0.x, a1.x), min(a2.x, a3.x)),
              max(max(a0.y, a1.y), max(a2.y, a3.y)),
              max(max(a0.z, a1.z), max(a2.z, a3.z)),
              min(min(a0.w, a1.w), min(a2.w, a3.w)));
          reinterpret_cast<int4*>(tile_bounds_out + (b * nqt + qt) * 4)[0] = merged;
        }
      }
    }
  }
}

// H25: one block per batch row scans that row's BOS mask into an inclusive int64 prefix sum
// (the segment id of every key): each thread sums its own contiguous run of the mask, then a
// block-wide Hillis-Steele scan of the kScanThreads per-thread totals gives every thread's
// exclusive base, and a second pass over the same run adds it in and writes the result. Integer
// counts: any correct scan gives the exact same bits as torch's cumsum, so this needs no
// tolerance and no fallback to agree with it.
constexpr int kScanThreads = 1024;

__global__ void SegmentIdsFromBosKernel(const unsigned char* __restrict__ bos,
                                        int64_t* __restrict__ ids, int64_t total) {
  __shared__ int64_t s_block[kScanThreads];
  const int64_t b = blockIdx.x;
  const unsigned char* row = bos + b * total;
  int64_t* out = ids + b * total;
  const int tid = threadIdx.x;
  const int64_t per = (total + kScanThreads - 1) / kScanThreads;
  const int64_t start = tid * per;
  const int64_t end = min(start + per, total);

  int64_t local = 0;
  for (int64_t i = start; i < end; ++i) local += row[i] != 0 ? 1 : 0;
  s_block[tid] = local;
  __syncthreads();

#pragma unroll
  for (int offset = 1; offset < kScanThreads; offset <<= 1) {
    int64_t val = s_block[tid];
    if (tid >= offset) val += s_block[tid - offset];
    __syncthreads();
    s_block[tid] = val;
    __syncthreads();
  }

  int64_t running = tid == 0 ? 0 : s_block[tid - 1];
  for (int64_t i = start; i < end; ++i) {
    running += row[i] != 0 ? 1 : 0;
    out[i] = running;
  }
}

}  // namespace

void LaunchRowBoundsFromBos(const unsigned char* bos, int32_t* lo, int32_t* hi, int64_t batch,
                            int64_t total, int64_t q_length, int64_t shift, int32_t chunk_size,
                            bool strict_past, bool reset, cudaStream_t stream,
                            int32_t* tile_bounds_out, bool tile128) {
  const dim3 grid(static_cast<unsigned>((q_length + kRowsPerBlock - 1) / kRowsPerBlock),
                  static_cast<unsigned>(batch));
  if (tile_bounds_out != nullptr) {
    if (tile128) {
      RowBoundsKernel<2><<<grid, kRowsPerBlock, 0, stream>>>(
          bos, lo, hi, tile_bounds_out, total, q_length, shift, chunk_size,
          strict_past ? 1 : 0, reset ? 1 : 0);
    } else {
      RowBoundsKernel<1><<<grid, kRowsPerBlock, 0, stream>>>(
          bos, lo, hi, tile_bounds_out, total, q_length, shift, chunk_size,
          strict_past ? 1 : 0, reset ? 1 : 0);
    }
  } else {
    RowBoundsKernel<0><<<grid, kRowsPerBlock, 0, stream>>>(
        bos, lo, hi, nullptr, total, q_length, shift, chunk_size,
        strict_past ? 1 : 0, reset ? 1 : 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchSegmentIdsFromBos(const unsigned char* bos, int64_t* ids, int64_t batch,
                             int64_t total, cudaStream_t stream) {
  if (batch <= 0 || total <= 0) return;
  SegmentIdsFromBosKernel<<<static_cast<unsigned>(batch), kScanThreads, 0, stream>>>(bos, ids,
                                                                                     total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xattn::ops::flash_sca_native