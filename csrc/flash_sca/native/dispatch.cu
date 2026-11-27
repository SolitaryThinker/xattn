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
template <typename E>
void BackwardDeltaOnly(const Params& p, cudaStream_t stream);

namespace detail {

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

// Fused per-batch TileBounds + KeyTileRange kernel:
// Phase 1: 16-lane coalesced int4 row_lo/row_hi reduction (256 B coalesced/tile) + warp shuffle.
// Phase 2: O(1) parallel monotonicity check + O(log tiles) binary search when monotone (100% of
// BOS/dense masks), with automatic exact linear fallback if non-monotone. ~2-3 us total.
__global__ void TileBoundsKeyRangeKernel(const Params p) {
  extern __shared__ int2 sm_tiles[];
  const int64_t b = blockIdx.x;
  const int64_t tiles = p.tile_count;
  const bool explicit_bounds = p.row_lo != nullptr;
  const int32_t* const rlo_b = explicit_bounds ? (p.row_lo + b * p.q_length) : nullptr;
  const int32_t* const rhi_b = explicit_bounds ? (p.row_hi + b * p.q_length) : nullptr;
  const int max_k = static_cast<int>(p.total_length - 1);

  if (explicit_bounds && ((reinterpret_cast<uintptr_t>(rlo_b) & 15u) == 0u) &&
      ((reinterpret_cast<uintptr_t>(rhi_b) & 15u) == 0u)) {
    // 16 lanes collaboratively reduce one 64-row tile (4 rows = 1 int4 per lane, 256 B coalesced).
    const int lane16 = threadIdx.x & 15;
    const int slot = threadIdx.x >> 4;
    const int64_t tiles_round = ((tiles + 15) / 16) * 16;
    for (int64_t t = slot; t < tiles_round; t += 16) {
      int min_lo = INT_MAX, max_hi = -1, max_lo = INT_MIN, min_hi = INT_MAX;
      if (t < tiles) {
        const int64_t r0 = t * kTileRows;
        if (r0 + kTileRows <= p.q_length) {
          const int4 lv = reinterpret_cast<const int4*>(rlo_b + r0)[lane16];
          const int4 hv = reinterpret_cast<const int4*>(rhi_b + r0)[lane16];
          const int ls[4] = {lv.x, lv.y, lv.z, lv.w};
          const int hs[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int lo = ls[e] < 0 ? 0 : ls[e];
            const int hi = hs[e] > max_k ? max_k : hs[e];
            if (lo <= hi) {
              min_lo = min(min_lo, lo);
              max_hi = max(max_hi, hi);
              max_lo = max(max_lo, lo);
              min_hi = min(min_hi, hi);
            } else {
              max_lo = INT_MAX;
              min_hi = -1;
            }
          }
        } else {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const Bounds bd = RowBounds(p, b, r0 + 4 * lane16 + e);
            if (bd.lo <= bd.hi) {
              min_lo = min(min_lo, bd.lo);
              max_hi = max(max_hi, bd.hi);
            }
            max_lo = max(max_lo, bd.lo);
            min_hi = min(min_hi, bd.hi);
          }
        }
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) {
        min_lo = min(min_lo, __shfl_xor_sync(0xffffffffu, min_lo, off));
        max_hi = max(max_hi, __shfl_xor_sync(0xffffffffu, max_hi, off));
        max_lo = max(max_lo, __shfl_xor_sync(0xffffffffu, max_lo, off));
        min_hi = min(min_hi, __shfl_xor_sync(0xffffffffu, min_hi, off));
      }
      if (lane16 == 0 && t < tiles) {
        reinterpret_cast<int4*>(p.tile_bounds + (b * tiles + t) * 4)[0] =
            make_int4(min_lo, max_hi, max_lo, min_hi);
        if (p.key_tile_range != nullptr) sm_tiles[t] = make_int2(min_lo, max_hi);
      }
    }
  } else {
    for (int64_t t = threadIdx.x; t < tiles; t += blockDim.x) {
      int min_lo = INT_MAX, max_hi = -1, max_lo = INT_MIN, min_hi = INT_MAX;
      const int64_t r0 = t * kTileRows;
      if (!explicit_bounds && r0 + kTileRows <= p.q_length) {
        const Bounds first = RowBounds(p, b, r0);
        const Bounds last = RowBounds(p, b, r0 + kTileRows - 1);
        if (first.lo <= first.hi) {
          min_lo = first.lo;
          max_hi = last.hi;
          max_lo = last.lo;
          min_hi = first.hi;
        } else {
          for (int r = 0; r < kTileRows; ++r) {
            const Bounds bd = RowBounds(p, b, r0 + r);
            if (bd.lo <= bd.hi) {
              min_lo = min(min_lo, bd.lo);
              max_hi = max(max_hi, bd.hi);
            }
            max_lo = max(max_lo, bd.lo);
            min_hi = min(min_hi, bd.hi);
          }
        }
      } else {
        for (int r = 0; r < kTileRows; ++r) {
          const Bounds bd = RowBounds(p, b, r0 + r);
          if (bd.lo <= bd.hi) {
            min_lo = min(min_lo, bd.lo);
            max_hi = max(max_hi, bd.hi);
          }
          max_lo = max(max_lo, bd.lo);
          min_hi = min(min_hi, bd.hi);
        }
      }
      reinterpret_cast<int4*>(p.tile_bounds + (b * tiles + t) * 4)[0] =
          make_int4(min_lo, max_hi, max_lo, min_hi);
      if (p.key_tile_range != nullptr) sm_tiles[t] = make_int2(min_lo, max_hi);
    }
  }

  if (p.key_tile_range != nullptr) {
    __syncthreads();
    bool mono = true;
    for (int64_t t = threadIdx.x; t < tiles; t += blockDim.x) {
      const int2 cur = sm_tiles[t];
      if (cur.x > cur.y) {
        mono = false;
      } else if (t > 0) {
        const int2 prev = sm_tiles[t - 1];
        if (prev.x > cur.x || prev.y > cur.y) mono = false;
      }
    }
    const bool all_mono = (__syncthreads_and(mono ? 1 : 0) != 0);
    const int64_t keys = p.key_tile_count;
    const int ntiles = static_cast<int>(tiles);
    if (all_mono) {
      for (int64_t n = threadIdx.x; n < keys; n += blockDim.x) {
        const int n0 = static_cast<int>(n * kTileRows);
        const int n1 = n0 + kTileRows - 1;
        int lo = 0, hi = ntiles;
        while (lo < hi) {
          const int mid = (lo + hi) >> 1;
          if (sm_tiles[mid].y >= n0) {
            hi = mid;
          } else {
            lo = mid + 1;
          }
        }
        int first = INT_MAX, last = -1;
        if (lo < ntiles && sm_tiles[lo].x <= n1) {
          first = lo;
          int l2 = lo, h2 = ntiles;
          while (l2 < h2) {
            const int mid = (l2 + h2) >> 1;
            if (sm_tiles[mid].x <= n1) {
              l2 = mid + 1;
            } else {
              h2 = mid;
            }
          }
          last = l2 - 1;
        }
        reinterpret_cast<int2*>(p.key_tile_range + (b * keys + n) * 2)[0] = make_int2(first, last);
      }
    } else {
      for (int64_t n = threadIdx.x; n < keys; n += blockDim.x) {
        const int n0 = static_cast<int>(n * kTileRows);
        const int n1 = n0 + kTileRows - 1;
        int first = INT_MAX, last = -1;
        for (int t = 0; t < ntiles; ++t) {
          const int2 e = sm_tiles[t];
          if (e.x <= n1 && e.y >= n0) {
            first = min(first, t);
            last = max(last, t);
          }
        }
        reinterpret_cast<int2*>(p.key_tile_range + (b * keys + n) * 2)[0] = make_int2(first, last);
      }
    }
  }
}

// Warp-accelerated 1-CTA bitonic sort in shared memory (ascending uint32 keys, P power of 2 <= 2048).
// Uses __shfl_xor_sync in registers for all intra-warp stages (k <= 32 and j <= 16), reducing
// __syncthreads() count from 45 to 10 for P = 512.
__device__ __forceinline__ void BitonicSortU32(uint32_t* sm, int P) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // Stage 1: intra-warp bitonic sort for k = 2, 4, 8, 16, 32 (zero __syncthreads!).
  const int k_max_warp = P < 32 ? P : 32;
  for (int base = warp * 32; base < P; base += blockDim.x) {
    const int idx = base + lane;
    uint32_t val = (idx < P) ? sm[idx] : 0xffffffffu;
    for (int k = 2; k <= k_max_warp; k <<= 1) {
      const bool asc = ((idx & k) == 0);
      for (int j = k >> 1; j > 0; j >>= 1) {
        const uint32_t partner = __shfl_xor_sync(0xffffffffu, val, j);
        const bool is_hi = (lane & j) != 0;
        val = (is_hi == asc) ? max(val, partner) : min(val, partner);
      }
    }
    if (idx < P) sm[idx] = val;
  }
  __syncthreads();

  // Stage 2: inter-warp merge for k = 64 .. P, with j <= 16 fused into warp registers.
  // Each thread maps directly to a unique compare-swap pair (lo, hi) with 100% thread utilization.
  const int half_P = P >> 1;
  for (int k = 64; k <= P; k <<= 1) {
    for (int j = k >> 1; j >= 32; j >>= 1) {
      const int mask_lo = j - 1;
      for (int pair = tid; pair < half_P; pair += blockDim.x) {
        const int lo = ((pair & ~mask_lo) << 1) | (pair & mask_lo);
        const int hi = lo | j;
        const uint32_t a = sm[lo];
        const uint32_t b = sm[hi];
        const bool asc = ((lo & k) == 0);
        if ((a > b) == asc) {
          sm[lo] = b;
          sm[hi] = a;
        }
      }
      __syncthreads();
    }
    for (int base = warp * 32; base < P; base += blockDim.x) {
      const int idx = base + lane;
      uint32_t val = sm[idx];
      const bool asc = ((idx & k) == 0);
#pragma unroll
      for (int j = 16; j > 0; j >>= 1) {
        const uint32_t partner = __shfl_xor_sync(0xffffffffu, val, j);
        const bool is_hi = (lane & j) != 0;
        val = (is_hi == asc) ? max(val, partner) : min(val, partner);
      }
      sm[idx] = val;
    }
    __syncthreads();
  }
}

constexpr int kKvSchedMaxB = 64;
constexpr int kKvSchedMaxPairs = 2048;
constexpr int kKvSchedBuckets = 6;

// Emits F1's deterministic L2-aware adaptive 5/6-bucket schedule for BackwardKvR128Kernel:
// Order: work bucket (heavy -> light -> dead), then batch (descending max/sum work), then
// kv_head hk (contiguous per (b, hk) so <= 2 (b, hk) groups fit in B300's 132 MB L2 cache),
// then kt ascending within each (bucket, b, hk). Each entry packs (b << 24) | (hk << 16) | kt.
__device__ __forceinline__ void EmitKvBucketSchedule(
    const Params& p, const uint16_t* sm_w, const int32_t* sm_bmax, const int32_t* sm_bsum,
    int32_t tiny_pairs, uint8_t* sm_bkt, uint16_t* sm_pos,
    int32_t (&sm_cnt)[kKvSchedBuckets][kKvSchedMaxB],
    int32_t (&sm_base)[kKvSchedBuckets][kKvSchedMaxB], int32_t* sm_border,
    int32_t* kv_ljf_out, int kv_n) {
  const int tid = threadIdx.x;
  const int B = static_cast<int>(p.batch);
  const int Hkv = static_cast<int>(p.kv_heads);
  const int nkt = static_cast<int>((p.total_length + 127) / 128);
  const int total_tiles = B * Hkv * nkt;

  for (int i = tid; i < kKvSchedBuckets * B; i += blockDim.x) {
    sm_cnt[i / B][i % B] = 0;
  }
  if (tid == 0) {
    for (int b = 0; b < B; ++b) sm_border[b] = b;
    for (int i = 1; i < B; ++i) {
      const int cur = sm_border[i];
      const int32_t cur_max = sm_bmax[cur];
      const int32_t cur_sum = sm_bsum[cur];
      int j = i - 1;
      while (j >= 0) {
        const int prev = sm_border[j];
        const bool better = (cur_max > sm_bmax[prev]) ||
                            (cur_max == sm_bmax[prev] && cur_sum > sm_bsum[prev]);
        if (!better) break;
        sm_border[j + 1] = prev;
        --j;
      }
      sm_border[j + 1] = cur;
    }
  }
  __syncthreads();

  const bool split_mid = (tiny_pairs * Hkv < 256);
  for (int i = tid; i < kv_n; i += blockDim.x) {
    const int32_t w = static_cast<int32_t>(sm_w[i]);
    uint8_t bkt;
    if (w == 0) {
      bkt = 5;
    } else if (w > 32) {
      bkt = 0;
    } else if (w > 16) {
      bkt = 1;
    } else if (split_mid && w > 8) {
      bkt = 2;
    } else if (w > 2) {
      bkt = 3;
    } else {
      bkt = 4;
    }
    sm_bkt[i] = bkt;
  }
  __syncthreads();

  const int nb_pairs = B * kKvSchedBuckets;
  for (int idx = tid; idx < nb_pairs; idx += blockDim.x) {
    const int b = idx / kKvSchedBuckets;
    const int bkt = idx % kKvSchedBuckets;
    const int row_base = b * nkt;
    int cnt = 0;
    for (int kt = 0; kt < nkt; ++kt) {
      if (sm_bkt[row_base + kt] == static_cast<uint8_t>(bkt)) {
        sm_pos[row_base + kt] = static_cast<uint16_t>(cnt++);
      }
    }
    sm_cnt[bkt][b] = cnt;
  }
  __syncthreads();

  if (tid == 0) {
    int off = 0;
    for (int bkt = 0; bkt < kKvSchedBuckets; ++bkt) {
      for (int br = 0; br < B; ++br) {
        const int b = sm_border[br];
        sm_base[bkt][b] = off;
        off += sm_cnt[bkt][b] * Hkv;
      }
    }
  }
  __syncthreads();

  for (int t = tid; t < total_tiles; t += blockDim.x) {
    const int hk = t % Hkv;
    const int pair = t / Hkv;
    const int b = pair / nkt;
    const int kt = pair % nkt;
    const int bkt = static_cast<int>(sm_bkt[pair]);
    const int pos = static_cast<int>(sm_pos[pair]);
    const int cnt = sm_cnt[bkt][b];
    const int out_idx = sm_base[bkt][b] + hk * cnt + pos;
    const uint32_t packed = (static_cast<uint32_t>(b) << 24) |
                            (static_cast<uint32_t>(hk) << 16) |
                            static_cast<uint32_t>(kt);
    kv_ljf_out[out_idx] = static_cast<int32_t>(packed);
  }
}

// 2-CTA parallel worklist builder: CTA 0 builds F1's L2-aware packed kv_ljf_out + zeros kv_counter,
// CTA 1 builds DqFlat's q_ljf_out + zeros dq_counter in parallel using warp-register bitonic sort.
__global__ void BuildLjfWorklistsKernel(const Params p, int32_t* kv_ljf_out, int32_t* q_ljf_out,
                                        unsigned int* kv_counter, unsigned int* dq_counter,
                                        int kv_n, int kv_P, int q_n, int q_P) {
  (void)kv_P;
  extern __shared__ uint32_t sm[];
  __shared__ uint16_t sm_w[kKvSchedMaxPairs];
  __shared__ uint8_t sm_bkt[kKvSchedMaxPairs];
  __shared__ uint16_t sm_pos[kKvSchedMaxPairs];
  __shared__ int32_t sm_cnt[kKvSchedBuckets][kKvSchedMaxB];
  __shared__ int32_t sm_base[kKvSchedBuckets][kKvSchedMaxB];
  __shared__ int32_t sm_bmax[kKvSchedMaxB];
  __shared__ int32_t sm_bsum[kKvSchedMaxB];
  __shared__ int32_t sm_border[kKvSchedMaxB];
  __shared__ int32_t sm_tiny_pairs;
  const int tid = threadIdx.x;

  if (blockIdx.x == 0) {
    if (tid == 0) {
      if (kv_counter != nullptr) *kv_counter = 0u;
      sm_tiny_pairs = 0;
    }
    if (kv_ljf_out != nullptr && kv_n > 0 && kv_n <= kKvSchedMaxPairs && p.batch > 0 &&
        p.batch <= kKvSchedMaxB) {
      const int B = static_cast<int>(p.batch);
      const int64_t nkt = (p.total_length + 127) / 128;
      const int64_t offset = p.total_length - p.q_length;
      for (int i = tid; i < B; i += blockDim.x) {
        sm_bmax[i] = 0;
        sm_bsum[i] = 0;
      }
      __syncthreads();
      for (int i = tid; i < kv_n; i += blockDim.x) {
        const int64_t b = i / nkt;
        const int64_t kt = i % nkt;
        int64_t qt_first = 0, qt_last = -1;
        if (p.key_tile_range != nullptr) {
          auto sub_first = [&](int64_t k) -> int64_t {
            const int32_t* e = p.key_tile_range + (b * p.key_tile_count + k) * 2;
            return e[0] > e[1] ? int64_t{0x7fffffff} : int64_t{e[0] / 2};
          };
          auto sub_last = [&](int64_t k) -> int64_t {
            const int32_t* e = p.key_tile_range + (b * p.key_tile_count + k) * 2;
            return e[1] < 0 ? int64_t{-1} : int64_t{e[1] / 2};
          };
          const int64_t k1 = (2 * kt + 1 < p.key_tile_count) ? 2 * kt + 1 : 2 * kt;
          const int64_t f = min(sub_first(2 * kt), sub_first(k1));
          const int64_t l = max(sub_last(2 * kt), sub_last(k1));
          qt_first = (f == 0x7fffffff) ? 0 : f;
          qt_last = (f == 0x7fffffff || l < 0) ? -1 : l;
        } else {
          const int64_t n0 = kt * 128;
          const int64_t n1 = n0 + 127;
          const int64_t first_row = n0 - offset + (p.strict ? 1 : 0);
          const int64_t last_row = (n1 / p.chunk + 2) * p.chunk - 1 - offset;
          qt_first = (first_row > 0 ? first_row : 0) / 128;
          int64_t z = last_row < p.q_length - 1 ? last_row : p.q_length - 1;
          qt_last = z < 0 ? -1 : z / 128;
        }
        const int32_t w = (qt_first <= qt_last) ? static_cast<int32_t>(qt_last - qt_first + 1) : 0;
        const int32_t wc = w > 65535 ? 65535 : w;
        sm_w[i] = static_cast<uint16_t>(wc);
        atomicMax(&sm_bmax[b], wc);
        atomicAdd(&sm_bsum[b], wc);
        if (wc >= 1 && wc <= 2) atomicAdd(&sm_tiny_pairs, 1);
      }
      __syncthreads();
      EmitKvBucketSchedule(p, sm_w, sm_bmax, sm_bsum, sm_tiny_pairs, sm_bkt, sm_pos, sm_cnt,
                           sm_base, sm_border, kv_ljf_out, kv_n);
    }
  } else {
    if (tid == 0 && dq_counter != nullptr) *dq_counter = 0u;
    if (q_ljf_out != nullptr && q_n > 0) {
      const int64_t nqt = (p.q_length + 127) / 128;
      for (int i = tid; i < q_P; i += blockDim.x) {
        if (i < q_n) {
          const int64_t b = i / nqt;
          const int64_t qt = i % nqt;
          const umma::QueryTileMask m = umma::FromTable(p, b, 2 * qt, 2);
          const uint32_t steps =
              (m.lo <= m.hi) ? static_cast<uint32_t>(m.hi / 64 - m.lo / 64 + 1) : 0u;
          const uint32_t sc = steps > 65534u ? 65534u : steps;
          sm[i] = ((65534u - sc) << 16) | static_cast<uint32_t>(i);
        } else {
          sm[i] = 0xffffffffu;
        }
      }
      __syncthreads();
      BitonicSortU32(sm, q_P);
      for (int i = tid; i < q_n; i += blockDim.x) {
        q_ljf_out[i] = static_cast<int32_t>(sm[i] & 0xffffu);
      }
    }
  }
}

// Single-launch fused KeyTileRange (or analytic TileBounds) + F1/F3 LJF worklist builder.
//   - CTA 0 writes key_tile_range (or analytic tile_bounds) + F1's L2-aware packed kv_ljf_out.
//   - CTA 1 builds DqFlat's q_ljf_out in parallel + zeros *dq_counter.
__global__ void BuildKeyRangeAndLjfKernel(const Params p, bool tile_bounds_ready,
                                          int32_t* kv_ljf_out, int32_t* q_ljf_out,
                                          unsigned int* kv_counter, unsigned int* dq_counter,
                                          int kv_n, int kv_P, int q_n, int q_P) {
  (void)kv_P;
  extern __shared__ uint32_t sm_raw[];
  __shared__ uint16_t sm_w[kKvSchedMaxPairs];
  __shared__ uint8_t sm_bkt[kKvSchedMaxPairs];
  __shared__ uint16_t sm_pos[kKvSchedMaxPairs];
  __shared__ int32_t sm_cnt[kKvSchedBuckets][kKvSchedMaxB];
  __shared__ int32_t sm_base[kKvSchedBuckets][kKvSchedMaxB];
  __shared__ int32_t sm_bmax[kKvSchedMaxB];
  __shared__ int32_t sm_bsum[kKvSchedMaxB];
  __shared__ int32_t sm_border[kKvSchedMaxB];
  __shared__ int32_t sm_tiny_pairs;
  const int tid = threadIdx.x;
  const int64_t tiles = p.tile_count;
  const int64_t keys = p.key_tile_count;

  if (blockIdx.x == 0) {
    if (tid == 0) {
      if (kv_counter != nullptr) *kv_counter = 0u;
      sm_tiny_pairs = 0;
    }
    const bool do_kv = (kv_ljf_out != nullptr && kv_n > 0 && kv_n <= kKvSchedMaxPairs &&
                        p.batch > 0 && p.batch <= kKvSchedMaxB);
    const int B = static_cast<int>(p.batch);
    if (do_kv) {
      for (int i = tid; i < B; i += blockDim.x) {
        sm_bmax[i] = 0;
        sm_bsum[i] = 0;
      }
      __syncthreads();
    }
    const int64_t nkt = (p.total_length + 127) / 128;
    int2* sm_tiles = reinterpret_cast<int2*>(sm_raw);
    int2* sm_kr = sm_tiles + tiles;

    if (!tile_bounds_ready && p.row_lo == nullptr) {
      // Dense analytic tile_bounds: O(1) per 64-row tile!
      for (int64_t b = 0; b < p.batch; ++b) {
        for (int64_t t = tid; t < tiles; t += blockDim.x) {
          const int64_t r0 = t * kTileRows;
          int min_lo = INT_MAX, max_hi = -1, max_lo = INT_MIN, min_hi = INT_MAX;
          if (r0 + kTileRows <= p.q_length) {
            const Bounds first = RowBounds(p, b, r0);
            const Bounds last = RowBounds(p, b, r0 + kTileRows - 1);
            if (first.lo <= first.hi) {
              min_lo = first.lo;
              max_hi = last.hi;
              max_lo = last.lo;
              min_hi = first.hi;
            } else {
              for (int r = 0; r < kTileRows; ++r) {
                const Bounds bd = RowBounds(p, b, r0 + r);
                if (bd.lo <= bd.hi) {
                  min_lo = min(min_lo, bd.lo);
                  max_hi = max(max_hi, bd.hi);
                }
                max_lo = max(max_lo, bd.lo);
                min_hi = min(min_hi, bd.hi);
              }
            }
          } else {
            for (int r = 0; r < kTileRows; ++r) {
              const Bounds bd = RowBounds(p, b, r0 + r);
              if (bd.lo <= bd.hi) {
                min_lo = min(min_lo, bd.lo);
                max_hi = max(max_hi, bd.hi);
              }
              max_lo = max(max_lo, bd.lo);
              min_hi = min(min_hi, bd.hi);
            }
          }
          reinterpret_cast<int4*>(p.tile_bounds + (b * tiles + t) * 4)[0] =
              make_int4(min_lo, max_hi, max_lo, min_hi);
        }
      }
      if (do_kv) {
        const int64_t offset = p.total_length - p.q_length;
        for (int i = tid; i < kv_n; i += blockDim.x) {
          const int64_t b = i / nkt;
          const int64_t kt = i % nkt;
          const int64_t n0 = kt * 128;
          const int64_t n1 = n0 + 127;
          const int64_t first_row = n0 - offset + (p.strict ? 1 : 0);
          const int64_t last_row = (n1 / p.chunk + 2) * p.chunk - 1 - offset;
          const int64_t qt_first = (first_row > 0 ? first_row : 0) / 128;
          const int64_t z = last_row < p.q_length - 1 ? last_row : p.q_length - 1;
          const int64_t qt_last = z < 0 ? -1 : z / 128;
          const int32_t w =
              (qt_first <= qt_last) ? static_cast<int32_t>(qt_last - qt_first + 1) : 0;
          const int32_t wc = w > 65535 ? 65535 : w;
          sm_w[i] = static_cast<uint16_t>(wc);
          atomicMax(&sm_bmax[b], wc);
          atomicAdd(&sm_bsum[b], wc);
          if (wc >= 1 && wc <= 2) atomicAdd(&sm_tiny_pairs, 1);
        }
        __syncthreads();
        EmitKvBucketSchedule(p, sm_w, sm_bmax, sm_bsum, sm_tiny_pairs, sm_bkt, sm_pos, sm_cnt,
                             sm_base, sm_border, kv_ljf_out, kv_n);
      }
    } else if (p.key_tile_range != nullptr) {
      // Explicit bounds with tile_bounds already written by RowBoundsKernel:
      // Compute key_tile_range per batch in SMEM via O(log tiles) binary search and directly
      // populate sm_w without round-tripping key_tile_range through global memory!
      const int ntiles = static_cast<int>(tiles);
      for (int64_t b = 0; b < p.batch; ++b) {
        for (int64_t t = tid; t < tiles; t += blockDim.x) {
          const int4 v = reinterpret_cast<const int4*>(p.tile_bounds + (b * tiles + t) * 4)[0];
          sm_tiles[t] = make_int2(v.x, v.y);
        }
        __syncthreads();
        bool mono = true;
        for (int64_t t = tid; t < tiles; t += blockDim.x) {
          const int2 cur = sm_tiles[t];
          if (cur.x > cur.y) {
            mono = false;
          } else if (t > 0) {
            const int2 prev = sm_tiles[t - 1];
            if (prev.x > cur.x || prev.y > cur.y) mono = false;
          }
        }
        const bool all_mono = (__syncthreads_and(mono ? 1 : 0) != 0);
        if (all_mono) {
          for (int64_t n = tid; n < keys; n += blockDim.x) {
            const int n0 = static_cast<int>(n * kTileRows);
            const int n1 = n0 + kTileRows - 1;
            int lo = 0, hi = ntiles;
            while (lo < hi) {
              const int mid = (lo + hi) >> 1;
              if (sm_tiles[mid].y >= n0) {
                hi = mid;
              } else {
                lo = mid + 1;
              }
            }
            int first = INT_MAX, last = -1;
            if (lo < ntiles && sm_tiles[lo].x <= n1) {
              first = lo;
              int l2 = lo, h2 = ntiles;
              while (l2 < h2) {
                const int mid = (l2 + h2) >> 1;
                if (sm_tiles[mid].x <= n1) {
                  l2 = mid + 1;
                } else {
                  h2 = mid;
                }
              }
              last = l2 - 1;
            }
            const int2 kr = make_int2(first, last);
            reinterpret_cast<int2*>(p.key_tile_range + (b * keys + n) * 2)[0] = kr;
            sm_kr[n] = kr;
          }
        } else {
          for (int64_t n = tid; n < keys; n += blockDim.x) {
            const int n0 = static_cast<int>(n * kTileRows);
            const int n1 = n0 + kTileRows - 1;
            int first = INT_MAX, last = -1;
            for (int t = 0; t < ntiles; ++t) {
              const int2 e = sm_tiles[t];
              if (e.x <= n1 && e.y >= n0) {
                first = min(first, t);
                last = max(last, t);
              }
            }
            const int2 kr = make_int2(first, last);
            reinterpret_cast<int2*>(p.key_tile_range + (b * keys + n) * 2)[0] = kr;
            sm_kr[n] = kr;
          }
        }
        if (do_kv) {
          __syncthreads();
          for (int64_t kt = tid; kt < nkt; kt += blockDim.x) {
            const int2 e0 = sm_kr[2 * kt];
            const int2 e1 = (2 * kt + 1 < keys) ? sm_kr[2 * kt + 1] : e0;
            const int f0 = e0.x > e0.y ? 0x7fffffff : (e0.x >> 1);
            const int f1 = e1.x > e1.y ? 0x7fffffff : (e1.x >> 1);
            const int l0 = e0.y < 0 ? -1 : (e0.y >> 1);
            const int l1 = e1.y < 0 ? -1 : (e1.y >> 1);
            const int f = min(f0, f1);
            const int l = max(l0, l1);
            const int32_t w = (f != 0x7fffffff && l >= f) ? (l - f + 1) : 0;
            const int32_t wc = w > 65535 ? 65535 : w;
            sm_w[b * nkt + kt] = static_cast<uint16_t>(wc);
            atomicMax(&sm_bmax[b], wc);
            atomicAdd(&sm_bsum[b], wc);
            if (wc >= 1 && wc <= 2) atomicAdd(&sm_tiny_pairs, 1);
          }
          __syncthreads();
        }
      }
      if (do_kv) {
        EmitKvBucketSchedule(p, sm_w, sm_bmax, sm_bsum, sm_tiny_pairs, sm_bkt, sm_pos, sm_cnt,
                             sm_base, sm_border, kv_ljf_out, kv_n);
      }
    }
  } else {
    if (tid == 0 && dq_counter != nullptr) *dq_counter = 0u;
    if (q_ljf_out != nullptr && q_n > 0) {
      uint32_t* sm_sort = sm_raw;
      const int64_t nqt = (p.q_length + 127) / 128;
      for (int i = tid; i < q_P; i += blockDim.x) {
        if (i < q_n) {
          const int64_t b = i / nqt;
          const int64_t qt = i % nqt;
          int lo = INT_MAX, hi = -1;
          if (tile_bounds_ready) {
            const umma::QueryTileMask m = umma::FromTable(p, b, 2 * qt, 2);
            lo = m.lo;
            hi = m.hi;
          } else {
            const int64_t r0 = qt * 128;
            const int64_t r1 = min(p.q_length - 1, r0 + 127);
            const Bounds f = RowBounds(p, b, r0);
            const Bounds l = RowBounds(p, b, r1);
            if (f.lo <= f.hi) {
              lo = f.lo;
              hi = l.hi;
            } else {
              for (int64_t r = r0; r <= r1; ++r) {
                const Bounds bd = RowBounds(p, b, r);
                if (bd.lo <= bd.hi) {
                  lo = min(lo, bd.lo);
                  hi = max(hi, bd.hi);
                }
              }
            }
          }
          const uint32_t steps = (lo <= hi) ? static_cast<uint32_t>(hi / 64 - lo / 64 + 1) : 0u;
          const uint32_t sc = steps > 65534u ? 65534u : steps;
          sm_sort[i] = ((65534u - sc) << 16) | static_cast<uint32_t>(i);
        } else {
          sm_sort[i] = 0xffffffffu;
        }
      }
      __syncthreads();
      BitonicSortU32(sm_sort, q_P);
      for (int i = tid; i < q_n; i += blockDim.x) {
        q_ljf_out[i] = static_cast<int32_t>(sm_sort[i] & 0xffffu);
      }
    }
  }
}

}  // namespace detail

__global__ void RowBoundsKernel(const RowBoundsArgs a) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t b = blockIdx.y;
  if (i >= a.q_length) return;
  const int64_t* const row = a.ids + b * a.total;
  const int64_t p = i + (a.total - a.q_length);
  const int64_t id = row[p];
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
  int64_t lo = (p / a.chunk - 1) * a.chunk;
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

void LaunchTileBoundsAndKeyRange(const Params& p, cudaStream_t stream) {
  if (p.batch <= 0 || p.tile_bounds == nullptr) return;
  constexpr int64_t kMaxSmemTiles = 6144;
  if (p.key_tile_range == nullptr || p.tile_count <= kMaxSmemTiles) {
    const size_t smem =
        p.key_tile_range != nullptr ? static_cast<size_t>(p.tile_count) * sizeof(int2) : 0;
    detail::TileBoundsKeyRangeKernel<<<static_cast<unsigned>(p.batch), 256, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    Params mutable_p = p;
    LaunchTileBounds(mutable_p, stream);
    LaunchKeyTileRange(mutable_p, stream);
  }
}

void LaunchBuildLjfWorklists(const Params& p, int32_t* kv_ljf_out, int32_t* q_ljf_out,
                             unsigned int* kv_counter, unsigned int* dq_counter,
                             cudaStream_t stream) {
  const int kv_n = kv_ljf_out != nullptr
                       ? static_cast<int>(p.batch * ((p.total_length + 127) / 128))
                       : 0;
  const int q_n = q_ljf_out != nullptr
                      ? static_cast<int>(p.batch * ((p.q_length + 127) / 128))
                      : 0;
  auto next_pow2 = [](int x) {
    int r = 1;
    while (r < x) r <<= 1;
    return r;
  };
  const int q_P = q_n > 0 ? next_pow2(q_n) : 0;
  const size_t smem = static_cast<size_t>(q_P) * sizeof(uint32_t);
  detail::BuildLjfWorklistsKernel<<<2, 256, smem, stream>>>(
      p, kv_ljf_out, q_ljf_out, kv_counter, dq_counter, kv_n, 0, q_n, q_P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchKeyRangeAndLjf(const Params& p, bool tile_bounds_ready, int32_t* kv_ljf_out,
                          int32_t* q_ljf_out, unsigned int* kv_counter, unsigned int* dq_counter,
                          cudaStream_t stream) {
  const int kv_n = kv_ljf_out != nullptr
                       ? static_cast<int>(p.batch * ((p.total_length + 127) / 128))
                       : 0;
  const int q_n = q_ljf_out != nullptr
                      ? static_cast<int>(p.batch * ((p.q_length + 127) / 128))
                      : 0;
  auto next_pow2 = [](int x) {
    int r = 1;
    while (r < x) r <<= 1;
    return r;
  };
  const int q_P = q_n > 0 ? next_pow2(q_n) : 0;
  const int64_t kr_count = p.key_tile_range != nullptr ? p.key_tile_count : 0;
  const size_t cta0_smem =
      p.key_tile_range != nullptr
          ? static_cast<size_t>(p.tile_count + kr_count) * sizeof(int2)
          : 0;
  const size_t cta1_smem = static_cast<size_t>(q_P) * sizeof(uint32_t);
  const size_t smem = cta0_smem > cta1_smem ? cta0_smem : cta1_smem;
  // Static SMEM in BuildKeyRangeAndLjfKernel is ~14 KB; keep dynamic <= 32 KB (<= 46 KB total).
  if ((tile_bounds_ready || p.row_lo == nullptr) && smem <= 32 * 1024) {
    detail::BuildKeyRangeAndLjfKernel<<<2, 256, smem, stream>>>(
        p, tile_bounds_ready, kv_ljf_out, q_ljf_out, kv_counter, dq_counter,
        kv_n, 0, q_n, q_P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    if (!tile_bounds_ready || p.key_tile_range != nullptr) {
      LaunchTileBoundsAndKeyRange(p, stream);
    }
    LaunchBuildLjfWorklists(p, kv_ljf_out, q_ljf_out, kv_counter, dq_counter, stream);
  }
}

void LaunchRowBounds(const RowBoundsArgs& args, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const dim3 grid(static_cast<unsigned>((args.q_length + kThreads - 1) / kThreads),
                  static_cast<unsigned>(args.batch));
  RowBoundsKernel<<<grid, kThreads, 0, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
constexpr int64_t kMaskChunk = 32;
}  // namespace

__global__ void RowBoundsMaskKernel(const RowBoundsMaskArgs a) {
  extern __shared__ int32_t smem[];
  const int64_t nchunk = (a.total + kMaskChunk - 1) / kMaskChunk;
  const int64_t nwords = (nchunk + 31) / 32;
  uint32_t* const starts = reinterpret_cast<uint32_t*>(smem);
  uint32_t* const occupied = reinterpret_cast<uint32_t*>(smem + nchunk);
  const int64_t b = blockIdx.y;
  const uint8_t* const row = a.mask + b * a.total;
  for (int64_t c = threadIdx.x; c < nchunk; c += blockDim.x) starts[c] = 0u;
  for (int64_t w = threadIdx.x; w < nwords; w += blockDim.x) occupied[w] = 0u;
  __syncthreads();
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
  const bool in_range = (i < a.q_length);
  int32_t lo_i = 0, hi_i = -1;
  if (in_range) {
    const int64_t p = i + (a.total - a.q_length);
    const int64_t c = p >> 5;
    const unsigned bit = static_cast<unsigned>(p) & 31u;
    int64_t first = 0;
    {
      const uint32_t m =
          starts[c] & (bit == 31u ? 0xffffffffu : ((uint32_t(1) << (bit + 1u)) - 1u));
      if (m != 0u) {
        first = (c << 5) + (31 - __clz(m));
      } else if (c > 0) {
        int64_t w = (c - 1) >> 5;
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
    int64_t last = a.total - 1;
    {
      const uint32_t m = bit == 31u ? 0u : (starts[c] & (0xffffffffu << (bit + 1u)));
      if (m != 0u) {
        last = (c << 5) + (__ffs(m) - 1) - 1;
      } else if (c + 1 < nchunk) {
        int64_t w = (c + 1) >> 5;
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
    int64_t lo = (p / a.chunk - 1) * a.chunk;
    if (lo < 0) lo = 0;
    int64_t hi = p - a.strict;
    if (first > lo) lo = first;
    if (last < hi) hi = last;
    lo_i = static_cast<int32_t>(lo);
    hi_i = static_cast<int32_t>(hi);
    a.row_lo[b * a.q_length + i] = lo_i;
    a.row_hi[b * a.q_length + i] = hi_i;
  }
  if (a.tile128_bounds != nullptr) {
    const bool live = in_range && lo_i <= hi_i;
    const int32_t w_min_lo = __reduce_min_sync(0xffffffffu, live ? lo_i : INT_MAX);
    const int32_t w_max_hi = __reduce_max_sync(0xffffffffu, live ? hi_i : -1);
    const int32_t w_max_lo = __reduce_max_sync(0xffffffffu, live ? lo_i : INT_MAX);
    const int32_t w_min_hi = __reduce_min_sync(0xffffffffu, live ? hi_i : INT_MIN);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
      smem[warp * 4 + 0] = w_min_lo;
      smem[warp * 4 + 1] = w_max_hi;
      smem[warp * 4 + 2] = w_max_lo;
      smem[warp * 4 + 3] = w_min_hi;
    }
    __syncthreads();
    if (threadIdx.x < 2) {
      const int half = threadIdx.x;
      const int64_t nqt = (a.q_length + 127) / 128;
      const int64_t qt = int64_t(blockIdx.x) * 2 + half;
      if (qt < nqt) {
        const int w0 = half * 4;
        int32_t min_lo = smem[w0 * 4 + 0];
        int32_t max_hi = smem[w0 * 4 + 1];
        int32_t max_lo = smem[w0 * 4 + 2];
        int32_t min_hi = smem[w0 * 4 + 3];
#pragma unroll
        for (int dw = 1; dw < 4; ++dw) {
          min_lo = min(min_lo, smem[(w0 + dw) * 4 + 0]);
          max_hi = max(max_hi, smem[(w0 + dw) * 4 + 1]);
          max_lo = max(max_lo, smem[(w0 + dw) * 4 + 2]);
          min_hi = min(min_hi, smem[(w0 + dw) * 4 + 3]);
        }
        reinterpret_cast<int4*>(a.tile128_bounds + (b * nqt + qt) * 4)[0] =
            make_int4(min_lo, max_hi, max_lo, min_hi);
      }
    }
  }
}

void LaunchRowBoundsMask(const RowBoundsMaskArgs& args, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const int64_t nchunk = (args.total + kMaskChunk - 1) / kMaskChunk;
  const int64_t nwords = (nchunk + 31) / 32;
  const size_t min_ints = args.tile128_bounds != nullptr ? size_t{32} : size_t{0};
  const size_t ints =
      static_cast<size_t>(nchunk + nwords) > min_ints ? static_cast<size_t>(nchunk + nwords)
                                                      : min_ints;
  const size_t bytes = ints * sizeof(int32_t);
  const dim3 grid(static_cast<unsigned>((args.q_length + kThreads - 1) / kThreads),
                  static_cast<unsigned>(args.batch));
  RowBoundsMaskKernel<<<grid, kThreads, bytes, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void BuildFwdLjfKernel(const int32_t* __restrict__ tile128_bounds,
                                  int32_t* __restrict__ q_ljf_out,
                                  unsigned int* __restrict__ fwd_counter, int total_q, int P,
                                  int nqt, int64_t q_length, int64_t total_length) {
  extern __shared__ uint32_t sm[];
  const int tid = threadIdx.x;
  if (tid == 0 && fwd_counter != nullptr) *fwd_counter = 0u;
  const int64_t shift = total_length - q_length;
  for (int i = tid; i < P; i += blockDim.x) {
    if (i < total_q) {
      const int qt = i % nqt;
      const int4 tb = reinterpret_cast<const int4*>(tile128_bounds + int64_t(i) * 4)[0];
      const int64_t q_last = min(int64_t(qt) * 128 + 127, q_length - 1);
      const int top_tile = static_cast<int>(min(q_last + shift, total_length - 1) / 64);
      const uint32_t steps =
          (tb.x <= tb.y) ? static_cast<uint32_t>(top_tile - (tb.x / 64) + 1) : 0u;
      const uint32_t sc = steps > 65534u ? 65534u : steps;
      sm[i] = ((65534u - sc) << 16) | static_cast<uint32_t>(i);
    } else {
      sm[i] = 0xffffffffu;
    }
  }
  __syncthreads();
  detail::BitonicSortU32(sm, P);
  for (int i = tid; i < total_q; i += blockDim.x) {
    q_ljf_out[i] = static_cast<int32_t>(sm[i] & 0xffffu);
  }
}

// Fast 1-CTA 512-thread stable 2-pass 4-bit radix sort for total_q <= 512 (steps < 255).
// Falls back to bitonic sort inside the same kernel if any tile has steps >= 255.
__global__ void BuildFwdLjfFast512Kernel(const int32_t* __restrict__ tile128_bounds,
                                         int32_t* __restrict__ q_ljf_out,
                                         unsigned int* __restrict__ fwd_counter, int total_q,
                                         int nqt, int64_t q_length, int64_t total_length) {
  __shared__ uint32_t sm_vals[512];
  __shared__ uint32_t sm_cnt[256];
  __shared__ uint32_t sm_wsum[8];
  const int tid = threadIdx.x;
  if (tid == 0 && fwd_counter != nullptr) *fwd_counter = 0u;
  const int64_t shift = total_length - q_length;
  uint32_t steps = 0u;
  if (tid < total_q) {
    const int qt = tid % nqt;
    const int4 tb = reinterpret_cast<const int4*>(tile128_bounds + int64_t(tid) * 4)[0];
    const int64_t q_last = min(int64_t(qt) * 128 + 127, q_length - 1);
    const int top_tile = static_cast<int>(min(q_last + shift, total_length - 1) / 64);
    steps = (tb.x <= tb.y) ? static_cast<uint32_t>(top_tile - (tb.x / 64) + 1) : 0u;
  }
  if (!__syncthreads_or(tid < total_q && steps >= 255u)) {
    uint32_t key = (tid < total_q) ? (((254u - steps) << 16) | static_cast<uint32_t>(tid))
                                   : ((255u << 16) | static_cast<uint32_t>(tid));
    const int w = tid >> 5;
    const int l = tid & 31;
    const uint32_t lt_mask = (1u << l) - 1u;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const uint32_t digit = (key >> (16 + pass * 4)) & 15u;
      int local_rank = 0;
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        const uint32_t m = __ballot_sync(0xffffffffu, digit == static_cast<uint32_t>(d));
        if (l == 0) sm_cnt[d * 16 + w] = __popc(m);
        if (digit == static_cast<uint32_t>(d)) local_rank = __popc(m & lt_mask);
      }
      __syncthreads();
      uint32_t v = 0u, x = 0u;
      if (tid < 256) {
        v = sm_cnt[tid];
        x = v;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          const uint32_t t = __shfl_up_sync(0xffffffffu, x, off);
          if (l >= off) x += t;
        }
        if (l == 31) sm_wsum[w] = x;
      }
      __syncthreads();
      if (tid < 8) {
        const uint32_t orig = sm_wsum[tid];
        uint32_t ws = orig;
#pragma unroll
        for (int off = 1; off < 8; off <<= 1) {
          const uint32_t t = __shfl_up_sync(0xffu, ws, off);
          if (tid >= off) ws += t;
        }
        sm_wsum[tid] = ws - orig;
      }
      __syncthreads();
      if (tid < 256) sm_cnt[tid] = (x - v) + sm_wsum[w];
      __syncthreads();
      sm_vals[sm_cnt[digit * 16 + w] + local_rank] = key;
      __syncthreads();
      key = sm_vals[tid];
    }
    if (tid < total_q) q_ljf_out[tid] = static_cast<int32_t>(key & 0xffffu);
    return;
  }
  if (tid < total_q) {
    const uint32_t sc = steps > 65534u ? 65534u : steps;
    sm_vals[tid] = ((65534u - sc) << 16) | static_cast<uint32_t>(tid);
  } else {
    sm_vals[tid] = 0xffffffffu;
  }
  __syncthreads();
  for (int k = 2; k <= 512; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int ixj = tid ^ j;
      if (ixj > tid) {
        const uint32_t a = sm_vals[tid];
        const uint32_t b = sm_vals[ixj];
        const bool asc = ((tid & k) == 0);
        if ((a > b) == asc) {
          sm_vals[tid] = b;
          sm_vals[ixj] = a;
        }
      }
      __syncthreads();
    }
  }
  if (tid < total_q) q_ljf_out[tid] = static_cast<int32_t>(sm_vals[tid] & 0xffffu);
}

void LaunchBuildFwdLjf(const int32_t* tile128_bounds, int32_t* q_ljf_out,
                       unsigned int* fwd_counter, int64_t batch, int64_t q_length,
                       int64_t total_length, cudaStream_t stream) {
  const int nqt = static_cast<int>((q_length + 127) / 128);
  const int total_q = static_cast<int>(batch) * nqt;
  if (total_q <= 0 || total_q > 2048) return;
  if (total_q <= 512) {
    BuildFwdLjfFast512Kernel<<<1, 512, 0, stream>>>(tile128_bounds, q_ljf_out, fwd_counter, total_q,
                                                    nqt, q_length, total_length);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  int P = 1;
  while (P < total_q) P <<= 1;
  const size_t smem = static_cast<size_t>(P) * sizeof(uint32_t);
  BuildFwdLjfKernel<<<1, 256, smem, stream>>>(tile128_bounds, q_ljf_out, fwd_counter, total_q, P,
                                              nqt, q_length, total_length);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

__device__ __forceinline__ int4 ReduceTile128FromRowBounds(const Params& p, int64_t b, int64_t qt,
                                                           int lane) {
  const int64_t r0 = qt * 128;
  const int max_k = static_cast<int>(p.total_length - 1);
  const int32_t* const rlo_b = p.row_lo + b * p.q_length;
  const int32_t* const rhi_b = p.row_hi + b * p.q_length;
  int min_lo = INT_MAX, max_hi = -1, max_lo = INT_MIN, min_hi = INT_MAX;
  const bool aligned16 = ((reinterpret_cast<uintptr_t>(rlo_b + r0) & 15u) == 0u) &&
                         ((reinterpret_cast<uintptr_t>(rhi_b + r0) & 15u) == 0u);
  if (aligned16 && r0 + 128 <= p.q_length) {
    const int4 lv = reinterpret_cast<const int4*>(rlo_b + r0)[lane];
    const int4 hv = reinterpret_cast<const int4*>(rhi_b + r0)[lane];
    const int ls[4] = {lv.x, lv.y, lv.z, lv.w};
    const int hs[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int lo = ls[e] < 0 ? 0 : ls[e];
      const int hi = hs[e] > max_k ? max_k : hs[e];
      if (lo <= hi) {
        min_lo = min(min_lo, lo);
        max_hi = max(max_hi, hi);
        max_lo = max(max_lo, lo);
        min_hi = min(min_hi, hi);
      } else {
        max_lo = INT_MAX;
        min_hi = INT_MIN;
      }
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int64_t r = r0 + 4 * lane + e;
      if (r < p.q_length) {
        const int lo = rlo_b[r] < 0 ? 0 : rlo_b[r];
        const int hi = rhi_b[r] > max_k ? max_k : rhi_b[r];
        if (lo <= hi) {
          min_lo = min(min_lo, lo);
          max_hi = max(max_hi, hi);
          max_lo = max(max_lo, lo);
          min_hi = min(min_hi, hi);
        } else {
          max_lo = INT_MAX;
          min_hi = INT_MIN;
        }
      } else {
        max_lo = INT_MAX;
        min_hi = INT_MIN;
      }
    }
  }
  min_lo = __reduce_min_sync(0xffffffffu, min_lo);
  max_hi = __reduce_max_sync(0xffffffffu, max_hi);
  max_lo = __reduce_max_sync(0xffffffffu, max_lo);
  min_hi = __reduce_min_sync(0xffffffffu, min_hi);
  return make_int4(min_lo, max_hi, max_lo, min_hi);
}

__global__ void Tile128BoundsAndFwdLjfKernel(const Params p, int32_t* __restrict__ tile128_bounds,
                                             int32_t* __restrict__ q_ljf_out,
                                             unsigned int* __restrict__ fwd_counter, int total_q,
                                             int P, int nqt) {
  extern __shared__ uint32_t sm[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int num_warps = blockDim.x >> 5;
  if (tid == 0 && fwd_counter != nullptr) *fwd_counter = 0u;
  const int64_t shift = p.total_length - p.q_length;

  for (int i = warp; i < total_q; i += num_warps) {
    const int64_t b = i / nqt;
    const int64_t qt = i % nqt;
    const int4 tb = ReduceTile128FromRowBounds(p, b, qt, lane);
    if (lane == 0) {
      reinterpret_cast<int4*>(tile128_bounds + int64_t(i) * 4)[0] = tb;
      if (q_ljf_out != nullptr) {
        const int64_t q_last = min(qt * 128 + 127, p.q_length - 1);
        const int top_tile = static_cast<int>(min(q_last + shift, p.total_length - 1) / 64);
        const uint32_t steps =
            (tb.x <= tb.y) ? static_cast<uint32_t>(top_tile - (tb.x / 64) + 1) : 0u;
        const uint32_t sc = steps > 65534u ? 65534u : steps;
        sm[i] = ((65534u - sc) << 16) | static_cast<uint32_t>(i);
      }
    }
  }
  if (q_ljf_out != nullptr) {
    for (int i = total_q + tid; i < P; i += blockDim.x) sm[i] = 0xffffffffu;
    __syncthreads();
    detail::BitonicSortU32(sm, P);
    for (int i = tid; i < total_q; i += blockDim.x) {
      q_ljf_out[i] = static_cast<int32_t>(sm[i] & 0xffffu);
    }
  }
}

__global__ void Tile128BoundsOnlyKernel(const Params p, int32_t* __restrict__ tile128_bounds,
                                        int total_q, int nqt) {
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_global >= total_q) return;
  const int64_t b = warp_global / nqt;
  const int64_t qt = warp_global % nqt;
  const int4 tb = ReduceTile128FromRowBounds(p, b, qt, lane);
  if (lane == 0) {
    reinterpret_cast<int4*>(tile128_bounds + int64_t(warp_global) * 4)[0] = tb;
  }
}

}  // namespace

void LaunchTile128BoundsAndFwdLjf(const Params& p, int32_t* tile128_bounds, int32_t* q_ljf_out,
                                  unsigned int* fwd_counter, cudaStream_t stream) {
  if (p.batch <= 0 || p.q_length <= 0 || p.row_lo == nullptr || tile128_bounds == nullptr) return;
  const int nqt = static_cast<int>((p.q_length + 127) / 128);
  const int total_q = static_cast<int>(p.batch) * nqt;
  if (total_q <= 0) return;
  if (q_ljf_out != nullptr && total_q <= 2048) {
    int P = 1;
    while (P < total_q) P <<= 1;
    const size_t smem = static_cast<size_t>(P) * sizeof(uint32_t);
    Tile128BoundsAndFwdLjfKernel<<<1, 256, smem, stream>>>(p, tile128_bounds, q_ljf_out,
                                                           fwd_counter, total_q, P, nqt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    constexpr int kWarpsPerBlock = 8;
    const unsigned blocks = static_cast<unsigned>((total_q + kWarpsPerBlock - 1) / kWarpsPerBlock);
    Tile128BoundsOnlyKernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(p, tile128_bounds,
                                                                        total_q, nqt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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
  TORCH_CHECK(!((stages & kStageDq) != 0 && p.dq_accum != nullptr &&
                (stages & kStageDkdv) == 0),
              "LaunchBackwardStages: dq_accum is set (deterministic=False) but the fused dK+dV "
              "stage is cleared; dq would be exactly zero. Keep kStageDkdv, or clear kStageDq "
              "and have the serving body accumulate into dq_accum -- on the atomics route the "
              "host fold is the only writer of dq, so a body writing dq directly would be "
              "overwritten.");
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

void LaunchDeltaOnly(const Params& p, cudaStream_t stream) {
  if (p.element == Element::kHalf) {
    BackwardDeltaOnly<__half>(p, stream);
  } else {
    BackwardDeltaOnly<__nv_bfloat16>(p, stream);
  }
}

}  // namespace xattn::ops::flash_sca_native
