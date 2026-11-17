#pragma once
// Portable native row-statistics contraction.
//
// Same mathematical contract as RowStatisticsSm90 (delta, log_z and the six
// metadata planes), but with no wgmma/GMMA dependency: the two contractions
// (scores = Q K^T and anchored dP = dO V^T) use nvcuda::wmma 16x16x16 with FP32
// accumulation, and every memory path is a plain 16-byte vector load. This
// compiles for SM80 and newer, including Blackwell (sm_100/sm_103).
//
// The row recurrence is the one the Hopper kernel uses: online row maximum,
// earliest-key anchor, normalizer z, and mean = sum_j p_j (dp_j - dp_anchor)
// over the admitted keys, maintained incrementally with the same rescaling.
// Empty rows keep the upstream convention: delta 0, log_z +inf, anchor -1,
// first nk, last -1.
#include <climits>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace xattn::ops::attention::precision {
namespace native_statistics {

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Shared layout. Padded row strides keep wmma 256-bit alignment (both strides
// are even and every column offset is a multiple of 16 elements) and spread the
// per-row staging reads over banks.
template <class E, int D, int DV>
struct Shared {
  static constexpr int QStride = D + 8;   // elements per row, multiple of 8
  static constexpr int VStride = DV + 8;
  static constexpr int SStride = 68;      // float tile row, multiple of 4
  alignas(32) E q[64 * QStride];
  alignas(32) E go[64 * VStride];
  alignas(32) E k[64 * QStride];
  alignas(32) E v[64 * VStride];
  alignas(32) float s[64 * SStride];
  alignas(32) float dp[64 * SStride];
  int lower, upper;
};

// 16-byte vector copy that zero-fills a chunk that is not admissible. The source
// pointer is formed only in the admissible arm, so a tail chunk of a short row or
// a chunk past the last row never forms an out-of-range pointer at all.
template <class E>
__device__ __forceinline__ void Copy16(E* destination, const E* base, int64_t offset,
                                       bool valid) {
  uint4 value = make_uint4(0u, 0u, 0u, 0u);
  if (valid) value = *reinterpret_cast<const uint4*>(base + offset);
  *reinterpret_cast<uint4*>(destination) = value;
}

// One CTA handles 64 query rows of one (batch, head). Four warps own 16 rows
// each; inside a warp one row is owned by two adjacent lanes, each covering
// half of the 64-key tile, and the halves are combined by one shfl each.
//
// Every warp-wide operation (wmma tile contraction and shfl) runs for the whole
// warp; only the per-row scan and the state commit are predicated per row.
template <class E, int D, int DV, bool Segments>
__global__ void __launch_bounds__(128, 1) RowStatistics(
    const E* Q, const E* K, const E* V, const E* G, const int* L, const int* R,
    const int64_t* QI, const int64_t* KI, const int* Exact, float* Delta,
    float* LogZ, float* Meta, int nq, int nk, int h, int hk, int d, int dv,
    int batch, float alpha) {
  extern __shared__ __align__(32) char buffer[];
  using Layout = Shared<E, D, DV>;
  auto& smem = *reinterpret_cast<Layout*>(buffer);

  const int heads = batch * h;
  const int qblocks = (nq + 63) / 64;
  const int linear = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bh = linear % heads;
  const int b = bh / h;
  const int head = bh % h;
  const int kh = head / (h / hk);
  // Heaviest query block first, so causal tails retire early.
  const int q0 = (qblocks - 1 - linear / heads) * 64;
  const bool segments = Segments && !(Exact && Exact[b]);
  const float ax = alpha * 0.7213475204444817f;  // scale * log2(e) / 2

  if (tid == 0) {
    smem.lower = nk;
    smem.upper = -1;
  }
  __syncthreads();
  if (tid < 64 && q0 + tid < nq) {
    atomicMin(&smem.lower, L[b * nq + q0 + tid]);
    atomicMax(&smem.upper, R[b * nq + q0 + tid]);
  }
  __syncthreads();

  constexpr int ChunksQ = D / 8, ChunksV = DV / 8;
  // Q and dO stay resident for the whole row block, zero padded.
  for (int x = tid; x < 64 * ChunksQ; x += 128) {
    const int row = x / ChunksQ, c = (x % ChunksQ) * 8;
    Copy16(&smem.q[row * Layout::QStride + c], Q,
           ((int64_t(b) * nq + q0 + row) * h + head) * d + c,
           q0 + row < nq && c < d);
  }
  for (int x = tid; x < 64 * ChunksV; x += 128) {
    const int row = x / ChunksV, c = (x % ChunksV) * 8;
    Copy16(&smem.go[row * Layout::VStride + c], G,
           ((int64_t(b) * nq + q0 + row) * h + head) * dv + c,
           q0 + row < nq && c < dv);
  }

  const int begin = max(smem.lower, 0) / 64 * 64;
  const int end = min(smem.upper + 1, nk);

  // Row ownership inside this warp.
  const int row = warp * 16 + (lane >> 1);
  const int half = lane & 1;
  const int qi = q0 + row;
  const bool rowlive = qi < nq;
  const int left = rowlive ? L[b * nq + qi] : nk;
  const int right = rowlive ? R[b * nq + qi] : -1;
  int64_t qid = 0;
  if constexpr (Segments) {
    if (qi < nq) qid = QI[b * nq + qi];
  }

  float maximum = 0.f;    // running maximum of the admitted scores
  float anchor_dp = 0.f;  // dP at the anchored key
  float normalizer = 0.f;
  float mean = 0.f;
  int anchor = -1;
  int first = nk, last = -1;

  for (int k0 = begin; k0 < end; k0 += 64) {
    // K/V tile of 64 keys, zero padded, single stage.
    for (int x = tid; x < 64 * ChunksQ; x += 128) {
      const int j = x / ChunksQ, c = (x % ChunksQ) * 8;
      Copy16(&smem.k[j * Layout::QStride + c], K,
             ((int64_t(b) * nk + k0 + j) * hk + kh) * d + c,
             k0 + j < nk && c < d);
    }
    for (int x = tid; x < 64 * ChunksV; x += 128) {
      const int j = x / ChunksV, c = (x % ChunksV) * 8;
      Copy16(&smem.v[j * Layout::VStride + c], V,
             ((int64_t(b) * nk + k0 + j) * hk + kh) * dv + c,
             k0 + j < nk && c < dv);
    }
    __syncthreads();

    // Contractions for this warp's 16 rows over all 64 keys of the tile.
    const int row0 = warp * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs[4], fd[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      wmma::fill_fragment(fs[n], 0.f);
      wmma::fill_fragment(fd[n], 0.f);
    }
    for (int c0 = 0; c0 < D; c0 += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, E, wmma::row_major> fa;
      wmma::load_matrix_sync(fa, &smem.q[row0 * Layout::QStride + c0], Layout::QStride);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, E, wmma::col_major> fb;
        wmma::load_matrix_sync(fb, &smem.k[(n * 16) * Layout::QStride + c0], Layout::QStride);
        wmma::mma_sync(fs[n], fa, fb, fs[n]);
      }
    }
    for (int c0 = 0; c0 < DV; c0 += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, E, wmma::row_major> fa;
      wmma::load_matrix_sync(fa, &smem.go[row0 * Layout::VStride + c0], Layout::VStride);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, E, wmma::col_major> fb;
        wmma::load_matrix_sync(fb, &smem.v[(n * 16) * Layout::VStride + c0], Layout::VStride);
        wmma::mma_sync(fd[n], fa, fb, fd[n]);
      }
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      wmma::store_matrix_sync(&smem.s[row0 * Layout::SStride + n * 16], fs[n],
                              Layout::SStride, wmma::mem_row_major);
      wmma::store_matrix_sync(&smem.dp[row0 * Layout::SStride + n * 16], fd[n],
                              Layout::SStride, wmma::mem_row_major);
    }
    __syncwarp();

    const bool active = rowlive && k0 <= right && k0 + 63 >= left;
    const int base = row * Layout::SStride;
    const int chunk = half * 32;

    // Tile maximum with its earliest attaining key, plus the admitted extremes.
    float tile_max = -INFINITY, tile_dp = 0.f;
    int tile_key = INT_MAX, tile_first = nk, tile_last = -1;
    if (active) {
#pragma unroll
      for (int kk = 0; kk < 32; ++kk) {
        const int j = chunk + kk, key = k0 + j;
        bool valid = key < nk && key >= left && key <= right;
        if constexpr (Segments) {
          if (valid && segments) valid = KI[b * nk + key] == qid;
        }
        if (valid) {
          tile_first = min(tile_first, key);
          tile_last = max(tile_last, key);
          const float score = smem.s[base + j];
          if (tile_key == INT_MAX || score > tile_max) {
            tile_max = score;
            tile_key = key;
            tile_dp = smem.dp[base + j];
          }
        }
      }
    }
    {
      const float other_max = __shfl_xor_sync(0xffffffffu, tile_max, 1);
      const float other_dp = __shfl_xor_sync(0xffffffffu, tile_dp, 1);
      const int other_key = __shfl_xor_sync(0xffffffffu, tile_key, 1);
      const int other_first = __shfl_xor_sync(0xffffffffu, tile_first, 1);
      const int other_last = __shfl_xor_sync(0xffffffffu, tile_last, 1);
      if (other_key != INT_MAX &&
          (tile_key == INT_MAX || other_max > tile_max ||
           (other_max == tile_max && other_key < tile_key))) {
        tile_max = other_max;
        tile_key = other_key;
        tile_dp = other_dp;
      }
      tile_first = min(tile_first, other_first);
      tile_last = max(tile_last, other_last);
    }

    const bool update = active && tile_key != INT_MAX;
    int next_anchor = anchor;
    float next_max = maximum, next_dp = anchor_dp;
    if (update && (anchor < 0 || tile_max > next_max ||
                   (tile_max == next_max && tile_key < next_anchor))) {
      next_max = tile_max;
      next_anchor = tile_key;
      next_dp = tile_dp;
    }

    float roots[32];
    float partial_z = 0.f;
    if (update) {
#pragma unroll
      for (int kk = 0; kk < 32; ++kk) {
        const int j = chunk + kk, key = k0 + j;
        bool valid = key < nk && key >= left && key <= right;
        if constexpr (Segments) {
          if (valid && segments) valid = KI[b * nk + key] == qid;
        }
        const float root =
            valid ? exp2f(__fmul_rn(__fsub_rn(smem.s[base + j], next_max), ax)) : 0.f;
        roots[kk] = root;
        partial_z += root * root;
      }
    } else {
#pragma unroll
      for (int kk = 0; kk < 32; ++kk) roots[kk] = 0.f;
    }
    partial_z += __shfl_xor_sync(0xffffffffu, partial_z, 1);

    float factor = 0.f, nz = 0.f, inv_root = 0.f, partial = 0.f;
    if (update) {
      factor = anchor < 0 ? 0.f : exp2f(__fmul_rn(__fsub_rn(maximum, next_max), ax));
      nz = (normalizer * factor) * factor + partial_z;
      inv_root = rsqrtf(nz);
#pragma unroll
      for (int kk = 0; kk < 32; ++kk) {
        const float root = roots[kk] * inv_root;
        if (root > 0.f) {
          partial += root * (root * (smem.dp[base + chunk + kk] - next_dp));
        }
      }
    }
    partial += __shfl_xor_sync(0xffffffffu, partial, 1);

    if (update) {
      mean = factor * (factor * ((normalizer / nz) * (mean + (anchor_dp - next_dp)))) + partial;
      normalizer = nz;
      maximum = next_max;
      anchor_dp = next_dp;
      anchor = next_anchor;
      first = min(first, tile_first);
      last = max(last, tile_last);
    }
    __syncthreads();
  }

  if (half == 0 && rowlive) {
    const int64_t count = int64_t(batch) * h * nq, index = int64_t(bh) * nq + qi;
    Delta[index] = normalizer > 0.f ? mean : 0.f;
    LogZ[index] = normalizer > 0.f ? logf(normalizer) : INFINITY;
    Meta[index] = maximum;
    Meta[count + index] = anchor_dp;
    Meta[2 * count + index] = 1.f;
    Meta[3 * count + index] = float(anchor);
    Meta[4 * count + index] = float(first);
    Meta[5 * count + index] = float(last);
  }
}

// Launch helper: one CTA of 128 threads per 64 query rows and (batch, head).
template <class E, int D, int DV, bool Segments>
inline cudaError_t Launch(const E* q, const E* k, const E* v, const E* g, const int* left,
                   const int* right, const int64_t* q_segment, const int64_t* k_segment,
                   const int* exact, float* delta, float* logz, float* meta, int nq,
                   int nk, int h, int hk, int d, int dv, int batch, float scale,
                   cudaStream_t stream) {
  constexpr int smem = sizeof(Shared<E, D, DV>);
  auto kernel = native_statistics::RowStatistics<E, D, DV, Segments>;
  // The opt-in dynamic shared-memory limit is a device property, so report the
  // requirement here rather than failing later with an opaque launch error.
  const cudaError_t attribute =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attribute != cudaSuccess) return attribute;
  const dim3 grid(unsigned((nq + 63) / 64) * unsigned(batch) * unsigned(h));
  kernel<<<grid, 128, smem, stream>>>(q, k, v, g, left, right, q_segment, k_segment,
                                      exact, delta, logz, meta, nq, nk, h, hk, d, dv,
                                      batch, scale);
  return cudaGetLastError();
}

}  // namespace native_statistics
}  // namespace xattn::ops::attention::precision
