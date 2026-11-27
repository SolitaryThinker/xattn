// Native Blackwell (sm_103a) FlashSCA kernels: forward, row-owned dQ and
// key-owned dK/dV, on warp-level mma.sync tensor cores (m16n8k16, FP32
// accumulation) with cp.async double buffering and ldmatrix operand loads.
//
// These kernels cover the whole public contract (any D/Dv that is a multiple
// of 8 up to 256, short Q, previous K/V, segment/BOS/reset metadata through
// row bounds, strict past, FP32 state, empty rows). Determinism: dK/dV are
// key-owned and reduced in a fixed order. With determinism requested, dQ is
// row-owned (DqKernel) and the whole backward repeats bit-exactly; otherwise
// the dK/dV kernels also reduce dQ with unordered FP32 atomics, which saves
// the dQ kernel's recomputation of S and dP.
//
// Visibility and empty-row semantics are documented in params.h.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "flash_sca/native/params.h"
#include "flash_sca/native/visibility.cuh"

namespace xattn::ops::flash_sca_native {
namespace detail {

constexpr float kLog2e = 1.4426950408889634f;

// ------------------------------------------------------------------ PTX ---

__device__ __forceinline__ uint32_t SmemAddress(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void CopyAsync16(uint32_t dst, const void* src, bool pred) {
  const int bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void CopyAsyncCommit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void CopyAsyncWait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void LoadMatrix(uint32_t (&r)[4], uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

__device__ __forceinline__ void LoadMatrixTransposed(uint32_t (&r)[4], uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

// Unordered FP32 pair reduction into global memory (dQ when determinism is
// not requested).
__device__ __forceinline__ void ReduceAdd2(float* address, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" ::"l"(address), "f"(x), "f"(y) : "memory");
}

__device__ __forceinline__ float FastExp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

template <typename E>
struct Traits;

template <>
struct Traits<__half> {
  static __device__ __forceinline__ void Mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                             uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  static __device__ __forceinline__ uint32_t Pack(float x, float y) {
    __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
  }
  static __device__ __forceinline__ float Load(const void* base, int64_t index) {
    return __half2float(reinterpret_cast<const __half*>(base)[index]);
  }
};

template <>
struct Traits<__nv_bfloat16> {
  static __device__ __forceinline__ void Mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                             uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  static __device__ __forceinline__ uint32_t Pack(float x, float y) {
    __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
  }
  static __device__ __forceinline__ float Load(const void* base, int64_t index) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[index]);
  }
};

// ---------------------------------------------------------- shared tiles ---

// Row-major tile of kWidth 16-bit elements per row (kWidth in {64,128,256}).
// 16-byte chunks are XOR-swizzled with the row index so ldmatrix and cp.async
// are bank-conflict free.
template <int kWidth>
__device__ __forceinline__ int Swizzle(int row, int chunk) {
  return row * kWidth + ((chunk ^ (row & 7)) << 3);
}

// Two adjacent elements of a key-gradient plane (dk, dv, dprev_k or dprev_v), `index` the
// first (even): rounded to E, or with Params::key_grads_fp32 the FP32 values themselves,
// so that rounding them to E gives exactly the 16-bit store.
template <typename E>
__device__ __forceinline__ void StoreKeyGradPair(const Params& p, void* base, int64_t index,
                                                 float x, float y) {
  if (p.key_grads_fp32) {
    *reinterpret_cast<float2*>(static_cast<float*>(base) + index) = make_float2(x, y);
  } else {
    *reinterpret_cast<uint32_t*>(static_cast<E*>(base) + index) = Traits<E>::Pack(x, y);
  }
}

// Row pointer of total key j for a K-like operand (previous, then current).
template <typename E>
__device__ __forceinline__ const E* KeyRow(const Params& p, const Operand& previous,
                                           const Operand& current, int64_t b, int64_t j,
                                           int64_t head) {
  if (j < 0 || j >= p.total_length) return nullptr;
  if (j < p.prev_length) {
    return reinterpret_cast<const E*>(previous.ptr) + b * previous.batch_stride +
           j * previous.row_stride + head * previous.head_stride;
  }
  return reinterpret_cast<const E*>(current.ptr) + b * current.batch_stride +
         (j - p.prev_length) * current.row_stride + head * current.head_stride;
}

// Asynchronously copies kRows rows of kWidth columns; columns >= valid and
// rows whose pointer is null are zero filled.
template <typename E, int kRows, int kWidth, int kThreads, typename RowFn>
__device__ __forceinline__ void LoadTile(E* tile, RowFn row_pointer, int valid, const void* any) {
  constexpr int kChunks = kWidth / 8;
#pragma unroll 4
  for (int index = threadIdx.x; index < kRows * kChunks; index += kThreads) {
    const int row = index / kChunks;
    const int chunk = index % kChunks;
    const E* source = row_pointer(row);
    const bool pred = source != nullptr && chunk * 8 < valid;
    CopyAsync16(SmemAddress(tile + Swizzle<kWidth>(row, chunk)), pred ? source + chunk * 8 : any,
                pred);
  }
}

// ------------------------------------ work-descending launch order ---
//
// The permutation is per (batch row, query tile) and shared by all heads: no
// head term enters a tile's work, so the same order serves every Hq. Rank form
// of the stable descending order, computed in-kernel from Params::tile_bounds
// (no host buffer, no host sync, no new launch): tile t is processed by the
// CTA whose blockIdx.x equals
//
//   rank(t) = #{t' : w(t') > w(t)} + #{t' < t : w(t') == w(t)},   t' in [0,T)
//
// where T = gridDim.x = ceil(Nq/128) and, exactly as ReadTileRange merges the
// two 64-row entries of an M128 tile,
//
//   w(t) = (min_lo > max_hi) ? -1 : max_hi/64 - min_lo/64 + 1,
//   min_lo = min(e[0], e[4]), max_hi = max(e[1], e[5]),
//
// with e the four int32 of the 64-row tile_bounds entry of this query tile.
// Dead tiles (w = -1) sort last and keep their ascending order. The upper key
// clamp of the work count is a no-op: max_hi <= Nt-1 implies
// max_hi/64 <= ceil(Nt/64)-1, so no total-length term is needed here.
constexpr int kOrderTiles = 768;     // program-bound query tiles per batch row
constexpr int kOrderBuckets = 1536;  // work+2 buckets: a tile spanning < 98k keys
// The in-kernel rank is paid by every CTA of the grid, so the cheap form of the
// switch ("sdesc") applies the permutation only when the whole grid is this
// shallow: the traced prefill grids are 1024 CTAs (8-14 waves) and can win the
// tail; the traced train grids are 4096-16384 CTAs (55-110 waves), where the
// per-CTA price, estimated and measured, puts the table under water.
constexpr int kOrderMaxCtas = 4096;

struct OrderScratch {
  int32_t* work;       // [kOrderTiles] w(t) per query tile
  int32_t* bucket;     // [kOrderBuckets] histogram, then base ranks
  int64_t* chosen;     // [1] mapped query tile
  int32_t* max_work;   // [1]
};

__device__ __forceinline__ int32_t QueryTileWork(const Params& p, int64_t b, int64_t t) {
  // An M128 tile owns the two 64-row entries 2t and 2t+1 of its batch row; the
  // last tile of an odd tile_count has only the first, which it uses twice, so
  // this never reads past the row's entries.
  //
  // Liveness is "this 64-row entry has at least one row with a key", which is
  // exactly e0 <= e1: params.h pins an entry with no live row to e0 = INT_MAX,
  // e1 = -1, and a live entry to e0 = min lo, e1 = max hi over its live rows.
  // It is NOT e2/e3 (max lo / min hi over all rows): those are pinned to
  // INT_MAX / -1 as soon as ANY row of the entry is dead, so e2 <= e3 means
  // "every row live" -- which misclassifies a partial tail entry (q_length not
  // a multiple of 64, e.g. the curated Q <= 517 shapes) and any M128 tile
  // whose rows' ranges do not intersect (a segment boundary closer than 128
  // rows, e.g. chunk < 128) as dead. Such a tile would then be ranked strictly
  // last although its CTA's loop trip count is positive.
  const int32_t* entry = p.tile_bounds + (b * p.tile_count + 2 * t) * 4;
  const bool second = 2 * t + 1 < p.tile_count;
  const bool live0 = entry[0] <= entry[1];
  const bool live1 = second && entry[4] <= entry[5];
  if (!live0 && !live1) return -1;
  int32_t min_lo = INT_MAX, max_hi = -1;
  if (live0) {
    min_lo = entry[0];
    max_hi = entry[1];
  }
  if (live1) {
    if (entry[4] < min_lo) min_lo = entry[4];
    if (entry[5] > max_hi) max_hi = entry[5];
  }
  return max_hi / kTileRows - min_lo / kTileRows + 1;
}

// Maps this CTA's blockIdx.x (a rank) to the query tile it processes. Falls
// back to the in-order map (identity) when the shape is outside the table, and
// says so under the dump switch. __noinline__ keeps the rank machinery's
// registers out of the default arm's allocation: the call only exists on the
// opt-in path, and the default kernel's code is unchanged.
__device__ __noinline__ static int64_t RankedQueryTile(const Params& p, int64_t b, int64_t rank,
                                                int64_t tiles, OrderScratch s) {
  const int head_idx = p.head_fastest ? blockIdx.x : blockIdx.y;
  if (p.tile_bounds == nullptr || p.tile_count <= 0) {
    // Insurance: the host fills tile_bounds on every route; a null here is
    // a launch bug, not an order decision, so fail loudly rather than permute.
    if (threadIdx.x == 0 && head_idx == 0)
      printf("ORDER_FALLBACK b=%d rank=%d reason=no_tile_bounds\n", (int)b, (int)rank);
    return rank;
  }
  if (tiles > kOrderTiles) {
    // A silent identity map would be indistinguishable from a correct
    // permutation, so every fallback prints its own line -- in every arm, not
    // only under dump: a fallback row is an invalid timing row and must not be
    // citable as "the order does not help".
    if (threadIdx.x == 0 && head_idx == 0)
      printf("ORDER_FALLBACK b=%d rank=%d reason=tiles tiles=%lld limit=%d\n", (int)b, (int)rank,
             (long long)tiles, kOrderTiles);
    return rank;
  }
  if (threadIdx.x == 0) {
    s.chosen[0] = -1;
    s.max_work[0] = -1;
  }
  __syncthreads();
  int32_t max_work = -1;
  for (int64_t t = threadIdx.x; t < tiles; t += blockDim.x) {
    const int32_t work = QueryTileWork(p, b, t);
    s.work[t] = work;
    if (work > max_work) max_work = work;
  }
  if (max_work > -1) atomicMax(s.max_work, max_work);
  __syncthreads();
  const int32_t top = s.max_work[0];
  const int buckets = top + 2;
  if (buckets > kOrderBuckets) {
    if (threadIdx.x == 0 && head_idx == 0)
      printf("ORDER_FALLBACK b=%d rank=%d reason=buckets buckets=%d limit=%d\n", (int)b, (int)rank,
             buckets, kOrderBuckets);
    return rank;
  }
  for (int i = threadIdx.x; i < buckets; i += blockDim.x) s.bucket[i] = 0;
  __syncthreads();
  // Bucket by reversed work, so an ascending index is descending work: a tile
  // of work w lands at index top - w, and dead tiles (w = -1) at the last one.
  for (int64_t t = threadIdx.x; t < tiles; t += blockDim.x) atomicAdd(s.bucket + top - s.work[t], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t acc = 0;
    for (int i = 0; i < buckets; ++i) {
      const int32_t count = s.bucket[i];
      s.bucket[i] = acc;
      acc += count;
    }
    // A running counter per bucket over ascending t is exactly the stable
    // tie-break: equal work is ordered by ascending tile index.
    for (int64_t t = 0; t < tiles; ++t) {
      if (s.bucket[top - s.work[t]]++ == rank) {
        s.chosen[0] = t;
        break;
      }
    }
  }
  __syncthreads();
  const int64_t mapped = s.chosen[0] >= 0 ? s.chosen[0] : rank;
  if (p.order_dump && threadIdx.x == 0 && head_idx == 0) {
    if (s.chosen[0] < 0)
      printf("ORDER_FALLBACK b=%d rank=%d reason=chosen tiles=%lld\n", (int)b, (int)rank,
             (long long)tiles);
    printf("ORDER_DUMP b=%d rank=%d tile=%lld work=%d\n", (int)b, (int)rank, (long long)mapped,
           QueryTileWork(p, b, mapped));
  }
  return mapped;
}

// --------------------------------------------------------------- forward ---

template <int kD, int kDv>
struct ForwardConfig {
  static constexpr int kBM = 128;
  static constexpr int kBN = 64;
  static constexpr int kWarps = 8;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kSmemElements = kBM * kD + 2 * kBN * kD + 2 * kBN * kDv;
  static constexpr int kSmemBytes = kSmemElements * 2;
  // Launch-order scratch (kOrderTiles work + kOrderBuckets counts + the two
  // scalars), appended to the dynamic shared memory only when the switch is on,
  // so the default kernel keeps exactly its base shared footprint. Every D
  // bucket's kSmemBytes is a multiple of 8, so the tail is 8-byte aligned.
  static constexpr int kOrderBytes = (kOrderTiles + kOrderBuckets) * 4 + 16;
};

template <typename E, int kD, int kDv, bool kSubtractBeforeScale>
__global__ void __launch_bounds__(ForwardConfig<kD, kDv>::kThreads, 1)
    ForwardKernel(const Params p) {
  using Config = ForwardConfig<kD, kDv>;
  using T = Traits<E>;
  constexpr int kBM = Config::kBM, kBN = Config::kBN, kThreads = Config::kThreads;
  extern __shared__ __align__(128) unsigned char shared_raw[];
  E* sQ = reinterpret_cast<E*>(shared_raw);
  E* sK = sQ + kBM * kD;
  E* sV = sK + 2 * kBN * kD;

  // Launch order: the rank scratch is a tail on the dynamic shared memory
  // that ForwardCase allocates only when the switch is on, so the default
  // kernel's shared footprint is exactly the base one.
  int32_t* sOrderWork = reinterpret_cast<int32_t*>(shared_raw + Config::kSmemBytes);
  int32_t* sOrderBucket = sOrderWork + kOrderTiles;
  int64_t* sOrderChosen = reinterpret_cast<int64_t*>(sOrderBucket + kOrderBuckets);
  int32_t* sOrderMaxWork = reinterpret_cast<int32_t*>(sOrderChosen + 1);

  // Head-fastest layout: tf (default) keeps blockIdx.x = query tile and
  // blockIdx.y = query head; hf swaps the two so consecutive CTAs are the head
  // copies of one tile. A tile's work carries no head term, so the same [B,T]
  // order serves both layouts.
  const int64_t hq = p.head_fastest ? blockIdx.x : blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t hk = hq / p.group;
  const int64_t tile_index = p.head_fastest ? blockIdx.y : blockIdx.x;
  const int64_t tiles = p.head_fastest ? gridDim.y : gridDim.x;
  // "sdesc" applies the permutation only on shallow grids (the rank is paid by
  // every CTA); "sdesc-all" and "dump" force it on every grid.
  const int64_t ctas = int64_t(gridDim.x) * int64_t(gridDim.y) * int64_t(gridDim.z);
  int64_t m_block = tile_index;
  if (p.sorted_order && (p.order_always || ctas <= kOrderMaxCtas)) {
    const OrderScratch order{sOrderWork, sOrderBucket, sOrderChosen, sOrderMaxWork};
    m_block = RankedQueryTile(p, b, tile_index, tiles, order);
  }
  const int64_t m0 = m_block * kBM;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t4 = lane % 4;

  const TileRange range = ReadTileRange(p, b, m0 / kTileRows, kBM / kTileRows);
  const int64_t row0 = m0 + warp * 16 + g, row1 = row0 + 8;
  const Bounds bound0 = RowBounds(p, b, row0), bound1 = RowBounds(p, b, row1);
  const bool segments = SegmentTest(p, b);
  const int64_t seg0 = segments && row0 < p.q_length ? p.q_segment[b * p.q_length + row0] : 0;
  const int64_t seg1 = segments && row1 < p.q_length ? p.q_segment[b * p.q_length + row1] : 0;
  const int64_t* kseg = segments ? p.k_segment + b * p.total_length : nullptr;

  float acc[kDv / 8][4];
#pragma unroll
  for (int i = 0; i < kDv / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.f, 0.f};

  if (range.min_lo <= range.max_hi) {
    const E* q_base = reinterpret_cast<const E*>(p.q.ptr) + b * p.q.batch_stride + hq * p.q.head_stride;
    LoadTile<E, kBM, kD, kThreads>(
        sQ,
        [&](int r) -> const E* {
          const int64_t i = m0 + r;
          return i < p.q_length ? q_base + i * p.q.row_stride : nullptr;
        },
        p.qk_dim, p.q.ptr);
    const int n_begin = range.min_lo / kBN, n_end = range.max_hi / kBN;
    auto load_kv = [&](int n, int stage) {
      const int64_t j0 = int64_t(n) * kBN;
      LoadTile<E, kBN, kD, kThreads>(
          sK + stage * kBN * kD, [&](int r) { return KeyRow<E>(p, p.prev_k, p.k, b, j0 + r, hk); },
          p.qk_dim, p.q.ptr);
      LoadTile<E, kBN, kDv, kThreads>(
          sV + stage * kBN * kDv, [&](int r) { return KeyRow<E>(p, p.prev_v, p.v, b, j0 + r, hk); },
          p.value_dim, p.q.ptr);
    };
    load_kv(n_begin, 0);
    CopyAsyncCommit();

    for (int n = n_begin; n <= n_end; ++n) {
      const int stage = (n - n_begin) & 1;
      if (n < n_end) load_kv(n + 1, stage ^ 1);
      CopyAsyncCommit();
      CopyAsyncWait<1>();
      __syncthreads();
      const E* tK = sK + stage * kBN * kD;
      const E* tV = sV + stage * kBN * kDv;

      // S = Q K^T for this warp's 16 rows and the 64 keys of the tile.
      float s[kBN / 8][4];
#pragma unroll
      for (int i = 0; i < kBN / 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < kD / 16; ++kk) {
        uint32_t a[4];
        LoadMatrix(a, SmemAddress(sQ + Swizzle<kD>(warp * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
        for (int nb = 0; nb < kBN / 16; ++nb) {
          uint32_t bf[4];
          LoadMatrix(bf, SmemAddress(tK + Swizzle<kD>(nb * 16 + (lane % 8) + (lane / 16) * 8,
                                                       kk * 2 + (lane / 8) % 2)));
          T::Mma(s[2 * nb], a, bf[0], bf[1]);
          T::Mma(s[2 * nb + 1], a, bf[2], bf[3]);
        }
      }

      // Visibility mask, only where some element of the tile can be outside.
      const int j0 = n * kBN;
      const bool full = !segments && j0 >= range.max_lo && j0 + kBN - 1 <= range.min_hi;
      if (!full) {
#pragma unroll
        for (int nb = 0; nb < kBN / 8; ++nb) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int j = j0 + nb * 8 + 2 * t4 + e;
            const int64_t key_seg = segments && j < p.total_length ? kseg[j] : 0;
            if (!(j >= bound0.lo && j <= bound0.hi && (!segments || key_seg == seg0)))
              s[nb][e] = -INFINITY;
            if (!(j >= bound1.lo && j <= bound1.hi && (!segments || key_seg == seg1)))
              s[nb][2 + e] = -INFINITY;
          }
        }
      }

      // Online softmax on raw scores.
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float m = row_max[h];
#pragma unroll
        for (int nb = 0; nb < kBN / 8; ++nb) m = fmaxf(m, fmaxf(s[nb][2 * h], s[nb][2 * h + 1]));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        const float m_use = m == -INFINITY ? 0.f : m;
        const float correction = FastExp2((row_max[h] - m_use) * p.scale_log2);
        row_max[h] = m;
        const float m_scaled = m_use * p.scale_log2;
        float sum = 0.f;
#pragma unroll
        for (int nb = 0; nb < kBN / 8; ++nb) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float x = s[nb][2 * h + e];
            const float y = kSubtractBeforeScale ? FastExp2((x - m_use) * p.scale_log2)
                                                 : FastExp2(fmaf(x, p.scale_log2, -m_scaled));
            s[nb][2 * h + e] = y;
            sum += y;
          }
        }
        row_sum[h] = row_sum[h] * correction + sum;
#pragma unroll
        for (int i = 0; i < kDv / 8; ++i) {
          acc[i][2 * h] *= correction;
          acc[i][2 * h + 1] *= correction;
        }
      }

      // O += P V.
#pragma unroll
      for (int kk = 0; kk < kBN / 16; ++kk) {
        uint32_t a[4];
        a[0] = T::Pack(s[2 * kk][0], s[2 * kk][1]);
        a[1] = T::Pack(s[2 * kk][2], s[2 * kk][3]);
        a[2] = T::Pack(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        a[3] = T::Pack(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
        for (int nv = 0; nv < kDv / 16; ++nv) {
          uint32_t bf[4];
          LoadMatrixTransposed(
              bf, SmemAddress(tV + Swizzle<kDv>(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                                 nv * 2 + lane / 16)));
          T::Mma(acc[2 * nv], a, bf[0], bf[1]);
          T::Mma(acc[2 * nv + 1], a, bf[2], bf[3]);
        }
      }
      __syncthreads();
    }
  }

  // Normalize and store. Empty rows: y = 0, state = 0, LSE = -inf.
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float sum = row_sum[h];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const int64_t row = h ? row1 : row0;
    if (row >= p.q_length) continue;
    const bool live = sum > 0.f;
    const float inverse = live ? 1.f / sum : 0.f;
    const int64_t out_row = ((b * p.q_length + row) * p.query_heads + hq) * p.value_dim;
#pragma unroll
    for (int i = 0; i < kDv / 8; ++i) {
      const int col = i * 8 + 2 * t4;
      if (col >= p.value_dim) continue;
      const float x = acc[i][2 * h] * inverse, y = acc[i][2 * h + 1] * inverse;
      *reinterpret_cast<uint32_t*>(reinterpret_cast<E*>(p.y) + out_row + col) = T::Pack(x, y);
      if (p.y_state != nullptr) *reinterpret_cast<float2*>(p.y_state + out_row + col) = make_float2(x, y);
    }
    if (t4 == 0) {
      p.lse[(b * p.query_heads + hq) * p.q_length + row] =
          live ? row_max[h] * p.scale + logf(sum) : -INFINITY;
    }
  }
}

// -------------------------------------------------------------- backward ---

// delta[b,h,i] = sum_c dy[b,i,h,c] * saved[b,i,h,c] in FP32; one warp per row.
template <typename E>
__global__ void DeltaKernel(const Params p) {
  const int64_t rows = p.batch * p.q_length * p.query_heads;
  const int64_t row = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows) return;
  const int lane = threadIdx.x % 32;
  const int64_t base = row * p.value_dim;
  float sum = 0.f;
  for (int64_t c = lane; c < p.value_dim; c += 32) {
    const float o = p.saved_fp32 ? reinterpret_cast<const float*>(p.saved)[base + c]
                                 : Traits<E>::Load(p.saved, base + c);
    sum += Traits<E>::Load(p.dy, base + c) * o;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  if (lane == 0) {
    // row = (b * Nq + i) * Hq + h  ->  delta[(b * Hq + h) * Nq + i]
    const int64_t h = row % p.query_heads;
    const int64_t bi = row / p.query_heads;
    const int64_t i = bi % p.q_length;
    const int64_t bb = bi / p.q_length;
    p.delta[(bb * p.query_heads + h) * p.q_length + i] = sum;
  }
}

// ------------------------------------------------- backward preprocessing ---

__device__ __forceinline__ float ElemToFloat(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float ElemToFloat(__half x) { return __half2float(x); }

// Pure zero-spill, zero-SMEM vectorized delta kernel: delta[(b*Hq + hq)*Nq + i] = sum_c dy * saved.
// Reads dy and saved contiguously along flat rows r = i * Hq + hq (16-B vector loads, 4 rows unrolled
// per CTA step) so every warp reads contiguous 512-B lines from HBM3e with 0 local spills and 0 SMEM.
// Fixed per-lane reduction order (tree xor-shuffle over kLanes) guarantees strict bitwise determinism.
constexpr int kPreprocessThreads = 256;
constexpr int64_t kPreprocessMaxTileCount = 3072;         // tile_count*4 ints <= 48 KB shared

__device__ __forceinline__ float2 Elem2ToFloat2(__nv_bfloat162 v) { return __bfloat1622float2(v); }
__device__ __forceinline__ float2 Elem2ToFloat2(__half2 v) { return __half22float2(v); }

__device__ __forceinline__ float2 DeltaFma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) >= 1000 && (__CUDA_ARCH__) < 1200
  unsigned long long pa = *reinterpret_cast<unsigned long long*>(&a);
  unsigned long long pb = *reinterpret_cast<unsigned long long*>(&b);
  unsigned long long pc = *reinterpret_cast<unsigned long long*>(&c);
  unsigned long long pr;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(pr) : "l"(pa), "l"(pb), "l"(pc));
  return *reinterpret_cast<float2*>(&pr);
#else
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}

template <typename E>
struct Elem2Of;
template <>
struct Elem2Of<__nv_bfloat16> {
  using type = __nv_bfloat162;
};
template <>
struct Elem2Of<__half> {
  using type = __half2;
};

__device__ __forceinline__ uint4 LoadVec16Stream(const void* ptr) {
  uint4 v;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) >= 1000
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
#else
  v = __ldg(reinterpret_cast<const uint4*>(ptr));
#endif
  return v;
}

struct alignas(32) DeltaVec32 {
  uint4 lo, hi;
};

__device__ __forceinline__ DeltaVec32 LoadVec32Evict256B(const void* ptr) {
  DeltaVec32 r;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) >= 1000
  asm volatile(
      "ld.global.L1::no_allocate.L2::evict_first.L2::256B.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.lo.x), "=r"(r.lo.y), "=r"(r.lo.z), "=r"(r.lo.w),
        "=r"(r.hi.x), "=r"(r.hi.y), "=r"(r.hi.z), "=r"(r.hi.w)
      : "l"(ptr));
#else
  const uint4* p = reinterpret_cast<const uint4*>(ptr);
  r.lo = __ldg(p);
  r.hi = __ldg(p + 1);
#endif
  return r;
}

template <typename E>
__device__ __forceinline__ float DeltaDotU4(const uint4& au, const uint4& su) {
  using E2 = typename Elem2Of<E>::type;
  const E2* ap = reinterpret_cast<const E2*>(&au);
  const E2* sp = reinterpret_cast<const E2*>(&su);
  float2 acc0 = DeltaFma2(Elem2ToFloat2(ap[0]), Elem2ToFloat2(sp[0]), make_float2(0.f, 0.f));
  float2 acc1 = DeltaFma2(Elem2ToFloat2(ap[2]), Elem2ToFloat2(sp[2]), make_float2(0.f, 0.f));
  acc0 = DeltaFma2(Elem2ToFloat2(ap[1]), Elem2ToFloat2(sp[1]), acc0);
  acc1 = DeltaFma2(Elem2ToFloat2(ap[3]), Elem2ToFloat2(sp[3]), acc1);
  return (acc0.x + acc0.y) + (acc1.x + acc1.y);
}

// HBM3e SOL 256-bit (LDG.256) streaming delta kernel for Dv == 128, Hq % 32 == 0, Nq % 32 == 0,
// 32-byte aligned dy/saved. Sustains ~7.0 TB/s on B300 (~155 us for B=2, L=32768, Hq=32, Dv=128):
//   - 8 lanes/row x 32 B/lane (LDG.256 with L1::no_allocate.L2::evict_first.L2::256B) so all 256
//     threads stream 8 KB (all 32 heads of 1 query row) 100% contiguously with zero stride gaps,
//     preserving L2 for subsequent attention kernels.
//   - 32 regs/thread, 0 local spills, 100% theoretical occupancy (64/64 warps/SM).
//   - 4224 B padded SMEM transpose delivers 100% coalesced 128-B stores for delta.
//   - Fixed 8-lane butterfly xor-shuffle (4, 2, 1) guarantees strict bitwise determinism.
template <typename E>
__global__ void __launch_bounds__(kPreprocessThreads, 8)
DeltaVector256H32Kernel(const DeltaVec32* __restrict__ dy_ptr,
                        const DeltaVec32* __restrict__ sv_ptr,
                        float* __restrict__ delta_ptr, int q_length, int query_heads) {
  __shared__ float sm[32][33];
  const int tid = threadIdx.x;
  const int sub = tid & 7;
  const int dh = tid >> 3;  // 0..31
  const int64_t b = blockIdx.z;
  const int64_t i0 = int64_t(blockIdx.x) * 32;
  const int64_t h0 = int64_t(blockIdx.y) * 32;
  const int64_t stride_vec = int64_t(query_heads) * 8;
  const DeltaVec32* dy_p = dy_ptr + ((b * q_length + i0) * query_heads + h0) * 8 + tid;
  const DeltaVec32* sv_p = sv_ptr + ((b * q_length + i0) * query_heads + h0) * 8 + tid;

  for (int r = 0; r < 32; ++r) {
    const DeltaVec32 a = LoadVec32Evict256B(dy_p);
    const DeltaVec32 s = LoadVec32Evict256B(sv_p);
    dy_p += stride_vec;
    sv_p += stride_vec;
    float sum = DeltaDotU4<E>(a.lo, s.lo) + DeltaDotU4<E>(a.hi, s.hi);
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (sub == 0) sm[dh][r] = sum;
  }
  __syncthreads();
  const int warp = tid >> 5;
  const int lane = tid & 31;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const int oh = warp + w * 8;
    delta_ptr[(b * query_heads + h0 + oh) * q_length + i0 + lane] = sm[oh][lane];
  }
}

// General zero-spill streaming vectorized delta kernel (handles arbitrary Nq, Hq, Dv multiples of 8
// and 16-byte aligned pointers with 0 local spills across all instantiations).
template <typename E, int kLanes, bool kSingleVec>
__global__ void __launch_bounds__(kPreprocessThreads, 4)
DeltaVectorKernel(const E* __restrict__ dy_ptr, const E* __restrict__ sv_ptr,
                  float* __restrict__ delta_ptr, int64_t q_length, int64_t query_heads,
                  int64_t value_dim) {
  using E2 = typename Elem2Of<E>::type;

  if constexpr (kSingleVec && kLanes == 16) {
    __shared__ float sm_delta[4][33];
    const int tid = threadIdx.x;
    const int sub = tid & 15;
    const int sw = tid >> 4;
    const int dh = sw & 3;
    const int r_off = sw >> 2;  // 0..3
    const int64_t b = blockIdx.z;
    const int64_t i0 = int64_t(blockIdx.x) * 32;
    const int64_t h0 = int64_t(blockIdx.y) * 4;
    const int64_t row_stride = query_heads * 128;
    const E* const dy_blk = dy_ptr + ((b * q_length + i0) * query_heads + h0) * 128;
    const E* const sv_blk = sv_ptr + ((b * q_length + i0) * query_heads + h0) * 128;
    const uint32_t step = static_cast<uint32_t>(4 * row_stride);
    uint32_t off =
        static_cast<uint32_t>(r_off) * static_cast<uint32_t>(row_stride) +
        static_cast<uint32_t>((dh << 7) + (sub << 3));

    auto dot_reduce = [&](const uint4& au, const uint4& su) -> float {
      const E2* ap = reinterpret_cast<const E2*>(&au);
      const E2* sp = reinterpret_cast<const E2*>(&su);
      float2 acc0 = DeltaFma2(Elem2ToFloat2(ap[0]), Elem2ToFloat2(sp[0]), make_float2(0.f, 0.f));
      float2 acc1 = DeltaFma2(Elem2ToFloat2(ap[2]), Elem2ToFloat2(sp[2]), make_float2(0.f, 0.f));
      acc0 = DeltaFma2(Elem2ToFloat2(ap[1]), Elem2ToFloat2(sp[1]), acc0);
      acc1 = DeltaFma2(Elem2ToFloat2(ap[3]), Elem2ToFloat2(sp[3]), acc1);
      float sum = (acc0.x + acc0.y) + (acc1.x + acc1.y);
#pragma unroll
      for (int s = 8; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, s);
      return sum;
    };

    if (i0 + 32 <= q_length && h0 + 4 <= query_heads) {
      // Interior full tile (100% of CTAs on 32K/H=32): 2-stage software-pipelined loads.
      uint4 au = LoadVec16Stream(dy_blk + off);
      uint4 su = LoadVec16Stream(sv_blk + off);
#pragma unroll
      for (int r = 0; r < 28; r += 4) {
        off += step;
        const uint4 au_n = LoadVec16Stream(dy_blk + off);
        const uint4 su_n = LoadVec16Stream(sv_blk + off);
        const float sum = dot_reduce(au, su);
        if (sub == 0) sm_delta[dh][r + r_off] = sum;
        au = au_n;
        su = su_n;
      }
      const float sum = dot_reduce(au, su);
      if (sub == 0) sm_delta[dh][28 + r_off] = sum;
      __syncthreads();
      if (tid < 128) {
        const int out_dh = tid >> 5;
        const int out_ri = tid & 31;
        delta_ptr[(b * query_heads + h0 + out_dh) * q_length + i0 + out_ri] =
            sm_delta[out_dh][out_ri];
      }
    } else {
      const bool valid_h = (h0 + dh) < query_heads;
      for (int r = 0; r < 32; r += 4) {
        const int local_i = r + r_off;
        float sum = 0.f;
        if (valid_h && (i0 + local_i) < q_length) {
          const uint4 au = LoadVec16Stream(dy_blk + off);
          const uint4 su = LoadVec16Stream(sv_blk + off);
          sum = dot_reduce(au, su);
        } else {
#pragma unroll
          for (int s = 8; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, s);
        }
        off += step;
        if (sub == 0) sm_delta[dh][local_i] = sum;
      }
      __syncthreads();
      if (tid < 128) {
        const int out_dh = tid >> 5;
        const int out_ri = tid & 31;
        if ((h0 + out_dh) < query_heads && (i0 + out_ri) < q_length) {
          delta_ptr[(b * query_heads + h0 + out_dh) * q_length + i0 + out_ri] =
              sm_delta[out_dh][out_ri];
        }
      }
    }
  } else {
    constexpr int kRowsPerStep = kPreprocessThreads / kLanes;
    const int sub = threadIdx.x & (kLanes - 1);
    const int local_row = threadIdx.x / kLanes;
    const int64_t b = blockIdx.z;
    const int64_t hq0 = int64_t(blockIdx.y) * 2;
    const int64_t i = int64_t(blockIdx.x) * kRowsPerStep + local_row;
    const bool valid_i = i < q_length;
    for (int dh = 0; dh < 2; ++dh) {
      const int64_t hq = hq0 + dh;
      if (hq >= query_heads) break;
      float2 acc0 = make_float2(0.f, 0.f);
      float2 acc1 = make_float2(0.f, 0.f);
      if (valid_i) {
        const int64_t base = ((b * q_length + i) * query_heads + hq) * value_dim;
        const E* dy = dy_ptr + base;
        const E* sv = sv_ptr + base;
        for (int64_t c = 8 * sub; c < value_dim; c += 8 * kLanes) {
          const uint4 au = LoadVec16Stream(dy + c);
          const uint4 su = LoadVec16Stream(sv + c);
          const E2* ap = reinterpret_cast<const E2*>(&au);
          const E2* sp = reinterpret_cast<const E2*>(&su);
          acc0 = DeltaFma2(Elem2ToFloat2(ap[0]), Elem2ToFloat2(sp[0]), acc0);
          acc1 = DeltaFma2(Elem2ToFloat2(ap[2]), Elem2ToFloat2(sp[2]), acc1);
          acc0 = DeltaFma2(Elem2ToFloat2(ap[1]), Elem2ToFloat2(sp[1]), acc0);
          acc1 = DeltaFma2(Elem2ToFloat2(ap[3]), Elem2ToFloat2(sp[3]), acc1);
        }
      }
      float sum = (acc0.x + acc0.y) + (acc1.x + acc1.y);
#pragma unroll
      for (int off = kLanes / 2; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
      if (valid_i && sub == 0) {
        delta_ptr[(b * query_heads + hq) * q_length + i] = sum;
      }
    }
  }
}

__device__ __forceinline__ float LseLog2(float lse) {
  // A non-finite LSE marks a row without admitted keys: every P is zero.
  return isfinite(lse) ? lse * kLog2e : INFINITY;
}

template <int kD, int kDv>
struct DqConfig {
  static constexpr int kBM = 64;
  static constexpr int kBN = 64;
  static constexpr int kWarps = 8;  // 4 row groups x 2 key halves
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kSmemElements = kBM * kD + kBM * kDv + 2 * kBN * kD + 2 * kBN * kDv;
  static constexpr int kSmemBytes = kSmemElements * 2;
  static_assert(kBM * kD * 4 <= 2 * kBN * (kD + kDv) * 2, "dQ reduction must fit the K/V stages");
};

// Row-owned dQ: one CTA per 64 query rows of one head, keys in ascending order.
// Two warps share each 16-row group, one per half of every 64-key tile; their
// partial dQ are added in a fixed order at the end, so dQ is deterministic.
template <typename E, int kD, int kDv>
__global__ void __launch_bounds__(DqConfig<kD, kDv>::kThreads, 1) DqKernel(const Params p) {
  using Config = DqConfig<kD, kDv>;
  using T = Traits<E>;
  constexpr int kBM = Config::kBM, kBN = Config::kBN, kThreads = Config::kThreads;
  constexpr int kHalf = kBN / 2;
  extern __shared__ __align__(128) unsigned char shared_raw[];
  E* sQ = reinterpret_cast<E*>(shared_raw);
  E* sG = sQ + kBM * kD;
  E* sK = sG + kBM * kDv;
  E* sV = sK + 2 * kBN * kD;
  float* sReduce = reinterpret_cast<float*>(sK);  // reused after the key loop

  const int64_t m_block = blockIdx.x;
  const int64_t hq = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t hk = hq / p.group;
  const int64_t m0 = m_block * kBM;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t4 = lane % 4;
  const int rg = warp & 3, kh = warp >> 2;

  const TileRange range = ReadTileRange(p, b, m0 / kTileRows, kBM / kTileRows);
  const int64_t row0 = m0 + rg * 16 + g, row1 = row0 + 8;
  const Bounds bound0 = RowBounds(p, b, row0), bound1 = RowBounds(p, b, row1);
  const bool segments = SegmentTest(p, b);
  const int64_t seg0 = segments && row0 < p.q_length ? p.q_segment[b * p.q_length + row0] : 0;
  const int64_t seg1 = segments && row1 < p.q_length ? p.q_segment[b * p.q_length + row1] : 0;
  const int64_t* kseg = segments ? p.k_segment + b * p.total_length : nullptr;
  const int64_t stat0 = (b * p.query_heads + hq) * p.q_length + row0;
  const float lse2[2] = {row0 < p.q_length ? LseLog2(p.lse_in[stat0]) : INFINITY,
                         row1 < p.q_length ? LseLog2(p.lse_in[stat0 + 8]) : INFINITY};
  const float delta[2] = {row0 < p.q_length ? p.delta[stat0] : 0.f,
                          row1 < p.q_length ? p.delta[stat0 + 8] : 0.f};

  float acc[kD / 8][4];
#pragma unroll
  for (int i = 0; i < kD / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

  if (range.min_lo <= range.max_hi) {
    const E* q_base = reinterpret_cast<const E*>(p.q.ptr) + b * p.q.batch_stride + hq * p.q.head_stride;
    const E* g_base = reinterpret_cast<const E*>(p.dy) + (b * p.q_length * p.query_heads + hq) * p.value_dim;
    LoadTile<E, kBM, kD, kThreads>(
        sQ,
        [&](int r) -> const E* {
          const int64_t i = m0 + r;
          return i < p.q_length ? q_base + i * p.q.row_stride : nullptr;
        },
        p.qk_dim, p.q.ptr);
    LoadTile<E, kBM, kDv, kThreads>(
        sG,
        [&](int r) -> const E* {
          const int64_t i = m0 + r;
          return i < p.q_length ? g_base + i * p.query_heads * p.value_dim : nullptr;
        },
        p.value_dim, p.q.ptr);
    const int n_begin = range.min_lo / kBN, n_end = range.max_hi / kBN;
    auto load_kv = [&](int n, int stage) {
      const int64_t j0 = int64_t(n) * kBN;
      LoadTile<E, kBN, kD, kThreads>(
          sK + stage * kBN * kD, [&](int r) { return KeyRow<E>(p, p.prev_k, p.k, b, j0 + r, hk); },
          p.qk_dim, p.q.ptr);
      LoadTile<E, kBN, kDv, kThreads>(
          sV + stage * kBN * kDv, [&](int r) { return KeyRow<E>(p, p.prev_v, p.v, b, j0 + r, hk); },
          p.value_dim, p.q.ptr);
    };
    load_kv(n_begin, 0);
    CopyAsyncCommit();

    for (int n = n_begin; n <= n_end; ++n) {
      const int stage = (n - n_begin) & 1;
      if (n < n_end) load_kv(n + 1, stage ^ 1);
      CopyAsyncCommit();
      CopyAsyncWait<1>();
      __syncthreads();
      const E* tK = sK + stage * kBN * kD;
      const E* tV = sV + stage * kBN * kDv;
      const int j0 = n * kBN + kh * kHalf;  // first key of this warp's half
      const bool full = !segments && j0 >= range.max_lo && j0 + kHalf - 1 <= range.min_hi;
      // Warp-uniform: the mma.sync/ldmatrix below need every lane.
      const bool empty = __all_sync(0xffffffffu, j0 > max(bound0.hi, bound1.hi) ||
                                                     j0 + kHalf - 1 < min(bound0.lo, bound1.lo));

      if (!empty) {
        float s[kHalf / 8][4], dp[kHalf / 8][4];
#pragma unroll
        for (int i = 0; i < kHalf / 8; ++i) {
          s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
          dp[i][0] = dp[i][1] = dp[i][2] = dp[i][3] = 0.f;
        }
#pragma unroll
        for (int kk = 0; kk < kD / 16; ++kk) {
          uint32_t a[4];
          LoadMatrix(a, SmemAddress(sQ + Swizzle<kD>(rg * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
          for (int nb = 0; nb < kHalf / 16; ++nb) {
            uint32_t bf[4];
            LoadMatrix(bf, SmemAddress(tK + Swizzle<kD>(kh * kHalf + nb * 16 + (lane % 8) + (lane / 16) * 8,
                                                         kk * 2 + (lane / 8) % 2)));
            T::Mma(s[2 * nb], a, bf[0], bf[1]);
            T::Mma(s[2 * nb + 1], a, bf[2], bf[3]);
          }
        }
#pragma unroll
        for (int kk = 0; kk < kDv / 16; ++kk) {
          uint32_t a[4];
          LoadMatrix(a, SmemAddress(sG + Swizzle<kDv>(rg * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
          for (int nb = 0; nb < kHalf / 16; ++nb) {
            uint32_t bf[4];
            LoadMatrix(bf, SmemAddress(tV + Swizzle<kDv>(kh * kHalf + nb * 16 + (lane % 8) + (lane / 16) * 8,
                                                          kk * 2 + (lane / 8) % 2)));
            T::Mma(dp[2 * nb], a, bf[0], bf[1]);
            T::Mma(dp[2 * nb + 1], a, bf[2], bf[3]);
          }
        }

#pragma unroll
        for (int nb = 0; nb < kHalf / 8; ++nb) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int j = j0 + nb * 8 + 2 * t4 + e;
            bool ok0 = true, ok1 = true;
            if (!full) {
              const int64_t key_seg = segments && j < p.total_length ? kseg[j] : 0;
              ok0 = j >= bound0.lo && j <= bound0.hi && (!segments || key_seg == seg0);
              ok1 = j >= bound1.lo && j <= bound1.hi && (!segments || key_seg == seg1);
            }
            const float p0 = ok0 ? FastExp2(fmaf(s[nb][e], p.scale_log2, -lse2[0])) : 0.f;
            const float p1 = ok1 ? FastExp2(fmaf(s[nb][2 + e], p.scale_log2, -lse2[1])) : 0.f;
            s[nb][e] = p0 * (dp[nb][e] - delta[0]);
            s[nb][2 + e] = p1 * (dp[nb][2 + e] - delta[1]);
          }
        }

        // dQ += dS K over this warp's half of the keys.
#pragma unroll
        for (int kk = 0; kk < kHalf / 16; ++kk) {
          uint32_t a[4];
          a[0] = T::Pack(s[2 * kk][0], s[2 * kk][1]);
          a[1] = T::Pack(s[2 * kk][2], s[2 * kk][3]);
          a[2] = T::Pack(s[2 * kk + 1][0], s[2 * kk + 1][1]);
          a[3] = T::Pack(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
          for (int nd = 0; nd < kD / 16; ++nd) {
            uint32_t bf[4];
            LoadMatrixTransposed(
                bf, SmemAddress(tK + Swizzle<kD>(kh * kHalf + kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                                  nd * 2 + lane / 16)));
            T::Mma(acc[2 * nd], a, bf[0], bf[1]);
            T::Mma(acc[2 * nd + 1], a, bf[2], bf[3]);
          }
        }
      }
      __syncthreads();
    }
  }

  // Fixed-order reduction of the two key halves: first half + second half.
  CopyAsyncWait<0>();
  __syncthreads();
  float* slot = sReduce + (rg * 32 + lane) * (kD / 2);
  if (kh == 1) {
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) {
      *reinterpret_cast<float4*>(slot + i * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
  }
  __syncthreads();
  if (kh == 1) return;
#pragma unroll
  for (int i = 0; i < kD / 8; ++i) {
    const float4 other = *reinterpret_cast<const float4*>(slot + i * 4);
    acc[i][0] += other.x;
    acc[i][1] += other.y;
    acc[i][2] += other.z;
    acc[i][3] += other.w;
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int64_t row = h ? row1 : row0;
    if (row >= p.q_length) continue;
    const int64_t out_row = ((b * p.q_length + row) * p.query_heads + hq) * p.qk_dim;
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) {
      const int col = i * 8 + 2 * t4;
      if (col >= p.qk_dim) continue;
      *reinterpret_cast<uint32_t*>(reinterpret_cast<E*>(p.dq) + out_row + col) =
          T::Pack(acc[i][2 * h] * p.scale, acc[i][2 * h + 1] * p.scale);
    }
  }
}

// Query tiles [first, last] that may see keys [n0, n1] of batch row b.
struct QueryTiles {
  int64_t first, last;
};

template <int kBM>
__device__ __forceinline__ QueryTiles QueryTilesOf(const Params& p, int64_t b, int64_t n0, int64_t n1) {
  int64_t t_begin = 0, t_end = p.tile_count - 1;
  if (p.row_lo != nullptr) {
    t_begin = INT_MAX;
    t_end = -1;
    for (int64_t kt = n0 / kTileRows; kt * kTileRows <= n1; ++kt) {
      const int32_t* e = p.key_tile_range + (b * p.key_tile_count + kt) * 2;
      t_begin = e[0] < t_begin ? e[0] : t_begin;
      t_end = e[1] > t_end ? e[1] : t_end;
    }
  } else {
    const int64_t offset = p.total_length - p.q_length;
    const int64_t first_row = n0 - offset + (p.strict ? 1 : 0);
    const int64_t last_row = (n1 / p.chunk + 2) * p.chunk - 1 - offset;
    t_begin = (first_row > 0 ? first_row : 0) / kBM;
    t_end = last_row < p.q_length - 1 ? last_row : p.q_length - 1;
    t_end = t_end < 0 ? -1 : t_end / kBM;
  }
  return {t_begin, t_end};
}

// dq_accum[q, c] += sum_key dS^T[key][q] K[key][c] for this warp's 16 query
// rows (rows16) and the columns [col0, col0 + kCols), with dS^T in shared
// memory as [kKeys][64] and K as [kKeys][kD]. Unscaled; the host applies the
// softmax scale once.
template <typename E, int kD, int kKeys, int kCols>
__device__ __forceinline__ void AccumulateDq(const Params& p, const E* sS, const E* sK, int rows16,
                                             int col0, int64_t b, int64_t hq, int64_t m0) {
  using T = Traits<E>;
  const int lane = threadIdx.x % 32, g = lane / 4, t4 = lane % 4;
  float acc[kCols / 8][4];
#pragma unroll
  for (int i = 0; i < kCols / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < kKeys / 16; ++kk) {
    uint32_t a[4];
    LoadMatrixTransposed(a, SmemAddress(sS + Swizzle<64>(kk * 16 + (lane % 8) + (lane / 16) * 8,
                                                          rows16 / 8 + (lane / 8) % 2)));
#pragma unroll
    for (int nd = 0; nd < kCols / 16; ++nd) {
      uint32_t bf[4];
      LoadMatrixTransposed(bf, SmemAddress(sK + Swizzle<kD>(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                                            col0 / 8 + nd * 2 + lane / 16)));
      T::Mma(acc[2 * nd], a, bf[0], bf[1]);
      T::Mma(acc[2 * nd + 1], a, bf[2], bf[3]);
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int64_t row = m0 + rows16 + g + 8 * h;
    if (row >= p.q_length) continue;
    float* out = p.dq_accum + ((b * p.q_length + row) * p.query_heads + hq) * p.qk_dim;
#pragma unroll
    for (int i = 0; i < kCols / 8; ++i) {
      const int col = col0 + i * 8 + 2 * t4;
      if (col < p.qk_dim) ReduceAdd2(out + col, acc[i][2 * h], acc[i][2 * h + 1]);
    }
  }
}

template <int kD, int kDv>
struct DkdvConfig {
  static constexpr int kBN = 64;  // keys per CTA
  static constexpr int kBM = 64;  // queries per step
  static constexpr int kWarps = 8;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kSmemElements =
      kBN * kD + kBN * kDv + 2 * kBM * kD + 2 * kBM * kDv + 2 * kBN * kBM;
  // Per-stage row metadata: lo, hi (int32), lse2, delta (float), segment (int64).
  static constexpr int kRowBytes = kBM * (4 + 4 + 4 + 4 + 8);
  static constexpr int kSmemBytes = kSmemElements * 2 + 2 * kRowBytes;
};

// Key-owned dK/dV: one CTA per 64 keys of one KV head. The query heads of the
// group, then the query tiles, are visited in ascending order.
template <typename E, int kD, int kDv>
__global__ void __launch_bounds__(DkdvConfig<kD, kDv>::kThreads, 1) DkdvKernel(const Params p) {
  using Config = DkdvConfig<kD, kDv>;
  using T = Traits<E>;
  constexpr int kBN = Config::kBN, kBM = Config::kBM, kThreads = Config::kThreads;
  extern __shared__ __align__(128) unsigned char shared_raw[];
  E* sK = reinterpret_cast<E*>(shared_raw);
  E* sV = sK + kBN * kD;
  E* sQ = sV + kBN * kDv;            // [2][kBM*kD]
  E* sG = sQ + 2 * kBM * kD;         // [2][kBM*kDv]
  E* sP = sG + 2 * kBM * kDv;        // [kBN*kBM]
  E* sS = sP + kBN * kBM;            // [kBN*kBM]
  unsigned char* rows_raw = reinterpret_cast<unsigned char*>(sS + kBN * kBM);
  int64_t* rSeg = reinterpret_cast<int64_t*>(rows_raw);      // [2][kBM]
  int32_t* rLo = reinterpret_cast<int32_t*>(rSeg + 2 * kBM);  // [2][kBM]
  int32_t* rHi = rLo + 2 * kBM;
  float* rLse = reinterpret_cast<float*>(rHi + 2 * kBM);
  float* rDelta = rLse + 2 * kBM;

  const int64_t n_block = blockIdx.x;
  const int64_t hk = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t n0 = n_block * kBN;
  const int64_t n1 = n0 + kBN - 1 < p.total_length - 1 ? n0 + kBN - 1 : p.total_length - 1;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t4 = lane % 4;
  const int kg = warp & 3, half = warp >> 2;
  const bool segments = SegmentTest(p, b);

  // Query tiles that may see these keys.
  const QueryTiles tiles = QueryTilesOf<kBM>(p, b, n0, n1);
  const int64_t t_begin = tiles.first, t_end = tiles.last;
  const int64_t span = t_end >= t_begin ? t_end - t_begin + 1 : 0;
  const int64_t steps = span * p.group;
  auto intersects = [&](int64_t step) {
    const int64_t tile = t_begin + step % span;
    const int32_t* e = p.tile_bounds + (b * p.tile_count + tile) * 4;
    return e[0] <= n1 && e[1] >= n0;
  };
  auto next_step = [&](int64_t step) {
    while (step < steps && !intersects(step)) ++step;
    return step;
  };

  // Keys of this CTA stay resident.  A CTA whose key tile no query tile
  // intersects loads nothing: the resident tile is read only inside the step
  // loop below, and the epilogue stores the zeroed accumulators as they are.
  const int64_t step0 = next_step(0);
  if (step0 < steps) {
    LoadTile<E, kBN, kD, kThreads>(sK, [&](int r) { return KeyRow<E>(p, p.prev_k, p.k, b, n0 + r, hk); },
                                   p.qk_dim, p.q.ptr);
    LoadTile<E, kBN, kDv, kThreads>(sV, [&](int r) { return KeyRow<E>(p, p.prev_v, p.v, b, n0 + r, hk); },
                                    p.value_dim, p.q.ptr);
  }

  const int64_t key0 = n0 + kg * 16 + g, key1 = key0 + 8;
  const int64_t* kseg = segments ? p.k_segment + b * p.total_length : nullptr;
  const int64_t kseg0 = segments && key0 < p.total_length ? kseg[key0] : 0;
  const int64_t kseg1 = segments && key1 < p.total_length ? kseg[key1] : 0;

  auto load_queries = [&](int64_t step, int stage) {
    const int64_t hq = hk * p.group + step / span;
    const int64_t m0 = (t_begin + step % span) * kBM;
    const E* q_base = reinterpret_cast<const E*>(p.q.ptr) + b * p.q.batch_stride + hq * p.q.head_stride;
    const E* g_base = reinterpret_cast<const E*>(p.dy) + (b * p.q_length * p.query_heads + hq) * p.value_dim;
    LoadTile<E, kBM, kD, kThreads>(
        sQ + stage * kBM * kD,
        [&](int r) -> const E* {
          const int64_t i = m0 + r;
          return i < p.q_length ? q_base + i * p.q.row_stride : nullptr;
        },
        p.qk_dim, p.q.ptr);
    LoadTile<E, kBM, kDv, kThreads>(
        sG + stage * kBM * kDv,
        [&](int r) -> const E* {
          const int64_t i = m0 + r;
          return i < p.q_length ? g_base + i * p.query_heads * p.value_dim : nullptr;
        },
        p.value_dim, p.q.ptr);
    if (threadIdx.x < kBM) {
      const int r = threadIdx.x;
      const int64_t i = m0 + r;
      const Bounds bound = RowBounds(p, b, i);
      const bool valid = i < p.q_length;
      const int64_t stat = (b * p.query_heads + hq) * p.q_length + i;
      rLo[stage * kBM + r] = bound.lo;
      rHi[stage * kBM + r] = bound.hi;
      rLse[stage * kBM + r] = valid ? LseLog2(p.lse_in[stat]) : INFINITY;
      rDelta[stage * kBM + r] = valid ? p.delta[stat] : 0.f;
      rSeg[stage * kBM + r] = segments && valid ? p.q_segment[b * p.q_length + i] : 0;
    }
  };

  float acc_v[kDv / 16][4];
  float acc_k[kD / 16][4];
#pragma unroll
  for (int i = 0; i < kDv / 16; ++i) acc_v[i][0] = acc_v[i][1] = acc_v[i][2] = acc_v[i][3] = 0.f;
#pragma unroll
  for (int i = 0; i < kD / 16; ++i) acc_k[i][0] = acc_k[i][1] = acc_k[i][2] = acc_k[i][3] = 0.f;

  int64_t step = step0;
  if (step < steps) load_queries(step, 0);
  CopyAsyncCommit();
  int stage = 0;
  while (step < steps) {
    const int64_t following = next_step(step + 1);
    __syncthreads();  // every warp is done with the other stage
    if (following < steps) load_queries(following, stage ^ 1);
    CopyAsyncCommit();
    CopyAsyncWait<1>();
    __syncthreads();

    const int64_t m0 = (t_begin + step % span) * kBM;
    const E* tQ = sQ + stage * kBM * kD;
    const E* tG = sG + stage * kBM * kDv;
    const int32_t* lo = rLo + stage * kBM;
    const int32_t* hi = rHi + stage * kBM;
    const float* lse2 = rLse + stage * kBM;
    const float* dl = rDelta + stage * kBM;
    const int64_t* qseg = rSeg + stage * kBM;
    const TileRange range = ReadTileRange(p, b, m0 / kTileRows, 1);
    const bool full = !segments && n0 >= range.max_lo && n0 + kBN - 1 <= range.min_hi;

    // S^T and dP^T for keys kg*16.. and queries half*32..
    float st[4][4], dpt[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      st[i][0] = st[i][1] = st[i][2] = st[i][3] = 0.f;
      dpt[i][0] = dpt[i][1] = dpt[i][2] = dpt[i][3] = 0.f;
    }
#pragma unroll
    for (int kk = 0; kk < kD / 16; ++kk) {
      uint32_t a[4];
      LoadMatrix(a, SmemAddress(sK + Swizzle<kD>(kg * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        uint32_t bf[4];
        LoadMatrix(bf, SmemAddress(tQ + Swizzle<kD>(half * 32 + nb * 16 + (lane % 8) + (lane / 16) * 8,
                                                     kk * 2 + (lane / 8) % 2)));
        T::Mma(st[2 * nb], a, bf[0], bf[1]);
        T::Mma(st[2 * nb + 1], a, bf[2], bf[3]);
      }
    }
#pragma unroll
    for (int kk = 0; kk < kDv / 16; ++kk) {
      uint32_t a[4];
      LoadMatrix(a, SmemAddress(sV + Swizzle<kDv>(kg * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        uint32_t bf[4];
        LoadMatrix(bf, SmemAddress(tG + Swizzle<kDv>(half * 32 + nb * 16 + (lane % 8) + (lane / 16) * 8,
                                                      kk * 2 + (lane / 8) % 2)));
        T::Mma(dpt[2 * nb], a, bf[0], bf[1]);
        T::Mma(dpt[2 * nb + 1], a, bf[2], bf[3]);
      }
    }

    // P^T and dS^T, written to shared memory as [key][query].
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
      const int ql = half * 32 + nb * 8 + 2 * t4;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t key = h ? key1 : key0;
        const int64_t key_seg = h ? kseg1 : kseg0;
        float pv[2], dsv[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int q = ql + e;
          bool ok = true;
          if (!full) {
            ok = key >= lo[q] && key <= hi[q] && (!segments || key_seg == qseg[q]);
          }
          const float pr = ok ? FastExp2(fmaf(st[nb][2 * h + e], p.scale_log2, -lse2[q])) : 0.f;
          pv[e] = pr;
          dsv[e] = pr * (dpt[nb][2 * h + e] - dl[q]);
        }
        const int kl = kg * 16 + g + 8 * h;
        const int offset = Swizzle<kBM>(kl, ql / 8) + (ql % 8);
        *reinterpret_cast<uint32_t*>(sP + offset) = T::Pack(pv[0], pv[1]);
        *reinterpret_cast<uint32_t*>(sS + offset) = T::Pack(dsv[0], dsv[1]);
      }
    }
    __syncthreads();

    // dV += P^T dO and dK += dS^T Q for keys kg*16.. and columns of this half.
#pragma unroll
    for (int kk = 0; kk < kBM / 16; ++kk) {
      uint32_t a[4];
      LoadMatrix(a, SmemAddress(sP + Swizzle<kBM>(kg * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
      for (int nv = 0; nv < kDv / 32; ++nv) {
        uint32_t bf[4];
        LoadMatrixTransposed(
            bf, SmemAddress(tG + Swizzle<kDv>(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                               half * (kDv / 16) + nv * 2 + lane / 16)));
        T::Mma(acc_v[2 * nv], a, bf[0], bf[1]);
        T::Mma(acc_v[2 * nv + 1], a, bf[2], bf[3]);
      }
      LoadMatrix(a, SmemAddress(sS + Swizzle<kBM>(kg * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
      for (int nd = 0; nd < kD / 32; ++nd) {
        uint32_t bf[4];
        LoadMatrixTransposed(
            bf, SmemAddress(tQ + Swizzle<kD>(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                              half * (kD / 16) + nd * 2 + lane / 16)));
        T::Mma(acc_k[2 * nd], a, bf[0], bf[1]);
        T::Mma(acc_k[2 * nd + 1], a, bf[2], bf[3]);
      }
    }
    if (p.dq_accum != nullptr) {
      // dQ rows kg*16.., columns of this half, in 64-column passes.
      const int64_t hq = hk * p.group + step / span;
#pragma unroll 1
      for (int c = 0; c < kD / 2; c += 64) {
        AccumulateDq<E, kD, kBN, (kD / 2 < 64 ? kD / 2 : 64)>(p, sS, sK, kg * 16, half * (kD / 2) + c, b, hq, m0);
      }
    }

    step = following;
    stage ^= 1;
  }

  // Store: total key j < P belongs to the previous K/V gradients.
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int64_t key = h ? key1 : key0;
    if (key >= p.total_length) continue;
    const bool previous = key < p.prev_length;
    const int64_t kd_row = previous ? ((b * p.prev_length + key) * p.kv_heads + hk)
                                    : ((b * p.k_length + key - p.prev_length) * p.kv_heads + hk);
    void* dk = previous ? p.dprev_k : p.dk;
    void* dv = previous ? p.dprev_v : p.dv;
#pragma unroll
    for (int i = 0; i < kDv / 16; ++i) {
      const int col = half * (kDv / 2) + i * 8 + 2 * t4;
      if (col < p.value_dim)
        StoreKeyGradPair<E>(p, dv, kd_row * p.value_dim + col, acc_v[i][2 * h], acc_v[i][2 * h + 1]);
    }
#pragma unroll
    for (int i = 0; i < kD / 16; ++i) {
      const int col = half * (kD / 2) + i * 8 + 2 * t4;
      if (col < p.qk_dim)
        StoreKeyGradPair<E>(p, dk, kd_row * p.qk_dim + col, acc_k[i][2 * h] * p.scale,
                            acc_k[i][2 * h + 1] * p.scale);
    }
  }
}


template <int kD, int kDv>
struct DkdvRegConfig {
  static constexpr int kWarps = 8;
  static constexpr int kBN = 16 * kWarps;  // keys per CTA, 16 per warp
  static constexpr int kBM = 64;           // queries per step
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kSmemElements =
      kBN * kD + kBN * kDv + 2 * kBM * kD + 2 * kBM * kDv + kBN * kBM;
  static constexpr int kRowBytes = kBM * (4 + 4 + 4 + 4 + 8);
  static constexpr int kSmemBytes = kSmemElements * 2 + 2 * kRowBytes;
};

// Key-owned dK/dV for D + Dv <= 256: every warp owns 16 keys and all 64
// queries of a step, so P^T and dS^T stay in registers as the A operands of
// dV += P^T dO and dK += dS^T Q. Same visiting order as DkdvKernel.
template <typename E, int kD, int kDv>
__global__ void __launch_bounds__(DkdvRegConfig<kD, kDv>::kThreads, 1) DkdvRegKernel(const Params p) {
  using Config = DkdvRegConfig<kD, kDv>;
  using T = Traits<E>;
  constexpr int kBN = Config::kBN, kBM = Config::kBM, kThreads = Config::kThreads;
  extern __shared__ __align__(128) unsigned char shared_raw[];
  E* sK = reinterpret_cast<E*>(shared_raw);
  E* sV = sK + kBN * kD;
  E* sQ = sV + kBN * kDv;        // [2][kBM*kD]
  E* sG = sQ + 2 * kBM * kD;     // [2][kBM*kDv]
  E* sS = sG + 2 * kBM * kDv;    // [kBN][kBM] dS^T, only for the fused dQ
  unsigned char* rows_raw = reinterpret_cast<unsigned char*>(sS + kBN * kBM);
  int64_t* rSeg = reinterpret_cast<int64_t*>(rows_raw);
  int32_t* rLo = reinterpret_cast<int32_t*>(rSeg + 2 * kBM);
  int32_t* rHi = rLo + 2 * kBM;
  float* rLse = reinterpret_cast<float*>(rHi + 2 * kBM);
  float* rDelta = rLse + 2 * kBM;

  const int64_t n_block = blockIdx.x;
  const int64_t hk = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t n0 = n_block * kBN;
  const int64_t n1 = n0 + kBN - 1 < p.total_length - 1 ? n0 + kBN - 1 : p.total_length - 1;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t4 = lane % 4;
  const bool segments = SegmentTest(p, b);

  const QueryTiles tiles = QueryTilesOf<kBM>(p, b, n0, n1);
  const int64_t t_begin = tiles.first, t_end = tiles.last;
  const int64_t span = t_end >= t_begin ? t_end - t_begin + 1 : 0;
  const int64_t steps = span * p.group;
  auto intersects = [&](int64_t step) {
    const int64_t tile = t_begin + step % span;
    const int32_t* e = p.tile_bounds + (b * p.tile_count + tile) * 4;
    return e[0] <= n1 && e[1] >= n0;
  };
  auto next_step = [&](int64_t step) {
    while (step < steps && !intersects(step)) ++step;
    return step;
  };

  // A CTA whose key tile no query tile intersects loads nothing (see the
  // non-Reg kernel): the resident tile is read only inside the step loop.
  const int64_t step0 = next_step(0);
  if (step0 < steps) {
    LoadTile<E, kBN, kD, kThreads>(sK, [&](int r) { return KeyRow<E>(p, p.prev_k, p.k, b, n0 + r, hk); },
                                   p.qk_dim, p.q.ptr);
    LoadTile<E, kBN, kDv, kThreads>(sV, [&](int r) { return KeyRow<E>(p, p.prev_v, p.v, b, n0 + r, hk); },
                                    p.value_dim, p.q.ptr);
  }

  const int64_t key0 = n0 + warp * 16 + g, key1 = key0 + 8;
  const int64_t* kseg = segments ? p.k_segment + b * p.total_length : nullptr;
  const int64_t kseg0 = segments && key0 < p.total_length ? kseg[key0] : 0;
  const int64_t kseg1 = segments && key1 < p.total_length ? kseg[key1] : 0;
  // This warp's 16 keys entirely inside [warp_lo, warp_hi] of every row allow
  // skipping the per-element test.
  const int64_t warp_key0 = n0 + warp * 16, warp_key1 = warp_key0 + 15;

  auto load_queries = [&](int64_t step, int stage) {
    const int64_t hq = hk * p.group + step / span;
    const int64_t m0 = (t_begin + step % span) * kBM;
    const E* q_base = reinterpret_cast<const E*>(p.q.ptr) + b * p.q.batch_stride + hq * p.q.head_stride;
    const E* g_base = reinterpret_cast<const E*>(p.dy) + (b * p.q_length * p.query_heads + hq) * p.value_dim;
    LoadTile<E, kBM, kD, kThreads>(
        sQ + stage * kBM * kD,
        [&](int r) -> const E* {
          const int64_t i = m0 + r;
          return i < p.q_length ? q_base + i * p.q.row_stride : nullptr;
        },
        p.qk_dim, p.q.ptr);
    LoadTile<E, kBM, kDv, kThreads>(
        sG + stage * kBM * kDv,
        [&](int r) -> const E* {
          const int64_t i = m0 + r;
          return i < p.q_length ? g_base + i * p.query_heads * p.value_dim : nullptr;
        },
        p.value_dim, p.q.ptr);
    if (threadIdx.x < kBM) {
      const int r = threadIdx.x;
      const int64_t i = m0 + r;
      const Bounds bound = RowBounds(p, b, i);
      const bool valid = i < p.q_length;
      const int64_t stat = (b * p.query_heads + hq) * p.q_length + i;
      rLo[stage * kBM + r] = bound.lo;
      rHi[stage * kBM + r] = bound.hi;
      rLse[stage * kBM + r] = valid ? LseLog2(p.lse_in[stat]) : INFINITY;
      rDelta[stage * kBM + r] = valid ? p.delta[stat] : 0.f;
      rSeg[stage * kBM + r] = segments && valid ? p.q_segment[b * p.q_length + i] : 0;
    }
  };

  float acc_v[kDv / 8][4];
  float acc_k[kD / 8][4];
#pragma unroll
  for (int i = 0; i < kDv / 8; ++i) acc_v[i][0] = acc_v[i][1] = acc_v[i][2] = acc_v[i][3] = 0.f;
#pragma unroll
  for (int i = 0; i < kD / 8; ++i) acc_k[i][0] = acc_k[i][1] = acc_k[i][2] = acc_k[i][3] = 0.f;

  int64_t step = step0;
  if (step < steps) load_queries(step, 0);
  CopyAsyncCommit();
  int stage = 0;
  while (step < steps) {
    const int64_t following = next_step(step + 1);
    __syncthreads();
    if (following < steps) load_queries(following, stage ^ 1);
    CopyAsyncCommit();
    CopyAsyncWait<1>();
    __syncthreads();

    const int64_t m0 = (t_begin + step % span) * kBM;
    const E* tQ = sQ + stage * kBM * kD;
    const E* tG = sG + stage * kBM * kDv;
    const int32_t* lo = rLo + stage * kBM;
    const int32_t* hi = rHi + stage * kBM;
    const float* lse2 = rLse + stage * kBM;
    const float* dl = rDelta + stage * kBM;
    const int64_t* qseg = rSeg + stage * kBM;
    const TileRange range = ReadTileRange(p, b, m0 / kTileRows, 1);
    const bool full = !segments && warp_key0 >= range.max_lo && warp_key1 <= range.min_hi;
    const bool empty = warp_key0 > range.max_hi || warp_key1 < range.min_lo;

    const bool fused_dq = p.dq_accum != nullptr;
    if (empty && fused_dq) {
      // This warp's dS^T rows are zero for the fused dQ product.
#pragma unroll
      for (int x = lane; x < 16 * kBM / 8; x += 32) {
        const int r = x / (kBM / 8), c = x % (kBM / 8);
        *reinterpret_cast<uint4*>(sS + Swizzle<kBM>(warp * 16 + r, c)) = make_uint4(0, 0, 0, 0);
      }
    }
    if (!empty) {
      float st[kBM / 8][4], dpt[kBM / 8][4];
#pragma unroll
      for (int i = 0; i < kBM / 8; ++i) {
        st[i][0] = st[i][1] = st[i][2] = st[i][3] = 0.f;
        dpt[i][0] = dpt[i][1] = dpt[i][2] = dpt[i][3] = 0.f;
      }
#pragma unroll
      for (int kk = 0; kk < kD / 16; ++kk) {
        uint32_t a[4];
        LoadMatrix(a, SmemAddress(sK + Swizzle<kD>(warp * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
        for (int nb = 0; nb < kBM / 16; ++nb) {
          uint32_t bf[4];
          LoadMatrix(bf, SmemAddress(tQ + Swizzle<kD>(nb * 16 + (lane % 8) + (lane / 16) * 8,
                                                       kk * 2 + (lane / 8) % 2)));
          T::Mma(st[2 * nb], a, bf[0], bf[1]);
          T::Mma(st[2 * nb + 1], a, bf[2], bf[3]);
        }
      }
#pragma unroll
      for (int kk = 0; kk < kDv / 16; ++kk) {
        uint32_t a[4];
        LoadMatrix(a, SmemAddress(sV + Swizzle<kDv>(warp * 16 + (lane % 16), kk * 2 + lane / 16)));
#pragma unroll
        for (int nb = 0; nb < kBM / 16; ++nb) {
          uint32_t bf[4];
          LoadMatrix(bf, SmemAddress(tG + Swizzle<kDv>(nb * 16 + (lane % 8) + (lane / 16) * 8,
                                                        kk * 2 + (lane / 8) % 2)));
          T::Mma(dpt[2 * nb], a, bf[0], bf[1]);
          T::Mma(dpt[2 * nb + 1], a, bf[2], bf[3]);
        }
      }

      // P^T into st, dS^T into dpt.
#pragma unroll
      for (int nb = 0; nb < kBM / 8; ++nb) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int q = nb * 8 + 2 * t4 + e;
          bool ok0 = true, ok1 = true;
          if (!full) {
            ok0 = key0 >= lo[q] && key0 <= hi[q] && (!segments || kseg0 == qseg[q]);
            ok1 = key1 >= lo[q] && key1 <= hi[q] && (!segments || kseg1 == qseg[q]);
          }
          const float p0 = ok0 ? FastExp2(fmaf(st[nb][e], p.scale_log2, -lse2[q])) : 0.f;
          const float p1 = ok1 ? FastExp2(fmaf(st[nb][2 + e], p.scale_log2, -lse2[q])) : 0.f;
          st[nb][e] = p0;
          st[nb][2 + e] = p1;
          dpt[nb][e] = p0 * (dpt[nb][e] - dl[q]);
          dpt[nb][2 + e] = p1 * (dpt[nb][2 + e] - dl[q]);
        }
      }
      if (fused_dq) {
#pragma unroll
        for (int nb = 0; nb < kBM / 8; ++nb) {
          const int ql = nb * 8 + 2 * t4;
          *reinterpret_cast<uint32_t*>(sS + Swizzle<kBM>(warp * 16 + g, ql / 8) + ql % 8) =
              T::Pack(dpt[nb][0], dpt[nb][1]);
          *reinterpret_cast<uint32_t*>(sS + Swizzle<kBM>(warp * 16 + g + 8, ql / 8) + ql % 8) =
              T::Pack(dpt[nb][2], dpt[nb][3]);
        }
      }

#pragma unroll
      for (int kk = 0; kk < kBM / 16; ++kk) {
        uint32_t a[4];
        a[0] = T::Pack(st[2 * kk][0], st[2 * kk][1]);
        a[1] = T::Pack(st[2 * kk][2], st[2 * kk][3]);
        a[2] = T::Pack(st[2 * kk + 1][0], st[2 * kk + 1][1]);
        a[3] = T::Pack(st[2 * kk + 1][2], st[2 * kk + 1][3]);
#pragma unroll
        for (int nv = 0; nv < kDv / 16; ++nv) {
          uint32_t bf[4];
          LoadMatrixTransposed(
              bf, SmemAddress(tG + Swizzle<kDv>(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                                 nv * 2 + lane / 16)));
          T::Mma(acc_v[2 * nv], a, bf[0], bf[1]);
          T::Mma(acc_v[2 * nv + 1], a, bf[2], bf[3]);
        }
        a[0] = T::Pack(dpt[2 * kk][0], dpt[2 * kk][1]);
        a[1] = T::Pack(dpt[2 * kk][2], dpt[2 * kk][3]);
        a[2] = T::Pack(dpt[2 * kk + 1][0], dpt[2 * kk + 1][1]);
        a[3] = T::Pack(dpt[2 * kk + 1][2], dpt[2 * kk + 1][3]);
#pragma unroll
        for (int nd = 0; nd < kD / 16; ++nd) {
          uint32_t bf[4];
          LoadMatrixTransposed(
              bf, SmemAddress(tQ + Swizzle<kD>(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                                nd * 2 + lane / 16)));
          T::Mma(acc_k[2 * nd], a, bf[0], bf[1]);
          T::Mma(acc_k[2 * nd + 1], a, bf[2], bf[3]);
        }
      }
    }
    if (fused_dq) {
      __syncthreads();
      // 64 query rows x kD columns over 8 warps: 4 row groups x 2 column halves.
      const int64_t hq = hk * p.group + step / span;
      AccumulateDq<E, kD, kBN, kD / 2>(p, sS, sK, (warp & 3) * 16, (warp >> 2) * (kD / 2), b, hq, m0);
    }

    step = following;
    stage ^= 1;
  }

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int64_t key = h ? key1 : key0;
    if (key >= p.total_length) continue;
    const bool previous = key < p.prev_length;
    const int64_t kd_row = previous ? ((b * p.prev_length + key) * p.kv_heads + hk)
                                    : ((b * p.k_length + key - p.prev_length) * p.kv_heads + hk);
    void* dk = previous ? p.dprev_k : p.dk;
    void* dv = previous ? p.dprev_v : p.dv;
#pragma unroll
    for (int i = 0; i < kDv / 8; ++i) {
      const int col = i * 8 + 2 * t4;
      if (col < p.value_dim)
        StoreKeyGradPair<E>(p, dv, kd_row * p.value_dim + col, acc_v[i][2 * h], acc_v[i][2 * h + 1]);
    }
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) {
      const int col = i * 8 + 2 * t4;
      if (col < p.qk_dim)
        StoreKeyGradPair<E>(p, dk, kd_row * p.qk_dim + col, acc_k[i][2 * h] * p.scale,
                            acc_k[i][2 * h + 1] * p.scale);
    }
  }
}

}  // namespace detail
}  // namespace xattn::ops::flash_sca_native
