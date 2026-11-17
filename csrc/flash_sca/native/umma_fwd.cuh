// Native Blackwell (sm_103a) FlashSCA forward: one CTA per (batch, query head,
// 128-query-row tile), tcgen05/UMMA contractions with the accumulator in TMEM,
// TMA operand loads, online softmax in registers.
//
// Precision coverage, stated exactly - and by the public baseline's own
// semantics, not by feature names. The forward's observable form at levels 1-4
// is the same one level 1 is defined as: FP32 running state and the
// subtract-before-scale softmax ordering. Levels 2-4 add no separate forward
// algebra in the shipped baseline (their K/V centering, stable dS, row-sum
// repair and dual-BF16 P/mass repair are backward-side machinery), so this leaf
// serves the forward for every level with the level 0/1 form and no refusal
// (reconciled against the actual public
// baseline, never inferred from a feature list). The backward's per-level algebra
// is the backward leaves' business, not this one's.
//
// What this leaf does NOT claim: a bitwise match against any other family. See
// the LSE convention below.
//
// Scope of the visibility mask: D and Dv buckets of 64/128/256 (a 32 bucket has
// no 64-column swizzle slab and stays on the mma.sync path), FP16/BF16,
// previous K/V (whole key tiles, or a packed [previous;current] map for any
// previous length), unequal current lengths, BOS/segment metadata, per-sequence
// resets, empty rows and arbitrary chunk sizes. An empty row writes y = 0,
// an FP32 state of 0 and LSE = -INFINITY - all three - and contributes nothing
// to any gradient; -INFINITY is this family's no-visible-key LSE convention (not
// the generic seed's +INFINITY), so this leaf is gated bitwise only against a
// native envelope baseline, never across families. Visibility is read from the shared tables
// only (`row_lo`/`row_hi`/`q_segment`/`k_segment`/`segment_exact`) through
// `native/visibility.cuh`, so this kernel has no second opinion about the mask.
//
// Per-tile work, all on one warpgroup (warps 0-3, thread t owns query row t):
//   S = Q K^T      tcgen05.mma SS, SMEM operands, FP32 accumulator in TMEM
//   P = softmax(S) tcgen05.ld -> registers -> pack -> tcgen05.st
//   O += P V       tcgen05.mma TS, A from TMEM, B (V) from SMEM, MN-major
// y/y_state/lse are ordinary predicated stores.
//
// TMEM columns (one allocation per CTA, one CTA per SM), S kBN + O kDv + P kBN/2:
//   224 at kD=kDv=128 (allocates 256), 352 at 256 (allocates 512).
// SMEM: Q resident, K and V in `kStages` stages; D256 = 64K + 2x32K + 2x32K.
//
// CTA-uniform facts this kernel relies on: the PV mma is issued for every key
// tile with a CTA-uniform accumulate flag (rows with no visible key in a tile
// store a zero P row, so their contribution is exactly zero); the S mma, the
// K/V pipeline and every barrier are CTA-wide; only the softmax state (m2,
// lsum, alpha) is per-row, held in registers of the row's thread.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/params.h"
#include "flash_sca/native/umma_fwd_pipe_host.h"
#include "flash_sca/native/umma_params.h"
#include "flash_sca/native/visibility.cuh"
// Primitive set, bound EXPLICITLY: choosing between the shared header and this
// tree's pinned copy with __has_include
// would be a silent switch - the same TU could bind to either vocabulary depending on
// an include path, and the two sets are not interchangeable (no MmaTs, and no 4-D map
// load, in the shared set; different I32/I64 widths elsewhere). This file binds the
// pinned copy and the two headers carry mutual #error guards, so a build that sees both
// fails loudly instead of taking whichever came first.
//
// The port itself is small: the shared header's 4-D load takes this
// file's verbatim body, and the three spellings in play - this leaf's 4-D
// no-`.tile` form, the dQ leaf's 4-D `.tile` form and the shared 5-D
// `.tile` form - assemble to the same UTMALDG.4D instruction. So the port
// owes no map encoder and no host parity proof: it is an include swap plus
// deleting this pinned copy in the same commit, with the reciprocal
// `#error` half landing in umma_common.cuh.
#include "flash_sca/native/umma_common_local.h"

namespace xattn::ops::flash_sca_native {
namespace umma {

inline constexpr int kBM = 128;  // query rows per CTA; thread t owns row t
inline constexpr int kBN = 64;   // keys per tile; one TMA box width
inline constexpr int kThreads = 128;
inline constexpr int kWarps = kThreads / 32;

// kD/kDv are padded buckets of 64 (the 128 B swizzle MN atom is 64 elements
// wide, so a smaller logical width is zero-padded by the TMA, never re-swizzled).
template <int kD, int kDv>
struct ForwardPlan {
  static_assert(kD % 64 == 0 && kDv % 64 == 0, "buckets are multiples of 64");
  static constexpr int kSlabsQ = kD / 64;
  static constexpr int kSlabsK = kD / 64;
  static constexpr int kSlabsV = kDv / 64;
  static constexpr int kQSlab = kBM * 128;  // bytes of one 64-column slab
  // TMA box: 64 rows x 64 columns x 2 B = 8 KiB at every D; an operand tile is a
  // COUNT of boxes (D=256 Q tile = 8, D=128 Q tile = 4). The 64-row box is the
  // only shape executed end to end and the one the 4D map was proven with,
  // so it stays: a 128-row box halves the issue count per 128-key
  // stage at D=128 but is a separate arm with its own criterion.
  static constexpr int kQTmaRows = 64;
  static constexpr int kKvSlab = kBN * 128;
  static constexpr int kStages = (kD == 256 || kDv == 256) ? 2 : 3;
  static constexpr int kQBytes = kSlabsQ * kQSlab;
  static constexpr int kKBytes = kStages * kSlabsK * kKvSlab;
  static constexpr int kVBytes = kStages * kSlabsV * kKvSlab;
  static constexpr int kBarOff = kQBytes + kKBytes + kVBytes;  // 128-aligned
  // bars[0] = Q, bars[1 .. kStages] = K/V per stage, then S-mma, PV-mma.
  static constexpr int kBarCount = 2 + kStages + 2;
  static constexpr int kSlotOff = kBarOff + kBarCount * 8;
  static constexpr int kRedOff = kSlotOff + 16;  // kWarps x 4 int32
  static constexpr int kSegOff = kRedOff + kWarps * 16;
  static constexpr int kSmemBytes = kSegOff + kBN * 8;
  // TMEM columns.
  static constexpr int kSCol = 0;
  static constexpr int kOCol = kBN;
  static constexpr int kPCol = kOCol + kDv;
  static constexpr int kUsedCols = kPCol + kBN / 2;
  static constexpr int kAllocCols = kUsedCols <= 256 ? 256 : 512;
  static_assert(kUsedCols <= 512, "TMEM column budget");
};

// Host-visible so the budget audits can harvest the column count. Must agree
// with ForwardPlan<kD,kDv>::kAllocCols for every bucket: the used columns are
// S (kBN = 64) + O (kDv) + P (kBN/2 = 32) and the allocation is the next whole
// power-of-two window (256 or 512). kD does not enter it.
constexpr int ForwardTmemColumns(int d_bucket, int dv_bucket) {
  (void)d_bucket;
  const int used = 64 + dv_bucket + 32;
  return used <= 256 ? 256 : 512;
}

// ------------------------------------------------------------------ helpers
template <class Elem>
struct ElemTraits;
template <>
struct ElemTraits<__nv_bfloat16> {
  static constexpr int kFormat = 1;
  __device__ static __nv_bfloat16 From(float x) { return __float2bfloat16_rn(x); }
  __device__ static uint32_t Pack2(float lo, float hi) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
  }
};
template <>
struct ElemTraits<__half> {
  static constexpr int kFormat = 0;
  __device__ static __half From(float x) { return __float2half_rn(x); }
  __device__ static uint32_t Pack2(float lo, float hi) {
    return static_cast<uint32_t>(__half_as_ushort(__float2half_rn(lo))) |
           (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(hi))) << 16);
  }
};

__device__ __forceinline__ int WarpMin(int v) {
  for (int o = 16; o; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ int WarpMax(int v) {
  for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// `cols` must be a multiple of 32; `addr` includes the warp's lane field.
__device__ __forceinline__ void TmemLoadCols(uint32_t addr, float* v, int cols) {
  for (int c = 0; c < cols; c += 32) TmemLoad32(addr + c, *reinterpret_cast<float(*)[32]>(v + c));
  TmemWaitLd();
}

__device__ __forceinline__ uint32_t Pack2Fmt(uint32_t lo, uint32_t hi) { return lo | (hi << 16); }

// Keys of this tile lie inside every live row's visible interval and no
// per-element segment test is needed.
//
// Proof obligation: the `!seg` relaxation is sound only because of
// the host predicate behind `segment_exact[b]` - on the sorted/BOS route
// visibility() returns the row's key-run intersection as row_lo/row_hi and
// leaves q_segment null, so the bounds already exclude the previous segment;
// on the unsorted-id route the host sets segment_exact[b] only when that batch
// row's key ids are nondecreasing, i.e. exactly when the run intersection is
// exact. The traced prefill pattern (previous K/V present, bos_mask marks the
// first current token, so the previous chunk is another segment and the hull of
// a new-segment row contains previous positions) is therefore covered: either
// segment_exact is 0 and this returns false, or the bounds exclude them. The
// exactness itself is the host predicate's claim, not this leaf's; if it is ever
// withdrawn, gate this on `!segments` (the form the generic kernels use) and mask always.
__device__ __forceinline__ bool TileIsFull(bool seg, int64_t kb, int max_lo, int min_hi) {
  if (seg) return false;  // exact-segment metadata is required to trust an interval
  return kb >= max_lo && kb + kBN - 1 <= min_hi;
}

// Map and row offset holding the global key tile that starts at `j0`.
// PRECONDITION: the split `prev_k`/`prev_v` maps are used
// only when the previous K/V length is a whole number of key tiles
// (`packed_kv` or `prev_length % kBN == 0`), because a single TMA box cannot
// straddle the previous/current boundary. The contract's prev length is exactly
// chunk_size (1..257 in sca-contract), so most contract cases are NOT multiples
// of 64 and must come in as the packed [prev;current] map, which the host
// packing rule (pack iff prev_length % kBN != 0) guarantees. Reading
// the split maps with a partial key tile is a silent wrong answer, so
// LaunchUmmaForward refuses that combination instead of launching, and dispatch
// must route prev_length % kBN != 0 to the packed map.
__device__ __forceinline__ void ResolveKeyMap(const UmmaLaunchParams& lp, int64_t j0,
                                              const CUtensorMap** kmap, const CUtensorMap** vmap,
                                              int* row0) {
  const Params& p = lp.params;
  if (lp.maps.packed_kv || p.prev_length == 0 || j0 >= p.prev_length) {
    *kmap = &lp.maps.k;
    *vmap = &lp.maps.v;
    *row0 = static_cast<int>((lp.maps.packed_kv || p.prev_length == 0) ? j0 : j0 - p.prev_length);
  } else {
    *kmap = &lp.maps.prev_k;
    *vmap = &lp.maps.prev_v;
    *row0 = static_cast<int>(j0);
  }
}

// tcgen05 cross-thread ordering. tcgen05.wait::st /
// ::ld only order the ISSUING thread's own TMEM accesses; a TMEM buffer handed
// from one set of threads to another (P written by 128 threads and read by the
// PV mma, O rescaled then read by the next PV mma, S read by 128 threads then
// overwritten by the next S mma) needs the fence pair around the __syncthreads
// that closes the handoff. These are CUTLASS's fence_view_async_tmem_store/load.
__device__ __forceinline__ void TmemFenceBeforeSync() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
}
__device__ __forceinline__ void TmemFenceAfterSync() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}

// ------------------------------------------------------------------ kernel
template <class Elem, int kD, int kDv>
__global__ void __launch_bounds__(kThreads) ForwardKernel(
    const __grid_constant__ UmmaLaunchParams lp) {
  using Plan = ForwardPlan<kD, kDv>;
  const Params& p = lp.params;
  extern __shared__ __align__(128) unsigned char smem[];

  unsigned char* const sq = smem;
  unsigned char* const sk = smem + Plan::kQBytes;
  unsigned char* const sv = smem + Plan::kQBytes + Plan::kKBytes;
  uint32_t const bar0 = SmemAddr(smem + Plan::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Plan::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Plan::kRedOff);
  int64_t* const kseg = reinterpret_cast<int64_t*>(smem + Plan::kSegOff);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hq = blockIdx.y;
  const int qt = blockIdx.x;
  const int64_t kvh = p.group > 0 ? hq / p.group : hq;
  const int64_t i = static_cast<int64_t>(qt) * kBM + tid;
  const bool row_in_range = i < p.q_length;

  const uint32_t bar_q = bar0;
  const uint32_t bar_s = bar0 + (Plan::kStages + 1) * 8;
  const uint32_t bar_pv = bar0 + (Plan::kStages + 2) * 8;

  if (tid == 0) {
    MbarInit(bar_q, 1);
    for (int s = 0; s < Plan::kStages; ++s) MbarInit(bar0 + (1 + s) * 8, 1);
    MbarInit(bar_s, 1);
    MbarInit(bar_pv, 1);
  }
  if (warp == 0) {
    TmemAlloc(Plan::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) MbarInitFence();

  // Per-row visibility, from the shared tables only.
  const detail::Bounds bd = detail::RowBounds(p, b, i);
  const bool live = row_in_range && bd.lo <= bd.hi;
  {
    // min_lo/max_hi drive the KEY-TILE LOOP (live rows only: a dead row must not
    // widen it). max_lo/min_hi drive the FULL test and are reduced over ALL rows:
    // a dead row contributes INT_MAX/INT_MIN, so a CTA holding any dead row never
    // takes the unmasked path. That is the shipped tile_bounds rule, and it is the
    // one an independent CPU proof on these headers showed never marks a masked
    // tile FULL (988 CTA sets x 494 cases, 0 false-full); reducing over live
    // rows only is refuted - 32 empty rows x 64 keys of unmasked scores in the
    // hostile/dead-in-full-live-rule case. The per-row `live` guard on the P pack
    // below is defence in depth, not the argument.
    const int w_lo = WarpMin(live ? bd.lo : INT_MAX);
    const int w_hi = WarpMax(live ? bd.hi : -1);
    const int w_mlo = WarpMax(live ? bd.lo : INT_MAX);
    const int w_mhi = WarpMin(live ? bd.hi : INT_MIN);
    if (lane == 0) {
      red[warp * 4 + 0] = w_lo;
      red[warp * 4 + 1] = w_hi;
      red[warp * 4 + 2] = w_mlo;
      red[warp * 4 + 3] = w_mhi;
    }
  }
  __syncthreads();

  int min_lo = red[0], max_hi = red[1], max_lo = red[2], min_hi = red[3];
  for (int w = 1; w < kWarps; ++w) {
    min_lo = min(min_lo, red[w * 4 + 0]);
    max_hi = max(max_hi, red[w * 4 + 1]);
    max_lo = max(max_lo, red[w * 4 + 2]);
    min_hi = min(min_hi, red[w * 4 + 3]);
  }
  const uint32_t tbase = *tmem_slot & 0x1FFu;  // 1-CTA allocation: lane 0, column base
  const uint32_t lane_base = static_cast<uint32_t>(32 * warp) << 16;

  auto write_dead_row = [&]() {
    if (!row_in_range) return;
    Elem* yrow =
        reinterpret_cast<Elem*>(p.y) + ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
    for (int64_t c = 0; c < p.value_dim; ++c) yrow[c] = ElemTraits<Elem>::From(0.f);
    if (p.y_state != nullptr) {
      float* srow = p.y_state + ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
      for (int64_t c = 0; c < p.value_dim; ++c) srow[c] = 0.f;
    }
    if (p.lse != nullptr) p.lse[(b * p.query_heads + hq) * p.q_length + i] = -INFINITY;
  };
  auto dealloc_and_exit = [&]() {
    TmemFenceBeforeSync();
    __syncthreads();
    TmemFenceAfterSync();
    if (warp == 0) TmemDealloc(tbase, Plan::kAllocCols);
  };

  if (min_lo > max_hi) {  // no live row in this CTA
    write_dead_row();
    dealloc_and_exit();
    return;
  }

  // The base-alignment condition, made a check instead of an assumption. Every slab and
  // stage start in this Plan is an exact multiple of 8192 or 16384 B from `smem`
  // (kKvSlab = 8192, kQSlab = 16384, stage strides of whole slabs), and the
  // descriptors encode base_offset = 0, which is only correct for a tile whose
  // base sits on a 1024 B boundary - the swizzle atom period. The driver picks
  // the dynamic shared memory base, so assert its phase once per CTA: if it is
  // ever not 1024 B aligned, the right answer is a loud trap and not a swizzle
  // read of the wrong operand lanes. Never fires on a conforming launch.
  if (tid == 0 && (SmemAddr(smem) & 1023u) != 0u) __trap();

  // Q tile: always the full box set; rows past Nq and columns past D are the
  // TMA's zero fill, and those rows are excluded from every store.
  if (tid == 0) {
    MbarArriveExpectTx(bar_q, Plan::kQBytes);
    for (int s = 0; s < Plan::kSlabsQ; ++s) {
      for (int r = 0; r < kBM; r += Plan::kQTmaRows) {
        TmaLoad4D(sq + s * Plan::kQSlab + r * 128, &lp.maps.q, bar_q, s * 64, static_cast<int>(hq),
                  qt * kBM + r, static_cast<int>(b));
      }
    }
  }

  const int t0 = min_lo / kBN;
  const int t1 = max_hi / kBN;
  const int steps = t1 - t0 + 1;
  const int64_t nt_total = p.total_length;

  auto issue_kv = [&](int t) {
    if (tid != 0) return;
    const int stage = (t - t0) % Plan::kStages;
    const int64_t j0 = static_cast<int64_t>(t) * kBN;
    const CUtensorMap* kmap;
    const CUtensorMap* vmap;
    int row0;
    ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);
    MbarArriveExpectTx(bar0 + (1 + stage) * 8,
                       Plan::kSlabsK * Plan::kKvSlab + Plan::kSlabsV * Plan::kKvSlab);
    unsigned char* kdst = sk + stage * Plan::kSlabsK * Plan::kKvSlab;
    unsigned char* vdst = sv + stage * Plan::kSlabsV * Plan::kKvSlab;
    for (int s = 0; s < Plan::kSlabsK; ++s) {
      for (int r = 0; r < kBN; r += kTmaRows) {
        TmaLoad4D(kdst + s * Plan::kKvSlab + r * 128, kmap, bar0 + (1 + stage) * 8, s * 64,
                  static_cast<int>(kvh), row0 + r, static_cast<int>(b));
      }
    }
    for (int s = 0; s < Plan::kSlabsV; ++s) {
      for (int r = 0; r < kBN; r += kTmaRows) {
        TmaLoad4D(vdst + s * Plan::kKvSlab + r * 128, vmap, bar0 + (1 + stage) * 8, s * 64,
                  static_cast<int>(kvh), row0 + r, static_cast<int>(b));
      }
    }
  };

  for (int t = t0; t < t0 + min(steps, Plan::kStages); ++t) issue_kv(t);
  MbarWait(bar_q, 0);

  const int elem_fmt = ElemTraits<Elem>::kFormat;
  const uint32_t s_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/0);
  // V arrives as kSlabsV slabs of 64 columns (one TMA 128 B-swizzle atom each), not
  // as one Dv-wide MN-major image, so PV is kSlabsV MMAs of N=64 - each with its own
  // V slab descriptor and its own O column window - not one MMA with N=Dv. A single
  // N=kDv instruction would stride one descriptor across slabs that are not adjacent
  // in the layout it describes, and would read the wrong V columns for every Dv > 64.
  const uint32_t pv_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/1);

  const bool seg = detail::SegmentTest(p, b);
  // PRECONDITION, written at the read site: `p.q_segment`
  // must be a DENSE plane, i.e. its row stride is `p.q_length`. This leaf indexes it as
  // `q_segment[b * q_length + i]`, and the k_segment read below as `k_segment[b * nt_total
  // + j]`, so a strided view reads another batch row's metadata. On the only host path in
  // this tree it holds: sliding_chunk_attention.py passes `q_segment_idx = k_segment_idx
  // [:, -N:]`, whose row stride is C+N exactly when previous K/V is given, and
  // host.cc:159-160 calls `.contiguous()` on both planes before Params is built. The case
  // a caller could get wrong is `prev_length > 0 && batch > 1`; no traced shape reaches it
  // (training has no previous K/V, prefill is batch 1). Any future UMMA host path that
  // bypasses PrepareCommon must hand over dense storage, or carry a density flag - not
  // read across rows, which the contract cannot carry.
  const int64_t qseg = (seg && row_in_range) ? p.q_segment[b * p.q_length + i] : 0;
  const bool sub = p.subtract_before_scale;
  const float sl2 = p.scale_log2;

  float m2 = -INFINITY;  // running max: log2-scaled units at level 0, raw at level >= 1
  float lsum = 0.f;

  for (int t = t0; t <= t1; ++t) {
    const int nt = t - t0;
    const int stage = nt % Plan::kStages;
    MbarWait(bar0 + (1 + stage) * 8, static_cast<uint32_t>((nt / Plan::kStages) & 1));

    uint32_t const kb_smem = SmemAddr(sk + stage * Plan::kSlabsK * Plan::kKvSlab);
    uint32_t const vb_smem = SmemAddr(sv + stage * Plan::kSlabsV * Plan::kKvSlab);

    // S = Q K^T. The contraction and its commit are issued by ONE warp: every
    // warp's ElectOne() would otherwise issue the whole MMA again (4x the work,
    // 4 commit arrivals into a count-1 barrier).
    if (warp == 0) {
      for (int kk = 0; kk < kD / 16; ++kk) {
        const int slab = kk >> 2, within = kk & 3;
        MmaSs(tbase + Plan::kSCol, MakeSmemDesc(SmemAddr(sq) + slab * Plan::kQSlab + within * 32),
              MakeSmemDesc(kb_smem + slab * Plan::kKvSlab + within * 32), s_idesc,
              kk == 0 ? 0u : 1u);
      }
      MmaCommit(bar_s);
    }
    MbarWait(bar_s, static_cast<uint32_t>(nt & 1));

    const int64_t kb = static_cast<int64_t>(t) * kBN;
    const bool full = TileIsFull(seg, kb, max_lo, min_hi);
    if (seg && !full) {
      if (tid < kBN) {
        const int64_t j = kb + tid;
        kseg[tid] = j < nt_total ? p.k_segment[b * nt_total + j] : (qseg + 1);
      }
      __syncthreads();
    }

    float s[kBN];
    TmemLoadCols(lane_base | (tbase + Plan::kSCol), s, kBN);

    // Mask, then the online-softmax update of this row. The mask is applied as a
    // select on the exponent's ARGUMENT - masked entries are dropped before any
    // arithmetic - never by carrying -inf through the scale: for scale_log2 == 0
    // or < 0 such a product is NaN or +inf and one masked element would poison the
    // row sum and the lse. Valid for every finite scale
    // including 0 and negative, at every level.
    float tmax = -INFINITY;
    if (live) {
      if (full) {
#pragma unroll
        for (int c = 0; c < kBN; ++c) tmax = fmaxf(tmax, sub ? s[c] : s[c] * sl2);
      } else {
#pragma unroll
        for (int c = 0; c < kBN; ++c) {
          const int64_t j = kb + c;
          const bool ok = j <= bd.hi && j >= bd.lo && (!seg || kseg[c] == qseg);
          if (ok) tmax = fmaxf(tmax, sub ? s[c] : s[c] * sl2);
          else s[c] = -INFINITY;
        }
      }
    }

    float alpha = 1.f;  // O/lsum rescale for this key tile; stays 1 when tmax is -inf
    if (tmax > -INFINITY) {
      if (sub) {
        const float mr = fmaxf(m2, tmax);
        alpha = (m2 == -INFINITY) ? 0.f : exp2f(sl2 * (m2 - mr));
        m2 = mr;
      } else {
        const float mn = fmaxf(m2, tmax);
        alpha = (m2 == -INFINITY) ? 0.f : exp2f(m2 - mn);
        m2 = mn;
      }
    }
    // The O rescale is per-row arithmetic, but tcgen05.ld/st are WARP-COLLECTIVE
    // (.sync.aligned): rows of one warp disagree on alpha (a row can be dead, or
    // already at its final max), so the collective step runs whenever ANY lane
    // needs it and multiplies by 1.0f where a row does not. Skipping it per row
    // would diverge around a collective instruction, which is undefined; the
    // alpha == 1 skip survives at warp granularity, so the common no-change case
    // still costs no TMEM round trip. A value multiplied by 1.0f is unchanged,
    // so this does not disturb the exact-rescale property.
    // alpha == 0 is NOT a skip: it means the old row maximum is more than a
    // bf16-representable factor below the new one, so every earlier key's
    // contribution has underflowed to nothing and the old O accumulator must be
    // CLEARED, not carried. Leaving it in place adds stale, unscaled output to
    // the new tile's P*V - a wrong answer, not a rounding one. So the round trip
    // runs for alpha != 1 (including 0) and the store is a select: 0 where the
    // row underflowed, o * alpha everywhere else. The select also keeps the clear
    // exact if O ever holds an infinity, where 0 * inf would be NaN.
    const bool need_rescale = (tmax > -INFINITY) && alpha != 1.f;
    if (__any_sync(0xffffffffu, need_rescale)) {
      const float a = need_rescale ? alpha : 1.f;
      const bool clear = need_rescale && alpha == 0.f;
      for (int c = 0; c < kDv; c += 32) {
        float o[32];
        TmemLoad32(lane_base | (tbase + Plan::kOCol + c), o);
        TmemWaitLd();
#pragma unroll
        for (int j = 0; j < 32; ++j) o[j] = clear ? 0.f : o[j] * a;
        TmemStore32(lane_base | (tbase + Plan::kOCol + c), *reinterpret_cast<uint32_t(*)[32]>(o));
      }
    }

    // P row of this thread: exponentials of the visible keys, zero elsewhere.
    uint32_t pv[kBN / 2];
    {
      float tsum = 0.f;
#pragma unroll
      for (int c = 0; c < kBN; c += 2) {
        float e0 = 0.f, e1 = 0.f;
        if (live && s[c] != -INFINITY) {
          e0 = sub ? exp2f(sl2 * (s[c] - m2)) : exp2f(s[c] * sl2 - m2);
        }
        if (live && s[c + 1] != -INFINITY) {
          e1 = sub ? exp2f(sl2 * (s[c + 1] - m2)) : exp2f(s[c + 1] * sl2 - m2);
        }
        tsum += e0 + e1;
        pv[c >> 1] = ElemTraits<Elem>::Pack2(e0, e1);
      }
      if (tmax > -INFINITY) lsum = lsum * alpha + tsum;
    }
    TmemStore32(lane_base | (tbase + Plan::kPCol), pv);
    TmemWaitSt();
    // Closes three TMEM handoffs at once: this row's P store -> the PV mma, the O
    // rescale -> the next PV mma, and the S read -> the next S mma (WAR).
    TmemFenceBeforeSync();
    __syncthreads();
    TmemFenceAfterSync();

    // O += P V, CTA-uniform accumulate flag; one warp issues, as above.
    if (warp == 0) {
      for (int sl = 0; sl < Plan::kSlabsV; ++sl) {
        for (int kk = 0; kk < kBN / 16; ++kk) {
          // B (V) is MN-major and this MMA's N is 64 (one V slab), so the leading
          // byte-offset field of its descriptor is 0u, permanently. One 64-column slab
          // per MMA never walks the image stride:
          // with 64-row images and one 64-column group the image stride is kBN * 8 =
          // 512 u128, which 0 and 512 address identically at this geometry, so there is
          // no positive measurement of 512 at N = 64 and none is needed - the observed
          // fault at this site was a stride-versus-image-rows mismatch, not this value
          // (measured). It becomes 1024 only if kBN ever moves to 128 columns. The backward
          // dQ leaf's DqMnMajor is the same form with more than one slab - `kSlabs > 1 ?
          // kRows * 8 : 0u` - and the two leaves' PV-style B descriptors otherwise agree
          // (same MN-major k-step, 16 keys per 128 u128).
          //
          // The accumulate flag resets the FIRST k-step of EVERY V slab at the
          // first key tile, not only slab 0: O windows are per slab, and the
          // rescale above is not the initializer a slab may rely on (it is
          // skipped when the tile is fully dead), so a slab that is not reset
          // would accumulate onto uninitialised TMEM (the precise form is
          // `(nt == 0 && kk == 0) ? 0u : 1u`).
          MmaTs(tbase + Plan::kOCol + sl * 64, tbase + Plan::kPCol + kk * 8,
                MakeSmemDesc(vb_smem + sl * Plan::kKvSlab + kk * 16 * 128, 0u, 64), pv_idesc,
                (nt == 0 && kk == 0) ? 0u : 1u);
        }
      }
      MmaCommit(bar_pv);
    }
    MbarWait(bar_pv, static_cast<uint32_t>(nt & 1));

    if (t + Plan::kStages <= t1) issue_kv(t + Plan::kStages);
  }

  // ---------------------------------------------------------------- store
  const bool row_ok = row_in_range && live;
  const float inv = lsum > 0.f ? 1.f / lsum : 0.f;
  Elem* yrow = row_ok ? reinterpret_cast<Elem*>(p.y) +
                            ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim
                      : nullptr;
  float* srow = (row_ok && p.y_state != nullptr)
                    ? p.y_state + ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim
                    : nullptr;
  for (int c = 0; c < kDv; c += 32) {
    float o[32];
    TmemLoad32(lane_base | (tbase + Plan::kOCol + c), o);
    TmemWaitLd();
    if (row_ok) {
      for (int j = 0; j < 32; ++j) {
        const int64_t col = c + j;
        if (col >= p.value_dim) break;
        const float v = o[j] * inv;
        yrow[col] = ElemTraits<Elem>::From(v);
        if (srow != nullptr) srow[col] = v;
      }
    }
  }
  if (row_ok) {
    const float base = 0.6931471805599453f * (sub ? sl2 * m2 : m2);
    if (p.lse != nullptr) {
      // A live row whose visible set is empty has lsum == 0 and m2 still at its
      // level's initial sentinel. `base + __logf(lsum)` is not a safe way to say that:
      // at level >= 1 with a negative scale, `sub ? sl2 * m2 : m2` multiplies a
      // -INFINITY sentinel by a negative number and yields +INFINITY, so the sum is
      // +INF + (-INF) = NaN. That is a NaN LSE written for a row the contract requires
      // to be exactly zero, and it poisons the backward's `exp2(s*scale - lse)` on
      // every column of that row rather than only the invisible ones. Write the
      // empty-row convention the rest of the stack already reads (a non-finite LSE
      // means P == 0, see LseLog2 and the leaves' lse_finite predicate) instead
      // of deriving it through INF arithmetic.
      p.lse[(b * p.query_heads + hq) * p.q_length + i] =
          lsum > 0.f ? base + __logf(lsum) : -INFINITY;
    }
  } else if (row_in_range) {
    write_dead_row();
  }

  dealloc_and_exit();
}

// The pipelined level-0 forward at (256,256) and (128,128) (umma_fwd_pipe.cuh, instantiated in
// instantiations/sca_umma_fwd_pipe_*.cu), for the body host.cc resolved for the call
// (fwdpipe::ForwardBodySlot). 0 = not taken (the call resolved this file's leaf), 1 = launched,
// 2 = refused before the launch call, with nothing enqueued. A launch call that reports an error
// is none of these: TryLaunchPipe raises it, so it never reaches the fallback below.
namespace fwdpipe {
template <class E>
int TryLaunchPipe(const UmmaLaunchParams& lp, cudaStream_t stream);
}  // namespace fwdpipe

// Returns true when the kernel was launched and false when the call was refused
// before any launch call. The refusal is a status the caller MUST branch on - never
// ignore it, and never treat the outputs as written. A launch call that reports an
// error is not a refusal: it raises (fwdpipe::ThrowPostAttemptError), at both native
// launch sites, so the caller's generic fallback never runs after a launch attempt.
template <class E, int kD, int kDv>
bool LaunchUmmaForward(const UmmaLaunchParams& params, cudaStream_t stream) {
  using Plan = ForwardPlan<kD, kDv>;
  const Params& hp = params.params;
  // No level gate: the public baseline's forward is the same SBS + FP32-state
  // form at levels 1-4, so there is no level this leaf
  // must refuse. Level-specific algebra is backward-side and belongs to those
  // leaves.
  // The split previous-K/V maps can only be read a whole
  // key tile at a time (a 64-row TMA box never starts inside a swizzle atom), so
  // a partial previous-K/V tile must come in packed. Refusing here keeps the read
  // exact; the caller routes prev_length % kBN != 0 through the packed map and
  // falls back when it cannot pack. Two different alignments are in play and only
  // one of them is a kernel requirement: the host's PACKING rule is
  // kKeyMapAlignment (128, i.e. pack whenever prev_length % 128 != 0, which also
  // keeps a packed map tile-aligned for 128-key stages later), while the kernel
  // test here is the looser kBN = 64 a 64-row box actually needs. A split map
  // that satisfies 128 therefore always satisfies this test; the reverse is not
  // true, and passing a split map with prev_length % 64 != 0 is refused below.
  if (!params.maps.packed_kv && hp.prev_length > 0 && (hp.prev_length % kBN) != 0) return false;
  // The attribute is set once and its status is NOT ignorable: if the 197 KiB (or
  // 128 KiB) dynamic shared memory request is refused, the launch below fails or,
  // worse, runs with a default limit and indexes past the end of the allocation.
  // Refuse the call instead.
  // The attribute is set on EVERY launch, and both its status and the launch's own
  // status are checked. `static const` would make this a once-per-process call whose
  // failure is then cached for the rest of the process: any first call that failed for a
  // transient reason (no current context yet, a driver-internal retry) would silently
  // disable the opt-in for every later call, and the next launch would then run with
  // the default 48 KiB dynamic limit while the kernel indexes to kSmemBytes - a silent
  // out-of-bounds shared access, not a refused call. The opt-in costs one runtime call
  // per launch, which is orders below this kernel's own time and is counted in host
  // time like every other launch cost.
  // Three sites, one rule: no
  // `cudaPeekAtLastError` anywhere, and no sticky error handed to the next caller.
  //  (1) entry drain - anything latched before this call is not ours to report, and if
  //      it stayed set every check below would read it as its own;
  //  (2) the attr refusal consumes the error it raised, so a transient environment
  //      refusal leaves no residue (acceptance: the same configuration launches after);
  //  (3) the tail reads the launch's own error and raises it: an error reported after the
  //      launch call is never a refusal status.
  (void)cudaGetLastError();
  // Capability gate (at the launcher, matching the sentence
  // `umma_common_local.h` writes: "the host dispatch decides reachability
  // (capability 10.3 only)"). Every body this launcher starts is inside
  // `#if XATTN_UMMA_DEVICE_ASM`, so on a build or a device outside the compiled family the
  // kernel is EMPTY: it returns success and writes zeros and -INF for every row, which is
  // the one failure mode a caller cannot detect from the status. Fail closed here instead.
  // The bound mirrors that guard's family window (`>= 1000 && < 1200`) narrowed to the
  // only gencode this package emits for the leaf, sm_103a; a wider gencode widens this
  // line. A future dispatcher should do this once per device, not once per call.
  int cc_device = 0;
  int cc_major = 0;
  int cc_minor = 0;
  const bool cc_known =
      cudaGetDevice(&cc_device) == cudaSuccess &&
      cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, cc_device) ==
          cudaSuccess &&
      cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, cc_device) ==
          cudaSuccess;
  if (!cc_known || cc_major != 10 || cc_minor < 3) {
    (void)cudaGetLastError();
    return false;
  }
  // At the two buckets that have an instantiation of the pipelined body (umma_fwd_pipe.cuh) -
  // (256,256) and (128,128) - that body takes every call host.cc resolved to it
  // (fwdpipe::ForwardBody::kPipe), and its launch or its pre-launch refusal is this launcher's
  // result: a refusal (2) returns false here, so the caller's refusal row and generic fallback
  // apply to a call that enqueued nothing. A post-attempt launch error never arrives here as 2;
  // TryLaunchPipe raises it. Every other call - a kLeaf call in these two buckets, every other
  // bucket - reaches ForwardKernel below unchanged.
  if constexpr ((kD == 256 && kDv == 256) || (kD == 128 && kDv == 128)) {
    const int pipe = fwdpipe::TryLaunchPipe<E>(params, stream);
    if (pipe == 1) return true;
    if (pipe == 2) return false;
  }
  const cudaError_t attr = cudaFuncSetAttribute(
      reinterpret_cast<const void*>(&ForwardKernel<E, kD, kDv>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, Plan::kSmemBytes);
  if (attr != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  dim3 grid(static_cast<unsigned>((params.params.q_length + kBM - 1) / kBM),
            static_cast<unsigned>(params.params.query_heads),
            static_cast<unsigned>(params.params.batch));
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
  // Fault control, probe builds only (fwdpipe::ProbeHook; never in a shipped build): an empty grid,
  // rejected AT the launch call (cudaErrorInvalidValue under CUDA 13.4) with nothing enqueued and no
  // device fault.
  if (fwdpipe::ProbeHook("leaf-posterr")) grid.x = 0u;
#endif
  ForwardKernel<E, kD, kDv><<<grid, kThreads, Plan::kSmemBytes, stream>>>(params);
  // Site (3): the launch's own error, read here (the get also clears it, so a failed launch does
  // not hand its error to the next caller). A launch call that reports an error is NOT a
  // refusal: its status is no proof that nothing was enqueued, so it raises with the CUDA error
  // and the caller never falls back to a second producer over the same outputs - the rule of the
  // pipelined body's launch site too (fwdpipe::ThrowPostAttemptError). This launcher returns false
  // only from the checks above, all made before the launch call.
  const cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fwdpipe::PostAttemptErrorSlot() = launch_err;
    fwdpipe::ThrowPostAttemptError("the native forward leaf (umma::ForwardKernel)");
  }
  // The leaf's own launch count (_flash_sca_native_fwd_leaf_launches), beside the pipelined
  // body's: both record umma -> umma/ran, and these two counts tell them apart.
  fwdpipe::LeafLaunchCount().fetch_add(1, std::memory_order_relaxed);
  return true;
}

// Compile invariant: the
// instantiation TUs are nine-line files that expand this macro at FILE scope, so the
// template-id must be fully qualified inside the macro body. Without the qualification
// each TU sees a bare `LaunchUmmaForward` with no template of that name in scope and no
// `UmmaLaunchParams`, i.e. six errors per TU and zero instantiated kernels - a build in
// which `nm ... umma` is empty and every value row reads "blocked".
#define XATTN_UMMA_FWD_INSTANTIATE(E, D, DV)                                    \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaForward<E, D, DV>( \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
