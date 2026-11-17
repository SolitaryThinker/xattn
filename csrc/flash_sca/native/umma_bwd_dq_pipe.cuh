// Native Blackwell (sm_103a) SCA backward: the dQ leaf at (256,256), warp-specialized and
// pipelined across 64-key tiles ("dq_pipe"). LaunchUmmaDq<E, 256, 256> (umma_bwd_dq.cuh) runs it,
// BF16 and FP16, for every level-0 call it accepts while XATTN_SCA_BWD_DQ_PIPE is unset, empty or
// "1" (bwd_pipe_host.h); DqKernel serves every other call, and every call under "0".
//
// It is the QUERY-MAJOR mirror of the pipelined dK-only key leaf (umma_bwd_dk_pipe.cuh) and
// computes exactly what DqKernel<E, 256, 256> (umma_bwd_dq.cuh) computes: one CTA per (batch,
// query head, 128-query tile), the same grid, the same key-tile range t0..t1 (min live lo / max
// live hi), the same TileIsFull test, the same per-element mask (RowBounds, then the
// exact-segment test on the staged key ids), the same LSE / delta / dead-row rules, the same
// unscaled dS packed to 16 bits, and the same MMA instructions in the same per-accumulator order:
//   S(n)  = Q K(n)^T    16 x MmaSs M128 N64 K16 (k-step kk: slab kk>>2, 32 B * (kk&3)), reset kk 0
//   dP(n) = dO V(n)^T   16 x MmaSs M128 N64 K16 (slab sl, k-step kk), reset at sl 0, kk 0
//   dQ   += dS(n) K(n)  16 x MmaTs M128 N64 K16 (per K slab sl, 16-key step kk ascending; K
//                       MN-major, LBO kBN*8), reset at the first tile's kk 0 of every slab
// so every dQ element receives the same K=16 key steps in the same order from bitwise the same
// dS and K operands: dQ is BIT-EQUAL to DqKernel's. (S and dP are reset per tile, so their
// position in the issue order cannot change a value; dQ accumulates tiles ascending, as before.)
//
// What changes is only who waits for what. DqKernel runs issue -> wait -> elementwise -> dQ MMA
// -> wait -> next tile in one 128-thread CTA, so the tensor core idles while dS is formed and the
// pack idles while the MMAs run. Here, 320 threads (__launch_bounds__(320, 1)):
//   warps 0..7  pack. Warps w and w + 4 share the TMEM lane quadrant 32 (w & 3) .. +31, i.e. the
//               query ROWS 32 (w & 3) + lane - the row comes from the lane, never from tid - and
//               split the tile's 64 key columns: warp w columns 0..31, warp w + 4 columns 32..63.
//               Each warp loads its 32 S and 32 dP columns, forms 32 dS values with DqKernel's
//               branchless line, and stores its 16 packed words over the FIRST 16 of ITS OWN 32
//               columns of the S buffer (its loads completed before the store, so no warp writes
//               TMEM another warp still has to read). dS of key step kk (16 keys) is therefore at
//               column DsBase(buf) + 32 (kk >> 1) + 8 (kk & 1). No __syncthreads: the key-id
//               staging of the full-plane segment route uses a named barrier over the 256 pack
//               threads (bar.sync 1, 256), and only on tiles that need it.
//   warp 8      TMA producer (one thread): Q and dO once, then two streams polled with test_wait:
//               K(n) into K stage n & 1 once dQ(n - 2) - its last reader - completed; V(n) into
//               the single V stage once dP(n - 1) - its only reader - completed.
//   warp 9      MMA issuer (one elected lane per instruction, as every leaf here).
// MMA order: prologue dP(0) S(0) dP(1) S(1); then per tile n:
//     dQ(n) [pack(n) done]; dP(n+2) [V(n+2) landed]; S(n+2) [K(n+2) landed, dQ(n) done].
//   dS lives IN PLACE over the S buffer: the dP buffer is free as soon as pack(n) has loaded it,
//   so dP(n+2) needs no MMA completion; S(n+2) overwrites dS(n) and waits for dQ(n). The pack of
//   tile n+1 then runs under dQ(n), dP(n+2) and S(n+2), the K(n+2) load under dP(n+2), the V(n+2)
//   load under S(n+1) and dQ(n).
//
// TMEM (512 columns allocated, one CTA per SM): dQ accumulator FP32 cols 0..255 (M128, lane =
// query row); S buffer b at 256 + 64 b; dP buffer b at 384 + 64 b.
// SMEM (dynamic, 1024 B rebased, every operand base 1024-aligned):
//   Q        0 ..  65,535   128 x 256 16-bit, 4 slabs of 16 KB (two 64-row TMA boxes each)
//   dO  65,536 .. 131,071
//   K   131,072 .. 196,607   two stages of 64 x 256 (4 slabs of 8 KB)
//   V   196,608 .. 229,375   one stage
//   mbarriers 229,376 (16 slots, 12 used), TMEM slot 229,504, row reductions 229,520 (4 x 16 B),
//   key ids 229,584 .. 230,095 (64 x 8 B); rounded up to 1024 = 230,400, plus 1,024 B of
//   rebase slack = 231,424 B <= 232,448 (the sm_103a opt-in ceiling). A second V stage (+32 KB)
//   or a third K stage does not fit.
//
// Barrier phases. Tile n uses K stage, S buffer, dP buffer and pack barrier n & 1. bar_v,
// bar_vfree complete once per tile (parity n & 1); bar_k[s], bar_s[s], bar_dq[s], bar_pack[s]
// once per tile of their parity (parity (n >> 1) & 1). Each parity wait sees the previous phase
// of its barrier completed before it starts, and the next phase cannot complete before it
// returns:
//   MMA  waits bar_v (V(m)): V(m-1) observed before dP(m-1); V(m+1) needs dP(m) done (vfree).
//   MMA  waits bar_k[m&1] (K(m)): K(m-2) observed before S(m-2); K(m+2) needs dQ(m), later.
//   MMA  waits bar_pack[n&1] (pack(n)): pack(n-2) observed before dQ(n-2); pack(n+2) needs
//        S(n+2), issued after this wait.
//   MMA  waits bar_dq[x&1] (dQ(x)): dQ(x-2) observed by the same wait two tiles earlier; dQ(x+2)
//        is issued by this warp later.
//   pack waits bar_s[n&1] (S(n), dP(n)): S(n-2) observed by pack(n-2); S(n+2) needs pack(n)
//        arrived (through dQ(n)).
//   prod tests bar_dq[n&1] for dQ(n-2) before K(n): dQ(n-4) observed before K(n-2); dQ(n) needs
//        K(n). Tests bar_vfree for dP(n-1) before V(n): dP(n-2) observed before V(n-1); dP(n+1)
//        needs V(n+1), issued after this test.
//   pack (epilogue) waits bar_dq of the last tile L: dQ(L-2) was observed by the MMA warp
//        before it issued S(L), and pack(L) needed S(L).
//   Every expect_tx opens its phase only after the previous load into that slot was observed.
#pragma once

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/bwd_pipe_host.h"
#include "flash_sca/native/umma_bwd_dq.cuh"

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace dqpipe {

inline constexpr int kPackWarps = 8;
inline constexpr int kPackThreads = 32 * kPackWarps;    // 256
inline constexpr int kProducerWarp = kPackWarps;        // warp 8
inline constexpr int kMmaWarp = kPackWarps + 1;         // warp 9
inline constexpr int kThreads = 32 * (kPackWarps + 2);  // 320
inline constexpr int kPackBarrier = 1;  // bar.sync id of the 256 pack threads (0 = __syncthreads)
inline constexpr int kHalf = kBN / 2;   // key columns per pack warp
static_assert(kBM == 128 && kBN == 64, "DqKernel's tile: 128 query rows x 64 keys");
static_assert(kBN == kTmaRows, "one 64-row TMA box per K / V slab");
static_assert(kPackThreads == 2 * kBM, "two pack warps per TMEM lane quadrant");
static_assert(kHalf == 32, "one 32x32b.x32 load per pack warp and operand");

// mbarrier slots (8 B each) after the operand image.
inline constexpr int kBarQ = 0;      // Q landed (TMA, expect_tx)
inline constexpr int kBarDy = 1;     // dO landed (TMA, expect_tx)
inline constexpr int kBarK = 2;      // + (n & 1): K(n) landed in K stage n & 1 (TMA)
inline constexpr int kBarV = 4;      // V(n) landed in the single V stage (TMA)
inline constexpr int kBarVfree = 5;  // dP(n) complete: the V stage may be refilled (commit)
inline constexpr int kBarS = 6;      // + (n & 1): S(n) and dP(n) complete (commit)
inline constexpr int kBarDq = 8;     // + (n & 1): dQ(n) complete (commit)
inline constexpr int kBarPack = 10;  // + (n & 1): dS(n) packed (kPackThreads arrivals)
inline constexpr int kBarCount = 12;

template <int kD, int kDv>
struct Plan {
  // Two buckets, both on the objective: (256,256) is the 0.9B training cell, (128,128) the 4B one.
  // The slab geometry is DqPlan<kD,kDv>'s, reused and not re-derived, so the per-bucket quantities
  // here are only the byte totals, and each is asserted below.
  static_assert((kD == 256 && kDv == 256) || (kD == 128 && kDv == 128),
                "the pipelined dQ leaf is instantiated at (256,256) and (128,128)");
  using Base = DqPlan<kD, kDv>;  // DqKernel's slab geometry, reused, not re-derived
  static constexpr int kSlabsQ = Base::kSlabsQ;    // 4
  static constexpr int kSlabsK = Base::kSlabsK;    // 4
  static constexpr int kSlabsV = Base::kSlabsV;    // 4 (dO's slab count as well)
  static constexpr int kSlab = Base::kSlab;        // 16,384: 128 rows x 128 B
  static constexpr int kKvSlab = Base::kKvSlab;    // 8,192: 64 rows x 128 B
  static constexpr int kQBytes = Base::kQBytes;    // 65,536
  static constexpr int kDyBytes = Base::kDyBytes;  // 65,536
  static constexpr int kKStages = 2;
  static constexpr int kKStageBytes = kSlabsK * kKvSlab;  // 32,768
  static constexpr int kVBytes = kSlabsV * kKvSlab;       // 32,768 (one stage)
  static constexpr int kQOff = 0;
  static constexpr int kDyOff = kQOff + kQBytes;                 // 65,536
  static constexpr int kKOff = kDyOff + kDyBytes;                // 131,072
  static constexpr int kVOff = kKOff + kKStages * kKStageBytes;  // 196,608
  static constexpr int kBarOff = kVOff + kVBytes;                // 229,376
  static constexpr int kSlotOff = kBarOff + 128;                 // 229,504: 16 barrier slots
  static constexpr int kRedOff = kSlotOff + 16;                  // 229,520: 4 warps x 4 int32
  static constexpr int kSegOff = kRedOff + 4 * 16;               // 229,584: 64 key ids
  static constexpr int kEnd = kSegOff + kBN * 8;                 // 230,096
  static constexpr int kSmemBytes = ((kEnd + 1023) / 1024) * 1024 + 1024;  // 231,424
  // TMEM columns.
  static constexpr int kDqCol = 0;
  static constexpr int kSCol = 256;   // S buffer b at 256 + 64 b (dS packed in place)
  static constexpr int kDpCol = 384;  // dP buffer b at 384 + 64 b
  static constexpr int kBufCols = kBN;
  static constexpr int kAllocCols = 512;

  static_assert(kDqCol + kD <= kSCol, "dQ sits below both S buffers");
  static_assert(kSCol + 2 * kBufCols <= kDpCol, "two S buffers below the dP buffers");
  static_assert(kDpCol + 2 * kBufCols <= kAllocCols, "two dP buffers fit the allocation");
  static_assert(kBarCount * 8 <= kSlotOff - kBarOff, "barrier area");
  static_assert(kSlab % 1024 == 0 && kKvSlab % 1024 == 0 && kKStageBytes % 1024 == 0,
                "descriptor tile strides must stay 1024-aligned");
  static_assert(kQOff % 1024 == 0 && kDyOff % 1024 == 0 && kKOff % 1024 == 0 &&
                    kVOff % 1024 == 0 && kBarOff % 1024 == 0,
                "descriptor tile bases must stay 1024-aligned");
  static_assert(kSegOff % 8 == 0, "the key-id table must stay 8-aligned");
  // (256,256): 4 Q slabs + 4 dO slabs + 2x4 K slabs + 4 V slabs = 231,424 B, one CTA per SM
  // beside a 232,448 B window. (128,128): the same layout at half the row width = 116,736 B, still
  // one CTA per SM - two would be 233,472 B and miss the window by 1,024.
  static_assert(kD != 256 || kSmemBytes == 231424, "SMEM total at (256,256), computed in the header comment");
  static_assert(kD != 128 || kSmemBytes == 116736, "SMEM total at (128,128), same layout");
  static_assert(kSmemBytes <= 232448, "the sm_103a opt-in dynamic SMEM ceiling");
};

// TMEM column (relative to the allocation base) of the first packed dS word of buffer `buf`: dS
// lives in place over the S buffer.
__host__ __device__ constexpr uint32_t DsBase(uint32_t buf) {
  return static_cast<uint32_t>(256 + 64 * static_cast<int>(buf));
}
// Key step kk (16 keys): the pack warp of column half kk >> 1 wrote its 16 words at the start
// of its own 32 columns, two 16-key steps of 8 words each.
__host__ __device__ constexpr uint32_t DsColumn(uint32_t buf, int kk) {
  return DsBase(buf) + static_cast<uint32_t>(32 * (kk >> 1) + 8 * (kk & 1));
}
static_assert(DsColumn(1, 3) == 256 + 64 + 40 && DsColumn(0, 2) == 256 + 32,
              "dS columns: buffer base + 32 per column half + 8 per 16-key step");

// ---------------------------------------------------------------- local primitives
// (dk_pipe / dv_pipe spell the same instructions in their own namespaces; this header keeps its
// own copies so that it includes nothing but the dQ leaf header.)
__device__ __forceinline__ bool MbarTestWait(uint32_t mbar, uint32_t phase) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t ok;
  asm volatile("{\n\t.reg .pred P;\n\t"
               "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, P;\n\t}\n"
               : "=r"(ok)
               : "r"(mbar), "r"(phase)
               : "memory");
  return ok != 0;
#else
  (void)mbar;
  (void)phase;
  return true;
#endif
}

__device__ __forceinline__ void MbarArrive(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mbar) : "memory");
#else
  (void)mbar;
#endif
}

__device__ __forceinline__ void PackBarrierSync() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("bar.sync %0, %1;" ::"n"(kPackBarrier), "n"(kPackThreads) : "memory");
#endif
}

// 32 lanes x 16 columns, warp-collective, warp-uniform address (32x32b.x16).
__device__ __forceinline__ void TmemStore16(uint32_t addr, const uint32_t (&v)[16]) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};" ::"r"(v[0]),
      "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]),
      "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]),
      "r"(addr));
#else
  (void)addr;
  (void)v;
#endif
}

// ------------------------------------------------------------------ kernel
template <class Elem, int kD, int kDv>
__global__ void __launch_bounds__(kThreads, 1) DqPipeKernel(
    const __grid_constant__ UmmaLaunchParams lp) {
  using Pl = Plan<kD, kDv>;
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  // DqKernel's 1024-byte rebase (descriptor base_offset = 0); the slack is inside kSmemBytes.
  unsigned char* const smem = smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);

  unsigned char* const sq = smem + Pl::kQOff;
  unsigned char* const sdy = smem + Pl::kDyOff;
  unsigned char* const sk = smem + Pl::kKOff;
  unsigned char* const sv = smem + Pl::kVOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Pl::kRedOff);
  int64_t* const kseg = reinterpret_cast<int64_t*>(smem + Pl::kSegOff);
#define XATTN_DQP_BAR(slot) (bar0 + 8u * static_cast<uint32_t>(slot))

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hq = blockIdx.y;
  const int qt = blockIdx.x;
  const int64_t kvh = p.group > 0 ? hq / p.group : hq;  // GQA/MQA: as DqKernel
  const bool pack_warp = warp < kPackWarps;
  const int w4 = warp & 3;         // TMEM lane quadrant of this warp
  const int ch = (warp >> 2) & 1;  // column half of a pack warp: 0 -> keys 0..31, 1 -> 32..63
  // THE QUERY ROW OF THIS THREAD comes from the TMEM lane it may access, never from tid: in
  // DqKernel tid == row because kThreads == kBM; here warps w and w + 4 own the same 32 rows.
  const int row = 32 * w4 + lane;
  const int64_t i = static_cast<int64_t>(qt) * kBM + row;
  const bool row_in_range = pack_warp && i < p.q_length;

  if (tid == 0) {
    for (int s = 0; s < kBarCount; ++s) {
      MbarInit(XATTN_DQP_BAR(s), (s == kBarPack || s == kBarPack + 1) ? kPackThreads : 1);
    }
  }
  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) MbarInitFence();

  // Per-row visibility from the shared tables (as DqKernel). Warps 0..3 hold the 128 rows
  // exactly once, so they alone feed the CTA reductions (same values, same reduction as
  // DqKernel's four warps: min/max are order-free).
  const detail::Bounds bd =
      row_in_range ? detail::RowBounds(p, b, i) : detail::Bounds{INT_MAX, -1};
  const bool live = row_in_range && bd.lo <= bd.hi;
  if (warp < 4) {
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
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();

  int min_lo = red[0], max_hi = red[1], max_lo = red[2], min_hi = red[3];
  for (int w = 1; w < 4; ++w) {
    min_lo = min(min_lo, red[w * 4 + 0]);
    max_hi = max(max_hi, red[w * 4 + 1]);
    max_lo = max(max_lo, red[w * 4 + 2]);
    min_hi = min(min_hi, red[w * 4 + 3]);
  }
  const uint32_t tbase = *tmem_slot & 0x1FFu;  // 1-CTA allocation: lane 0, column base

  if (min_lo > max_hi) {
    // No live row in this CTA: dQ = 0 for every row in range (DqKernel's write_dead_row),
    // the two pack warps of a row writing complementary column halves.
    if (row_in_range) {
      Elem* dqrow = reinterpret_cast<Elem*>(p.dq) +
                    ((b * p.q_length + i) * p.query_heads + hq) * p.qk_dim;
      const int64_t half = (p.qk_dim + 1) / 2;
      const int64_t c_begin = ch == 0 ? 0 : half;
      const int64_t c_end = ch == 0 ? half : p.qk_dim;
      for (int64_t c = c_begin; c < c_end; ++c) dqrow[c] = ElemTraits<Elem>::From(0.f);
    }
    TmemFenceBeforeSync();
    __syncthreads();
    TmemFenceAfterSync();
    if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
    return;
  }

  const int t0 = min_lo / kBN;
  const int t1 = max_hi / kBN;
  const int steps = t1 - t0 + 1;  // >= 1 here
  const bool seg = detail::SegmentTest(p, b);
  const int64_t nt_total = p.total_length;

  if (warp == kProducerWarp) {
    // ------------------------------------------------------------- TMA producer
    if (lane == 0) {
      // Resident tiles, as DqKernel: Q and dO, kSlabs x (128 / 64) boxes each.
      MbarArriveExpectTx(XATTN_DQP_BAR(kBarQ), Pl::kQBytes);
      for (int s = 0; s < Pl::kSlabsQ; ++s) {
        for (int r = 0; r < kBM; r += kTmaRows) {
          TmaLoad4D(sq + s * Pl::kSlab + r * 128, &lp.maps.q, XATTN_DQP_BAR(kBarQ), s * 64,
                    static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
        }
      }
      MbarArriveExpectTx(XATTN_DQP_BAR(kBarDy), Pl::kDyBytes);
      for (int s = 0; s < Pl::kSlabsV; ++s) {
        for (int r = 0; r < kBM; r += kTmaRows) {
          TmaLoad4D(sdy + s * Pl::kSlab + r * 128, &lp.maps.dy, XATTN_DQP_BAR(kBarDy), s * 64,
                    static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
        }
      }
      // Two streams over the same tile sequence t0 .. t1; neither waits for the other.
      int nk = 0, nv = 0;
      uint32_t idle = 0;
      while (nk < steps || nv < steps) {
        bool moved = false;
        if (nk < steps &&
            (nk < 2 || MbarTestWait(XATTN_DQP_BAR(kBarDq + (nk & 1)),
                                    static_cast<uint32_t>(((nk - 2) >> 1) & 1)))) {
          const int64_t j0 = static_cast<int64_t>(t0 + nk) * kBN;
          const CUtensorMap* kmap;
          const CUtensorMap* vmap;
          int row0;
          ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);  // as DqKernel, packed map included
          const uint32_t bk = XATTN_DQP_BAR(kBarK + (nk & 1));
          MbarArriveExpectTx(bk, Pl::kKStageBytes);
          unsigned char* const kdst = sk + (nk & 1) * Pl::kKStageBytes;
          for (int s = 0; s < Pl::kSlabsK; ++s) {
            for (int r = 0; r < kBN; r += kTmaRows) {
              TmaLoad4D(kdst + s * Pl::kKvSlab + r * 128, kmap, bk, s * 64,
                        static_cast<int>(kvh), row0 + r, static_cast<int>(b));
            }
          }
          ++nk;
          moved = true;
        }
        if (nv < steps &&
            (nv == 0 || MbarTestWait(XATTN_DQP_BAR(kBarVfree),
                                     static_cast<uint32_t>((nv - 1) & 1)))) {
          const int64_t j0 = static_cast<int64_t>(t0 + nv) * kBN;
          const CUtensorMap* kmap;
          const CUtensorMap* vmap;
          int row0;
          ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);
          MbarArriveExpectTx(XATTN_DQP_BAR(kBarV), Pl::kVBytes);
          for (int s = 0; s < Pl::kSlabsV; ++s) {
            for (int r = 0; r < kBN; r += kTmaRows) {
              TmaLoad4D(sv + s * Pl::kKvSlab + r * 128, vmap, XATTN_DQP_BAR(kBarV), s * 64,
                        static_cast<int>(kvh), row0 + r, static_cast<int>(b));
            }
          }
          ++nv;
          moved = true;
        }
        if (moved) {
          idle = 0;
        } else {
          // Bounded like MbarWait: a stream that never becomes issuable is a kernel bug.
          if (++idle > (1u << 22)) __trap();
          __nanosleep(20);
        }
      }
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    // ------------------------------------------------------------- MMA issuer
    const int elem_fmt = ElemTraits<Elem>::kFormat;
    // DqKernel's two instruction descriptors and dQ's B leading byte offset, verbatim.
    const uint32_t ss_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/0);
    const uint32_t ts_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/1);
    const uint32_t dq_lbo = Pl::kSlabsK > 1 ? static_cast<uint32_t>(kBN * 8) : 0u;
    const uint32_t qb_smem = SmemAddr(sq);
    const uint32_t dyb_smem = SmemAddr(sdy);
    const uint32_t k0_smem = SmemAddr(sk);
    const uint32_t vb_smem = SmemAddr(sv);
    // (1) S(n) = Q K(n)^T into S buffer n & 1: DqKernel's k-loop.
    auto issue_s = [&](int n) {
      const uint32_t kb_smem = k0_smem + static_cast<uint32_t>((n & 1) * Pl::kKStageBytes);
      const uint32_t dst = tbase + Pl::kSCol + static_cast<uint32_t>((n & 1) * Pl::kBufCols);
      for (int kk = 0; kk < kD / 16; ++kk) {
        const int slab = kk >> 2, within = kk & 3;
        MmaSs(dst, MakeSmemDesc(qb_smem + slab * Pl::kSlab + within * 32),
              MakeSmemDesc(kb_smem + slab * Pl::kKvSlab + within * 32), ss_idesc,
              kk == 0 ? 0u : 1u);
      }
    };
    // (2) dP(n) = dO V(n)^T into dP buffer n & 1: DqKernel's (slab, k-step) loop.
    auto issue_dp = [&](int n) {
      const uint32_t dst = tbase + Pl::kDpCol + static_cast<uint32_t>((n & 1) * Pl::kBufCols);
      for (int sl = 0; sl < Pl::kSlabsV; ++sl) {
        for (int kk = 0; kk < 4; ++kk) {
          MmaSs(dst, MakeSmemDesc(dyb_smem + sl * Pl::kSlab + kk * 32),
                MakeSmemDesc(vb_smem + sl * Pl::kKvSlab + kk * 32), ss_idesc,
                (sl == 0 && kk == 0) ? 0u : 1u);
        }
      }
    };
    // (5) dQ += dS(n) K(n): DqKernel's per-slab loop, A = dS of buffer n & 1 from TMEM.
    auto issue_dq = [&](int n) {
      const uint32_t kb_smem = k0_smem + static_cast<uint32_t>((n & 1) * Pl::kKStageBytes);
      const uint32_t buf = static_cast<uint32_t>(n & 1);
      for (int sl = 0; sl < Pl::kSlabsK; ++sl) {
        for (int kk = 0; kk < kBN / 16; ++kk) {
          MmaTs(tbase + Pl::kDqCol + sl * 64, tbase + DsColumn(buf, kk),
                MakeSmemDesc(kb_smem + sl * Pl::kKvSlab + kk * 16 * 128, dq_lbo, 64), ts_idesc,
                (n == 0 && kk == 0) ? 0u : 1u);
        }
      }
    };
    MbarWait(XATTN_DQP_BAR(kBarQ), 0u);
    MbarWait(XATTN_DQP_BAR(kBarDy), 0u);
    const int pro = steps < 2 ? steps : 2;
    for (int m = 0; m < pro; ++m) {
      MbarWait(XATTN_DQP_BAR(kBarV), static_cast<uint32_t>(m & 1));
      TmemFenceAfterSync();
      __syncwarp();
      issue_dp(m);
      MmaCommit(XATTN_DQP_BAR(kBarVfree));
      MbarWait(XATTN_DQP_BAR(kBarK + m), 0u);
      __syncwarp();
      issue_s(m);
      MmaCommit(XATTN_DQP_BAR(kBarS + m));  // S(m) and, issued before it, dP(m)
    }
    for (int n = 0; n < steps; ++n) {
      // dQ(n): dS(n) packed.
      MbarWait(XATTN_DQP_BAR(kBarPack + (n & 1)), static_cast<uint32_t>((n >> 1) & 1));
      TmemFenceAfterSync();
      __syncwarp();
      issue_dq(n);
      MmaCommit(XATTN_DQP_BAR(kBarDq + (n & 1)));
      const int m = n + 2;
      if (m < steps) {
        // dP(m) into dP buffer n & 1: its last reader was pack(n) (waited above); V(m) landed.
        MbarWait(XATTN_DQP_BAR(kBarV), static_cast<uint32_t>(m & 1));
        __syncwarp();
        issue_dp(m);
        MmaCommit(XATTN_DQP_BAR(kBarVfree));
        // S(m) into S buffer n & 1, where dS(n) lives: dQ(n) must have completed; K(m) landed
        // (its stage was released by the same dQ(n)).
        MbarWait(XATTN_DQP_BAR(kBarK + (m & 1)), static_cast<uint32_t>((m >> 1) & 1));
        MbarWait(XATTN_DQP_BAR(kBarDq + (n & 1)), static_cast<uint32_t>((n >> 1) & 1));
        TmemFenceAfterSync();
        __syncwarp();
        issue_s(m);
        MmaCommit(XATTN_DQP_BAR(kBarS + (m & 1)));
      }
    }
  } else {
    // ------------------------------------------------------------- pack warps 0..7
    const uint32_t lane_base = static_cast<uint32_t>(32 * w4) << 16;
    const int c0 = ch * kHalf;
    // Per-row constants, DqKernel's expressions verbatim (row i of this thread).
    const void* const saved_base =
        p.saved != nullptr ? p.saved
                           : (p.saved_fp32 ? static_cast<const void*>(p.y_state)
                                           : static_cast<const void*>(p.y));
    const float lse_i =
        (live && p.lse_in != nullptr)
            ? p.lse_in[(b * p.query_heads + hq) * p.q_length + i]
            : -INFINITY;
    const bool lse_ok = live && lse_i > -INFINITY && lse_i < INFINITY;
    const float lse_log2 = lse_ok ? lse_i * kDqLog2e : INFINITY;
    float delta = 0.f;
    if (lse_ok && p.delta != nullptr) {
      delta = p.delta[(b * p.query_heads + hq) * p.q_length + i];
    } else if (lse_ok) {
      const int64_t row0 = ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
      const Elem* const dy_row = reinterpret_cast<const Elem*>(p.dy) + row0;
      float acc = 0.f;
      if (p.saved_fp32) {
        const float* const y_row = static_cast<const float*>(saved_base) + row0;
        for (int64_t c = 0; c < p.value_dim; ++c) acc += ToFloat(dy_row[c]) * y_row[c];
      } else {
        const Elem* const y_row = static_cast<const Elem*>(saved_base) + row0;
        for (int64_t c = 0; c < p.value_dim; ++c) acc += ToFloat(dy_row[c]) * ToFloat(y_row[c]);
      }
      delta = acc;
    }
    const int64_t qseg = (seg && row_in_range) ? p.q_segment[b * p.q_length + i] : 0;
    const float sl2 = p.scale_log2;

    for (int n = 0; n < steps; ++n) {
      const uint32_t buf = static_cast<uint32_t>(n & 1);
      const int64_t kb = static_cast<int64_t>(t0 + n) * kBN;
      const bool full = TileIsFull(seg, kb, max_lo, min_hi);  // CTA-uniform
      if (seg && !full) {
        // DqKernel's key-id staging (thread t < kBN writes key t; thread t is row t here too),
        // between two pack-only barriers: the first orders it after every read of the previous
        // staged tile.
        PackBarrierSync();
        if (tid < kBN) {
          const int64_t j = kb + tid;
          kseg[tid] = j < nt_total ? p.k_segment[b * nt_total + j] : (qseg + 1);
        }
        PackBarrierSync();
      }
      auto col_visible = [&](int c) {
        if (full) return true;
        const int64_t j = kb + c;
        if (j > bd.hi || j < bd.lo) return false;
        return !seg || kseg[c] == qseg;
      };
      MbarWait(XATTN_DQP_BAR(kBarS + buf), static_cast<uint32_t>((n >> 1) & 1));
      TmemFenceAfterSync();
      float s[32];
      float dpr[32];
      TmemLoad32(lane_base | (tbase + Pl::kSCol + buf * Pl::kBufCols + c0), s);
      TmemLoad32(lane_base | (tbase + Pl::kDpCol + buf * Pl::kBufCols + c0), dpr);
      TmemWaitLd();
      uint32_t dsv[kHalf / 2];
#pragma unroll
      for (int c2 = 0; c2 < kHalf; c2 += 2) {
        const int c = c0 + c2;
        // DqKernel's branchless line, verbatim.
        const bool v0 = lse_ok && col_visible(c);
        const bool v1 = lse_ok && col_visible(c + 1);
        const float e0 = exp2f(sl2 * s[c2] - lse_log2);
        const float e1 = exp2f(sl2 * s[c2 + 1] - lse_log2);
        const float d0 = v0 ? e0 * (dpr[c2] - delta) : 0.f;
        const float d1 = v1 ? e1 * (dpr[c2 + 1] - delta) : 0.f;
        dsv[c2 >> 1] = ElemTraits<Elem>::Pack2(d0, d1);
      }
      // In place over this warp's own (already loaded) columns of the S buffer.
      TmemStore16(lane_base | (tbase + DsBase(buf) + c0), dsv);
      TmemWaitSt();
      TmemFenceBeforeSync();
      MbarArrive(XATTN_DQP_BAR(kBarPack + buf));
    }

    // ----------------------------------------------------------------- epilogue
    // DqKernel's store: lse_ok ? acc * p.scale : 0 for every row in range (a live row with an
    // empty visible set and a dead row both store exact zeros), scale applied once, here. Warps
    // 0..3 drain columns 0..127, warps 4..7 columns 128..255 of the same rows.
    MbarWait(XATTN_DQP_BAR(kBarDq + ((steps - 1) & 1)),
             static_cast<uint32_t>(((steps - 1) >> 1) & 1));
    TmemFenceAfterSync();
    Elem* const dqrow = row_in_range ? reinterpret_cast<Elem*>(p.dq) +
                                           ((b * p.q_length + i) * p.query_heads + hq) * p.qk_dim
                                     : nullptr;
    const float scale = p.scale;
    const int cbeg = ch * (kD / 2);
    for (int cc = cbeg; cc < cbeg + kD / 2; cc += 32) {
      float o[32];
      TmemLoad32(lane_base | (tbase + Pl::kDqCol + cc), o);
      TmemWaitLd();
      if (dqrow != nullptr) {
        if (cc + 32 <= p.qk_dim && (reinterpret_cast<uintptr_t>(dqrow + cc) & 15u) == 0u) {
          union {
            uint4 u4[4];
            Elem e[32];
          } pk;
#pragma unroll
          for (int x = 0; x < 32; ++x) pk.e[x] = ElemTraits<Elem>::From(lse_ok ? o[x] * scale : 0.f);
          uint4* const dst = reinterpret_cast<uint4*>(dqrow + cc);
          dst[0] = pk.u4[0];
          dst[1] = pk.u4[1];
          dst[2] = pk.u4[2];
          dst[3] = pk.u4[3];
        } else {
          for (int x = 0; x < 32; ++x) {
            const int64_t col = cc + x;
            if (col >= p.qk_dim) break;  // padding columns of the bucket are not written
            dqrow[col] = ElemTraits<Elem>::From(lse_ok ? o[x] * scale : 0.f);
          }
        }
      }
    }
  }

  // ------------------------------------------------------------------ CTA exit
  // All 320 threads: the pack warps after the epilogue (which waited for the last dQ), the
  // producer after its last issue, the MMA warp after its last commit.
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
#undef XATTN_DQP_BAR
}

// ------------------------------------------------------------------ host
// Called by LaunchUmmaDq<E, kD, kDv> at either instantiated bucket, only AFTER
// CanLaunchUmmaDq<E, kD, kDv> accepted the call
// (element, bucket and real widths, shapes, chunk and length identity, segment pairing, the
// split-map rule at kBN = 64, dY / LSE / dq presence and no dq_accum, device 10.3): this adds its
// own scope (precision level 0: no FP32 saved state, no subtract-before-scale) and its larger
// SMEM request. `false` means nothing was launched and DqKernel runs.
template <class E, int kD, int kDv>
bool CanLaunchUmmaDqPipe(const UmmaLaunchParams& lp) {
  using Pl = Plan<kD, kDv>;
  const Params& p = lp.params;
  if (p.saved_fp32 || p.subtract_before_scale) return false;
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: cudaGetDevice", q);
  int smem_cap = 0;
  q = cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (q != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: shared memory capacity", q);
  }
  return smem_cap >= Pl::kSmemBytes;
}

template <class E, int kD, int kDv>
void LaunchDqPipeKernel(const UmmaLaunchParams& lp, cudaStream_t stream) {
  using Pl = Plan<kD, kDv>;
  const Params& p = lp.params;
  const cudaError_t attr = cudaFuncSetAttribute(DqPipeKernel<E, kD, kDv>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: cudaFuncSetAttribute", attr);
  }
  const dim3 grid(static_cast<unsigned>((p.q_length + kBM - 1) / kBM),
                  static_cast<unsigned>(p.query_heads), static_cast<unsigned>(p.batch));
  DqPipeKernel<E, kD, kDv><<<grid, kThreads, Pl::kSmemBytes, stream>>>(lp);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: kernel launch", launched);
  }
}

// Declared in umma_bwd_dq.cuh. 0 = not taken (XATTN_SCA_BWD_DQ_PIPE is "0", or the pipe refused
// the call: the caller launches DqKernel); 1 = launched, counted in DqPipeLaunchCount(). The
// variable is read on every call; a value that names no arm raises (host.cc refuses it before
// the backward allocates anything, so only a caller outside that entry can reach this raise).
template <class E, int kD, int kDv>
int TryLaunchDqPipe(const UmmaLaunchParams& lp, cudaStream_t stream) {
  const char* const value = DqPipeArmValue();
  const DqPipeArm arm = ParseDqPipeArm(value);
  if (arm == DqPipeArm::kInvalid) {
    throw std::runtime_error(
        std::string("XATTN_SCA_BWD_DQ_PIPE must be unset, empty, '0' or '1'; got '") + value + "'");
  }
  if (arm == DqPipeArm::kOff || !CanLaunchUmmaDqPipe<E, kD, kDv>(lp)) return 0;
  LaunchDqPipeKernel<E, kD, kDv>(lp, stream);
  DqPipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
  return 1;
}

}  // namespace dqpipe
}  // namespace umma
}  // namespace xattn::ops::flash_sca_native

// One instantiation per (element, bucket): the TU names the bucket, the macro only the element.
#define XATTN_UMMA_DQ_PIPE_INSTANTIATE_BUCKET(E, D, DV)                                         \
  template int xattn::ops::flash_sca_native::umma::dqpipe::TryLaunchDqPipe<E, D, DV>(           \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// The (256,256) bucket's TU pair, unchanged in what it emits: this name has always meant the
// bucket whose filenames end in _pipe_<element>.cu, and it keeps meaning exactly that so the
// other bucket is not silently defined twice. Every further bucket gets its own TU pair that
// calls XATTN_UMMA_DQ_PIPE_INSTANTIATE_BUCKET with the bucket in the file name.
#define XATTN_UMMA_DQ_PIPE_INSTANTIATE(E) XATTN_UMMA_DQ_PIPE_INSTANTIATE_BUCKET(E, 256, 256)
