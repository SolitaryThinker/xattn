// Native Blackwell (sm_103a) SCA backward: the dK-only M128 key leaf at (256,256),
// warp-specialized and pipelined across 64-query rounds. It is the second leaf of the
// (256,256) key pair that the in-situ launch runs at level 0 (umma_bwd_dkdv.cuh,
// LaunchUmmaBackwardInSituDkdv): the pipelined dV-only leaf (umma_bwd_dv_pipe.cuh) runs first on
// the stream and writes dV and dPrevV; this one writes dK and dPrevK (dV and dPrevV are NOT
// written here).
//
// It computes the dK / dPrevK of the fused M64 leaf (BackwardDkdvKernel<E, 256, 256, 3, 64, 0>,
// the in-situ body at this bucket for every call the pair does not serve). One CTA owns one
// (batch, KV head, 128-key tile). The 128-query tiles are those the fused leaf's round loop
// visits for this key range (query tiles ascending inside the envelope, MaySee skipping, heads
// ascending inside a tile), and each (tile, head) is processed as TWO consecutive 64-query
// rounds (h = 0, 1), both always. Round r:
//   S^T  = K Q(r)^T     tcgen05.mma SS, M128 N64, K = D   (16 k-steps), S^T buffer r & 1
//   dP^T = V dO(r)^T    tcgen05.mma SS, M128 N64, K = Dv  (16 k-steps), dP^T buffer r & 1
//                       S^T committed on bar_s[r & 1], then dP^T on bar_dp[r & 1], so the
//                       pack computes P^T while dP^T(r) still runs
//   dS^T = P^T (dP^T - delta), P^T = exp2(S^T scale_log2 - lse2), masked: the fused leaf's
//          per-element line (pt::MakePtBlock / MakePtBlockFull on the same ColumnMeta), packed
//          to 16 bits round-to-nearest-even IN PLACE over the dP^T buffer
//   dK  += dS^T Q(r)    tcgen05.mma TS, M128 N256, K = 64 (4 k-steps of 16 queries), A = dS^T
//                       in TMEM, B = Q(r) MN-major; commit bar_dk[r & 1]
// Every dK element therefore receives the same K = 16 query steps in the same order as in the
// fused leaf (tile, head, then the eight 16-query steps of the 128-query tile, now 4 + 4), from
// bitwise the same dS^T and Q operands, so dK / dPrevK are expected BIT-EQUAL to the fused leaf.
// (A 64-key half of the tile that a visited tile cannot see adds exact zeros, as the dV-only
// M128 leaves do against the M64 leaf.)
//
// Roles, 320 threads (__launch_bounds__(320, 1)):
//   warps 0..7  pack. Warps w and w + 4 share the TMEM lane group 32 (w & 3) .. +31 (key rows
//               32 (w & 3) + lane) and split the round's 64 query columns: warp w packs columns
//               0..31, warp w + 4 columns 32..63. Each warp reads 32 S^T and 32 dP^T columns of
//               its own rows and writes its 16 packed dS^T words over the first 16 of ITS OWN 32
//               dP^T columns (its loads have completed - tcgen05.wait::ld - before the store), so
//               no warp writes TMEM another warp still has to read and the pack needs no WAR
//               barrier. dS^T(r) of query step kk (16 queries) is therefore at TMEM column
//               kdPCol + 64 (r & 1) + 32 (kk >> 1) + 8 (kk & 1).
//               The round's 64 ColumnMeta (lse2, delta, row bounds, query segment) are staged in
//               SMEM by threads 0..63 between two named barriers (bar.sync 1, 256) BEFORE the wait
//               for S^T(r), and the global gathers for round r + 1 are issued one round ahead,
//               so neither the barriers nor the gathers sit between bar_s(r) and pack_done(r).
//               After the last round: the epilogue (drain dK once the last dK MMA completed,
//               store dK / dPrevK with the scale folded, the fused leaf's rows and rules; warps
//               0..3 drain columns 0..127 and warps 4..7 columns 128..255 of the same rows).
//   warp 8      TMA producer, ONE thread polling two streams with test_wait, in round order:
//               Q(r) into Q stage r & 1 once that stage is free (r < 2, or dK(r - 2) - its last
//               reader - has completed), dO(r) into the single dO stage once dP^T(r - 1) - its
//               only reader - has completed (bar_s).
//   warp 9      MMA issuer: S^T/dP^T(0); then for each r: S^T/dP^T(r + 1) into buffers
//               (r + 1) & 1 once Q(r + 1) and dO(r + 1) have landed and dK(r - 1) - the last
//               reader of those buffers and of that Q stage - has completed; then dK(r) once
//               pack(r) is done. The tensor core runs S^T/dP^T(r + 1) under pack(r).
//               (Issue order: S^T(r + 1) once Q(r + 1) landed, BEFORE the wait for
//               pack(r); then dK(r); then dP^T(r + 1) once dO(r + 1) landed and dK(r - 1) is
//               complete - see the loop.)
// The pack warps never use __syncthreads (it would deadlock against warps 8 and 9); the CTA-wide
// barriers left are the prologue's and the tile exit's, which all 320 threads reach once.
//
// TMEM (512 columns allocated): dK accumulator FP32 cols 0..255 (M128: lane = key row);
// S^T buffer b at 256 + 64 b (256..319, 320..383); dP^T buffer b at 384 + 64 b (384..447,
// 448..511), its packed dS^T in place (16 words per 32-column half, see above).
// SMEM (dynamic, operand bases 1024-aligned after the prologue's rebase):
//   Q stage 0      0 ..  32,767   64 x 256 16-bit, 4 slabs of 8 KB (one 64-row TMA box each)
//   Q stage 1 32,768 ..  65,535
//   dO        65,536 ..  98,303   one stage
//   K         98,304 .. 163,839   128 x 256 16-bit, 4 slabs of 16 KB
//   V        163,840 .. 229,375
//   mbarriers 229,376 (16 slots, 12 used; do not widen past 16 - the area is 128 B), TMEM slot 229,504, envelope words 229,520,
//   ColumnMeta[64] 229,552 .. 231,087 (24 B each); rounded up to 1024 = 231,424, plus 1,024 B
//   of alignment slack = 232,448 B = the sm_103a opt-in ceiling, exactly. No static SMEM. The
//   key segments are not staged (each pack thread keeps its own row's in a register).
//
// Barrier phases. Round r uses buffer / Q stage r & 1. bar_g completes once per round (parity
// r & 1); bar_q[st], bar_s[b], bar_dk[b] and bar_pack[b] complete once per round of their stage
// or buffer (parity (r >> 1) & 1). Each parity wait below sees the previous phase of its barrier
// completed before it starts, and the next phase cannot complete before it returns:
//   MMA  waits bar_q[(r+1)&1] (Q(r+1)): Q(r-1) observed before S(r-1); Q(r+3) needs dK(r+1),
//        which it issues later.
//   MMA  waits bar_g (dO(r+1)): dO(r) observed before dP^T(r); dO(r+2) needs dP^T(r+1).
//   MMA  waits bar_dk[(r-1)&1] (dK(r-1)): dK(r-3) observed before S(r-1); dK(r+1) is its own.
//   MMA  waits bar_pack[r&1] (pack(r)): pack(r-2) observed before dK(r-2); pack(r+2) needs
//        S(r+2), which it issues only after this wait.
//   pack waits bar_s[r&1] (S^T(r)) and then bar_dp[r&1] (dP^T(r)): S(r-2)/dP(r-2) observed by
//        pack(r-2); S(r+2)/dP(r+2) need dK(r) complete, which needs pack(r) arrived, i.e.
//        every pack thread past both waits.
//   prod tests bar_dk[r&1] for dK(r-2) before Q(r): dK(r-4) was observed before Q(r-2); dK(r)
//        needs Q(r). Tests bar_dp[(r-1)&1] for dP^T(r-1) before dO(r): dP(r-3) was observed
//        before dO(r-2); dP(r+1) needs dO(r+1), issued after this test.
//   The expect_tx of Q(r) opens bar_q[r&1]'s phase only after Q(r-2) was consumed (dK(r-2)
//   done implies S(r-2) issued after its wait), and that of dO(r) only after dO(r-1) was
//   consumed (S/dP(r-1) done), so no barrier ever holds two pending expect_tx.
//   pack (epilogue) waits bar_dk of the last round L: dK(L-2) completed before S(L) was issued,
//        and nothing is committed to that barrier after dK(L).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/umma_bwd_dkdv.cuh"
#include "flash_sca/native/umma_bwd_dv_pipe.cuh"  // dvpipe::MbarTestWait/MbarArrive/NextTile/...

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace dkpipe {

inline constexpr int kPackWarps = 8;
inline constexpr int kPackThreads = 32 * kPackWarps;    // 256
inline constexpr int kProducerWarp = kPackWarps;        // warp 8
inline constexpr int kMmaWarp = kPackWarps + 1;         // warp 9
inline constexpr int kThreads = 32 * (kPackWarps + 2);  // 320
inline constexpr int kKeyRows = 128;
inline constexpr int kRoundQueries = 64;  // queries per round: half of a 128-query tile
inline constexpr int kHalves = kBwdQueryRows / kRoundQueries;
static_assert(kHalves == 2, "a 128-query tile is two 64-query rounds");
static_assert(kPackThreads == dvpipe::kPackThreads,
              "the pack barrier is dvpipe's named barrier over 256 threads");
static_assert(kRoundQueries == kTmaRows, "one 64-row TMA box per slab of a round's Q / dO");
static_assert(kRoundQueries == 2 * pt::kBlock, "two pack warps split a round's columns");
static_assert(kRoundQueries <= kPackThreads, "one ColumnMeta writer per column");

// mbarrier slots (8 B each) after the operand image.
inline constexpr int kBarKv = 0;    // K and V landed (once per CTA)
inline constexpr int kBarQ = 1;     // + (r & 1): Q(r) landed in stage r & 1 (TMA, expect_tx)
inline constexpr int kBarG = 3;     // dO(r) landed (TMA, expect_tx), single stage
inline constexpr int kBarS = 4;     // + (r & 1): S^T(r) complete (its own commit)
inline constexpr int kBarDk = 6;    // + (r & 1): dK(r) complete (tcgen05.commit)
inline constexpr int kBarPack = 8;  // + (r & 1): dS^T(r) packed (kPackThreads arrivals)
inline constexpr int kBarDp = 10;   // + (r & 1): dP^T(r) complete (committed after S^T(r))
inline constexpr int kBarCount = 12;

template <int kD, int kDv>
struct Plan {
  // General by capacity, not by bucket name: dK needs kD <= 256 TMEM columns (asserted
  // below, kdKCol + kD <= kSCol), and the S^T/dP^T round buffers are sized by kRoundQueries
  // (64, half a query tile) regardless of kD or kDv. kDv only sets how much V this leaf
  // stages (kSlabsV below); it carries no TMEM-column requirement of its own.
  static_assert(kD % 64 == 0 && kDv % 64 == 0 && kD > 0 && kD <= 256 && kDv > 0 && kDv <= 256,
                "the pipelined dK-only leaf fits every (D, Dv) with D, Dv <= 256, each a "
                "multiple of 64");
  static constexpr int kSlabsK = kD / 64;                         // 4
  static constexpr int kSlabsV = kDv / 64;                        // 4
  static constexpr int kKeySlabBytes = kKeyRows * 128;            // 16,384
  static constexpr int kRoundSlabBytes = kRoundQueries * 128;     // 8,192
  static constexpr int kKBytes = kSlabsK * kKeySlabBytes;         // 65,536
  static constexpr int kVBytes = kSlabsV * kKeySlabBytes;         // 65,536
  static constexpr int kQStageBytes = kSlabsK * kRoundSlabBytes;  // 32,768
  static constexpr int kQStages = 2;
  static constexpr int kGBytes = kSlabsV * kRoundSlabBytes;       // 32,768
  static constexpr int kQOff = 0;
  static constexpr int kGOff = kQOff + kQStages * kQStageBytes;   // 65,536
  static constexpr int kKOff = kGOff + kGBytes;                   // 98,304
  static constexpr int kVOff = kKOff + kKBytes;                   // 163,840
  static constexpr int kBarOff = kVOff + kVBytes;                 // 229,376
  static constexpr int kSlotOff = kBarOff + 128;                  // 229,504: 16 barrier slots
  static constexpr int kRedOff = kSlotOff + 16;                   // 229,520
  // ColumnMeta staged as struct-of-arrays: the key-major pack reads one block
  // column's lse2/delta/lo/hi/segment for all 64 query columns, so an AoS table
  // forces a strided 24 B load per column (the DkPipe LDS a profile measured). As SoA
  // the lse2/delta planes are contiguous floats (vectorizable) and the pack stores
  // are bank-conflict-free. Same 1,536 bytes as the AoS table.
  static constexpr int kMetaOff = kRedOff + 32;                   // 229,552
  static constexpr int kLse2Off = kMetaOff;                       // 64 x float
  static constexpr int kDeltaOff = kLse2Off + kRoundQueries * 4;
  static constexpr int kLoOff = kDeltaOff + kRoundQueries * 4;
  static constexpr int kHiOff = kLoOff + kRoundQueries * 4;
  static constexpr int kSegOff = kHiOff + kRoundQueries * 4;
  // One int per 32-column block of the round: the block passed pt::BlockIsSuffix.
  static constexpr int kBlkOff = kSegOff + kRoundQueries * 8;     // 231,088
  static constexpr int kEnd = kBlkOff + (kRoundQueries / 32) * 4; // 231,096
  static constexpr int kSmemBytes = ((kEnd + 1023) / 1024) * 1024 + 1024;  // 232,448
  // TMEM columns.
  static constexpr int kdKCol = 0;
  static constexpr int kSCol = 256;   // S^T buffer b at 256 + 64 b
  static constexpr int kdPCol = 384;  // dP^T buffer b at 384 + 64 b, packed dS^T in place
  static constexpr int kBufCols = kRoundQueries;
  static constexpr int kAllocCols = 512;
  // dK B operand (Q, MN-major): the next 64-wide D group is one 64-row slab further.
  static constexpr uint32_t kQSlabLbo = static_cast<uint32_t>(kRoundSlabBytes / 16);  // 512

  static_assert(sizeof(pt::ColumnMeta) == 24 && alignof(pt::ColumnMeta) <= 8,
                "the ColumnMeta table is 64 x 24 B");
  static_assert(kdKCol + kD <= kSCol, "dK sits below both S^T buffers");
  static_assert(kSCol + 2 * kBufCols <= kdPCol, "two S^T buffers below the dP^T buffers");
  static_assert(kdPCol + 2 * kBufCols <= kAllocCols, "two dP^T buffers fit the allocation");
  static_assert(kBarCount * 8 <= kSlotOff - kBarOff, "barrier area");
  static_assert(kKeySlabBytes % 1024 == 0 && kRoundSlabBytes % 1024 == 0 &&
                    kQStageBytes % 1024 == 0,
                "descriptor tile bases must stay 1024-aligned");
  static_assert(kQOff % 1024 == 0 && kGOff % 1024 == 0 && kKOff % 1024 == 0 &&
                    kVOff % 1024 == 0,
                "descriptor tile bases must stay 1024-aligned");
  static_assert(kMetaOff % 8 == 0, "the ColumnMeta table must stay 8-aligned");
  static_assert(kRoundSlabBytes / 16 < (1 << 14), "the slab hop must encode in the LBO field");
  // At (256,256) kSmemBytes lands exactly on 232,448 B (the opt-in ceiling), which the
  // original leaf asserted with ==: any future byte added to this Plan's (256,256) staging
  // breaks the build there, which is the point. Generalized to every (D, Dv) <= 256 the
  // total is smaller (fewer K/V/Q/dO bytes staged), so the exact-232448 reading is no longer
  // a universal truth about this Plan - only a fact about its largest bucket - and the
  // assert that must hold at every (D, Dv) is the ceiling alone; CanLaunchUmmaBackwardDkPipe
  // also checks it at run time against the device's actual opt-in limit.
  static_assert(kSmemBytes <= 232448, "the sm_103a opt-in dynamic SMEM ceiling");
  static_assert(kD != 256 || kDv != 256 || kSmemBytes == 232448,
                "(256,256) stages the most bytes this Plan ever stages; if this no longer "
                "holds exactly, something shrank its SMEM use silently");
};

// TMEM column of the packed dS^T of round buffer `buf`, query step kk (16 queries): the pack
// warp of column half kk >> 1 writes its 16 words at the start of its own 32 dP^T columns.
__host__ __device__ constexpr uint32_t DsColumn(uint32_t buf, int kk) {
  return static_cast<uint32_t>(384 + 64 * static_cast<int>(buf) + 32 * (kk >> 1) + 8 * (kk & 1));
}
static_assert(DsColumn(0, 0) == Plan<256, 256>::kdPCol &&
                  DsColumn(1, 3) == Plan<256, 256>::kdPCol + Plan<256, 256>::kBufCols + 40,
              "dS^T columns: dP^T buffer base + 32 per column half + 8 per 16-query step");

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

// One round of the sequence: query tile t (128 rows), query head gh of the KV group, half h.
struct Round {
  int64_t t = 0;
  int64_t gh = 0;
  int h = 0;
  QueryTileMask m;
};

__device__ __forceinline__ void FirstRound(const Params& p, int64_t b, int64_t kb, int64_t ke,
                                           int64_t first, int64_t last, Round* c) {
  c->t = dvpipe::NextTile(p, b, kb, ke, first, last, &c->m);
  c->gh = 0;
  c->h = 0;
}

// Halves ascend inside a (tile, head), heads inside a tile, tiles by NextTile (the fused leaf's
// own skip test over this key range). c->t > last means the sequence has ended.
__device__ __forceinline__ void NextRound(const Params& p, int64_t b, int64_t kb, int64_t ke,
                                          int64_t last, Round* c) {
  if (++c->h < kHalves) return;
  c->h = 0;
  if (++c->gh < p.group) return;
  c->gh = 0;
  c->t = dvpipe::NextTile(p, b, kb, ke, c->t + 1, last, &c->m);
}

// kFp32Keys: dK/dPrevK in FP32 (Params::key_grads_fp32), a separate instantiation so that the
// 16-bit one keeps its own store code.
template <typename E, int kD, int kDv, bool kFp32Keys = false>
__global__ void __launch_bounds__(kThreads, 1) BackwardDkPipeKernel(
    const __grid_constant__ UmmaLaunchParams lp) {
  using Pl = Plan<kD, kDv>;
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  // Same 1024-byte rebase as the fused leaf (descriptor base_offset = 0).
  unsigned char* const smem = smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);

  unsigned char* const sq = smem + Pl::kQOff;
  unsigned char* const sg = smem + Pl::kGOff;
  unsigned char* const sk = smem + Pl::kKOff;
  unsigned char* const sv = smem + Pl::kVOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Pl::kRedOff);
  float* const smeta_lse2 = reinterpret_cast<float*>(smem + Pl::kLse2Off);
  float* const smeta_delta = reinterpret_cast<float*>(smem + Pl::kDeltaOff);
  int32_t* const smeta_lo = reinterpret_cast<int32_t*>(smem + Pl::kLoOff);
  int32_t* const smeta_hi = reinterpret_cast<int32_t*>(smem + Pl::kHiOff);
  int64_t* const smeta_seg = reinterpret_cast<int64_t*>(smem + Pl::kSegOff);
  int32_t* const smeta_blk = reinterpret_cast<int32_t*>(smem + Pl::kBlkOff);
  const uint32_t bar_kv = bar0 + 8 * kBarKv;
  const uint32_t bar_g = bar0 + 8 * kBarG;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hk = blockIdx.y;
  const int64_t kt = blockIdx.x;
  const int64_t kb = kt * kKeyRows;  // first global key of this tile
  const int64_t ke = kb + kKeyRows;  // exclusive
  const int64_t nt_total = p.total_length;

  if (tid == 0) {
    for (int i = 0; i < kBarCount; ++i) {
      MbarInit(bar0 + 8 * i, (i == kBarPack || i == kBarPack + 1) ? kPackThreads : 1);
    }
  }
  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) MbarInitFence();

  // ---- the key tile, once: K and V into SMEM
  const bool seg = detail::SegmentTest(p, b);
  if (tid == 0) {
    const CUtensorMap* kmap;
    const CUtensorMap* vmap;
    int row0;
    ResolveKeyMap(lp, kb, &kmap, &vmap, &row0);
    MbarArriveExpectTx(bar_kv, Pl::kKBytes + Pl::kVBytes);
    for (int s = 0; s < Pl::kSlabsK; ++s) {
      for (int r = 0; r < kKeyRows; r += kTmaRows) {
        TmaLoad4D(sk + s * Pl::kKeySlabBytes + r * 128, kmap, bar_kv, s * 64,
                  static_cast<int>(hk), row0 + r, static_cast<int>(b));
      }
    }
    for (int s = 0; s < Pl::kSlabsV; ++s) {
      for (int r = 0; r < kKeyRows; r += kTmaRows) {
        TmaLoad4D(sv + s * Pl::kKeySlabBytes + r * 128, vmap, bar_kv, s * 64,
                  static_cast<int>(hk), row0 + r, static_cast<int>(b));
      }
    }
  }

  int64_t qt_first, qt_last;
  if (tid == 0) {
    QueryTileEnvelope<kKeyRows>(p, b, kt, kb, ke - 1, &qt_first, &qt_last);
    red[0] = static_cast<int32_t>(qt_first);
    red[1] = static_cast<int32_t>(qt_last);
  }
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  qt_first = red[0];
  qt_last = red[1];

  const uint32_t tbase = *tmem_slot & 0x1FFu;  // 1-CTA allocation: lane 0, column base
  const int elem_fmt = BwdTok<E>::kFormat;

  if (warp == kProducerWarp) {
    // ------------------------------------------------------------- TMA producer
    // One thread, two cursors over the same round sequence. Q(rq) into stage rq & 1 once that
    // stage's last reader dK(rq - 2) has completed; dO(rg) into the single dO stage once its
    // only reader dP^T(rg - 1) has completed. Neither stream waits for the other.
    if (lane == 0) {
      Round qc, gc;
      FirstRound(p, b, kb, ke, qt_first, qt_last, &qc);
      gc = qc;
      uint32_t rq = 0, rg = 0;
      uint32_t idle = 0;
      while (qc.t <= qt_last || gc.t <= qt_last) {
        bool moved = false;
        if (qc.t <= qt_last &&
            (rq < 2 ||
             dvpipe::MbarTestWait(bar0 + 8 * (kBarDk + (rq & 1)), ((rq - 2) >> 1) & 1))) {
          const uint32_t st = rq & 1;
          const uint32_t bq = bar0 + 8 * (kBarQ + st);
          const int hq = static_cast<int>(hk * p.group + qc.gh);
          const int i0 = static_cast<int>(qc.t * kBwdQueryRows + qc.h * kRoundQueries);
          MbarArriveExpectTx(bq, Pl::kQStageBytes);
          for (int s = 0; s < Pl::kSlabsK; ++s) {
            TmaLoad4D(sq + st * Pl::kQStageBytes + s * Pl::kRoundSlabBytes, &lp.maps.q, bq,
                      s * 64, hq, i0, static_cast<int>(b));
          }
          ++rq;
          NextRound(p, b, kb, ke, qt_last, &qc);
          moved = true;
        }
        if (gc.t <= qt_last &&
            (rg == 0 ||
             dvpipe::MbarTestWait(bar0 + 8 * (kBarDp + ((rg - 1) & 1)), ((rg - 1) >> 1) & 1))) {
          const int hq = static_cast<int>(hk * p.group + gc.gh);
          const int i0 = static_cast<int>(gc.t * kBwdQueryRows + gc.h * kRoundQueries);
          MbarArriveExpectTx(bar_g, Pl::kGBytes);
          for (int s = 0; s < Pl::kSlabsV; ++s) {
            TmaLoad4D(sg + s * Pl::kRoundSlabBytes, &lp.maps.dy, bar_g, s * 64, hq, i0,
                      static_cast<int>(b));
          }
          ++rg;
          NextRound(p, b, kb, ke, qt_last, &gc);
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
    // K/V are waited for unconditionally: a CTA with no round must not exit while its K/V load
    // is still writing shared memory.
    MbarWait(bar_kv, 0);
    Round c;
    FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    if (c.t <= qt_last) {
      const uint32_t s_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kRoundQueries, /*a=*/0, /*b=*/0);
      const uint32_t k_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kD, /*a=*/0, /*b=*/1);
      // S^T = K Q^T and dP^T = V dO^T into buffer r & 1 (the fused leaf's k-loops, N = 64).
      auto issue_sdp = [&](uint32_t r, int parts = 3) {
        const uint32_t buf = r & 1;
        const uint32_t qs = SmemAddr(sq) + buf * Pl::kQStageBytes;
        if (parts & 1) {
        for (int kk = 0; kk < kD / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(tbase + Pl::kSCol + buf * Pl::kBufCols,
                MakeSmemDesc(SmemAddr(sk) + slab * Pl::kKeySlabBytes + within * 32),
                MakeSmemDesc(qs + slab * Pl::kRoundSlabBytes + within * 32), s_idesc,
                kk == 0 ? 0u : 1u);
        }
        // S^T(r) is published on its own, so the pack computes P^T under dP^T(r).
        MmaCommit(bar0 + 8 * (kBarS + buf));
        }
        if (!(parts & 2)) return;
        for (int kk = 0; kk < kDv / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(tbase + Pl::kdPCol + buf * Pl::kBufCols,
                MakeSmemDesc(SmemAddr(sv) + slab * Pl::kKeySlabBytes + within * 32),
                MakeSmemDesc(SmemAddr(sg) + slab * Pl::kRoundSlabBytes + within * 32), s_idesc,
                kk == 0 ? 0u : 1u);
        }
        MmaCommit(bar0 + 8 * (kBarDp + buf));
      };
      // dK += dS^T Q: one N = 256 instruction per 16-query step (the fused leaf's wide form),
      // B = Q(r) MN-major with the 64-row image's slab hop as LBO.
      auto issue_dk = [&](uint32_t r) {
        const uint32_t buf = r & 1;
        const uint32_t qs = SmemAddr(sq) + buf * Pl::kQStageBytes;
        for (int kk = 0; kk < kRoundQueries / 16; ++kk) {
          MmaTs(tbase + Pl::kdKCol, tbase + DsColumn(buf, kk),
                MakeSmemDesc(qs + kk * 16 * 128, Pl::kQSlabLbo, 64), k_idesc,
                (r == 0 && kk == 0) ? 0u : 1u);
        }
        MmaCommit(bar0 + 8 * (kBarDk + buf));
      };
      uint32_t r = 0;
      MbarWait(bar0 + 8 * kBarQ, 0u);
      MbarWait(bar_g, 0u);
      TmemFenceAfterSync();
      __syncwarp();
      issue_sdp(0u);
      // S^T(r+1) is issued BEFORE the wait for pack(r): its S^T buffer was last read by
      // pack(r-1), already observed here, and its Q stage holds Q(r+1) once bar_q says so. The
      // tensor pipe runs it under pack(r)'s chain instead of idling from dP^T(r)'s end to dK(r).
      // dK(r) follows pack(r), and dP^T(r+1) follows dK(r): its dP^T buffer holds dS^T(r-1)
      // until dK(r-1) completes, and dO(r+1) lands only after dP^T(r). Only S^T's 16 MMAs go
      // ahead of dK(r) (an earlier order put all 32 there and dK(r) queued behind them). The MMAs, their
      // operands and dK's accumulation order are unchanged, so dK is bit-identical.
      for (;;) {
        Round n = c;
        NextRound(p, b, kb, ke, qt_last, &n);
        const bool more = n.t <= qt_last;
        if (more) {
          MbarWait(bar0 + 8 * (kBarQ + ((r + 1) & 1)), ((r + 1) >> 1) & 1);
          TmemFenceAfterSync();
          __syncwarp();
          issue_sdp(r + 1, 1);
        }
        MbarWait(bar0 + 8 * (kBarPack + (r & 1)), (r >> 1) & 1);
        TmemFenceAfterSync();
        __syncwarp();
        issue_dk(r);
        if (!more) break;
        MbarWait(bar_g, (r + 1) & 1);
        if (r >= 1) MbarWait(bar0 + 8 * (kBarDk + ((r - 1) & 1)), ((r - 1) >> 1) & 1);
        TmemFenceAfterSync();
        __syncwarp();
        issue_sdp(r + 1, 2);
        c = n;
        ++r;
      }
    }
  } else {
    // ------------------------------------------------------------- pack warps 0..7
    const int w4 = warp & 3;
    const int ch = warp >> 2;  // column half of the round: 0 -> 0..31, 1 -> 32..63
    const uint32_t lane_base = static_cast<uint32_t>(32 * w4) << 16;
    const int64_t row_in_tile = static_cast<int64_t>(w4 * 32 + lane);
    const int64_t j = kb + row_in_tile;
    // The fused leaf's kseg entry of this row, kept in a register (the row never changes).
    const int64_t ks = (seg && j < nt_total) ? p.k_segment[b * nt_total + j] : 0;
    // Precondition of the suffix-mask path, and it is not the analytic-bounds test alone:
    // the visible set of key row j is a suffix of a 32-column block only if no chunk boundary
    // falls inside the block. A block's query positions are p = i + (total_length - q_length),
    // so that holds when the chunk and that shift are both multiples of the block. Nothing in
    // the API requires it (the contract runs chunk sizes 1..257), so it is tested here, once
    // per launch, and any other shape takes the general masked path below.
    const bool suffix_ok = !seg && p.row_lo == nullptr && p.chunk % pt::kBlock == 0 &&
        (p.total_length - p.q_length) % pt::kBlock == 0;
    bool any = false;
    uint32_t r = 0;
    Round c;
    FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    // Column `tid` of the NEXT round to stage (threads 0..63 only).
    pt::ColumnMeta cm;
    if (c.t <= qt_last && tid < kRoundQueries) {
      const int64_t ic = c.t * kBwdQueryRows + c.h * kRoundQueries + tid;
      cm = pt::ColumnOf(p, b, hk * p.group + c.gh, ic, seg, ic < p.q_length);
    }
    while (c.t <= qt_last) {
      Round n = c;
      NextRound(p, b, kb, ke, qt_last, &n);
      // FullTileExact(c.m, p, b, kb, ke) with its segment test hoisted: `seg` is the same
      // detail::SegmentTest(p, b), computed once per CTA.
      const bool full = !seg && FullTile(c.m, kb, ke);
      const uint32_t buf = r & 1;
      // smeta(r) replaces smeta(r-1) only after every pack thread has finished reading it.
      dvpipe::PackBarrierSync();
      if (tid < kRoundQueries) {
        smeta_lse2[tid] = cm.lse2;
        smeta_delta[tid] = cm.delta;
        smeta_lo[tid] = cm.lo;
        smeta_hi[tid] = cm.hi;
        smeta_seg[tid] = cm.segment;
        // Warps 0 and 1 hold the round's two 32-column blocks, one column per lane.
        const bool blk_ok = pt::BlockIsSuffix(cm.lo, cm.hi, cm.segment, seg);
        if (lane == 0) smeta_blk[tid >> 5] = blk_ok ? 1 : 0;
      }
      dvpipe::PackBarrierSync();
      // Round r+1's gathers, one round ahead (consumed at the next round's staging).
      if (n.t <= qt_last && tid < kRoundQueries) {
        const int64_t ic = n.t * kBwdQueryRows + n.h * kRoundQueries + tid;
        cm = pt::ColumnOf(p, b, hk * p.group + n.gh, ic, seg, ic < p.q_length);
      }
      MbarWait(bar0 + 8 * (kBarS + buf), (r >> 1) & 1);
      TmemFenceAfterSync();
      // The fused leaf's per-element line on this thread's key row and 32 query columns, in
      // two phases. S^T(r) is committed before dP^T(r), so P^T = exp2(S^T scale - lse2) and
      // its mask are computed while the tensor pipe still runs dP^T(r); only
      // dS^T = P^T (dP^T - delta), its convert and the store wait for dP^T. Per element the
      // expressions, operands and mask are those of MakePtBlock{FullSoA,Suffix,SoA}
      // (ProbabilityX2 / ScoreGradientX2 with the same ok bits), so dK is bit-identical.
      const int c0 = ch * pt::kBlock;
      float s[pt::kBlock];
      TmemLoad32(lane_base | (tbase + Pl::kSCol + buf * Pl::kBufCols + c0), s);
      uint32_t mask;
      if (full) {
        mask = ~0u;
      } else if ((suffix_ok && smeta_lo[c0] <= smeta_hi[c0]) || smeta_blk[ch] != 0) {
        // The visible columns of this key row are a suffix of the 32-column block (lo
        // constant, hi slope 1, one segment - see pt::SuffixMask, pt::BlockIsSuffix): for
        // the whole launch by the analytic guard, or for this block by its staged metadata.
        // Two scalars and a shift in place of the per-column compares. The analytic guard
        // also needs the block's first column live: RowBounds encodes an empty row as
        // {INT_MAX, -1}, which a strict-past read gives its first query (position 0), and
        // a suffix built from that column would drop the whole block; such a block takes
        // the per-block test, which it fails, and the general mask.
        mask = pt::BlockSuffixMask(j, smeta_hi[c0], smeta_lo[c0], smeta_seg[c0], ks, seg,
                                   pt::kBlock);
      } else {
        mask = 0u;
#pragma unroll
        for (int x = 0; x < pt::kBlock; ++x) {
          if (pt::VisibleSoA(smeta_lo[c0 + x], smeta_hi[c0 + x], smeta_seg[c0 + x], j, ks, seg)) {
            mask |= 1u << x;
          }
        }
      }
      TmemWaitLd();
#pragma unroll
      for (int t = 0; t < pt::kBlock / 2; ++t) {
        const int x = 2 * t;
        const float2 pp = pt::ProbabilityX2(make_float2(s[x], s[x + 1]),
                                            make_float2(smeta_lse2[c0 + x], smeta_lse2[c0 + x + 1]),
                                            p.scale_log2, (mask >> x) & 1u, (mask >> (x + 1)) & 1u);
        s[x] = pp.x;
        s[x + 1] = pp.y;
      }
      MbarWait(bar0 + 8 * (kBarDp + buf), (r >> 1) & 1);
      TmemFenceAfterSync();
      float dp[pt::kBlock];
      TmemLoad32(lane_base | (tbase + Pl::kdPCol + buf * Pl::kBufCols + c0), dp);
      TmemWaitLd();
      uint32_t dsv[pt::kBlock / 2];
#pragma unroll
      for (int t = 0; t < pt::kBlock / 2; ++t) {
        const int x = 2 * t;
        const float2 ds = pt::ScoreGradientX2(make_float2(s[x], s[x + 1]),
                                              make_float2(dp[x], dp[x + 1]),
                                              make_float2(smeta_delta[c0 + x], smeta_delta[c0 + x + 1]),
                                              (mask >> x) & 1u, (mask >> (x + 1)) & 1u);
        dsv[t] = pt::Pack2Fast<E>::Of(ds.x, ds.y);
      }
      // In place over this warp's own (already loaded) dP^T columns.
      TmemStore16(lane_base | (tbase + Pl::kdPCol + buf * Pl::kBufCols + c0), dsv);
      TmemWaitSt();
      TmemFenceBeforeSync();
      dvpipe::MbarArrive(bar0 + 8 * (kBarPack + buf));
      any = true;
      ++r;
      c = n;
    }

    // ----------------------------------------------------------------- epilogue
    // The fused leaf's dK sweep (kWrite 1): lane L of warp group ch holds key row 32 (w & 3) + L;
    // the two groups drain complementary column halves of the same rows. A tile with no round
    // stores exact zeros.
    if (any) {
      MbarWait(bar0 + 8 * (kBarDk + ((r - 1) & 1)), ((r - 1) >> 1) & 1);
      TmemFenceAfterSync();
    }
    // This leaf writes dK only, whose accumulator holds exactly kD TMEM columns
    // (kdKCol..kdKCol+kD-1): at kD != kDv (the unequal buckets) using max(kD, kDv) here
    // read columns past the accumulator (still inside the kSCol/kdPCol windows' allocation,
    // so no fault, but wasted TMEM traffic and iterations StoreKeyGrad32's width bound
    // below discarded anyway) instead of the dK width alone.
    const int cols = kD;
    const int64_t width = p.qk_dim;
    const float mul = p.scale;
    void* row_out = nullptr;
    if (j < p.prev_length) {
      row_out = KeyGradRow<E, kFp32Keys>(p.dprev_k, (b * p.prev_length + j) * p.kv_heads + hk, width);
    } else if (j >= p.prev_length && j < nt_total) {
      const int64_t lk = j - p.prev_length;
      row_out = KeyGradRow<E, kFp32Keys>(p.dk, (b * p.k_length + lk) * p.kv_heads + hk, width);
    }
    const int col_begin = ch * (cols / 2);
    for (int cc = col_begin; cc < col_begin + cols / 2; cc += 32) {
      float v[32];
      if (any) {
        TmemLoad32(lane_base | (tbase + Pl::kdKCol + cc), v);
        TmemWaitLd();
      } else {
#pragma unroll
        for (int x = 0; x < 32; ++x) v[x] = 0.f;
      }
      if (row_out != nullptr) StoreKeyGrad32<E, kFp32Keys>(row_out, cc, width, v, mul);
    }
  }

  // ------------------------------------------------------------------ tile exit
  // All 320 threads: the pack warps after the epilogue (which waited for the last dK), the
  // producer after its last issue, the MMA warp after its last commit.
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
}

}  // namespace dkpipe

// Preflight of the pipelined dK-only leaf (declared in umma_bwd_dkdv.cuh, instantiated for BF16
// and FP16 at every (D, Dv) the pair serves: (256,256) in
// instantiations/sca_umma_bwd_dkdv_d256_v256.cu, the four unequal buckets in
// instantiations/sca_umma_bwd_dkdv_pair_d*.cu): the fused leaf's refusal set keyed on the dK
// output (Accepted with kWrite 1 at the 128-row tile), the (kD, kDv) bucket and real
// dimensions, the element, device 10.3, and this plan's dynamic SMEM. `false` means nothing was
// written; after it every driver error raises, exactly like UmmaBackwardDkdvCanLaunch.
template <typename E, int kD, int kDv>
bool CanLaunchUmmaBackwardDkPipe(const UmmaLaunchParams& lp) {
  using Pl = dkpipe::Plan<kD, kDv>;
  const Params& p = lp.params;
  if (!Accepted(lp, 1, dkpipe::kKeyRows)) return false;
  if (QkBucket(p.qk_dim) != kD || ValueBucket(p.value_dim) != kDv) return false;
  if (p.qk_dim > kD || p.value_dim > kDv) return false;
  if (p.element != BwdTok<E>::kElement) return false;
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd dK pipe: cudaGetDevice", q);
  int major = 0, minor = 0;
  q = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd dK pipe: capability major", q);
  q = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd dK pipe: capability minor", q);
  if (major != 10 || minor != 3) return false;
  int smem_cap = 0;
  q = cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (q != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dK pipe: shared memory capacity", q);
  }
  if (static_cast<int>(Pl::kSmemBytes) > smem_cap) return false;
  return true;
}

// Launch entry of the pipelined dK-only leaf. Writes dK and dPrevK only (every row of every key
// tile, exact zeros for a tile no query sees); dV and dPrevV are left untouched.
template <typename E, int kD, int kDv>
bool LaunchUmmaBackwardDkPipe(const UmmaLaunchParams& lp, cudaStream_t stream) {
  using Pl = dkpipe::Plan<kD, kDv>;
  const Params& p = lp.params;
  if (!CanLaunchUmmaBackwardDkPipe<E, kD, kDv>(lp)) return false;
  const auto kernel = p.key_grads_fp32 ? dkpipe::BackwardDkPipeKernel<E, kD, kDv, true>
                                       : dkpipe::BackwardDkPipeKernel<E, kD, kDv, false>;
  const cudaError_t attr =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dK pipe: cudaFuncSetAttribute", attr);
  }
  const unsigned gx =
      static_cast<unsigned>((p.total_length + dkpipe::kKeyRows - 1) / dkpipe::kKeyRows);
  if (gx == 0) return true;  // no work (see LaunchUmmaBackwardDkdv)
  const dim3 grid(gx, static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
  kernel<<<grid, dkpipe::kThreads, Pl::kSmemBytes, stream>>>(lp);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dK pipe: kernel launch", launched);
  }
  return true;
}

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
