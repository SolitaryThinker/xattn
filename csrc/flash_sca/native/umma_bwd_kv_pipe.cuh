// Native Blackwell (sm_103a) SCA backward: the FUSED dK/dV M128 key leaf at (128,128),
// warp-specialized and pipelined across 64-query rounds ("kv_pipe"). The in-situ key launch
// (umma_bwd_dkdv.cuh, DecideInSituKeyPair / LaunchUmmaBackwardInSituDkdv) runs it at level 0,
// BF16 and FP16, for every call its preflight accepts while XATTN_SCA_BWD_KEY_ARM is unset or
// empty; every call it refuses, every call at levels 1-4 and every other value of that variable
// run the two-stage fused M128 leaf (or its M64 fallback) unchanged.
//
// It computes what that fused leaf - BackwardDkdvKernel<E, 128, 128, 3, 128, 4> - computes: dK,
// dV, dPrevK and dPrevV, every row of every key tile, one writer per row, nothing atomic, no dq.
// One CTA owns one (batch, KV head, 128-key tile). The 128-query tiles are the ones the fused
// leaf's round loop visits for this key range (query tiles ascending inside the envelope, MaySee
// skipping, heads ascending inside a tile), and each (tile, head) is processed as TWO consecutive
// 64-query rounds (h = 0, 1), both always. Round r:
//   S^T  = K Q(r)^T     tcgen05.mma SS, M128 N64, K = D  (8 k-steps), S^T buffer r & 1
//   dP^T = V dO(r)^T    tcgen05.mma SS, M128 N64, K = Dv (8 k-steps), dP^T buffer r & 1
//                       one commit for both: bar_s[r & 1]
//   P^T  = exp2(S^T scale_log2 - lse2), dS^T = P^T (dP^T - delta), masked: the fused leaf's
//          per-element values (pt::MakePtBlock / MakePtBlockFull, 32-column blocks, the same
//          column metadata), packed to 16 bits round-to-nearest-even IN PLACE over the S^T and
//          dP^T buffers of the round
//   dV  += P^T(r) dO(r) tcgen05.mma TS, M128 N = Dv (all slabs in one instruction, the slab hop
//                       in the descriptor's LBO), K = 16 per step (4 steps)
//   dK  += dS^T(r) Q(r) tcgen05.mma TS, M128 N = D, 4 steps
//                       one commit for both: bar_acc[r % kStages]
// The accumulators therefore receive, element by element, the fused leaf's K = 16 query steps in
// its order (tile, head, then the eight 16-query steps of the tile, now 4 + 4) with bitwise the
// same A (P^T / dS^T) and B (dO / Q) operands; the wide instruction (one N = D / Dv issue per
// step instead of one N = 64 issue per slab, as the fused leaf's kWide form) only regroups
// columns, so dK / dV / dPrevK / dPrevV are BIT-EQUAL to the fused leaf's (digests identical
// to the per-slab form at six objective rows). The scores are an N = 64 split of the fused leaf's N = 128 SS MMAs over the same K
// order (the regrouping umma_bwd_dk_pipe.cuh uses, bit-identical as well).
//
// Roles, 352 threads (__launch_bounds__(352, 1)), as in umma_bwd_dk_pipe.cuh plus warp 10:
//   warps 0..7  pack. Warps w and w + 4 share the TMEM lane group 32 (w & 3) .. +31 (key rows
//               32 (w & 3) + lane) and split the round's 64 query columns: warp w packs columns
//               0..31, warp w + 4 columns 32..63. Each warp reads 32 S^T and 32 dP^T columns of its
//               own rows and writes its 16 packed P^T words over the first 16 of ITS OWN 32 S^T
//               columns and its 16 packed dS^T words over the first 16 of ITS OWN 32 dP^T
//               columns, after its own loads completed (tcgen05.wait::ld): no warp writes TMEM
//               another warp still reads, so the pack needs no WAR barrier. The packed operand of
//               query step kl (16 queries) of buffer b is at window + 64 b + 32 (kl >> 1) +
//               8 (kl & 1) (PackedCol). Threads 0..63 stage the round's column table in SMEM before
//               one named barrier (bar.sync 1, 256), BEFORE the wait for S^T(r), and the global
//               gathers for round r + 1 are issued one round ahead. After the last round, the
//               epilogue: warps 0..3 drain dK (TMEM cols 0..127) and warps 4..7 dV (cols
//               128..255) of the same rows, once the last dV/dK commit completed; dK is stored
//               with the scale folded, dV unscaled, with the fused leaf's row rules (dPrevK /
//               dPrevV rows below prev_length, dK / dV rows after it, nothing past the key axis,
//               exact zeros for a tile no query sees).
//               Where no metadata warp is taken (Plan::kGroups, the capacity predicate
//               kMetaWarps == 0), the pack is grouped instead: round r is packed whole by warps
//               4g .. 4g + 3, g = r & 1 (both 32-column blocks of each key row, in turn, to the same
//               packed columns), staged by that group's threads 0..63 into column-table slot r % 4
//               before a 128-thread group barrier (bar.sync 1 + g), and bar_pack[r & 1] counts 128
//               arrivals. A missed arrival count would hang, not slow: bar_pack's one MbarInit
//               (the prologue loop) takes its count from Plan::kGroups.
//   warp 8      TMA producer, ONE thread: Q(r) and dO(r) into stage r % kStages (one expect_tx,
//               32 KB) once that stage's last readers - dV(r - kStages) and dK(r - kStages) -
//               have completed (bar_acc).
//   warp 9      score issuer: S^T/dP^T(0); then for each r: S^T/dP^T(r + 1) into buffers
//               (r + 1) & 1 once Q/dO(r + 1) landed and dV/dK(r - 1) - the last readers of those
//               TMEM buffers - completed (bar_acc, committed by warp 10). It never waits on the
//               pack, so the next round's scores are issued while pack(r) runs.
//   warp 10     accumulator issuer: for each r, dV(r) then dK(r) once pack(r) is done; commits
//               bar_acc. Each accumulator has this one issuer in round order, so dK / dV keep the
//               one-issuer bits.
//   warps 11, 12 (only where Plan::kMetaWarps is 2, see kMetaMaxDims): the column-metadata
//               warps. They gather each round's column table into a ring of four slots ahead of
//               the pack and publish it by mbarrier; the pack warps then neither gather nor stage
//               it and take no named barrier per round.
// The pack warps never use __syncthreads (it would deadlock against warps 8..12); the CTA-wide
// barriers left are the prologue's and the tile exit's, which all threads (352, or 416 with
// the metadata warps) reach once.
//
// The pack forms the fused leaf's values from the same operands, bit for bit, in two forms that
// cost fewer instructions:
//   - each 16-bit pair is packed by ONE cvt.rn.{bf16,f16}x2.f32 (PackPair) instead of two scalar
//     cvt.rn.{bf16,f16}.f32 and a byte permute: the same round-to-nearest-even conversion of the
//     same two floats into the same halves;
//   - the round's column table (lse2, delta, first and last visible key, query segment) is staged
//     as structure-of-arrays in two round-parity buffers, one staging barrier per round, read 16
//     bytes at a time, and a masked block's visibility is evaluated branch-free in 32 bits
//     (lo <= j <= hi and, with segments, key_segment == q_segment: the fused leaf's predicate,
//     exact while the key axis fits in int32, which the preflight checks).
//
// TMEM (512 columns allocated): dK FP32 cols 0..127, dV FP32 cols 128..255 (M128: lane = key row);
// S^T buffer b at 256 + 64 b (256..319, 320..383) with its packed P^T in place; dP^T buffer b at
// 384 + 64 b (384..447, 448..511) with its packed dS^T in place.
// SMEM (dynamic, operand bases 1024-aligned after the prologue's rebase):
//   stage st (st < kStages) at 32,768 st: Q(r) 64 x 128 16-bit (2 slabs of 8 KB, one 64-row TMA
//   box each), dO(r) at +16,384 (2 slabs); then K 128 x 128 (2 slabs of 16 KB), V the same; then
//   16 mbarrier slots, the TMEM slot, the envelope words and the column table (two round-parity
//   structure-of-arrays buffers, 3,072 B). With 1,024 B of alignment slack, three stages take
//   168,960 B (static_assert below). No static SMEM. The key segment of a row stays in a register.
//
// Barrier phases. Round r uses TMEM buffer r & 1 and SMEM stage r % S (S = kStages). bar_s[b] and
// bar_pack[b] complete once per round of their buffer (parity (r >> 1) & 1); bar_ld[st] and
// bar_acc[st] once per round of their stage (parity (r / S) & 1). Each parity wait below sees the
// previous phase of its barrier completed before it starts, and the next phase cannot complete
// before it returns:
//   prod waits bar_acc[r%S] (dV/dK(r-S)) before Q/dO(r): dV/dK(r-2S) was observed before Q/dO(r-S);
//        dV/dK(r) needs Q/dO(r), issued after this wait. The expect_tx of round r opens bar_ld's
//        phase only after round r-S was consumed, so no barrier holds two pending expect_tx.
//   MMA  waits bar_ld[(r+1)%S] (Q/dO(r+1)): round r+1-S observed before S(r+1-S); round r+1+S
//        needs dV/dK(r+1), which it issues later.
//   MMA  waits bar_acc[(r-1)%S] (dV/dK(r-1)): dV/dK(r-1-S) observed at iteration r-S; dV/dK(r-1+S)
//        is issued by itself later.
//   MMA  waits bar_pack[r&1] (pack(r)): pack(r-2) observed before dV/dK(r-2); pack(r+2) needs
//        S(r+2), which it issues only after this wait.
//   pack waits bar_s[r&1] (S(r)): S(r-2) observed by pack(r-2); S(r+2) is issued after the MMA
//        warp observed pack(r), i.e. every pack thread past this wait.
//   pack (epilogue) waits bar_acc of the last round L: dV/dK(L-S) was observed by the MMA warp
//        before it issued S(L-S+2) (L-S+2 <= L, which pack(L) needed), and nothing is committed to
//        that barrier after dV/dK(L).
//   column table: the buffer of round r (parity r & 1) is rewritten at round r by threads 0..63
//        before the round's one staging barrier; its previous readers (pack(r-2)) had all finished
//        before any thread passed the staging barrier of round r-1, which every writer passed
//        before this write.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/bwd_pipe_host.h"
#include "flash_sca/native/umma_bwd_dkdv.cuh"
#include "flash_sca/native/umma_bwd_dv_pipe.cuh"  // dvpipe::MbarArrive/PackBarrierSync
#include "flash_sca/native/umma_bwd_dk_pipe.cuh"  // dkpipe::TmemStore16, Round, FirstRound, NextRound

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace kvpipe {

inline constexpr int kPackWarps = 8;
inline constexpr int kPackThreads = 32 * kPackWarps;    // 256
inline constexpr int kProducerWarp = kPackWarps;        // warp 8
inline constexpr int kMmaWarp = kPackWarps + 1;         // warp 9: S^T / dP^T issuer
inline constexpr int kAccWarp = kPackWarps + 2;         // warp 10: dV / dK issuer
inline constexpr int kBaseWarps = kPackWarps + 3;       // 11: pack, producer, two issuers
inline constexpr int kMetaWarp0 = kBaseWarps;           // warp 11: first column-metadata warp
// The column-metadata warps (warps 11, 12). Warp 11 + m gathers the column table of every
// round r with r % 2 == m into ring slot r % kMetaRingSlots, publishes it on bar_meta_full[slot]
// (32 arrivals, its lanes) and reuses the slot once the eight pack warps released its previous
// round on bar_meta_empty[slot] (one arrival per pack warp). Round 0's table is written like every
// other round: the pack waits bar_meta_full[0], phase 0, before its first read, so no slot is read
// before it was written. The pack warps then wait on mbarriers only: no warp gathers, no CTA-wide
// staging barrier per round.
//
// Where they are taken (the capacity rule, Plan::kMetaWarps): a round's gather is a fixed cost,
// 64 columns of global loads (~1,400 cycles of latency) whatever the bucket,
// paid by the two pack warps that stage it while the other six wait at the staging barrier; the
// round's tensor work grows with D + Dv. Where D + Dv <= kMetaMaxDims the round's MMA chain is too
// short to hide that straggler and the two warps take ~10% off the whole backward
// (the three D = 64 rows); at (128,128) the chain hides it (the ceiling arm is 2-3% of the row) and
// the two extra warps cost more than they save (+2-5%), so the leaf keeps the
// per-round staging path there. One warp is not enough: a round's gather takes longer than a
// D = 64 round (1 warp slower than none).
#ifndef XATTN_KV_META_MAX_DIMS
#define XATTN_KV_META_MAX_DIMS 128
#endif
inline constexpr int kMetaMaxDims = XATTN_KV_META_MAX_DIMS;
inline constexpr int kMetaWarpCount = 2;
inline constexpr int kMetaRingSlots = 4;
inline constexpr int kKeyRows = 128;
inline constexpr int kRoundQueries = 64;  // queries per round: half of a 128-query tile
inline constexpr int kSteps = kRoundQueries / 16;  // 16-query k-steps of the TS MMAs per round
inline constexpr int kRingStages = 3;     // Q/dO SMEM stages of the launched instantiation
static_assert(kPackThreads == dvpipe::kPackThreads,
              "the pack barrier is dvpipe's named barrier over 256 threads");
static_assert(kRoundQueries == dkpipe::kRoundQueries && dkpipe::kHalves == 2,
              "the round sequence is dk_pipe's: two 64-query rounds per 128-query tile");
static_assert(kRoundQueries == kTmaRows, "one 64-row TMA box per slab of a round's Q / dO");
static_assert(kRoundQueries == 2 * pt::kBlock, "two pack warps split a round's columns");
static_assert(kRoundQueries <= kPackThreads, "one column-table writer per column");

// mbarrier slots (8 B each) after the operand image.
template <int kStages, int kMetaSlots>
struct Bars {
  static constexpr int kKv = 0;                // K and V landed (once per CTA)
  static constexpr int kLd = 1;                // + st: Q(r) and dO(r) landed in stage st
  static constexpr int kS = kLd + kStages;     // + (r & 1): S^T(r) and dP^T(r) complete
  static constexpr int kPack = kS + 2;         // + (r & 1): pack(r) done (256 arrivals)
  static constexpr int kAcc = kPack + 2;       // + st: dV(r) and dK(r) complete
  static constexpr int kMetaFull = kAcc + kStages;       // + slot: column table written
  static constexpr int kMetaEmpty = kMetaFull + kMetaSlots;  // + slot: its pack readers done
  static constexpr int kCount = kMetaEmpty + kMetaSlots;
  static_assert(kCount <= 32, "32 barrier slots");
};

// The round's column table: kMetaSlots buffers of 64 columns, structure of arrays (two round-
// parity buffers on the staging path, the metadata warps' ring of four otherwise).
template <int kMetaSlots>
struct ColumnTable {
  static constexpr int kL2 = 0;                    // float   [slots][64]  lse2
  static constexpr int kDl = 256 * kMetaSlots;     // float   [slots][64]  delta
  static constexpr int kLo = 512 * kMetaSlots;     // int32_t [slots][64]  first visible key
  static constexpr int kHi = 768 * kMetaSlots;     // int32_t [slots][64]  last visible key
  static constexpr int kSg = 1024 * kMetaSlots;    // int64_t [slots][64]  q_segment
  static constexpr int kBytes = 1536 * kMetaSlots;
  // The one writer form of a column (every staging site uses it): lse2 staged as +inf where it
  // is not finite, the other fields as pt::ColumnOf gave them.
  __device__ __forceinline__ static void Store(unsigned char* tb, uint32_t slot, int col,
                                               const pt::ColumnMeta& cm) {
    const int o = static_cast<int>(slot) * kRoundQueries + col;
    reinterpret_cast<float*>(tb + kL2)[o] = isfinite(cm.lse2) ? cm.lse2 : __int_as_float(0x7f800000);
    reinterpret_cast<float*>(tb + kDl)[o] = cm.delta;
    reinterpret_cast<int32_t*>(tb + kLo)[o] = cm.lo;
    reinterpret_cast<int32_t*>(tb + kHi)[o] = cm.hi;
    reinterpret_cast<int64_t*>(tb + kSg)[o] = cm.segment;
  }
};

// The plan is templated on the head-dimension buckets: every (D, Dv) with D, Dv multiples of 64
// and D + Dv <= 256 (the dK and dV accumulators share TMEM columns 0..255 below the score
// buffers): (128,128), (64,64), (64,128) and (128,64). Every quantity below follows from the
// slab counts D / 64 and Dv / 64; at (128,128) it is the original plan, byte for byte.
template <int kStages, int kD_ = 128, int kDv_ = 128>
struct Plan {
  static_assert(kStages >= 2 && kStages <= 4, "two to four Q/dO stages");
  static_assert(kD_ % 64 == 0 && kDv_ % 64 == 0 && kD_ >= 64 && kDv_ >= 64,
                "one 64-column slab per MMA step");
  static_assert(kD_ + kDv_ <= 256, "dK and dV accumulators share TMEM columns 0..255");
  static constexpr int kD = kD_;
  static constexpr int kDv = kDv_;
  static constexpr int kSlabsK = kD / 64;                         // Q / K slabs
  static constexpr int kSlabsV = kDv / 64;                        // dO / V slabs
  static constexpr int kKeySlabBytes = kKeyRows * 128;            // 16,384
  static constexpr int kRoundSlabBytes = kRoundQueries * 128;     // 8,192
  static constexpr int kKBytes = kSlabsK * kKeySlabBytes;         // 32,768 at D = 128
  static constexpr int kVBytes = kSlabsV * kKeySlabBytes;         // 32,768 at Dv = 128
  static constexpr int kQBytes = kSlabsK * kRoundSlabBytes;       // 16,384 per stage at D = 128
  static constexpr int kGBytes = kSlabsV * kRoundSlabBytes;       // 16,384 per stage at Dv = 128
  static constexpr int kStageBytes = kQBytes + kGBytes;           // 32,768
  static constexpr int kKOff = kStages * kStageBytes;
  static constexpr int kVOff = kKOff + kKBytes;
  static constexpr int kBarOff = kVOff + kVBytes;
  static constexpr int kSlotOff = kBarOff + 256;                  // 32 barrier slots
  static constexpr int kRedOff = kSlotOff + 16;
  static constexpr int kMetaOff = kRedOff + 32;
  // The capacity rule (see kMetaMaxDims): two metadata warps and a ring of four tables where
  // the round's tensor work is too short to hide the gather, else the per-round staging path.
  static constexpr int kMetaWarps = (kD + kDv <= kMetaMaxDims) ? kMetaWarpCount : 0;
  // Group-per-round pack: wherever no metadata warp is taken (the capacity predicate above,
  // kMetaWarps == 0), round r is packed by ONE 4-warp group g = r & 1 - warps 4g .. 4g + 3, which
  // between them reach all 128 TMEM lanes - over all 64 columns of their key rows, with a
  // group-local staging barrier (bar.sync 1 + g, 128 threads) and four column-table slots
  // (r % 4). The eight pack warps never rendezvous: each group waits only on its own rounds, so
  // the two groups' packs overlap instead of meeting at a CTA-wide barrier every round (dev A/B:
  // the D+Dv > 128 rows 3-6% faster, bit-identical). Where the metadata warps are taken
  // the pack keeps their ring and its 8-warp form.
  static constexpr bool kGroups = kMetaWarps == 0;
  static constexpr int kMetaSlots = (kMetaWarps > 0 || kGroups) ? kMetaRingSlots : 2;
  static constexpr int kThreads = 32 * (kBaseWarps + kMetaWarps);  // 352 or 416
  using Table = ColumnTable<kMetaSlots>;
  static constexpr int kMetaBytes = Table::kBytes;                // 3,072 or 6,144
  static constexpr int kEnd = kMetaOff + kMetaBytes;
  static constexpr int kSmemBytes = ((kEnd + 1023) / 1024) * 1024 + 1024;
  // TMEM columns.
  static constexpr int kdKCol = 0;
  static constexpr int kdVCol = kD;
  static constexpr int kSCol = 256;   // S^T buffer b at 256 + 64 b, packed P^T in place
  static constexpr int kdPCol = 384;  // dP^T buffer b at 384 + 64 b, packed dS^T in place
  static constexpr int kBufCols = kRoundQueries;
  static constexpr int kAllocCols = 512;

  static_assert(kdKCol + kD <= kdVCol && kdVCol + kDv <= kSCol, "dK and dV below the scores");
  static_assert(kSCol + 2 * kBufCols <= kdPCol, "two S^T buffers below the dP^T buffers");
  static_assert(kdPCol + 2 * kBufCols <= kAllocCols, "two dP^T buffers fit the allocation");
  static_assert(kKeySlabBytes % 1024 == 0 && kRoundSlabBytes % 1024 == 0 &&
                    kStageBytes % 1024 == 0 && kQBytes % 1024 == 0,
                "descriptor tile bases must stay 1024-aligned");
  static_assert(kKOff % 1024 == 0 && kVOff % 1024 == 0, "descriptor tile bases 1024-aligned");
  static_assert(kMetaOff % 16 == 0, "the column table is read 16 bytes at a time");
  // Totals: kStages x (Q + dO round slabs) + K + V + 304 B of barriers, TMEM slot and envelope
  // + the 1,536 B per slot column table, rounded to 1 KB, + 1 KB rebase slack.
  static_assert(kStages != 3 || kD != 128 || kDv != 128 ||
                    kSmemBytes == (kMetaSlots == 4 ? 172032 : 168960),
                "SMEM total at three stages, (128,128)");
  static_assert(kStages != 4 || kD != 64 || kDv != 64 ||
                    kSmemBytes == (kMetaSlots == 4 ? 106496 : 103424),
                "SMEM total at four stages, (64,64)");
  static_assert(kStages != 4 || kD + kDv != 192 ||
                    kSmemBytes == (kMetaSlots == 4 ? 155648 : 152576),
                "SMEM total at four stages, (64,128) and (128,64)");
  static_assert(kSmemBytes <= 232448, "the sm_103a opt-in dynamic SMEM ceiling");
};

// TMEM column of the packed operand (P^T in an S^T window, dS^T in a dP^T window) of round
// buffer `buf`, query step kl (16 queries): the pack warp of column half kl >> 1 writes its 16
// words at the start of its own 32 columns.
__host__ __device__ constexpr uint32_t PackedCol(int window, uint32_t buf, int kl) {
  return static_cast<uint32_t>(window + 64 * static_cast<int>(buf) + 32 * (kl >> 1) +
                               8 * (kl & 1));
}
static_assert(PackedCol(256, 0, 0) == 256 && PackedCol(256, 1, 3) == 256 + 64 + 40 &&
                  PackedCol(384, 1, 2) == 384 + 64 + 32,
              "packed columns: window + 64 per buffer + 32 per column half + 8 per step");

// ------------------------------------------------------------------ pack
// One 16-bit pair, element `lo` in the low half: cvt.rn.{bf16,f16}x2.f32 d, a, b puts a's
// conversion in the upper half and b's in the lower, both round-to-nearest-even with subnormals
// kept - the conversion pt::Pack2 performs with two scalar cvt.rn.{bf16,f16}.f32.
template <class E>
__device__ __forceinline__ uint32_t PackPair(float lo, float hi);
template <>
__device__ __forceinline__ uint32_t PackPair<__nv_bfloat16>(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
template <>
__device__ __forceinline__ uint32_t PackPair<__half>(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// The pack's exponent. `exp2f` compiles to MUFU.EX2 plus a per-element denormal guard;
// `ex2.approx.ftz.f32` is that instruction alone. The two differ only where P is FP32-SUBNORMAL
// (P < 2^-126), which `ftz` flushes to zero instead of up-converting: FP16's smallest subnormal is
// 2^-24, so for FP16 the convert stores zero either way, but BF16 shares FP32's exponent range
// (subnormals down to 2^-133), so a subnormal P would pack to a nonzero BF16 P^T here and to zero
// there - and dS = P*(dP - delta) is formed in FP32, where |dP - delta| > 1 can lift a subnormal P
// to a normal value that `ftz` loses. That is a value change in the subnormal tail, judged by the
// frozen contract's FP64 tolerances (which the contract runs passed with this instruction),
// not a bit-identical rewrite. Compile-time, so one build's A/B can put the two arms side by side;
// no gate check sets it to 0.
#ifndef XATTN_SCA_KV_EX2
#define XATTN_SCA_KV_EX2 1
#endif
__device__ __forceinline__ float KvExp2(float x) {
#if XATTN_SCA_KV_EX2
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return exp2f(x);
#endif
}

// The column table stages lse2 as +inf where it is not finite, so the pack's exponential
// gives P = 0 (ex2.approx.ftz(-inf) = +0) exactly where a per-element isfinite select used to
// write 0 - and that select, one FSEL plus one FSETP per element, is gone. delta is 0 on such
// columns, so dS = P*(dP - delta) stays 0. An arm-for-arm A/B measured it.

// A block proved fully visible with no segment test (the caller's `full`): the fused leaf's
// Probability(.., true) and ScoreGradient(.., true) expressions, the table read 4 columns at a time.
template <class E>
__device__ __forceinline__ void PackBlockFull(const float (&s)[pt::kBlock],
                                              const float (&dp)[pt::kBlock], const float* l2,
                                              const float* dl, float scale_log2, uint32_t* out_p,
                                              uint32_t* out_ds) {
#pragma unroll
  for (int q = 0; q < pt::kBlock / 4; ++q) {
    const float4 L = *reinterpret_cast<const float4*>(l2 + 4 * q);
    const float4 D = *reinterpret_cast<const float4*>(dl + 4 * q);
    const float lv[4] = {L.x, L.y, L.z, L.w};
    const float dv[4] = {D.x, D.y, D.z, D.w};
    float pe[4], ge[4];
    // Two lanes per instruction (fma/sub/mul .rn.f32x2: per-lane IEEE round-to-nearest-even,
    // the same values as the scalar fmaf / - / * they replace).
    const float2 sc = make_float2(scale_log2, scale_log2);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 x = pt::FmaX2(make_float2(s[4 * q + 2 * h], s[4 * q + 2 * h + 1]), sc,
                                 make_float2(-lv[2 * h], -lv[2 * h + 1]));
      const float2 pv = make_float2(KvExp2(x.x), KvExp2(x.y));
      const float2 gv = pt::MulX2(pv, pt::SubX2(make_float2(dp[4 * q + 2 * h], dp[4 * q + 2 * h + 1]),
                                                make_float2(dv[2 * h], dv[2 * h + 1])));
      pe[2 * h] = pv.x;
      pe[2 * h + 1] = pv.y;
      ge[2 * h] = gv.x;
      ge[2 * h + 1] = gv.y;
    }
    out_p[2 * q] = PackPair<E>(pe[0], pe[1]);
    out_p[2 * q + 1] = PackPair<E>(pe[2], pe[3]);
    out_ds[2 * q] = PackPair<E>(ge[0], ge[1]);
    out_ds[2 * q + 1] = PackPair<E>(ge[2], ge[3]);
  }
}

// A masked block: pt::Visible's predicate (lo <= j <= hi, then the segment test when the row
// needs it) evaluated without branches on a 32-bit key index, then the fused leaf's Probability /
// ScoreGradient selects.
template <class E, bool kSegs>
__device__ __forceinline__ void PackBlockMasked(const float (&s)[pt::kBlock],
                                                const float (&dp)[pt::kBlock], const float* l2,
                                                const float* dl, const int32_t* lo,
                                                const int32_t* hi, const int64_t* sg, int32_t j32,
                                                int64_t key_segment, float scale_log2,
                                                uint32_t* out_p, uint32_t* out_ds) {
#pragma unroll
  for (int q = 0; q < pt::kBlock / 4; ++q) {
    const float4 L = *reinterpret_cast<const float4*>(l2 + 4 * q);
    const float4 D = *reinterpret_cast<const float4*>(dl + 4 * q);
    const int4 LO = *reinterpret_cast<const int4*>(lo + 4 * q);
    const int4 HI = *reinterpret_cast<const int4*>(hi + 4 * q);
    const float lv[4] = {L.x, L.y, L.z, L.w};
    const float dv[4] = {D.x, D.y, D.z, D.w};
    const int32_t lov[4] = {LO.x, LO.y, LO.z, LO.w};
    const int32_t hiv[4] = {HI.x, HI.y, HI.z, HI.w};
    int64_t sv[4] = {0, 0, 0, 0};
    if constexpr (kSegs) {
      const longlong2 A = *reinterpret_cast<const longlong2*>(sg + 4 * q);
      const longlong2 B = *reinterpret_cast<const longlong2*>(sg + 4 * q + 2);
      sv[0] = A.x;
      sv[1] = A.y;
      sv[2] = B.x;
      sv[3] = B.y;
    }
    float pe[4], ge[4];
    const float2 sc = make_float2(scale_log2, scale_log2);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      bool vis[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        vis[e] = (j32 >= lov[2 * h + e]) && (j32 <= hiv[2 * h + e]);
        if constexpr (kSegs) vis[e] = vis[e] && (key_segment == sv[2 * h + e]);
      }
      const float2 x = pt::FmaX2(make_float2(s[4 * q + 2 * h], s[4 * q + 2 * h + 1]), sc,
                                 make_float2(-lv[2 * h], -lv[2 * h + 1]));
      const float2 pv = make_float2(vis[0] ? KvExp2(x.x) : 0.f, vis[1] ? KvExp2(x.y) : 0.f);
      const float2 gv = pt::MulX2(pv, pt::SubX2(make_float2(dp[4 * q + 2 * h], dp[4 * q + 2 * h + 1]),
                                                make_float2(dv[2 * h], dv[2 * h + 1])));
      pe[2 * h] = pv.x;
      pe[2 * h + 1] = pv.y;
      ge[2 * h] = vis[0] ? gv.x : 0.f;
      ge[2 * h + 1] = vis[1] ? gv.y : 0.f;
    }
    out_p[2 * q] = PackPair<E>(pe[0], pe[1]);
    out_p[2 * q + 1] = PackPair<E>(pe[2], pe[3]);
    out_ds[2 * q] = PackPair<E>(ge[0], ge[1]);
    out_ds[2 * q + 1] = PackPair<E>(ge[2], ge[3]);
  }
}

// kFp32Keys: dK/dV/dPrevK/dPrevV in FP32 (Params::key_grads_fp32), a separate instantiation so
// that the 16-bit one keeps its own store code.
template <typename E, int kStages, bool kFp32Keys = false, int kDB = 128, int kDvB = 128>
__global__ void __launch_bounds__((Plan<kStages, kDB, kDvB>::kThreads), 1) BackwardKvPipeKernel(
    const __grid_constant__ UmmaLaunchParams lp) {
  using Pl = Plan<kStages, kDB, kDvB>;
  using Br = Bars<kStages, Pl::kMetaSlots>;
  using ColumnTable = typename Pl::Table;
  constexpr int kD = Pl::kD;
  constexpr int kDv = Pl::kDv;
  constexpr int kMetaWarps = Pl::kMetaWarps;
  constexpr int kMetaSlots = Pl::kMetaSlots;
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  // Same 1024-byte rebase as the fused leaf (descriptor base_offset = 0).
  unsigned char* const smem = smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);

  unsigned char* const sk = smem + Pl::kKOff;
  unsigned char* const sv = smem + Pl::kVOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Pl::kRedOff);
  const uint32_t bar_kv = bar0 + 8u * Br::kKv;

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
    for (int i = 0; i < Br::kCount; ++i) {
      int count = 1;
      if (i == Br::kPack || i == Br::kPack + 1) count = Pl::kGroups ? kPackThreads / 2 : kPackThreads;
      if (kMetaWarps > 0 && i >= Br::kMetaFull && i < Br::kMetaEmpty) count = 32;
      if (kMetaWarps > 0 && i >= Br::kMetaEmpty && i < Br::kMetaEmpty + kMetaSlots) {
        count = kPackWarps;
      }
      MbarInit(bar0 + 8u * i, count);
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
    // One thread, one cursor over the round sequence: Q(r) and dO(r) into stage r % S once that
    // stage's last readers (dV(r - S) and dK(r - S), one commit) have completed.
    if (lane == 0) {
      dkpipe::Round c;
      dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
      uint32_t r = 0;
      while (c.t <= qt_last) {
        const uint32_t st = r % kStages;
        if (r >= static_cast<uint32_t>(kStages)) {
          MbarWait(bar0 + 8u * (Br::kAcc + st), ((r - kStages) / kStages) & 1u);
        }
        unsigned char* const qs = smem + st * Pl::kStageBytes;
        unsigned char* const gs = qs + Pl::kQBytes;
        const uint32_t bl = bar0 + 8u * (Br::kLd + st);
        const int hq = static_cast<int>(hk * p.group + c.gh);
        const int i0 = static_cast<int>(c.t * kBwdQueryRows + c.h * kRoundQueries);
        MbarArriveExpectTx(bl, Pl::kStageBytes);
        for (int s = 0; s < Pl::kSlabsK; ++s) {
          TmaLoad4D(qs + s * Pl::kRoundSlabBytes, &lp.maps.q, bl, s * 64, hq, i0,
                    static_cast<int>(b));
        }
        for (int s = 0; s < Pl::kSlabsV; ++s) {
          TmaLoad4D(gs + s * Pl::kRoundSlabBytes, &lp.maps.dy, bl, s * 64, hq, i0,
                    static_cast<int>(b));
        }
        ++r;
        dkpipe::NextRound(p, b, kb, ke, qt_last, &c);
      }
    }
    __syncwarp();
  } else if (warp == kMmaWarp || warp == kAccWarp) {
    // ------------------------------------------------------------- MMA issuers
    // Two warps: warp 9 (A) issues S^T/dP^T(r+1) once Q/dO(r+1) landed and dV/dK(r-1) completed
    // (kAcc, committed by B); warp 10 (B) issues dV(r) then dK(r) once pack(r) is done and commits
    // kAcc. Each accumulator has one issuer and the one-issuer order, so dK/dV keep their bits;
    // A no longer waits on pack(r) before it can issue the next round's S^T/dP^T.
    const bool is_a = warp == kMmaWarp;
    // K/V are waited for unconditionally: a CTA with no round must not exit while its K/V load is
    // still writing shared memory.
    MbarWait(bar_kv, 0);
    dkpipe::Round c;
    dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    if (c.t <= qt_last) {
      const uint32_t s_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kRoundQueries, /*a=*/0, /*b=*/0);
      // dV / dK: ONE instruction per 16-query step covering the whole accumulator width (N = Dv
      // for dV, N = D for dK, at most 128 here), B MN-major: its 64-column slabs are
      // Pl::kRoundSlabBytes apart, which the SMEM descriptor's leading byte offset expresses (the
      // fused leaf's kWide form). Each accumulator element still receives the same K = 16 steps
      // in the same order with the same operands as the per-slab form: same bits.
      const uint32_t v_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kDv, /*a=*/0, /*b=*/1);
      const uint32_t k_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kD, /*a=*/0, /*b=*/1);
      constexpr uint32_t kSlabLbo = static_cast<uint32_t>(Pl::kRoundSlabBytes / 16);
      static_assert(Pl::kRoundSlabBytes % 16 == 0 && kSlabLbo < (1u << 14), "LBO field");
      const uint32_t s_base = SmemAddr(smem);
      // S^T(r) = K Q(r)^T into S^T buffer r & 1 (the fused leaf's k-loop, N = 64).
      auto issue_s = [&](uint32_t r) {
        const uint32_t qs = s_base + (r % kStages) * Pl::kStageBytes;
        for (int kk = 0; kk < kD / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(tbase + Pl::kSCol + (r & 1u) * Pl::kBufCols,
                MakeSmemDesc(SmemAddr(sk) + slab * Pl::kKeySlabBytes + within * 32),
                MakeSmemDesc(qs + slab * Pl::kRoundSlabBytes + within * 32), s_idesc,
                kk == 0 ? 0u : 1u);
        }
      };
      // dP^T(r) = V dO(r)^T into dP^T buffer r & 1.
      auto issue_dp = [&](uint32_t r) {
        const uint32_t gs = s_base + (r % kStages) * Pl::kStageBytes + Pl::kQBytes;
        for (int kk = 0; kk < kDv / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(tbase + Pl::kdPCol + (r & 1u) * Pl::kBufCols,
                MakeSmemDesc(SmemAddr(sv) + slab * Pl::kKeySlabBytes + within * 32),
                MakeSmemDesc(gs + slab * Pl::kRoundSlabBytes + within * 32), s_idesc,
                kk == 0 ? 0u : 1u);
        }
      };
      // dV += P^T(r) dO(r): A = P^T packed in the S^T buffer, B = dO(r) MN-major, all slabs.
      auto issue_dv = [&](uint32_t r) {
        const uint32_t gs = s_base + (r % kStages) * Pl::kStageBytes + Pl::kQBytes;
        for (int kl = 0; kl < kSteps; ++kl) {
          MmaTs(tbase + Pl::kdVCol, tbase + PackedCol(Pl::kSCol, r & 1u, kl),
                MakeSmemDesc(gs + kl * 16 * 128, kSlabLbo, 64), v_idesc,
                (r == 0 && kl == 0) ? 0u : 1u);
        }
      };
      // dK += dS^T(r) Q(r): A = dS^T packed in the dP^T buffer, B = Q(r) MN-major, all slabs.
      auto issue_dk = [&](uint32_t r) {
        const uint32_t qs = s_base + (r % kStages) * Pl::kStageBytes;
        for (int kl = 0; kl < kSteps; ++kl) {
          MmaTs(tbase + Pl::kdKCol, tbase + PackedCol(Pl::kdPCol, r & 1u, kl),
                MakeSmemDesc(qs + kl * 16 * 128, kSlabLbo, 64), k_idesc,
                (r == 0 && kl == 0) ? 0u : 1u);
        }
      };
      uint32_t r = 0;
      if (is_a) {
        MbarWait(bar0 + 8u * (Br::kLd + 0), 0u);
        TmemFenceAfterSync();
        __syncwarp();
        issue_s(0u);
        issue_dp(0u);
        MmaCommit(bar0 + 8u * (Br::kS + 0));
      }
      for (;;) {
        dkpipe::Round n = c;
        dkpipe::NextRound(p, b, kb, ke, qt_last, &n);
        const bool more = n.t <= qt_last;
        if (is_a) {
          if (!more) break;
          // S^T/dP^T(r+1) into buffers (r+1)&1: Q/dO(r+1) landed, and the last readers of those
          // TMEM buffers - dV(r-1) and dK(r-1), committed by warp B - have completed.
          const uint32_t r1 = r + 1;
          MbarWait(bar0 + 8u * (Br::kLd + r1 % kStages), (r1 / kStages) & 1u);
          if (r >= 1) {
            MbarWait(bar0 + 8u * (Br::kAcc + (r - 1) % kStages), ((r - 1) / kStages) & 1u);
          }
          TmemFenceAfterSync();
          __syncwarp();
          issue_s(r1);
          issue_dp(r1);
          MmaCommit(bar0 + 8u * (Br::kS + (r1 & 1u)));
        } else {
          // dV(r), dK(r): P^T(r) and dS^T(r) packed.
          MbarWait(bar0 + 8u * (Br::kPack + (r & 1u)), (r >> 1) & 1u);
          TmemFenceAfterSync();
          __syncwarp();
          issue_dv(r);
          issue_dk(r);
          MmaCommit(bar0 + 8u * (Br::kAcc + r % kStages));
          if (!more) break;
        }
        c = n;
        ++r;
      }
    }
  } else if (warp >= kMetaWarp0) {
    // ------------------------------------------------------------- column-metadata warps
    // Warp kMetaWarp0 + m gathers the column table of rounds r = m, m + kMetaWarps, ... into slot
    // r % kMetaSlots (lane L: columns L and L + 32), once the eight pack warps released that
    // slot's previous round (r - kMetaSlots). The values are pt::ColumnOf's and the staging
    // expressions are the per-round path's, so the table holds the same bits.
    if constexpr (kMetaWarps > 0) {
      const int mw = warp - kMetaWarp0;
      unsigned char* const tb = smem + Pl::kMetaOff;
      dkpipe::Round c;
      dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
      uint32_t r = 0;
      while (c.t <= qt_last) {
        if (static_cast<int>(r % kMetaWarps) == mw) {
          const uint32_t slot = r % kMetaSlots;
          if (r >= static_cast<uint32_t>(kMetaSlots)) {
            MbarWait(bar0 + 8u * (Br::kMetaEmpty + slot), ((r / kMetaSlots) - 1u) & 1u);
          }
          constexpr int kPer = kRoundQueries / 32;
          pt::ColumnMeta cm[kPer];
#pragma unroll
          for (int k = 0; k < kPer; ++k) {
            const int64_t ic = c.t * kBwdQueryRows + c.h * kRoundQueries + lane + 32 * k;
            cm[k] = pt::ColumnOf(p, b, hk * p.group + c.gh, ic, seg, ic < p.q_length);
          }
#pragma unroll
          for (int k = 0; k < kPer; ++k) ColumnTable::Store(tb, slot, lane + 32 * k, cm[k]);
          dvpipe::MbarArrive(bar0 + 8u * (Br::kMetaFull + slot));
        }
        ++r;
        dkpipe::NextRound(p, b, kb, ke, qt_last, &c);
      }
    }
    __syncwarp();
  } else {
    // ------------------------------------------------------------- pack warps 0..7
    const int w4 = warp & 3;
    const int ch = warp >> 2;  // column half of the round: 0 -> 0..31, 1 -> 32..63
    const uint32_t lane_base = static_cast<uint32_t>(32 * w4) << 16;
    const int64_t row_in_tile = static_cast<int64_t>(w4 * 32 + lane);
    const int64_t j = kb + row_in_tile;
    // The fused leaf's kseg entry of this row, kept in a register (the row never changes).
    const int64_t ks = (seg && j < nt_total) ? p.k_segment[b * nt_total + j] : 0;
    bool any = false;
    uint32_t r = 0;
    if constexpr (Pl::kGroups) {
      // Step 1: group g = ch packs rounds g, g + 2, ...; both 32-column blocks of its rows.
      const int g = ch;
      const int gtid = tid & 127;
      dkpipe::Round c;
      dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
      uint32_t total = c.t <= qt_last ? 1u : 0u;  // rounds of the tile (all groups)
      if (g == 1 && c.t <= qt_last) {
        dkpipe::NextRound(p, b, kb, ke, qt_last, &c);
        r = 1;
      }
      pt::ColumnMeta cm;
      if (c.t <= qt_last && gtid < kRoundQueries) {
        const int64_t ic = c.t * kBwdQueryRows + c.h * kRoundQueries + gtid;
        cm = pt::ColumnOf(p, b, hk * p.group + c.gh, ic, seg, ic < p.q_length);
      }
      while (c.t <= qt_last) {
        dkpipe::Round n = c;
        dkpipe::NextRound(p, b, kb, ke, qt_last, &n);
        dkpipe::Round n2 = n;  // this group's next round
        if (n.t <= qt_last) dkpipe::NextRound(p, b, kb, ke, qt_last, &n2);
        total = (n.t <= qt_last) ? r + 2 : r + 1;
        const bool full = !seg && FullTile(c.m, kb, ke);
        const uint32_t buf = r & 1u;  // == g
        const uint32_t tslot = r % 4u;
        // Slot r % 4 was last read at round r - 4 by this group, before every one of its threads
        // reached round r - 2's group barrier, which this writer passed.
        if (gtid < kRoundQueries) ColumnTable::Store(smem + Pl::kMetaOff, tslot, gtid, cm);
        asm volatile("bar.sync %0, %1;" ::"r"(1 + g), "r"(kPackThreads / 2) : "memory");
        if (n2.t <= qt_last && gtid < kRoundQueries) {
          const int64_t ic = n2.t * kBwdQueryRows + n2.h * kRoundQueries + gtid;
          cm = pt::ColumnOf(p, b, hk * p.group + n2.gh, ic, seg, ic < p.q_length);
        }
        MbarWait(bar0 + 8u * (Br::kS + buf), (r >> 1) & 1u);
        TmemFenceAfterSync();
#pragma unroll 1
        for (int blk = 0; blk < 2; ++blk) {
          const int c0 = blk * pt::kBlock;
          const uint32_t scol = tbase + Pl::kSCol + buf * Pl::kBufCols + c0;
          const uint32_t dcol = tbase + Pl::kdPCol + buf * Pl::kBufCols + c0;
          float s[pt::kBlock], dp[pt::kBlock];
          TmemLoad32(lane_base | scol, s);
          TmemLoad32(lane_base | dcol, dp);
          TmemWaitLd();
          uint32_t pv[pt::kBlock / 2], dsv[pt::kBlock / 2];
          const unsigned char* const tb = smem + Pl::kMetaOff;
          const int o = static_cast<int>(tslot) * kRoundQueries + c0;
          const float* const l2 = reinterpret_cast<const float*>(tb + ColumnTable::kL2) + o;
          const float* const dl = reinterpret_cast<const float*>(tb + ColumnTable::kDl) + o;
          if (full) {
            PackBlockFull<E>(s, dp, l2, dl, p.scale_log2, pv, dsv);
          } else {
            const int32_t* const lo = reinterpret_cast<const int32_t*>(tb + ColumnTable::kLo) + o;
            const int32_t* const hi = reinterpret_cast<const int32_t*>(tb + ColumnTable::kHi) + o;
            const int64_t* const sg = reinterpret_cast<const int64_t*>(tb + ColumnTable::kSg) + o;
            const int32_t j32 = static_cast<int32_t>(j);
            if (seg) {
              PackBlockMasked<E, true>(s, dp, l2, dl, lo, hi, sg, j32, ks, p.scale_log2, pv, dsv);
            } else {
              PackBlockMasked<E, false>(s, dp, l2, dl, lo, hi, sg, j32, ks, p.scale_log2, pv, dsv);
            }
          }
          dkpipe::TmemStore16(lane_base | scol, pv);
          dkpipe::TmemStore16(lane_base | dcol, dsv);
        }
        TmemWaitSt();
        TmemFenceBeforeSync();
        dvpipe::MbarArrive(bar0 + 8u * (Br::kPack + buf));
        r += 2;
        c = n2;
      }
      r = total;
      any = total > 0;
    } else {
    dkpipe::Round c;
    dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    // Staging path only: column `tid` of the NEXT round to stage (threads 0..63 only).
    pt::ColumnMeta cm;
    if constexpr (kMetaWarps == 0) {
      if (c.t <= qt_last && tid < kRoundQueries) {
        const int64_t ic = c.t * kBwdQueryRows + c.h * kRoundQueries + tid;
        cm = pt::ColumnOf(p, b, hk * p.group + c.gh, ic, seg, ic < p.q_length);
      }
    }
    while (c.t <= qt_last) {
      dkpipe::Round n = c;
      dkpipe::NextRound(p, b, kb, ke, qt_last, &n);
      // FullTileExact(c.m, p, b, kb, ke), with its segment test (detail::SegmentTest) read once.
      const bool full = !seg && FullTile(c.m, kb, ke);
      const uint32_t buf = r & 1u;
      const uint32_t tslot = r % kMetaSlots;  // = buf on the staging path
      if constexpr (kMetaWarps > 0) {
        // The metadata warps published slot tslot for round r (round 0 included).
        MbarWait(bar0 + 8u * (Br::kMetaFull + tslot), (r / kMetaSlots) & 1u);
      } else {
      // Round parity buffer `buf`: its last readers (pack(r-2)) finished before the barrier of
      // round r-1, which this writer passed (see the header). One barrier publishes it.
      if (tid < kRoundQueries) ColumnTable::Store(smem + Pl::kMetaOff, buf, tid, cm);
      dvpipe::PackBarrierSync();
      // Round r+1's gathers, one round ahead (consumed at the next round's staging).
      if (n.t <= qt_last && tid < kRoundQueries) {
        const int64_t ic = n.t * kBwdQueryRows + n.h * kRoundQueries + tid;
        cm = pt::ColumnOf(p, b, hk * p.group + n.gh, ic, seg, ic < p.q_length);
      }
      }
      MbarWait(bar0 + 8u * (Br::kS + buf), (r >> 1) & 1u);
      TmemFenceAfterSync();
      // The fused leaf's per-element values on this thread's key row and 32 query columns.
      const int c0 = ch * pt::kBlock;
      const uint32_t scol = tbase + Pl::kSCol + buf * Pl::kBufCols + c0;
      const uint32_t dcol = tbase + Pl::kdPCol + buf * Pl::kBufCols + c0;
      float s[pt::kBlock], dp[pt::kBlock];
      TmemLoad32(lane_base | scol, s);
      TmemLoad32(lane_base | dcol, dp);
      TmemWaitLd();
      uint32_t pv[pt::kBlock / 2], dsv[pt::kBlock / 2];
      {
        const unsigned char* const tb = smem + Pl::kMetaOff;
        const int o = static_cast<int>(tslot) * kRoundQueries + c0;
        const float* const l2 = reinterpret_cast<const float*>(tb + ColumnTable::kL2) + o;
        const float* const dl = reinterpret_cast<const float*>(tb + ColumnTable::kDl) + o;
        if (full) {
          PackBlockFull<E>(s, dp, l2, dl, p.scale_log2, pv, dsv);
        } else {
          const int32_t* const lo = reinterpret_cast<const int32_t*>(tb + ColumnTable::kLo) + o;
          const int32_t* const hi = reinterpret_cast<const int32_t*>(tb + ColumnTable::kHi) + o;
          const int64_t* const sg = reinterpret_cast<const int64_t*>(tb + ColumnTable::kSg) + o;
          // Exact: the preflight refused any call whose key axis does not fit in int32.
          const int32_t j32 = static_cast<int32_t>(j);
          if (seg) {
            PackBlockMasked<E, true>(s, dp, l2, dl, lo, hi, sg, j32, ks, p.scale_log2, pv, dsv);
          } else {
            PackBlockMasked<E, false>(s, dp, l2, dl, lo, hi, sg, j32, ks, p.scale_log2, pv, dsv);
          }
        }
      }
      if constexpr (kMetaWarps > 0) {
        // This warp's reads of slot `tslot` are done (their values are in pv / dsv): release it.
        __syncwarp();
        if (lane == 0) dvpipe::MbarArrive(bar0 + 8u * (Br::kMetaEmpty + tslot));
      }
      // In place over this warp's own (already loaded) S^T and dP^T columns.
      dkpipe::TmemStore16(lane_base | scol, pv);
      dkpipe::TmemStore16(lane_base | dcol, dsv);
      TmemWaitSt();
      TmemFenceBeforeSync();
      dvpipe::MbarArrive(bar0 + 8u * (Br::kPack + buf));
      any = true;
      ++r;
      c = n;
    }
    }  // !kGroups

    // ----------------------------------------------------------------- epilogue
    // The fused leaf's two sweeps (kKeyRows 128, kWrite 3), split across the warp groups: lane L
    // of warp w holds key row 32 (w & 3) + L; warps 0..3 drain and store dK (TMEM cols 0..127,
    // scale folded), warps 4..7 dV (cols 128..255, unscaled) of the same rows. A tile with no round
    // stores exact zeros.
    if (any) {
      const uint32_t last = r - 1;
      MbarWait(bar0 + 8u * (Br::kAcc + last % kStages), (last / kStages) & 1u);
      TmemFenceAfterSync();
    }
    const bool is_k = (ch == 0);
    const int cols = is_k ? kD : kDv;  // this product's own accumulator columns
    const int64_t width = is_k ? p.qk_dim : p.value_dim;
    const float mul = is_k ? p.scale : 1.f;
    void* row_out = nullptr;
    if (j < p.prev_length) {
      row_out = KeyGradRow<E, kFp32Keys>(is_k ? p.dprev_k : p.dprev_v,
                                         (b * p.prev_length + j) * p.kv_heads + hk, width);
    } else if (j >= p.prev_length && j < nt_total) {
      const int64_t lk = j - p.prev_length;
      row_out = KeyGradRow<E, kFp32Keys>(is_k ? p.dk : p.dv,
                                         (b * p.k_length + lk) * p.kv_heads + hk, width);
    }
    const uint32_t acc_col = is_k ? Pl::kdKCol : Pl::kdVCol;
    for (int cc = 0; cc < cols; cc += 32) {
      float v[32];
      if (any) {
        TmemLoad32(lane_base | (tbase + acc_col + cc), v);
        TmemWaitLd();
      } else {
#pragma unroll
        for (int x = 0; x < 32; ++x) v[x] = 0.f;
      }
      if (row_out != nullptr) StoreKeyGrad32<E, kFp32Keys>(row_out, cc, width, v, mul);
    }
  }

  // ------------------------------------------------------------------ tile exit
  // All threads (352, or 416 with the metadata warps): the pack warps after the epilogue (which
  // waited for the last dV/dK), the producer after its last issue, the two MMA warps after their
  // last commits, the metadata warps after their last table (whose readers, the pack warps, are
  // past it by then).
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
}

}  // namespace kvpipe

// The preflight (declared in umma_bwd_dkdv.cuh): the two-stage fused M128 leaf's own
// (UmmaBackwardDkdvCanLaunch<E, 128, 128, 3, 128, 4>: accepted set at the 128-row tile, buckets
// and real dimensions, element, device 10.3, its SMEM), a key axis that fits in int32 (the
// branch-free visibility test above) and this plan's dynamic SMEM. `false` means refused. The
// level gate is not here: DecideInSituKeyPair tests it before asking.
template <typename E>
bool CanLaunchUmmaBackwardKvPipe(const UmmaLaunchParams& lp) {
  using Pl = kvpipe::Plan<kvpipe::kRingStages>;
  const Params& p = lp.params;
  if (!UmmaBackwardDkdvCanLaunch<E, 128, 128, 3, 128, 4>(lp)) return false;
  if (p.total_length + kvpipe::kKeyRows > static_cast<int64_t>(INT32_MAX)) return false;
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: cudaGetDevice", q);
  int smem_cap = 0;
  q = cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (q != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: shared memory capacity", q);
  }
  return static_cast<int>(Pl::kSmemBytes) <= smem_cap;
}

// The launch (declared in umma_bwd_dkdv.cuh): the preflight again, then the committed launch.
// `false` means refused and nothing was written; after the preflight every driver error raises.
// Each enqueued launch adds one to kvpipe::KvPipeLaunchCount() (bwd_pipe_host.h).
template <typename E>
bool LaunchUmmaBackwardKvPipe(const UmmaLaunchParams& lp, cudaStream_t stream) {
  using Pl = kvpipe::Plan<kvpipe::kRingStages>;
  const Params& p = lp.params;
  if (!CanLaunchUmmaBackwardKvPipe<E>(lp)) return false;
  const auto kernel = p.key_grads_fp32 ? kvpipe::BackwardKvPipeKernel<E, kvpipe::kRingStages, true>
                                       : kvpipe::BackwardKvPipeKernel<E, kvpipe::kRingStages, false>;
  const cudaError_t attr =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: cudaFuncSetAttribute", attr);
  }
  const unsigned gx =
      static_cast<unsigned>((p.total_length + kvpipe::kKeyRows - 1) / kvpipe::kKeyRows);
  if (gx == 0) return true;  // no work (see LaunchUmmaBackwardDkdv)
  const dim3 grid(gx, static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
  kernel<<<grid, Pl::kThreads, Pl::kSmemBytes, stream>>>(lp);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: kernel launch", launched);
  }
  kvpipe::KvPipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
  return true;
}

// ------------------------------------------------------------------ the other buckets
// The same leaf at every other (D, Dv) bucket pair its plan holds - D, Dv in {64, 128} with
// D + Dv <= 256: (64,64), (64,128), (128,64). These buckets had no tcgen05 key leaf: the
// registry rows that name these functions (instantiations/sca_umma_bwd_registry.cu; instantiated
// in instantiations/sca_umma_bwd_pipe_d*_v*.cu) are their
// only native key writer, and every call they refuse runs the generic body as before. Four Q/dO
// stages: one stage is 16 KB at (64,64), so SMEM is not the limit there.
namespace kvpipe {
template <int kD, int kDv>
struct StagesFor {
  static constexpr int value = (kD == 128 && kDv == 128) ? kRingStages : 4;
};
}  // namespace kvpipe

// Preflight at bucket (kD, kDv): precision level 0 (no FP32 saved state, no subtract-before-
// scale), the fused leaf's accepted set at the 128-row key tile (umma_bwd_dkdv.cuh:Accepted:
// operands, GQA, no dq accumulator, previous-plane pairing and alignment), the call's buckets
// and real widths, the element, device 10.3, a key axis that fits in int32, and this plan's
// dynamic SMEM. `false` means refused and nothing is written.
template <typename E, int kD, int kDv>
bool CanLaunchUmmaBackwardKvPipeAt(const UmmaLaunchParams& lp) {
  using Pl = kvpipe::Plan<kvpipe::StagesFor<kD, kDv>::value, kD, kDv>;
  const Params& p = lp.params;
  if (p.saved_fp32 || p.subtract_before_scale) return false;
  if (!Accepted(lp, 3, kvpipe::kKeyRows)) return false;
  if (p.qk_dim <= 0 || p.value_dim <= 0 || p.qk_dim > kD || p.value_dim > kDv) return false;
  if (QkBucket(p.qk_dim) != kD || ValueBucket(p.value_dim) != kDv) return false;
  if (p.element != BwdTok<E>::kElement) return false;
  if (p.total_length + kvpipe::kKeyRows > static_cast<int64_t>(INT32_MAX)) return false;
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: cudaGetDevice", q);
  int major = 0, minor = 0;
  q = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: capability", q);
  q = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: capability", q);
  if (major != 10 || minor != 3) return false;
  int smem_cap = 0;
  q = cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (q != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: shared memory capacity", q);
  }
  return static_cast<int>(Pl::kSmemBytes) <= smem_cap;
}

template <typename E, int kD, int kDv>
bool LaunchUmmaBackwardKvPipeAt(const UmmaLaunchParams& lp, cudaStream_t stream) {
  constexpr int kS = kvpipe::StagesFor<kD, kDv>::value;
  using Pl = kvpipe::Plan<kS, kD, kDv>;
  const Params& p = lp.params;
  if (!CanLaunchUmmaBackwardKvPipeAt<E, kD, kDv>(lp)) return false;
  const auto kernel = p.key_grads_fp32 ? kvpipe::BackwardKvPipeKernel<E, kS, true, kD, kDv>
                                       : kvpipe::BackwardKvPipeKernel<E, kS, false, kD, kDv>;
  const cudaError_t attr =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: cudaFuncSetAttribute", attr);
  }
  const unsigned gx =
      static_cast<unsigned>((p.total_length + kvpipe::kKeyRows - 1) / kvpipe::kKeyRows);
  if (gx == 0) return true;
  const dim3 grid(gx, static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
  kernel<<<grid, Pl::kThreads, Pl::kSmemBytes, stream>>>(lp);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv pipe: kernel launch", launched);
  }
  kvpipe::KvPipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
  InSituLaunchedKeyRows() = kvpipe::kKeyRows;
  return true;
}

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
