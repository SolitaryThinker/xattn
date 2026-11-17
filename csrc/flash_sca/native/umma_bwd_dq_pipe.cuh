// Native Blackwell (sm_103a) SCA backward: the dQ leaf at (256,256), warp-specialized and
// pipelined across 64-key tiles ("dq_pipe"). LaunchUmmaDq<E, 256, 256> (umma_bwd_dq.cuh) runs it,
// BF16 and FP16, for every level-0 call it accepts while XATTN_SCA_BWD_DQ_PIPE is unset, empty or
// "1" (bwd_pipe_host.h); DqKernel serves every other call, and every call under "0".
//
// The pack (every bucket): P = 2^(S scale2 - lse2) as one ex2.approx.ftz (a P below 2^-126 is 0,
// where exp2f returned the subnormal; dS differs from DqKernel's only there), dS packed two at
// a time with cvt.rn.{bf16,f16}x2.f32 (the bits of the scalar conversions), and no mask test
// when a warp's 32 x 32 block is fully visible. Two pack groups of 4 warps alternate tiles
// (Plan::kPingPong): group g packs tiles n with n & 1 == g (S / dP buffer g), every thread its
// row's 64 columns; key-id tables and named barriers are per group, and the epilogue waits the
// group's own last dQ before the last tile's (see there).
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
// At (128,128) (Plan::kTmemA) Q and dO are resident in TMEM (columns 128..191 and 192..255) and
// S, dP are TS MMAs (A from TMEM, B = K / V from SMEM), so each M128 N64 K16 step reads only its
// 2 KB B operand from SMEM instead of 4 KB A + 2 KB B. The products and the per-accumulator K-step
// order are the same. Q and dO arrive once per CTA (Plan::kCp): TMA boxes into SMEM landing
// tiles, then the MMA warp's tcgen05.cp.128x256b into the resident columns, ahead of its MMAs in
// issue order (XATTN_SCA_DQ_CP=0: the pack warps' per-row loads + tcgen05.st, for a dev A/B). The
// K and V rings are 4 stages deep there (kKStages / kVStages; (256,256) keeps 3 / 1 under
// kDoTmem; the two-CTA small map takes the deepest pair its SMEM window allows).
// In both buckets the pack forms the visibility of its 32 columns as one bit mask per tile
// (same predicate as before), not per element.
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

#include <map>          // the tile counter's per-(device, stream) table
#include <mutex>        //   ... and the lock that guards it
#include <utility>

#include "flash_sca/native/bwd_pipe_host.h"
#include "flash_sca/native/umma_bwd_dq.cuh"
#include "flash_sca/native/umma_bwd_pt.cuh"  // the shared packed core (pt::FmaX2, ...)


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

template <int kD, int kDv>
struct Plan {
  // Q and dO resident in TMEM as the A operands of S = Q K^T and dP = dO V^T (TS MMAs, only B is
  // read from SMEM) where the columns fit: dQ kD + Q kD/2 + dO kDv/2 + S 2x64 + dP 2x64 <= 512,
  // i.e. every bucket with D, Dv <= 128 ((128,128) exactly 512; (64,64), (64,128), (128,64)).
  // (256,256) keeps Q and dO in SMEM (SS MMAs).
  // Which bucket takes which map is a capacity order, measured (the small map with the
  // two CTAs is 9.7-11.3% faster than the resident one on the D=64 rows): the 256-column map first
  // where dQ + S (two buffers) + dP (ONE buffer) fits it, the resident-Q/dO map only where that does
  // not fit, and the resident-dO map only where neither does.
  static constexpr bool kSmallMap = (kD + 3 * kBN) <= 256;
  static constexpr bool kTmemA = ((kD / 2 + kDv / 2 + kD) <= 256) && !kSmallMap;
  // Where Q and dO do not both fit but dO does (dQ kD + dO kDv/2 + ONE S and ONE dP buffer of 64
  // <= 512; (256,256) exactly 512, the reference ScaBwdDq map): dO alone is resident in TMEM as
  // the A operand of dP = dO V^T (TS), S = Q K^T stays SS, and dS goes to a double-buffered SMEM
  // tile as the A operand of dQ += dS K (SS), so the single S / dP buffers are free again as soon
  // as the pack has read them into registers (kBarDpFree), not after dQ.
  static constexpr bool kDoTmem = !kTmemA && !kSmallMap && (kD + kDv / 2 + 2 * kBN) <= 512;
  // K / V ring depths. With Q and dO out of SMEM the freed bytes deepen both rings, so the K(n+KS)
  // and V(n+VS) loads are issued KS / VS tiles ahead instead of one tile ahead.
  // kDoTmem: K(n) is read by S(n) and dQ(n), and dQ(n) is issued after S(n+1), so K gets the
  // third stage (K(n+3) loads under S(n+1)..dQ(n+1)); V(n) is free once dP(n) completes.
  // kSmallMap (two CTAs per SM, Q and dO in SMEM): the deepest K / V ring, from 4 / 4 down to
  // 2 / 2, whose CTA image still lets two CTAs share the SM: 2 x (image + 1 KB reserved) <=
  // 233,472 B (the kTwoCta static_assert below). A one-stage V ring cannot load V(n) before dP(n-1)
  // completes, which put that TMA flight in front of every tile's dP. The image is the Q and
  // dO tiles, the rings and the 3 KB tail with the 1 KB rebase slack, as kSmemBytes counts it.
  static constexpr int SmallImage(int ks, int vs) {
    return 2 * kBM * kD + 2 * kBM * kDv + ks * 2 * kBN * kD + vs * 2 * kBN * kDv + 3072;
  }
  static constexpr bool SmallFits(int ks, int vs) {
    return 2 * (SmallImage(ks, vs) + 1024) <= 233472;
  }
  static constexpr int kSmallStages = SmallFits(4, 4) ? 4 : (SmallFits(3, 3) ? 3 : 2);
  static constexpr int kKStages = kTmemA ? 4 : (kDoTmem ? 3 : (kSmallMap ? kSmallStages : 2));
  static constexpr int kVStages =
      kTmemA ? 4 : (kSmallMap ? (SmallFits(kSmallStages, kSmallStages) ? kSmallStages : 1) : 1);
  // Two pack groups of 4 warps alternate tiles (group g packs the tiles n with n & 1 == g, all 64
// columns of its rows), instead of all 8 warps packing every tile. It needs two tiles' dP in
  // flight at once, so it is defined below, from the TMEM map: kPingPong = (kDpBufs == 2).
  // mbarrier slots (8 B each) after the operand image.
  static constexpr int kBarQ = 0;                       // Q landed (TMA) / Q,dO in TMEM (kTmemA)
  static constexpr int kBarDy = 1;                      // dO landed (TMA, expect_tx)
  static constexpr int kBarK = 2;                       // + n % KS: K(n) landed (TMA)
  static constexpr int kBarV = kBarK + kKStages;        // + n % VS: V(n) landed (TMA)
  static constexpr int kBarVfree = kBarV + kVStages;    // + n % VS: dP(n) complete (commit)
  static constexpr int kBarS = kBarVfree + kVStages;    // + (n & 1): S(n), dP(n) complete
  static constexpr int kBarDq = kBarS + 2;              // + (n & 1): dQ(n) complete (commit)
  static constexpr int kBarPack = kBarDq + 2;           // + (n & 1): dS(n) packed (256 arrivals)
  static constexpr int kBarKfree = kBarPack + 2;        // + n % KS: dQ(n) complete: K stage free
  // The single dP buffer (kDpBufs == 1, the two D = 64 buckets): pack(n) reports that it has read
  // dP(n) out of TMEM, so the MMA warp may overwrite that buffer with dP(n + 2). At every bucket
  // with two dP buffers nothing waits on it and nothing arrives on it.
  static constexpr int kBarDpFree = kBarKfree + kKStages;
  static constexpr int kBarCount = kBarDpFree + 1;
  // Two buckets, both on the objective: (256,256) is the 0.9B training cell, (128,128) the 4B one.
  // The slab geometry is DqPlan<kD,kDv>'s, reused and not re-derived, so the per-bucket quantities
  // here are only the byte totals, and each is asserted below.
  // Also (64,64), (64,128) and (128,64): the same layout with fewer slabs
  // (umma_bwd_pipe_small.cuh, instantiated in instantiations/sca_umma_bwd_pipe_d*_v*.cu).
  // One rule, not a bucket list: the map below is derived from (D, Dv), and it is asked for at
  // every pair of head dims the API accepts (multiples of 64 up to 256), not at the buckets that
  // happen to be instantiated today. Which map a pair gets follows from the capacity order above;
  // whether it fits is asserted further down, once the totals are known.
  static_assert(kD % 64 == 0 && kDv % 64 == 0 && kD <= 256 && kDv <= 256,
                "the pipelined dQ leaf takes every bucket the API accepts: multiples of 64 up to 256");
  using Base = DqPlan<kD, kDv>;  // DqKernel's slab geometry, reused, not re-derived
  static constexpr int kSlabsQ = Base::kSlabsQ;    // 4
  static constexpr int kSlabsK = Base::kSlabsK;    // 4
  static constexpr int kSlabsV = Base::kSlabsV;    // 4 (dO's slab count as well)
  static constexpr int kSlab = Base::kSlab;        // 16,384: 128 rows x 128 B
  static constexpr int kKvSlab = Base::kKvSlab;    // 8,192: 64 rows x 128 B
  // Under kTmemA, Q and dO land by TMA in SMEM tiles (the same boxes and barriers as the SS
  // buckets) and the MMA warp copies them into their TMEM columns with tcgen05.cp. The older
  // per-row path (global loads + tcgen05.st by the pack warps) is dead on every bucket and is kept
  // below as unreachable code only until a later change removes it; a measurement showed that the dev A/B
  // switch had no arm worth keeping, so it is gone.
  static constexpr bool kCp = kTmemA;
  // SMEM bytes of the Q / dO tiles: none under kTmemA without kCp (they live in TMEM only); under
  // kCp they are the TMA landing tiles of the copy.
  static constexpr int kQBytes = (kTmemA && !kCp) ? 0 : Base::kQBytes;  // 65,536 at (256,256)
  static constexpr int kDyBytes = ((kTmemA && !kCp) || kDoTmem) ? 0 : Base::kDyBytes;
  // kDoTmem: two dS tiles, 128 rows x 64 keys bf16 = one 128 B-swizzled slab each.
  static constexpr int kDsBytes = kDoTmem ? 2 * kSlab : 0;
  static constexpr int kKStageBytes = kSlabsK * kKvSlab;  // 32,768 / 16,384
  static constexpr int kVStageBytes = kSlabsV * kKvSlab;  // 32,768 / 16,384
  static constexpr int kVBytes = kVStageBytes;            // one stage
  static constexpr int kQOff = 0;
  static constexpr int kDyOff = kQOff + kQBytes;
  static constexpr int kKOff = kDyOff + kDyBytes;
  static constexpr int kVOff = kKOff + kKStages * kKStageBytes;
  static constexpr int kDsOff = kVOff + kVStages * kVStageBytes;
  static constexpr int kBarOff = kDsOff + kDsBytes;
  static constexpr int kSlotOff = kBarOff + 256;                 // 32 barrier slots
  static constexpr int kRedOff = kSlotOff + 16;                  // 4 warps x 4 int32
  static constexpr int kSegOff = kRedOff + 4 * 16;               // 64 key ids per pack group
  static constexpr int kEnd = kSegOff + 2 * kBN * 8;
  static constexpr int kSmemBytes = ((kEnd + 1023) / 1024) * 1024 + 1024;
  static constexpr int kQCol = kD;              // 128 at (128,128): Q packed, 2 elements per column
  static constexpr int kDoCol = kTmemA ? kD + kD / 2 : kD;  // 192 at (128,128); 256 at (256,256)
  static_assert(!kTmemA || kDoCol + kDv / 2 <= 256, "Q and dO columns below the S buffers");
  // TMEM columns.
// A bucket whose dQ + S (two buffers) + dP (ONE buffer) fits 256 columns takes a 256-column
  // map so that two CTAs share the SM's 512 columns - dQ 64 | S 2 x 64 | dP ONE buffer of 64. A
  // single dP buffer is safe because the pack reads dP into registers before it stores dS, and it
  // arrives on kBarDpFree once that read has landed; the MMA warp then waits that arrival before it
  // overwrites the buffer with dP(m). Every other bucket keeps the 512-column map (two dP buffers,
  // no kBarDpFree wait) exactly as it was.
  static constexpr int kDqCol = 0;
  static constexpr int kBufCols = kBN;
  // The maps are ordered by capacity, and where two of them both fit the order is measured, not
  // argued: the small map with its two CTAs is 9.7-11.3% faster at D=64 than the resident-Q/dO map
  // (+1.08% of the objective), so the small map is tried first and the
  // resident map only where it does not fit. The two never fit together (dQ + Q + dO + S + dP >
  // 256 columns), and kTmemA already carries !kSmallMap, so kTwoCta is the small map with a
  // second CTA beside it.
  //
  // The second CTA is a capacity fact, not a bucket name: the small map's 256 columns let
  // two CTAs share the SM's TMEM, and they share its SMEM only if both fit beside the 1 KB the
  // system reserves per CTA. Where they do not - (64,256), 2 x 135,168 against
  // 233,472 - the SAME map runs at one CTA, which only `__launch_bounds__`'s kMinBlocks reads.
  static constexpr bool kTwoCta = kSmallMap && (2 * (kSmemBytes + 1024) <= 233472);
  static constexpr int kDpBufs = (kTwoCta || kDoTmem) ? 1 : 2;
  static constexpr int kSBufs = kDoTmem ? 1 : 2;
  // The pack groups follow the buffers, not a bucket: two groups alternating tiles need two tiles'
  // S / dP in flight, and a single-buffer map reaches tile n + 1 only after tile n has been read.
  static constexpr bool kPingPong = (kDpBufs == 2);
  static constexpr int kPackArrivals = kPingPong ? kPackThreads / 2 : kPackThreads;
  static constexpr int kMinBlocks = kTwoCta ? 2 : 1;  // __launch_bounds__'s second argument
  // Where dS is the TMEM A operand of dQ and one CTA holds the SM (kTmemA), dQ += dS K is
  // issued by a warp of its own (warp 10), so the S / dP issuer's waits and commits and the dQ
  // issuer's run beside each other and the tensor pipe's queue is fed from two warps. The
  // two-CTA small map keeps one issuer: 352 threads would not fit its register budget.
  static constexpr bool kDqIssuer = kTmemA;
  static constexpr int kThreads = 32 * (kPackWarps + 2 + (kDqIssuer ? 1 : 0));
  // S buffer b at kSCol + 64 b (dS in place, except under kDoTmem where dS goes to SMEM).
  static constexpr int kSCol = kTwoCta ? kD : (kDoTmem ? kD + kDv / 2 : 256);
  static constexpr int kDpCol = kSCol + kSBufs * kBufCols;  // dP buffer b at kDpCol + 64 b
  static constexpr int kAllocCols = kTwoCta ? 256 : 512;
// Two pack groups alternate tiles. That needs two tiles' dP in flight at once, which the
  // 512-column maps give and the single-dP maps (kTwoCta's and kDoTmem's) cannot: with one buffer
  // the group that reached tile n + 1 would wait on the group packing tile n to finish reading
  // dP(n), serializing on the very barrier that exists to refill the buffer. A capacity
  // consequence, not a bucket name; the declarations sit above, with the buffers.
  //
  // The two maps are alternatives, never both: with Q/dO resident the S buffers start at 256 and a
  // two-CTA map would put Q and dO inside them, and S(n) would overwrite the operand of its own
  // TS MMA after the first tile. The asserts below make that overlap impossible to compile; they
  // sit after the TMEM-column block because a static_assert at class scope cannot name a member
  // declared further down.

  static_assert(!(kTmemA && kTwoCta), "the resident Q/dO map and the 256-column map exclude each other");
  static_assert(!(kDoTmem && kTwoCta), "the resident-dO map and the 256-column map exclude each other");
  static_assert(!kTmemA || kDoCol + kDv / 2 <= kSCol, "Q and dO columns below the S buffers");
  static_assert(!kDoTmem || (kDoCol >= kDqCol + kD && kDoCol + kDv / 2 <= kSCol),
                "dO columns between dQ and the S buffer");
  static_assert(kDqCol + kD <= kSCol, "dQ sits below both S buffers");
  static_assert(kSCol + kSBufs * kBufCols <= kDpCol, "the S buffers below the dP buffers");
  static_assert(!kDoTmem || 2 * kBN * 2 == 2 * 128, "a dS tile row is one 128 B swizzle row");
  static_assert(kDsOff % 1024 == 0, "dS tiles 1024-aligned (128 B swizzle atoms)");
  static_assert(kDpCol + kDpBufs * kBufCols <= kAllocCols, "the dP buffers fit the allocation");
  static_assert(!kTwoCta || (kSCol == 64 && kDpCol == 192 && kAllocCols == 256 && kDpBufs == 1),
                "the 256-column map at a 64-wide head dim");
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
  // (256,256) under kDoTmem: 4 Q slabs + 3 K stages + 1 V stage + 2 dS tiles = 229,376 B, plus
  // the tail, rounded: 232,448 B.
  // (128,128): 4 K stages + 4 V stages of 16 KB + the tail = 133,120 B; one CTA per SM either way
  // (the 512-column TMEM allocation).
  // The tail holds two 64-entry key-id tables (one per pack group): 3 KB with the barriers and
  // the 1 KB rebase slack.
  static_assert(kD != 256 || kDv != 256 || kSmemBytes == 232448,
                "SMEM total at (256,256), computed in the header comment");
  // Every bucket with D, Dv <= 128 holds Q and dO in TMEM (kTmemA) and SMEM is the two rings plus
  // the tail: (128,128) 4 x 16 KB + 4 x 16 KB; (64,64) 4 x 8 + 4 x 8 KB; (64,128) 4 x 8 + 4 x 16;
  // (128,64) 4 x 16 + 4 x 8.
  // Under kCp SMEM also holds the Q and dO landing tiles: (128,128) 2 x 32 KB more.
  static_assert(!kTmemA || kCp || kQBytes + kDyBytes == 0, "no Q / dO tiles in SMEM under kTmemA");
  static_assert(!kTmemA || kSmemBytes == kQBytes + kDyBytes + kKStages * kKStageBytes +
                                             kVStages * kVStageBytes + 3072,
                "SMEM total under kTmemA: the Q/dO landing tiles (kCp), the K and V rings + a 3 KB tail");
  static_assert(kD != 128 || kDv != 128 || kSmemBytes == (kCp ? 199680 : 134144),
                "SMEM total at (128,128)");
  static_assert(kSmemBytes <= 232448, "the sm_103a opt-in dynamic SMEM ceiling");
  // The second CTA shares the SM's SMEM only if both fit beside the 1 KB the system reserves
  // per CTA: 2 x (dynamic + reserved) against the SM's 233,472 B.
  static_assert(!kSmallMap || kSmemBytes == SmallImage(kKStages, kVStages),
                "the small map's ring rule counts the image kSmemBytes allocates");
  static_assert(!kTwoCta || 2 * (kSmemBytes + 1024) <= 233472,
                "the two-CTA map fits the SM's SMEM window");
};

// One 16-element K step of a 128-row operand tile from SMEM (the descriptor an SS MMA would
// read for that step) into 8 packed TMEM columns, 128 lanes x 256 bits. Issued by the elected
// lane of the MMA warp - the same lane that issues the MMAs reading those columns, so the copy
// and the MMAs after it are a pipelined tcgen05 pair, executed in issue order (PTX ISA, tcgen05
// memory consistency: tcgen05.cp -> tcgen05.mma from the same thread). Moving either to another
// lane or warp breaks that ordering and would need a commit + wait in between.
__device__ __forceinline__ void TmemCopy128x256b(uint32_t taddr, SmemDesc src) {
#if XATTN_UMMA_DEVICE_ASM
  if (ElectOne()) {
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" ::"r"(taddr), "l"(src.value)
                 : "memory");
  }
#else
  (void)taddr;
  (void)src;
#endif
}

// TMEM column (relative to the allocation base) of the first packed dS word of buffer `buf`: dS
// lives in place over the S buffer, whose base is the plan's kSCol.
__host__ __device__ constexpr uint32_t DsBase(int s_col, uint32_t buf) {
  return static_cast<uint32_t>(s_col + 64 * static_cast<int>(buf));
}
// Key step kk (16 keys): the pack warp of column half kk >> 1 wrote its 16 words at the start
// of its own 32 columns, two 16-key steps of 8 words each.
__host__ __device__ constexpr uint32_t DsColumn(int s_col, uint32_t buf, int kk) {
  return DsBase(s_col, buf) + static_cast<uint32_t>(32 * (kk >> 1) + 8 * (kk & 1));
}
static_assert(DsColumn(256, 1, 3) == 256 + 64 + 40 && DsColumn(256, 0, 2) == 256 + 32,
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

// Barrier of the 128 threads of pack group g (ids 2 and 3; ping-pong only).
__device__ __forceinline__ void GroupBarrierSync(int g) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("bar.sync %0, %1;" ::"r"(kPackBarrier + 1 + g), "n"(kPackThreads / 2) : "memory");
#else
  (void)g;
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
__global__ void __launch_bounds__(Plan<kD, kDv>::kThreads, Plan<kD, kDv>::kMinBlocks) DqPipeKernel(
    const __grid_constant__ UmmaLaunchParams lp, unsigned int* const tile_counter,
    const int total_tiles) {
  using Pl = Plan<kD, kDv>;
  // The single-dP refill barrier's slot is Plan's; four uses below name it in this scope, so it is
  // brought in here by hand. Everything else in Plan is reached through Pl::.
  constexpr int kBarDpFree = Pl::kBarDpFree;
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
  constexpr int kKS = Pl::kKStages, kVS = Pl::kVStages;
  constexpr int kBarQ = Pl::kBarQ, kBarDy = Pl::kBarDy, kBarK = Pl::kBarK, kBarV = Pl::kBarV;
  constexpr int kBarVfree = Pl::kBarVfree, kBarS = Pl::kBarS, kBarDq = Pl::kBarDq;
  constexpr int kBarPack = Pl::kBarPack, kBarKfree = Pl::kBarKfree, kBarCount = Pl::kBarCount;
  static_assert(kBarCount * 8 <= Pl::kSlotOff - Pl::kBarOff, "barrier area");

  const int tid = threadIdx.x;
  
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool pack_warp = warp < kPackWarps;
  const int w4 = warp & 3;         // TMEM lane quadrant of this warp
  const int ch = (warp >> 2) & 1;  // column half of a pack warp: 0 -> keys 0..31, 1 -> 32..63
  // THE QUERY ROW OF THIS THREAD comes from the TMEM lane it may access, never from tid: in
  // DqKernel tid == row because kThreads == kBM; here warps w and w + 4 own the same 32 rows.
  const int row = 32 * w4 + lane;

  // ------------------------------------------------------------- Persistent CTA
  // One CTA per SM. A CTA claims one query tile at a time from the call's counter (reset before the
  // launch) and runs the whole tile, so the launch, the TMEM allocation and the CTA's fixed set-up
  // are paid once per CTA instead of once per tile. A tile is computed by exactly one CTA with the
  // same code path and the same operands whatever order the tiles are claimed in, so dq is
  // bit-identical to the tile-per-CTA kernel. The claim word is broadcast through the 12 bytes of
  // slack between tmem_slot and red.
  int32_t* const claim = reinterpret_cast<int32_t*>(smem + Pl::kSlotOff) + 1;
  const int nqt = (p.q_length + kBM - 1) / kBM;  // query tiles per (head, batch)
  const int64_t nhq = p.query_heads;
  const int64_t tiles_per_batch = static_cast<int64_t>(nqt) * nhq;
  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);  // once per CTA; every tile reuses these columns
    TmemRelinquish();
  }
  __syncthreads();
  uint32_t tbase = *tmem_slot & 0x1FFu;  // 1-CTA allocation: lane 0, column base

  for (;;) {
  if (tid == 0) *claim = static_cast<int32_t>(atomicAdd(tile_counter, 1u));
  __syncthreads();
  const int32_t tile = *claim;
  __syncthreads();  // the claim is read by every thread before the next one overwrites it
  if (tile >= total_tiles) break;
  const int qt = tile % nqt;
  const int64_t hq = (tile / nqt) % nhq;
  const int64_t b = tile / tiles_per_batch;
  const int64_t kvh = p.group > 0 ? hq / p.group : hq;  // GQA/MQA: as DqKernel
  const int64_t i = static_cast<int64_t>(qt) * kBM + row;
  const bool row_in_range = pack_warp && i < p.q_length;

  if (tid == 0) {
    for (int s = 0; s < kBarCount; ++s) {
// kBarPack / kBarDpFree count the threads that pack one tile (one group of 128 under
      // ping-pong, else all 256); under kTmemA without kCp, kBarQ counts the 256 pack threads that
      // wrote Q (warps 0..3) and dO (warps 4..7) into TMEM instead of a TMA arrive. Under kCp
      // kBarQ / kBarDy are TMA barriers, as in the SS buckets.
      MbarInit(XATTN_DQP_BAR(s), (s == kBarPack || s == kBarPack + 1 || s == kBarDpFree)
                                     ? Pl::kPackArrivals
                                     : (((Pl::kTmemA && !Pl::kCp && s == kBarQ) ||
                                         (Pl::kDoTmem && s == kBarDy))
                                            ? kPackThreads
                                            : 1));
    }
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
    continue;  // next claim; the CTA's TMEM columns stay allocated for it
  }

  const int t0 = min_lo / kBN;
  const int t1 = max_hi / kBN;
  const int steps = t1 - t0 + 1;  // >= 1 here
  const bool seg = detail::SegmentTest(p, b);
  const int64_t nt_total = p.total_length;

  if (warp == kProducerWarp) {
    // ------------------------------------------------------------- TMA producer
    if (lane == 0) {
      // Resident tiles, as DqKernel: Q and dO, kSlabs x (128 / 64) boxes each (SMEM operands only;
      // under kTmemA they are the landing tiles of the MMA warp's tcgen05.cp (kCp), or the
      // pack warps write them into TMEM themselves (kCp off)).
      if constexpr (!Pl::kTmemA || Pl::kCp) {
      MbarArriveExpectTx(XATTN_DQP_BAR(kBarQ), Pl::kQBytes);
      for (int s = 0; s < Pl::kSlabsQ; ++s) {
        for (int r = 0; r < kBM; r += kTmaRows) {
          TmaLoad4D(sq + s * Pl::kSlab + r * 128, &lp.maps.q, XATTN_DQP_BAR(kBarQ), s * 64,
                    static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
        }
      }
      if constexpr (!Pl::kDoTmem) {
      MbarArriveExpectTx(XATTN_DQP_BAR(kBarDy), Pl::kDyBytes);
      for (int s = 0; s < Pl::kSlabsV; ++s) {
        for (int r = 0; r < kBM; r += kTmaRows) {
          TmaLoad4D(sdy + s * Pl::kSlab + r * 128, &lp.maps.dy, XATTN_DQP_BAR(kBarDy), s * 64,
                    static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
        }
      }
      }  // !kDoTmem
      }  // !kTmemA
      // Two streams over the same tile sequence t0 .. t1; neither waits for the other.
      int nk = 0, nv = 0;
      uint32_t idle = 0;
      while (nk < steps || nv < steps) {
        bool moved = false;
        if (nk < steps &&
            (nk < kKS || MbarTestWait(XATTN_DQP_BAR(kBarKfree + nk % kKS),
                                      static_cast<uint32_t>(((nk - kKS) / kKS) & 1)))) {
          // K stage nk % KS is free once dQ(nk - KS), its last reader, completed.
          const int64_t j0 = static_cast<int64_t>(t0 + nk) * kBN;
          const CUtensorMap* kmap;
          const CUtensorMap* vmap;
          int row0;
          ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);  // as DqKernel, packed map included
          const uint32_t bk = XATTN_DQP_BAR(kBarK + nk % kKS);
          MbarArriveExpectTx(bk, Pl::kKStageBytes);
          unsigned char* const kdst = sk + (nk % kKS) * Pl::kKStageBytes;
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
            (nv < kVS || MbarTestWait(XATTN_DQP_BAR(kBarVfree + nv % kVS),
                                      static_cast<uint32_t>(((nv - kVS) / kVS) & 1)))) {
          // V stage nv % VS is free once dP(nv - VS), its only reader, completed.
          const int64_t j0 = static_cast<int64_t>(t0 + nv) * kBN;
          const CUtensorMap* kmap;
          const CUtensorMap* vmap;
          int row0;
          ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);
          const uint32_t bv = XATTN_DQP_BAR(kBarV + nv % kVS);
          unsigned char* const vdst = sv + (nv % kVS) * Pl::kVStageBytes;
          MbarArriveExpectTx(bv, Pl::kVBytes);
          for (int s = 0; s < Pl::kSlabsV; ++s) {
            for (int r = 0; r < kBN; r += kTmaRows) {
              TmaLoad4D(vdst + s * Pl::kKvSlab + r * 128, vmap, bv, s * 64,
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
  } else if (warp == kMmaWarp || (Pl::kDqIssuer && warp == kMmaWarp + 1)) {
    const bool dq_role = Pl::kDqIssuer && warp == kMmaWarp + 1;  // the dQ issuer
    // ------------------------------------------------------------- MMA issuer
    const int elem_fmt = ElemTraits<Elem>::kFormat;
    // DqKernel's two instruction descriptors and dQ's B leading byte offset, verbatim.
    const uint32_t ss_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/0);
    const uint32_t ts_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/1);
    const uint32_t dq_lbo = Pl::kSlabsK > 1 ? static_cast<uint32_t>(kBN * 8) : 0u;
    // kDoTmem's dQ MMA: N = kD (<= 256) in one instruction.
    const uint32_t dq_idesc = MakeInstrDesc(elem_fmt, kBM, kD <= 256 ? kD : 256, 0, 1);
    const uint32_t qb_smem = SmemAddr(sq);
    const uint32_t dyb_smem = SmemAddr(sdy);
    const uint32_t k0_smem = SmemAddr(sk);
    const uint32_t v0_smem = SmemAddr(sv);
    const uint32_t ds0_smem = SmemAddr(smem + Pl::kDsOff);
    // (1) S(n) = Q K(n)^T into S buffer n & 1: DqKernel's k-loop.
    auto issue_s = [&](int n) {
      const uint32_t kb_smem = k0_smem + static_cast<uint32_t>((n % kKS) * Pl::kKStageBytes);
      const uint32_t dst = tbase + Pl::kSCol +
                           static_cast<uint32_t>((n & 1) * Pl::kBufCols * (Pl::kSBufs - 1));
      for (int kk = 0; kk < kD / 16; ++kk) {
        const int slab = kk >> 2, within = kk & 3;
        if constexpr (Pl::kTmemA) {
          // A = Q from TMEM: 16 head-dim elements of step kk are the 8 packed columns 8 kk..+7.
          MmaTs(dst, tbase + static_cast<uint32_t>(Pl::kQCol + 8 * kk),
                MakeSmemDesc(kb_smem + slab * Pl::kKvSlab + within * 32), ss_idesc,
                kk == 0 ? 0u : 1u);
        } else {
          MmaSs(dst, MakeSmemDesc(qb_smem + slab * Pl::kSlab + within * 32),
                MakeSmemDesc(kb_smem + slab * Pl::kKvSlab + within * 32), ss_idesc,
                kk == 0 ? 0u : 1u);
        }
      }
    };
    // (2) dP(n) = dO V(n)^T into dP buffer n & 1: DqKernel's (slab, k-step) loop.
    auto issue_dp = [&](int n) {
const uint32_t dst =
          tbase + Pl::kDpCol +
          static_cast<uint32_t>((n & 1) * Pl::kBufCols * (Pl::kDpBufs - 1));
      const uint32_t vb_smem = v0_smem + static_cast<uint32_t>((n % kVS) * Pl::kVStageBytes);
      for (int sl = 0; sl < Pl::kSlabsV; ++sl) {
        for (int kk = 0; kk < 4; ++kk) {
          if constexpr (Pl::kTmemA || Pl::kDoTmem) {
            MmaTs(dst, tbase + static_cast<uint32_t>(Pl::kDoCol + 8 * (4 * sl + kk)),
                  MakeSmemDesc(vb_smem + sl * Pl::kKvSlab + kk * 32), ss_idesc,
                  (sl == 0 && kk == 0) ? 0u : 1u);
          } else {
            MmaSs(dst, MakeSmemDesc(dyb_smem + sl * Pl::kSlab + kk * 32),
                  MakeSmemDesc(vb_smem + sl * Pl::kKvSlab + kk * 32), ss_idesc,
                  (sl == 0 && kk == 0) ? 0u : 1u);
          }
        }
      }
    };
    // (5) dQ += dS(n) K(n): DqKernel's per-slab loop, A = dS of buffer n & 1 from TMEM.
    auto issue_dq = [&](int n) {
      const uint32_t kb_smem = k0_smem + static_cast<uint32_t>((n % kKS) * Pl::kKStageBytes);
      const uint32_t buf = static_cast<uint32_t>(n & 1);
      for (int sl = 0; sl < Pl::kSlabsK; ++sl) {
        for (int kk = 0; kk < kBN / 16; ++kk) {
          if constexpr (Pl::kDoTmem) {
            // A = dS(n) from its SMEM tile (128 B-swizzled, K-major like a Q slab): 16 keys of
            // step kk are bytes 32 kk.. of each row. One N = kD MMA per step (B walks the kD / 64
            // slabs through dq_lbo), so A is read once per step, not once per slab.
            if (sl == 0) {
              MmaSs(tbase + Pl::kDqCol,
                    MakeSmemDesc(ds0_smem + buf * static_cast<uint32_t>(Pl::kSlab) + kk * 32),
                    MakeSmemDesc(kb_smem + kk * 16 * 128, dq_lbo, 64), dq_idesc,
                    (n == 0 && kk == 0) ? 0u : 1u);
            }
          } else {
            // A = dS(n) from TMEM. One N = kD MMA per 16-key step, B walking the kD / 64 K slabs
            // through dq_lbo as kDoTmem's does: kD / 16 x 4 instructions per tile become 4 (at
            // (128,128) the MMA warp's issue path is the round). Each dQ element receives
            // the same K = 16 steps in the same order from the same operands as the per-slab form.
            if (sl == 0) {
              MmaTs(tbase + Pl::kDqCol, tbase + DsColumn(Pl::kSCol, buf, kk),
                    MakeSmemDesc(kb_smem + kk * 16 * 128, dq_lbo, 64), dq_idesc,
                    (n == 0 && kk == 0) ? 0u : 1u);
            }
          }
        }
      }
    };
    if (dq_role) {
      // dQ(n) once pack(n) has stored dS(n); the commits that track dQ(n) are this warp's
      // (a commit tracks the MMAs its own thread issued). K(n)'s other reader S(n) completed
      // before pack(n) could start, so dQ(n)'s completion frees K stage n % KS.
      for (int n = 0; n < steps; ++n) {
        MbarWait(XATTN_DQP_BAR(kBarPack + (n & 1)), static_cast<uint32_t>((n >> 1) & 1));
        TmemFenceAfterSync();
        __syncwarp();
        issue_dq(n);
        MmaCommit(XATTN_DQP_BAR(kBarDq + (n & 1)));
        if (n + kKS < steps) MmaCommit(XATTN_DQP_BAR(kBarKfree + n % kKS));
      }
    } else {
    MbarWait(XATTN_DQP_BAR(kBarQ), 0u);
    if constexpr (Pl::kTmemA && !Pl::kCp) {
      TmemFenceAfterSync();  // the pack warps' tcgen05.st of Q and dO precede every MMA
    } else if constexpr (Pl::kTmemA) {
      // Q and dO landed in SMEM (TMA); copy each 16-element K step into its 8 packed TMEM
      // columns with the descriptor an SS MMA would read. The copies precede the S / dP MMAs in
      // this lane's issue order, which orders them (cp -> mma, same thread): no commit, no wait.
      MbarWait(XATTN_DQP_BAR(kBarDy), 0u);
      TmemFenceAfterSync();
      __syncwarp();
      for (int kk = 0; kk < kD / 16; ++kk) {
        TmemCopy128x256b(tbase + static_cast<uint32_t>(Pl::kQCol + 8 * kk),
                         MakeSmemDesc(qb_smem + (kk >> 2) * Pl::kSlab + (kk & 3) * 32));
      }
      for (int kk = 0; kk < kDv / 16; ++kk) {
        TmemCopy128x256b(tbase + static_cast<uint32_t>(Pl::kDoCol + 8 * kk),
                         MakeSmemDesc(dyb_smem + (kk >> 2) * Pl::kSlab + (kk & 3) * 32));
      }
    } else {
      MbarWait(XATTN_DQP_BAR(kBarDy), 0u);
      if constexpr (Pl::kDoTmem) TmemFenceAfterSync();  // the pack warps' tcgen05.st of dO
    }
    
    const int pro = Pl::kDoTmem ? 0 : (steps < 2 ? steps : 2);
    // V(m) landed: barrier kBarV + m % VS, its (m / VS)-th phase; K(m) likewise with KS.
    auto wait_v = [&](int m) {
      MbarWait(XATTN_DQP_BAR(kBarV + m % kVS), static_cast<uint32_t>((m / kVS) & 1));
    };
    auto wait_k = [&](int m) {
      MbarWait(XATTN_DQP_BAR(kBarK + m % kKS), static_cast<uint32_t>((m / kKS) & 1));
    };
    for (int m = 0; m < pro; ++m) {
      wait_v(m);
      TmemFenceAfterSync();
      __syncwarp();
      if (Pl::kDpBufs == 1 && m >= 1) {
        // One dP buffer: pack(m - 1) must have read dP(m - 1) out of it.
        MbarWait(XATTN_DQP_BAR(kBarDpFree), static_cast<uint32_t>((m - 1) & 1));
      }
      
      issue_dp(m);
      MmaCommit(XATTN_DQP_BAR(kBarVfree + m % kVS));
      wait_k(m);
      __syncwarp();
      issue_s(m);
      MmaCommit(XATTN_DQP_BAR(kBarS + m));  // S(m) and, issued before it, dP(m)
    }
    if constexpr (Pl::kDoTmem) {
      // One S and one dP buffer, dS in SMEM: S(n+1) / dP(n+1) are issued as soon as the pack has
      // read S(n) / dP(n) into registers (kBarDpFree, phase n), before dQ(n), so the tensor pipe
      // computes them while the pack forms dS(n).
      auto issue_sdp = [&](int m) {
        wait_v(m);
        TmemFenceAfterSync();
        __syncwarp();
        issue_dp(m);
        MmaCommit(XATTN_DQP_BAR(kBarVfree + m % kVS));
        wait_k(m);
        __syncwarp();
        issue_s(m);
        MmaCommit(XATTN_DQP_BAR(kBarS + (m & 1)));  // S(m) and, issued before it, dP(m)
      };
      issue_sdp(0);
      for (int n = 0; n < steps; ++n) {
        if (n + 1 < steps) {
          MbarWait(XATTN_DQP_BAR(kBarDpFree), static_cast<uint32_t>(n & 1));
          issue_sdp(n + 1);
        }
        MbarWait(XATTN_DQP_BAR(kBarPack + (n & 1)), static_cast<uint32_t>((n >> 1) & 1));
        __syncwarp();
        issue_dq(n);
        MmaCommit(XATTN_DQP_BAR(kBarDq + (n & 1)));
        if (n + kKS < steps) MmaCommit(XATTN_DQP_BAR(kBarKfree + n % kKS));
      }
    } else
    for (int n = 0; n < steps; ++n) {
      const int m = n + 2;
      if constexpr (Pl::kDqIssuer) {
        // dQ(n) is the other warp's. dP(m) overwrites dP buffer n & 1, whose last reader is
        // pack(n), so this warp still waits pack(n), but only when it has a dP(m) to issue.
        if (m < steps) {
          MbarWait(XATTN_DQP_BAR(kBarPack + (n & 1)), static_cast<uint32_t>((n >> 1) & 1));
          TmemFenceAfterSync();
        }
      } else {
      // dQ(n): dS(n) packed.
      MbarWait(XATTN_DQP_BAR(kBarPack + (n & 1)), static_cast<uint32_t>((n >> 1) & 1));
      TmemFenceAfterSync();
      __syncwarp();
      issue_dq(n);
      MmaCommit(XATTN_DQP_BAR(kBarDq + (n & 1)));
      // K stage n % KS free for K(n + KS); only committed when the producer will wait for it, so
      // no commit is left in flight behind the epilogue's last wait (on dQ(steps - 1)).
      if (n + kKS < steps) MmaCommit(XATTN_DQP_BAR(kBarKfree + n % kKS));
      }
      if (m < steps) {
// dP(m): with two buffers its last reader was pack(n) (waited above); with one buffer
        // it is pack(m - 1), which kBarDpFree reports. V(m) has landed in either case.
        if (Pl::kDpBufs == 1) {
          MbarWait(XATTN_DQP_BAR(kBarDpFree), static_cast<uint32_t>((m - 1) & 1));
        }
        wait_v(m);
        __syncwarp();
        issue_dp(m);
        MmaCommit(XATTN_DQP_BAR(kBarVfree + m % kVS));
        // S(m) into S buffer n & 1, where dS(n) lives: dQ(n) must have completed; K(m) landed.
        wait_k(m);
        MbarWait(XATTN_DQP_BAR(kBarDq + (n & 1)), static_cast<uint32_t>((n >> 1) & 1));
        TmemFenceAfterSync();
        __syncwarp();
        issue_s(m);
        MmaCommit(XATTN_DQP_BAR(kBarS + (m & 1)));
      }
    }
    }  // the S / dP issuer
    
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

    if constexpr (Pl::kTmemA && !Pl::kCp) {
      // (kCp off only; under kCp the MMA warp copies the TMA-landed tiles.)
      // Q (warps 0..3) or dO (warps 4..7) of this thread's row into TMEM as the A operand:
      // lane = query row, column j = elements 2j (low half) and 2j + 1. Rows past q_length and
      // columns past the real width are zero, as the TMA boxes' fill was.
      const int width = ch == 0 ? kD : kDv;
      const int64_t real = ch == 0 ? p.qk_dim : p.value_dim;
      const Elem* src = nullptr;
      if (row_in_range) {
        src = ch == 0 ? static_cast<const Elem*>(p.q.ptr) + b * p.q.batch_stride +
                            i * p.q.row_stride + hq * p.q.head_stride
                      : static_cast<const Elem*>(p.dy) +
                            ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
      }
      const uint32_t col0 = static_cast<uint32_t>(ch == 0 ? Pl::kQCol : Pl::kDoCol);
      for (int c32 = 0; c32 < width; c32 += 64) {  // 64 elements = 32 packed columns
        uint32_t w[32];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int c = c32 + 8 * v;
          uint4 u = make_uint4(0u, 0u, 0u, 0u);
          if (src != nullptr) {
            if (c + 8 <= real) {
              u = *reinterpret_cast<const uint4*>(src + c);
            } else if (c < real) {
              union {
                uint4 u4;
                Elem e[8];
              } t;
              for (int x = 0; x < 8; ++x) {
                t.e[x] = c + x < real ? src[c + x] : ElemTraits<Elem>::From(0.f);
              }
              u = t.u4;
            }
          }
          w[4 * v + 0] = u.x;
          w[4 * v + 1] = u.y;
          w[4 * v + 2] = u.z;
          w[4 * v + 3] = u.w;
        }
        TmemStore32(lane_base | (tbase + col0 + static_cast<uint32_t>(c32 / 2)), w);
      }
      TmemWaitSt();
      TmemFenceBeforeSync();
      MbarArrive(XATTN_DQP_BAR(kBarQ));
    } else if constexpr (Pl::kDoTmem) {
      // dO of this thread's row into TMEM as the A operand of dP (TS): warp ch writes the
      // elements ch * kDv / 2 .. of its rows, lane = query row, column j = elements 2j, 2j + 1.
      // Rows past q_length and columns past value_dim are zero, as the TMA box's fill was.
      const int64_t real = p.value_dim;
      const Elem* src = nullptr;
      if (row_in_range) {
        src = static_cast<const Elem*>(p.dy) +
              ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
      }
      // Two warps per row (ch = 0, 1) write the row's dO in 64-element blocks, taken alternately:
      // block b of the row is warp ch = b & 1's, so Dv = 128 is warp 0 blocks 0, 2.. and warp 1
      // blocks 1, 3.., and Dv = 64 - one block, where the per-warp window would be 32 elements -
      // is warp 0's block 0 with warp 1 writing nothing. The block width is the 64-element
      // step of the loop below and the 32 packed TMEM columns of one store.
      constexpr int kDoBlocks = kDv / 64;  // 1 at Dv = 64, 2 at 128, 4 at 256
      static_assert(kDv % 64 == 0 && kDoBlocks >= 1, "dO is staged in 64-element blocks");
      for (int blk = ch; blk < kDoBlocks; blk += 2) {
        const int c32 = blk * 64;
        uint32_t w[32];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int c = c32 + 8 * v;
          uint4 u = make_uint4(0u, 0u, 0u, 0u);
          if (src != nullptr) {
            if (c + 8 <= real) {
              u = *reinterpret_cast<const uint4*>(src + c);
            } else if (c < real) {
              union {
                uint4 u4;
                Elem e[8];
              } t;
              for (int x = 0; x < 8; ++x) {
                t.e[x] = c + x < real ? src[c + x] : ElemTraits<Elem>::From(0.f);
              }
              u = t.u4;
            }
          }
          w[4 * v + 0] = u.x;
          w[4 * v + 1] = u.y;
          w[4 * v + 2] = u.z;
          w[4 * v + 3] = u.w;
        }
        TmemStore32(lane_base | (tbase + static_cast<uint32_t>(Pl::kDoCol + c32 / 2)), w);
      }
      TmemWaitSt();
      TmemFenceBeforeSync();
      MbarArrive(XATTN_DQP_BAR(kBarDy));
    }

    // Ping-pong (Pl::kPingPong): pack group g = warps 4g .. 4g+3 packs the tiles n with
    // n & 1 == g, i.e. always S / dP buffer g, every thread its row's 64 columns as two 32-column
    // halves; the two groups' tiles overlap, so one group's waits (S landed, TMEM loads, stores)
    // run under the other group's arithmetic. Otherwise all 8 warps pack every tile, warp w
    // the 32 columns ch * 32 of its rows.
    constexpr bool kPP = Pl::kPingPong;
    const int grp = kPP ? (warp >> 2) : 0;
    int64_t* const kseg_g = kseg + (kPP ? grp * kBN : 0);
    // The visibility of 32 columns cb .. cb + 31 of tile kb for this thread's row as one bit mask,
    // formed once per tile instead of per element (the same predicate as DqKernel's col_visible,
    // lse_ok folded in): full tile -> all; else the row's [lo, hi] window, then the staged ids.
    auto visibility = [&](int64_t kb, bool full, int cb) -> uint32_t {
      uint32_t vis = 0u;
      if (lse_ok) {
        if (full) {
          vis = 0xFFFFFFFFu;
        } else {
          const int64_t base = kb + cb;
          const int64_t lo_r = static_cast<int64_t>(bd.lo) - base;
          const int64_t hi_r = static_cast<int64_t>(bd.hi) - base;
          if (hi_r >= 0 && lo_r <= 31) {
            const int a = lo_r < 0 ? 0 : static_cast<int>(lo_r);
            const int z = hi_r > 31 ? 31 : static_cast<int>(hi_r);
            vis = (0xFFFFFFFFu >> (31 - z)) & (0xFFFFFFFFu << a);
          }
          if (seg && vis != 0u) {
#pragma unroll 8
            for (int x = 0; x < 32; ++x) {
              if (kseg_g[cb + x] != qseg) vis &= ~(1u << x);
            }
          }
        }
      }
      return vis;
    };
    int cur_n = 0;  // the tile being packed (kDoTmem's dS tile reuse wait reads it)
    // dS of 32 columns cb .. cb + 31 of buffer buf, in place over the (already loaded) S words.
    auto pack_math = [&](uint32_t buf, int cb, uint32_t vis, const float (&s)[32],
                         const float (&dpr)[32]) {
      uint32_t dsv[kHalf / 2];
      // Two columns at a time. `mul/sub.rn.f32x2` are IEEE round-to-nearest-even PER
      // LANE, the same operands in the same order as the scalar forms below, so those change no
      // value. The exponential has no packed form and stays per lane (`FastExp2` is
      // `ex2.approx.ftz.f32`). The exponent argument is one `fma.rn.f32x2` where the scalar chain
      // was an unfused FMUL then FADD, so a lane's dQ may differ in its last bit from that chain:
      // it is deterministic and bit-repeatable, and the frozen contract's FP64 tolerances are
      // what judge it. This round's change also left the file's local `Ex2Ftz` and `PackPairRn`
      // helpers unreferenced, so they are removed here: `pt::FastExp2` and `pt::Pack2Fast` are
      // the same two instructions they were.
      const float2 sl2v = make_float2(sl2, sl2);
      const float2 nlse2 = make_float2(-lse_log2, -lse_log2);
      const float2 dlt2 = make_float2(delta, delta);
      if (__all_sync(0xFFFFFFFFu, vis == 0xFFFFFFFFu)) {
        // Every element of the warp visible (full tiles of live rows): no mask test.
#pragma unroll
        for (int c2 = 0; c2 < kHalf; c2 += 2) {
          const float2 e = pt::FmaX2(make_float2(s[c2], s[c2 + 1]), sl2v, nlse2);
          const float2 p = make_float2(pt::FastExp2(e.x), pt::FastExp2(e.y));
          const float2 d = pt::MulX2(p, pt::SubX2(make_float2(dpr[c2], dpr[c2 + 1]), dlt2));
          dsv[c2 >> 1] = pt::Pack2Fast<Elem>::Of(d.x, d.y);
        }
      } else {
#pragma unroll
        for (int c2 = 0; c2 < kHalf; c2 += 2) {
          // DqKernel's branchless line, the visibility read from the mask.
          const bool v0 = ((vis >> c2) & 1u) != 0u;
          const bool v1 = ((vis >> (c2 + 1)) & 1u) != 0u;
          const float2 e = pt::FmaX2(make_float2(s[c2], s[c2 + 1]), sl2v, nlse2);
          const float2 p = make_float2(pt::FastExp2(e.x), pt::FastExp2(e.y));
          const float2 d = pt::MulX2(p, pt::SubX2(make_float2(dpr[c2], dpr[c2 + 1]), dlt2));
          dsv[c2 >> 1] = pt::Pack2Fast<Elem>::Of(v0 ? d.x : 0.f, v1 ? d.y : 0.f);
        }
      }
      if constexpr (Pl::kDoTmem) {
        // dS(n) into SMEM tile buf (A of dQ(n), K-major, 128 B swizzle as the Q slabs): row
        // `row`, 16 B chunk c of the 128 B row at (c ^ (row & 7)). The tile's previous reader
        // was dQ(n - 2).
        if (cur_n >= 2) {
          MbarWait(XATTN_DQP_BAR(kBarDq + (cur_n & 1)), static_cast<uint32_t>(((cur_n - 2) >> 1) & 1));
        }
        unsigned char* const drow = smem + Pl::kDsOff + buf * Pl::kSlab + row * 128;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int c = (cb >> 3) + q;
          *reinterpret_cast<uint4*>(drow + ((c ^ (row & 7)) << 4)) =
              make_uint4(dsv[4 * q], dsv[4 * q + 1], dsv[4 * q + 2], dsv[4 * q + 3]);
        }
      } else {
        TmemStore16(lane_base | (tbase + DsBase(Pl::kSCol, buf) + static_cast<uint32_t>(cb)), dsv);
      }
    };
    auto pack_cols = [&](uint32_t buf, int cb, uint32_t vis) {
      float s[32];
      float dpr[32];
// One S (kDoTmem) / dP (kDpBufs == 1) buffer is shared by consecutive tiles; otherwise buf.
      const uint32_t sbuf = buf * static_cast<uint32_t>(Pl::kSBufs - 1);
      const uint32_t dpbuf = buf * static_cast<uint32_t>(Pl::kDpBufs - 1);
      TmemLoad32(lane_base | (tbase + Pl::kSCol + sbuf * Pl::kBufCols + cb), s);
      TmemLoad32(lane_base | (tbase + Pl::kDpCol + dpbuf * Pl::kBufCols + cb), dpr);
      TmemWaitLd();
      // kDoTmem: S(n) and dP(n) are in registers, so the MMA warp may issue S(n+1) / dP(n+1).
      if constexpr (Pl::kDoTmem) {
        TmemFenceBeforeSync();
        MbarArrive(XATTN_DQP_BAR(kBarDpFree));
      }
      pack_math(buf, cb, vis, s, dpr);
    };

    for (int n = grp; n < steps; n += kPP ? 2 : 1) {
      cur_n = n;
      const uint32_t buf = static_cast<uint32_t>(n & 1);
      const int64_t kb = static_cast<int64_t>(t0 + n) * kBN;
      const bool full = TileIsFull(seg, kb, max_lo, min_hi);  // CTA-uniform
      if (seg && !full) {
        // DqKernel's key-id staging (thread t < kBN of the packing threads writes key t), between
        // two barriers of the threads that pack this tile: the first orders it after every read
        // of the previous tile staged in this table.
        const int gt = kPP ? tid - 4 * 32 * grp : tid;
        if constexpr (kPP) GroupBarrierSync(grp); else PackBarrierSync();
        if (gt < kBN) {
          const int64_t j = kb + gt;
          kseg_g[gt] = j < nt_total ? p.k_segment[b * nt_total + j] : (qseg + 1);
        }
        if constexpr (kPP) GroupBarrierSync(grp); else PackBarrierSync();
      }
      if constexpr (kPP) {
        const uint32_t vis0 = visibility(kb, full, 0);
        const uint32_t vis1 = visibility(kb, full, kHalf);
        MbarWait(XATTN_DQP_BAR(kBarS + buf), static_cast<uint32_t>((n >> 1) & 1));
        TmemFenceAfterSync();
        pack_cols(buf, 0, vis0);
        pack_cols(buf, kHalf, vis1);
      } else {
        const uint32_t vis = visibility(kb, full, c0);
        MbarWait(XATTN_DQP_BAR(kBarS + buf), static_cast<uint32_t>((n >> 1) & 1));
        TmemFenceAfterSync();
        pack_cols(buf, c0, vis);
      }
      // One dP buffer: every dP word of tile n this thread reads is in its registers now
      // (each pack_cols waited its loads), so the MMA warp may overwrite the buffer. Once per
      // tile per packing thread, kPackArrivals in all.
      if constexpr (Pl::kDoTmem) {
        // (kBarDpFree arrived in pack_cols.) dS(n)'s generic SMEM stores before the async
        // proxy's (the dQ MMA's) reads.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      } else {
        if (Pl::kDpBufs == 1) MbarArrive(XATTN_DQP_BAR(kBarDpFree));
        TmemWaitSt();
        TmemFenceBeforeSync();
      }
      MbarArrive(XATTN_DQP_BAR(kBarPack + buf));
    }

    // ----------------------------------------------------------------- epilogue
    // DqKernel's store: lse_ok ? acc * p.scale : 0 for every row in range (a live row with an
    // empty visible set and a dead row both store exact zeros), scale applied once, here. Warps
    // 0..3 drain columns 0..127, warps 4..7 columns 128..255 of the same rows.
    // Under ping-pong the group whose last tile L is steps - 2 gets here while dQ(steps - 3) may
    // still be in flight: a parity wait on dQ(steps - 1)'s barrier would then match the phase of
    // dQ(steps - 3) one phase early. So it first waits its own dQ(L) (whose barrier's previous
    // phase, dQ(L - 2), completed before S(L) was issued); dQ(steps - 3) was issued before dQ(L)
    // and has completed with it.
    if (kPP) {
      const int last_own = ((steps - 1 - grp) & ~1) + grp;  // largest n <= steps-1 with n&1 == grp
      if (last_own >= 0 && last_own != steps - 1) {
        MbarWait(XATTN_DQP_BAR(kBarDq + (last_own & 1)),
                 static_cast<uint32_t>((last_own >> 1) & 1));
      }
    }
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
  // End of the tile body. This __syncthreads is load-bearing beyond the fences: every thread's last
  // wait on a barrier of this tile happens before it, and the next iteration's mbarrier re-init
  // happens after it, so a re-init can never race a wait still outstanding on the same barrier.
  }  // persistent tile loop
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
  
#undef XATTN_DQP_BAR
}

// ------------------------------------------------------------------ the flat persistent form
// DqFlatKernel: the same leaf, the same tiles, the same MMAs per tile in the same order with the
// same operands (dQ bit-equal), at the maps with Q / dO resident in TMEM from TMA landing tiles,
// two pack groups and a dQ issuer warp (Plan::kTmemA && kCp && kPingPong && kDqIssuer: (128,128)
// and (128,64)). What changes is that consecutive tiles of a CTA overlap instead of running one
// after the other: DqPipeKernel re-initializes its barriers, recomputes the tile's row bounds
// behind two CTA barriers, loads Q / dO and fills the K / V rings only after the previous tile's
// epilogue, so each tile pays a fill and a drain (several microseconds against ~0.4 us per key
// step), which is most of a tile at short-document masks. Here:
//   - barriers are initialized once; every ring (K, V, S / dP buffers, pack, dQ) counts GLOBAL
//     key steps g over the CTA's live tiles, so their phases run on across tiles;
//   - warp 11 (the scheduler) claims tiles from the counter and computes each tile's bounds
//     (min lo, max hi, max lo, min hi over its 128 rows: DqPipeKernel's reductions) into a
//     two-slot ring of descriptors, one tile ahead, claimed once the S / dP issuer is within
//     FlatPlan::kLook steps of the current tile's end;
//   - the producer issues Q / dO of tile T + 1 as soon as the copies of tile T into TMEM are done
//     (kQFree) and runs its K / V streams on into tile T + 1 while tile T drains;
//   - the S / dP issuer copies tile T + 1's Q / dO into TMEM once every S / dP of tile T has
//     completed, then issues S / dP(T + 1, 0..1) under the pack of tile T's last steps;
//   - dQ(T + 1, 0) (the accumulator reset) waits only for the epilogue of tile T to have read
//     the dQ columns (kDqDrained).
// The pack warps read the row constants of tile T + 1 (lse, delta, bounds) while they wait for
// dQ(T) to complete.
template <int kD, int kDv>
struct FlatPlan {
  using Pl = Plan<kD, kDv>;
  static constexpr bool kEligible = Pl::kTmemA && Pl::kCp && Pl::kPingPong && Pl::kDqIssuer;
  static constexpr int kSchedWarp = Pl::kThreads / 32;            // 11
  static constexpr int kThreads = Pl::kThreads + 32;              // 384
  // Extra barriers and the descriptor ring live after the plan's tail, inside its rounding.
  static constexpr int kXBarOff = ((Pl::kEnd + 7) / 8) * 8;
  static constexpr int kBarQFree = 0, kBarDrained = 1, kBarDescFull = 2, kBarDescEmpty = 4,
                       kBarFinal = 6, kXBarCount = 8;
  static constexpr int kDescOff = kXBarOff + 8 * kXBarCount;      // 2 x 32 B
  static constexpr int kProgOff = kDescOff + 64;                 // u32: S/dP steps issued
  static constexpr int kEnd = kProgOff + 8;
  // The scheduler claims tile it + 1 once the issuer is within kLook steps of tile it's end:
  // early enough to hide the claim and the bounds reduction, late enough that a claimed tile
  // does not wait behind a long one while other SMs run dry at the grid's tail.
  static constexpr uint32_t kLook = 4;
  static constexpr int kDescEmptyCount = 3 + kPackWarps;         // producer, 2 issuers, 8 pack warps
  static_assert(!kEligible || ((kEnd + 1023) / 1024) * 1024 + 1024 <= Pl::kSmemBytes,
                "the flat plan's extra barriers and descriptors fit the plan's SMEM image");
};

struct FlatDesc {
  int32_t tile, t0, steps, min_lo, max_hi, max_lo, min_hi, pad;
};

template <class Elem, int kD, int kDv>
__global__ void __launch_bounds__(FlatPlan<kD, kDv>::kThreads, 1) DqFlatKernel(
    const __grid_constant__ UmmaLaunchParams lp, unsigned int* const tile_counter,
    const int total_tiles) {
  using Pl = Plan<kD, kDv>;
  using Fp = FlatPlan<kD, kDv>;
  static_assert(Fp::kEligible, "the flat form serves the resident-Q/dO ping-pong maps only");
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const smem = smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);
  unsigned char* const sq = smem + Pl::kQOff;
  unsigned char* const sdy = smem + Pl::kDyOff;
  unsigned char* const sk = smem + Pl::kKOff;
  unsigned char* const sv = smem + Pl::kVOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  const uint32_t xbar0 = SmemAddr(smem + Fp::kXBarOff);
  FlatDesc* const descs = reinterpret_cast<FlatDesc*>(smem + Fp::kDescOff);
  volatile uint32_t* const sdp_issued = reinterpret_cast<volatile uint32_t*>(smem + Fp::kProgOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int64_t* const kseg = reinterpret_cast<int64_t*>(smem + Pl::kSegOff);
#define XATTN_DQF_BAR(slot) (bar0 + 8u * static_cast<uint32_t>(slot))
#define XATTN_DQF_XBAR(slot) (xbar0 + 8u * static_cast<uint32_t>(slot))
  constexpr int kKS = Pl::kKStages, kVS = Pl::kVStages;
  constexpr int kBarQ = Pl::kBarQ, kBarDy = Pl::kBarDy, kBarK = Pl::kBarK, kBarV = Pl::kBarV;
  constexpr int kBarVfree = Pl::kBarVfree, kBarS = Pl::kBarS, kBarDq = Pl::kBarDq;
  constexpr int kBarPack = Pl::kBarPack, kBarKfree = Pl::kBarKfree, kBarCount = Pl::kBarCount;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int w4 = warp & 3;
  const int ch = (warp >> 2) & 1;
  const int row = 32 * w4 + lane;
  const int nqt = (p.q_length + kBM - 1) / kBM;
  const int64_t nhq = p.query_heads;
  const int64_t tiles_per_batch = static_cast<int64_t>(nqt) * nhq;

  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) {
    for (int s = 0; s < kBarCount; ++s) {
      MbarInit(XATTN_DQF_BAR(s),
               (s == kBarPack || s == kBarPack + 1) ? Pl::kPackArrivals : 1);
    }
    for (int s = 0; s < Fp::kXBarCount; ++s) {
      int count = 1;
      if (s == Fp::kBarDrained) count = kPackThreads;
      if (s == Fp::kBarDescEmpty || s == Fp::kBarDescEmpty + 1) count = Fp::kDescEmptyCount;
      MbarInit(XATTN_DQF_XBAR(s), count);
    }
    MbarInitFence();
    *sdp_issued = 0u;
  }
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  const uint32_t tbase = *tmem_slot & 0x1FFu;

  // Every role but the scheduler walks the descriptor ring the same way.
  auto read_desc = [&](int it, FlatDesc* d) {
    const int slot = it & 1;
    MbarWait(XATTN_DQF_XBAR(Fp::kBarDescFull + slot), static_cast<uint32_t>((it >> 1) & 1));
    *d = descs[slot];
  };
  // One arrival per consuming warp (its lane 0; the producer runs on lane 0 only).
  auto release_desc = [&](int it) {
    __syncwarp(__activemask());
    if (lane == 0) MbarArrive(XATTN_DQF_XBAR(Fp::kBarDescEmpty + (it & 1)));
  };
  auto decode = [&](int tile, int* qt, int64_t* hq, int64_t* b) {
    *qt = tile % nqt;
    *hq = (tile / nqt) % nhq;
    *b = tile / tiles_per_batch;
  };

  if (warp == Fp::kSchedWarp) {
    // ------------------------------------------------------------- scheduler
    uint32_t g_end = 0;  // S/dP steps through the last tile described
    for (int it = 0;; ++it) {
      const int slot = it & 1;
      if (it >= 2) {
        MbarWait(XATTN_DQF_XBAR(Fp::kBarDescEmpty + slot), static_cast<uint32_t>(((it - 2) >> 1) & 1));
      }
      if (it >= 1 && g_end > Fp::kLook) {
        // A scheduling hint only: which tile an SM takes changes no tile's bits.
        while (*sdp_issued + Fp::kLook < g_end) __nanosleep(64);
      }
      int tile = 0;
      if (lane == 0) tile = static_cast<int>(atomicAdd(tile_counter, 1u));
      tile = __shfl_sync(0xFFFFFFFFu, tile, 0);
      FlatDesc d{};
      d.tile = tile < total_tiles ? tile : -1;
      if (tile < total_tiles) {
        int qt;
        int64_t hq, b;
        decode(tile, &qt, &hq, &b);
        int lo = INT_MAX, hi = -1, mlo = INT_MIN, mhi = INT_MAX;
        bool any_dead = false;
        for (int k = 0; k < 4; ++k) {
          const int64_t i = static_cast<int64_t>(qt) * kBM + 32 * k + lane;
          const bool in = i < p.q_length;
          const detail::Bounds bd = in ? detail::RowBounds(p, b, i) : detail::Bounds{INT_MAX, -1};
          const bool live = in && bd.lo <= bd.hi;
          lo = min(lo, live ? bd.lo : INT_MAX);
          hi = max(hi, live ? bd.hi : -1);
          // DqPipeKernel's max lo / min hi: a dead or absent row counts as INT_MAX / INT_MIN.
          mlo = max(mlo, live ? bd.lo : INT_MAX);
          mhi = min(mhi, live ? bd.hi : INT_MIN);
          any_dead |= !live;
        }
        (void)any_dead;
        d.min_lo = WarpMin(lo);
        d.max_hi = WarpMax(hi);
        d.max_lo = WarpMax(mlo);
        d.min_hi = WarpMin(mhi);
        if (d.min_lo > d.max_hi) {
          d.steps = 0;
          d.t0 = 0;
        } else {
          d.t0 = d.min_lo / kBN;
          d.steps = d.max_hi / kBN - d.t0 + 1;
        }
      }
      if (d.tile >= 0) g_end += static_cast<uint32_t>(d.steps);
      if (lane == 0) descs[slot] = d;
      __syncwarp();
      if (lane == 0) MbarArrive(XATTN_DQF_XBAR(Fp::kBarDescFull + slot));
      if (d.tile < 0) break;
    }
  } else if (warp == kProducerWarp) {
    // ------------------------------------------------------------- TMA producer
    if (lane == 0) {
      uint32_t gk = 0, gv = 0;
      int live_i = 0;
      for (int it = 0;; ++it) {
        FlatDesc d;
        read_desc(it, &d);
        release_desc(it);
        if (d.tile < 0) break;
        if (d.steps == 0) continue;
        int qt;
        int64_t hq, b;
        decode(d.tile, &qt, &hq, &b);
        const int64_t kvh = p.group > 0 ? hq / p.group : hq;
        // The landing tiles are free once the previous live tile's copies into TMEM are done.
        if (live_i >= 1) MbarWait(XATTN_DQF_XBAR(Fp::kBarQFree), static_cast<uint32_t>((live_i - 1) & 1));
        MbarArriveExpectTx(XATTN_DQF_BAR(kBarQ), Pl::kQBytes);
        for (int s = 0; s < Pl::kSlabsQ; ++s) {
          for (int r = 0; r < kBM; r += kTmaRows) {
            TmaLoad4D(sq + s * Pl::kSlab + r * 128, &lp.maps.q, XATTN_DQF_BAR(kBarQ), s * 64,
                      static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
          }
        }
        MbarArriveExpectTx(XATTN_DQF_BAR(kBarDy), Pl::kDyBytes);
        for (int s = 0; s < Pl::kSlabsV; ++s) {
          for (int r = 0; r < kBM; r += kTmaRows) {
            TmaLoad4D(sdy + s * Pl::kSlab + r * 128, &lp.maps.dy, XATTN_DQF_BAR(kBarDy), s * 64,
                      static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
          }
        }
        int nk = 0, nv = 0;
        uint32_t idle = 0;
        while (nk < d.steps || nv < d.steps) {
          bool moved = false;
          const uint32_t g = gk + nk;
          if (nk < d.steps &&
              (g < static_cast<uint32_t>(kKS) ||
               MbarTestWait(XATTN_DQF_BAR(kBarKfree + g % kKS), ((g - kKS) / kKS) & 1u))) {
            const int64_t j0 = static_cast<int64_t>(d.t0 + nk) * kBN;
            const CUtensorMap* kmap;
            const CUtensorMap* vmap;
            int row0;
            ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);
            const uint32_t bk = XATTN_DQF_BAR(kBarK + g % kKS);
            MbarArriveExpectTx(bk, Pl::kKStageBytes);
            unsigned char* const kdst = sk + (g % kKS) * Pl::kKStageBytes;
            for (int s = 0; s < Pl::kSlabsK; ++s) {
              for (int r = 0; r < kBN; r += kTmaRows) {
                TmaLoad4D(kdst + s * Pl::kKvSlab + r * 128, kmap, bk, s * 64,
                          static_cast<int>(kvh), row0 + r, static_cast<int>(b));
              }
            }
            ++nk;
            moved = true;
          }
          const uint32_t gvv = gv + nv;
          if (nv < d.steps &&
              (gvv < static_cast<uint32_t>(kVS) ||
               MbarTestWait(XATTN_DQF_BAR(kBarVfree + gvv % kVS), ((gvv - kVS) / kVS) & 1u))) {
            const int64_t j0 = static_cast<int64_t>(d.t0 + nv) * kBN;
            const CUtensorMap* kmap;
            const CUtensorMap* vmap;
            int row0;
            ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);
            const uint32_t bv = XATTN_DQF_BAR(kBarV + gvv % kVS);
            unsigned char* const vdst = sv + (gvv % kVS) * Pl::kVStageBytes;
            MbarArriveExpectTx(bv, Pl::kVBytes);
            for (int s = 0; s < Pl::kSlabsV; ++s) {
              for (int r = 0; r < kBN; r += kTmaRows) {
                TmaLoad4D(vdst + s * Pl::kKvSlab + r * 128, vmap, bv, s * 64,
                          static_cast<int>(kvh), row0 + r, static_cast<int>(b));
              }
            }
            ++nv;
            moved = true;
          }
          if (moved) {
            idle = 0;
          } else {
            if (++idle > (1u << 22)) __trap();
            __nanosleep(20);
          }
        }
        gk += d.steps;
        gv += d.steps;
        ++live_i;
      }
    }
    __syncwarp();
  } else if (warp == kMmaWarp || warp == kMmaWarp + 1) {
    const bool dq_role = warp == kMmaWarp + 1;
    const int elem_fmt = ElemTraits<Elem>::kFormat;
    const uint32_t ss_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, 0, 0);
    const uint32_t dq_lbo = Pl::kSlabsK > 1 ? static_cast<uint32_t>(kBN * 8) : 0u;
    const uint32_t dq_idesc = MakeInstrDesc(elem_fmt, kBM, kD <= 256 ? kD : 256, 0, 1);
    const uint32_t qb_smem = SmemAddr(sq);
    const uint32_t dyb_smem = SmemAddr(sdy);
    const uint32_t k0_smem = SmemAddr(sk);
    const uint32_t v0_smem = SmemAddr(sv);
    // DqPipeKernel's issue bodies at the resident-Q/dO map, indexed by the GLOBAL step g.
    auto issue_s = [&](uint32_t g) {
      const uint32_t kb_smem = k0_smem + (g % kKS) * Pl::kKStageBytes;
      const uint32_t dst = tbase + Pl::kSCol + (g & 1u) * Pl::kBufCols;
      for (int kk = 0; kk < kD / 16; ++kk) {
        const int slab = kk >> 2, within = kk & 3;
        MmaTs(dst, tbase + static_cast<uint32_t>(Pl::kQCol + 8 * kk),
              MakeSmemDesc(kb_smem + slab * Pl::kKvSlab + within * 32), ss_idesc,
              kk == 0 ? 0u : 1u);
      }
    };
    auto issue_dp = [&](uint32_t g) {
      const uint32_t dst = tbase + Pl::kDpCol + (g & 1u) * Pl::kBufCols;
      const uint32_t vb_smem = v0_smem + (g % kVS) * Pl::kVStageBytes;
      for (int sl = 0; sl < Pl::kSlabsV; ++sl) {
        for (int kk = 0; kk < 4; ++kk) {
          MmaTs(dst, tbase + static_cast<uint32_t>(Pl::kDoCol + 8 * (4 * sl + kk)),
                MakeSmemDesc(vb_smem + sl * Pl::kKvSlab + kk * 32), ss_idesc,
                (sl == 0 && kk == 0) ? 0u : 1u);
        }
      }
    };
    auto issue_dq = [&](uint32_t g, bool first) {
      const uint32_t kb_smem = k0_smem + (g % kKS) * Pl::kKStageBytes;
      for (int kk = 0; kk < kBN / 16; ++kk) {
        MmaTs(tbase + Pl::kDqCol, tbase + DsColumn(Pl::kSCol, g & 1u, kk),
              MakeSmemDesc(kb_smem + kk * 16 * 128, dq_lbo, 64), dq_idesc,
              (first && kk == 0) ? 0u : 1u);
      }
    };
    auto wait_v = [&](uint32_t g) {
      MbarWait(XATTN_DQF_BAR(kBarV + g % kVS), (g / kVS) & 1u);
    };
    auto wait_k = [&](uint32_t g) {
      MbarWait(XATTN_DQF_BAR(kBarK + g % kKS), (g / kKS) & 1u);
    };
    uint32_t G = 0;
    int live_i = 0;
    if (dq_role) {
      for (int it = 0;; ++it) {
        FlatDesc d;
        read_desc(it, &d);
        release_desc(it);
        if (d.tile < 0) break;
        if (d.steps == 0) continue;
        for (int n = 0; n < d.steps; ++n) {
          const uint32_t g = G + n;
          MbarWait(XATTN_DQF_BAR(kBarPack + (g & 1u)), (g >> 1) & 1u);
          if (n == 0 && live_i >= 1) {
            // The accumulator's previous tile has been read out by the epilogue.
            MbarWait(XATTN_DQF_XBAR(Fp::kBarDrained), static_cast<uint32_t>((live_i - 1) & 1));
          }
          TmemFenceAfterSync();
          __syncwarp();
          issue_dq(g, n == 0);
          MmaCommit(XATTN_DQF_BAR(kBarDq + (g & 1u)));
          MmaCommit(XATTN_DQF_BAR(kBarKfree + g % kKS));
        }
        G += d.steps;
        ++live_i;
      }
    } else {
      int64_t last_s = -1;  // global step of the last S / dP issued
      for (int it = 0;; ++it) {
        FlatDesc d;
        read_desc(it, &d);
        release_desc(it);
        if (d.tile < 0) break;
        if (d.steps == 0) continue;
        // Q / dO columns are read by every S / dP of the previous tile: all complete.
        if (last_s >= 0) {
          MbarWait(XATTN_DQF_BAR(kBarS + (last_s & 1)), static_cast<uint32_t>((last_s >> 1) & 1));
        }
        MbarWait(XATTN_DQF_BAR(kBarQ), static_cast<uint32_t>(live_i & 1));
        MbarWait(XATTN_DQF_BAR(kBarDy), static_cast<uint32_t>(live_i & 1));
        TmemFenceAfterSync();
        __syncwarp();
        for (int kk = 0; kk < kD / 16; ++kk) {
          TmemCopy128x256b(tbase + static_cast<uint32_t>(Pl::kQCol + 8 * kk),
                           MakeSmemDesc(qb_smem + (kk >> 2) * Pl::kSlab + (kk & 3) * 32));
        }
        for (int kk = 0; kk < kDv / 16; ++kk) {
          TmemCopy128x256b(tbase + static_cast<uint32_t>(Pl::kDoCol + 8 * kk),
                           MakeSmemDesc(dyb_smem + (kk >> 2) * Pl::kSlab + (kk & 3) * 32));
        }
        MmaCommit(XATTN_DQF_XBAR(Fp::kBarQFree));
        // S / dP(g) for the tile's steps; the buffers' previous readers are pack(g - 2) (dP)
        // and dQ(g - 2) (S, where dS(g - 2) lives), in this tile or the previous one.
        auto issue_sdp = [&](uint32_t g) {
          if (g >= 2) {
            MbarWait(XATTN_DQF_BAR(kBarPack + (g & 1u)), ((g - 2) >> 1) & 1u);
            TmemFenceAfterSync();
          }
          wait_v(g);
          __syncwarp();
          issue_dp(g);
          MmaCommit(XATTN_DQF_BAR(kBarVfree + g % kVS));
          wait_k(g);
          if (g >= 2) MbarWait(XATTN_DQF_BAR(kBarDq + (g & 1u)), ((g - 2) >> 1) & 1u);
          TmemFenceAfterSync();
          __syncwarp();
          issue_s(g);
          MmaCommit(XATTN_DQF_BAR(kBarS + (g & 1u)));
        };
        for (int n = 0; n < d.steps; ++n) {
          issue_sdp(G + n);
          if (lane == 0) *sdp_issued = G + n + 1;
        }
        last_s = static_cast<int64_t>(G) + d.steps - 1;
        G += d.steps;
        ++live_i;
      }
    }
    // Every commit of this warp has arrived before the CTA can exit.
    MmaCommit(XATTN_DQF_XBAR(Fp::kBarFinal + (dq_role ? 1 : 0)));
    MbarWait(XATTN_DQF_XBAR(Fp::kBarFinal + (dq_role ? 1 : 0)), 0u);
  } else {
    // ------------------------------------------------------------- pack warps 0..7
    const uint32_t lane_base = static_cast<uint32_t>(32 * w4) << 16;
    const int grp = warp >> 2;
    int64_t* const kseg_g = kseg + grp * kBN;
    const float sl2 = p.scale_log2;
    const void* const saved_base =
        p.saved != nullptr ? p.saved
                           : (p.saved_fp32 ? static_cast<const void*>(p.y_state)
                                           : static_cast<const void*>(p.y));
    // The row constants of one tile (DqPipeKernel's expressions).
    struct RowC {
      int64_t b, hq, i;
      bool in, lse_ok, seg;
      detail::Bounds bd;
      float lse_log2, delta;
      int64_t qseg;
    };
    auto row_consts = [&](const FlatDesc& d) {
      RowC r;
      int qt;
      decode(d.tile, &qt, &r.hq, &r.b);
      r.i = static_cast<int64_t>(qt) * kBM + row;
      r.in = r.i < p.q_length;
      // The row's loads are issued independently of each other (the values they feed are the
      // same as DqPipeKernel's chained form: lse and delta are only used where the row is live
      // and its lse finite), so their latencies overlap instead of adding up.
      const int64_t stat = (r.b * p.query_heads + r.hq) * p.q_length + r.i;
      const float lse_raw = (r.in && p.lse_in != nullptr) ? p.lse_in[stat] : -INFINITY;
      const float delta_raw = (r.in && p.delta != nullptr) ? p.delta[stat] : 0.f;
      r.seg = detail::SegmentTest(p, r.b);
      r.qseg = (r.seg && r.in) ? p.q_segment[r.b * p.q_length + r.i] : 0;
      r.bd = r.in ? detail::RowBounds(p, r.b, r.i) : detail::Bounds{INT_MAX, -1};
      const bool live = r.in && r.bd.lo <= r.bd.hi;
      const float lse_i = live ? lse_raw : -INFINITY;
      r.lse_ok = live && lse_i > -INFINITY && lse_i < INFINITY;
      r.lse_log2 = r.lse_ok ? lse_i * kDqLog2e : INFINITY;
      r.delta = 0.f;
      if (r.lse_ok && p.delta != nullptr) {
        r.delta = delta_raw;
      } else if (r.lse_ok) {
        const int64_t row0 = ((r.b * p.q_length + r.i) * p.query_heads + r.hq) * p.value_dim;
        const Elem* const dy_row = reinterpret_cast<const Elem*>(p.dy) + row0;
        float acc = 0.f;
        if (p.saved_fp32) {
          const float* const y_row = static_cast<const float*>(saved_base) + row0;
          for (int64_t c = 0; c < p.value_dim; ++c) acc += ToFloat(dy_row[c]) * y_row[c];
        } else {
          const Elem* const y_row = static_cast<const Elem*>(saved_base) + row0;
          for (int64_t c = 0; c < p.value_dim; ++c) acc += ToFloat(dy_row[c]) * ToFloat(y_row[c]);
        }
        r.delta = acc;
      }
      return r;
    };
    uint32_t G = 0;
    int live_i = 0;
    const int64_t nt_total = p.total_length;
    // The pack warps release a descriptor slot only when its tile is done (below, and on the dead
    // path), so the scheduler claims at most one tile ahead of the tile being packed: claiming
    // further ahead would lengthen the grid's tail on uniform heavy masks. The next tile's
    // descriptor and row constants are read during the current tile (`prefetch`), so their
    // global-load latency is off the path between two tiles.
    FlatDesc d;
    read_desc(0, &d);
    RowC rc{};
    if (d.tile >= 0) rc = row_consts(d);
    for (int it = 0;; ++it) {
      if (d.tile < 0) break;
      FlatDesc dn{};
      RowC rcn{};
      bool pre = false;
      auto prefetch = [&](bool block) {
        if (pre) return;
        const int nslot = (it + 1) & 1;
        const uint32_t nbar = XATTN_DQF_XBAR(Fp::kBarDescFull + nslot);
        const uint32_t nph = static_cast<uint32_t>(((it + 1) >> 1) & 1);
        const bool ready = __all_sync(0xFFFFFFFFu, MbarTestWait(nbar, nph));
        if (!ready && !block) return;
        read_desc(it + 1, &dn);
        if (dn.tile >= 0) rcn = row_consts(dn);
        pre = true;
      };
      if (d.steps == 0) {
        // No live row in this tile: dQ = 0 for every row in range (DqPipeKernel's dead tile).
        if (rc.in) {
          Elem* dqrow = reinterpret_cast<Elem*>(p.dq) +
                        ((rc.b * p.q_length + rc.i) * p.query_heads + rc.hq) * p.qk_dim;
          const int64_t half = (p.qk_dim + 1) / 2;
          const int64_t c_begin = ch == 0 ? 0 : half;
          const int64_t c_end = ch == 0 ? half : p.qk_dim;
          for (int64_t c = c_begin; c < c_end; ++c) dqrow[c] = ElemTraits<Elem>::From(0.f);
        }
        prefetch(true);
        release_desc(it);
        d = dn;
        rc = rcn;
        continue;
      }
      auto visibility = [&](int64_t kb, bool full, int cb) -> uint32_t {
        uint32_t vis = 0u;
        if (rc.lse_ok) {
          if (full) {
            vis = 0xFFFFFFFFu;
          } else {
            const int64_t base = kb + cb;
            const int64_t lo_r = static_cast<int64_t>(rc.bd.lo) - base;
            const int64_t hi_r = static_cast<int64_t>(rc.bd.hi) - base;
            if (hi_r >= 0 && lo_r <= 31) {
              const int a = lo_r < 0 ? 0 : static_cast<int>(lo_r);
              const int z = hi_r > 31 ? 31 : static_cast<int>(hi_r);
              vis = (0xFFFFFFFFu >> (31 - z)) & (0xFFFFFFFFu << a);
            }
            if (rc.seg && vis != 0u) {
#pragma unroll 8
              for (int x = 0; x < 32; ++x) {
                if (kseg_g[cb + x] != rc.qseg) vis &= ~(1u << x);
              }
            }
          }
        }
        return vis;
      };
      auto pack_cols = [&](uint32_t buf, int cb, uint32_t vis) {
        float s[32];
        float dpr[32];
        TmemLoad32(lane_base | (tbase + Pl::kSCol + buf * Pl::kBufCols + cb), s);
        TmemLoad32(lane_base | (tbase + Pl::kDpCol + buf * Pl::kBufCols + cb), dpr);
        TmemWaitLd();
        uint32_t dsv[kHalf / 2];
        const float2 sl2v = make_float2(sl2, sl2);
        const float2 nlse2 = make_float2(-rc.lse_log2, -rc.lse_log2);
        const float2 dlt2 = make_float2(rc.delta, rc.delta);
        if (__all_sync(0xFFFFFFFFu, vis == 0xFFFFFFFFu)) {
#pragma unroll
          for (int c2 = 0; c2 < kHalf; c2 += 2) {
            const float2 e = pt::FmaX2(make_float2(s[c2], s[c2 + 1]), sl2v, nlse2);
            const float2 pp = make_float2(pt::FastExp2(e.x), pt::FastExp2(e.y));
            const float2 dd = pt::MulX2(pp, pt::SubX2(make_float2(dpr[c2], dpr[c2 + 1]), dlt2));
            dsv[c2 >> 1] = pt::Pack2Fast<Elem>::Of(dd.x, dd.y);
          }
        } else {
#pragma unroll
          for (int c2 = 0; c2 < kHalf; c2 += 2) {
            const bool v0 = ((vis >> c2) & 1u) != 0u;
            const bool v1 = ((vis >> (c2 + 1)) & 1u) != 0u;
            const float2 e = pt::FmaX2(make_float2(s[c2], s[c2 + 1]), sl2v, nlse2);
            const float2 pp = make_float2(pt::FastExp2(e.x), pt::FastExp2(e.y));
            const float2 dd = pt::MulX2(pp, pt::SubX2(make_float2(dpr[c2], dpr[c2 + 1]), dlt2));
            dsv[c2 >> 1] = pt::Pack2Fast<Elem>::Of(v0 ? dd.x : 0.f, v1 ? dd.y : 0.f);
          }
        }
        TmemStore16(lane_base | (tbase + DsBase(Pl::kSCol, buf) + static_cast<uint32_t>(cb)), dsv);
      };
      // This group's steps of the tile: global g with g & 1 == grp.
      const int n0 = static_cast<int>((static_cast<uint32_t>(grp) - (G & 1u)) & 1u);
      for (int n = n0; n < d.steps; n += 2) {
        const uint32_t g = G + n;
        const uint32_t buf = g & 1u;
        const int64_t kb = static_cast<int64_t>(d.t0 + n) * kBN;
        const bool full = TileIsFull(rc.seg, kb, d.max_lo, d.min_hi);
        if (rc.seg && !full) {
          const int gt = tid - 4 * 32 * grp;
          GroupBarrierSync(grp);
          if (gt < kBN) {
            const int64_t j = kb + gt;
            kseg_g[gt] = j < nt_total ? p.k_segment[rc.b * nt_total + j] : (rc.qseg + 1);
          }
          GroupBarrierSync(grp);
        }
        const uint32_t vis0 = visibility(kb, full, 0);
        const uint32_t vis1 = visibility(kb, full, kHalf);
        MbarWait(XATTN_DQF_BAR(kBarS + buf), (g >> 1) & 1u);
        TmemFenceAfterSync();
        pack_cols(buf, 0, vis0);
        pack_cols(buf, kHalf, vis1);
        TmemWaitSt();
        TmemFenceBeforeSync();
        MbarArrive(XATTN_DQF_BAR(kBarPack + buf));
        prefetch(false);
      }
      prefetch(true);
      // ----------------------------------------------------------------- epilogue
      // dQ(last - 1) (when the tile has it) and then dQ(last): see the parity argument in the
      // flat form's header comment (each wait's previous phase completed before it is taken).
      const uint32_t gl = G + d.steps - 1;
      if (d.steps >= 2) {
        MbarWait(XATTN_DQF_BAR(kBarDq + ((gl - 1) & 1u)), ((gl - 1) >> 1) & 1u);
      }
      MbarWait(XATTN_DQF_BAR(kBarDq + (gl & 1u)), (gl >> 1) & 1u);
      TmemFenceAfterSync();
      float o[kD / 2];
      const int cbeg = ch * (kD / 2);
#pragma unroll
      for (int cc = 0; cc < kD / 2; cc += 32) {
        float t[32];
        TmemLoad32(lane_base | (tbase + Pl::kDqCol + cbeg + cc), t);
        TmemWaitLd();
#pragma unroll
        for (int x = 0; x < 32; ++x) o[cc + x] = t[x];
      }
      TmemFenceBeforeSync();
      MbarArrive(XATTN_DQF_XBAR(Fp::kBarDrained));
      release_desc(it);
      if (rc.in) {
        Elem* const dqrow =
            reinterpret_cast<Elem*>(p.dq) + ((rc.b * p.q_length + rc.i) * p.query_heads + rc.hq) * p.qk_dim;
        const float scale = p.scale;
        for (int cc = 0; cc < kD / 2; cc += 32) {
          const int col0 = cbeg + cc;
          if (col0 + 32 <= p.qk_dim && (reinterpret_cast<uintptr_t>(dqrow + col0) & 15u) == 0u) {
            union {
              uint4 u4[4];
              Elem e[32];
            } pk;
#pragma unroll
            for (int x = 0; x < 32; ++x) pk.e[x] = ElemTraits<Elem>::From(rc.lse_ok ? o[cc + x] * scale : 0.f);
            uint4* const dst = reinterpret_cast<uint4*>(dqrow + col0);
            dst[0] = pk.u4[0];
            dst[1] = pk.u4[1];
            dst[2] = pk.u4[2];
            dst[3] = pk.u4[3];
          } else {
            for (int x = 0; x < 32; ++x) {
              const int64_t col = col0 + x;
              if (col >= p.qk_dim) break;
              dqrow[col] = ElemTraits<Elem>::From(rc.lse_ok ? o[cc + x] * scale : 0.f);
            }
          }
        }
      }
      G += d.steps;
      ++live_i;
      d = dn;
      rc = rcn;
    }
  }

  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
#undef XATTN_DQF_BAR
#undef XATTN_DQF_XBAR
}

// XATTN_SCA_DQ_FLAT: "0" keeps DqPipeKernel at the maps the flat form serves (a dev A/B switch,
// read per call; unset in every gate check, so the gate runs the flat form).
inline bool DqFlatEnabled() {
  const char* v = std::getenv("XATTN_SCA_DQ_FLAT");
  return v == nullptr || v[0] != '0';
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

// The persistent leaf's tile counter and its CTA count.
// The counter is reset on the call's stream before the launch (one 4-byte memset, no host wait).
//
// One counter per (device, stream) pair, and never shared: a call resets its counter and its grid
// consumes it, so two calls that can overlap must not draw from the same word. Keying by the pair
// both prevents that and fixes the device: an allocation is made on the device the call runs on and
// is only ever handed back to a call on that same device, so one process driving two GPUs never
// passes a pointer from one device's context to the other's kernel. Slots are kept for the life of
// the process (4 bytes each) and looked up by the pair, so nothing is ever reused and no bound on
// concurrent calls has to be argued. Reset here rather than in the preprocessing launch: this file
// launches nothing else, and the preprocessing kernel belongs to the metadata line.
inline unsigned int* DqPipeTileCounter(cudaStream_t stream) {
  static std::mutex ring_mutex;
  static std::map<std::pair<int, cudaStream_t>, unsigned int*> ring;
  int device = 0;
  const cudaError_t d = cudaGetDevice(&device);
  if (d != cudaSuccess) dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: cudaGetDevice", d);
  unsigned int* p = nullptr;
  {
    const std::lock_guard<std::mutex> lock(ring_mutex);
    unsigned int*& slot = ring[std::make_pair(device, stream)];
    if (slot == nullptr) {
      const cudaError_t a = cudaMalloc(reinterpret_cast<void**>(&slot), sizeof(unsigned int));
      if (a != cudaSuccess) dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: tile counter", a);
    }
    p = slot;
  }
  const cudaError_t z = cudaMemsetAsync(p, 0, sizeof(unsigned int), stream);
  if (z != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: tile counter reset", z);
  }
  return p;
}

// The SMs of the device this call runs on, read once: the persistent grid is one CTA per SM (two
// where the bucket's map admits a second CTA), capped by the number of tiles.
inline int DqPipeSmCount() {
  static int cached_dev = -1;
  static int cached_sms = 0;
  int device = 0;
  const cudaError_t d = cudaGetDevice(&device);
  if (d != cudaSuccess) dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: cudaGetDevice", d);
  if (device != cached_dev) {
    int sms = 0;
    const cudaError_t q = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (q != cudaSuccess) {
      dq_detail::RaiseCuda("flash_sca native bwd dQ pipe: SM count", q);
    }
    cached_sms = sms > 0 ? sms : 1;
    cached_dev = device;
  }
  return cached_sms;
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
  const int nqt = (p.q_length + kBM - 1) / kBM;
  const int total_tiles = nqt * static_cast<int>(p.query_heads) * static_cast<int>(p.batch);
  int blocks = DqPipeSmCount() * (Pl::kTwoCta ? 2 : 1);
  if (blocks > total_tiles) blocks = total_tiles;
  if (blocks < 1) blocks = 1;
  unsigned int* const counter = DqPipeTileCounter(stream);
  const dim3 grid(static_cast<unsigned>(blocks));
  if constexpr (FlatPlan<kD, kDv>::kEligible) {
    if (DqFlatEnabled()) {
      const cudaError_t fattr = cudaFuncSetAttribute(
          DqFlatKernel<E, kD, kDv>, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
      if (fattr != cudaSuccess) {
        dq_detail::RaiseCuda("flash_sca native bwd dQ flat: cudaFuncSetAttribute", fattr);
      }
      DqFlatKernel<E, kD, kDv><<<grid, FlatPlan<kD, kDv>::kThreads, Pl::kSmemBytes, stream>>>(
          lp, counter, total_tiles);
      const cudaError_t flaunched = cudaGetLastError();
      if (flaunched != cudaSuccess) {
        dq_detail::RaiseCuda("flash_sca native bwd dQ flat: kernel launch", flaunched);
      }
      return;
    }
  }

  DqPipeKernel<E, kD, kDv><<<grid, Pl::kThreads, Pl::kSmemBytes, stream>>>(lp, counter, total_tiles);
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
