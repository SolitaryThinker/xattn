// Native Blackwell (sm_103a) FlashSCA BACKWARD dQ leaf: one CTA per
// (batch, query head, 128-query-row tile), tcgen05/UMMA contractions with the
// accumulators in TMEM, TMA operand loads, dS formed in registers and packed
// back to TMEM as the A operand of the dQ contraction.
//
// Placement: csrc/flash_sca/native/umma_bwd_dq.cuh, beside the forward leaf
// umma_fwd.cuh. This file is modelled on that one line for
// line; every block taken from it says "as umma_fwd.cuh" and does not re-derive
// the argument it carries. The semantics this leaf must equal are the generic
// seed's DQ kernel (`cuda/template.cuh:FlashSCARegDQKernel`, code read for what
// dQ equals - never for how to compute it), and the public contract. The FP64
// leaf probe that tests this file is shipped beside it in
// `csrc/flash_sca/native/probes/` (`dq_probe.cu` + `run_dq_probe.sh`): it builds
// this header's kernel standalone and torch-free against a hand-written host FP64
// reference, and its per-case table is the numeric evidence for this leaf. The two
// places this file departs from the obvious spelling (dP's B descriptor is
// K-major over the VALUE dim, and dS is packed to 16 bits before the dQ MMA) are
// argued at their own sites, so nothing here defers to a document outside this header.
//
// ---------------------------------------------------------------- the algebra
// With s_raw = Q K^T, sl2 = p.scale_log2 (= scale * log2 e), lse the NATURAL-log
// LSE the forward wrote and LOG2E = 1/ln2:
//
//   P_ij    = exp2(sl2 * s_raw_ij - lse_i * LOG2E)      0 if j is invisible
//                                                        or lse_i is not finite
//   delta_i = sum_j P_ij dP_ij          = dY_i . y_i     (identity; see below)
//   dP_ij   = dY_i . V_j                                 (mma  dY (128 x kDv) V^T)
//   dS_ij   = P_ij * (dP_ij - delta_i)                   (dL/dS with S = scale*s_raw,
//                                                        NOT scaled: see the ABI below)
//   dQ_i    = sum_j dS_ij K_j     (unscaled here; the store multiplies by p.scale)
//
// ---------------------------------------------------------------- scale ABI
// The scale is applied EXACTLY ONCE, after the dQ MMA, in this leaf's own
// epilogue store — never inside dS. Source, four sites (each verified at its own
// site):
//   * `host.cc:341-356`: `dq_accum` is allocated iff `!deterministic`, and the
//     host folds the scale only on that arm (`dq.copy_(dq_accum.mul_(p.scale))`).
//   * `params.h:120-125`: `dq_accum`, when set, holds the UNSCALED dQ — the dK/dV
//     kernels write it and no dQ kernel runs.
//   * `kernels.cuh:754-755`, the seed's deterministic dQ store, is
//     `T::Pack(acc * p.scale, ...)`: on the det arm the single scale site is
//     INSIDE the dQ kernel, because that arm has no host scale at all.
// Consequences this file obeys: (a) `P*(dP-delta)` is formed unscaled and packed
// unscaled in BOTH arms — scaling before the 16-bit pack would change the
// rounding for no gain; (b) the store below multiplies by `p.scale` and is the
// only place in this file that does (a leaf that skipped it would return dq 16x
// too large at D=256 and 11.3x at D=128 — self-consistent, so only an oracle
// catches it); (c) `p.scale_log2` is read only inside P, where it is the logit
// scale of the forward being differentiated, not a gradient scale.
//
// Which determinism arm mounts this leaf is NOT this file's choice: `dq_accum !=
// nullptr` means "the key-owned pass produces dQ into it" and this launcher
// refuses that call. Changing what that nullability means (so the leaf could own
// dQ in both arms) is a `Params`/host change outside this leaf; the four-cell
// (D bucket x deterministic) table is the current reading, and the
// zero-double-count requirement is the same one the seed's `fused_dq` flag obeys
// (`kernels.cuh:1051`, `:1229`).
//
// P is exactly the forward's own normalized probability, not an approximation of
// it: at level 0 the forward's lse = ln2*m2 + ln(lsum) with m2 = max(s*sl2), at
// level 1 its lse = ln2*sl2*m2 + ln(lsum) with m2 = max(s); substituting either
// into exp2(sl2*s - lse*LOG2E) gives P_forward/lsum in both cases. Level 1's
// subtract-before-scale ordering is a FORWARD property that changes lse by
// rounding only; a backward differentiates the forward it is handed, so there is
// no second algebra per level here and levels 2-4 are the advanced reader's
// business (dispatch_tag.cuh's scope note), not this leaf's.
//
// delta without a second pass: y_i = sum_j P_ij V_j, so
// dY_i . y_i = sum_j P_ij dP_ij. It is computed ONCE per row, before the key
// loop, from the saved output (FP32 at levels 1-4, input dtype at level 0) and
// the incoming dY. The naive sum_j P dP would need the whole row before any dS
// can be formed, i.e. a second pass over every key tile.
//
// Dead rows (out of range, or no visible key -> lse = -INFINITY, which is this
// family's empty-row convention in the forward): delta = 0, every P entry stays
// 0, dS = 0, dQ row = 0 exactly, never NaN. The P select is on the mask and on
// the finiteness of lse BEFORE any arithmetic - never by carrying -inf through
// the scale, which for scale_log2 <= 0 would turn one invisible entry into NaN
// or +inf (umma_fwd.cuh's mask-select rule, reused).
//
// deterministic: dQ is row-owned. One CTA owns one (b, hq, 128-row tile) and one
// query head maps to one dQ row, so no atomic, no split-K and no reduction order
// exists to vary; `deterministic=True` is bit-exact by construction and this leaf
// needs nothing special for it. (dK/dV are key-owned and do need the argument -
// not this leaf's claim to make.) The unordered `p.dq_accum` route is the OTHER
// dQ path (no dQ kernel runs when the host supplies it); this leaf must never be
// launched with dq_accum and without dq, which the launcher refuses.
//
// ---------------------------------------------------------------- the pipeline
// One warpgroup (warps 0-3, thread t owns query row t); warp 0 issues every MMA.
//
//   resident in SMEM, TMA-loaded once:  Q (128 x kD), dY (128 x kDv)
//   streamed per key tile, kStages:     K (64 x kD), V (64 x kDv)
//
//   per key tile t in [min_lo/kBN .. max_hi/kBN]:
//     (1) S  = Q  K^T   MmaSs, K-major both sides, contract over kD   -> kSCol
//     (2) dP = dY V^T   MmaSs, K-major both sides, contract over kDv  -> kDpCol
//         (1) and (2) are issued back to back under ONE MmaCommit, one barrier:
//         both are M=128, N=kBN, and both operands are K-major in both, so they
//         share one instruction descriptor; only slabs and k-step counts differ.
//     (3) ld S and dP (32 columns at a time), form dS = P*(dP - delta)
//         (UNSCALED: the single scale site is the store, see the ABI block)
//     (4) st dS as 16-bit pairs into kDsCol (kBN/2 = 32 columns, the fwd's P form)
//     (5) dQ += dS K     MmaTs, A = dS from TMEM, B = K MN-major, N = 64 per K
//         slab, kSlabsK slabs                                    -> kDqCol
//   after the loop: ld dQ from TMEM, store to p.dq (input dtype).
//
// (2)'s B descriptor is NOT the forward's pv_idesc: dP contracts over the VALUE
// dim, so V is K-major there (its value axis is the contiguous one, 16 elements
// per k-step, exactly like K in the S mma), while pv_idesc is the MN-major form
// for a contraction over KEYS. (5) is where the forward's PV spelling is reused
// verbatim: same MN-major B, same 16-keys-per-k-step, same per-slab output
// window and accumulate flag. Naming pv_idesc for dP would
// be a mis-attribution of (5)'s form - the derivation is the paragraph above, and
// the probe's dP arm is what tests it. Getting this backwards is a silent wrong
// answer, not a slow one.
//
// ---------------------------------------------------------------- budgets
// TMEM columns: S kBN + dP kBN + dQ kD, with dS sharing dP's window
//   256 at kD=128 (allocates 256 - two CTAs per SM), 384 at kD=256 (allocates
//   512 - one CTA per SM). Neither shared window needs one of its own: S is
//   consumed by the leg before the next tile's S MMA overwrites it, and dP is
//   consumed by the leg before dS is stored into its columns. The occupancy note
//   above the plan states both orderings and what the sharing is worth.
// SMEM, against 232448 B of opt-in dynamic shared memory (the per-BLOCK opt-in
// ceiling, 227 KiB - 233472 B is the per-SM total and is not what the attribute
// or the preflight tests):
//   Q = 256*kD B, dY = 256*kDv B, K = kStages*128*kD B, V = kStages*128*kDv B,
//   then (kStages+5) mbarriers + 16 B TMEM slot + 4*16 B row reductions +
//   kBN*8 B key-segment scratch.
//   kD=kDv=128, kStages=1: 32768 + 32768 + 16384 + 16384 = 98304, +48+16+64+512
//     = 98944 B  (96 KiB + 640 B); with the 1024 B base-rebase slack the request
//     is 99968 B, inside the 116224 B that two CTAs per SM may hold.
//   kD=kDv=256, kStages=1: 65536 + 65536 + 32768 + 32768 = 196608, +48+16+64+512
//     = 197248 B  (192 KiB + 640 B).
//   kStages = 1 for every bucket, and the reason is occupancy rather than
//   capacity: at (128,128) two K/V stages would make the request 131720 B, over
//   the 116224 B two CTAs may hold, and a second resident CTA is worth more than
//   the stage overlap it costs (two CTAs at one stage 5834 cycles/round against
//   one CTA at two stages 10293, same work, same rounded walk). At 256/256 the
//   two-stage request would be 262800 B, over the 232448 B opt-in ceiling; that
//   bucket is one CTA either way. kStages=1 costs overlap, never correctness: the
//   K/V stage is refilled only after the dQ MMA that reads it has completed.
//
// ---------------------------------------------------------------- scope
// FP16 and BF16; D and Dv buckets 128 and 256 (the traced training shapes); levels
// 0 and 1 (identical algebra, see above); GQA/MQA through p.group; previous K/V
// whole key tiles or a packed map; unequal Q/K lengths; BOS/segment metadata;
// resets; empty rows. No torch, no new primitive, no public API change.
//
// ---------------------------------------------------------------- mask arms
// Two licences, and the header states which one is taken when:
//   ARM A, interval-only (every traced call): the host converts bos_mask to
//     per-row row_lo/row_hi and hands `q_segment`/`k_segment` as NULL, so
//     detail::SegmentTest(p,b) is FALSE and `seg` is false for the batch row. The
//     per-element predicate is then `bd.lo <= j <= bd.hi` alone, and any 64-key
//     tile whose span lies inside every live row's interval (max_lo/min_hi) takes
//     the no-mask branch of `col_visible` - the traced shapes actually get the
//     fast path. `segment_exact[b]` is the same licence expressed for the id
//     route, and TileIsFull consumes it unchanged (its proof obligation is
//     umma_fwd.cuh's, reused not re-derived: the host may only leave the planes
//     null / set segment_exact when the bounds already exclude every other
//     segment).
//   ARM B, full planes (the explicit `segment_idx` / `q_segment_idx` route, which
//     sca-contract covers and the traced packings never reach): `seg` is true, so
//     TileIsFull refuses every full tile, the 64 key ids of the tile are staged in
//     SMEM and the predicate is `RowBounds ∩ (kseg[c] == qseg)` per element. A
//     leaf that implemented only ARM A is silently wrong on sca-contract; one that
//     implemented only ARM B is correct everywhere but pays the elementwise test on
//     the shapes that carry the primary number. Both are in this file, selected by
//     the shared tables and nothing else.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// The forward leaf is included, not copied: kBM/kBN/kThreads/kWarps, ElemTraits,
// WarpMin/WarpMax, TileIsFull, ResolveKeyMap, TmemFenceBeforeSync/AfterSync are
// shared vocabulary, and re-declaring them here would be an ODR violation in any
// TU that wants both leaves (the instantiation TUs of the two leaves are
// separate files, but a host or a probe TU may include both). It also pulls in
// params.h, umma_params.h, visibility.cuh and the pinned primitive set
// (umma_common_local.h) exactly once, with umma_fwd.cuh's own mutual-exclusion
// guards intact.
#include "flash_sca/native/umma_fwd.cuh"

namespace xattn::ops::flash_sca_native {
namespace umma {

// 1/ln2, the factor between the forward's natural-log LSE and the log2 domain
// this leaf exponentiates in. Written out rather than taken from <cmath>'s
// M_LOG2E: the forward leaf does not include <cmath> and this file keeps the
// same include surface.
inline constexpr float kDqLog2e = 1.4426950408889634f;

// 16-bit element -> float. Explicit intrinsics rather than static_cast: the
// CUDA 16-bit types' conversion operators are conditional on
// __CUDA_NO_BFLOAT16_CONVERSIONS__, which a caller's flags may define.
template <class Elem>
__device__ __forceinline__ float ToFloat(Elem x);
template <>
__device__ __forceinline__ float ToFloat<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <>
__device__ __forceinline__ float ToFloat<__half>(__half x) {
  return __half2float(x);
}

// ------------------------------------------------------------------ plan
// Occupancy decides this kernel's per-round cost, so the column budget comes first.
// S and dP are full FP32 accumulator windows of kBN columns each, dS is kBN/2 columns
// of packed 16-bit pairs, and dQ is a full accumulator window of kD columns. Laid out
// one after another they take kD + 2*kBN + kBN/2 columns - 288 at (128,128) - which is
// over the 256 that two CTAs need, so the allocator takes the whole 512 columns and the
// SM holds exactly one CTA of four warps: every declared wait of the round (two MMA
// completions, the S and dP loads, the dS store, two __syncthreads) is then paid in
// sequence, and the round costs 10,293 cycles where its own MMAs need 1,015.
// Two changes hand that time back, measured on the traced 4B bucket at BF16, level 0,
// with dQ bit-identical to the layout above:
//   * dS goes into the window dP has already released, which is 256 columns used. The
//     leg loads dP into registers and waits on those loads (TmemWaitLd) before it
//     stores dS into the same columns, so read-before-write there is program order in
//     the warps that own the rows - no barrier sits between them, and none is needed,
//     which is also true of the merged layout. The fence plus syncthreads keeps its
//     old place, AFTER the dS store and BEFORE the dQ MMA; the cross-round hazard
//     (the next tile's dP MMA writing the window this tile's dQ MMA reads) is closed
//     by the pre-existing MbarWait(bar_dq) that gates the next MMA issue on the dQ
//     MMA's completion.
//   * kStages drops to 1, for the same reason a second CTA is wanted: two K/V stages
//     push the dynamic shared-memory request over the 116,224 B that two CTAs can
//     have. At one CTA the exposed K/V refill is worth 5.4% of the round (an equal-work
//     pair: 10,877 cycles/round with one stage against 10,293 with two); at two CTAs
//     that wait is filled by the other CTA's work.
// 10,293 -> 5,834 cycles/round, -43.3% at the traced 4B bucket. The 0.9B bucket
// (D=Dv=256) cannot take it - dQ alone is 256 columns, so its floor is 384 - and keeps
// one CTA and its 12,822 cycles/round.

// Constants only; no state. Mirrors ForwardPlan<kD,kDv> and keeps its naming so
// a diff between the two leaves reads as the differences and nothing else.
template <int kD, int kDv>
struct DqPlan {
  static_assert(kD % 64 == 0 && kDv % 64 == 0, "buckets are multiples of 64");
  static constexpr int kSlabsQ = kD / 64;
  static constexpr int kSlabsK = kD / 64;
  static constexpr int kSlabsV = kDv / 64;  // dY's slab count as well: same layout as Q
  // One 64-column slab of a 128-row operand: 128 rows * 64 columns * 2 B.
  static constexpr int kSlab = kBM * 128;
  static constexpr int kQBytes = kSlabsQ * kSlab;
  static constexpr int kDyBytes = kSlabsV * kSlab;
  // One 64-row key tile of K or V: 64 rows * 64 columns * 2 B (as umma_fwd.cuh).
  static constexpr int kKvSlab = kBN * 128;
  // Refuted, not adopted: two stages at one CTA measured 505 cycles of saving
  // (9,373 -> 8,868 cycles/round) against the 3,215 a 1-CTA leaf would need to
  // match two co-resident CTAs at 6,135 cycles/round/SM, i.e. 15.7 % of the
  // requirement. The ring is correct at depth 2 (56 case-arms, 0 failed) and is
  // worth the 5.7 % of a round that the K/V wait costs, no more. Kept at 1.
  static constexpr int kStages = 1;  // two CTAs, not two stages: see the note above
  static constexpr int kKBytes = kStages * kSlabsK * kKvSlab;
  static constexpr int kVBytes = kStages * kSlabsV * kKvSlab;
  static constexpr int kBarOff = kQBytes + kDyBytes + kKBytes + kVBytes;  // 1024-aligned
  // bars[0] = Q, bars[1] = dY, bars[2 .. 1+kStages] = K/V per stage,
  // bars[2+kStages] = S+dP MMAs, bars[3+kStages] = dQ MMA (two spare).
  static constexpr int kBarCount = 3 + kStages + 2;
  static constexpr int kSlotOff = kBarOff + kBarCount * 8;
  static constexpr int kRedOff = kSlotOff + 16;  // kWarps x 4 int32
  static constexpr int kSegOff = kRedOff + kWarps * 16;
  static constexpr int kSmemBytes = kSegOff + kBN * 8;
  // The kernel REBASES its dynamic shared-memory base to the next 1024 B boundary
  // instead of assuming the driver hands it one (the family rule for native backward
  // leaves - a leaf must make its own alignment true). The rebase can move
  // the base by up to 1023 B, so THE CORRECTION BUDGET LIVES INSIDE THE REQUEST, not
  // at the <<<>>> site: an under-requested launch would let the last stage stride
  // run past the allocation.
  static constexpr int kSmemSlack = 1024;
  static constexpr int kSmemRequest = kSmemBytes + kSmemSlack;
  static_assert(kSmemRequest <= 232448,
                "SMEM budget: the 227 KiB opt-in ceiling, rebase slack included");
  // The half of the invariant that is a compile-time fact: every start offset a
  // descriptor encodes or a TMA base is derived from is a multiple of 1024 B, so a
  // uniform rebase shifts the whole window and leaves every relative offset on the
  // same 1024 B phase (the descriptor's base_offset stays 0). A future stage count
  // or slab size that breaks this fails the build rather than the hardware.
  static_assert(kSlab % 1024 == 0, "operand slab stride is not a 1024 B multiple");
  static_assert(kQBytes % 1024 == 0 && kDyBytes % 1024 == 0, "Q and dY bases");
  static_assert(kKvSlab % 1024 == 0 && kKBytes % 1024 == 0 && kVBytes % 1024 == 0,
                "K/V tile stride and per-stage bases");
  static_assert(kBarOff % 1024 == 0, "the barrier block follows the KV slabs");
  // The TMA loops write box r at byte offset `r * 128` inside a slab, so a box
  // height that is not a multiple of 8 puts those destinations off the 1024 B phase
  // the single rebase preserves. kTmaRows lives in the shared header (one box shape
  // serves both leaves); asserting it HERE is what makes this leaf's invariant a
  // compile-time fact instead of a property of a constant in another file - the
  // failure mode would be a silent one, a box that reads its own wrong rows.
  static_assert(kTmaRows % 8 == 0, "TMA box height must keep r * 128 on a 1024 B phase");

  // TMEM columns. dS shares the dP window: the leg has already loaded dP into
  // registers when it stores dS, and the fence plus syncthreads between the two is
  // the ordering both windows relied on. That is what brings the used count to 256
  // and the second CTA in at (128,128) - see the occupancy note above the plan.
  static constexpr int kSCol = 0;
  static constexpr int kDpCol = kSCol + kBN;
  static constexpr int kDsCol = kDpCol;
  static constexpr int kDqCol = kDpCol + kBN;
  static constexpr int kUsedCols = kDqCol + kD;
  static constexpr int kAllocCols = kUsedCols <= 256 ? 256 : 512;
  static_assert(kUsedCols <= 512, "TMEM column budget");
};

// Host-visible so the budget audits can harvest the column count, exactly as
// ForwardTmemColumns is. Must agree with DqPlan<kD,kDv>::kAllocCols: used is
// S (kBN) + dP (kBN) + dQ (kD), because dS (kBN/2 packed columns) is stored into
// the dP window once the leg has consumed it - see the occupancy note above the
// plan - and the total is allocated in the next whole window of 256 or 512.
// dQ is the only term kD enters.
constexpr int DqTmemColumns(int d_bucket, int dv_bucket) {
  (void)dv_bucket;
  const int used = 64 + 64 + d_bucket;
  return used <= 256 ? 256 : 512;
}

// ------------------------------------------------------------------ kernel
template <class Elem, int kD, int kDv>
__global__ void __launch_bounds__(kThreads) DqKernel(
    const __grid_constant__ UmmaLaunchParams lp) {
  using Plan = DqPlan<kD, kDv>;
  const Params& p = lp.params;
  // 128 B is the only alignment CUDA guarantees the dynamic shared-memory base, and
  // the TMA descriptors of this leaf encode tile bases that assume 1024 B, so the
  // base is MADE true here rather than asserted. The correction
  // budget is inside Plan::kSmemRequest, so the rebased window is always inside the
  // allocation.
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* const smem =
      smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);

  unsigned char* const sq = smem;
  unsigned char* const sdy = smem + Plan::kQBytes;
  unsigned char* const sk = smem + Plan::kQBytes + Plan::kDyBytes;
  unsigned char* const sv = smem + Plan::kQBytes + Plan::kDyBytes + Plan::kKBytes;
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
  const int64_t kvh = p.group > 0 ? hq / p.group : hq;  // GQA/MQA: as umma_fwd.cuh
  const int64_t i = static_cast<int64_t>(qt) * kBM + tid;
  const bool row_in_range = i < p.q_length;

  const uint32_t bar_q = bar0;
  const uint32_t bar_dy = bar0 + 8;
  const uint32_t bar_sdp = bar0 + (Plan::kStages + 2) * 8;
  const uint32_t bar_dq = bar0 + (Plan::kStages + 3) * 8;

  if (tid == 0) {
    MbarInit(bar_q, 1);
    MbarInit(bar_dy, 1);
    for (int s = 0; s < Plan::kStages; ++s) MbarInit(bar0 + (2 + s) * 8, 1);
    MbarInit(bar_sdp, 1);
    MbarInit(bar_dq, 1);
  }
  if (warp == 0) {
    TmemAlloc(Plan::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) MbarInitFence();

  // Per-row visibility, from the shared tables only - no second opinion about
  // the mask anywhere in this file (as umma_fwd.cuh).
  const detail::Bounds bd = detail::RowBounds(p, b, i);
  const bool live = row_in_range && bd.lo <= bd.hi;
  {
    // min_lo/max_hi drive the KEY-TILE LOOP (live rows only); max_lo/min_hi drive
    // the FULL test and are reduced over ALL rows, a dead row contributing
    // INT_MAX/INT_MIN so a CTA holding any dead row never takes the unmasked
    // path. Verbatim from umma_fwd.cuh including why the reduction is over all
    // rows (the 988-CTA-set proof, and the refuted live-only variant).
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

  // A row with no visible key writes dQ = 0 exactly (the contract's empty-row
  // requirement, and this leaf's only output - the forward's y/y_state/lse
  // versions of this block collapse to the one plane).
  auto write_dead_row = [&]() {
    if (!row_in_range) return;
    Elem* dqrow = reinterpret_cast<Elem*>(p.dq) +
                  ((b * p.q_length + i) * p.query_heads + hq) * p.qk_dim;
    for (int64_t c = 0; c < p.qk_dim; ++c) dqrow[c] = ElemTraits<Elem>::From(0.f);
  };
  auto dealloc_and_exit = [&]() {  // as umma_fwd.cuh
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

  // No alignment trap here (the forward leaf's __trap() is that leaf's
  // own). The prologue above MAKES the base a 1024 B boundary, and
  // every slab, stage and barrier start is a 1024 B multiple of it - the
  // static_asserts in DqPlan hold that half at compile time. A trap left here would
  // guard a premise that can no longer fail, and it would fire AFTER the seam has
  // chosen this arm: the one failure a caller cannot fall back from is a process
  // that dies mid-call.

  // Resident tiles: Q over kSlabsQ slabs and dY over kSlabsV slabs, each a box
  // count of 64-row x 64-column TMA transfers. Rows past Nq and columns past the
  // logical D/Dv are the TMA's zero fill and are excluded from every store (as
  // umma_fwd.cuh). dY's map is `lp.maps.dy`, built by the host from p.dy with the
  // same 4-D [column, head, row, batch] encoder as q/k/v.
  if (tid == 0) {
    MbarArriveExpectTx(bar_q, Plan::kQBytes);
    for (int s = 0; s < Plan::kSlabsQ; ++s) {
      for (int r = 0; r < kBM; r += kTmaRows) {
        TmaLoad4D(sq + s * Plan::kSlab + r * 128, &lp.maps.q, bar_q, s * 64,
                  static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
      }
    }
    MbarArriveExpectTx(bar_dy, Plan::kDyBytes);
    for (int s = 0; s < Plan::kSlabsV; ++s) {
      for (int r = 0; r < kBM; r += kTmaRows) {
        TmaLoad4D(sdy + s * Plan::kSlab + r * 128, &lp.maps.dy, bar_dy, s * 64,
                  static_cast<int>(hq), qt * kBM + r, static_cast<int>(b));
      }
    }
  }

  // ---------------------------------------------------------------- delta
  // delta_i = dY_i . y_i, once per row, in FP32, before the key loop (see the
  // algebra block). Source plane: the contract's saved output (`p.saved`, FP32 when
  // `p.saved_fp32`) if the host set it, else the forward's own planes (`p.y` /
  // `p.y_state`, which is how the delta block names them). The launcher applies the
  // SAME expression as its refusal test, so the host and the kernel can never
  // disagree about which pointer is read.
  const void* const saved_base =
      p.saved != nullptr ? p.saved
                         : (p.saved_fp32 ? static_cast<const void*>(p.y_state)
                                         : static_cast<const void*>(p.y));
  const float lse_i =
      (live && p.lse_in != nullptr)
          ? p.lse_in[(b * p.query_heads + hq) * p.q_length + i]
          : -INFINITY;
  // The LSE -> log2-domain offset: ONE definition, under the seed's name.
  // Every NON-FINITE LSE maps
  // to +inf, so exp2(finite - inf) == 0 for ANY sign of scale_log2 - which is why
  // the seed's value form and the predicate form agree bit-exactly on every input
  // this ABI can produce, so the value form is the one kept. `ok` stays a
  // separate test: the STORE writes an exact 0 for such a row instead of relying
  // on the arithmetic being 0. The
  // forward's ABI writes -INFINITY for a row with no visible key and never NaN; a
  // NaN or +inf LSE (which this ABI does not produce) also yields ok == false, so
  // no column of such a row can be poisoned. When the shared native header carries
  // this helper, this local definition goes away in favour of it.
  struct DqLse2 {
    float off;
    bool ok;
  };
  const DqLse2 lse2 = [&]() -> DqLse2 {
    const bool ok = live && lse_i > -INFINITY && lse_i < INFINITY;
    return DqLse2{ok ? lse_i * kDqLog2e : INFINITY, ok};
  }();
  const bool lse_ok = lse2.ok;
  const float lse_log2 = lse2.off;

  float delta = 0.f;
  if (lse_ok && p.delta != nullptr) {
    // Shared preprocess plane (`Params::delta`; "the same expression the
    // key-owned leaves subtract"): reading it is the primary form, so the two dQ
    // producers cannot round delta differently.
    delta = p.delta[(b * p.query_heads + hq) * p.q_length + i];
  } else if (lse_ok) {
    // Fallback: the dY.y identity of the block above, computed in-kernel. This is
    // what a standalone leaf probe exercises (no preprocess launch), and it is the
    // only path that needs the saved-output plane at all.
    const int64_t row0 = ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
    const Elem* const dy_row = reinterpret_cast<const Elem*>(p.dy) + row0;
    float acc = 0.f;
    if (p.saved_fp32) {
      const float* const y_row = static_cast<const float*>(saved_base) + row0;
      for (int64_t c = 0; c < p.value_dim; ++c) {
        acc += ToFloat(dy_row[c]) * y_row[c];
      }
    } else {
      const Elem* const y_row = static_cast<const Elem*>(saved_base) + row0;
      for (int64_t c = 0; c < p.value_dim; ++c) {
        acc += ToFloat(dy_row[c]) * ToFloat(y_row[c]);
      }
    }
    delta = acc;
  }

  const int t0 = min_lo / kBN;
  const int t1 = max_hi / kBN;
  const int steps = t1 - t0 + 1;

#if defined(XATTN_DQ_ROUND_STAT)
  // --------------------------------------------------------------- round stat
  // An INDEPENDENT loop total.
  //
  // Why it exists: a phase table whose `loop_total` is the sum of its own phase
  // deltas closes as arithmetic and not as attribution. Two
  // clock64() calls bracketing the round give a number the phases cannot
  // manufacture, and every existing phase share can then be tested against it
  // instead of against itself.
  //
  // Cost, deliberately: the counters live in __shared__ and are touched by tid 0
  // only, and NO register is live across the body. That is not fussiness. The
  // (128,256) row sits at 251 of 256 registers in the ptxas report, so a 64-bit temporary
  // held across the round would spill and change the very round the arm exists to
  // measure. `dt` below is one 32-bit value live for six instructions at the loop
  // tail; whether even that costs a register is MEASURED by cuobjdump on the row,
  // not asserted here.
  //
  // The rule is `dt > 2 * running_min` against the CTA's OWN floor: no cross-job
  // constant, no warm-up beyond the first round, and immune to two consecutive
  // stalls (a previous-round test misses the second of a pair).
  __shared__ unsigned int s_t0c, s_rmin, s_nround, s_nflag;
  __shared__ unsigned long long s_excess;
  // Phase accumulators, one per segment of the round's own text: 0 = the K/V
  // arrival wait, 1 = S+dP issue and its wait, 2 = the dS elementwise block and
  // its TMEM store, 3 = the dQ issue, its wait and the next tile's re-issue.
  // Same rule as the counters above: __shared__, never registers, and written
  // only by tid 0 so the four sums are one thread's view of one CTA's round.
  __shared__ unsigned long long s_ph[4];
  __shared__ unsigned int s_mark;
  // Inner split of ph_elem: [0] the two TmemLoad32 + TmemWaitLd per h iteration,
  // [2] the TmemStore32 + TmemWaitSt. [1] is the remainder - the exp2f/fma loop and
  // its col_visible predicates - and is not accumulated, it is ph_elem minus these.
  __shared__ unsigned long long s_pe[3];
  __shared__ unsigned int s_pmark;
  if (tid == 0) {
    s_rmin = 0u;
    s_nround = 0u;
    s_nflag = 0u;
    s_excess = 0ull;
    s_ph[0] = s_ph[1] = s_ph[2] = s_ph[3] = 0ull;
    s_pe[0] = s_pe[1] = s_pe[2] = 0ull;
    s_pmark = 0u;
    s_mark = 0u;
  }
  __syncthreads();
#endif
  const int64_t nt_total = p.total_length;

  auto issue_kv = [&](int t) {  // as umma_fwd.cuh, ResolveKeyMap included
    if (tid != 0) return;
    const int stage = (t - t0) % Plan::kStages;
    const int64_t j0 = static_cast<int64_t>(t) * kBN;
    const CUtensorMap* kmap;
    const CUtensorMap* vmap;
    int row0;
    ResolveKeyMap(lp, j0, &kmap, &vmap, &row0);
    MbarArriveExpectTx(bar0 + (2 + stage) * 8,
                       Plan::kSlabsK * Plan::kKvSlab + Plan::kSlabsV * Plan::kKvSlab);
    unsigned char* kdst = sk + stage * Plan::kSlabsK * Plan::kKvSlab;
    unsigned char* vdst = sv + stage * Plan::kSlabsV * Plan::kKvSlab;
    for (int s = 0; s < Plan::kSlabsK; ++s) {
      for (int r = 0; r < kBN; r += kTmaRows) {
        TmaLoad4D(kdst + s * Plan::kKvSlab + r * 128, kmap, bar0 + (2 + stage) * 8,
                  s * 64, static_cast<int>(kvh), row0 + r, static_cast<int>(b));
      }
    }
    for (int s = 0; s < Plan::kSlabsV; ++s) {
      for (int r = 0; r < kBN; r += kTmaRows) {
        TmaLoad4D(vdst + s * Plan::kKvSlab + r * 128, vmap, bar0 + (2 + stage) * 8,
                  s * 64, static_cast<int>(kvh), row0 + r, static_cast<int>(b));
      }
    }
  };

  for (int t = t0; t < t0 + min(steps, Plan::kStages); ++t) issue_kv(t);
  MbarWait(bar_q, 0);
  MbarWait(bar_dy, 0);

  const int elem_fmt = ElemTraits<Elem>::kFormat;
  // (1) S = Q K^T and (2) dP = dY V^T: M=128, N=kBN, K-major A and K-major B on
  // both, so one instruction descriptor serves both. The contraction lengths
  // differ (kD vs kDv) and so do the slab walks; the major-ness does not.
  const uint32_t ss_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/0);
  // (5) dQ += dS K: A from TMEM, B = K MN-major, N = 64 per K slab - the forward's
  // pv_idesc verbatim (same MN-major k-step, 16 keys per 128 u128).
  const uint32_t ts_idesc = MakeInstrDesc(elem_fmt, kBM, kBN, /*a_major=*/0, /*b_major=*/1);
  // Leading byte offset of that MN-major B descriptor, the same form as the forward
  // leaf's PV descriptor (umma_fwd.cuh): zero for a single slab, kBN*8 u128 (one
  // 64-row image, 8192 B) once the operand is a multi-slab image. This is a
  // DERIVATION, not a measurement: at 64-row images and one 64-column group per
  // MMA the two spellings address the same bytes, so no passing arm can separate
  // them; a kSlabsK > 1 arm (the D=256 bucket, the 256/256 probe case) exercises
  // this value but would also pass with the other one. Separating them needs the
  // descriptor the hardware consumed, which nothing here prints. Recorded as this
  // leaf's open uncertainty 1.
  const uint32_t dq_lbo = Plan::kSlabsK > 1 ? static_cast<uint32_t>(kBN * 8) : 0u;

  const bool seg = detail::SegmentTest(p, b);
  // As umma_fwd.cuh (PRECONDITION, written at the read site): p.q_segment must be
  // a DENSE plane with row stride p.q_length, and p.k_segment dense with row
  // stride p.total_length. host.cc's PrepareCommon calls .contiguous() on both;
  // any future host path must keep that or carry a density flag.
  const int64_t qseg = (seg && row_in_range) ? p.q_segment[b * p.q_length + i] : 0;
  // The scale enters P through scale_log2 only. The ONE place `p.scale` is applied
  // is the dQ store at the epilogue, per the scale-once ABI; nothing
  // between this line and that store reads it.
  const float sl2 = p.scale_log2;

  const uint32_t qb_smem = SmemAddr(sq);
  const uint32_t dyb_smem = SmemAddr(sdy);

  for (int t = t0; t <= t1; ++t) {
    const int nt = t - t0;
#if defined(XATTN_DQ_ROUND_STAT)
    if (tid == 0) { s_t0c = static_cast<unsigned int>(clock64()); s_mark = s_t0c; }
#endif
    const int stage = nt % Plan::kStages;
    MbarWait(bar0 + (2 + stage) * 8, static_cast<uint32_t>((nt / Plan::kStages) & 1));
#if defined(XATTN_DQ_ROUND_STAT)
    // ph0 closes here: the wait for this tile's K/V to arrive.
    if (tid == 0) {
      const unsigned int now0 = static_cast<unsigned int>(clock64());
      s_ph[0] += now0 - s_mark;
      s_mark = now0;
    }
#endif

    uint32_t const kb_smem = SmemAddr(sk + stage * Plan::kSlabsK * Plan::kKvSlab);
    uint32_t const vb_smem = SmemAddr(sv + stage * Plan::kSlabsV * Plan::kKvSlab);

    // (1)+(2), one warp, one commit (as umma_fwd.cuh's "one warp issues" rule:
    // every warp's ElectOne would issue the whole MMA again). Two MMAs into two
    // different TMEM windows, so a single completion barrier is enough and the
    // second one's latency hides behind the first.
    if (warp == 0) {
      for (int kk = 0; kk < kD / 16; ++kk) {
        const int slab = kk >> 2, within = kk & 3;
        MmaSs(tbase + Plan::kSCol, MakeSmemDesc(qb_smem + slab * Plan::kSlab + within * 32),
              MakeSmemDesc(kb_smem + slab * Plan::kKvSlab + within * 32), ss_idesc,
              kk == 0 ? 0u : 1u);
      }
      for (int sl = 0; sl < Plan::kSlabsV; ++sl) {
        for (int kk = 0; kk < 4; ++kk) {  // 64 value columns per slab, 16 per k-step
          // ONE output window accumulating over (slab, k-step), so the reset is
          // the very first step of the very first slab OF THIS TILE - NOT the
          // forward's per-slab reset, which exists because PV gives each V slab
          // its own output window. LIFETIMES, all three contractions together:
          //   S  : reset per key tile     (kk == 0)
          //   dP : reset per key tile     (sl == 0 && kk == 0) - the window's
          //        columns name THIS tile's keys and are consumed by step (3)
          //        before the next tile issues its MMAs.
          //   dQ : reset once per kernel  (nt == 0 && kk == 0) - the output
          //        accumulates across every key tile, which is what it is for.
          // A per-kernel `nt == 0` on dP is a silent multi-tile bug: from the
          // second key tile on, dP adds onto the previous tile's dP.
          MmaSs(tbase + Plan::kDpCol,
                MakeSmemDesc(dyb_smem + sl * Plan::kSlab + kk * 32),
                MakeSmemDesc(vb_smem + sl * Plan::kKvSlab + kk * 32), ss_idesc,
                (sl == 0 && kk == 0) ? 0u : 1u);
        }
      }
      MmaCommit(bar_sdp);
    }
    MbarWait(bar_sdp, static_cast<uint32_t>(nt & 1));
#if defined(XATTN_DQ_ROUND_STAT)
    // ph1 closes here: the S+dP issue by warp 0 and the whole CTA's wait on it.
    if (tid == 0) {
      const unsigned int now1 = static_cast<unsigned int>(clock64());
      s_ph[1] += now1 - s_mark;
      s_mark = now1;
    }
#endif

    const int64_t kb = static_cast<int64_t>(t) * kBN;
    const bool full = TileIsFull(seg, kb, max_lo, min_hi);
    if (seg && !full) {
      if (tid < kBN) {
        const int64_t j = kb + tid;
        kseg[tid] = j < nt_total ? p.k_segment[b * nt_total + j] : (qseg + 1);
      }
      __syncthreads();
    }

    // The forward's two mask branches folded into one predicate: a FULL tile has
    // every key inside every live row's interval and needs no key test
    // (TileIsFull's proof obligation is umma_fwd.cuh's, reused not re-derived);
    // otherwise the same bounds + exact-segment test, on the same tables.
    auto col_visible = [&](int c) {
      if (full) return true;
      const int64_t j = kb + c;
      if (j > bd.hi || j < bd.lo) return false;
      return !seg || kseg[c] == qseg;
    };

    // (3)+(4): dS rows. 32 TMEM columns at a time keeps the live register set at
    // two 32-wide float windows instead of two 64-wide ones; the loads are
    // warp-collective (.sync.aligned) and therefore run for EVERY row of the warp
    // including dead ones (the forward's alpha comment: never diverge around a
    // collective). Dead and invisible entries are excluded by the select on e,
    // before any arithmetic.
    uint32_t dsv[kBN / 2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float s[32];
      float dpr[32];
#if defined(XATTN_DQ_ROUND_STAT)
      if (tid == 0) s_pmark = static_cast<unsigned int>(clock64());
#endif
      TmemLoad32(lane_base | (tbase + Plan::kSCol + 32 * h), s);
      TmemLoad32(lane_base | (tbase + Plan::kDpCol + 32 * h), dpr);
      TmemWaitLd();  // orders both of this thread's loads (TmemLoadCols' rule)
#if defined(XATTN_DQ_ROUND_STAT)
      if (tid == 0) {
        const unsigned int npl = static_cast<unsigned int>(clock64());
        s_pe[0] += npl - s_pmark;
      }
#endif
      // [w15_i279_hoist] On a FULL tile `col_visible` is identically true, so both per-column
      // predicates are identically `lse_ok`, a ROW property.  Same arithmetic, same discard of
      // invisible columns, one branch per 32-column window instead of 64.  The invariant this
      // relies on: the row-varying test below encloses no TMEM collective and no CTA barrier -
      // the `TmemLoad32` pair and its `TmemWaitLd` stay above the branch, and the only thing
      // skipped is arithmetic - which is why a divergent `lse_ok` is safe here.  Levels 2-4 never
      // reach this leaf (`native_adv::LaunchMainDq`); the non-full loop below is unchanged.
      if (full && lse_ok) {
#pragma unroll
        for (int c2 = 0; c2 < 32; c2 += 2) {
          const float e0 = exp2f(sl2 * s[c2] - lse_log2);
          const float e1 = exp2f(sl2 * s[c2 + 1] - lse_log2);
          dsv[(32 * h + c2) >> 1] = ElemTraits<Elem>::Pack2(e0 * (dpr[c2] - delta),
                                                            e1 * (dpr[c2 + 1] - delta));
        }
      } else if (full) {
#pragma unroll
        for (int c2 = 0; c2 < 32; c2 += 2) {
          dsv[(32 * h + c2) >> 1] = ElemTraits<Elem>::Pack2(0.f, 0.f);
        }
      } else
#pragma unroll
      for (int c2 = 0; c2 < 32; c2 += 2) {
        const int c = 32 * h + c2;
        // Branchless form of the same expression. The previous shape branched
        // per column (two `col_visible` tests per step, 64 per thread per round),
        // which puts a data-dependent branch between every pair of `exp2f` and
        // leaves the MUFU pipe issuing one transcendental at a time: the round
        // measures ~1.45 instructions per cycle against 4 available, and this
        // block is 74-78 % of it. Computing both columns unconditionally and
        // selecting is the SAME arithmetic for visible columns and discards the
        // invisible ones exactly as the branch did - the discarded `exp2f` can
        // reach no output, and the select cannot form a NaN from it because the
        // untaken operand is never the one stored.
        const bool v0 = lse_ok && col_visible(c);
        const bool v1 = lse_ok && col_visible(c + 1);
        const float e0 = exp2f(sl2 * s[c2] - lse_log2);
        const float e1 = exp2f(sl2 * s[c2 + 1] - lse_log2);
        const float d0 = v0 ? e0 * (dpr[c2] - delta) : 0.f;
        const float d1 = v1 ? e1 * (dpr[c2 + 1] - delta) : 0.f;
        dsv[(32 * h + c2) >> 1] = ElemTraits<Elem>::Pack2(d0, d1);
      }
    }
#if defined(XATTN_DQ_ROUND_STAT)
    if (tid == 0) s_pmark = static_cast<unsigned int>(clock64());
#endif
    TmemStore32(lane_base | (tbase + Plan::kDsCol), dsv);
    TmemWaitSt();
#if defined(XATTN_DQ_ROUND_STAT)
    if (tid == 0) {
      const unsigned int nps = static_cast<unsigned int>(clock64());
      s_pe[2] += nps - s_pmark;
    }
#endif
    // Closes three TMEM handoffs at once, as umma_fwd.cuh's identical block: this
    // row's dS store -> the dQ MMA, the dQ MMA's read of dS -> the next tile's dS
    // store (WAR), and the S/dP reads -> the next tile's S and dP MMAs (WAR).
    TmemFenceBeforeSync();
    __syncthreads();
    TmemFenceAfterSync();
#if defined(XATTN_DQ_ROUND_STAT)
    // ph2 closes here: the dS elementwise block, its TMEM store, and the sync.
    if (tid == 0) {
      const unsigned int now2 = static_cast<unsigned int>(clock64());
      s_ph[2] += now2 - s_mark;
      s_mark = now2;
    }
#endif

    // (5) dQ += dS K, the forward's PV loop with P->dS, V->K, O->dQ. The
    // accumulate flag resets the FIRST k-step of EVERY K slab at the first key
    // tile: each slab owns its own 64-column window of dQ, and nothing else
    // initialises it.
    if (warp == 0) {
      for (int sl = 0; sl < Plan::kSlabsK; ++sl) {
        for (int kk = 0; kk < kBN / 16; ++kk) {
          MmaTs(tbase + Plan::kDqCol + sl * 64, tbase + Plan::kDsCol + kk * 8,
                MakeSmemDesc(kb_smem + sl * Plan::kKvSlab + kk * 16 * 128, dq_lbo, 64),
                ts_idesc, (nt == 0 && kk == 0) ? 0u : 1u);
        }
      }
      MmaCommit(bar_dq);
    }
    // Also the point after which this stage's K/V may be overwritten: the dQ MMA
    // above is the last reader of K, and the dP MMA's V read completed with
    // bar_sdp. (With kStages == 1 this wait is what makes the single stage safe.)
    MbarWait(bar_dq, static_cast<uint32_t>(nt & 1));

    if (t + Plan::kStages <= t1) issue_kv(t + Plan::kStages);
#if defined(XATTN_DQ_ROUND_STAT)
    // ph3 closes here: the dQ issue and wait, and the next tile's re-issue.
    if (tid == 0) {
      const unsigned int now3 = static_cast<unsigned int>(clock64());
      s_ph[3] += now3 - s_mark;
      s_mark = now3;
    }
#endif

#if defined(XATTN_DQ_ROUND_STAT)
    // `s_rmin == 0` is the "unset" sentinel, so the first round seeds the floor and
    // no unsigned 2*UINT_MAX wrap is reachable. A round is ~1.2e4 cycles here, so
    // 2*s_rmin cannot overflow a uint; if a future bucket ever exceeds 2^31 cycles
    // per round this test is wrong and the compiler will not tell you.
    if (tid == 0) {
      const unsigned int dt = static_cast<unsigned int>(clock64() - s_t0c);
      if (s_rmin == 0u) {
        s_rmin = dt;
      } else if (dt > 2u * s_rmin) {
        ++s_nflag;
        s_excess += static_cast<unsigned long long>(dt - s_rmin);
      } else if (dt < s_rmin) {
        s_rmin = dt;
      }
      ++s_nround;
      if (s_nround == static_cast<unsigned int>(steps)) {
        printf("DQ_PHASE_STAT cta=%d of=%d steps=%d ph_kv=%llu ph_sdp=%llu ph_elem=%llu ph_dq=%llu total=%llu pe_load=%llu pe_store=%llu\n",
               static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x), steps,
               s_ph[0], s_ph[1], s_ph[2], s_ph[3],
               s_ph[0] + s_ph[1] + s_ph[2] + s_ph[3], s_pe[0], s_pe[2]);
      }
    }
#endif
  }

#if defined(XATTN_DQ_ROUND_STAT)
  // First and last CTA only, which is exactly the sample the
  // phase table draws, so this row is comparable to it rather than
  // adjacent to it. An all-CTA histogram is a different patch: this
  // one is the independent denominator, the field no phase-sum can
  // produce.
  __syncthreads();
  if (tid == 0 && (blockIdx.x == 0u || blockIdx.x + 1u == gridDim.x)) {
    printf("DQ_ROUND_STAT cta=%u of=%u steps=%d rmin=%u nround=%u nflag=%u excess=%llu\n",
           blockIdx.x, gridDim.x, steps, s_rmin, s_nround, s_nflag, s_excess);
  }
#endif

  // ---------------------------------------------------------------- store
  // Only dQ leaves this leaf. A live row whose visible set is empty (lse = -inf)
  // writes zeros: its dS row is identically zero, so the TMEM window already
  // holds exactly 0, and the select makes that a property of the code rather than
  // of the accumulator's initialisation. `p.scale` is applied HERE, after the MMA,
  // once: on this (deterministic) arm the host folds no scale at all (host.cc:341-
  // 356) and the seed's equivalent site is kernels.cuh:754-755 (see the ABI block).
  const bool row_ok = row_in_range && live;
  Elem* dqrow = row_in_range ? reinterpret_cast<Elem*>(p.dq) +
                                   ((b * p.q_length + i) * p.query_heads + hq) * p.qk_dim
                             : nullptr;
  for (int c = 0; c < kD; c += 32) {
    float o[32];
    TmemLoad32(lane_base | (tbase + Plan::kDqCol + c), o);
    TmemWaitLd();
    if (row_ok) {
      for (int j = 0; j < 32; ++j) {
        const int64_t col = c + j;
        if (col >= p.qk_dim) break;  // padding columns of the bucket are not written
        dqrow[col] = ElemTraits<Elem>::From(lse_ok ? o[j] * p.scale : 0.f);
      }
    }
  }
  if (!row_ok && row_in_range) write_dead_row();

  dealloc_and_exit();
}

// Raised, never returned: a driver or launch error is not a refusal of the accepted
// set. A host header that throws works in both worlds it is compiled into - the
// packaged extension (pybind turns std::runtime_error into a Python RuntimeError) and
// a standalone nvcc compile of an instantiation TU. The sibling key-owned leaf
// (csrc/flash_sca/native/umma_bwd_dkdv.cuh, dkdv_detail::RaiseCuda) uses the same
// idiom, so both leaves raise and are caught identically by the same caller.
namespace dq_detail {
[[noreturn]] inline void RaiseCuda(const char* what, cudaError_t err) {
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
}  // namespace dq_detail

// The ONE predicate: true exactly when LaunchUmmaDq would accept this call.
// Pure - no launch, no allocation, no stream; the only CUDA calls are the two
// device-attribute reads of the capability gate below, which are queries.
// Torch-free by construction: the element type is the template argument E, so the
// host passes the E it is about to launch and no at::ScalarType enters a device
// header (this file is also built standalone by the probe, with no torch).
// The host MUST preflight with this and launch only on true: a
// preflight that said yes where the launch refuses would clear the stage mask and
// leave dq unwritten with no error anywhere.
// The element kind a Params plane must carry for this instantiation. Params has a
// single `element` field covering every operand (params.h), so one comparison
// stands for q, k, v, prev_k, prev_v, dy and dq: the host refuses mixed dtypes
// before it builds a Params, and this refuses an instantiation that does not
// match the planes it was handed.
template <class E>
constexpr Element DqElementOf();
template <>
constexpr Element DqElementOf<__half>() {
  return Element::kHalf;
}
template <>
constexpr Element DqElementOf<__nv_bfloat16>() {
  return Element::kBFloat16;
}

template <class E, int kD, int kDv>
bool CanLaunchUmmaDq(const UmmaLaunchParams& params) {
  using Plan = DqPlan<kD, kDv>;
  const Params& hp = params.params;
  // (0a) Element kind. Every body below is compiled for one element type, and the
  // TMA descriptors are encoded with it. A bf16 call routed to an fp16
  // instantiation would read the same bytes as the wrong format: a wrong answer,
  // not a crash, so it is refused here instead of launched.
  if (hp.element != DqElementOf<E>()) return false;
  // (0b) The BUCKET, not the raw dimension, must match, and the logical width must
  // fit inside the instantiation. The host dispatches on QkBucket/ValueBucket
  // (params.h), so a 160-wide call arrives at the 256 instantiation by design and
  // must run: the TMA maps carry the real 160 columns, the columns past them are
  // clipped to zero by the descriptor, and the store is masked at p.qk_dim.
  // What cannot run is refused here, before the first launch:
  //   - a call of a DIFFERENT bucket: the TMEM slabs, the descriptor box and the
  //     store are laid out for kD / kDv columns, so a 256-wide call in a 128
  //     instantiation would write half a row and a 128-wide call in a 256
  //     instantiation would read past its slabs;
  //   - a width ABOVE the instantiation's, which bucket equality alone cannot
  //     catch: QkBucket saturates at 256, so a 2048-wide call has the same bucket
  //     as a 200-wide one and would be served by a 256-wide writer that writes the
  //     first 256 columns of a 2048-column row and leaves the rest untouched. This
  //     predicate is fail-closed on its own terms: it does not inherit the public
  //     API's <= 256 validation, because it cannot assume the public API is its
  //     only caller. A zero or negative width indexes nothing and goes with them.
  if (hp.qk_dim <= 0 || hp.value_dim <= 0) return false;
  if (hp.qk_dim > kD || hp.value_dim > kDv) return false;
  if (QkBucket(hp.qk_dim) != kD || ValueBucket(hp.value_dim) != kDv) return false;
  // (1) Shapes the kernel indexes without a further test: one 128-row CTA per
  // (batch, query-head, row-tile) and the GQA map kvh = hq / group.
  if (hp.q_length <= 0 || hp.total_length <= 0 || hp.query_heads <= 0 ||
      hp.kv_heads <= 0 || hp.group <= 0 || hp.group * hp.kv_heads != hp.query_heads) {
    return false;
  }
  // (1b) Chunk and length identity. `chunk` is the DIVISOR of the visibility bound
  // (visibility.cuh:32 `lo = (position / p.chunk - 1) * p.chunk`), so a non-positive
  // chunk is a device integer division by zero - undefined, not a crash - and the
  // contract's range starts at 1. The key axis is `prev_length` rows of previous K/V
  // followed by `k_length` current rows, which params.h:43 states as Nt = P + Nk; a
  // call that breaks the identity does not crash, it reads the previous plane for
  // every tile (prev_length > total_length) or shifts every current key row by
  // |prev_length| (prev_length < 0), i.e. a silent wrong answer.
  if (hp.chunk <= 0 || hp.k_length <= 0) return false;
  if (hp.prev_length < 0 || hp.prev_length > hp.total_length) return false;
  if (hp.total_length != hp.prev_length + hp.k_length) return false;
  // (1c) Segment planes come in pairs. The mask path is selected by the QUERY plane
  // alone (visibility.cuh:41-42 SegmentTest tests q_segment, and only then consults
  // segment_exact), after which the kernel indexes the key plane per key. A lone
  // q_segment is therefore a device read of a null pointer - so the pairing is
  // refused here rather than trusted to the caller. (The public API makes it a
  // validation error; this predicate does not assume the public API is its caller.)
  if ((hp.q_segment == nullptr) != (hp.k_segment == nullptr)) return false;
  // The split-map rule, verbatim from LaunchUmmaForward: the split
  // previous-K/V maps can only be read a whole key tile at a time, so a partial
  // previous-K/V tile must come in packed (`maps.packed_kv`). The host's packing
  // rule is the looser-of-the-two kKeyMapAlignment = 128 test, so a split map it
  // produces always satisfies this kBN test; the reverse is not true.
  if (!params.maps.packed_kv && hp.prev_length > 0 && (hp.prev_length % kBN) != 0) return false;
  // Backward-specific preconditions, also refused BEFORE the first launch
  // (dispatch_tag.cuh: a partial launch followed by a fallback writes the same
  // gradient twice and no numerical check can see it).
  //   - dY and LSE are this leaf's inputs: P cannot be formed without lse, and
  //     dP/delta without dY;
  //   - the saved output must exist: the contract's `saved` plane, or the
  //     forward's own y / y_state - the SAME expression the kernel evaluates;
  //   - dq must be writable, and dq_accum must NOT be the route (when the host
  //     supplies dq_accum no dQ kernel runs at all and the phase is tagged
  //     kSkipped).
  const void* const saved_base =
      hp.saved != nullptr ? hp.saved
                          : (hp.saved_fp32 ? static_cast<const void*>(hp.y_state)
                                           : static_cast<const void*>(hp.y));
  // `saved` is read ONLY when the preprocess has not supplied `delta` (the dY.y
  // fallback), so it is required exactly then: a call with a delta plane and no
  // saved plane must run (case A-D of the probe refused on the stricter test).
  if (hp.dy == nullptr || hp.lse_in == nullptr || hp.dq == nullptr ||
      hp.dq_accum != nullptr) {
    return false;
  }
  if (hp.delta == nullptr && saved_base == nullptr) return false;
  // Capability gate, as LaunchUmmaForward: every body this launcher starts is
  // inside `#if XATTN_UMMA_DEVICE_ASM`, so outside the compiled family the kernel
  // is EMPTY and would "succeed" by writing nothing - the one failure mode a
  // caller cannot see from the status. Fail closed.
  int cc_device = 0;
  int cc_major = 0;
  int cc_minor = 0;
  // An API error is not an eligibility verdict: fail loud, never fall back.
  cudaError_t cc_err = cudaGetDevice(&cc_device);
  if (cc_err != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ: cudaGetDevice", cc_err);
  }
  cc_err = cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, cc_device);
  if (cc_err != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ: capability major", cc_err);
  }
  cc_err = cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, cc_device);
  if (cc_err != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ: capability minor", cc_err);
  }
  // EXACT capability, not a floor: every body this launcher starts is compiled for
  // sm_103a alone, so a 10.5 or a 12.0 device must refuse here rather than run an
  // sm_103a cubin that may not be forward-compatible. The seam then falls back to
  // the generic body, which is correct everywhere.
  if (cc_major != 10 || cc_minor != 3) return false;
  // The plan's dynamic shared-memory request must fit what this device will grant
  // as an opt-in. The launcher sets cudaFuncAttributeMaxDynamicSharedMemorySize to
  // exactly Plan::kSmemRequest on every call; if the device could not grant that,
  // the attribute would fail and the launch would run with the DEFAULT limit and
  // index past its allocation. Asking first turns that into a refusal the caller
  // can fall back from. (Plan reserves kSmemSlack on top of kSmemBytes so the
  // 1024 B base rebase always fits inside the granted window.)
  int smem_optin = 0;
  const cudaError_t smem_err =
      cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cc_device);
  // Same rule: an API error is not an eligibility verdict (fail loud, never fall
  // back); only a SUCCESSFUL query below the plan's request is a refusal.
  if (smem_err != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ: shared memory capacity", smem_err);
  }
  if (smem_optin < Plan::kSmemRequest) return false;
  return true;
}

// The pipelined dQ leaf (umma_bwd_dq_pipe.cuh, instantiated for BF16 and FP16 at (256,256) in
// instantiations/sca_umma_bwd_dq_pipe_*.cu and at (128,128) in
// instantiations/sca_umma_bwd_dq_pipe_d128_v128_*.cu), which LaunchUmmaDq tries first at either
// bucket. 0 = not taken (XATTN_SCA_BWD_DQ_PIPE is "0", or its preflight refused the call:
// DqKernel runs), 1 = launched.
namespace dqpipe {
template <class E, int kD, int kDv>
int TryLaunchDqPipe(const UmmaLaunchParams& lp, cudaStream_t stream);
}  // namespace dqpipe

// Returns true when the kernel was launched and false when the call was refused.
// The refusal is a status the caller MUST branch on - never ignore it, and never
// treat p.dq as written. The refusal set is CanLaunchUmmaDq's; this function adds
// no test of its own, so the two cannot drift.
template <class E, int kD, int kDv>
bool LaunchUmmaDq(const UmmaLaunchParams& params, cudaStream_t stream) {
  using Plan = DqPlan<kD, kDv>;
  const Params& hp = params.params;
  if (!CanLaunchUmmaDq<E, kD, kDv>(params)) return false;
  // At both instantiated buckets (256,256) and (128,128) the pipelined dQ leaf serves every call
  // it accepts while XATTN_SCA_BWD_DQ_PIPE is unset, empty or "1": a level-0 call this predicate
  // accepted, on a device that grants its dynamic SMEM. It writes what DqKernel writes, bit for
  // bit - hashed in measurement runs, not asserted. A refusal launches nothing and DqKernel
  // runs below; "0" selects DqKernel for every call.
  if constexpr ((kD == 256 && kDv == 256) || (kD == 128 && kDv == 128)) {
    if (dqpipe::TryLaunchDqPipe<E, kD, kDv>(params, stream) == 1) return true;
  }
  // SEPARATE PREFLIGHT, COMMITTED LAUNCH: every
  // refusal is in CanLaunchUmmaDq and is decided before anything is issued; from
  // the kernel launch down this function cannot say no. A false returned after the
  // launch would make the driver run the generic body as well, and two bodies
  // writing one gradient is a double write no numerical check can see. Hence no
  // entry drain (a sticky error from someone else's call is not this call's
  // refusal) and exactly one status read after the launch, which RAISES: a failed
  // launch is an API error surfacing here, where it belongs, never a refusal.
  // The attribute is set on EVERY launch and its status is not ignorable: if the
  // dynamic shared memory request is refused, the launch would run with the
  // default limit and index past the allocation.
  const cudaError_t attr = cudaFuncSetAttribute(
      reinterpret_cast<const void*>(&DqKernel<E, kD, kDv>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, Plan::kSmemRequest);
  // An API error here is not an eligibility verdict: fail loud, never fall back.
  if (attr != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ: cudaFuncSetAttribute", attr);
  }
  const dim3 grid(static_cast<unsigned>((hp.q_length + kBM - 1) / kBM),
                  static_cast<unsigned>(hp.query_heads),
                  static_cast<unsigned>(hp.batch));
  DqKernel<E, kD, kDv><<<grid, kThreads, Plan::kSmemRequest, stream>>>(params);
  const cudaError_t launch_err = cudaGetLastError();
  // The one status read: an API error is not an eligibility verdict - fail loud,
  // never fall back.
  if (launch_err != cudaSuccess) {
    dq_detail::RaiseCuda("flash_sca native bwd dQ: kernel launch", launch_err);
  }
  return true;  // committed: the launch was issued (see the block above)
}

// Instantiation TUs expand this at FILE scope, so the template-id must be fully
// qualified inside the macro body - the same compile invariant
// umma_fwd.cuh records (six errors per TU otherwise, and zero instantiated
// kernels).
#define XATTN_UMMA_DQ_INSTANTIATE(E, D, DV)                                    \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaDq<E, D, DV>(    \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t); \
  template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaDq<E, D, DV>( \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
