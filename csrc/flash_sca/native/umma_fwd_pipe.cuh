// The native Blackwell (sm_103a) FlashSCA forward at
// (D, Dv) = (256, 256), (128, 128), (64, 64) and the unequal (64, 128), level 0, warp-specialized
// and pipelined across key tiles.
//
// Same work decomposition, tile walk and per-element semantics as umma_fwd.cuh's
// ForwardKernel (one CTA per (batch, query head, 128-query tile); key tiles of Pl::kKeys from
// min_lo / Pl::kKeys to max_hi / Pl::kKeys over the CTA's live rows; TileIsFull on max_lo / min_hi
// over ALL
// rows; per-element row bounds and, when SegmentTest holds, the per-key segment test; empty
// row y = 0 and LSE = -inf; the mask is a select on the exponent's argument, so zero and
// negative scales stay defined). What changes is the schedule:
//
//   warps 0 .. kSmWarps-1  softmax (4: thread t owns query row t and the 64 columns of a tile;
//                          8: warps w and w + 4 share row 32 (w & 3) + lane, columns 0..31 and
//                          32..63, row max and row sum combined through SMEM per tile)
//   warp kSmWarps          TMA producer (one thread): Q once, then K(j) and V(j) into separate
//                          rings, each stage released by its own reader's tcgen05.commit
//   warp kSmWarps + 1      MMA issuer: S(0..kSAhead-1); then per tile j: S(j+kSAhead) =
//                          Q K(j+kSAhead)^T into S buffer (j+kSAhead) % kSB, then O += P(j) V(j)
//                          once softmax(j) has arrived (SAheadFor / SBuffersFor); at a two-CTA
//                          bucket each group of MMAs and its commits is issued under one
//                          election; at a one-CTA bucket each MMA elects its own
//
// TMEM (512 columns): O FP32 at 0..255 (lane = query row); S buffer b at 256 + 64 b; the
// softmax packs P(j) (16-bit, round to nearest even) IN PLACE over its own S(j) columns, so the
// PV MMA reads A from TMEM (kind::f16 TS form, N = 256 per instruction, B = V MN-major with the
// 64-row slab hop as LBO, the backward dK pipe's form).
//
// Online softmax with a LAZY rescale (log2 domain, level 0 form P = exp2(s * scale_log2 - m)):
// a row keeps the maximum it last rescaled to (m_used); a tile whose maximum exceeds m_used by
// more than `rescale_thr` (8, so P <= 2^8) waits for the previous PV, rescales the row's
// O and l by exp2(m_used - m_new) (a select to exact 0 when that underflows) and moves m_used.
// The first visible tile of a row only sets m_used (its O row is exactly zero). LSE =
// ln2 * m_used + ln(l), which is the same quantity ForwardKernel writes.
// A row with no visible key (dead by its bounds, or live by its bounds with no key of its segment)
// keeps m_used = -inf; every P it packs is exp2(-inf + 0) = 0, so its O row and its l stay exactly
// 0, and the epilogue writes y = 0 (inv = 0 when l == 0; a dead row writes zeros through
// write_dead) and LSE = -inf (l > 0 is the only finite branch): never 0/0 and never NaN, in a CTA
// with live rows as in a CTA without any.
// Rounding: a FULL tile forms the exponent's argument as fmaf(s, scale_log2, -m) (one rounding); a
// masked tile scales first (s * scale_log2, which the -inf select needs) and then adds -m (two
// roundings). The two forms of one score can differ in the last FP32 ulp, so P and y are not
// bit-equal between them, and this body is bit-equal neither to the older leaf nor to the generic
// body; it is held to the contract's tolerances, not to their bits.
//
// Barrier phases (tile j uses K stage j % kKS, V stage j % 2, S buffer j % kSB; each barrier
// completes once per use of its slot, so a wait for use u is on parity u & 1, and every wait
// below sees the previous phase of its barrier completed and cannot be overtaken by the next):
//   k_full / v_full   producer expect_tx + TMA bytes; waited by the MMA warp
//   k_empty / v_empty commit after S(j) / PV(j), only when a later load reuses the stage;
//                     waited by the producer before K(j + kKS) / V(j + 2)
//   s[b]              commit after S(j); waited by the softmax warps
//   p[b]              kSmThreads arrivals after P(j) is stored (and O rescaled); waited by MMA
//   pv[b]             commit after PV(j); waited by MMA before S(j + kSB) (last reader of the
//                     buffer), by a rescaling softmax warp in tile j + 1, and by the epilogue
// The last commit the MMA warp issues is pv of the last tile, which the epilogue waits for, so
// no commit can arrive on this CTA's shared memory after it exits.
//
// Compiled configuration: one kernel per element type and bucket, PipeForwardKernel<E, 8, 2, 2,
// 256> and <E, 8, 2, 2, 128> (8 softmax warps, 2 K stages, 2 S buffers; the last argument is D =
// Dv and every width-dependent constant of the plan derives from it; TryLaunchPipe below).
//
// KEYS PER TILE is one more width-dependent constant (Plan::kKeys): 128 at a one-CTA bucket whose
// budgets admit it (only when kDv_ + 2 * 128 <= 512 TMEM columns and
// PipeRingBytes(W, Dv, kKS, 128) + 4096 <= 232,448 B, which holds at D = Dv = 128), 64 at every
// two-CTA bucket and at D = Dv = 256. The width is read from the two-buffer budget, so a bucket
// that goes two-CTA (kW_ <= kTwoCtaMaxWidth) never widens: D = Dv = 64 takes that limb, so it stays
// at 64 keys and does NOT inherit the 128 of D = Dv = 128. Nothing else changes with it: the TMA maps the host prepared
// are still 64-key boxes, so a 128-key slab is two boxes written contiguously into one tile, the
// softmax still loads and stores one thread's kCols = kKeys / kHalves columns
// (Ld64 / TmemStore32 at scol + col0 for the 8-warp 128-key form, Ld32 / St16 at 64 keys), and the
// packed P(j) keeps its rule - each thread's kCols keys become kCols / 2 TMEM columns at its own
// offset h * kCols (PCol). The per-element arithmetic is unchanged, so a 64-key tile's numbers are
// exactly main's; a 128-key tile's differ only where the tile boundary changes which scores are
// formed with one rounding and which with two (see the Rounding note above), and it is held to the
// contract's tolerances. Which
// calls it serves, the XATTN_SCA_FWD_PIPE arm and the per-call body slot are in
// umma_fwd_pipe_host.h, the precedence of the two forward arms is host.cc's ResolveForwardBody,
// and the launch sites are LaunchUmmaForward<E, 256, 256> and <E, 128, 128> (umma_fwd.cuh), after
// those launchers' own refusals.
#pragma once

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "flash_sca/native/dimension_bucket.h"
#include "flash_sca/native/tma_load.cuh"  // TmaStoreAtom / the staged-store fence
#include "flash_sca/native/umma_fwd.cuh"
#include "flash_sca/native/umma_fwd_pipe_host.h"

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace fwdpipe {

inline constexpr int kPipeRows = 128;  // query rows per CTA (M128: TMEM lane = row)
// Keys per tile are NOT one value: Plan::kKeys picks 128 or 64 from the budgets (see Plan::kKeys
// below). A tile is always built out of the maps' 64-key TMA boxes.
// Every (D, Dv) bucket pair of 64/128/256 has an instantiation. The plan's last two template
// arguments are the two widths - D, the K extent of S = Q K^T, and Dv, the N extent of O += P V -
// and every constant that depends on them (the 64-column slab counts of Q/K and of V, their stage
// byte sizes, keys per tile, resident CTAs, S's TMEM column and the O columns per softmax thread)
// derives from them, so another bucket costs another body and changes no arithmetic for the others.
inline constexpr int kPipeWidthMax = 256;
inline constexpr int kSmBarrierId = 5;  // named barrier of all softmax threads (the staged O store's
                                        // hand-off to the elected TMA issuer; the seg staging this
                                        // label used to name is no longer a use of it)
inline constexpr int kInitBarrierId = 6;  // named barrier of the producer and MMA warps (prologue)

// SMEM bytes of the Q tile and the K (kKS stages) and V (2 stages) rings at `keys` per tile.
constexpr int PipeRingBytes(int d, int dv, int kks, int keys) {
  return (d / 64) * kPipeRows * 128 + kks * (d / 64) * keys * 128 + 2 * (dv / 64) * keys * 128;
}

template <int kSmWarps_, int kKS_, int kSB_, int kW_, int kDv_ = kW_>
struct Plan {
  static constexpr int kSmWarps = kSmWarps_;
  static constexpr int kKS = kKS_;  // K stages
  static constexpr int kVS = 2;     // V stages
  static constexpr int kSB = kSB_;  // S buffers in TMEM
  static constexpr int kPipeWidth = kW_;         // D of the bucket: the K extent of S = Q K^T
  static constexpr int kPipeWidthO = kDv_;       // Dv of the bucket: the N extent of O += P V
  static constexpr int kPipeSlabs = kW_ / 64;    // 64-column TMA slabs of Q and K per row
  static constexpr int kPipeSlabsV = kDv_ / 64;  // ... of V
  // Resident CTAs per SM and keys per tile, from ONE budget rule over (D, Dv), not per shape:
  //   two CTAs   when the score is at most kTwoCtaMaxWidth wide and O plus a 64-key S ring fit
  //              in 256 TMEM columns (kDv + kSB * 64 <= 256): (64,64), (64,128), (128,64) and
  //              (128,128); then 64 keys per tile and 256 columns each, so the two allocations
  //              fit side by side in the SM's 512 (a 128-key tile would need kDv + 2*128 > 256).
  //   one CTA    otherwise; then 128 keys per tile when TMEM (kDv + 2 * 128 <= 512, i.e. room for
  //              O plus two 128-key S buffers) and the SMEM of a 128-key ring (PipeRingBytes) fit,
  //              else 64: (256,256) is 65,536 + 2*4*16,384 + 2*4*16,384 = 327,680 B (64 keys).
  //              The key-tile width is decided from the two-buffer budget on purpose, so that it
  //              does not move when SBuffersFor (below) spends a third buffer: one rule gives the
  //              width first and the S-ring depth second, and a deeper ring never narrows a tile.
  // kTwoCtaMaxWidth is the rule's only measured input: 128, from the gate's rounds of the two-CTA
  // form (D = 64: 15-22% over one CTA at 128 keys, in two paired rounds; D = 128:
  // 4-8% on the D = 128 rows). D = 256 is excluded by the TMEM limb itself (256 + 128 > 256).
  // XATTN_FWD_PIPE_ONE_CTA (a probe define, in no build) turns the two-CTA form off, for the
  // paired measurement of the two forms.
#if defined(XATTN_FWD_PIPE_ONE_CTA)
  static constexpr int kTwoCtaMaxWidth = 0;
#else
  static constexpr int kTwoCtaMaxWidth = 128;
#endif
  static constexpr bool kTwoCtas = kW_ <= kTwoCtaMaxWidth && kDv_ + kSB_ * 64 <= 256;  // TMEM limb
  static constexpr int kKeys =
      kTwoCtas ? 64
               : (kDv_ + 2 * 128 <= 512 && PipeRingBytes(kW_, kDv_, kKS_, 128) + 4096 <= 232448
                      ? 128
                      : 64);
  static constexpr int kSmThreads = 32 * kSmWarps;
  static constexpr int kProducerWarp = kSmWarps;
  static constexpr int kMmaWarp = kSmWarps + 1;
  static constexpr int kThreads = 32 * (kSmWarps + 2);
  static constexpr int kQSlabBytes = kPipeRows * 128;   // 16,384
  static constexpr int kKvSlabBytes = kKeys * 128;                // keys x 64 dims x 2 B
  static constexpr int kQBytes = kPipeSlabs * kQSlabBytes;        // 65,536 at D = 256
  static constexpr int kKTileBytes = kPipeSlabs * kKvSlabBytes;
  static constexpr int kVTileBytes = kPipeSlabsV * kKvSlabBytes;  // TMA bytes of one V stage
  static constexpr int kQOff = 0;
  static constexpr int kKOff = kQOff + kQBytes;
  static constexpr int kVOff = kKOff + kKS * kKTileBytes;
  static constexpr int kBarQ = 0;
  static constexpr int kBarKFull = 1;
  static constexpr int kBarKEmpty = kBarKFull + kKS;
  static constexpr int kBarVFull = kBarKEmpty + kKS;
  static constexpr int kBarVEmpty = kBarVFull + kVS;
  static constexpr int kBarS = kBarVEmpty + kVS;
  static constexpr int kBarP = kBarS + kSB;
  static constexpr int kBarPv = kBarP + kSB;
  static constexpr int kBarCount = kBarPv + kSB;
  static constexpr int kXchBytes = kSmWarps == 8 ? 2 * 2 * kPipeRows * 4 : 0;
  // SMEM without the row sum's ones slabs (the layout below with kVStageBytes = kVTileBytes).
  static constexpr int kSmemBaseBytes =
      kVOff + kVS * kVTileBytes + ((kBarCount * 8 + 15) / 16) * 16 + 16 + 4 * 16 + kXchBytes;
  // The softmax row sum on the tensor core, one budget rule over (D, Dv). It holds where the PV
  // MMA may grow to N = Dv + 16 (N <= 256), the CTA's TMEM has those 16 columns beside O and the
  // S ring, and SMEM has one more 64-column slab per V stage within the ceiling and the residency
  // share. There each V stage carries a slab of 16-bit ones after its Dv / 64 TMA slabs, so the
  // same PV instructions also accumulate L = P x ones into TMEM column Dv: the row sum of the very
  // 16-bit P that O sums, with no extra MMA issue or commit. The softmax then drops its per-element
  // row-sum adds and the 8-warp lsum exchange; L is rescaled with O. Today it takes (64,64),
  // (128,64), (256,64) and (256,128); (128,128) and (64,128) have no free TMEM columns and every
  // Dv = 256 bucket would need N > 256. A bucket whose TMEM columns go to another use (a second
  // O block, or an S ring deepened to kSB 3 at two CTAs) takes no slab, by the same column limb.
  static constexpr bool kRowSumMma =
      kDv_ + 16 <= 256 && kDv_ + 16 + kSB_ * kKeys <= (kTwoCtas ? 256 : 512) &&
      kSmemBaseBytes + kVS * kKvSlabBytes <= 232448 &&
      (kTwoCtas ? 2 : 1) * (kSmemBaseBytes + kVS * kKvSlabBytes + 1024) <= 233472;
  static constexpr int kVStageBytes = kVTileBytes + (kRowSumMma ? kKvSlabBytes : 0);
  static constexpr int kBarOff = kVOff + kVS * kVStageBytes;
  static constexpr int kSlotOff = kBarOff + ((kBarCount * 8 + 15) / 16) * 16;
  static constexpr int kRedOff = kSlotOff + 16;
  static constexpr int kXchOff = kRedOff + 4 * 16;          // 4 row warps x 4 int32
  // No staged segment table: the k_segment plane is read at the mask test, in int64. A staged
  // table would be 2 * kKeys * 8 B (int64) and a third K stage does not fit beside it, and it may
  // not be narrowed to int32: the ids are the caller's - host.cc copies q_segment/k_segment as
  // given, contiguity only, and `_segment_metadata.py` validates shape and dtype, not range - so
  // ids that differ by 2^32, or an id whose low word is 0x80000000, would compare equal to each
  // other or to a sentinel and a key of another segment would become visible (silent wrong y).
  // Freeing the table entirely is what lets a third K stage fit the sm_103a SMEM ceiling, and it
  // costs the timed rows nothing: a BOS-masked call hands the kernel no segment planes, so
  // SegmentTest is false and the plane is never read there.
  static constexpr int kSmemBytes = kXchOff + kXchBytes;
  // TMEM columns: O at 0..Dv-1, then (kRowSumMma) L at Dv..Dv+15 (the PV accumulator's tail),
  // then the S ring.
  static constexpr int kOCol = 0;
  static constexpr int kLCol = kPipeWidthO;
  static constexpr int kPvN = kPipeWidthO + (kRowSumMma ? 16 : 0);
  static constexpr int kSCol = kPvN;
  static constexpr int kBufCols = kKeys;
  // Q resident in TMEM, one-CTA buckets only: the A operand of S = Q K^T is then a TMEM tile
  // (the TS form PV already uses) instead of an SMEM descriptor, which is worth what the S group
  // spends on the shared-memory read port: at N = 64 it reads 6 KB per 16-deep step, 48 cycles at
  // 128 B/clk against 32 tensor cycles (a microbenchmark: S alone 952 cyc/tile in the real SS
  // form, 516 with A in TMEM; the S+PV floor 1,265 -> 1,028). An A tile of kD half-precision dims
  // is kD/2 columns (a TMEM lane holds 2 values per column), and 8 columns are consumed per 16-deep
  // step. XATTN_H49_QDIV picks the arm: 0 the SS form (the shipped body, and every two-CTA bucket
  // whatever it says), 2 full Q (all kD dims, kD/2 columns), 4 half Q (dims 0..kD/2-1 in TMEM,
  // kD/4 columns, the rest SS - it keeps one more S buffer and halves the port pressure of A).
#ifndef XATTN_H49_QDIV
#define XATTN_H49_QDIV 2
#endif
  static constexpr int kQWantCols =
      (!kTwoCtas && XATTN_H49_QDIV) ? (kPipeWidth * 2 / XATTN_H49_QDIV) / 2 : 0;
  // Q's columns are spent only where the S ring keeps its two-buffer floor beside them and beside
  // O with its row-sum columns (kSCol = kPvN, which counts the 16 L columns where kRowSumMma): a bucket
  // whose kKeys leave less than that stays in the SS form (e.g. (64,256), whose 128-key buffers do
  // not fit beside Q's 32 columns). kSCol, kKeys and 512 do not depend on kSB, so this test is not
  // circular with SAheadFor / SBuffersFor below.
  static constexpr bool kQResident =
      kQWantCols > 0 && kSCol + 2 * kKeys + kQWantCols <= 512;
  static constexpr int kQTmemCols = kQResident ? kQWantCols : 0;  // 8 per 16-deep step
  static constexpr int kQCol = kSCol + kSB * kBufCols;  // after O and the S ring
  // TMEM allocation. A one-CTA plan takes the SM's whole 512 columns; that is safe only while no
  // second CTA of this kernel (or another TMEM user) is resident on the SM, which a one-CTA plan's
  // __launch_bounds__(..., 1) and its SMEM (199,888 B at (256,256)) keep true at D = 256.
  // kCtasPerSm is the kernel's __launch_bounds__ minimum (kTwoCtas above): a two-CTA plan
  // allocates 256 columns, so the two allocations fit side by side and neither CTA waits in
  // tcgen05.alloc. The three limbs of a two-CTA plan, checked by the asserts below or by the build:
  //   TMEM       kSCol + kSB * kBufCols <= 256 per CTA (O at Dv columns plus the S ring);
  //   SMEM       2 * (kSmemBytes + 1024) <= 233,472 B;
  //   registers  65,536 / (2 * kThreads) = 102.4 per thread (96 after ptxas' granularity): the
  //              two-CTA bodies compile at 85-96 (see the PR's register table), LOCAL 0.
  //              At one CTA the budget is 204.8, above the 136 the widest body uses.
  static constexpr int kCtasPerSm = kTwoCtas ? 2 : 1;
  static constexpr int kAllocCols = kCtasPerSm == 2 ? 256 : 512;
  // <E, 4 or 8, 2, 2, D, Dv> are instantiated (TryLaunchPipe, one TU per (D, Dv) bucket pair in
  // instantiations/sca_umma_fwd_d*_v*.cu, softmax warps by SmWarpsFor): the 4-softmax-warp form
  // (thread = row) at two-CTA buckets whose MMA work per tile covers the softmax, 8 elsewhere.
  // 3 K stages / 3 S buffers were tuning arms of an earlier probe that nothing here compiles or
  // validates; widening those asserts needs its own validation.
  static_assert(kSmWarps == 4 || kSmWarps == 8, "4 (thread = row) or 8 (two threads per row) softmax warps");
  static_assert(kKS == 2 || kKS == 3, "2 or 3 K stages: KStagesFor's SMEM budget decides");
  static_assert(kSB >= 2 && kSB <= 4, "2 to 4 S buffers: SBuffersFor's TMEM budget decides");
  static_assert((kPipeWidth == 64 || kPipeWidth == 128 || kPipeWidth == 256) &&
                    (kPipeWidthO == 64 || kPipeWidthO == 128 || kPipeWidthO == 256),
                "D and Dv buckets are 64, 128 or 256 (every pair is instantiated)");
  static_assert(kSCol + kSB * kBufCols + kQTmemCols <= kAllocCols, "TMEM column budget");
  static_assert(kCtasPerSm * kAllocCols <= 512, "resident CTAs share the SM's 512 TMEM columns");
  static_assert(kCtasPerSm * (kSmemBytes + 1024) <= 233472,
                "resident CTAs share the SM's shared memory");
  static_assert(kQOff % 1024 == 0 && kKOff % 1024 == 0 && kVOff % 1024 == 0 &&
                    kKTileBytes % 1024 == 0 && kVTileBytes % 1024 == 0 && kKvSlabBytes % 1024 == 0,
                "descriptor tile bases must stay 1024-aligned");
static_assert(kRedOff % 8 == 0 && kXchOff % 8 == 0, "SMEM tables stay 8-aligned");
  // The epilogue may stage the O tile (kPipeRows rows x kPipeWidthO) over the Q/K/V region, which
  // is dead once the last PV has completed: Q's last reader is the last S MMA and the K/V rings'
  // last reader is the PV of the last tile, and the staged store follows the wait for that PV.
  // The stage must also stay below kBarOff, where the mbarriers and the row tables live. Not every
  // bucket stages (kStageO in the kernel); the bound is asserted for every bucket.
  static_assert(kPipeRows * kPipeWidthO * 2 <= kBarOff, "the staged O tile fits the dead Q/K/V region");
  static_assert(kSmemBytes <= 232448, "the sm_103a opt-in dynamic SMEM ceiling");
};

// ---------------------------------------------------------------- device primitives
template <class E>
struct PackOf;
template <>
struct PackOf<__nv_bfloat16> {
  __device__ __forceinline__ static uint32_t Pack2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
  }
  __device__ __forceinline__ static __nv_bfloat16 From(float x) { return __float2bfloat16_rn(x); }
};
template <>
struct PackOf<__half> {
  __device__ __forceinline__ static uint32_t Pack2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
  }
  __device__ __forceinline__ static __half From(float x) { return __float2half_rn(x); }
};

__device__ __forceinline__ float Ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Packed FP32 pair arithmetic (sm_100+ FFMA2 / FADD2 / FMUL2), the same form the two-tile body
// uses (umma_fwd_pp.cuh, Ffma2/Fadd2): each lane performs exactly the scalar instruction on its
// half, so a pair op gives the scalar op's bits and costs one instruction where two scalar ops
// cost two. The softmax tile's per-element work is one scale-subtract, one exp2, one accumulate
// and half a pack per score, and the exp2 (MUFU) has no packed form - so packing the other three
// is what is left to cut on that path.
__device__ __forceinline__ float2 Ffma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{\n\t.reg .b64 ra, rb, rc, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\tmov.b64 rc, {%6, %7};\n\t"
      "fma.rn.f32x2 rd, ra, rb, rc;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}
__device__ __forceinline__ float2 Fadd2(float2 a, float2 b) {
  float2 d;
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "add.rn.f32x2 rd, ra, rb;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 Fmul2(float2 a, float2 b) {
  float2 d;
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "mul.rn.f32x2 rd, ra, rb;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}

__device__ __forceinline__ void Arrive(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mbar) : "memory");
#endif
}

__device__ __forceinline__ void NamedSync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// 32 lanes x 64 columns, then tcgen05.wait::ld, in ONE asm statement: no use of the loaded
// registers can be scheduled between the load and its wait.
__device__ __forceinline__ void Ld64(uint32_t addr, float (&v)[64]) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t r[64];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]),
        "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]),
        "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]),
        "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]),
        "=r"(r[62]), "=r"(r[63])
      : "r"(addr)
      : "memory");
#pragma unroll
  for (int x = 0; x < 64; ++x) v[x] = __uint_as_float(r[x]);
#else
  (void)addr;
#pragma unroll
  for (int x = 0; x < 64; ++x) v[x] = 0.f;
#endif
}

__device__ __forceinline__ void Ld32(uint32_t addr, float (&v)[32]) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(addr)
      : "memory");
#pragma unroll
  for (int x = 0; x < 32; ++x) v[x] = __uint_as_float(r[x]);
#else
  (void)addr;
#pragma unroll
  for (int x = 0; x < 32; ++x) v[x] = 0.f;
#endif
}

// 32 lanes x 16 columns, warp-collective (32x32b.x16), no wait.
__device__ __forceinline__ void St16(uint32_t addr, const uint32_t (&v)[16]) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};" ::"r"(v[0]),
      "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]),
      "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]),
      "r"(addr)
      : "memory");
#else
  (void)addr;
  (void)v;
#endif
}

// One TMEM column per lane (32x32b.x1), with its wait in the same asm statement; and its store.
// Used only for the row-sum column L (read by the epilogue, rescaled with O).
__device__ __forceinline__ float Ld1(uint32_t addr) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t r;
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r)
      : "r"(addr)
      : "memory");
  return __uint_as_float(r);
#else
  (void)addr;
  return 0.f;
#endif
}
__device__ __forceinline__ void St1(uint32_t addr, float v) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(addr),
               "r"(__float_as_uint(v))
               : "memory");
#else
  (void)addr;
  (void)v;
#endif
}

// TMEM column of the packed P(j) for the 16-key k-step kk of buffer column `scol`. Each softmax
// thread owns kCols keys (kCols / 16 k-steps) and writes kCols / 2 columns of 16-bit P at its own
// half's offset h * kCols; within a thread's block the k-steps are 8 columns apart, so the form is
// the same for 4 warps (one 64-key block per thread), 8 warps at 64 keys (2 steps each) and
// 8 warps at 128 keys (4 steps each).
template <int kSmWarps, int kCols>
__device__ __forceinline__ uint32_t PCol(uint32_t scol, int kk) {
  constexpr int kPer = kCols / 16;  // 16-key k-steps per softmax thread
  return scol + static_cast<uint32_t>(kCols) * static_cast<uint32_t>(kk / kPer) +
         8u * static_cast<uint32_t>(kk % kPer);
}

// The issuer's MMA forms with the descriptors passed as precomputed 64-bit words and without the
// all-zero disable-output-lane vector (optional in tcgen05.mma; zero disables nothing), so the
// issue path carries no per-MMA descriptor rebuild and no per-MMA zero materialisation. One
// elected lane issues; the instruction issued is the same as MmaSs / MmaTs / MmaCommit
// (umma_common_local.h, shared with the backward, unchanged).
//
// First try (measured, refuted): folding the elect and the MMA into one asm block guarded
// by `@pe` instead of `if (!ElectOne()) return;` left the WARPSYNC.COLLECTIVE/ELECT/R2UR SASS
// counts bit-for-bit unchanged and added ~0.6% more instructions (CPU sm_103a build,
// <8,3,3,256,256> and <8,2,2,64,64> bf16: WARPSYNC 270/172, ELECT 464/158, R2UR 2544/842, all
// identical before/after; total SASS lines +288/+160). tcgen05.mma is a single-thread
// instruction at the hardware level, so predicating it with `@p` does not let ptxas avoid the
// divergent-branch-plus-reconvergence it already builds for `if (!ElectOne()) return;`: the two
// forms compile to the same control structure.
//
// Second try (this one): elect ONCE per group instead of once per MMA. `FwdMmaSs` /
// `FwdMmaTs` / `FwdMmaCommit` still do the per-call elect (unchanged below, so every other call
// site - none remain after this change - keeps working), but the issuer's `issue_s` / `issue_pv`
// closures (further down this file) now wrap their whole run of MMA issues and commits - 16 S
// issues + 2 commits, or 4 PV issues + 2 commits - in ONE `if (ElectOne())`, calling the `*Raw`
// forms below that assume the caller already elected. This turns 18 branches (S) / 6 branches
// (PV) per tile into 1 each: the same number of lanes diverge, for the same instructions, in the
// same order, but ptxas has one join to reconverge instead of many, so (if a fallback is paid at
// all) it is paid once per tile instead of once per MMA. Bit-identical by construction: the
// elected lane and the instruction stream it issues are unchanged.
__device__ __forceinline__ void FwdMmaSs(uint32_t dtmem, uint64_t a, uint64_t b, uint32_t idesc,
                                         uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  if (!ElectOne()) return;
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(dtmem),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate));
#endif
}

__device__ __forceinline__ void FwdMmaTs(uint32_t dtmem, uint32_t atmem, uint64_t b, uint32_t idesc,
                                         uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  if (!ElectOne()) return;
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(dtmem),
      "r"(atmem), "l"(b), "r"(idesc), "r"(accumulate));
#endif
}

// The Q tile copied from shared memory into TMEM, so that S = Q K^T can take A from TMEM (the
// TS form, FwdMmaTsRaw) instead of from an SMEM descriptor. One .128x256b copy moves 32 bytes of
// every one of the 128 rows (256 b = 8 TMEM columns), so a kD-wide bf16/f16 row of kD * 2 bytes
// needs kD/16 copies, each landing 8 columns further; the source descriptor is the SAME K-major
// 128B-swizzle descriptor the SS-form A uses, so the per-step arithmetic is the one the S loop
// already does (slab * kQSlabBytes + within * 32). PTX 9.7.16.9.2: a single thread of a single CTA
// issues tcgen05.cp, and a copy followed by an MMA from that same thread needs no fence and no wait
// (9.7.16.6.2 pairing 2: cp -> mma is implicitly ordered; tcgen05.wait::st is for tcgen05.st). The
// issuer elects once for its whole group of issues, so this is the Raw form.
__device__ __forceinline__ void FwdCopyQToTmemRaw(uint32_t qtmem, uint64_t qdesc) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t" ::"r"(qtmem), "l"(qdesc));
#endif
}

// Forward-local commit: one elected lane, as MmaCommit (umma_common_local.h, shared with the
// backward, unchanged) does.
__device__ __forceinline__ void FwdMmaCommit(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  if (ElectOne()) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar)
        : "memory");
  }
#endif
}

// Raw forms: no election. The caller has already tested ElectOne() once for the whole group of
// issues these sit in, so every lane that reaches these has already been selected - issuing the
// elect again per instruction would be redundant (and, per the first try above, would not remove
// anything ptxas emits anyway).
__device__ __forceinline__ void FwdMmaSsRaw(uint32_t dtmem, uint64_t a, uint64_t b, uint32_t idesc,
                                            uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(dtmem),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate));
#endif
}

__device__ __forceinline__ void FwdMmaTsRaw(uint32_t dtmem, uint32_t atmem, uint64_t b,
                                             uint32_t idesc, uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(dtmem),
      "r"(atmem), "l"(b), "r"(idesc), "r"(accumulate));
#endif
}

__device__ __forceinline__ void FwdMmaCommitRaw(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar)
      : "memory");
#endif
}

// A ring position carried as a counter: stage index and the parity of the stage's current use, so
// the waits need no j % kStages / j / kStages (non-power-of-two at three stages).
template <int kStages>
struct RingPos {
  int st = 0;
  uint32_t ph = 0;
  __device__ __forceinline__ void Next() {
    if (++st == kStages) {
      st = 0;
      ph ^= 1u;
    }
  }
};

#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
// Probe builds only (XATTN_FWD_PROBE_HOOKS, see umma_fwd_pipe_host.h; never in a shipped build):
// lazy-rescale events and, among them, the events whose factor underflowed to exactly 0 (the clear
// branch), one per softmax thread that takes the branch (both threads of a row in the 8-warp
// form), read back by TryLaunchPipe after every launch of a probe build.
static __device__ unsigned long long g_pipe_rescale_events = 0;
static __device__ unsigned long long g_pipe_clear_events = 0;
#endif

// The issuer's S issue distance: S(j + kSAhead) is issued in PV(j)'s iteration. It is the shipped
// distance, unchanged: two tiles where ONE CTA is resident and O plus three S buffers fit its TMEM
// (kDv + 3 * kKeys <= kAllocCols), one tile elsewhere (the two-CTA buckets, where the other CTA's
// MMAs fill the pipe, and the one-CTA buckets whose 128-key tile leaves no room). The S RING depth
// is decided from it, below (SBuffersFor): the distance and the depth are separate knobs.
template <int kD, int kDv>
constexpr int SAheadFor() {
#if defined(XATTN_PIPE_SB2)  // measurement build only: the two-buffer, one-tile-ahead schedule
  return 1;
#else
  using P2 = Plan<8, 2, 2, kD, kDv>;  // kCtasPerSm, kKeys, kQTmemCols and kAllocCols do not depend on kSB
  return (P2::kCtasPerSm == 1 && P2::kSCol + P2::kQTmemCols + 3 * P2::kKeys <= P2::kAllocCols) ? 2 : 1;
#endif
}

// ---------------------------------------------------------------- kernel
template <class E, int kSmWarps, int kKS, int kSB, int kW, int kWv = kW>
__global__ void __launch_bounds__(32 * (kSmWarps + 2), Plan<kSmWarps, kKS, kSB, kW, kWv>::kCtasPerSm)
    PipeForwardKernel(const __grid_constant__ UmmaLaunchParams lp, const float rescale_thr) {
  using Pl = Plan<kSmWarps, kKS, kSB, kW, kWv>;
  constexpr int kHalves = kSmWarps / 4;
  constexpr int kCols = Pl::kKeys / kHalves;         // S columns per softmax thread
  constexpr int kOCols = Pl::kPipeWidthO / kHalves;  // O columns per softmax thread
  // Which form the epilogue's O store takes, one rule over the plan's residency and the warp form.
  // A softmax thread writes kOCols columns of an O row = kOCols * 2 bytes. A one-CTA bucket, or a
  // thread whose slice is at most one 128 B line, stages O into the dead Q/K/V region and writes it
  // with one TMA store per 64-column slab. A two-CTA bucket whose thread slice spans two lines keeps
  // the direct per-row store, which avoids the staged path's serial tail (one named barrier, one
  // elected issuer, `wait_group.read 0`). The slice follows the warp form, so a bucket whose warp
  // count changes takes its form from this rule without a separate choice.
  constexpr bool kStageO = Pl::kCtasPerSm == 1 || (kOCols * 2 <= 128);
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem[];

  unsigned char* const sq = smem + Pl::kQOff;
  unsigned char* const sk = smem + Pl::kKOff;
  unsigned char* const sv = smem + Pl::kVOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Pl::kRedOff);
  float* const xch = reinterpret_cast<float*>(smem + Pl::kXchOff);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hq = blockIdx.y;
  const int qt = blockIdx.x;
  const int64_t kvh = p.group > 0 ? hq / p.group : hq;
  auto bar = [&](int idx) -> uint32_t { return bar0 + 8u * static_cast<uint32_t>(idx); };

  // The walk's top key tile, from the geometry alone: a query row at position p sees only keys
  // <= p, so the tile of the CTA's last in-range row is an upper bound on the walk's top tile on
  // EVERY route - BOS mask, segment_idx, the q/k segment pair and strict_past each can only lower
  // a row's `hi`. Making it the walk's top tile is therefore exact when nothing clips that row,
  // and costs one extra, fully masked S+PV group when something does; what it buys is the diagonal
  // tile's address at CTA entry, with no row-bounds reduction and no barrier, which is what lets
  // the producer issue K/V for step 0 with Q.
  const int64_t q_row0 = static_cast<int64_t>(qt) * kPipeRows;
  const int64_t q_last = std::min<int64_t>(q_row0 + kPipeRows - 1, p.q_length - 1);
  const int kTopTile =
      (q_last < q_row0)
          ? -1
          : static_cast<int>(std::min<int64_t>(q_last + p.total_length - p.q_length,
                                               p.total_length - 1) /
                             Pl::kKeys);
  // Set by the producer warp's prologue: the walk's step 0 (the tile above) is already in stage 0
  // of the K and V rings, so the loop starts at step 1. Never true for a CTA with no in-range row.
  bool pre_issued = false;
  // Prologue. Q's address needs no row bound (one 128-row slab of one (b, hq)), so the producer
  // initialises Q's barrier itself and issues Q's TMA first, before anything else of the CTA; the
  // MMA warp initialises the other barriers and allocates TMEM; the softmax warps only load the
  // row bounds. The three run side by side, and Q's copy is in flight while the bounds are reduced.
  // Nothing here changes what any later phase computes.
  if (warp == Pl::kProducerWarp) {
    if (lane == 0) {
      // The descriptors encode base_offset = 0: every operand tile must sit on a 1024 B boundary.
      if ((SmemAddr(smem) & 1023u) != 0u) __trap();
      MbarInit(bar(Pl::kBarQ), 1);
      MbarInitFence();
      MbarArriveExpectTx(bar(Pl::kBarQ), Pl::kQBytes);
      for (int s = 0; s < Pl::kPipeSlabs; ++s) {
        for (int r = 0; r < kPipeRows; r += kTmaRows) {
          TmaLoad4D(sq + s * Pl::kQSlabBytes + r * 128, &lp.maps.q, bar(Pl::kBarQ), s * 64,
                    static_cast<int>(hq), qt * kPipeRows + r, static_cast<int>(b));
        }
      }
    }
    // The walk's top tile is known here from the geometry (kTopTile), so step 0 of the walk - the
    // diagonal tile - can be issued now with Q. No reduction, no table, no barrier: the only
    // precondition is that the MMA warp has initialised the K/V barriers these TMAs arrive on.
    pre_issued = kTopTile >= 0;
    // The K/V barriers are the MMA warp's: wait for its init, not for the CTA-wide barrier the
    // row-bounds reduction publishes through. Both warps arrive here unconditionally.
    NamedSync(kInitBarrierId, 64);
    if (lane == 0 && pre_issued) {
      // Stage 0 of the K and V rings for the walk's step 0. load() is not reachable from here (it
      // is defined after the reduction), and it would only repeat this: no empty-bar wait (step 0
      // is the ring's first use), the same box layout, the same maps.
      const int64_t kb0 = static_cast<int64_t>(kTopTile) * Pl::kKeys;
      const CUtensorMap* kmap;
      const CUtensorMap* vmap;
      int row0;
      ResolveKeyMap(lp, kb0, &kmap, &vmap, &row0);
      MbarArriveExpectTx(bar(Pl::kBarKFull), static_cast<uint32_t>(Pl::kKTileBytes));
      for (int s = 0; s < Pl::kPipeSlabs; ++s) {
        for (int kx = 0; kx < Pl::kKeys / 64; ++kx) {
          TmaLoad4D(sk + s * Pl::kKvSlabBytes + kx * 8192, kmap, bar(Pl::kBarKFull), s * 64,
                    static_cast<int>(kvh), row0 + kx * 64, static_cast<int>(b));
        }
      }
      MbarArriveExpectTx(bar(Pl::kBarVFull), static_cast<uint32_t>(Pl::kVTileBytes));
      for (int s = 0; s < Pl::kPipeSlabsV; ++s) {
        for (int kx = 0; kx < Pl::kKeys / 64; ++kx) {
          TmaLoad4D(sv + s * Pl::kKvSlabBytes + kx * 8192, vmap, bar(Pl::kBarVFull), s * 64,
                    static_cast<int>(kvh), row0 + kx * 64, static_cast<int>(b));
        }
      }
    }
  } else if (warp == Pl::kMmaWarp) {
    if (lane == 0) {
      for (int x = 0; x < Pl::kBarCount; ++x) {
        if (x == Pl::kBarQ) continue;  // the producer's
        const bool is_p = x >= Pl::kBarP && x < Pl::kBarP + kSB;
        MbarInit(bar(x), is_p ? Pl::kSmThreads : 1);
      }
      MbarInitFence();
    }
    __syncwarp();
    // Publish the barrier init alone, to the producer warp only, so its first K/V box (stage 0 of
    // both rings) can leave before the softmax warps' row-bounds reduction reaches __syncthreads.
    // Two warps, 64 threads - the same 64 that arrive at this id in the producer's block above.
    NamedSync(kInitBarrierId, 64);
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }

  // Row ownership: softmax warp w holds row 32 (w & 3) + lane (TMEM lane group w & 3).
  const bool is_sm = warp < kSmWarps;
  const int g = warp & 3;
  const int h = (warp >> 2) & 1;
  const int row = 32 * g + lane;
  const int64_t i = static_cast<int64_t>(qt) * kPipeRows + row;
  const bool row_in_range = is_sm && i < p.q_length;
  const detail::Bounds bd = is_sm ? detail::RowBounds(p, b, i) : detail::Bounds{INT_MAX, -1};
  const bool live = row_in_range && bd.lo <= bd.hi;
  if (warp < 4) {
    // ForwardKernel's reduction, verbatim: min_lo/max_hi over live rows drive the walk,
    // max_lo/min_hi over ALL rows drive the FULL test (a dead row never allows it).
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
  if constexpr (Pl::kRowSumMma) {
    // B's ones slab of every V stage. Generic stores read by the tensor core: each writer fences
    // the async proxy before the CTA barrier the issuer passes before its first MMA.
    if (tid < Pl::kSmThreads) {
      const uint32_t one2 = PackOf<E>::Pack2(1.f, 1.f);
      for (int st = 0; st < Pl::kVS; ++st) {
        uint32_t* const ones =
            reinterpret_cast<uint32_t*>(smem + Pl::kVOff + st * Pl::kVStageBytes + Pl::kVTileBytes);
        for (int x = tid; x < Pl::kKvSlabBytes / 4; x += Pl::kSmThreads) ones[x] = one2;
      }
      FenceAsyncShared();
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

  const int ocol0 = h * kOCols;
  auto write_dead = [&](bool with_lse, bool with_y) {
    // ForwardKernel's empty-row convention: y = 0, LSE = -inf (level 0 writes no state).
    // `with_y` is taken only where nothing is staged: an empty row inside a CTA that has a live row
    // gets its zeros from the staged O tile and the elected thread's TMA store, and a second plain
    // store to the same global bytes would be an unordered writer against that store.
    if (!row_in_range) return;
    if (with_y) {
      E* const yrow =
          reinterpret_cast<E*>(p.y) + ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
      if (p.value_dim == Pl::kPipeWidthO) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        for (int c = ocol0; c < ocol0 + kOCols; c += 8) *reinterpret_cast<uint4*>(yrow + c) = z;
      } else {
        for (int64_t c = ocol0; c < ocol0 + kOCols && c < p.value_dim; ++c) {
          yrow[c] = PackOf<E>::From(0.f);
        }
      }
    }
    if (with_lse && p.lse != nullptr) p.lse[(b * p.query_heads + hq) * p.q_length + i] = -INFINITY;
  };

  if (min_lo > max_hi) {
// No live row in this CTA: nothing but Q (and the prologue's pre-issued step 0, on the path that
    // pre-issues it) is loaded, and nothing is issued or staged, so this path writes y = 0 itself.
    if (is_sm) write_dead(h == 0, true);
    if (warp == Pl::kProducerWarp && lane == 0) {
      MbarWait(bar(Pl::kBarQ), 0u);
      // A CTA with rows but no visible key still pre-issued step 0's K and V: they land in this
      // CTA's shared memory, so they must complete before the CTA may exit.
      if (pre_issued) {
        MbarWait(bar(Pl::kBarKFull), 0u);
        MbarWait(bar(Pl::kBarVFull), 0u);
      }
    }
  } else {
    const int t0 = min_lo / Pl::kKeys;
    // The top tile is the geometry one the prologue issued, which is >= max_hi / kKeys on every
    // route, so the producer's pre-issued stage 0 is exactly step 0 of this walk.
    const int t1 = kTopTile;
    const int n = t1 - t0 + 1;

    // The walk runs DESCENDING: tile index t1 - j at step j, so step 0 is the diagonal tile, the
    // one holding this CTA's own key positions. The mask and the visibility are positional, so
    // only the order changes; what the order buys is that the producer's first TMA box, the
    // issuer's first S and the softmax's first P all describe the tile whose address depends on
    // the query tile alone. A different walk is a different summation order, so y and LSE change
    // against the base; they are still bit-exactly reproducible, because the order is fixed.
    // The diagonal tile is also the tile the prologue issues first (see `pre_issued` above), so on
    // that path the walk's first tile is in flight with Q: the order is what makes its index
    // readable from the bound table at CTA entry, with no reduction and no barrier.

    if (warp == Pl::kProducerWarp) {
      // ------------------------------------------------------------ TMA producer
      if (lane == 0) {
        // Q was issued in the prologue.
        auto load = [&](int j, bool is_k) {
          const int stages = is_k ? kKS : Pl::kVS;
          const int st = j % stages;
          const int full_bar = (is_k ? Pl::kBarKFull : Pl::kBarVFull) + st;
          const int empty_bar = (is_k ? Pl::kBarKEmpty : Pl::kBarVEmpty) + st;
          // The stage's previous reader (S(j - kKS) for K, PV(j - 2) for V) has completed.
          if (j >= stages) MbarWait(bar(empty_bar), static_cast<uint32_t>(((j - stages) / stages) & 1));
          const CUtensorMap* kmap;
          const CUtensorMap* vmap;
          int row0;
          ResolveKeyMap(lp, static_cast<int64_t>(t1 - j) * Pl::kKeys, &kmap, &vmap, &row0);
          const int slabs = is_k ? Pl::kPipeSlabs : Pl::kPipeSlabsV;
          const int tile_bytes = is_k ? Pl::kKTileBytes : Pl::kVTileBytes;
          unsigned char* const dst = (is_k ? sk : sv) + st * (is_k ? Pl::kKTileBytes : Pl::kVStageBytes);
          MbarArriveExpectTx(bar(full_bar), static_cast<uint32_t>(tile_bytes));
          // One TMA box covers 64 keys of one 64-dim slab (the prepared maps are unchanged): a
          // kKeys-wide slab needs kKeys / 64 boxes, placed so the keys stay contiguous - box kx
          // lands at +kx * 64 * 128 B, which is exactly row 64 * kx of the slab, because the box
          // is 64 rows of 128 B with no padding and 8192 B is a whole number of 1024 B swizzle
          // periods. The UMMA descriptors below then read the slab as one kKeys-row tile.
          for (int s = 0; s < slabs; ++s) {
            for (int kx = 0; kx < Pl::kKeys / 64; ++kx) {
              TmaLoad4D(dst + s * Pl::kKvSlabBytes + kx * 8192, is_k ? kmap : vmap, bar(full_bar),
                        s * 64, static_cast<int>(kvh), row0 + kx * 64, static_cast<int>(b));
            }
          }
        };
        // Release order of the stages: S(j - 1) completes before PV(j - 2), so K runs kKS - 1
        // tiles ahead of V and every blocking wait below is met in the order it happens. A K lead
        // of kKS was measured and not kept: flat at one CTA on the deeper
        // ring (gekko-0.9B 1.0029 against the same tree without it) and ~0.5-1% slower at
        // two CTAs.
        // Step 0's K and V, the diagonal tile, left the prologue above into stage 0 of both rings
        // whenever the bound table was there (pre_issued), so the loop starts at step 1 and skips
        // step 0's V; where it was not, the loop issues step 0 itself and this is the branch head.
        // The K lookahead is otherwise exactly what it was: K(1..kKS-2) here, K(j + kKS - 1) at
        // step j.
        int jk = pre_issued ? 1 : 0;
        for (; jk < n && jk < kKS - 1; ++jk) load(jk, true);
        for (int j = 0; j < n; ++j) {
          if (jk < n) {
            load(jk, true);
            ++jk;
          }
          if (j > 0 || !pre_issued) load(j, false);
        }
      }
      __syncwarp();
    } else if (warp == Pl::kMmaWarp) {
      // ------------------------------------------------------------ MMA issuer
      const int fmt = ElemTraits<E>::kFormat;
      const uint32_t s_idesc = MakeInstrDesc(fmt, kPipeRows, Pl::kKeys, /*a_major=*/0, /*b_major=*/0);
      // N = Dv, plus the row-sum column group (L) where Plan::kRowSumMma holds.
      const uint32_t pv_idesc = MakeInstrDesc(fmt, kPipeRows, Pl::kPvN, /*a_major=*/0, /*b_major=*/1);
      MbarWait(bar(Pl::kBarQ), 0u);
      // Descriptors built once per CTA. The start-address field is (address >> 4) in 14 bits and
      // every operand address of the CTA's SMEM is below 2^18 B, so adding a 16-byte-unit offset
      // to the 64-bit descriptor equals MakeSmemDesc of the offset address bit for bit; per tile
      // only the stage offset is added, per k-step a compile-time constant.
      const uint64_t q_desc = MakeSmemDesc(SmemAddr(sq)).value;
      const uint64_t k_desc0 = MakeSmemDesc(SmemAddr(sk)).value;
      const uint64_t v_desc0 =
          MakeSmemDesc(SmemAddr(sv), static_cast<uint32_t>(Pl::kKeys * 8), 64u).value;
      // Q resident in TMEM, once per CTA and before S(0). kQTmemCols/8 copies of the 32 bytes
      // that are one 16-deep step of every row, at the SAME K-major descriptor offsets the SS-form
      // S loop uses, landing 8 columns further along Q's tile each time. The issuer thread issues
      // the copies and then S(0), and cp -> mma from one thread is a pipelined pair (PTX
      // 9.7.16.6.2), so no fence or wait stands between them; kQTmemCols == 0 compiles this away.
      if constexpr (Pl::kQTmemCols > 0) {
        constexpr int kQCopies = Pl::kQTmemCols / 8;
        if (ElectOne()) {
#pragma unroll
          for (int i = 0; i < kQCopies; ++i) {
            const int slab = i >> 2, within = i & 3;
            FwdCopyQToTmemRaw(
                tbase + Pl::kQCol + 8u * static_cast<uint32_t>(i),
                q_desc + static_cast<uint64_t>((slab * Pl::kQSlabBytes + within * 32) >> 4));
          }
        }
      }
      // The S stream (S(0), S(1), ... in order) and the PV stream (PV(0), PV(1), ...) each carry
      // their ring positions; the waits name the same barriers and parities as j % / j / did.
      // The counters count STEPS, not tile indices: the walk is descending (tile t1 - j at step j),
      // so the counters are the same arithmetic as on an ascending walk and the walk order lives
      // only in the tile address each step resolves (kb = (t1 - j) * kKeys, and ResolveKeyMap for
      // the producer's loads). The two are therefore independent: changing the walk order cannot
      // change a stage or a parity, and changing the ring depth cannot change which tile is read.
      // Step 0 is the diagonal tile and, when the prologue pre-issued it, its K/V are already in
      // stage 0 of both rings - which is exactly where these counters start, so no counter needs a
      // seed value for the pre-issued step.
      RingPos<kKS> s_k;    // K stage of the next S
      RingPos<kSB> s_b;    // S buffer of the next S
      RingPos<Pl::kVS> p_v;  // V stage of the next PV
      RingPos<kSB> p_b;    // S/P buffer of the next PV
      // Issue form, one rule over the plan's residency: where one CTA holds the SM, each
      // MMA and commit elects its own lane (kPerMmaElect); where two CTAs share it, one election
      // covers each group. Same MMAs, operands, order and commits either way.
      constexpr bool kPerMmaElect = Pl::kCtasPerSm == 1;
      auto issue_s = [&](int j) {
        const int st = s_k.st;
        const int buf = s_b.st;
        MbarWait(bar(Pl::kBarKFull + st), s_k.ph);
        // PV(j - kSB) read P(j - kSB) from this buffer: it must have completed. Its use of the
        // buffer was the previous lap, so the parity is the opposite of this lap's.
        if (j >= kSB) MbarWait(bar(Pl::kBarPv + buf), s_b.ph ^ 1u);
        TmemFenceAfterSync();
        __syncwarp();
        const uint64_t kd = k_desc0 + static_cast<uint64_t>(st * (Pl::kKTileBytes >> 4));
        const uint32_t dst = tbase + Pl::kSCol + buf * Pl::kBufCols;
        // The first kQSteps steps take A from Q's TMEM tile (the TS form, 8 columns per step) and
        // the rest, if any, take it from Q's SMEM descriptor (the SS form); kQTmemCols == 0 gives
        // the SS form at every step.
        constexpr int kQSteps = Pl::kQTmemCols / 8;
        auto s_mma = [&](int kk) {
          const int slab = kk >> 2, within = kk & 3;
          const uint64_t bd =
              kd + static_cast<uint64_t>((slab * Pl::kKvSlabBytes + within * 32) >> 4);
          const uint32_t acc = kk == 0 ? 0u : 1u;
          if (kk < kQSteps) {
            FwdMmaTsRaw(dst, tbase + Pl::kQCol + 8u * static_cast<uint32_t>(kk), bd, s_idesc, acc);
          } else {
            FwdMmaSsRaw(dst,
                        q_desc + static_cast<uint64_t>((slab * Pl::kQSlabBytes + within * 32) >> 4),
                        bd, s_idesc, acc);
          }
        };
        constexpr int kSSteps = Pl::kPipeWidth / 16;
        if constexpr (kPerMmaElect) {
          // One election per MMA and per commit.
#pragma unroll
          for (int kk = 0; kk < kSSteps; ++kk) {
            if (ElectOne()) s_mma(kk);
          }
          if (j + kKS < n) FwdMmaCommit(bar(Pl::kBarKEmpty + st));
          FwdMmaCommit(bar(Pl::kBarS + buf));
        } else {
          // One election for the whole group (the S issues and up to 2 commits). ElectOne() is
          // warp-deterministic, so the same lane issues the same instructions in the same order
          // as per-MMA election would.
          if (ElectOne()) {
#pragma unroll
            for (int kk = 0; kk < kSSteps; ++kk) s_mma(kk);
            if (j + kKS < n) FwdMmaCommitRaw(bar(Pl::kBarKEmpty + st));
            FwdMmaCommitRaw(bar(Pl::kBarS + buf));
          }
        }
        s_k.Next();
        s_b.Next();
      };
      auto issue_pv = [&](int j) {
        const int st = p_v.st;
        const int buf = p_b.st;
        MbarWait(bar(Pl::kBarP + buf), p_b.ph);
        MbarWait(bar(Pl::kBarVFull + st), p_v.ph);
        TmemFenceAfterSync();
        __syncwarp();
        const uint64_t vd = v_desc0 + static_cast<uint64_t>(st * (Pl::kVStageBytes >> 4));
        const uint32_t scol = tbase + Pl::kSCol + buf * Pl::kBufCols;
        // The issue form follows kPerMmaElect, as in issue_s above.
        // B = V(j) MN-major, N = Dv in one instruction: the next 64-column group is one
        // kKeys-row slab (kKeys * 128 B, i.e. kKeys * 8 in 16-byte units) further, the
        // backward dK pipe's form.
        auto pv_mma = [&](int kk) {
          FwdMmaTsRaw(tbase + Pl::kOCol, PCol<kSmWarps, kCols>(scol, kk),
                      vd + static_cast<uint64_t>((kk * 16 * 128) >> 4), pv_idesc,
                      (j == 0 && kk == 0) ? 0u : 1u);
        };
        if constexpr (kPerMmaElect) {
#pragma unroll
          for (int kk = 0; kk < Pl::kKeys / 16; ++kk) {
            if (ElectOne()) pv_mma(kk);
          }
          if (j + Pl::kVS < n) FwdMmaCommit(bar(Pl::kBarVEmpty + st));
          FwdMmaCommit(bar(Pl::kBarPv + buf));
        } else {
          if (ElectOne()) {
#pragma unroll
            for (int kk = 0; kk < Pl::kKeys / 16; ++kk) pv_mma(kk);
            if (j + Pl::kVS < n) FwdMmaCommitRaw(bar(Pl::kBarVEmpty + st));
            FwdMmaCommitRaw(bar(Pl::kBarPv + buf));
          }
        }
        p_v.Next();
        p_b.Next();
      };
      // S(j + kSAhead) is issued from PV(j)'s iteration (the distance, SAheadFor) into buffer
      // (j + kSAhead) % kSB (the depth, SBuffersFor), whose last reader is PV(j + kSAhead - kSB).
      // With kSB = kSAhead + 1 that reader is PV(j - 1), issued moments before, so every tile pays
      // its commit-to-wake latency with the pipe idle; with kSB = kSAhead + 2 it is PV(j - 2),
      // retired a whole iteration earlier, and the reuse wait is met on arrival.
      constexpr int kSAhead = SAheadFor<kW, kWv>();
      static_assert(kSAhead >= 1 && kSAhead + 1 <= kSB, "S ring must hold the issue distance");
      for (int p = 0; p < kSAhead && p < n; ++p) issue_s(p);
      for (int j = 0; j < n; ++j) {
        if (j + kSAhead < n) issue_s(j + kSAhead);
        issue_pv(j);
      }
    } else {
      // ------------------------------------------------------------ softmax warps
      const int col0 = h * kCols;
      const uint32_t lane_base = static_cast<uint32_t>(32 * g) << 16;
      const float sl2 = p.scale_log2;
      const bool neg_scale = sl2 < 0.f;
      const bool seg = detail::SegmentTest(p, b);
      // q_segment is a dense [B, Nq] plane (host.cc makes it contiguous), as in ForwardKernel.
      const int64_t qseg = (seg && row_in_range) ? p.q_segment[b * p.q_length + i] : 0;
      const int64_t nt_total = p.total_length;
      float m_used = -INFINITY;
      float lsum = 0.f;
      // The segment plane, read at the mask test below in int64 - never staged, never narrowed.
      // The keys of a warp's columns are one contiguous run, so the 32 lanes of a warp read the
      // same addresses and the loads broadcast; on the timed rows seg is false and this is dead.
      // Loop-invariant: formed once before the tile loop, not per tile.
      const int64_t* const kplane = seg ? p.k_segment + b * nt_total : nullptr;

      for (int j = 0; j < n; ++j) {
        const int buf = j % kSB;
        const int kb = (t1 - j) * Pl::kKeys;  // descending walk: see t0/t1 above
        // The FULL test over THIS tile's width. umma_fwd.cuh's TileIsFull keeps the 64-key (kBN)
        // span on purpose - the backward dQ leaves (umma_bwd_dq.cuh, umma_bwd_dq_pipe.cuh) call it
        // with their own 64-key tiles - so do not route this body through it: on a 128-key tile
        // its test admits a tile whose second half lies past a live row's last visible key, and
        // the unmasked path then folds those keys into the row (wrong y/LSE, no NaN). That was
        // reachable at chunks that are not multiples of 128 with previous K/V.
        const bool full = !seg && kb >= max_lo && kb + Pl::kKeys - 1 <= min_hi;
        const uint32_t scol = tbase + Pl::kSCol + buf * Pl::kBufCols;
        MbarWait(bar(Pl::kBarS + buf), static_cast<uint32_t>((j / kSB) & 1));
        TmemFenceAfterSync();
        float s[kCols];
        if constexpr (kHalves == 1) {
          Ld64(lane_base | scol, s);
        } else if constexpr (kCols == 64) {
          Ld64(lane_base | (scol + col0), s);
        } else {
          Ld32(lane_base | (scol + col0), s);
        }

        // Tile maximum in log2-scaled units over this thread's visible columns. FULL: every row
        // of the CTA is live and sees all 64 keys; the extreme of the raw scores times the scale
        // equals the maximum of the scaled scores exactly (rounding is monotonic), for a
        // negative scale through the minimum. Otherwise the mask is a select on the scaled
        // score: masked entries become -inf and never enter any arithmetic.
        float tmax = -INFINITY;
        if (full) {
          float e = s[0];
          if (!neg_scale) {
#pragma unroll
            for (int c = 1; c < kCols; ++c) e = fmaxf(e, s[c]);
          } else {
#pragma unroll
            for (int c = 1; c < kCols; ++c) e = fminf(e, s[c]);
          }
          tmax = e * sl2;
        } else {
          const int lo_c = live ? bd.lo - kb - col0 : 1;
          const int hi_c = live ? bd.hi - kb - col0 : 0;
          if (seg) {
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
              bool ok = c >= lo_c && c <= hi_c;
              // Exact int64 compare: the ids are the caller's (host.cc copies q_segment/k_segment
              // as given; `_segment_metadata.py` checks shape and dtype, not range), so narrowing
              // them would alias ids equal mod 2^32 or hit a sentinel. A key past the end of the
              // call is never visible (the bounds above already exclude it).
              const int64_t key = static_cast<int64_t>(kb + col0 + c);
              ok = ok && key < nt_total && kplane[key] == qseg;
              const float x = s[c] * sl2;
              s[c] = ok ? x : -INFINITY;
              tmax = fmaxf(tmax, s[c]);
            }
          } else {
            // The visible columns of a row are one contiguous run [lo_c, hi_c]. Build it once as a
            // bit mask per 32-column word (the segment test is the branch above, not interleaved
            // per column), so each column costs one predicate from a constant-bit test instead of
            // two range compares and a branch. Same values as the range test, same scaling
            // (one rounded multiply, packed in pairs) and the same -inf for masked columns, so
            // the masked tile's bits are unchanged.
            uint32_t vis[kCols / 32];
#pragma unroll
            for (int w = 0; w < kCols / 32; ++w) {
              const int lo = min(max(lo_c - 32 * w, 0), 32);
              const int hi = min(max(hi_c - 32 * w, -1), 31);
              vis[w] = lo > hi ? 0u : ((0xffffffffu >> (31 - hi)) & (0xffffffffu << lo));
            }
            const float2 sl2p = make_float2(sl2, sl2);
#pragma unroll
            for (int c = 0; c < kCols; c += 2) {
              const float2 x = Fmul2(make_float2(s[c], s[c + 1]), sl2p);
              s[c] = (vis[c / 32] & (1u << (c % 32))) ? x.x : -INFINITY;
              s[c + 1] = (vis[c / 32] & (1u << ((c + 1) % 32))) ? x.y : -INFINITY;
              tmax = fmaxf(tmax, fmaxf(s[c], s[c + 1]));
            }
          }
        }
        if constexpr (kHalves == 2) {
          float* const xb = xch + (j & 1) * 2 * kPipeRows;
          xb[h * kPipeRows + row] = tmax;
          NamedSync(1 + g, 64);
          tmax = fmaxf(tmax, xb[(1 - h) * kPipeRows + row]);
        }

        // Lazy rescale (identical decision in both threads of a row in the 8-warp form). The
        // decision is formed with predicates and selects, and the only branch is warp-uniform
        // (the vote below, which the O rescale takes as well): a per-thread if / else-if here
        // costs a reconvergence barrier on every tile although a rescale is rare. Same values:
        // the first visible tile of a row only sets m_used (its O row is exactly zero); a later
        // tile whose maximum exceeds m_used by more than rescale_thr gets alpha = exp2(m_used -
        // tmax) and moves m_used; every other thread keeps alpha = 1 and m_used.
        const bool vis_tile = tmax > -INFINITY;
        const bool first = vis_tile && m_used == -INFINITY;
        const bool need = vis_tile && !first && tmax > m_used + rescale_thr;
        const bool any_need = __any_sync(0xffffffffu, need);
        float alpha = 1.f;
        if (any_need) {
          const float a = Ex2(m_used - tmax);
          alpha = need ? a : 1.f;
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
          if (need) {
            atomicAdd(&g_pipe_rescale_events, 1ull);  // probe builds only (see TryLaunchPipe)
            if (alpha == 0.f) atomicAdd(&g_pipe_clear_events, 1ull);
          }
#endif
        }
        m_used = (first || need) ? tmax : m_used;
        // A row with no visible key in this tile gets P = 0: its scores are -inf (masked path),
        // and m is then 0 so that -inf - m never meets -inf.
        const float mneg = tmax > -INFINITY ? -m_used : 0.f;
        uint32_t pk[kCols / 2];
        // Packed pairs (Ffma2 / Fadd2): two adjacent columns per instruction, and the accumulation
        // order is bit-for-bit the scalar form's - acc0.x holds what was a0 (columns 4k + 0),
        // acc0.y a1 (4k + 1), acc1.x a2 (4k + 2), acc1.y a3 (4k + 3), each summed in the same
        // column order as before, and the row sum is still (a0 + a1) + (a2 + a3), which the two
        // scalar adds on the packed lanes form exactly. ex2 has no packed form (MUFU is
        // per-element), so the exp itself is unchanged.
        float2 acc0 = make_float2(0.f, 0.f), acc1 = make_float2(0.f, 0.f);
        const float2 sl2x2 = make_float2(sl2, sl2);
        const float2 mx2 = make_float2(mneg, mneg);
        // FULL rounds the exponent's argument once (fmaf), the masked path twice (s was scaled
        // above, then + mneg): not bit-equal for one score (see the header's Rounding note).
        if (full) {
#pragma unroll
          for (int c = 0; c < kCols; c += 4) {
            const float2 y01 = Ffma2(make_float2(s[c + 0], s[c + 1]), sl2x2, mx2);
            const float2 y23 = Ffma2(make_float2(s[c + 2], s[c + 3]), sl2x2, mx2);
            const float2 e01 = make_float2(Ex2(y01.x), Ex2(y01.y));
            const float2 e23 = make_float2(Ex2(y23.x), Ex2(y23.y));
            if constexpr (!Pl::kRowSumMma) {
              acc0 = Fadd2(acc0, e01);
              acc1 = Fadd2(acc1, e23);
            }
            pk[c / 2] = PackOf<E>::Pack2(e01.x, e01.y);
            pk[c / 2 + 1] = PackOf<E>::Pack2(e23.x, e23.y);
          }
        } else {
#pragma unroll
          for (int c = 0; c < kCols; c += 4) {
            const float2 y01 = Fadd2(make_float2(s[c + 0], s[c + 1]), mx2);
            const float2 y23 = Fadd2(make_float2(s[c + 2], s[c + 3]), mx2);
            const float2 e01 = make_float2(Ex2(y01.x), Ex2(y01.y));
            const float2 e23 = make_float2(Ex2(y23.x), Ex2(y23.y));
            if constexpr (!Pl::kRowSumMma) {
              acc0 = Fadd2(acc0, e01);
              acc1 = Fadd2(acc1, e23);
            }
            pk[c / 2] = PackOf<E>::Pack2(e01.x, e01.y);
            pk[c / 2 + 1] = PackOf<E>::Pack2(e23.x, e23.y);
          }
        }
        // Under the row-sum rule L accumulates in TMEM by the PV MMA (Plan::kRowSumMma).
        if constexpr (!Pl::kRowSumMma) {
          lsum = lsum * alpha + ((acc0.x + acc0.y) + (acc1.x + acc1.y));
        }
        // P(j) in place over this thread's own (already loaded) S(j) columns.
        if constexpr (kHalves == 1) {
          TmemStore32(lane_base | scol, pk);
        } else if constexpr (kCols == 64) {
          TmemStore32(lane_base | (scol + col0), pk);
        } else {
          St16(lane_base | (scol + col0), pk);
        }
        // O rescale: tcgen05.ld/st are warp-collective, so the warp runs it when any of its rows
        // needs it, multiplying by 1 elsewhere. O was last written by PV(j - 1); need implies an
        // earlier visible tile, so j >= 1.
        if (any_need) {
          MbarWait(bar(Pl::kBarPv + (j - 1) % kSB), static_cast<uint32_t>(((j - 1) / kSB) & 1));
          TmemFenceAfterSync();
          const float a = need ? alpha : 1.f;
          const bool clear = need && alpha == 0.f;
          if constexpr (kOCols % 64 == 0) {
#pragma unroll 1
            for (int c = 0; c < kOCols; c += 64) {
              float o[64];
              Ld64(lane_base | (tbase + Pl::kOCol + ocol0 + c), o);
              uint32_t w0[32], w1[32];
              const float2 a2 = make_float2(a, a);
#pragma unroll
              for (int x = 0; x < 16; ++x) {
                const float2 p0 = Fmul2(make_float2(o[2 * x], o[2 * x + 1]), a2);
                const float2 p1 = Fmul2(make_float2(o[32 + 2 * x], o[32 + 2 * x + 1]), a2);
                w0[2 * x] = __float_as_uint(clear ? 0.f : p0.x);
                w0[2 * x + 1] = __float_as_uint(clear ? 0.f : p0.y);
                w1[2 * x] = __float_as_uint(clear ? 0.f : p1.x);
                w1[2 * x + 1] = __float_as_uint(clear ? 0.f : p1.y);
              }
              TmemStore32(lane_base | (tbase + Pl::kOCol + ocol0 + c), w0);
              TmemStore32(lane_base | (tbase + Pl::kOCol + ocol0 + c + 32), w1);
            }
          } else {
            static_assert(kOCols % 32 == 0, "O columns per softmax thread are whole 32-column groups");
#pragma unroll 1
            for (int c = 0; c < kOCols; c += 32) {
              float o[32];
              Ld32(lane_base | (tbase + Pl::kOCol + ocol0 + c), o);
              uint32_t w0[32];
              const float2 a2 = make_float2(a, a);
#pragma unroll
              for (int x = 0; x < 16; ++x) {
                const float2 p0 = Fmul2(make_float2(o[2 * x], o[2 * x + 1]), a2);
                w0[2 * x] = __float_as_uint(clear ? 0.f : p0.x);
                w0[2 * x + 1] = __float_as_uint(clear ? 0.f : p0.y);
              }
              TmemStore32(lane_base | (tbase + Pl::kOCol + ocol0 + c), w0);
            }
          }
          if constexpr (Pl::kRowSumMma) {
            // L, rescaled with O, once per row (h == 0: the decision is identical in both
            // threads of a row in the 8-warp form; the 4-warp form has h == 0 only).
            if (h == 0) {
              const uint32_t la = lane_base | (tbase + Pl::kLCol);
              const float l = Ld1(la);
              St1(la, clear ? 0.f : l * a);
            }
          }
        }
        TmemWaitSt();
        TmemFenceBeforeSync();
        Arrive(bar(Pl::kBarP + buf));
      }

      // -------------------------------------------------------------- epilogue
      // Where the row sum accumulates in TMEM beside O (kRowSumMma) it is complete only after the
      // last PV, so it is read after that wait. Otherwise it is complete when the key-tile loop
      // ends, and the 8-warp exchange and 1/lsum are formed before the wait for the last PV, which
      // is needed only before O is read from TMEM.
      if constexpr (Pl::kRowSumMma) {
        MbarWait(bar(Pl::kBarPv + (n - 1) % kSB), static_cast<uint32_t>(((n - 1) / kSB) & 1));
        TmemFenceAfterSync();
        lsum = Ld1(lane_base | (tbase + Pl::kLCol));
      } else if constexpr (kHalves == 2) {
        // Buffer n & 1 was last read in tile n - 2, before both threads passed tile n - 1's
        // barrier.
        float* const xb = xch + (n & 1) * 2 * kPipeRows;
        xb[h * kPipeRows + row] = lsum;
        NamedSync(1 + g, 64);
        lsum = lsum + xb[(1 - h) * kPipeRows + row];
      }
      const float inv = lsum > 0.f ? 1.f / lsum : 0.f;
      if constexpr (!Pl::kRowSumMma) {
        MbarWait(bar(Pl::kBarPv + (n - 1) % kSB), static_cast<uint32_t>(((n - 1) / kSB) & 1));
        TmemFenceAfterSync();
      }
      // Which form the epilogue uses, by one rule over the plan's residency (kStageO, stated with
      // kOCols above): a CTA with the SM to itself, or a thread whose O slice is at most one line
      // wide, stages O in shared memory and writes it by TMA; every other bucket keeps the direct
      // per-row store.
      if constexpr (kStageO) {
      E* const stage = reinterpret_cast<E*>(sq);
      constexpr int kSlabElems = kPipeRows * 64;  // one 64-column slab of the staged tile
      const bool vec = p.value_dim == Pl::kPipeWidthO;
#pragma unroll 1
      for (int s = 0; s < Pl::kPipeSlabsV; ++s) {
#pragma unroll 1
      for (int c = 0; c < kOCols; c += 32) {
        // This thread's 32-column group belongs to exactly one 64-column slab (kOCols is a multiple
        // of 32 and ocol0 is a multiple of kOCols), and h is warp-constant, so a whole warp takes
        // this branch together: the tcgen05.ld below stays warp-collective.
        if (((ocol0 + c) >> 6) != s) continue;
        // tcgen05.ld is warp-collective: every softmax lane issues it, in range or not, so the
        // sequencing of the collective never depends on a row's state; only the staging write is
        // guarded.
        float o[32];
        Ld32(lane_base | (tbase + Pl::kOCol + ocol0 + c), o);
        if (!row_in_range) continue;  // rows past Nq stage nothing: the TMA clips them
        {
          const int cc = ocol0 + c;
          // Where this 32-column group lands: one slab (kOCols is a multiple of 32 and a slab is 64
          // columns wide, so a group never crosses a slab), at 16-byte chunk (cc & 63) / 8 of the
          // row, whose chunks are permuted by the row's phase (row & 7) - the permutation the map
          // declares (CU_TENSOR_MAP_SWIZZLE_128B, tma_load.cuh's kSwizzleBytes). Consecutive rows
          // then land on 8 different bank groups, so a warp's 16-byte stores conflict-free.
          E* const slabrow = stage + (cc >> 6) * kSlabElems + row * 64;
          uint4* const chunks = reinterpret_cast<uint4*>(slabrow);
          const int chunk0 = (cc & 63) >> 3;
          const int swz = row & 7;
          if (!live) {
            // Empty row: y = 0.
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int i = 0; i < 4; ++i) chunks[(chunk0 + i) ^ swz] = z;
            continue;
          }
          if (vec) {
            union {
              uint4 u4[4];
              uint32_t w[16];
            } pk;
            const float2 inv2 = make_float2(inv, inv);
#pragma unroll
            for (int x = 0; x < 16; ++x) {
              const float2 m2 = Fmul2(make_float2(o[2 * x], o[2 * x + 1]), inv2);
              pk.w[x] = PackOf<E>::Pack2(m2.x, m2.y);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) chunks[(chunk0 + i) ^ swz] = pk.u4[i];
          } else {
#pragma unroll
            for (int x = 0; x < 32; ++x) {
              const int ccx = cc + x;
              if (ccx >= p.value_dim) break;
              const int col = ccx & 63;
              slabrow[(((col >> 3) ^ swz) << 3) | (col & 7)] = PackOf<E>::From(o[x] * inv);
            }
          }
        }
      }
      // Slab s is complete in shared memory: make it visible to the async proxy and have one thread
      // issue its stores while the rest go on converting slab s + 1, so the first stores do not wait
      // for every slab to be staged. Every softmax
      // thread arrives at the barrier, in range or not, so the count is fixed. kPipeRows is a whole
      // number of the map's 64-row boxes, so a slab is 2 boxes.
      FenceProxyAsyncShared();
      NamedSync(kSmBarrierId, Pl::kSmThreads);
      if (h == 0 && row == 0) {
        const uint32_t sbase = SmemAddr(stage);
#pragma unroll
        for (int r = 0; r < kPipeRows; r += kTmaRows) {
          TmaStoreAtom(&lp.maps.y, sbase + static_cast<uint32_t>((s * kPipeRows + r) * 128), s * 64,
                       static_cast<int>(hq), qt * kPipeRows + r, static_cast<int>(b));
        }
        TmaStoreCommit();
      }
      }
      } else {
        // The pre-change form, bit for bit: each thread normalizes its own 32-column group out of
        // TMEM and stores it straight to y, 16 B per lane, per row.
        E* const yrow = live ? reinterpret_cast<E*>(p.y) +
                                   ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim
                             : nullptr;
        const bool vec_d = p.value_dim == Pl::kPipeWidthO;
#pragma unroll 1
        for (int c = 0; c < kOCols; c += 32) {
          float o[32];
          Ld32(lane_base | (tbase + Pl::kOCol + ocol0 + c), o);
          if (yrow != nullptr) {
            const int cc = ocol0 + c;
            if (vec_d) {
              union {
                uint4 u4[4];
                uint32_t w[16];
              } pk;
              const float2 inv2 = make_float2(inv, inv);
#pragma unroll
              for (int x = 0; x < 16; ++x) {
                const float2 m2 = Fmul2(make_float2(o[2 * x], o[2 * x + 1]), inv2);
                pk.w[x] = PackOf<E>::Pack2(m2.x, m2.y);
              }
              uint4* const dst = reinterpret_cast<uint4*>(yrow + cc);
              dst[0] = pk.u4[0];
              dst[1] = pk.u4[1];
              dst[2] = pk.u4[2];
              dst[3] = pk.u4[3];
            } else {
              for (int x = 0; x < 32; ++x) {
                if (cc + x >= p.value_dim) break;
                yrow[cc + x] = PackOf<E>::From(o[x] * inv);
              }
            }
          }
        }
      }
      if (live) {
        if (h == 0 && p.lse != nullptr) {
          p.lse[(b * p.query_heads + hq) * p.q_length + i] =
              lsum > 0.f ? 0.6931471805599453f * m_used + __logf(lsum) : -INFINITY;
        }
      } else {
        // Empty row. The staged (one-CTA) path has already zeroed its bytes in the tile that the
        // elected thread ships, so only the LSE is written there; the direct (two-CTA) path stores
        // y itself, so there it writes both.
        write_dead(h == 0, !kStageO);
      }
      if constexpr (kStageO) {
        // The last slab's read is the only one still in flight. Release it here, after the LSE
        // store, which does not depend on y: the engine keeps reading while this thread writes lse.
        if (h == 0 && row == 0) TmaStoreWaitRead();
      }
    }
  }

  // ------------------------------------------------------------------ exit
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == Pl::kMmaWarp) TmemDealloc(tbase, Pl::kAllocCols);  // the allocating warp
}

// ---------------------------------------------------------------- host
// The compiled configuration: SmWarpsFor<kW>() softmax warps (below), 2 K stages, 2 S buffers.
// Softmax warps per CTA, by one rule over the plan's budget rather than per shape: the 4-warp form
// (thread = row: a thread loads its row's whole kKeys-column S tile, keeps max, sum and the rescale
// decision in registers, no cross-half exchange through SMEM and no named barrier per tile) is taken
// wherever TWO CTAs are resident, so the SM still holds 8 softmax warps; the one-CTA buckets (D or
// Dv = 256) keep the two-threads-per-row form with 8 warps, because one resident CTA would otherwise
// leave only 4 softmax warps on the SM. (128,128) already ran 4 warps; now (64,64), (64,128)
// and (128,64) do too - three buckets move, not one: at kKeys 64 the old second condition
// (D + Dv >= 4 * kKeys) held only where 256 >= 256, i.e. at (128,128), and read 192 >= 256 for the
// unequal pair and 128 >= 256 at (64,64). Before this rule the softmax warp count was therefore 4 at
// (128,128) and 8 at (64,64), (64,128) and (128,64); now it is 4 at all four two-CTA buckets.
// Measured inputs, one job of three alternations: 4 warps 4-11% faster on the six
// D = 128 rows, 3.7% slower at (256,256) and 1-6% slower on two of three ladder rows at (64,64) -
// that A/B is what the old second condition (D + Dv >= 4 * kKeys) encoded. What changed since is all
// on the side of fewer instructions: one change cut 21% of the executed instructions per (64,64)
// tile and took 20.3% of its time at unchanged issue-active, and another
// removed the masked path's per-column reconvergence barriers (+13.6% objective) which the
// 4-warp form paid twice (64 columns per thread against 32). At (64,64) the two-threads-per-row
// form pays 2.03 instructions per score element of per-tile bookkeeping against 0.84 for
// thread = row, plus 0.78 for the cross-half exchange; a later run measured the 4-warp form
// at (64,128)/(128,64) at 0.8912 on the unequal guard row. On the tree that carries the
// descending fill the same one condition reads ladder 0.9013 / 0.8673 / 0.8598 and mixed-64-128
// 0.7505 against this tree with the width limb restored, no row slower.
//
template <int kD, int kDv>
constexpr int SmWarpsFor() {
#if defined(XATTN_PIPE_SM8)  // measurement build only: the two-threads-per-row form everywhere
  return 8;
#else
  using P8 = Plan<8, 2, 2, kD, kDv>;  // kCtasPerSm and kKeys do not depend on the warp count
  return (P8::kCtasPerSm == 2) ? 4 : 8;
#endif
}
inline constexpr int kKStagesShipped = 2;
inline constexpr int kSBuffersShipped = 2;

// S buffers (TMEM). History of the rule (the issue distance it used to set is SAheadFor's now):
//   * a third buffer is taken only where ONE CTA is resident - with two CTAs the other CTA's MMAs
//     fill the tensor pipe through any wait this body takes, and the two-CTA column budget
//     (kDv + 3 * 64 <= 256) leaves no room for a third buffer anyway;
//   * and only where the TMEM columns allow it: kDv + 3 * kKeys <= kAllocCols, with kKeys the
//     width this Plan already chose (Plan decides the width from the two-buffer budget, so the
//     depth never narrows a tile).
// Two buffers issue S(j+1) from PV(j)'s iteration and S(j+1) waits for PV(j-1): the tensor pipe
// drains between PV(j-1) completing and S(j+1) being issued. Three buffers issue S(j+2) from the
// same iteration and it waits for the same PV(j-1), while S(j+1) is still queued, so the pipe
// never empties. Measured inputs: the MMA_FREE arm (mode 1 with the reuse waits removed) is
// 1,866 cycles/tile against MMA_ONLY's 2,076, so the S-ring reuse wait is ~210 cycles/tile on the
// (256,256) body, i.e. the whole ceiling of this schedule; and the kSB = 3 arm
// (-0.43%) changed the buffer count WITHOUT changing the issue distance, which is why it measured
// nothing - this rule changes both.
//
// The ring is now two deeper than the issue distance (SAheadFor) wherever TMEM admits it, else one
// deeper (the shipped ring): kSB = kSAhead + 2 when kDv + (kSAhead + 2) * kKeys <= kAllocCols.
// With one buffer of slack, S(j + kSAhead) always reused PV(j - 1)'s buffer, the PV the issuer had
// just issued, at every depth; measured inputs: K-full wait 205 -> 79 cycles and the row unmoved
// (0.9998), so the reuse wait gates S; a commit-to-wake of ~204 cycles per tile and a
// ~415-cycle group cost paid only when the pipe drains; the softmax waits ~690 cycles a tile
// on S-full, so it can take an earlier S. Two buffers of slack name PV(j - 2) instead.
// The extra buffer is taken only where ONE CTA is resident, for the reason SAheadFor gives: at two
// CTAs the co-resident CTA's MMAs already fill the pipe across PV(j - 1) -> S(j + kSAhead), and the
// deeper ring measured slower there (ladder-300M/1B 1.012-1.015 at
// (64,64) with 3 buffers). Result per bucket: (256, Dv) and (128,256) at 64 keys: 4 buffers,
// distance 2 ((256,256) uses all 512 columns); every other bucket keeps the shipped ring.
template <int kD, int kDv>
constexpr int SBuffersFor() {
#if defined(XATTN_PIPE_SB2)  // measurement build only: the two-buffer, one-tile-ahead schedule
  return 2;
#else
  using P2 = Plan<8, 2, 2, kD, kDv>;  // kCtasPerSm, kKeys, kQTmemCols and kAllocCols do not depend on kSB
  constexpr int ahead = SAheadFor<kD, kDv>();
  return (P2::kCtasPerSm == 1 && P2::kSCol + P2::kQTmemCols + (ahead + 2) * P2::kKeys <= P2::kAllocCols)
             ? ahead + 2
             : ahead + 1;
#endif
}

// K stages (SMEM), by the same budget rule as the S ring above: a third stage only where ONE CTA is
// resident - a two-CTA bucket has no SMEM for it, and its schedule stays one tile ahead anyway - and
// only where the whole plan still fits the sm_103a ceiling. The producer needs no change for it: its
// loop already loads K at jk = j + kKS - 1, so kKS = 3 is exactly "K one iteration earlier than the
// S schedule is deep", which is what S two tiles ahead consumes.
// Measured need: the deeper S schedule with the shipped two K stages measured 0.9B
// 1.0358 - the issuer stopped waiting on PV(j-1) and started waiting on K(j+2), whose TMA is
// requested in the same iteration. a role trace puts the issuer's K_wait ->
// S_reuse_wait at 271 cycles, the interval that absorbs the third stage's load.
// What pays for it: at (256,256) the third stage is 32,768 B and the rings leave 240 B of the
// ceiling, so the budget closes only because the softmax no longer stages a segment table - it
// reads the int64 k_segment plane at the mask test instead, which frees 1,024 B of SMEM and costs
// the timed rows nothing (a BOS-masked call passes no segment planes at all).
template <int kD, int kDv>
constexpr int KStagesFor() {
#if defined(XATTN_PIPE_KS2)  // measurement build only: the shipped two-stage K ring
  return 2;
#else
  using P2 = Plan<8, 2, 2, kD, kDv>;
  if constexpr (P2::kCtasPerSm == 1) {
    // Only formed for a one-CTA bucket: instantiating a 3-stage Plan at a two-CTA bucket would
    // evaluate ITS two-CTA SMEM assert and fail the build over a plan nothing uses.
    using P3 = Plan<8, 3, SBuffersFor<kD, kDv>(), kD, kDv>;
    return P3::kSmemBytes <= 232448 ? 3 : 2;
  } else {
    return 2;
  }
#endif
}
// The lazy-rescale threshold in log2 units: a row's O and l are rescaled only when a tile's
// maximum exceeds the maximum they were last scaled to by more than this, so P <= 2^8.
inline constexpr float kRescaleThreshold = 8.f;

// Called by LaunchUmmaForward<E, 256, 256> and <E, 128, 128> after their own refusals (split-map alignment, device
// capability), for the body host.cc resolved for this call (ForwardBodySlot, set once per call;
// the environment is not read here). Returns
//   0  not taken: the call resolved the older leaf (XATTN_SCA_UMMA_FWD=1), and the caller launches
//      ForwardKernel unchanged;
//   1  launched (PipeLaunchCount moved);
//   2  refused BEFORE the launch call, with nothing enqueued: a call PipeAccepts does not admit
//      (host.cc resolves none such to kPipe, so this is a guard), a shared-memory opt-in the device
//      does not grant, or a failed attribute call. The caller records the refusal and the generic
//      body carries the call (dispatch_tag.cuh's two-row fallback).
// A launch call that reports an error does not return: it raises (ThrowPostAttemptError) with the
// CUDA error's name, code and text, and no other body runs over the call's outputs. One
// precondition is checked on the device and nowhere on the host: the 1024-byte alignment of the
// dynamic SMEM base that the UMMA descriptors (base_offset 0) need. The host cannot see the address
// a launch will get, so the kernel traps on it - a device fault that poisons the context, not a
// refusal and not a fallback.
template <class E, int kD, int kDv>
int TryLaunchPipe(const UmmaLaunchParams& lp, cudaStream_t stream) {
  PostAttemptErrorSlot() = cudaSuccess;
  if (ForwardBodySlot() != ForwardBody::kPipe) return 0;
  const Params& hp = lp.params;
  // The call's dims must be this launcher's buckets (ForwardCase dispatches on QkBucket and
  // ValueBucket, so this holds for every caller); padded columns are the TMA's zero fill.
  if (!PipeAccepts(hp) || QkBucket(hp.qk_dim) != kD || ValueBucket(hp.value_dim) != kDv) return 2;
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
  // Probe builds only: a pre-launch refusal, so that the caller's refusal row and generic fallback
  // can be observed (a healthy sm_103 device grants every check below).
  if (ProbeHook("pipe-refuse")) return 2;
#endif
  // The bucket pair is a compile-time constant of the plan: every constant (slab counts, tile
  // bytes, SMEM, TMEM, threads, resident CTAs, softmax warps) follows from (kD, kDv).
  constexpr int kSm = SmWarpsFor<kD, kDv>();
  constexpr int kSBuffers = SBuffersFor<kD, kDv>();
  constexpr int kKStages = KStagesFor<kD, kDv>();
  using Pl = Plan<kSm, kKStages, kSBuffers, kD, kDv>;
  const int smem_bytes = Pl::kSmemBytes;
  const int threads = Pl::kThreads;
  const auto kernel =
      &PipeForwardKernel<E, kSm, kKStages, kSBuffers, kD, kDv>;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    return 2;
  }
  // The opt-in capacity is a constant of the device and the dynamic-SMEM attribute, once set, stays
  // set on this instantiation for the device's context, so both are settled once per (device,
  // instantiation): the bit is published only after the attribute call succeeded, and a device
  // without its bit (or one numbered 64 or above) takes the checks below on every call.
  static std::atomic<uint64_t> attribute_ready{0};
  const uint64_t bit = device >= 0 && device < 64 ? (uint64_t(1) << device) : 0;
  if (bit == 0 || (attribute_ready.load(std::memory_order_acquire) & bit) == 0) {
    int cap = 0;
    if (cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) {
      (void)cudaGetLastError();
      return 2;
    }
    if (smem_bytes > cap) return 2;
    if (cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) != cudaSuccess) {
      (void)cudaGetLastError();
      return 2;
    }
    attribute_ready.fetch_or(bit, std::memory_order_release);
  }
  dim3 grid(static_cast<unsigned>((hp.q_length + kPipeRows - 1) / kPipeRows),
            static_cast<unsigned>(hp.query_heads), static_cast<unsigned>(hp.batch));
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
  // Fault control, probe builds only: an empty grid, which the runtime rejects AT the launch call
  // (cudaErrorInvalidValue under CUDA 13.4) without enqueueing work or faulting the device, so the
  // post-attempt path below runs on a healthy context.
  if (ProbeHook("pipe-posterr")) grid.x = 0u;
#endif
  // Drop any error latched before this point, so the status read after the launch call is that
  // launch's own (LaunchUmmaForward drains at its entry too; the attribute calls consume theirs).
  (void)cudaGetLastError();
  kernel<<<grid, threads, smem_bytes, stream>>>(lp, kRescaleThreshold);
  const cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    PostAttemptErrorSlot() = launch_err;
    char body[96];
    std::snprintf(body, sizeof(body), "the pipelined (%d,%d) body (fwdpipe::PipeForwardKernel)", kD,
                  kDv);
    ThrowPostAttemptError(body);
  }
  PipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
  {
    // Probe builds only: the cumulative lazy-rescale and clear counts of this element type after
    // the launch.
    unsigned long long events = 0, clears = 0;
    if (cudaStreamSynchronize(stream) == cudaSuccess &&
        cudaMemcpyFromSymbol(&events, g_pipe_rescale_events, sizeof(events)) == cudaSuccess &&
        cudaMemcpyFromSymbol(&clears, g_pipe_clear_events, sizeof(clears)) == cudaSuccess) {
      std::fprintf(stderr, "XATTN_FWD_PIPE_RESCALE_EVENTS %llu CLEAR_EVENTS %llu\n", events, clears);
    } else {
      (void)cudaGetLastError();
      std::fprintf(stderr, "XATTN_FWD_PIPE_RESCALE_EVENTS unreadable\n");
    }
  }
#endif
  return 1;
}

}  // namespace fwdpipe
}  // namespace umma
}  // namespace xattn::ops::flash_sca_native

#define XATTN_FWD_PIPE_INSTANTIATE(E, D, DV)                                            \
  template int xattn::ops::flash_sca_native::umma::fwdpipe::TryLaunchPipe<E, D, DV>(    \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
