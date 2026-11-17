// Native Blackwell (sm_103a) SCA backward: the key-owned dK/dV leaf.
//
// One CTA owns one (batch, KV head, 64-key tile) and accumulates the gradient of
// every visible pair into that tile's dK and dV, in TMEM, over the query tiles and
// query heads that can see it. Nothing is atomically reduced: every dK/dV (and
// dPrevK/dPrevV) row has exactly one writer, so `deterministic=True` is exact by
// construction rather than by a second pass.
//
// This leaf does NOT compute dQ (a separate row-owned leaf does) and does not
// implement precision levels 2-4 (a separate advanced-precision pass): levels 0 and 1
// share the elementwise algebra here, because the level change is in the FORWARD's
// saved state and softmax form, not in P^T = exp2(S^T*scale_log2 - lse2) or
// dS^T = P^T*(dP^T - delta).
//
// dq_accum is the one Params field this body cannot honour, and it is REFUSED rather
// than ignored. params.h:121-125 makes it a dK/dV-body obligation - "the dK/dV kernels
// also reduce the unscaled dQ into this buffer ... and no dQ kernel runs" - and the
// host builds it exactly when determinism is not requested (host.cc:341-346: `if
// (!deterministic) dq_accum = at::zeros(...)`), then copies `dq_accum * scale` into dq
// (host.cc:356). A call that accepted it would leave dq exactly zero under a `native`
// tag, which is the silent-zero class this file exists to keep out. So `Accepted()`
// refuses `p.dq_accum != nullptr`, and the seam that mounts this leaf must pass
// dq_accum == nullptr and let a row-owned dQ kernel write dq in a fixed order (the
// family-B composition: this leaf sets the dkdv support bit and never
// dkdv_atomic_dq - it writes no dq in any mode), or not mount it for that call.
//
// What `Accepted()` refuses is therefore a status the caller routes to the generic
// path, and it is exactly the set this body can see: levels 0-1 (the level is not a
// Params field, so levels 2-4 are the host's decision and never reach here), shape,
// previous-plane length, the dq_accum rule, the buckets and their real dimensions, the
// element format, the device capability and the SMEM capacity. Nothing after the
// preflight may turn an error into a refusal.
//
// It writes the WHOLE dK/dV/dPrevK/dPrevV plane whenever it launches: every row of
// the key tile is stored, previous-plane rows included, and a key tile that no query
// can see stores exact zeros instead of being skipped. The host's zero-fill of the
// previous planes is therefore a belt, not a precondition, for this leaf. A later
// grid change that launched only key tiles with a non-empty query envelope would
// break that property and must not, unless it stores those rows some other way; the
// kernel's "tile with no live query" arm exists for exactly this reason.
//
// Segment planes are read DENSELY (`q_segment[b * q_length + i]`, `k_segment[b *
// total_length + j]`). The host makes them dense in `PrepareCommon`
// (csrc/flash_sca/native/host.cc, the `.contiguous()` on both segment planes, lines
// 159-160 at main 44100a5eb1). A future host path that handed over a strided view -
// `k_segment_idx[:, -N:]` without that copy - would make this leaf read a
// neighbouring batch row's metadata, and nothing in the leaf would fail loudly.
//
// ---------------------------------------------------------------------------
// The contraction set (one query tile, one query head, one round)
//
//   S^T  = K_tile @ Q_tile^T      tcgen05.mma SS   M=64 keys, N=128 queries, K=D
//   dP^T = V_tile @ dO_tile^T     tcgen05.mma SS   M=64 keys, N=128 queries, K=Dv
//   dV  += P^T @ dO               tcgen05.mma TS   A = P^T packed in TMEM, K=128
//   dK  += dS^T @ Q               tcgen05.mma TS   A = dS^T packed in TMEM, K=128
//
// The two SS results land as FP32 in TMEM, and P^T / dS^T are written back as packed
// 16-bit pairs (`umma_bwd_pt.cuh` does the elementwise step and the pack: mask, exp2,
// dS, round-to-nearest-even pairs, element 2t low). At M128 they OVERWRITE the S^T and
// dP^T windows in place. At M64 both scores share S^T's columns (dP^T at lane field 16,
// see the map), P^T is packed in place over them, and dS^T goes to the window that dP^T
// no longer uses (384..447). In place is safe in ascending 32-column blocks: the block
// that reads columns 32b..32b+31 writes its 16 packed words to 16b..16b+15, and the
// next unread input starts at 32(b+1).
// A computed SMEM operand would have needed the 128-byte swizzle written by hand;
// from TMEM it needs nothing.
//
// ---------------------------------------------------------------------------
// TMEM map (128 lanes x 512 columns x 4 B, one CTA per SM)
//
//   columns   0 .. D-1    dK accumulator, FP32, M64 rows at LANE OFFSET 0
//   columns   0 .. Dv-1   dV accumulator, FP32, M64 rows at LANE OFFSET 16
//   columns 256 .. 383    S^T   FP32 at LANE OFFSET 0 and dP^T FP32 at LANE OFFSET 16:
//                         the dP^T SS MMA writes lane field 16 of S^T's columns, so one
//                         32-lane load returns both scores of the same key row;
//                         packed P^T in place at 256..319, lane offset 16
//   columns 384 .. 447    packed dS^T, lane offset 0
//   columns 448 .. 511    unreferenced at M64 (allocated: the allocation stays 512)
//
// Two distinctions the map has to keep, because the same columns play two roles:
//   - each packed window holds ONE valid lane half and one don't-care half, and
//     correctness rests on the don't-care halves never being read. In the kPCol window
//     (256..319) only lane offset 16 is P^T (written over the dP^T it came from, block by
//     block, after that block was loaded); it is what the dV MMA's A operand reads at
//     lane offset 16. Lane offset 0 of that window is overwritten with words nothing
//     reads. In the kdSCol window (384..447) only lane offset 0 is dS^T, which the dK
//     MMA's A operand reads; lane offset 16 there is a don't-care. Nothing may read
//     lane offset 0 of kPCol or lane offset 16 of kdSCol at M64;
//   - every tcgen05 load/store/wait in this file is a FULL-WARP instruction at a
//     warp-uniform address. Only ordinary scalar work is per lane: `pt::ColumnOf`,
//     `pt::MakePtBlock*` (at M64 on both lane halves, each computing 16 of every
//     block's 32 columns of its key row, see the M64 branch of the pack), and the
//     global stores in the epilogue.
//
// The lane offsets 0 and 16 are the measured M64 datapath layout, not a reading of
// the naming: an M64 accumulator holds its 64 rows in sixteen lanes of each 32-lane
// group and the destination lane field selects which sixteen (0 -> lanes 0-15, 32-47,
// 64-79, 96-111; 16 -> lanes 16-31, 48-63, 80-95, 112-127), so the two offsets are
// complementary halves of the same columns and that is what makes dK and dV share
// columns at D=Dv=256. Key row r therefore lives at lane 32*(r/16) + (r%16) of the
// window its lane offset selects.
//
// That layout is also a REQUIREMENT on the TS MMAs, not a preference: a TS MMA whose
// A operand's lane field differs from its D operand's lane field raises a
// misaligned-address fault on this hardware, while matched pairs compute correctly at
// either offset. So P^T is packed into the lane-half the dV MMA reads (offset 16) and
// dS^T into the half the dK MMA reads (offset 0) - the pack has no freedom there.
//
// Every tcgen05 load/store/wait in this file is issued by ALL 256 threads at one
// warp-uniform address (`lane_base | column`), because those instructions are
// warp-collective and their address must not vary by lane: the hardware assigns
// thread (w, L) the lane 32w + L itself. The gain is that one such read at lane field
// 0 hands half the warp dK's rows and the other half dV's same rows, and one field-0
// store fills both halves - so the epilogue and the packed store need no branch at all.
//
// ---------------------------------------------------------------------------
// Order, and why it is the deterministic one
//
// Query tiles ascend, and within a tile the query head ascends; each MMA only adds
// into an accumulator, so the accumulation order is fixed by the launch geometry
// alone. `deterministic` therefore changes nothing in this leaf's arithmetic: dQ is not
// its output, and the flag's only effect on the Params ABI is that the host builds
// dq_accum when it is false (host.cc:341-346) - which `Accepted()` refuses, so a call
// that reaches this body always has a row-owned dQ write behind it, at either setting.
// det=True and det=False must agree with this leaf's dK/dV bit for bit; that is an
// acceptance row, not an assumption.
//
// The scale is folded ONCE, at the dK store, and dV gets none (kernels.cuh:1085/1364).
// The host's own `dq.copy_(dq_accum.mul_(p.scale))` (host.cc:356) is the dQ path and
// never touches these tensors, so there is no second fold to avoid.
#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/params.h"
#include "flash_sca/native/tma_load.cuh"
#include "flash_sca/native/umma_bwd_entry.h"  // InSituKeyRowsFor: the in-situ key-tile rule
#include "flash_sca/native/umma_bwd_pt.cuh"
#include "flash_sca/native/umma_common_local.h"
#include "flash_sca/native/umma_fwd.cuh"  // ResolveKeyMap, TmemLoadCols, the TMEM fence pair
#include "flash_sca/native/umma_params.h"
#include "flash_sca/native/umma_tile_mask.cuh"
#include "flash_sca/native/visibility.cuh"

namespace xattn::ops::flash_sca_native {
namespace umma {

// 64 keys per CTA (one TMEM M64 accumulator pair), 128 queries per round (the
// native forward's kBM, so S^T/dP^T are 128 columns each and the two of them plus
// the input windows are exactly 512 TMEM columns).
inline constexpr int kBwdKeyRows = 64;
inline constexpr int kBwdQueryRows = 128;
// Eight warps: the 128-query axis splits into two halves of four warps, each packing
// two of the four 32-column blocks. The key axis does NOT split - the CTA owns 64 key
// rows and 16 lanes of one warp carry one, so every key-row index derived from `warp`
// must be reduced modulo 4 (`warp & 3`); the second half of the warps differs from the
// first only in which query columns it owns. `kseg` has kBwdKeyRows entries as well.
inline constexpr int kBwdThreads = 256;

// Output element type: the conversion the store uses, and the MMA element format
// the instruction descriptors carry (bf16 = 1, f16 = 0, as MakeInstrDesc expects).
template <class E>
struct BwdTok;
template <>
struct BwdTok<__nv_bfloat16> {
  static constexpr int kFormat = 1;
  static constexpr Element kElement = Element::kBFloat16;
  __device__ static __nv_bfloat16 To(float x) { return __float2bfloat16_rn(x); }
};
template <>
struct BwdTok<__half> {
  static constexpr int kFormat = 0;
  static constexpr Element kElement = Element::kHalf;
  __device__ static __half To(float x) { return __float2half_rn(x); }
};

// Row `row` of a key-gradient plane (dk, dv, dprev_k or dprev_v; `row` counts rows of
// `width` elements): in the input dtype, or in FP32 (kFp32: the call asked for FP32 key
// gradients, Params::key_grads_fp32). Null when the plane is.
template <class E, bool kFp32>
__device__ __forceinline__ void* KeyGradRow(void* base, int64_t row, int64_t width) {
  if (base == nullptr) return nullptr;
  return static_cast<char*>(base) + row * width * int64_t(kFp32 ? sizeof(float) : sizeof(E));
}

// Columns [c, c + 32) of a key-gradient row from the FP32 accumulator values v (columns at
// or past `width` are not stored): v[x] * mul rounded to E, or with kFp32 that FP32 product
// itself, so rounding the FP32 row to E gives exactly the 16-bit row. `row + c` is 16-byte
// aligned: a row starts at a multiple of `width` (a multiple of 8) elements from a tensor
// base and `c` is a multiple of 32. The pipelined leaves take kFp32 as a template parameter
// (their 16-bit instantiations keep their own store code); the fused leaf branches on the
// Params field.
template <class E, bool kFp32>
__device__ __forceinline__ void StoreKeyGrad32(void* row, int64_t c, int64_t width,
                                               const float (&v)[32], float mul) {
  if constexpr (kFp32) {
    float* out = static_cast<float*>(row) + c;
    if (c + 32 <= width) {
#pragma unroll
      for (int x = 0; x < 32; x += 4) {
        reinterpret_cast<float4*>(out)[x / 4] =
            make_float4(v[x] * mul, v[x + 1] * mul, v[x + 2] * mul, v[x + 3] * mul);
      }
    } else {
      for (int x = 0; x < 32; ++x) {
        if (c + x >= width) break;
        out[x] = v[x] * mul;
      }
    }
  } else {
    E* out = static_cast<E*>(row) + c;
    if (c + 32 <= width) {
      // 32 contiguous elements = 64 bytes, stored as four 16-byte stores.
      union {
        uint4 u4[4];
        E e[32];
      } pk;
#pragma unroll
      for (int x = 0; x < 32; ++x) pk.e[x] = BwdTok<E>::To(v[x] * mul);
      reinterpret_cast<uint4*>(out)[0] = pk.u4[0];
      reinterpret_cast<uint4*>(out)[1] = pk.u4[1];
      reinterpret_cast<uint4*>(out)[2] = pk.u4[2];
      reinterpret_cast<uint4*>(out)[3] = pk.u4[3];
    } else {
      for (int x = 0; x < 32; ++x) {
        if (c + x >= width) break;
        out[x] = BwdTok<E>::To(v[x] * mul);
      }
    }
  }
}

// `kKeyRows` is the key-tile height M: 64 (the measured M64 datapath layout, the
// default, bit-identical to the leaf as it was) or 128. An M128 accumulator holds
// its rows in ALL 128 TMEM lanes, so the two products can no longer share columns
// through complementary lane fields and dV moves to its own column range; that is
// the only reason the map changes. At (128,128) the two maps are:
//   M64 : dK cols 0..127 @field 0, dV cols 0..127 @field 16, S^T 256..383 @field 0 and
//         dP^T 256..383 @field 16, packed P^T 256..319 (valid @field 16), packed dS^T
//         384..447 (valid @field 0), 448..511 unreferenced
//   M128: dK cols 0..127 @field 0, dV cols 128..255 @field 0, S^T 256..383, dP^T 384..511
// both allocate exactly kAllocCols = 512, so both keep the TELEMETRY-neutral 1-CTA
// allocation.
// `kStages` is the number of Q/dO SMEM stages: 1 - the tree's single stage and the default -
// or 2, the ABL == 4 arm: the ablation ladder's ring, which the in-situ M128 key tile at
// (128,128) also ships (InSituKeyArm below). A second stage adds one (kQBytes + kGBytes)
// slice and moves where K/V start; at kStages == 1 every offset below, and kSmemBytes
// with them, is textually the tree's own expression, which is what makes ABL == 0 inert.
// `kWrite` is the launch's output set (see the kernel); the default 3 is the fused leaf, so
// every plan named without it keeps its meaning. It changes the plan in exactly one case,
// the dV-only launch at M128 (kWrite == 2, kKeyRows == 128): that launch holds no dK
// accumulator and skips dP^T, so dV takes columns 0..Dv-1, S^T and the packed P^T keep
// 256..383, and no V tile is staged. That map fits Dv = 256, which is why it is the one
// M128 plan admitted at (256,256). Every other combination is unchanged.
template <int kD, int kDv, int kKeyRows = kBwdKeyRows, int kStages = 1, int kWrite = 3>
struct BackwardDkdvPlan {
  static_assert(kD % 64 == 0 && kDv % 64 == 0, "one 64-column slab per MMA");
  static_assert(kD <= 256 && kDv <= 256, "public API cap, not a TMEM cap");
  static constexpr bool kDvOnlyM128 = (kWrite == 2 && kKeyRows == 128);
  // The M128 map is only budgeted for the bucket it is instantiated for. At (256,256)
  // dK and dV alone would need 512 columns and S^T/dP^T would have nowhere to live;
  // a dV-only launch there needs dV 0..255 + S^T 256..383 only.
  static_assert(kKeyRows == 64 || (kKeyRows == 128 && kD == 128 && kDv == 128) ||
                    (kDvOnlyM128 && kD == 256 && kDv == 256),
                "an M128 key tile is only mapped for the (128,128) bucket and for the "
                "dV-only launch at (256,256)");

  static constexpr int kSlabsK = kD / 64;   // D  in 64-column slabs (Q and K tiles)
  static constexpr int kSlabsV = kDv / 64;  // Dv in 64-column slabs (V and dO tiles)

  // One TMA 128-byte-swizzle atom: 64 rows x 64 columns of 16-bit elements.
  static constexpr int kKeySlabBytes = kKeyRows * 128;    // 8192 at M64, 16384 at M128
  static constexpr int kQuerySlabBytes = kBwdQueryRows * 128;  // 16384

  static constexpr int kKBytes = kSlabsK * kKeySlabBytes;
  // The dV-only M128 launch reads no V (V only feeds dP^T), so it stages none.
  static constexpr int kVBytes = kDvOnlyM128 ? 0 : kSlabsV * kKeySlabBytes;
  static constexpr int kQBytes = kSlabsK * kQuerySlabBytes;
  static constexpr int kGBytes = kSlabsV * kQuerySlabBytes;  // dO

  // TMEM columns. The SS windows are 128 wide because a query tile is 128 columns.
  static constexpr int kSCol = 256;
  static constexpr int kdPCol = kSCol + kBwdQueryRows;   // 384
  static constexpr int kPCol = kSCol;                    // packed P^T, in place
  static constexpr int kdSCol = kdPCol;                  // packed dS^T, in place
  static constexpr int kAllocCols = 512;

  // Where the dV accumulator lives. At M64 the accumulator holds 64 key rows in
  // sixteen lanes of each 32-lane group, so dK and dV share columns 0..max(D,Dv)-1
  // and are separated by the destination LANE FIELD (0 and 16) instead. At M128 the
  // accumulator holds 128 key rows in all 128 lanes: there is no second field, so dV
  // takes its own columns directly above dK. Both products stay at field 0 there, so
  // the "A lane field must equal D lane field" rule is trivially met.
  // The dV-only M128 launch has no dK accumulator, so dV starts at column 0.
  static constexpr int kdVColBase = kDvOnlyM128 ? 0 : ((kKeyRows == 128) ? kD : 0);
  static constexpr unsigned kdVLaneField = (kKeyRows == 128) ? 0u : 16u;
  static_assert(kKeyRows != 128 || kdVColBase + kDv <= kSCol,
                "dV must not run into the S^T window");

  // SMEM: one stage. Q and dO are 128x256 of 16-bit at D=Dv=256 (64 KB each), K and
  // V 32 KB each, so 192 KB + barriers and tables.
  static constexpr int kQOff = 0;
  static constexpr int kGOff = kQOff + kQBytes;
  // One Q + dO stage. Stage s of Q is at kQOff + s*kStageBytes and stage s of dO at
  // kGOff + s*kStageBytes, so a stage is one contiguous 2*slice image and the shift
  // between stages is one constant (the kernel adds it to sq and to sg).
  static constexpr int kStageBytes = kQBytes + kGBytes;
  // kStages == 1 gives kQOff + kStageBytes == kGOff + kGBytes, the tree's expression.
  static constexpr int kKOff = kQOff + kStages * kStageBytes;
  static constexpr int kVOff = kKOff + kKBytes;
  static constexpr int kBarOff = kVOff + kVBytes;  // 8 barriers, 8 B each
  static constexpr int kSlotOff = kBarOff + 64;    // tcgen05 TMEM allocation slot
  static constexpr int kRedOff = kSlotOff + 16;    // two int32 for the tile envelope
  static constexpr int kSegOff = kRedOff + 32;     // 64 int64 key segments
  // +1024 is slack so the kernel can rebase its operand window to a 1024-byte
  // boundary whatever base the driver gives dynamic SMEM (see the prologue).
  static constexpr int kSmemBytes = ((kSegOff + kKeyRows * 8 + 1023) / 1024) * 1024 + 1024;
  // kStages == 1 is the default plan. ABL 4's kStages == 2 is
  //   BackwardDkdvPlan<128, 128,  64, 2> = 165,888 B,
  //   BackwardDkdvPlan<128, 128, 128, 2> = 199,680 B (the shipping in-situ M128 tile),
  //   BackwardDkdvPlan<256, 256,  64, 2> = 329,728 B.
  // A byte count alone does not name a plan (other template arguments land on the same
  // totals), so quote the arguments with it. The first two fit the part's 232,448 B
  // opt-in ceiling and the third does not; CanLaunch's capacity test turns that into a
  // refusal rather than a silent one-stage fall-back. A third stage would be 231,424 B at
  // (128,128) M64 - inside the ceiling by 1 KB with nothing left over - which is why the
  // ring stops at two.

  // Every offset a descriptor encodes is a tile base, and MakeSmemDesc's base_offset is
  // the swizzle phase - 0, which is only right for a tile base a whole number of 1024 B
  // units from the start of the dynamic SMEM. The prologue establishes that for the
  // start; these keep it true for every stage and every slab, and they are compile-time
  // facts, so they cost nothing and cannot drift. (The run-time half of the same
  // invariant is the prologue's rebase into the slack - not a trap.)
  static_assert(kKeySlabBytes % 1024 == 0, "descriptor tile base must stay 1024-aligned");
  static_assert(kQuerySlabBytes % 1024 == 0, "descriptor tile base must stay 1024-aligned");
  static_assert(kQOff % 1024 == 0 && kGOff % 1024 == 0, "stage base must stay 1024-aligned");
  static_assert(kStageBytes % 1024 == 0, "a stage base must stay 1024-aligned");
  static_assert(kKOff % 1024 == 0 && kVOff % 1024 == 0, "stage base must stay 1024-aligned");
  static_assert(kSegOff % 8 == 0, "key-segment table must stay 8-aligned");

  // Scratch of the SELECTED plan, per instantiation - nothing is chosen at run time:
  //   D = Dv = 128:  100352 B of opt-in dynamic SMEM (98 KiB), 512 TMEM columns allocated
  //                  (M64: dK cols 0..127 at lane offset 0, dV cols 0..127 at lane offset
  //                  16, S^T 256..383 at lane offset 0 and dP^T 256..383 at lane offset
  //                  16; packed P^T 256..319 valid at offset 16, packed dS^T 384..447
  //                  valid at offset 0; 448..511 unreferenced).
  //   D = Dv = 256:  198656 B (194 KiB), the same 512-column map with both
  //                  accumulators spanning columns 0..255.
  //   M128, both products (D = Dv = 128 only): dK 0..127, dV 128..255, S^T 256..383, dP^T
  //                  384..511, all lane offset 0, packed in place.
  //   M128, dV only (kWrite == 2, D = Dv = 256, the plan of the pipelined dV-only leaf of
  //                  the (256,256) pair): dV 0..255, S^T 256..383 with the packed P^T in
  //                  place, lane offset 0, no V tile.
  // Both numbers include the 1024 B of alignment slack, and both fit the 227 KiB
  // opt-in ceiling of the target part; a bucket whose stages would not fit the
  // device's ceiling is refused in CanLaunch rather than at the launch.
};

// Dimension buckets, the same rule dispatch.cu uses to pick an instantiation.
// Every precondition this leaf needs, answered before anything is launched so a
// refusal leaves the outputs untouched and the caller can fall back.
//
// The precision LEVEL is not a Params field: the host decides which body runs, so
// levels 2-4 never reach this leaf. `Accepted` is the set this body implements -
// levels 0 and 1, which share it - and everything it can see. A refusal here is a
// status the caller routes to the generic path.
// kWrite keys the output rules the launch owns: (kWrite & 1) -> dk / dprev_k,
// (kWrite & 2) -> dv / dprev_v (3 = the fused leaf's original rule, so the
// default argument is a no-op for every existing caller). An output this
// instantiation owns must be present and, when a previous plane exists, its
// previous half must be present too; the OTHER output's pointer and pairing are
// not required - an independent launch contracts to leave that product's rows
// bitwise untouched, and "half a previous plane written" is precisely the bug
// the fused leaf's pairing rule exists to catch.
inline bool Accepted(const UmmaLaunchParams& lp, int kWrite = 3, int kKeyRows = kBwdKeyRows) {
  const Params& p = lp.params;
  if (p.dy == nullptr || p.lse_in == nullptr || p.delta == nullptr) return false;
  // delta stays required for kWrite == 2 although the dV-only leaf never reads
  // it: the planner builds one Params for the whole backward and the uniformity
  // costs it no field; a missing delta is a broken call, not a narrower launch.
  // The owned-output nulls below are KEYED (an unconditional pair would be the
  // kWrite == 3 case of exactly these two lines), so an independent leaf may be
  // given a launch whose other-product pointer is null - and it must be, per its
  // contract to leave that product's rows bitwise untouched.
  if ((kWrite & 1) && p.dk == nullptr) return false;
  if ((kWrite & 2) && p.dv == nullptr) return false;
  // The round reads the query-tile mask table before any other test can skip its
  // tile (`FromTable(p, b, 2 * t, 2)` is the first statement of the loop body), so a
  // tile table that is not there is a dereference, not a refusal. Every caller builds
  // one; this term is what makes that a checked precondition instead of an assumption.
  if (p.tile_bounds == nullptr) return false;
  if (p.qk_dim <= 0 || p.value_dim <= 0) return false;
  if (p.kv_heads <= 0 || p.query_heads <= 0) return false;
  if (p.group <= 0 || p.query_heads != p.kv_heads * p.group) return false;
  // dq_accum is a dK/dV-body obligation this body does not implement (see the header,
  // and params.h:121-125): the host builds it whenever determinism is not requested
  // (host.cc:341-346) and copies `dq_accum * scale` into dq afterwards (host.cc:356),
  // so accepting it here would leave dq exactly zero while the phase row says native.
  // The seam passes nullptr and lets a row-owned dQ kernel write dq in fixed order.
  if (p.dq_accum != nullptr) return false;
  if (p.prev_length < 0 || p.k_length < 0 || p.q_length < 0) return false;
  if (p.total_length != p.prev_length + p.k_length) return false;
  if (p.chunk <= 0) return false;
  if ((kWrite & 1) && ((p.prev_length > 0) != (p.dprev_k != nullptr))) return false;
  if ((kWrite & 2) && ((p.prev_length > 0) != (p.dprev_v != nullptr))) return false;
  // The key axis is what this CTA enumerates over, and the host rule is that an
  // unpacked previous plane never cuts a key tile (kKeyMapAlignment, params.h:20,
  // which is 128): at M128 a `prev_length == 64` would straddle the prev/current
  // boundary inside one tile, so the alignment rule is the tile height's, not a
  // constant 64.
  if (!lp.maps.packed_kv && p.prev_length % kKeyRows != 0) return false;
  return true;
}

// Query tiles (in kBwdQueryRows units) that may see keys [n0, n1]. Envelope only:
// every candidate is still tested with MaySee before a launch-sized amount of work
// is spent on it.
//
// `kKeyRows` must match the key-tile height the caller launched with: the mask table
// is built by the host in 64-row key tiles whatever the leaf's M is (host.cc builds
// key_tile_count = ceil(total_length / 64)), so an M128 caller must read the union of
// the two 64-row entries its tile covers. Reading only entry `kt` would take a
// different tile's envelope - a silent loss of dK/dV on every segmented call, with no
// fault and no shape error to catch it.
template <int kKeyRows = kBwdKeyRows>
__device__ __forceinline__ void QueryTileEnvelope(const Params& p, int64_t b, int64_t kt,
                                                  int64_t n0, int64_t n1, int64_t* first,
                                                  int64_t* last) {
  if (p.row_lo != nullptr && p.key_tile_range != nullptr) {
    // Sub-tile envelope in QUERY-tile (128-row) units; INT_MAX / -1 means empty.
    auto sub_first = [&](int64_t k) {
      const int32_t* e = p.key_tile_range + (b * p.key_tile_count + k) * 2;
      return e[1] < 0 ? int64_t{0x7fffffff} : (e[0] > 0 ? int64_t{e[0] / 2} : int64_t{0});
    };
    auto sub_last = [&](int64_t k) {
      const int32_t* e = p.key_tile_range + (b * p.key_tile_count + k) * 2;
      return e[1] < 0 ? int64_t{-1} : int64_t{e[1] / 2};
    };
    if (kKeyRows == 64) {
      *first = sub_first(kt);
      *last = sub_last(kt);
      if (*first == 0x7fffffff) *first = 0;
      return;
    } else {
      const int64_t k1 = (2 * kt + 1 < p.key_tile_count) ? 2 * kt + 1 : 2 * kt;
      const int64_t f = sub_first(2 * kt) < sub_first(k1) ? sub_first(2 * kt) : sub_first(k1);
      const int64_t l = sub_last(2 * kt) > sub_last(k1) ? sub_last(2 * kt) : sub_last(k1);
      *first = f == 0x7fffffff ? 0 : f;
      *last = (f == 0x7fffffff || l < 0) ? -1 : l;
      return;
    }
  }
  if (p.row_lo != nullptr) {  // no table: every tile is a candidate
    *first = 0;
    *last = p.tile_count > 0 ? (p.tile_count + 1) / 2 - 1 : -1;
    return;
  }
  // Plain chunk window (kernels.cuh:QueryTilesOf, same arithmetic).
  const int64_t offset = p.total_length - p.q_length;
  const int64_t first_row = n0 - offset + (p.strict ? 1 : 0);
  const int64_t last_row = (n1 / p.chunk + 2) * p.chunk - 1 - offset;
  const int64_t a = (first_row > 0 ? first_row : 0) / kBwdQueryRows;
  int64_t z = last_row < p.q_length - 1 ? last_row : p.q_length - 1;
  z = z < 0 ? -1 : z / kBwdQueryRows;
  *first = a;
  *last = z;
}

// kWrite is the compile-time output set of a launch: 1 = dK only (4D+2Dv per
// admitted pair; the P^T pack, the dV slab MMAs and the dV stores are skipped),
// 2 = dV only (2D+2Dv; dP^T, its pack, the dK slab MMAs and the dK stores
// skipped), 3 = both (the fused leaf, arithmetic unchanged at 4D+4Dv). The two
// restricted sets are independent leaves, family A's key side: each writes only
// its own product's rows and leaves the other product's region bitwise
// untouched, so two sequential launches reconstruct the fused result with the
// same determinism argument (one writer per row per product, ascending tiles
// then heads, no atomics, no dq accumulator). Cost of independence: the shared
// S^T contraction, the TMA loads, the envelope and the barriers are paid twice
// across the pair of launches.
// Minimum CTAs per SM the launch bounds ask for. One is the measured form: asking for
// two caps registers at 128 per thread to fit, which is bitwise identical and measured
// 1.25x (0.9B shape) and 1.58x (4B shape) SLOWER, so the design must not bake two in.
// The knob exists so that a build differing only in residency and registers is one -D
// away, which is the cheapest identity field this leaf has.
#ifndef XATTN_BWD_MIN_BLOCKS
#define XATTN_BWD_MIN_BLOCKS 1
#endif

// ---------------------------------------------------------------------------------
// The ABL template parameter is the ablation ladder's arm index (probe/leaf_bench.cu
// --only=abl<N>, launched through LaunchUmmaBackwardDkdvArm below). It is defaulted to
// 0 and every arm's edit is an `if constexpr`, so ABL == 0 is the shipping body and no
// run-time test enters the round loop:
//   ABL 0  the control, unchanged
//   ABL 1  TIMING ONLY - the in-place P^T/dS^T pack (both TMEM reads, both packed TMEM
//          stores, their WAR barrier and their hand-off fence pair) is skipped
//   ABL 2  TIMING ONLY - the epilogue's TMEM drain of the dK/dV accumulators is skipped
//   ABL 3  numerically identical - only the head-dependent fields of a ColumnMeta are
//          re-gathered on the head rounds after the first of one query tile
//   ABL 4  numerically identical - a second Q/dO SMEM stage with a one-round-lookahead
//          prefetch, i.e. the load overlaps the previous round's compute
// ABL 1 and 2 write wrong (or no) bytes on purpose and say so where they are used; 3 and
// 4 are gated bit-for-bit against ABL 0 by the bench's own dK/dV hashes.
template <typename E, int kD, int kDv, int kWrite = 3, int kKeyRows = kBwdKeyRows,
          int ABL = 0>
__global__ void __launch_bounds__(kBwdThreads, XATTN_BWD_MIN_BLOCKS) BackwardDkdvKernel(
    const __grid_constant__ UmmaLaunchParams lp) {
  using Plan = BackwardDkdvPlan<kD, kDv, kKeyRows, (ABL == 4 ? 2 : 1), kWrite>;
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem_raw[];

  // SMEM descriptor base. MakeSmemDesc puts the real, 16-byte-granular address in
  // the descriptor's start field and takes base_offset = 0, which is the 128-byte
  // swizzle PHASE of the matrix base, not an address. The phase is zero for every
  // operand here only because every operand base is a whole number of 32-byte
  // k-steps from `smem` and `smem` itself is 1024-byte aligned. That is made true
  // rather than assumed: the array is declared __align__(1024) and, if the driver
  // places dynamic SMEM elsewhere, the window is rebased into the slack that
  // kSmemBytes reserves. There is no longer a __trap() for a misaligned base: a
  // trap here would abort a call whose host arm had already been accepted and whose
  // phase row already says native, and the condition it checked can no longer hold.
  unsigned char* const smem =
      smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);

  unsigned char* const sq = smem + Plan::kQOff;
  unsigned char* const sg = smem + Plan::kGOff;
  unsigned char* const sk = smem + Plan::kKOff;
  unsigned char* const sv = smem + Plan::kVOff;
  uint32_t const bar0 = SmemAddr(smem + Plan::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Plan::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Plan::kRedOff);
  int64_t* const kseg = reinterpret_cast<int64_t*>(smem + Plan::kSegOff);

  const uint32_t bar_kv = bar0;
  const uint32_t bar_q = bar0 + 8;
  const uint32_t bar_s = bar0 + 16;
  const uint32_t bar_acc = bar0 + 24;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hk = blockIdx.y;
  const int64_t kt = blockIdx.x;
  const int64_t kb = kt * kKeyRows;                // first global key of this tile
  const int64_t ke = kb + kKeyRows;                // exclusive
  const int64_t nt_total = p.total_length;

  if (tid == 0) {
    MbarInit(bar_kv, 1);
    MbarInit(bar_q, 1);
    MbarInit(bar_s, 1);
    MbarInit(bar_acc, 1);
  }
  if (warp == 0) {
    TmemAlloc(Plan::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) MbarInitFence();

  // Every slab and window base below is a whole number of 32-byte k-steps from
  // `smem`, which the prologue has rebased to a 1024-byte boundary, so the
  // descriptors' base_offset = 0 is the true swizzle phase (see the prologue).

  // ---- the key tile, once: K and V into SMEM, and the key segments if needed
  const bool seg = detail::SegmentTest(p, b);
  if (seg && tid < kKeyRows) {
    const int64_t j = kb + tid;
    kseg[tid] = j < nt_total ? p.k_segment[b * nt_total + j] : 0;
  }
  if (tid == 0) {
    const CUtensorMap* kmap;
    const CUtensorMap* vmap;
    int row0;
    ResolveKeyMap(lp, kb, &kmap, &vmap, &row0);
    MbarArriveExpectTx(bar_kv, Plan::kKBytes + Plan::kVBytes);
    for (int s = 0; s < Plan::kSlabsK; ++s) {
      for (int r = 0; r < kKeyRows; r += kTmaRows) {
        TmaLoad4D(sk + s * Plan::kKeySlabBytes + r * 128, kmap, bar_kv, s * 64,
                  static_cast<int>(hk), row0 + r, static_cast<int>(b));
      }
    }
    if (Plan::kVBytes != 0) {  // a plan that stages no V tile (dV-only M128) loads none
      for (int s = 0; s < Plan::kSlabsV; ++s) {
        for (int r = 0; r < kKeyRows; r += kTmaRows) {
          TmaLoad4D(sv + s * Plan::kKeySlabBytes + r * 128, vmap, bar_kv, s * 64,
                    static_cast<int>(hk), row0 + r, static_cast<int>(b));
        }
      }
    }
  }

  int64_t qt_first, qt_last;
  if (tid == 0) {
    QueryTileEnvelope<kKeyRows>(p, b, kt, kb, ke - 1, &qt_first, &qt_last);
    red[0] = static_cast<int32_t>(qt_first);
    red[1] = static_cast<int32_t>(qt_last);
  }
  __syncthreads();
  qt_first = red[0];
  qt_last = red[1];

  const uint32_t tbase = *tmem_slot & 0x1FFu;  // 1-CTA allocation: lane 0, column base
  // The one TMEM base every collective below is issued at: warp w addresses lanes
  // 32w..32w+31, and tcgen05 assigns thread (w, L) the lane 32w + L, which holds key
  // row 16w + (L & 15) (umma_common_local.h:247 - "thread laneid reads accumulator row
  // (32*warp + laneid), address is warp-uniform"). The key row is therefore a property
  // of the thread's place in the warp and never of the address: folding a per-thread
  // lane into the address counts the same mapping twice, and it is not merely wrong but
  // illegal - a `.sync.aligned` collective whose address varies by lane raises
  // `misaligned address` on the device.
  //
  // This base is also why the key-owned epilogue needs no branch. An M64 accumulator
  // holds its 64 rows in sixteen lanes of each 32-lane group, and the destination
  // lane field selects which sixteen: field 0 gives {0-15, 32-47, 64-79, 96-111},
  // field 16 gives {16-31, 48-63, 80-95, 112-127}. So at one column base the low
  // sixteen lanes of every warp hold dK's rows and the high sixteen hold dV's same
  // key rows - the two windows are complementary halves of the same columns, which is
  // what lets dK and dV share columns at D=Dv=256. A single warp-uniform read at field
  // 0 therefore hands each thread one row of one of the two outputs. The +16 field
  // appears only in the dV MMA's own A and D operands, which one elected thread issues;
  // those are not collectives.
  // The TMEM lane field addresses one of 128 lanes, so a warp's row block is
  // 32 * (warp & 3): warps 4..7 must address the SAME 64 key rows as warps 0..3 (they
  // differ only in the query columns they pack). A bare `32 * warp` would put warps 4..7
  // at lane 128..223, outside the field, on the next CTA's key rows.
  const uint32_t lane_base = static_cast<uint32_t>(32 * (warp & 3)) << 16;

  // Phase 0 is a constant only because K/V is staged ONCE per CTA: bar_kv is armed
  // and arrived exactly once, so a fixed phase is correct here. A stage ring over
  // K/V would have to bring this phase along with it - the wait would otherwise ask
  // for a phase no arrival reaches and hang - which is why prefetching K/V changes
  // the barrier protocol rather than just the loop.
  MbarWait(bar_kv, 0);

  const int elem_fmt = BwdTok<E>::kFormat;

  // dV += P^T dO and dK += dS^T Q read their B operand (dO, Q) as kSlabs 64-column slabs of
  // one MN-major SWIZZLE_128B image, Plan::kQuerySlabBytes apart. That hop is what the SMEM
  // descriptor's leading byte offset expresses (the distance to the next 64-wide MN group),
  // so one instruction can cover the whole accumulator width instead of one slab. A wide
  // instruction is not free, so the wide form is taken only by the fused equal-width body
  // (both products, D = Dv) where it replaces at least three slab instructions per k-step,
  // and only up to the instruction's N limit of 256 columns. Every other instantiation (the
  // single-product leaves, rectangular cells, narrower buckets) keeps one instruction per
  // slab. The reduction over k-steps and its order are the same in both forms: same bits.
  constexpr bool kWide =
      kWrite == 3 && kD == kDv && Plan::kSlabsK >= 3 && Plan::kSlabsK * 64 <= 256;
  constexpr bool kWideV = kWide;
  constexpr bool kWideK = kWide;
  constexpr int kAccIssuesV = kWideV ? 1 : Plan::kSlabsV;  // instructions per k-step
  constexpr int kAccIssuesK = kWideK ? 1 : Plan::kSlabsK;
  constexpr int kAccIssueColsV = kWideV ? Plan::kSlabsV * 64 : 64;  // columns per instruction
  constexpr int kAccIssueColsK = kWideK ? Plan::kSlabsK * 64 : 64;
  constexpr uint32_t kSlabLboV = kWideV ? static_cast<uint32_t>(Plan::kQuerySlabBytes / 16) : 0u;
  constexpr uint32_t kSlabLboK = kWideK ? static_cast<uint32_t>(Plan::kQuerySlabBytes / 16) : 0u;
  static_assert(Plan::kQuerySlabBytes % 16 == 0 && Plan::kQuerySlabBytes / 16 < (1 << 14),
                "the slab hop must encode in the descriptor's 14-bit leading byte offset");

  // ------------------------------------------------------------------ the round
  bool any = false;
  if (qt_first <= qt_last) {
    const uint32_t s_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kBwdQueryRows, /*a=*/0, /*b=*/0);
    const uint32_t v_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kAccIssueColsV, /*a=*/0, /*b=*/1);
    const uint32_t k_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kAccIssueColsK, /*a=*/0, /*b=*/1);
    int round = 0;
    for (int64_t t = qt_first; t <= qt_last; ++t) {
      const QueryTileMask m = FromTable(p, b, 2 * t, 2);
      if (!MaySee(m, kb, ke)) continue;  // no visible pair in this tile: nothing to add
      const int64_t i0 = t * kBwdQueryRows;
      const bool full = FullTileExact(m, p, b, kb, ke);
      for (int64_t gh = 0; gh < p.group; ++gh) {
        const int64_t hq = hk * p.group + gh;
        // ABL == 4 only: which of the plan's two Q/dO stages THIS round reads. For every
        // other arm the offset is a compile-time zero, so qcur/gcur fold to sq/sg and the
        // operand addresses below are the tree's own.
        unsigned char* const qcur =
            sq + ((ABL == 4) ? static_cast<int>(round & 1) * Plan::kStageBytes : 0);
        unsigned char* const gcur =
            sg + ((ABL == 4) ? static_cast<int>(round & 1) * Plan::kStageBytes : 0);

        // Q and dO of this tile and head, one TMA box set each.
        if constexpr (ABL != 4) {
          if (tid == 0) {
            MbarArriveExpectTx(bar_q, Plan::kQBytes + Plan::kGBytes);
            for (int s = 0; s < Plan::kSlabsK; ++s) {
              for (int r = 0; r < kBwdQueryRows; r += kTmaRows) {
                TmaLoad4D(sq + s * Plan::kQuerySlabBytes + r * 128, &lp.maps.q, bar_q, s * 64,
                          static_cast<int>(hq), i0 + r, static_cast<int>(b));
              }
            }
            for (int s = 0; s < Plan::kSlabsV; ++s) {
              for (int r = 0; r < kBwdQueryRows; r += kTmaRows) {
                TmaLoad4D(sg + s * Plan::kQuerySlabBytes + r * 128, &lp.maps.dy, bar_q, s * 64,
                          static_cast<int>(hq), i0 + r, static_cast<int>(b));
              }
            }
          }
          MbarWait(bar_q, static_cast<uint32_t>(round & 1));
        } else {
          // ABL 4: two stages and one round of lookahead. This round's Q/dO were issued by
          // the PREVIOUS round (or, at round 0, just below - the one load with nothing to
          // hide behind), so the wait comes FIRST and the issue for the successor follows
          // it. The successor's image goes into the OTHER stage, whose last reader was the
          // round before this one and is therefore already done (that round waited on its
          // own bar_acc), so the ring needs no extra barrier and stays two deep: a third
          // image is 231,424 B at (128,128) - 1 KB under the 232,448 B ceiling, nothing
          // left over - and two-deep is what the ablation ladder specifies.
          auto stage_issue = [&](int64_t ni0, int64_t nhq, int nst) {
            if (tid != 0) return;
            unsigned char* const qs = sq + nst * Plan::kStageBytes;
            unsigned char* const gs = sg + nst * Plan::kStageBytes;
            MbarArriveExpectTx(bar_q, Plan::kQBytes + Plan::kGBytes);
            for (int s = 0; s < Plan::kSlabsK; ++s) {
              for (int r = 0; r < kBwdQueryRows; r += kTmaRows) {
                TmaLoad4D(qs + s * Plan::kQuerySlabBytes + r * 128, &lp.maps.q, bar_q, s * 64,
                          static_cast<int>(nhq), ni0 + r, static_cast<int>(b));
              }
            }
            for (int s = 0; s < Plan::kSlabsV; ++s) {
              for (int r = 0; r < kBwdQueryRows; r += kTmaRows) {
                TmaLoad4D(gs + s * Plan::kQuerySlabBytes + r * 128, &lp.maps.dy, bar_q, s * 64,
                          static_cast<int>(nhq), ni0 + r, static_cast<int>(b));
              }
            }
          };
          if (round == 0) {
            stage_issue(i0, hq, 0);
            MbarWait(bar_q, 0u);
          } else {
            MbarWait(bar_q, static_cast<uint32_t>(round & 1));
          }
          // Both stages share bar_q, so the successor's expect-tx below opens the NEXT phase
          // of the barrier every thread has just waited on. It must not be posted before the
          // slowest thread has passed this round's wait: if the successor's bytes landed
          // first, that thread would find the parity flipped twice, read this round's phase
          // as still pending, and wait for a phase that is only posted after the round's own
          // CTA barriers - which it never reaches. With this barrier the parity is at most
          // one phase ahead of any waiting thread, which is what a parity wait can tell apart.
          __syncthreads();
          // The successor of (t, gh) in the admitted round sequence: the next head of this
          // query tile while one is left, else the next query tile that MaySee this key
          // tile. `-1` means this is the last round, and then nothing is issued - a
          // dangling expect-tx would leave a phase no wait ever reaches.
          int64_t ni0 = -1, nhq = -1;
          if (gh + 1 < p.group) {
            ni0 = i0;
            nhq = hq + 1;
          } else {
            for (int64_t t2 = t + 1; t2 <= qt_last; ++t2) {
              if (MaySee(FromTable(p, b, 2 * t2, 2), kb, ke)) {
                ni0 = t2 * kBwdQueryRows;
                nhq = hk * p.group;
                break;
              }
            }
          }
          if (ni0 >= 0) stage_issue(ni0, nhq, static_cast<int>((round + 1) & 1));
        }

        // S^T = K Q^T and dP^T = V dO^T. One warp issues; every warp's ElectOne()
        // would issue the same MMA again into a count-1 barrier (the forward's own
        // duplicate-issue defect).
        if (warp == 0) {
          for (int kk = 0; kk < kD / 16; ++kk) {
            const int slab = kk >> 2, within = kk & 3;
            MmaSs(tbase + Plan::kSCol,
                  MakeSmemDesc(SmemAddr(sk) + slab * Plan::kKeySlabBytes + within * 32),
                  MakeSmemDesc(SmemAddr(qcur) + slab * Plan::kQuerySlabBytes + within * 32),
                  s_idesc, kk == 0 ? 0u : 1u);
          }
          if (kWrite & 1) {  // dP^T feeds the dS^T pack only; a dV-only leaf skips it
            for (int kk = 0; kk < kDv / 16; ++kk) {
              const int slab = kk >> 2, within = kk & 3;
              MmaSs(tbase + (kKeyRows == 64 ? ((16u << 16) + Plan::kSCol)  // M64: field 16
                                            : static_cast<uint32_t>(Plan::kdPCol)),
                    MakeSmemDesc(SmemAddr(sv) + slab * Plan::kKeySlabBytes + within * 32),
                    MakeSmemDesc(SmemAddr(gcur) + slab * Plan::kQuerySlabBytes + within * 32),
                    s_idesc, kk == 0 ? 0u : 1u);
            }
          }
          MmaCommit(bar_s);
        }
        MbarWait(bar_s, static_cast<uint32_t>(round & 1));

        // P^T and dS^T, 32-column blocks. The loads and the stores are WARP-COLLECTIVE
        // (.sync.aligned): all 128 threads issue them, and only the arithmetic between
        // them is per lane. Which lane half of a window must hold the operand follows
        // from the lane field of the MMA that reads it.
        // M64 (the `kKeyRows == 64` branch below): lane L and lane L+16 of a warp carry
        // the SAME key row. One 32-lane load returns S^T on lanes 0..15 (field 0) and
        // dP^T on lanes 16..31 (field 16) of that row; a shfl_xor 16 exchange hands each
        // half the other score, and each half computes 16 of the block's 32 columns. A
        // second exchange of the packed words leaves the row's full P^T on the high lanes
        // and its full dS^T on the low lanes. The dV MMA's A operand is addressed at lane
        // offset 16 of kPCol (`tbase + (16 << 16) + kPCol`, 256..319) and the dK MMA's at
        // lane offset 0 of kdSCol (`tbase + kdSCol`, 384..447). The other half of each
        // window holds words nothing may read (see the TMEM map).
        // M128 (the else-branch): lane L of the 128 is key row L and computes its own
        // row; S^T, dP^T, P^T and dS^T are all at lane offset 0, and P^T/dS^T are
        // packed in place over S^T/dP^T, where both MMAs read them.
        // The key row is the thread's place in the warp reduced to the key axis, and the
        // key segment with it. Both are read ONLY by the masked call below, which is why
        // a defect here is invisible on a full tile and decides the causal and segment
        // tests on a masked one - the analytic walk would stay bit-identical.
        if constexpr (ABL != 1) {  // ABL 1 (TIMING ONLY) skips the whole pack round trip
        const int w4 = warp & 3;
        // M64: sixteen lanes of each 32-lane group carry one key row, so the row is the
        // thread's place in the warp masked to the lane half. M128: the accumulator spans
        // all 128 TMEM lanes and lane L IS key row L - there is no second half to mask to,
        // and warps 4..7 cover the same 128 rows as warps 0..3, differing only in which
        // query columns they pack.
        const int64_t row_in_tile =
            (kKeyRows == 128) ? static_cast<int64_t>(w4 * 32 + lane)
                              : static_cast<int64_t>(w4 * 16 + (lane & 15));
        const int64_t j = kb + row_in_tile;
        const int64_t ks = seg ? kseg[row_in_tile] : 0;  // both lane halves pack at M64
        uint32_t pv[32], dsv[32];
        // One pt::ColumnMeta per query column per BODY, staged in shared memory by every
        // thread. Rebuilding pt::ColumnMeta meta[kBlock] inside each of the
        // four block iterations for lanes < 16 repeats the same gathers four times, in a
        // 768-byte local array per thread that ptxas spills (308 B stores, 492 B loads,
        // 448 B frame). The values and the arithmetic that reads them are the same
        // either way, so dK/dV are bit-identical; only the array's home differs.
        __shared__ pt::ColumnMeta smeta[kBwdQueryRows];
        // ABL == 3: three of a ColumnMeta's five fields - lo, hi and segment - are
        // properties of the query COLUMN (batch row and position), never of the head,
        // while lse2 and delta are per (head, column). The gh loop is inner and every
        // admitted (t, gh) is visited, so the rounds with gh != 0 re-visit a query tile
        // whose lo/hi/segment half is already in `smeta`; ABL 3 re-gathers only the two
        // head-dependent fields there. Same values, fewer fetches - every other arm,
        // ABL == 0 included, runs the tree's single full fill unchanged.
        for (int sc = threadIdx.x; sc < kBwdQueryRows; sc += blockDim.x) {
          const int64_t ic = i0 + sc;
          if constexpr (ABL == 3) {
            if (gh != 0) {
              // ColumnOf's own two cases, so the values are identical: a dead column keeps
              // the struct defaults it would have been given (lo=1, hi=0, lse2=+inf,
              // delta=0) and is never read out of range, a live one gets exactly
              // ColumnOf's expressions.
              if (ic >= p.q_length) {
                smeta[sc].lse2 = INFINITY;
                smeta[sc].delta = 0.f;
              } else {
                const int64_t stat = (b * p.query_heads + hq) * p.q_length + ic;
                const float lse = p.lse_in != nullptr ? p.lse_in[stat] : -INFINITY;
                smeta[sc].lse2 = isfinite(lse) ? lse * pt::kLog2e : INFINITY;
                smeta[sc].delta = p.delta != nullptr ? p.delta[stat] : 0.f;
              }
              continue;
            }
          }
          smeta[sc] = pt::ColumnOf(p, b, hq, ic, seg, ic < p.q_length);
        }
        __syncthreads();
        // The four 32-column blocks split across the two warp groups: warps 0..3 pack
        // blocks {0,1} and warps 4..7 pack blocks {2,3}. Both groups execute exactly two
        // iterations, so `blk & 1` (the pair) and `blk >> 1` keep their meaning, and blk0
        // is even so each half still issues one pair store per window. The slabs are
        // disjoint: PackedColumn(kPCol, 0) and (kPCol, 2) are +0 and +32, and the pairs
        // are separate windows. What is NOT free here is the order of the stores against
        // the block-1 source read - see the barrier below.
        static_assert(kBwdThreads == 2 * kBwdQueryRows,
                      "the pack split keys on threadIdx.x >= kBwdQueryRows: that is half of the "
                      "CTA only while kBwdThreads is twice the query-tile rows, and a geometry "
                      "constant moving alone would silently cross the groups");
        const int blk0 = (threadIdx.x >= kBwdQueryRows) ? kBwdQueryRows / pt::kBlock / 2 : 0;
        for (int blk = blk0; blk < blk0 + kBwdQueryRows / pt::kBlock / 2; ++blk) {
          const int c0 = blk * pt::kBlock;
          if constexpr (kKeyRows == 64) {
            // Full-lane M64 pack. The dP^T MMA wrote lane field 16 of S^T's columns, so
            // one 32-lane load returns S^T of key row 16*w4 + (lane & 15) on lanes 0..15 and
            // dP^T of the same row on lanes 16..31. The low lane computes block columns 0..15,
            // the high lane 16..31, each with its half's other operand from its partner.
            float x[pt::kBlock];
            TmemLoadCols(lane_base | (tbase + Plan::kSCol + c0), x, pt::kBlock);
            const bool lo_half = lane < 16;
            float s16[pt::kBlock / 2], dp16[pt::kBlock / 2];
#pragma unroll
            for (int k = 0; k < pt::kBlock / 2; ++k) {
              const float r =
                  __shfl_xor_sync(0xffffffffu, lo_half ? x[pt::kBlock / 2 + k] : x[k], 16);
              s16[k] = lo_half ? x[k] : r;
              dp16[k] = lo_half ? r : x[pt::kBlock / 2 + k];
            }
            const int cb = c0 + (lo_half ? 0 : pt::kBlock / 2);
            uint32_t p8[pt::kBlock / 4], ds8[pt::kBlock / 4];
            if (full) {
              pt::MakePtBlockFull<E, pt::kBlock / 2>(s16, dp16, &smeta[cb], p.scale_log2, p8, ds8);
            } else {
              pt::MakePtBlock<E, pt::kBlock / 2>(s16, dp16, &smeta[cb], j, ks, seg, p.scale_log2,
                                                 p8, ds8);
            }
            // The high lane takes its row's P^T words 0..7 (the low lane's columns), the low
            // lane its row's dS^T words 8..15 (the high lane's columns). The other halves of
            // pv/dsv are don't-care: only the high lanes' pv (P^T, lane field 16) and the
            // low lanes' dsv (dS^T, lane field 0) are ever read, by the dV and dK MMAs.
            uint32_t* const pw = pv + 16 * (blk & 1);
            uint32_t* const dw = dsv + 16 * (blk & 1);
#pragma unroll
            for (int k = 0; k < pt::kBlock / 4; ++k) {
              const uint32_t r = __shfl_xor_sync(0xffffffffu, lo_half ? p8[k] : ds8[k], 16);
              pw[k] = r;
              pw[pt::kBlock / 4 + k] = p8[k];
              dw[k] = ds8[k];
              dw[pt::kBlock / 4 + k] = r;
            }
          } else {
          float s[pt::kBlock], dp[pt::kBlock] = {};
          TmemLoadCols(lane_base | (tbase + Plan::kSCol + c0), s, pt::kBlock);
          if (kWrite & 1) {  // the dP^T window is written only when dS^T is packed
            TmemLoadCols(lane_base | (tbase + Plan::kdPCol + c0), dp, pt::kBlock);
          }
          // Every lane packs its own key row at M128 - all 32 lanes of all eight warps
          // are live, because the accumulator has 128 rows in 128 lanes. At M64 only the
          // low half of each warp holds a row. The per-row arithmetic is untouched either
          // way; per KEY ROW the round cost is identical.
          if (kKeyRows == 128 || lane < 16) {
            // The unmasked arm is exact only when every column of this block is LIVE
            // (row_lo <= i <= row_hi, same segment, i < q_length); on a dead column it
            // has no bound to test and would drop the slot only through the non-finite
            // `lse2` the host writes for an empty row, whereas MakePtBlock tests that
            // column against lo/hi explicitly. `FullTileExact` is what proves the
            // precondition, and any other tile takes the masked call below.
            if (full) {
              pt::MakePtBlockFull<E, pt::kBlock>(s, dp, &smeta[c0], p.scale_log2,
                                                pv + 16 * (blk & 1), dsv + 16 * (blk & 1));
            } else {
              pt::MakePtBlock<E, pt::kBlock>(s, dp, &smeta[c0], j, ks, seg, p.scale_log2,
                                             pv + 16 * (blk & 1), dsv + 16 * (blk & 1));
            }
          }
          }  // the half-lane pack in this else-branch serves M128 only
          if (blk & 1) {  // two blocks make the 32 words one tcgen05.st takes
            const int pair = blk >> 1;
            // The packed windows are written IN PLACE over the block-1 source: with
            // pair = 1 the store lands on columns 288-319, which is where block 1 was
            // read from (TmemLoadCols at kSCol + 32). A sequential four-warp body has
            // read it by the time it stores; with the blocks split across two warp
            // groups nothing orders this group's store against the other group's read of
            // it, and tcgen05.wait::ld / ::st order only the issuing warp - so a group
            // would store packed pairs over a source the other half has not read yet.
            // That is a WAR race and it was measured as one: 29% of key rows wrong,
            // max_rel 0.99, nondeterministic between runs of one binary. Both groups
            // reach this barrier on their SECOND iteration, after that iteration's loads
            // and before its stores, which is exactly the order the window needs.
            TmemFenceBeforeSync();
            __syncthreads();
            TmemFenceAfterSync();
            if (kWrite & 2) {  // P^T feed: the dV MMA's A operand is at lane offset 16
              uint32_t pvx[32];
              if constexpr (kKeyRows == 128) {
                // M128: lane L IS key row L and the dV MMA reads this window at lane
                // offset 0, so there is nothing to carry - every lane stores its own row.
                // The shuffle of the M64 path would be actively wrong here: it would
                // overwrite row 16+x with row x.
#pragma unroll
                for (int x = 0; x < 32; ++x) pvx[x] = pv[x];
              } else {
#pragma unroll
                for (int x = 0; x < 32; ++x) {
                  pvx[x] = pv[x];  // the high lanes already hold their rows' P^T
                }
              }
              TmemStore32(lane_base | (tbase + pt::PackedColumn(Plan::kPCol, 2 * pair)), pvx);
            }
            if (kWrite & 1) {  // dS^T feed: the dK MMA's A operand is at lane offset 0
              uint32_t dsx[32];
              if constexpr (kKeyRows == 128) {
                // M128: the dK MMA reads this window at lane offset 0 across all 128 lanes,
                // and every lane holds a real dS row, so nothing is zeroed.
#pragma unroll
                for (int x = 0; x < 32; ++x) dsx[x] = dsv[x];
              } else {
#pragma unroll
                for (int x = 0; x < 32; ++x) {
                  dsx[x] = lane < 16 ? dsv[x] : 0u;
                }
              }
              TmemStore32(lane_base | (tbase + pt::PackedColumn(Plan::kdSCol, 2 * pair)), dsx);
            }
          }
        }
        TmemWaitSt();
        // Hand the packed windows to the warp that issues the MMAs.
        TmemFenceBeforeSync();
        __syncthreads();
        TmemFenceAfterSync();
        }  // ABL != 1: ABL 1 issues no TMEM read, no pack and no TMEM store here, and so
           // also needs neither the WAR barrier inside the block loop nor this hand-off

        // dV += P^T dO and dK += dS^T Q: kAccIssuesV/K instructions per k-step, each covering
        // kAccIssueColsV/K accumulator columns (one slab, or the whole width where kWideV/K
        // holds). The first k-step of every instruction resets the columns it covers, not
        // only those of the first: the windows are per instruction and there is no other
        // initializer.
        if (warp == 0) {
          if (kWrite & 2) {  // dV accumulator: A = P^T packed at lane offset 16
            for (int sl = 0; sl < kAccIssuesV; ++sl) {
              for (int kk = 0; kk < kBwdQueryRows / 16; ++kk) {
                MmaTs(tbase + Plan::kdVColBase + (Plan::kdVLaneField << 16) + sl * 64,
                      tbase + (Plan::kdVLaneField << 16) + Plan::kPCol + kk * 8,
                      MakeSmemDesc(SmemAddr(gcur) + sl * Plan::kQuerySlabBytes + kk * 16 * 128,
                                   kSlabLboV, 64),
                      v_idesc, (round == 0 && kk == 0) ? 0u : 1u);
              }
            }
          }
          if (kWrite & 1) {  // dK accumulator: A = dS^T packed at lane offset 0
            for (int sl = 0; sl < kAccIssuesK; ++sl) {
              for (int kk = 0; kk < kBwdQueryRows / 16; ++kk) {
                MmaTs(tbase + sl * 64, tbase + Plan::kdSCol + kk * 8,
                      MakeSmemDesc(SmemAddr(qcur) + sl * Plan::kQuerySlabBytes + kk * 16 * 128,
                                   kSlabLboK, 64),
                      k_idesc, (round == 0 && kk == 0) ? 0u : 1u);
              }
            }
          }
          MmaCommit(bar_acc);
        }
        MbarWait(bar_acc, static_cast<uint32_t>(round & 1));
        any = true;
        ++round;
      }
    }
  }

  // -------------------------------------------------------------------- epilogue
  // One warp-uniform collective for all 256 threads: at column base c the low sixteen
  // lanes of each warp hold dK of key row 16w + (lane & 15) and the high sixteen hold
  // dV of the SAME key row, so a thread's half decides which output it owns and its
  // place in the warp decides the row - no branch around the collective and no lane in
  // the address. Rows are 32 columns at a time; a row past the key axis is not stored
  // at all, and a tile with no visible query stores zeros.
  // One sweep per owned product. At M64 a thread's lane HALF decides which product it
  // owns - the low half holds dK of key row 16w+(lane&15) and the high half dV of the
  // SAME row, both at the same columns - so exactly one sweep runs per thread and one
  // collective TMEM read feeds it. At M128 the accumulator spans all 128 lanes, so lane
  // L holds key row L for BOTH products and every lane that owns a row owns a dK row and
  // a dV row. Both sweeps then run, against different column ranges of the same lane.
  // The store volume per CTA is unchanged: half the warps own rows and each stores twice
  // as many.
  const int r = (kKeyRows == 128) ? (32 * (warp & 3) + lane)
                                  : ((warp & 3) * 16 + (lane & 15));
  const int64_t j = kb + r;
  // ABL 2 (TIMING ONLY) skips the CTA's single drain of the dK/dV accumulators out of
  // TMEM. This is the ONLY accumulator store in the leaf - the accumulators are never
  // copied per round - so the arm's delta is everything this CTA pays to get its products
  // out, and an upper bound on any per-round accumulator-store stall (that is the whole
  // point of the arm: candidate 2 assumes a per-round TMEM->SMEM store that this source
  // does not contain).
  if constexpr (ABL != 2) {
  for (int sweep = 0; sweep < ((kKeyRows == 128) ? 2 : 1); ++sweep) {
  // A single-product launch at M128 drains only the accumulator it owns: the other sweep
  // would read columns that nothing stores (and, dV-only, the same columns again). The
  // test is uniform over the CTA and compiled out of the fused leaf and of every M64 body.
  if constexpr (kKeyRows == 128 && kWrite != 3) {
    if ((sweep == 0) ? ((kWrite & 1) == 0) : ((kWrite & 2) == 0)) continue;
  }
  const bool is_k = (kKeyRows == 128) ? (sweep == 0) : (lane < 16);
  // The loop runs over the wider of the two windows, so a rectangular bucket
  // (kD != kDv) still drains both; each thread's store is gated by its own width, so a
  // thread whose window is the narrower one simply stores fewer columns.
  const int cols = kD > kDv ? kD : kDv;
  const int64_t width = is_k ? p.qk_dim : p.value_dim;
  const float mul = is_k ? p.scale : 1.f;

  void* row_out = nullptr;
  if (j < p.prev_length) {
    const int64_t row = (b * p.prev_length + j) * p.kv_heads + hk;
    void* base = is_k ? p.dprev_k : p.dprev_v;
    row_out = p.key_grads_fp32 ? KeyGradRow<E, true>(base, row, width)
                               : KeyGradRow<E, false>(base, row, width);
  } else if (j >= p.prev_length && j < nt_total) {
    const int64_t lk = j - p.prev_length;
    const int64_t row = (b * p.k_length + lk) * p.kv_heads + hk;
    void* base = is_k ? p.dk : p.dv;
    row_out = p.key_grads_fp32 ? KeyGradRow<E, true>(base, row, width)
                               : KeyGradRow<E, false>(base, row, width);
  }
  // The output set this launch does not own is never STORED: the collective TMEM
  // read below is guarded only by `any`, which is warp-uniform (a lane-variant guard
  // around a full-warp .sync.aligned read would be illegal), so ownership cannot skip
  // the read; instead a non-owned row's destination clears here, and a garbage
  // accumulator half can never reach a byte. `any` and the envelope are
  // metadata-derived, never TMEM-derived, for the same reason.
  if ((is_k && (kWrite & 1) == 0) || (!is_k && (kWrite & 2) == 0)) row_out = nullptr;
  // Warps 4..7 packed the second half of the query columns and own no key row, so their
  // destination is cleared here for the same reason a non-owned output's is: the
  // collective read is not ownership-guarded, the STORE is what ownership decides.
  // Only the first warp group stores: at M64 warps 4..7 pack the second half of the
  // query columns and own no key row, and at M128 they cover the SAME 128 rows as warps
  // 0..3 (32*(warp&3)+lane), differing only in the query columns they packed. In both
  // geometries the key axis of a 256-thread CTA is one warp group wide.
  if (warp >= 4) row_out = nullptr;

  for (int c = 0; c < cols; c += 32) {
    float v[32];
    if (any) {
      // At M128 the two products sit in different column ranges of the SAME lane; at M64
      // they share columns and the lane half selects which one the read feeds.
      TmemLoad32(lane_base | (tbase + (is_k ? 0 : Plan::kdVColBase) + c), v);
      TmemWaitLd();
    } else {
#pragma unroll
      for (int x = 0; x < 32; ++x) v[x] = 0.f;
    }
    if (row_out != nullptr) {
      if (p.key_grads_fp32) {
        StoreKeyGrad32<E, true>(row_out, c, width, v, mul);
      } else {
        StoreKeyGrad32<E, false>(row_out, c, width, v, mul);
      }
    }
  }
  }  // sweep
  }  // ABL != 2: this arm keeps the drain (ABL 2 skips it and writes no dK/dV byte)

  // ------------------------------------------------------------------ tile exit
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Plan::kAllocCols);
}

// Raised, never returned: a driver or launch error is not a refusal of the accepted
// set. A host header that throws works in both worlds it is compiled into - the
// packaged extension (pybind turns std::runtime_error into a Python RuntimeError) and
// a standalone nvcc compile of an instantiation TU.
namespace dkdv_detail {
[[noreturn]] inline void RaiseCuda(const char* what, cudaError_t err) {
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
}  // namespace dkdv_detail

// Launch entry. Returns false when the case is outside this leaf's accepted set, with
// nothing written: the caller falls back to the generic path rather than reading a
// partial tile. `false` means exactly that and nothing else, and everything after it
// raises (see dkdv_detail::RaiseCuda).
//
// Complete preflight for this leaf, and the ONLY place `false` is decided: the accepted
// set (Accepted, which includes the dq_accum rule and the kWrite-keyed owned-output
// rules), the two dimension buckets and their real dimensions, the element format, the
// device capability, and the dynamic-SMEM capacity Plan::kSmemBytes needs. A `false`
// writes no byte, configures nothing and launches nothing, so the seam may run the
// generic body. It is split out from the launch so that a caller which only needs the
// answer - a dispatch census, the seam's phase tag, a capability probe - can ask it
// without launching anything.
//
// The property queries below are CUDA calls: the first one may initialize a context, so
// this is not a "no CUDA call" path, and a driver failure there raises rather than
// turning into a refusal (a failed query says nothing about the accepted set). What is
// guaranteed is the part a caller depends on: no output is written, no function
// attribute is configured and no launch happens before the decision. kWrite is the
// output-set key of the independent leaves (family A; its default value is the fused
// leaf and a no-op for every existing caller).
template <typename E, int kD, int kDv, int kWrite = 3, int kKeyRows = kBwdKeyRows,
          int ABL = 0>
bool UmmaBackwardDkdvCanLaunch(const UmmaLaunchParams& lp) {
  using Plan = BackwardDkdvPlan<kD, kDv, kKeyRows, (ABL == 4 ? 2 : 1), kWrite>;
  const Params& p = lp.params;
  if (!Accepted(lp, kWrite, kKeyRows)) return false;
  if (QkBucket(p.qk_dim) != kD || ValueBucket(p.value_dim) != kDv) return false;
  // The bucket rule rounds every dimension above 128 UP to 256 (dispatch.cu:71-72), so
  // the bucket comparison alone would accept qk_dim = value_dim = 2048 at (256,256) and
  // this body would compute only the first 256 columns of each - the epilogue's per-
  // column guard tests against the real value_dim, so `col >= width` would never fire
  // and 1792 dV columns would stay unwritten. Compare the real dimensions, not only
  // their buckets: the public envelope caps them at 256 today, and the next stage raises
  // exactly that cap.
  if (p.qk_dim > kD || p.value_dim > kDv) return false;
  if (p.element != BwdTok<E>::kElement) return false;
  // Capability fail-closed. A build whose arch is outside
  // sm_100..sm_119 sets XATTN_UMMA_DEVICE_ASM to 0 and this kernel's body assembles to
  // nothing; the launcher would then report `native` having written no byte. That build
  // can only be selected by a device that is not sm_103a, so the check is the device,
  // not the macro - XATTN_UMMA_DEVICE_ASM is keyed off __CUDA_ARCH__ and is therefore
  // always 0 in this host code (umma_common_local.h:42-47), which is why a `#if` on it
  // here would refuse every call including the good one. Same rule as the host's own
  // gate at host.cc:366. A refusal is a decision, not an error: nothing has been
  // written, so the seam may run the generic body.
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd dK/dV: cudaGetDevice", q);
  int major = 0, minor = 0;
  q = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd dK/dV: capability major", q);
  q = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd dK/dV: capability minor", q);
  if (major != 10 || minor != 3) return false;
  // Resource preflight: the launch needs Plan::kSmemBytes of opt-in dynamic SMEM, and a
  // device whose per-block ceiling is below it cannot run the kernel at all. Asking
  // here turns "the driver would have refused the launch" into a decision made before
  // anything is configured or written.
  int smem_cap = 0;
  q = cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (q != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dK/dV: shared memory capacity", q);
  }
  if (static_cast<int>(Plan::kSmemBytes) > smem_cap) return false;
  return true;
}

// Committed launch. Everything above decides; from here on a driver or launch error
// raises.
template <typename E, int kD, int kDv, int kWrite = 3, int kKeyRows = kBwdKeyRows,
          int ABL = 0>
bool LaunchUmmaBackwardDkdv(const UmmaLaunchParams& lp, cudaStream_t stream) {
  using Plan = BackwardDkdvPlan<kD, kDv, kKeyRows, (ABL == 4 ? 2 : 1), kWrite>;
  const Params& p = lp.params;
  if (!UmmaBackwardDkdvCanLaunch<E, kD, kDv, kWrite, kKeyRows, ABL>(lp)) return false;
  // A driver error is not a refusal of the accepted set: by now the seam has accepted
  // the arm and owes a `native` phase row, so a `false` here would run the generic
  // body over a sticky error (the old code cached the first failure in a `static`, so
  // every later call would have fallen back silently for the life of the process) and
  // an unchecked launch would report a native body over uninitialised dk/dv/dprev.
  // The repo's own precedent for the wording is
  // csrc/attention/native_cuda/reader_host.cc - "a bad launch must not look like a
  // zero gradient".
  const cudaError_t attr =
      cudaFuncSetAttribute(BackwardDkdvKernel<E, kD, kDv, kWrite, kKeyRows, ABL>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, Plan::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dK/dV: cudaFuncSetAttribute", attr);
  }
  const unsigned gx = static_cast<unsigned>((p.total_length + kKeyRows - 1) / kKeyRows);
  // No work, not a native body: total_length == 0 implies shape-0 dk/dv/dprev, so this
  // returns after the attribute is set but with no launch and nothing written. A phase
  // tag keyed on this return value would read `native` over tensors no one wrote; this
  // arm is `no work`.
  if (gx == 0) return true;
  const dim3 grid(gx, static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
  BackwardDkdvKernel<E, kD, kDv, kWrite, kKeyRows, ABL><<<grid, kBwdThreads, Plan::kSmemBytes, stream>>>(lp);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dK/dV: kernel launch", launched);
  }
  return true;
}

// The fused twins (`CanLaunchUmmaBackwardDkdvAuto` / `LaunchUmmaBackwardDkdvAuto`)
// live BELOW the table: the earlier two-pair ladder that stood here is
// deleted at the fold, and the twins are now the table's kWrite=3 rows, so the
// fused entry and the split entries share ONE selection by construction.

// (The folded definition of LaunchUmmaBackwardDkdvAuto is below the table, with its
// query twin; the pre-fold pair ladder that stood here is deleted.)

// ---------------------------------------------------------------------------------
// The IN-SITU key-tile height: the one place the M of the public backward's FUSED key body is
// decided. (At (256,256), level 0, the in-situ launch runs a pair of single-product M128 leaves
// instead of that body - see LaunchUmmaBackwardInSituDkdv - and this trait does not change for
// it.)
//
// `kKeyRows` is the MMA's M dimension (the key tile). Moving it from 64 to 128 is a pure
// REGROUPING of the same MMA, not a different arithmetic: one M128 tile assembles the
// same 128 key rows into one accumulator that two M64 tiles above it would accumulate
// in two CTAs. The contraction direction (K = D / Dv, 64-column slabs in ascending
// order, one tcgen05 MMA per slab), the operand bytes, the scale fold at the dK store
// and the store rule are all unchanged; each key row still has exactly one writer and
// the rounds still ascend by query tile then query head, so the surviving association
// order is the same one the M64 tiles produce; and a key row the mask kills contributes
// exact zeros in both groupings, so the M64 and M128 groupings are expected to agree
// bit for bit (the contract and the traced-shape checks compare them against FP64; the
// two tile heights were also compared directly by exact hash of dK and dV at the 4B
// shape). Only the GRID changes: half as many key tiles, each twice as tall.
//
// The bucket that may run it is fixed by TMEM ARITHMETIC, not by taste. An M128
// accumulator holds its rows in all 128 lanes, so dV can no longer share dK's columns
// through the complementary lane field (Plan::kdVColBase) and needs its own 128
// columns. At (128,128) that is dK 0..127 + dV 128..255 + S^T 256..383 + dP^T 384..511
// = 512 = kAllocCols: exactly the one-CTA allocation, so it fits with nothing to spare.
// At 0.9B's (256,256) the same map would be 256 + 256 + 256 + 256 = 1024 columns against
// those 512, so that bucket CANNOT take a fused M128 key tile and MUST REFUSE it rather than
// run a 64-row body under an M128 label: a silent fall-back would print "B4 == B0" and
// be read as "the lever is empty" when the truth is that the lever does not exist there.
// (The M128 tiles that bucket does run belong to the single-product pair, dV then dK, whose
// maps hold one accumulator each: dV 0..255 + S^T 256..383, and dK 0..255 + two 64-column
// S^T and dP^T buffers.)
// BackwardDkdvPlan's static_assert makes an M128 instantiation at any other bucket a
// compile error; the asserts below pin the SELECTION rule to that same set, so a later
// edit cannot widen one without the other.
//
// One accepted-set consequence, not a correctness one: `Accepted`'s previous-plane
// alignment test is the tile height's (`p.prev_length % kKeyRows`), so at M128 an
// unpacked prev plane whose length is a multiple of 64 but not 128 is refused by this
// arm and served by the generic body instead - correct, not native. The traced shapes do
// not hit it (training passes no prev plane; prefill passes prev = C = 4096, a multiple
// of both tile heights).
//
// SMEM, checked from the plan's own constants rather than assumed: `kKeySlabBytes` doubles
// with the key tile (8,192 -> 16,384 B), so K and V grow by one slab each and the key
// segment table by kKeyRows*8; `kStageBytes = kQBytes + kGBytes` is QUERY-side and does
// NOT move, so the two-stage arm's per-stage admission is untouched by the key tile. At
// (128,128): one stage 100,352 -> 134,144 B (the shipping kStages == 1 form, 98,304 B
// under the part's 232,448 B opt-in ceiling), two stages 165,888 -> 199,680 B, still
// inside it. So the budget admits M128 at this bucket with room, and CanLaunch's own
// capacity test remains the run-time backstop for a smaller part.
//
// This is the IN-SITU arm only. LaunchUmmaBackwardDkdvAuto and the ladder entries above
// deliberately keep kKeyRows == kBwdKeyRows: their M64 form is the control arm of the
// bench's train/train128 pair, and an Auto that silently became M128 would make that A/B
// measure nothing. The public backward reaches the leaf through the registry row in
// instantiations/sca_umma_bwd_registry.cu, which names the pair below, and through
// nothing else.
template <int kD, int kDv>
struct InSituKeyRows {
  static constexpr int value = InSituKeyRowsFor(kD, kDv);  // (128,128) -> 128, else 64
};
static_assert(InSituKeyRowsFor(128, 128) == 128, "the M128 key tile is the (128,128) arm");
static_assert(InSituKeyRowsFor(256, 256) == kInSituKeyRowsDefault,
              "0.9B's (256,256) bucket must REFUSE the M128 key tile, never fall back to M64 "
              "silently: dK 0..255 + dV 256..511 + S^T + dP^T is 1024 TMEM columns against "
              "the 512 a CTA may allocate");
static_assert(kInSituKeyRowsDefault == kBwdKeyRows,
              "umma_bwd_entry.h's default key-tile height must be this header's M64");

// The in-situ M128 tile ships the ABL 4 body: a second Q/dO SMEM stage and a one-round
// lookahead, so the next round's Q/dO TMA is in flight while this round computes instead of
// being issued after it (the one-stage body issues, waits, then computes). The arm changes
// neither an operand nor an accumulation order, so dQ/dK/dV are expected bit-identical to
// ABL 0; the two builds were compared by exact digest of every gradient, not by tolerance.
// With this arm the (128,128) bucket's SHIPPING key body is the staged one: an ABL 0 row
// at that bucket is a control, no longer the shipped configuration.
// Its plan is BackwardDkdvPlan<128, 128, 128, 2> = 199,680 B against the 232,448 B opt-in
// ceiling, and CanLaunch's capacity test stays the run-time backstop. Every other bucket,
// and the M64 fallback below, keeps the one-stage body: (256,256) cannot hold two stages
// (BackwardDkdvPlan<256, 256, 64, 2> = 329,728 B). The fallback is taken whenever the
// preferred arm refuses: an unpacked previous plane whose length is not a multiple of the
// 128-row tile, or a device that grants less dynamic shared memory than the 199,680 B plan.
// Both outcomes stay native, and InSituLaunchedKeyRows() says which tile ran.
template <int kD, int kDv>
struct InSituKeyArm {
  static constexpr int value = (InSituKeyRows<kD, kDv>::value == 128) ? 4 : 0;
};
static_assert(InSituKeyArm<128, 128>::value == 4 && InSituKeyArm<256, 256>::value == 0,
              "the two-stage ring ships at the M128 tile only");
static_assert(BackwardDkdvPlan<128, 128, 128, 2>::kSmemBytes <= 232448,
              "the shipped M128 ring must fit the sm_103a opt-in shared-memory ceiling");
// Checked per instantiation as well, so that a future bucket inheriting a 128-row tile (and
// with it the arm) fails the BUILD when its two-stage plan does not fit, instead of losing
// the arm to CanLaunch's run-time refusal with no test noticing.
template <int kD, int kDv>
constexpr bool InSituKeyArmFits() {
  if constexpr (InSituKeyArm<kD, kDv>::value == 0) {
    return true;
  } else {
    return InSituKeyArm<kD, kDv>::value == 4 &&
           BackwardDkdvPlan<kD, kDv, InSituKeyRows<kD, kDv>::value, 2>::kSmemBytes <= 232448;
  }
}

// The in-situ key leaf: the fused kWrite == 3 body at the bucket's own key-tile height,
// with the query and the launch taken from ONE trait so the prepared arm and the launched
// body cannot diverge into "prepared 128 / launched 64". Every other template argument is
// the fused leaf's default. At (256,256) the launch below runs the single-product M128 pair
// instead when it accepts the call; the query stays the fused leaf's, because the fused M64
// leaf accepts every call the pair accepts. Not `inline`, exactly like the Auto pair above:
// the shipping instantiation is explicit (XATTN_UMMA_BWD_DKDV_M128_INSTANTIATE at the bottom
// of this file) so the registry TU can keep holding no device code.
template <typename E, int kD, int kDv>
bool CanLaunchUmmaBackwardInSituDkdv(const UmmaLaunchParams& lp) {
  constexpr int kM = InSituKeyRows<kD, kDv>::value;
  constexpr int kA = InSituKeyArm<kD, kDv>::value;
  static_assert(InSituKeyArmFits<kD, kDv>(),
                "a bucket that ships the two-stage ring must fit the opt-in SMEM ceiling");
  if (UmmaBackwardDkdvCanLaunch<E, kD, kDv, 3, kM, kA>(lp)) return true;
  // The bucket's preferred arm refused (at (128,128): an unpacked previous plane whose
  // length is not a multiple of 128, which Accepted tests against the TILE HEIGHT, or a
  // device granting less dynamic SMEM than the two-stage plan). The public call stays
  // native by falling back to the M64 body, which this bucket has in the same TU - the
  // alternative is a mixed call that runs a native dQ product beside a generic key
  // product, a coverage regression against the M64 leaf, which serves these calls.
  if constexpr (kM != kBwdKeyRows) {
    return UmmaBackwardDkdvCanLaunch<E, kD, kDv, 3, kBwdKeyRows>(lp);
  }
  return false;
}
// XATTN_SCA_BWD_KEY_ARM, read on every call, selects the key body of the in-situ launch at
// (256,256) and at (128,128), and nothing else:
//   - absent or empty: the shipping default. At level 0, BF16 and FP16, (256,256) runs the pair of
//     pipelined M128 single-product leaves declared below whenever both leaves' preflights accept
//     the call, and (128,128) runs the pipelined fused M128 leaf declared below whenever its
//     preflight accepts the call; every other call takes the fused key leaf;
//   - "pair128p" names the (256,256) pair and behaves as the default there with ONE exception: at
//     level 0 a refusal of the pair's preflight is REPORTED (the launch returns false, which the
//     caller raises on) instead of being served by the fused leaf, because a timing arm must not
//     measure a body it did not name. At levels 1-4 it falls through like the default. At
//     (128,128) it selects the fused key leaf;
//   - any other value, "m64" (the control arm) included, selects the fused key leaf at both
//     buckets.
// No value selects a leaf that writes only one of the two key products: such an arm hands
// autograd an unwritten dK or dV under a native tag. The query twin
// (CanLaunchUmmaBackwardInSituDkdv) does not read the variable: every call the pair accepts is a
// call the fused M64 leaf accepts, so every value answers the query alike.
inline bool KeyArmUnsetRequested() {
  const char* value = std::getenv("XATTN_SCA_BWD_KEY_ARM");
  return value == nullptr || value[0] == '\0';
}
inline bool KeyArmPair128pRequested() {
  const char* value = std::getenv("XATTN_SCA_BWD_KEY_ARM");
  return value != nullptr && std::strcmp(value, "pair128p") == 0;
}
// The two leaves of the (256,256) pair: the pipelined dV-only M128 leaf (umma_bwd_dv_pipe.cuh;
// its preflight is UmmaBackwardDkdvCanLaunch<E, 256, 256, 2, 128, 0>) and the pipelined dK-only
// M128 leaf (umma_bwd_dk_pipe.cuh). Defined in those headers and instantiated for BF16 and FP16
// in instantiations/sca_umma_bwd_dkdv_d256_v256.cu, the TU whose in-situ launch reaches them.
template <typename E>
bool LaunchUmmaBackwardDvPipe(const UmmaLaunchParams& lp, cudaStream_t stream);
template <typename E>
bool CanLaunchUmmaBackwardDkPipe(const UmmaLaunchParams& lp);
template <typename E>
bool LaunchUmmaBackwardDkPipe(const UmmaLaunchParams& lp, cudaStream_t stream);
// The pipelined fused M128 key leaf at (128,128) (umma_bwd_kv_pipe.cuh): its preflight - the
// two-stage fused M128 leaf's own (UmmaBackwardDkdvCanLaunch<E, 128, 128, 3, 128, 4>), a key axis
// that fits in int32 and its own dynamic SMEM - and its launch, which re-tests that preflight
// (`false`: refused, nothing written). Defined in that header and instantiated for BF16 and FP16
// in instantiations/sca_umma_bwd_dkdv_d128_v128.cu, the TU whose in-situ launch reaches them,
// which also defines InSituKvPipeWouldLaunch, the (128,128) half of InSituKeyPairWouldLaunch
// (umma_bwd_entry.h).
template <typename E>
bool CanLaunchUmmaBackwardKvPipe(const UmmaLaunchParams& lp);
template <typename E>
bool LaunchUmmaBackwardKvPipe(const UmmaLaunchParams& lp, cudaStream_t stream);
bool InSituKvPipeWouldLaunch(const UmmaLaunchParams& lp);

// What the in-situ launch at (256,256) and at (128,128) does with a call, decided in ONE place:
// the launch below acts on it, and InSituKeyPairWouldLaunch (umma_bwd_entry.h) reports it to host
// code that must know before the launch whether the fused key body will run. The pair and the
// (128,128) pipelined leaf write dK, dV, dPrevK and dPrevV and nothing else, so no by-product of
// the fused body exists for a call that takes either.
//   kPair     (256,256): both pair leaves accept the call: the dV-only leaf, then the dK-only
//             leaf, runs;
//   kRefused  (256,256): "pair128p" was named at level 0 and the pair's preflight refused:
//             nothing runs;
//   kKvPipe   (128,128): the variable is absent or empty, the call is level 0 and the pipelined
//             fused M128 leaf's preflight accepts it: that leaf runs;
//   kFused    every other call, every other bucket included: the fused key body decides.
// The level gate is tested apart from the leaves' own preflights on purpose. Levels >= 1 carry
// an FP32 saved state and the subtract-before-scale form, which is the state of a DIFFERENT
// body that neither pair leaf implements, so naming the pair there cannot change that body: a
// named arm at levels 1-4 falls through to the fused leaf exactly like the default.
enum class InSituKeyPairDecision { kFused, kPair, kRefused, kKvPipe };

template <typename E, int kD, int kDv>
inline InSituKeyPairDecision DecideInSituKeyPair(const UmmaLaunchParams& lp) {
  if constexpr (kD == 256 && kDv == 256) {
    const bool named = KeyArmPair128pRequested();
    const bool level0 = !lp.params.saved_fp32 && !lp.params.subtract_before_scale;
    if (!level0 || !(named || KeyArmUnsetRequested())) return InSituKeyPairDecision::kFused;
    if (UmmaBackwardDkdvCanLaunch<E, kD, kDv, 2, 128, 0>(lp) &&
        CanLaunchUmmaBackwardDkPipe<E>(lp)) {
      return InSituKeyPairDecision::kPair;
    }
    return named ? InSituKeyPairDecision::kRefused : InSituKeyPairDecision::kFused;
  }
  if constexpr (kD == 128 && kDv == 128) {
    const bool level0 = !lp.params.saved_fp32 && !lp.params.subtract_before_scale;
    if (level0 && KeyArmUnsetRequested() && CanLaunchUmmaBackwardKvPipe<E>(lp)) {
      return InSituKeyPairDecision::kKvPipe;
    }
  }
  return InSituKeyPairDecision::kFused;
}

template <typename E, int kD, int kDv>
bool LaunchUmmaBackwardInSituDkdv(const UmmaLaunchParams& lp, cudaStream_t stream) {
  constexpr int kM = InSituKeyRows<kD, kDv>::value;
  constexpr int kA = InSituKeyArm<kD, kDv>::value;
  if constexpr (kD == 256 && kDv == 256) {
    // The default route at (256,256), BF16 and FP16: the pipelined dV-only M128 leaf, then the
    // pipelined dK-only M128 leaf on the same stream. Together they write every row of dK, dV,
    // dPrevK and dPrevV - one writer per row per product, no atomics, the fused M64 leaf's
    // operands in its accumulation order - so their outputs are bit-equal to the fused leaf's at
    // either determinism setting. Both leaves are asked before either launches (in
    // DecideInSituKeyPair), so a refusal writes nothing. Their accepted sets are the fused
    // leaf's keyed on one output each, at the 128-row tile (a stricter previous-plane alignment)
    // and with more dynamic SMEM, so the fused M64 leaf accepts every call the pair accepts.
    const InSituKeyPairDecision decision = DecideInSituKeyPair<E, kD, kDv>(lp);
    if (decision == InSituKeyPairDecision::kPair) {
      const bool ran_v = LaunchUmmaBackwardDvPipe<E>(lp, stream);
      const bool ran_k = LaunchUmmaBackwardDkPipe<E>(lp, stream);
      if (!ran_v || !ran_k) {
        throw std::runtime_error(
            "flash_sca native bwd (256,256) M128 pair: a leaf refused after its preflight");
      }
      InSituLaunchedKeyRows() = 128;
      return true;
    }
    // A NAMED pair at LEVEL 0 whose preflight refused is REPORTED, not silently served by the
    // fused leaf: a timing arm must not measure a body it did not name. Levels 1-4 never decide
    // kRefused, and neither does an absent or empty variable.
    if (decision == InSituKeyPairDecision::kRefused) return false;
  }
  if constexpr (kD == 128 && kDv == 128) {
    // The default route at (128,128), level 0, BF16 and FP16: the pipelined fused M128 leaf
    // (umma_bwd_kv_pipe.cuh). It writes what the two-stage fused M128 leaf below writes - every
    // row of dK, dV, dPrevK and dPrevV, one writer per row, no atomics - from bitwise the same
    // operands in the same accumulation order, so its outputs are bit-equal to that leaf's at
    // either determinism setting. Its preflight is that leaf's plus its own, and every call it
    // refuses (DecideInSituKeyPair) reaches the run-time decision below unchanged.
    if (DecideInSituKeyPair<E, kD, kDv>(lp) == InSituKeyPairDecision::kKvPipe) {
      if (!LaunchUmmaBackwardKvPipe<E>(lp, stream)) {
        throw std::runtime_error(
            "flash_sca native bwd (128,128) pipelined M128 key leaf: refused after its preflight");
      }
      InSituLaunchedKeyRows() = 128;
      return true;
    }
  }
  // The run-time decision, made once, on the same predicate the query uses: preferred
  // arm, else the M64 body. LaunchUmmaBackwardDkdv re-tests its own acceptance and
  // refuses rather than writing, so a refusal here cannot be reported as a native body.
  const bool wide = UmmaBackwardDkdvCanLaunch<E, kD, kDv, 3, kM, kA>(lp);
  if (wide) {
    const bool ran = LaunchUmmaBackwardDkdv<E, kD, kDv, 3, kM, kA>(lp, stream);
    if (ran) InSituLaunchedKeyRows() = kM;
    return ran;
  }
  if constexpr (kM != kBwdKeyRows) {
    const bool ran = LaunchUmmaBackwardDkdv<E, kD, kDv, 3, kBwdKeyRows>(lp, stream);
    if (ran) InSituLaunchedKeyRows() = kBwdKeyRows;
    return ran;
  }
  return false;
}

// ---------------------------------------------------------------------------------
// The ablation ladder's entry (probe/leaf_bench.cu --only=abl<N>): one untemplated
// launch and one untemplated query per arm index, so the bench selects a COMPILE-TIME
// arm with a run-time flag while every shipping entry point above stays exactly as it
// is - ABL defaults to 0 in all of them except the in-situ M128 tile, which ships ABL 4
// (InSituKeyArm) - and the arm index never reaches a run-time branch inside the round loop.
//
// BF16 only: the ladder measures the primary objective's element type, and restricting
// it here keeps the instantiation count at 5 arms x 2 buckets instead of 5 x 2 x 2.
//
// `false` keeps its meaning (nothing written, the caller may run the generic body), so an
// arm the device cannot hold is REFUSED and never silently degraded to the control: ABL 4
// at (256,256) needs two Q/dO stages = 329,728 B against the 232,448 B opt-in ceiling of
// the target part, and reporting that as "ABL 4 == ABL 0" would be a null result where
// the truth is that the lever is unavailable at that bucket.
template <typename E, int kD, int kDv, int ABL>
inline bool CanLaunchUmmaBackwardDkdvArmed(const UmmaLaunchParams& lp) {
  return UmmaBackwardDkdvCanLaunch<E, kD, kDv, 3, kBwdKeyRows, ABL>(lp);
}
template <typename E, int kD, int kDv, int ABL>
inline bool LaunchUmmaBackwardDkdvArmed(const UmmaLaunchParams& lp, cudaStream_t stream) {
  return LaunchUmmaBackwardDkdv<E, kD, kDv, 3, kBwdKeyRows, ABL>(lp, stream);
}

inline bool CanLaunchUmmaBackwardDkdvArm(const UmmaLaunchParams& lp, int abl) {
  if (lp.params.element != Element::kBFloat16) return false;
  const int d = QkBucket(lp.params.qk_dim), dv = ValueBucket(lp.params.value_dim);
  if (d == 128 && dv == 128) {
    switch (abl) {
      case 0: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 0>(lp);
      case 1: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 1>(lp);
      case 2: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 2>(lp);
      case 3: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 3>(lp);
      case 4: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 4>(lp);
      default: return false;
    }
  }
  if (d == 256 && dv == 256) {
    switch (abl) {
      case 0: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 0>(lp);
      case 1: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 1>(lp);
      case 2: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 2>(lp);
      case 3: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 3>(lp);
      case 4: return CanLaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 4>(lp);
      default: return false;
    }
  }
  return false;
}

inline bool LaunchUmmaBackwardDkdvArm(const UmmaLaunchParams& lp, cudaStream_t stream,
                                      int abl) {
  if (lp.params.element != Element::kBFloat16) return false;
  const int d = QkBucket(lp.params.qk_dim), dv = ValueBucket(lp.params.value_dim);
  if (d == 128 && dv == 128) {
    switch (abl) {
      case 0: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 0>(lp, stream);
      case 1: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 1>(lp, stream);
      case 2: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 2>(lp, stream);
      case 3: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 3>(lp, stream);
      case 4: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 128, 128, 4>(lp, stream);
      default: return false;
    }
  }
  if (d == 256 && dv == 256) {
    switch (abl) {
      case 0: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 0>(lp, stream);
      case 1: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 1>(lp, stream);
      case 2: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 2>(lp, stream);
      case 3: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 3>(lp, stream);
      case 4: return LaunchUmmaBackwardDkdvArmed<__nv_bfloat16, 256, 256, 4>(lp, stream);
      default: return false;
    }
  }
  return false;
}

// Independent-leaf entries: the same kernel body with the compile-time output
// set restricted to dK (4D+2Dv per admitted pair) or to dV (2D+2Dv). Each
// writes only its own product's rows - the other product's region is left
// bitwise untouched - so the split form runs them as two independent launches
// over the same grid. The determinism argument of the fused form carries over
// term for term: one writer per row per product, ascending query tiles then
// heads, no atomics, no dq accumulator. The refusal predicate is the fused
// leaf's, keyed on the owned output set (see Accepted): every other term -
// buckets, element, device, SMEM capacity - is unchanged.
template <typename E, int kD, int kDv>
bool LaunchUmmaBackwardDk(const UmmaLaunchParams& lp, cudaStream_t stream) {
  return LaunchUmmaBackwardDkdv<E, kD, kDv, 1>(lp, stream);
}
template <typename E, int kD, int kDv>
bool LaunchUmmaBackwardDv(const UmmaLaunchParams& lp, cudaStream_t stream) {
  return LaunchUmmaBackwardDkdv<E, kD, kDv, 2>(lp, stream);
}

// Reachable-pair routing + query for the key-owned leaves: a TU line without
// an Auto row is a compiled-but-unreachable cubin body. The pair list exists
// exactly ONCE - ONE forty-row table ((9 pairs x the two split output sets
// {1,2} + the two fused pairs {128,128},{256,256} x the fused output set {3})
// x 2 dtypes = 36 + 4) that both the launch and the query enumerate - so
// query and launch share the selection (one Auto query per Auto launch) and a
// drift between them is impossible by construction. The count in this sentence is
// load-bearing: the table is verified against it row by row, so it moves
// WITH the table and never after it (the four kWrite=3 fused rows are folded in
// here, so there is no separate fused ladder and the fused Auto is a table
// selection like any other).
namespace dkdv_detail {
template <typename E, int kD, int kDv, int K>
bool KeyCanOne(const UmmaLaunchParams& lp) {
  return UmmaBackwardDkdvCanLaunch<E, kD, kDv, K>(lp);
}
template <typename E, int kD, int kDv, int K>
bool KeyLaunchOne(const UmmaLaunchParams& lp, cudaStream_t stream) {
  return LaunchUmmaBackwardDkdv<E, kD, kDv, K>(lp, stream);
}
struct KeyAutoEntry {
  int d, dv, kwrite;
  bool bf16;
  bool (*can)(const UmmaLaunchParams&);
  bool (*launch)(const UmmaLaunchParams&, cudaStream_t);
};
constexpr KeyAutoEntry kKeyAutoTable[] = {
    {64, 64, 1, true, KeyCanOne<__nv_bfloat16, 64, 64, 1>,
     KeyLaunchOne<__nv_bfloat16, 64, 64, 1>},
    {64, 64, 1, false, KeyCanOne<__half, 64, 64, 1>,
     KeyLaunchOne<__half, 64, 64, 1>},
    {64, 64, 2, true, KeyCanOne<__nv_bfloat16, 64, 64, 2>,
     KeyLaunchOne<__nv_bfloat16, 64, 64, 2>},
    {64, 64, 2, false, KeyCanOne<__half, 64, 64, 2>,
     KeyLaunchOne<__half, 64, 64, 2>},
    {64, 128, 1, true, KeyCanOne<__nv_bfloat16, 64, 128, 1>,
     KeyLaunchOne<__nv_bfloat16, 64, 128, 1>},
    {64, 128, 1, false, KeyCanOne<__half, 64, 128, 1>,
     KeyLaunchOne<__half, 64, 128, 1>},
    {64, 128, 2, true, KeyCanOne<__nv_bfloat16, 64, 128, 2>,
     KeyLaunchOne<__nv_bfloat16, 64, 128, 2>},
    {64, 128, 2, false, KeyCanOne<__half, 64, 128, 2>,
     KeyLaunchOne<__half, 64, 128, 2>},
    {64, 256, 1, true, KeyCanOne<__nv_bfloat16, 64, 256, 1>,
     KeyLaunchOne<__nv_bfloat16, 64, 256, 1>},
    {64, 256, 1, false, KeyCanOne<__half, 64, 256, 1>,
     KeyLaunchOne<__half, 64, 256, 1>},
    {64, 256, 2, true, KeyCanOne<__nv_bfloat16, 64, 256, 2>,
     KeyLaunchOne<__nv_bfloat16, 64, 256, 2>},
    {64, 256, 2, false, KeyCanOne<__half, 64, 256, 2>,
     KeyLaunchOne<__half, 64, 256, 2>},
    {128, 64, 1, true, KeyCanOne<__nv_bfloat16, 128, 64, 1>,
     KeyLaunchOne<__nv_bfloat16, 128, 64, 1>},
    {128, 64, 1, false, KeyCanOne<__half, 128, 64, 1>,
     KeyLaunchOne<__half, 128, 64, 1>},
    {128, 64, 2, true, KeyCanOne<__nv_bfloat16, 128, 64, 2>,
     KeyLaunchOne<__nv_bfloat16, 128, 64, 2>},
    {128, 64, 2, false, KeyCanOne<__half, 128, 64, 2>,
     KeyLaunchOne<__half, 128, 64, 2>},
    {128, 128, 1, true, KeyCanOne<__nv_bfloat16, 128, 128, 1>,
     KeyLaunchOne<__nv_bfloat16, 128, 128, 1>},
    {128, 128, 1, false, KeyCanOne<__half, 128, 128, 1>,
     KeyLaunchOne<__half, 128, 128, 1>},
    {128, 128, 2, true, KeyCanOne<__nv_bfloat16, 128, 128, 2>,
     KeyLaunchOne<__nv_bfloat16, 128, 128, 2>},
    {128, 128, 2, false, KeyCanOne<__half, 128, 128, 2>,
     KeyLaunchOne<__half, 128, 128, 2>},
    {128, 256, 1, true, KeyCanOne<__nv_bfloat16, 128, 256, 1>,
     KeyLaunchOne<__nv_bfloat16, 128, 256, 1>},
    {128, 256, 1, false, KeyCanOne<__half, 128, 256, 1>,
     KeyLaunchOne<__half, 128, 256, 1>},
    {128, 256, 2, true, KeyCanOne<__nv_bfloat16, 128, 256, 2>,
     KeyLaunchOne<__nv_bfloat16, 128, 256, 2>},
    {128, 256, 2, false, KeyCanOne<__half, 128, 256, 2>,
     KeyLaunchOne<__half, 128, 256, 2>},
    {256, 64, 1, true, KeyCanOne<__nv_bfloat16, 256, 64, 1>,
     KeyLaunchOne<__nv_bfloat16, 256, 64, 1>},
    {256, 64, 1, false, KeyCanOne<__half, 256, 64, 1>,
     KeyLaunchOne<__half, 256, 64, 1>},
    {256, 64, 2, true, KeyCanOne<__nv_bfloat16, 256, 64, 2>,
     KeyLaunchOne<__nv_bfloat16, 256, 64, 2>},
    {256, 64, 2, false, KeyCanOne<__half, 256, 64, 2>,
     KeyLaunchOne<__half, 256, 64, 2>},
    {256, 128, 1, true, KeyCanOne<__nv_bfloat16, 256, 128, 1>,
     KeyLaunchOne<__nv_bfloat16, 256, 128, 1>},
    {256, 128, 1, false, KeyCanOne<__half, 256, 128, 1>,
     KeyLaunchOne<__half, 256, 128, 1>},
    {256, 128, 2, true, KeyCanOne<__nv_bfloat16, 256, 128, 2>,
     KeyLaunchOne<__nv_bfloat16, 256, 128, 2>},
    {256, 128, 2, false, KeyCanOne<__half, 256, 128, 2>,
     KeyLaunchOne<__half, 256, 128, 2>},
    {256, 256, 1, true, KeyCanOne<__nv_bfloat16, 256, 256, 1>,
     KeyLaunchOne<__nv_bfloat16, 256, 256, 1>},
    {256, 256, 1, false, KeyCanOne<__half, 256, 256, 1>,
     KeyLaunchOne<__half, 256, 256, 1>},
    {256, 256, 2, true, KeyCanOne<__nv_bfloat16, 256, 256, 2>,
     KeyLaunchOne<__nv_bfloat16, 256, 256, 2>},
    {256, 256, 2, false, KeyCanOne<__half, 256, 256, 2>,
     KeyLaunchOne<__half, 256, 256, 2>},
    // Folded fused rows (output set 3): exactly the two pairs the pre-fold pair
    // ladder served, enumerated by the same selector as the split rows.
    {128, 128, 3, true, KeyCanOne<__nv_bfloat16, 128, 128, 3>,
     KeyLaunchOne<__nv_bfloat16, 128, 128, 3>},
    {128, 128, 3, false, KeyCanOne<__half, 128, 128, 3>,
     KeyLaunchOne<__half, 128, 128, 3>},
    {256, 256, 3, true, KeyCanOne<__nv_bfloat16, 256, 256, 3>,
     KeyLaunchOne<__nv_bfloat16, 256, 256, 3>},
    {256, 256, 3, false, KeyCanOne<__half, 256, 256, 3>,
     KeyLaunchOne<__half, 256, 256, 3>},
};
inline const KeyAutoEntry* KeyAutoSelect(const UmmaLaunchParams& lp, int kWrite) {
  const Params& p = lp.params;
  const int d = QkBucket(p.qk_dim), dv = ValueBucket(p.value_dim);
  const bool bf = p.element == Element::kBFloat16;
  // A non-bf16 element either is half (matches the bf16 == false rows, which
  // then test p.element != kElement inside the templated Can) or is neither,
  // which those rows refuse identically - the width of the table needs no
  // third column.
  if (p.element != Element::kBFloat16 && p.element != Element::kHalf) return nullptr;
  for (const KeyAutoEntry& e : kKeyAutoTable) {
    if (e.d == d && e.dv == dv && e.kwrite == kWrite && e.bf16 == bf) return &e;
  }
  return nullptr;
}
}  // namespace dkdv_detail

// One untemplated Auto query per entry point that has an Auto launch: the refusal
// the mount sees. This one is the table's for every
// key-owned output set - kWrite in {1, 2, 3}: after the fold, the fused output
// set is two table rows like any other, and the twins below are its named form.
// `Accepted` stays internal: a plan built on it alone can claim a native body
// the launch then refuses (a post-decision refusal, which the error contract
// forbids falling back over), because it misses every field that keys the
// table - bucket, element, device, capacity, product nulls.
inline bool CanLaunchUmmaBackwardKeyAuto(const UmmaLaunchParams& lp, int kWrite) {
  const dkdv_detail::KeyAutoEntry* e = dkdv_detail::KeyAutoSelect(lp, kWrite);
  return e != nullptr && e->can(lp);
}

inline bool LaunchUmmaBackwardKeyAuto(const UmmaLaunchParams& lp, cudaStream_t stream,
                                      int kWrite) {
  const dkdv_detail::KeyAutoEntry* e = dkdv_detail::KeyAutoSelect(lp, kWrite);
  return e != nullptr && e->launch(lp, stream);
}
inline bool LaunchUmmaBackwardDkAuto(const UmmaLaunchParams& lp, cudaStream_t stream) {
  return LaunchUmmaBackwardKeyAuto(lp, stream, 1);
}
inline bool LaunchUmmaBackwardDvAuto(const UmmaLaunchParams& lp, cudaStream_t stream) {
  return LaunchUmmaBackwardKeyAuto(lp, stream, 2);
}
// And their query twins (one Auto query per entry point with an Auto launch):
inline bool CanLaunchUmmaBackwardDkAuto(const UmmaLaunchParams& lp) {
  return CanLaunchUmmaBackwardKeyAuto(lp, 1);
}
inline bool CanLaunchUmmaBackwardDvAuto(const UmmaLaunchParams& lp) {
  return CanLaunchUmmaBackwardKeyAuto(lp, 2);
}

// The fused twins, folded: the same names and call shapes the host seam already
// uses, now routed through the one table (kWrite == 3). Anything a
// mount may ask of the key side - fused or split - is one table query and one
// table launch away.
inline bool CanLaunchUmmaBackwardDkdvAuto(const UmmaLaunchParams& lp) {
  return CanLaunchUmmaBackwardKeyAuto(lp, 3);
}
inline bool LaunchUmmaBackwardDkdvAuto(const UmmaLaunchParams& lp,
                                       cudaStream_t stream) {
  return LaunchUmmaBackwardKeyAuto(lp, stream, 3);
}

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native

// One line per (element type, D bucket, Dv bucket) so an ordinary package build
// emits the sm_103a cubin for the leaf (setup.py globs native/instantiations/*.cu).
#define XATTN_UMMA_BWD_DKDV_INSTANTIATE(E, D, DV)                          \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkdv<E, D, DV>( \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t); \
  template bool xattn::ops::flash_sca_native::umma::UmmaBackwardDkdvCanLaunch<E, D, DV>( \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
// Same for the two independent output sets (the split leaves: family A's key
// side). K carries the output set (1 = dK, 2 = dV): the macro takes it as a
// parameter so the pair lists in the two split TUs are call-site text only
// and cannot drift away from each other.
#define XATTN_UMMA_BWD_KEY_INSTANTIATE(E, D, DV, K)                        \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkdv<E, D, DV, K>( \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
// The IN-SITU key-tile arm, instantiated per (element, D bucket, Dv bucket). It is a
// general macro and not a (128,128)-only one because the HOST selects it for every
// backward bucket it is compiled against: the selection switch asks
// CanLaunchUmmaBackwardInSituDkdv<E, d, dv> for whichever bucket the call lands in, so
// every one of those specialisations needs a definition even when the bucket's
// InSituKeyRowsFor() is the M64 default. A missing one does not fail the link - a
// shared object is allowed to carry an unresolved reference - it fails the IMPORT with
// the mangled name of the bucket, after the whole build has finished.
#define XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(E, D, DV)                           \
  template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardInSituDkdv<E, \
      D, DV>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);            \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardInSituDkdv<E,   \
      D, DV>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&,             \
             cudaStream_t);
// The M128 arm is the (128,128) row of the same macro: one definition of the template
// list, so the bucket that takes the M128 key tile and the buckets that fall back to M64
// cannot drift into "prepared 128 / launched nothing".
#define XATTN_UMMA_BWD_DKDV_M128_INSTANTIATE(E) \
  XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(E, 128, 128)
