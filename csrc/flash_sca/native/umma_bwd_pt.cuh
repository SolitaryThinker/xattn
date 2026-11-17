// Native Blackwell (sm_103a) SCA backward: the P^T / dS^T producer of the
// key-owned dK/dV leaf, in key-major (transposed) form.
//
// One definition of the element algebra, used by both sides of the leaf and by
// the host-side checks. This header owns ONLY the elementwise step of the
// key-owned family - mask, exponential, dS - and the packing of its two results
// into the 16-bit A-operand TMEM window the dK/dV MMAs read (operand path settled
// by the operand map: A comes from TMEM, `tcgen05.mma TS`, in place). It
// issues no MMA, allocates no TMEM and launches nothing: the caller owns the
// columns and the store, and this file adds no field to Params.
//
// ---------------------------------------------------------------------------
// What it computes, and where each line comes from (main 44100a5eb1)
//
// For a global key j (row of the transposed tile) and a query i (column), the
// generic SM80 key-owned kernels compute, per visible pair,
//
//   P^T [j,i] = exp2(S^T[j,i] * scale_log2 - lse2(i))            kernels.cuh:1014, 1283
//   dS^T[j,i] = P^T[j,i] * (dP^T[j,i] - delta(i))
//   lse2(i)   = isfinite(lse(i)) ? lse(i) * log2(e) : +INFINITY  kernels.cuh:LseLog2
//
// and dV += P^T @ dO (no scale factor) and dK += p.scale * (dS^T @ Q), the scale
// folded once at the dK store (kernels.cuh:1085, 1364). The scale is NOT this
// header's business: it is applied by the dK epilogue, so nothing here multiplies
// by p.scale.
//
// Visibility is the shared rule and this header has no second opinion about it
// (params.h header comment, native/visibility.cuh):
//
//   visible(i,j)  iff  lo(i) <= j <= hi(i)
//                      and (no segment planes  or  segment_exact[b] != 0
//                           or  k_segment[b,j] == q_segment[b,i])
//
// lo/hi are the row bounds of the shared tables - analytic
// (max(0, (pos/C - 1)*C) .. pos - strict, pos = i + Nt - Nq) when row_lo is null,
// explicit otherwise, reset metadata already folded in by the Python boundary.
// A row with lo > hi is empty and contributes exactly zero, as does every column
// with a non-finite LSE.
//
// ---------------------------------------------------------------------------
// Rules this header keeps, each one a condition stated elsewhere in this tree
//
// 1. SELECT, NEVER CARRY -INF THROUGH THE SCALE. Masked entries are dropped by a
//    predicate on the exponent's ARGUMENT, never by feeding -INFINITY through
//    `s * scale_log2`: for scale_log2 == 0 or < 0 that product is NaN or +INF
//    and one masked element would poison the whole row (the same trap umma_fwd.cuh
//    records at its mask select). Finite zero and negative scales are contract cases.
// 2. A NON-FINITE LSE IS ZERO PROBABILITY, whatever its sign. The generic seed
//    writes -INFINITY for a row with no visible key and its reader maps any
//    non-finite value to +INFINITY so that P == 0 (kernels.cuh:LseLog2); the
//    native forward leaf writes -INFINITY for the same row (umma_fwd.cuh, its
//    LSE convention). Testing isfinite() once accepts both: the exponent may be
//    evaluated for such a column, but the select in Probability() discards it, so
//    no value derived from a sentinel is ever returned.
// 3. 16-BIT STAGING IS ROUND-TO-NEAREST-EVEN, the same cast the generic seed
//    applies when it packs sP/sS (`Traits<E>::Pack`, kernels.cuh:78/101) and the
//    same one the native forward packs P with (ElemTraits<Elem>::Pack2,
//    umma_fwd.cuh). Both P^T and dS^T are staged in the input element type: the
//    accumulation stays FP32 (TMEM), only the operand is 16-bit.
// 4. DETERMINISM. Everything here is elementwise, so it cannot reorder an
//    accumulation: the leaf's determinism is the caller's (one writer per dK/dV
//    row, ascending query-tile and head order, no atomics). Two runs of this
//    function on the same inputs are bit-identical by construction.
//
// ---------------------------------------------------------------------------
// Format of one row block (the shape the caller hands it)
//
// A key row of the 64-key tile owns BM = 128 query columns, which do not fit in
// registers next to dP^T (128 + 128 floats). So the row is processed in blocks
// of kBlock = 32 columns, the same step the native forward reads TMEM in
// (umma_fwd.cuh:TmemLoadCols). Per block the caller supplies
//   s[kBlock]   S^T[j, c0 .. c0+kBlock)   raw, unscaled  (tcgen05.ld of the S^T window)
//   dp[kBlock]  dP^T[j, c0 .. c0+kBlock)  raw              (tcgen05.ld of the dP^T window)
//   meta[]      one ColumnMeta per column of the block (lse2, delta, lo, hi, q_segment)
// and receives
//   out_p[kBlock/2]   P^T  pairs, columns (2t, 2t+1) in one uint32, element 2t low
//   out_ds[kBlock/2]  dS^T pairs, same order
// which are the TMEM columns 16*blk .. 16*blk+15 of the packed window: two
// consecutive blocks (2m, 2m+1) fill exactly one 32-word buffer, i.e. one
// tcgen05.st of the shape the forward already issues for its own P window
// (umma_fwd.cuh:516-533: `pv[c >> 1] = ElemTraits<Elem>::Pack2(e0, e1)` for
// c stepping 2 over 64 keys, then one `TmemStore32(lane_base | (tbase + kPCol), pv)`).
// That is also where the low/high order below is pinned: element 2t is the low
// half, so the packed pair the MMA reads as K element 2t, 2t+1 is the one this
// function returns.
//
// Thread mapping of the callers (the pack in umma_bwd_dkdv.cuh). All 256 threads run this
// code; warps 0..3 and 4..7 cover the same key rows and split the query axis.
//   M64 (datapath offsets 0 and 16, not 0 and 64): key row r lives at lane
//        32*(r/16) + (r%16) for lane field 0 and 16 lanes higher for field 16, so lane L
//        and lane L+16 own the same key row. The low half computes columns 0..15 and the
//        high half columns 16..31 of every 32-column block (MakePtBlock* with kBlock/2
//        columns). The CALLER moves values across the halves with shfl_xor 16 around
//        these functions: S^T arrives at lane offset 0 and dP^T at lane offset 16, and the
//        packed P^T leaves at offset 16 and the packed dS^T at offset 0.
//   M128: lane L of the 128 TMEM lanes is key row L and works on its own row; the fp32
//        and packed windows are all at lane offset 0, so no value crosses lanes.
// Nothing in this file crosses lanes; every function here is per thread.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/params.h"
#include "flash_sca/native/visibility.cuh"

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace pt {

inline constexpr int kBlock = 32;  // query columns per call (a TMEM read step)
constexpr float kLog2e = 1.4426950408889634f;

// One query column's inputs: everything about the query row i that the elementwise
// step needs. `lo > hi` marks a row with no visible key (it still has to produce
// exact zeros, not a skip, when other columns of the tile are live).
struct ColumnMeta {
  float lse2 = INFINITY;   // natural-log LSE * log2(e); any non-finite value -> P == 0
  float delta = 0.f;       // rowsum(dy * saved) in FP32 (DeltaKernel, kernels.cuh:524)
  int32_t lo = 1;          // first visible key on the GLOBAL key axis, inclusive
  int32_t hi = 0;          // last visible key, inclusive; lo > hi -> empty row
  int64_t segment = 0;     // q_segment[b,i], read only when the batch row needs the test
};

// Is the pair (query column of `c`, global key j) visible? Verbatim the shared
// rule, in the order that lets `lo > hi` short-circuit.
__host__ __device__ __forceinline__ bool Visible(const ColumnMeta& c, int64_t j,
                                                int64_t key_segment, bool segments) {
  if (j < static_cast<int64_t>(c.lo) || j > static_cast<int64_t>(c.hi)) return false;
  if (segments && key_segment != c.segment) return false;
  return true;
}

// P^T for one pair. `visible` is Visible(...) of the same pair; a masked pair and a
// column with a non-finite LSE both return exactly 0.f. The exponent is evaluated for
// every pair, these two included; only the select decides what is returned.
__host__ __device__ __forceinline__ float Probability(float s, const ColumnMeta& c, float scale_log2,
                                                     bool visible) {
  // Compute, then select (branch-free). For a masked pair or a non-finite lse2 the
  // select returns the literal 0.f and exp2f's result - whatever it is, a NaN or an
  // infinity from a non-finite argument included - is discarded, never returned.
  const float v = exp2f(fmaf(s, scale_log2, -c.lse2));
  return (visible && isfinite(c.lse2)) ? v : 0.f;
}

// dS^T for one pair, from the P^T of the same pair.
__host__ __device__ __forceinline__ float ScoreGradient(float p, float dp, const ColumnMeta& c,
                                                        bool visible) {
  // A masked pair returns exactly 0 even when dP^T is non-finite, where the
  // seed's `p0 * (dp - delta)` with p0 == 0 would propagate a NaN. On contract
  // inputs dP^T is finite and the two forms agree element for element.
  const float v = p * (dp - c.delta);
  return visible ? v : 0.f;
}

template <class Elem>
struct Pack2;
template <>
struct Pack2<__nv_bfloat16> {
  // Round-to-nearest-even, half the pair in the high 16 bits (the order the A
  // operand reads: element 2t low, 2t+1 high). Agrees with umma_fwd.cuh's
  // ElemTraits<__nv_bfloat16>::Pack2 and with kernels.cuh's Traits<__nv_bfloat16>.
  __host__ __device__ static uint32_t Of(float lo, float hi) {
    const uint32_t l = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo)));
    const uint32_t h = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi)));
    return l | (h << 16);
  }
};
template <>
struct Pack2<__half> {
  __host__ __device__ static uint32_t Of(float lo, float hi) {
    const uint32_t l = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(lo)));
    const uint32_t h = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(hi)));
    return l | (h << 16);
  }
};

// ---------------------------------------------------------------- fast core
//
// The instruction budget, not the shared traffic or the bank conflicts, is what
// binds the D=256 key leaves: DkPipe issues 0.73 warp instructions
// per visible pair against the reference's 0.31. Two of those instructions per
// element are avoidable without touching a single value the contract tolerates:
//
//   * two scalar round-to-nearest converts per pair lower to two `cvt`; the
//     packed `cvt.rn.bf16x2/f16x2` is one per pair, and it is the SAME
//     round-to-nearest-even, low element in the low 16 bits: Pack2Fast::Of(a,b)
//     is bit-identical to Pack2<Elem>::Of(a,b). This part changes no value.
//
// A faster exponential is here as well: `ex2.approx.ftz.f32` replaces `exp2f`'s
// per-element denormal guard. That is a value change in the subnormal tail (P below
// 2^-126), which BF16 partially preserves and an FP32 dS can lift back to normal - a tiny
// difference judged by the frozen contract's FP64 tolerances, which have accepted it at
// (128,128) and in every dQ bucket. See FastExp2 below for why an earlier
// contract failure was the unguarded suffix mask and not this.
//
// Both live here so every key-leaf caller (DkPipe, DvPipe, KvPipe, the TMEM key
// leaf) shares one definition instead of one per file.
// The elementwise exponent.
//
// `exp2f` compiles to `MUFU.EX2` plus a per-element denormal guard (des-query-1's SASS
// reading), so `ex2.approx.ftz.f32` on the same argument is the same instruction
// without that guard. It is a VALUE CHANGE, in the subnormal tail only: P differs where
// the result falls below 2^-126. That is not 'zero either way' - FP16's convert flushes
// those to zero, but BF16 keeps subnormals down to 2^-133, and dS = P*(dP-delta) is formed
// in FP32, so a |dP-delta| above 1 can lift such a P back to a normal FP32 dS. The
// magnitude is tiny, and what judges it is the frozen contract's FP64 tolerances, which
// have accepted it at (128,128) and in every dQ bucket including (256,256).
// A two-arm build of the KvPipe pack puts the instruction alone at -7.9..-12.1% on
// four rows and -6.8..-16.8% on nine.
//
// An earlier revision carried it; that revision's contract run failed 246 cases, and those were
// the UNGUARDED SUFFIX MASK, not the exponent: an error of ~2^-22
// relative cannot reach 1.5e-3 at sequence length 7, and `inf` relative differences mean a
// masked element came out nonzero, which a guardless MUFU.EX2 cannot produce. So it is
// here, and it is the only value change in this change set.
__host__ __device__ __forceinline__ float FastExp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

template <class Elem>
struct Pack2Fast;
template <>
struct Pack2Fast<__nv_bfloat16> {
  __host__ __device__ static uint32_t Of(float lo, float hi) {
    const __nv_bfloat162 v = __float22bfloat162_rn(make_float2(lo, hi));
    return *reinterpret_cast<const uint32_t*>(&v);
  }
};
template <>
struct Pack2Fast<__half> {
  __host__ __device__ static uint32_t Of(float lo, float hi) {
    const __half2 v = __float22half2_rn(make_float2(lo, hi));
    return *reinterpret_cast<const uint32_t*>(&v);
  }
};

// ---------------------------------------------------------------- packed f32x2
//
// The pack builds four independent floats per pair - `fma(s, scale, -lse2)` twice, then
// `dp - delta` twice, then `p * (...)` twice - and on sm_100 each of those has a packed
// two-lane form. `fma/mul/sub .rn.f32x2` are IEEE round-to-nearest-even PER LANE, the same
// rounding, the same operands and the same order as the scalar forms they replace, so this
// changes no value at all: it is one instruction where there were two.
//
// What a profile of these leaves measures is exactly this arithmetic: DkPipe at 0.9B runs
// 15.5 fp32 instructions per visible pair against the reference's 5.3, and fp32 is 81% of
// its excess over the reference. Three of the pair's four fp32 chains halve.
//
// The exponential stays scalar: there is no packed exponential, so `FastExp2` is applied
// per lane and the packed forms cover the fma, the subtract and the multiply.
__device__ __forceinline__ float2 FmaX2(float2 a, float2 b, float2 c) {
  unsigned long long pa = *reinterpret_cast<unsigned long long*>(&a);
  unsigned long long pb = *reinterpret_cast<unsigned long long*>(&b);
  unsigned long long pc = *reinterpret_cast<unsigned long long*>(&c);
  unsigned long long pr;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(pr) : "l"(pa), "l"(pb), "l"(pc));
  return *reinterpret_cast<float2*>(&pr);
}
__device__ __forceinline__ float2 MulX2(float2 a, float2 b) {
  unsigned long long pa = *reinterpret_cast<unsigned long long*>(&a);
  unsigned long long pb = *reinterpret_cast<unsigned long long*>(&b);
  unsigned long long pr;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(pr) : "l"(pa), "l"(pb));
  return *reinterpret_cast<float2*>(&pr);
}
__device__ __forceinline__ float2 SubX2(float2 a, float2 b) {
  unsigned long long pa = *reinterpret_cast<unsigned long long*>(&a);
  unsigned long long pb = *reinterpret_cast<unsigned long long*>(&b);
  unsigned long long pr;
  asm("sub.rn.f32x2 %0, %1, %2;" : "=l"(pr) : "l"(pa), "l"(pb));
  return *reinterpret_cast<float2*>(&pr);
}

// p = visible && isfinite(lse2) ? exp2(fma(s, scale, -lse2)) : 0, two columns at once.
__device__ __forceinline__ float2 ProbabilityX2(float2 s, float2 lse2, float scale_log2, bool ok0,
                                                bool ok1) {
  const float2 e = FmaX2(s, make_float2(scale_log2, scale_log2), make_float2(-lse2.x, -lse2.y));
  float2 p = make_float2(FastExp2(e.x), FastExp2(e.y));
  if (!ok0 || !isfinite(lse2.x)) p.x = 0.f;
  if (!ok1 || !isfinite(lse2.y)) p.y = 0.f;
  return p;
}

// dS pair = p * (dp - delta), masked per lane. Pack with Pack2Fast afterwards.
__device__ __forceinline__ float2 ScoreGradientX2(float2 p, float2 dp, float2 delta, bool ok0,
                                                  bool ok1) {
  float2 v = MulX2(p, SubX2(dp, delta));
  if (!ok0) v.x = 0.f;
  if (!ok1) v.y = 0.f;
  return v;
}

// The elementwise step for one key row and one block of query columns.
// `j` is the GLOBAL key index of this row, `key_segment` its k_segment (ignored
// when !segments), `meta` points at the block's first column.
template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlock(const float (&s)[kCols],
                                                     const float (&dp)[kCols],
                                                     const ColumnMeta* meta, int64_t j,
                                                     int64_t key_segment, bool segments,
                                                     float scale_log2, uint32_t* out_p,
                                                     uint32_t* out_ds) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const ColumnMeta& a = meta[c0];
    const bool ok0 = Visible(a, j, key_segment, segments);
    const float p0 = Probability(s[c0], a, scale_log2, ok0);
    const float ds0 = ScoreGradient(p0, dp[c0], a, ok0);
    const ColumnMeta& b = meta[c0 + 1];
    const bool ok1 = Visible(b, j, key_segment, segments);
    const float p1 = Probability(s[c0 + 1], b, scale_log2, ok1);
    const float ds1 = ScoreGradient(p1, dp[c0 + 1], b, ok1);
    out_p[t] = Pack2<Elem>::Of(p0, p1);
    out_ds[t] = Pack2<Elem>::Of(ds0, ds1);
  }
}

// Same, for a block the caller has PROVED fully visible with no segment test
// (umma_tile_mask.cuh: FullTileExact() on the same key range and query tile, and
// every column of the block inside its row's window). Skips the predicate only;
// the arithmetic is identical, so the two paths agree bit for bit whenever the
// caller's proof is right - which is what makes this switch testable.
template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockFull(const float (&s)[kCols],
                                                         const float (&dp)[kCols],
                                                         const ColumnMeta* meta,
                                                         float scale_log2, uint32_t* out_p,
                                                         uint32_t* out_ds) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const float p0 = Probability(s[c0], meta[c0], scale_log2, true);
    const float p1 = Probability(s[c0 + 1], meta[c0 + 1], scale_log2, true);
    out_p[t] = Pack2<Elem>::Of(p0, p1);
    out_ds[t] = Pack2<Elem>::Of(ScoreGradient(p0, dp[c0], meta[c0], true),
                                ScoreGradient(p1, dp[c0 + 1], meta[c0 + 1], true));
  }
}

// ---------------------------------------------------------------- SoA column meta
//
// The same elementwise step and result as MakePtBlock{,Full}, but the per-column
// inputs are read from STRUCT-OF-ARRAYS planes instead of an array of 24-byte
// ColumnMeta. The pipelined dK key leaf (umma_bwd_dk_pipe.cuh) is key-major: each
// pack thread owns one key row and must read the metadata of all of the block's
// QUERY columns, so with the AoS table it issues a strided 24 B load per column
// (the 194.8M LDS a profile measured on DkPipe). With SoA the block's lse2/delta are
// contiguous floats, so the same reads vectorize and the pack stores are
// bank-conflict-free. The arithmetic is IDENTICAL to Probability/ScoreGradient/
// Visible - the two paths agree bit for bit; only the gather layout differs.

__host__ __device__ __forceinline__ bool VisibleSoA(int32_t lo, int32_t hi, int64_t seg, int64_t j,
                                                    int64_t key_segment, bool segments) {
  if (j < static_cast<int64_t>(lo) || j > static_cast<int64_t>(hi)) return false;
  if (segments && key_segment != seg) return false;
  return true;
}

__host__ __device__ __forceinline__ float ProbabilitySoA(float s, float lse2v, float scale_log2,
                                                         bool visible) {
  const float v = FastExp2(fmaf(s, scale_log2, -lse2v));
  return (visible && isfinite(lse2v)) ? v : 0.f;
}

// dS^T only (the dK leaf never consumes P^T, so the packed-P write is dropped):
// out_ds[t] = two packed dS^T of columns (2t, 2t+1), element 2t low.
template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockSoA(const float (&s)[kCols],
                                                        const float (&dp)[kCols],
                                                        const float* lse2, const float* delta,
                                                        const int32_t* lo, const int32_t* hi,
                                                        const int64_t* seg, int64_t j,
                                                        int64_t key_segment, bool segments,
                                                        float scale_log2, uint32_t* out_ds) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const bool ok0 = VisibleSoA(lo[c0], hi[c0], seg[c0], j, key_segment, segments);
    const bool ok1 = VisibleSoA(lo[c0 + 1], hi[c0 + 1], seg[c0 + 1], j, key_segment, segments);
    const float2 p = ProbabilityX2(make_float2(s[c0], s[c0 + 1]),
                                   make_float2(lse2[c0], lse2[c0 + 1]), scale_log2, ok0, ok1);
    const float2 ds = ScoreGradientX2(p, make_float2(dp[c0], dp[c0 + 1]),
                                      make_float2(delta[c0], delta[c0 + 1]), ok0, ok1);
    out_ds[t] = Pack2Fast<Elem>::Of(ds.x, ds.y);
  }
}

// Same, for a block PROVED fully visible with no segment test (FullTileExact):
// needs only lse2 and delta, so it skips the lo/hi/segment plane reads entirely.
template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockFullSoA(const float (&s)[kCols],
                                                            const float (&dp)[kCols],
                                                            const float* lse2, const float* delta,
                                                            float scale_log2, uint32_t* out_ds) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const float2 p = ProbabilityX2(make_float2(s[c0], s[c0 + 1]),
                                   make_float2(lse2[c0], lse2[c0 + 1]), scale_log2, true, true);
    const float2 ds = ScoreGradientX2(p, make_float2(dp[c0], dp[c0 + 1]),
                                      make_float2(delta[c0], delta[c0 + 1]), true, true);
    out_ds[t] = Pack2Fast<Elem>::Of(ds.x, ds.y);
  }
}

// ---------------------------------------------------------------- suffix mask
//
// The dS^T-only pack for a key-major block whose bounds are the ANALYTIC interval
// (p.row_lo == nullptr), whose batch row needs no segment test, and whose block cannot
// straddle a chunk boundary - the caller checks that with `chunk % kCols == 0` and
// `(total_length - q_length) % kCols == 0`, since the block's positions are
// p = i + (total_length - q_length) and the contract's chunk sizes run 1..257. Given
// that alignment, the visible columns of a key row j are a SUFFIX of the block: `lo` is
// constant inside it and `hi(c) = hi(c0) + c` has slope exactly 1 (visibility.cuh
// RowBounds: hi = i + total - q - strict). So the mask is one scalar shift -
// `~0u << clamp(j - hi(c0), 0, kCols)` - and empty when j < lo(c0), in place of a 24 B
// per-column gather plus two compares and a select each.
// The per-element arithmetic is ProbabilityX2/ScoreGradientX2 over the same operands in
// the same order, so the values equal the scalar SoA path's except for the exponential,
// which is the one value change in this change set (FastExp2, see above). A block that
// fails either guard (explicit bounds, or a segment test) uses MakePtBlockSoA with the
// same core, so no caller loses the packed-convert/f32x2 win.
__host__ __device__ __forceinline__ uint32_t SuffixMask(int64_t j, int32_t hi_c0, int32_t lo_c0,
                                                        int kCols) {
  if (j < static_cast<int64_t>(lo_c0)) return 0u;
  const int64_t sh = j - static_cast<int64_t>(hi_c0);
  if (sh <= 0) return ~0u;
  if (sh >= kCols) return 0u;
  return ~0u << static_cast<unsigned>(sh);
}

// Per-block form of the suffix guard, for explicit row bounds and segments. A
// 32-column block whose columns all have the SAME lo, hi(c) = hi(c0) + c, and (when the
// batch row tests segments) the same query segment has, for every key row j, the visible
// set {c : lo <= j <= hi(c0) + c, key_segment(j) == segment} - the suffix SuffixMask
// builds, or empty when the key's segment differs. That is the exact per-column rule
// (Visible/VisibleSoA), so the values are the general path's bit for bit; the test is on
// the staged metadata itself, not on how it was produced, so it holds for analytic bounds,
// BOS row bounds, previous K/V and any chunk. Blocks that straddle a row (document) start,
// a chunk boundary or the end of the query axis fail it and keep the general path.
// Warp-collective: lane L passes column c0 + L's lo/hi/segment; every lane gets the answer.
#ifdef __CUDACC__
__device__ __forceinline__ bool BlockIsSuffix(int32_t lo, int32_t hi, int64_t segment,
                                              bool segments) {
  const int lane = static_cast<int>(threadIdx.x & 31u);
  const int32_t lo0 = __shfl_sync(0xffffffffu, lo, 0);
  const int32_t hi0 = __shfl_sync(0xffffffffu, hi, 0);
  const long long s0 = __shfl_sync(0xffffffffu, static_cast<long long>(segment), 0);
  const bool ok = lo == lo0 && static_cast<int64_t>(hi) == static_cast<int64_t>(hi0) + lane &&
                  (!segments || static_cast<long long>(segment) == s0);
  return __all_sync(0xffffffffu, ok) != 0;
}
#endif

// The visible mask of key row j in a block that passed BlockIsSuffix.
__host__ __device__ __forceinline__ uint32_t BlockSuffixMask(int64_t j, int32_t hi_c0,
                                                             int32_t lo_c0, int64_t seg_c0,
                                                             int64_t key_segment, bool segments,
                                                             int kCols) {
  if (segments && key_segment != seg_c0) return 0u;
  return SuffixMask(j, hi_c0, lo_c0, kCols);
}

template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockSuffix(const float (&s)[kCols],
                                                           const float (&dp)[kCols],
                                                           const float* lse2, const float* delta,
                                                           uint32_t mask, float scale_log2,
                                                           uint32_t* out_ds) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const bool ok0 = (mask >> c0) & 1u;
    const bool ok1 = (mask >> (c0 + 1)) & 1u;
    const float2 p = ProbabilityX2(make_float2(s[c0], s[c0 + 1]),
                                   make_float2(lse2[c0], lse2[c0 + 1]), scale_log2, ok0, ok1);
    const float2 ds = ScoreGradientX2(p, make_float2(dp[c0], dp[c0 + 1]),
                                      make_float2(delta[c0], delta[c0 + 1]), ok0, ok1);
    out_ds[t] = Pack2Fast<Elem>::Of(ds.x, ds.y);
  }
}

// ---------------------------------------------------------------- dV leaf (P^T only)
//
// The dV-only leaf (umma_bwd_dv_pipe.cuh) packs P^T and nothing else: dV += P^T dO,
// and its dS^T output is never stored. So its pack computes only the probability,
// through the same fast core, and skips the dS arithmetic and the second convert
// entirely. Same layout and lane mapping as the dS packs above, AoS ColumnMeta so
// the caller's existing smeta staging is unchanged.
template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockFullP(const float (&s)[kCols],
                                                          const ColumnMeta* meta, float scale_log2,
                                                          uint32_t* out_p) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const float2 p = ProbabilityX2(make_float2(s[c0], s[c0 + 1]),
                                   make_float2(meta[c0].lse2, meta[c0 + 1].lse2), scale_log2, true,
                                   true);
    out_p[t] = Pack2Fast<Elem>::Of(p.x, p.y);
  }
}

template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockSuffixP(const float (&s)[kCols],
                                                            const ColumnMeta* meta, uint32_t mask,
                                                            float scale_log2, uint32_t* out_p) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const bool ok0 = (mask >> c0) & 1u;
    const bool ok1 = (mask >> (c0 + 1)) & 1u;
    const float2 p = ProbabilityX2(make_float2(s[c0], s[c0 + 1]),
                                   make_float2(meta[c0].lse2, meta[c0 + 1].lse2), scale_log2, ok0,
                                   ok1);
    out_p[t] = Pack2Fast<Elem>::Of(p.x, p.y);
  }
}

template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockFastP(const float (&s)[kCols],
                                                          const ColumnMeta* meta, int64_t j,
                                                          int64_t key_segment, bool segments,
                                                          float scale_log2, uint32_t* out_p) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const bool ok0 = Visible(meta[c0], j, key_segment, segments);
    const bool ok1 = Visible(meta[c0 + 1], j, key_segment, segments);
    const float2 p = ProbabilityX2(make_float2(s[c0], s[c0 + 1]),
                                   make_float2(meta[c0].lse2, meta[c0 + 1].lse2), scale_log2, ok0,
                                   ok1);
    out_p[t] = Pack2Fast<Elem>::Of(p.x, p.y);
  }
}

// ---------------------------------------------------------------- dV leaf (P^T only), SoA lse2
//
// The same P^T-only values as MakePtBlock{Full,Suffix,Fast}P above, but lse2 is read
// from a contiguous SoA float plane instead of gathered from the 24-byte AoS ColumnMeta, and
// the per-element `isfinite(lse2)` select inside Probability/ProbabilityX2 is dropped. Every
// lse2 this leaf's own staging writes is already ColumnOf's `isfinite(lse) ? lse * kLog2e :
// INFINITY` (above), i.e. already +inf wherever the row's own lse was not finite, so
// FastExp2(-inf) alone already returns exactly 0.f (ex2.approx.ftz(-inf) = +0, no NaN): the
// select decided nothing an already-+inf lse2 did not decide for it. This is the KvPipe
// mechanism (umma_bwd_kv_pipe.cuh), applied here to the dV-only pack's probability only - the
// visibility mask (ok0/ok1, from the window and segment test) is UNCHANGED and still zeros
// every column the contract requires zeroed; only the redundant finite-lse test is gone. The
// lo/hi/segment plane stays the AoS ColumnMeta (unchanged callers, unchanged suffix-mask path);
// only the per-column lse2 read moves to the plane, so the gather this replaces is the one
//32-column loop's worth of strided 24 B loads for the ONE field it actually dereferences.
__device__ __forceinline__ float2 ProbabilityX2Fast(float2 s, float2 lse2, float scale_log2,
                                                     bool ok0, bool ok1) {
  const float2 e = FmaX2(s, make_float2(scale_log2, scale_log2), make_float2(-lse2.x, -lse2.y));
  float2 p = make_float2(FastExp2(e.x), FastExp2(e.y));
  if (!ok0) p.x = 0.f;
  if (!ok1) p.y = 0.f;
  return p;
}

template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockFullPSoA(const float (&s)[kCols],
                                                              const float* lse2, float scale_log2,
                                                              uint32_t* out_p) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const float2 p = ProbabilityX2Fast(make_float2(s[c0], s[c0 + 1]),
                                       make_float2(lse2[c0], lse2[c0 + 1]), scale_log2, true,
                                       true);
    out_p[t] = Pack2Fast<Elem>::Of(p.x, p.y);
  }
}

template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockSuffixPSoA(const float (&s)[kCols],
                                                                const float* lse2, uint32_t mask,
                                                                float scale_log2,
                                                                uint32_t* out_p) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const bool ok0 = (mask >> c0) & 1u;
    const bool ok1 = (mask >> (c0 + 1)) & 1u;
    const float2 p = ProbabilityX2Fast(make_float2(s[c0], s[c0 + 1]),
                                       make_float2(lse2[c0], lse2[c0 + 1]), scale_log2, ok0, ok1);
    out_p[t] = Pack2Fast<Elem>::Of(p.x, p.y);
  }
}

template <class Elem, int kCols = kBlock>
__host__ __device__ __forceinline__ void MakePtBlockFastPSoA(const float (&s)[kCols],
                                                              const float* lse2,
                                                              const ColumnMeta* meta, int64_t j,
                                                              int64_t key_segment, bool segments,
                                                              float scale_log2, uint32_t* out_p) {
  static_assert(kCols % 2 == 0, "pairs");
#pragma unroll
  for (int t = 0; t < kCols / 2; ++t) {
    const int c0 = 2 * t;
    const bool ok0 = Visible(meta[c0], j, key_segment, segments);
    const bool ok1 = Visible(meta[c0 + 1], j, key_segment, segments);
    const float2 p = ProbabilityX2Fast(make_float2(s[c0], s[c0 + 1]),
                                       make_float2(lse2[c0], lse2[c0 + 1]), scale_log2, ok0, ok1);
    out_p[t] = Pack2Fast<Elem>::Of(p.x, p.y);
  }
}

// ---------------------------------------------------------------- column gather

// ColumnMeta of query row i of head hq, from the shared tables only.
//
// PRECONDITION, written at the read site (the same one umma_fwd.cuh documents for
// q_segment): the metadata planes are DENSE, i.e. row stride == q_length / Nt.
// host.cc calls .contiguous() on both planes before Params is built, and the
// public boundary hands over q_segment_idx = k_segment_idx[:, -N:], whose row
// stride is C+N exactly when previous K/V is given. A strided view would read
// another batch row's metadata; the only shape that could reach it is
// prev_length > 0 && batch > 1, which no traced call has.
__device__ __forceinline__ ColumnMeta ColumnOf(const Params& p, int64_t b, int64_t hq, int64_t i,
                                               bool segments, bool row_in_range) {
  ColumnMeta c;
  if (!row_in_range) {  // past q_length: dead, and it must stay dead
    c.lo = 1;
    c.hi = 0;
    return c;
  }
  const detail::Bounds bd = detail::RowBounds(p, b, i);
  c.lo = bd.lo;
  c.hi = bd.hi;
  const int64_t stat = (b * p.query_heads + hq) * p.q_length + i;
  const float lse = p.lse_in != nullptr ? p.lse_in[stat] : -INFINITY;
  // any non-finite LSE -> +INFINITY, so Probability() returns exactly 0 (rule 2)
  c.lse2 = isfinite(lse) ? lse * kLog2e : INFINITY;
  c.delta = p.delta != nullptr ? p.delta[stat] : 0.f;
  c.segment = segments ? p.q_segment[b * p.q_length + i] : 0;
  return c;
}

// ---------------------------------------------------------- packed TMEM window

// Column of the packed window that takes block `blk` of a row: blocks are 32 fp32
// input columns wide and decimate to 16 packed words, so block blk lands at
// 16*blk .. 16*blk+15 (the in-place scheme). Ascending blocks are safe in place
// because the next unread input column (32*(blk+1)) is past this block's output
// (16*blk+15) for every blk >= 0 - the caller can therefore hand this function the
// same window it read `s` from, with no second allocation.
__host__ __device__ __forceinline__ int PackedColumn(int window_base, int blk) {
  return window_base + 16 * blk;
}

// The packed window needs NO pre-zeroing: MakePtBlock writes all kBlock/2 words of
// every block unconditionally, and a masked pair, a dead row (lo > hi) and a
// non-finite LSE all become exactly 0 through the predicates above rather than
// being skipped. So a tile with no visible pair writes zeros and its MMAs add
// nothing - the same property the forward leaf relies on when it issues the PV mma
// for every key tile with a CTA-uniform accumulate flag (umma_fwd.cuh header note),
// and the reason a fully dead query tile can be dropped by the caller without
// touching the accumulators.
//
// Handoff to the MMA (the caller's, not this header's): TmemWaitSt() after the
// tcgen05.st of both packed windows, then TmemFenceBeforeSync(), __syncthreads(),
// TmemFenceAfterSync() - the same fence pair the forward uses to hand P from 128
// threads to the warp that issues the MMA (umma_fwd.cuh TmemFenceBeforeSync /
// TmemFenceAfterSync, "tcgen05.wait::st only orders the ISSUING thread's own TMEM
// accesses"). Skipping the fences is a race that no test of the values here can
// see.

}  // namespace pt
}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
