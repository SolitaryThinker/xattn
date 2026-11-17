// Native SM103 advanced (level 2-4) reader backward: shared device math.
//
// Header only: __device__ __forceinline__ functions and constants, no kernels,
// no global state, no allocation.  Used by both ownership-partitioned leaves
// (adv_bwd_dq.cu and adv_bwd_dkdv.cu) so that the two
// kernels cannot disagree about admission, the P/dS/dV-operand algebra or the
// sink grids.
//
// Every formula below is the pinned Hopper mainloop
// (csrc/attention/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:1846-2050) written
// once, with the line links carried in adv_bwd_native.h items 1-7 and the units
// fixed there ("Units and conventions").  Nothing here changes a public
// signature or a public check.
//
// Conventions (identical in both leaves):
//   * row i of a reader sits at key position i + offset (offset = Nt - Nq);
//     reader 1 admits j <= pos, reader 2 (strict) j <= pos - 1, and when both
//     segment planes are present only keys with k_seg[b,j] == q_seg[b,i].
//   * meta planes are [6, B, Hq, Nq] with count = B*Hq*Nq:
//       0 raw-score row max m, 1 anchored dP a, 2 dO row scale 2^e (level 4),
//       3 anchor column index (-1 iff the admitted set is empty), 4/5 left/right.
//   * logz is the NATURAL log of the shifted row normalizer (the public LSE is
//     scale*m + logz); the exp2 needs lz2 = logz * log2(e), done here.
//   * stats planes are [4, B, Hq, Nq]: rho at +0*count, mass at +1*count,
//     variance at +2*count, anchor probability at +3*count -- FP64 sums in Main
//     mode, int64 fixed point in ordered mode (same buffer).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "attention/precision/native/adv_bwd_native.h"

namespace xattn {
namespace ops {
namespace native_adv {
namespace bwdmath {

// ------------------------------------------------------------------ constants

// log2(e): lz2 = logz * log2(e), and P = exp2((s - m)*scale_log2 - lz2).
constexpr float kLog2e = 1.44269504088896340735992468100189214f;

constexpr int kPlaneRho = 0;
constexpr int kPlaneMass = 1;
constexpr int kPlaneVariance = 2;
constexpr int kPlaneAnchor = 3;
constexpr int kPlaneCount = 4;

// Ordered mode rounds one partial per row and key tile onto the fixed grid, so
// a leaf that walks keys in tiles of this many keys flushes at each boundary.
constexpr int kSinkTile = 32;

// Fixed-point grids: rho 2^(62-e), variance 2^(68-2e), mass/anchor 2^60; the
// caller decodes by 2^(e-62), 2^(2e-68) and 2^-60.
constexpr int kGridBiasRho = 62;
constexpr int kGridBiasVariance = 68;
constexpr int kGridBiasMass = 60;

// ------------------------------------------------------------------- scalars

// log2 of the natural-log workspace normalizer (adv_bwd_native.h unit note).
__device__ __forceinline__ float lz2_from_logz(float logz_natural) {
  return __fmul_rn(logz_natural, kLog2e);
}

// Levels 2/3 receive the exact unit dO scale; level 4 stages 2^e in meta[2].
__device__ __forceinline__ float two_e_of(float meta_scale, int level) {
  return level == 4 ? meta_scale : 1.0f;
}

// Exact 2^-e for a power-of-two 2^e (:1993: __int_as_float(0x7f000000 - bits)).
__device__ __forceinline__ float inv_two_e(float two_e) {
  return __int_as_float(0x7f000000 - __float_as_int(two_e));
}

// 2^(e/2) from 2^e (:1901: (as_int(2^e) >> 1) + 0x1fc00000); exact for even e,
// which the level-4 dO preprocess guarantees (it clears the low bit).
__device__ __forceinline__ float half_scale(float two_e) {
  // 2^(e/2) from 2^e.  A bit-halved exponent is exact only for even e: for e = 1
  // it returns 1.5 and for e = 3 it returns 3.0, which squares to 2.25x and
  // 1.125x the right dV operand.  Halve the exponent and take it through exp2f.
  const int e = static_cast<int>(__float_as_uint(two_e) >> 23) - 127;
  return exp2f(0.5f * static_cast<float>(e));
}

// Integer e of the row scale, used only as the grid exponent when the caller
// does not supply rho_exponent.  The plane holds exactly 2^e, so ilogbf is
// exact; levels 2/3 (plane ignored) give 0.
__device__ __forceinline__ int exponent_of(float two_e) {
  if (!(two_e > 0.0f) || !isfinite(two_e)) return 0;
  return ilogbf(two_e);
}

// Grid exponent of this row: the caller's per-row e_i when present (ordered
// levels 3/4), else derived from the dO scale plane.
__device__ __forceinline__ int grid_exponent(const int32_t* rho_exponent, int64_t row,
                                            float two_e) {
  return rho_exponent != nullptr ? static_cast<int>(rho_exponent[row]) : exponent_of(two_e);
}

__device__ __forceinline__ int sink_grid_exp_rho(int e) { return kGridBiasRho - e; }
__device__ __forceinline__ int sink_grid_exp_variance(int e) { return kGridBiasVariance - 2 * e; }
__device__ __forceinline__ int sink_grid_exp_mass(int) { return kGridBiasMass; }
__device__ __forceinline__ int sink_grid_exp_anchor(int) { return kGridBiasMass; }

// ------------------------------------------------------------------ elements

// Narrow to the reader element type and widen again (one rounding).
__device__ __forceinline__ float elem_round(float v, Elem e) {
  if (e == Elem::kBf16) return __bfloat162float(__float2bfloat16_rn(v));
  return __half2float(__float2half_rn(v));
}

// Two-component split (:1912-1937): hi = elem(v), lo = elem(v - hi).
__device__ __forceinline__ void elem_split(float v, Elem e, float& hi, float& lo) {
  hi = elem_round(v, e);
  lo = elem_round(v - hi, e);
}

// ----------------------------------------------------------------- admission

// Full-causal right-aligned admission for query row i, key j.  Segment planes
// are compared per element (repeated and non-monotone ids are legal) and only
// when both are present.
__device__ __forceinline__ bool admitted(int64_t i, int64_t j, int64_t offset, bool strict,
                                         const int64_t* q_seg, const int64_t* k_seg,
                                         int64_t b, int64_t Nq, int64_t Nk) {
  const int64_t bound = i + offset - (strict ? 1 : 0);
  if (j < 0 || j > bound) return false;
  if (q_seg == nullptr || k_seg == nullptr) return true;
  return k_seg[b * Nk + j] == q_seg[b * Nq + i];
}

__device__ __forceinline__ bool admitted(int64_t i, int64_t j, const Params& p, int64_t b) {
  return admitted(i, j, p.offset, p.strict, p.q_seg, p.k_seg, b, p.Nq, p.Nk);
}

// A row with no admitted key has meta[3] == -1 (or a non-finite logz).  Gate on
// this BEFORE any exp; such a row contributes exactly zero.
__device__ __forceinline__ bool row_is_live(float anchor_index, float logz_natural) {
  return anchor_index >= 0.0f && isfinite(logz_natural);
}

// --------------------------------------------------------------- P and dS

// sqrt(P) in the doubled-exponent form (:1846-1857), level >= 2 only.
__device__ __forceinline__ float p_half(float s, float row_max, float lz2, float scale_log2) {
  const float shifted = __fmul_rn(__fsub_rn(s, row_max), scale_log2);
  return exp2f(0.5f * (shifted - lz2));
}

// Level < 2 form, kept for completeness; the reduced ABI is levels 2-4.
__device__ __forceinline__ float p_plain(float s, float lz2, float scale_log2) {
  return exp2f(__fsub_rn(__fmul_rn(s, scale_log2), lz2));
}

// dS = P_half * (P_half * ((dP*2^e - a) - delta))  (:1900-1902).
__device__ __forceinline__ float ds_from(float p_half_v, float dP, float two_e,
                                         float anchor_dP, float delta) {
  const float inner = __fsub_rn(__fsub_rn(__fmul_rn(dP, two_e), anchor_dP), delta);
  return __fmul_rn(p_half_v, __fmul_rn(p_half_v, inner));
}

// Level < 2 form, kept for completeness.
__device__ __forceinline__ float ds_from_plain(float p, float dP, float delta) {
  return __fmul_rn(p, __fsub_rn(dP, delta));
}

// dV operand P*2^e = (P_half * 2^(e/2))^2  (:1901-1903).
__device__ __forceinline__ float dv_operand(float p_half_v, float two_e) {
  const float root = __fmul_rn(p_half_v, half_scale(two_e));
  return __fmul_rn(root, root);
}

// ------------------------------------------------------ row sinks (statistics)

// Fixed-point conversion of one tile partial (:1982, :1987-1989).
__device__ __forceinline__ int64_t sink_grid(double partial, int grid_exp) {
  return static_cast<int64_t>(__double2ll_rn(scalbn(partial, grid_exp)));
}

// Per-row sink accumulator: init -> per-key add -> flush_tile at each key tile
// boundary -> store once per row.  This struct is the one place the sink policy
// lives; the shared surface is init / add / flush_tile / store.
//
// Main mode keeps one FP64 running total per row in the prescaled unit and
// de-scales it exactly once, in store (a per-tile de-scale cannot be right with
// one running total: a row crossing n boundaries would be divided by 2^e n
// times).  Ordered mode rounds each key-tile partial once onto the fixed grid
// (:1982-2005) and keeps only int64 totals, which is the same value the pin
// reaches by folding.  A row is owned by exactly one CTA, so no atomic is used
// and the result is bit-identical for any CTA order.
//
// Which planes exist (:1944-2005): rho at level >= 3, variance at level 3 only,
// mass and anchor probability at level 4 only.  A plane a level does not define
// is left exactly as the caller passed it (store returns before touching it);
// a plane a level does define is written for every row, exact zeros included
// (an empty row stores 0).
// Status bits, the same four measured in statistics_sink.cuh: a
// value that cannot be represented on its grid is flagged instead of being
// rounded into a plausible-looking number.  Opt-in: a null pointer costs one
// branch and no store.
enum : int {
  kSinkRhoOverflow = 1,
  kSinkDs2Overflow = 2,
  kSinkMassOverflow = 4,
  kSinkNonFinite = 8,
};

// Distance from the int64 clamp that a rounded partial may not reach.
__device__ __forceinline__ bool sink_overflows(double scaled) {
  return !(scaled > -9.0e18 && scaled < 9.0e18);
}

struct SinkRow {
  // Running sums: the current key-tile partial, folded in place at each tile
  // boundary (Main mode) or zeroed after being rounded onto the grid (ordered).
  double rho, mass, variance, anchor;
  int64_t grid_rho, grid_mass, grid_variance, grid_anchor;
  float inv_scale;
  int level, exponent;
  bool two_component, ordered;
  int* status;  // optional: OR of the kSink* bits, null when the caller has none

  __device__ __forceinline__ void flag(int bits) {
    if (status != nullptr) atomicOr(status, bits);
  }

  __device__ __forceinline__ void init(int level_, float two_e, bool two_component_, bool ordered_,
                                      int exponent_, int* status_ = nullptr) {
    status = status_;
    level = level_;
    two_component = two_component_;
    ordered = ordered_;
    inv_scale = inv_two_e(two_e);
    exponent = exponent_;
    rho = 0.0;
    mass = 0.0;
    variance = 0.0;
    anchor = 0.0;
    grid_rho = 0;
    grid_mass = 0;
    grid_variance = 0;
    grid_anchor = 0;
  }

  // One admitted key of this row.  ds_lo is zero below level 4; p_lo is zero
  // when the caller does not carry the dV operand in two components.
  __device__ __forceinline__ void add(float ds_hi, float ds_lo, float p_hi, float p_lo,
                                     bool is_anchor) {
    if (status != nullptr) {
      const bool ok = isfinite(ds_hi) && isfinite(ds_lo) && isfinite(p_hi) && isfinite(p_lo);
      if (!ok) flag(kSinkNonFinite);
    }
    if (level >= 3) {
      const float h = ds_hi + ds_lo;  // exact in FP32 (:1983-1985)
      rho += static_cast<double>(h);
      if (level == 3) variance += static_cast<double>(h) * static_cast<double>(h);
    }
    if (level == 4) {
      const float pv = p_hi + (two_component ? p_lo : 0.0f);
      mass += static_cast<double>(pv);
      if (is_anchor) anchor += static_cast<double>(pv);
    }
  }

  // End of one key tile: de-scale this tile's P mass and fold the partials in.
  __device__ __forceinline__ void flush_tile() {
    if (ordered) {
      const double descaled_mass = mass * static_cast<double>(inv_scale);
      const double descaled_anchor = anchor * static_cast<double>(inv_scale);
      // A partial whose scaled magnitude reaches the int64 clamp would be
      // rounded into a plausible number and then wrap in the int64 sum, with
      // no NaN and no exception.  Name it instead.
      if (sink_overflows(scalbn(rho, sink_grid_exp_rho(exponent)))) flag(kSinkRhoOverflow);
      if (level == 3 && sink_overflows(scalbn(variance, sink_grid_exp_variance(exponent))))
        flag(kSinkDs2Overflow);
      if (level == 4) {
        if (sink_overflows(scalbn(descaled_mass, sink_grid_exp_mass(exponent))))
          flag(kSinkMassOverflow);
        if (sink_overflows(scalbn(descaled_anchor, sink_grid_exp_anchor(exponent))))
          flag(kSinkMassOverflow);
      }
      grid_rho += sink_grid(rho, sink_grid_exp_rho(exponent));
      if (level == 3) {
        grid_variance += sink_grid(variance, sink_grid_exp_variance(exponent));
      }
      if (level == 4) {
        grid_mass += sink_grid(descaled_mass, sink_grid_exp_mass(exponent));
        grid_anchor += sink_grid(descaled_anchor, sink_grid_exp_anchor(exponent));
      }
      rho = 0.0;
      mass = 0.0;
      variance = 0.0;
      anchor = 0.0;
    } else {
      // Main mode keeps one running total for the whole row, so it cannot
      // de-scale here: a row that crosses n tile boundaries would be divided
      // by 2^e n times (mass = (P*2^e)/2^(n e)).  The total stays in the
      // prescaled unit and store() de-scales it exactly once, which is the
      // same value the pin reaches by folding (:1993-1998).
    }
  }

  // Write this row's planes once.  The pinned sink block is gated on
  // `PrecisionLevel >= 3` and stores plane 0 always, plane 2 only at level 3,
  // planes 1 and 3 only at level 4 (mainloop_bwd_sm90_tma_gmma_ws.hpp:1944,
  // :2021-2034).  No branch there stores a zero, so a level-2 call writes none
  // of the four planes, and a plane a level does not produce keeps whatever the
  // caller left in it.  That makes this buffer identical to the pinned route
  // for any pre-fill, not only for a zeroed one.
  __device__ __forceinline__ void store(double* stats, int64_t row, int64_t count) const {
    if (level < 3) return;
    if (ordered) {
      int64_t* fixed = reinterpret_cast<int64_t*>(stats);
      fixed[kPlaneRho * count + row] = grid_rho;
      if (level == 3) fixed[kPlaneVariance * count + row] = grid_variance;
      if (level == 4) {
        fixed[kPlaneMass * count + row] = grid_mass;
        fixed[kPlaneAnchor * count + row] = grid_anchor;
      }
    } else {
      // One de-scale per row, never per tile (see flush_tile).
      stats[kPlaneRho * count + row] = rho;
      if (level == 3) stats[kPlaneVariance * count + row] = variance;
      if (level == 4) {
        stats[kPlaneMass * count + row] = mass * static_cast<double>(inv_scale);
        stats[kPlaneAnchor * count + row] = anchor * static_cast<double>(inv_scale);
      }
    }
  }
};

}  // namespace bwdmath
}  // namespace native_adv
}  // namespace ops
}  // namespace xattn
