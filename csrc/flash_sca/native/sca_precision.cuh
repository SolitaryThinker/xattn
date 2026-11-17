// SCA precision algebra, levels 0-4. Header only: device helpers, no kernels,
// no allocation, no state.
//
// Purpose: the families that run SCA at levels 2-4 (the native advanced
// backward leaves today, the level-aware SCA kernels next) must apply exactly
// one copy of the documented level algebra, and one copy of the reader's chunk
// walk and admission.  The algebra already lives in
// attention/precision/native/adv_bwd_math.cuh (derived from the pinned Hopper
// mainloop, hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:1846-2050).  This header
// does not restate or re-derive it: it re-exports it under one name, adds the
// level plan and the walk, and fixes the units.  A numerical change belongs in
// adv_bwd_math.cuh, never here.
//
// Units (invariant, see adv_bwd_dq.cu:14-16):
//   * dY reaches the level-4 contractions already prescaled by 2^-e, so the dP
//     they compute is in 2^-e units and DsFrom's `two_e` restores it.
//   * meta plane 1 (anchored dP) and delta are in unscaled dY units.
//   * Below level 4 the 2^e/2^-e planes are exactly 1.
//
// ABI: this header adds no flag that could disagree with the level.  The plan
// is a function of Params (level plus the one field that is not a function of
// the level, two_component_dv), and the sink's grid is taken from the same
// planes and exponent the leaves read.
#pragma once

#include <cstdint>

#include "attention/precision/native/adv_bwd_math.cuh"
#include "attention/precision/native/adv_bwd_native.h"

namespace xattn::ops::sca_prec {

namespace bm = ::xattn::ops::native_adv::bwdmath;
using Elem = ::xattn::ops::native_adv::Elem;
using Params = ::xattn::ops::native_adv::Params;

// ------------------------------------------------------------- level plan

// The backward element algebra of a level, from docs/sliding_chunk_attention.md:
//   0      plain softmax gradient, single-component P and dS
//   1      + subtract-before-scale softmax, FP32 backward state
//   2      + sqrt(P) and anchored dS (the stable softmax gradient), centering
//   3      + dS row-sum repair (rho)
//   4      + dual element P/dS, dO range protection, probability-mass repair
// This is the *element* part of the level, not the whole level definition: the
// K/V centering and the level-4 dO range protection happen in the caller's
// preprocessing (preprocess.cu:51, :81) and are already baked into the operands
// and planes these leaves read, and `two_component_dv` is not a function of the
// level alone (adv_bwd_host.cc:151 sets it for level 4 or Nk > Nq).  Hence
// PlanOf, which reads it from Params.
struct LevelPlan {
  int level;
  bool sbs;            // level >= 1: the row max is taken on raw scores
  bool sqrt_p;         // level >= 2: p_half form, anchored dS
  bool rowsum_repair;  // level >= 3: rho accumulates sum(dS)
  bool dual_ds;        // level == 4: dS carried as hi + lo
  bool dual_dv;        // Params::two_component_dv: the dV operand is carried as hi + lo
};

__host__ __device__ __forceinline__ LevelPlan PlanForLevel(int level) {
  const bool sbs = level >= 1;
  const bool sqrt_p = level >= 2;
  const bool rowsum = level >= 3;
  const bool dual = level == 4;
  return LevelPlan{level, sbs, sqrt_p, rowsum, dual, dual};
}

__host__ __device__ __forceinline__ LevelPlan PlanOf(const Params& p) {
  LevelPlan plan = PlanForLevel(static_cast<int>(p.level));
  plan.dual_dv = p.two_component_dv != 0;
  return plan;
}

// ------------------------------------------------------- element algebra
// Every function is the adv_bwd_math counterpart, named for this family.

__device__ __forceinline__ float LogzToLz2(float logz_natural) {
  return bm::lz2_from_logz(logz_natural);
}

__device__ __forceinline__ float TwoE(float meta_scale, int level) {
  return bm::two_e_of(meta_scale, level);
}

__device__ __forceinline__ bool RowLive(float anchor_index, float logz_natural) {
  return bm::row_is_live(anchor_index, logz_natural);
}

// sqrt(P).  Callers gate on RowLive first; a dead row never reaches here.
__device__ __forceinline__ float PHalf(float s, float row_max, float lz2, float scale_log2) {
  return bm::p_half(s, row_max, lz2, scale_log2);
}

// dS = P_half * (P_half * ((dP * 2^e - a) - delta)); the level >= 2 form.
__device__ __forceinline__ float DsFrom(float p_half_v, float dP, float two_e, float anchor_dP,
                                        float delta) {
  return bm::ds_from(p_half_v, dP, two_e, anchor_dP, delta);
}

// Level < 2 form, kept so level 0/1 paths share the call shape.
__device__ __forceinline__ float DsPlain(float p, float dP, float delta) {
  return bm::ds_from_plain(p, dP, delta);
}

// dV operand P * 2^e = (P_half * 2^(e/2))^2, so dV comes out in unscaled units.
__device__ __forceinline__ float DvOperand(float p_half_v, float two_e) {
  return bm::dv_operand(p_half_v, two_e);
}

__device__ __forceinline__ float ElemRound(float v, Elem e) {
  return bm::elem_round(v, e);
}

__device__ __forceinline__ void Split(float v, Elem e, float& hi, float& lo) {
  bm::elem_split(v, e, hi, lo);
}

// --------------------------------------------------------------- row sinks

// One sink per (row, pass), fed exactly the operand the contraction consumes:
// zero_sum_repair defines rho as the sum of the dS the contraction used
// (zero_sum_repair.h:6), so below level 4 both take the element head and at
// level 4 both take hi + lo.
using Sink = bm::SinkRow;

// The sink's 2^e and fixed-point grid exponent, taken from the same planes and
// exponent the leaves read (formerly adv_bwd_dq.cu:77-82; the leaf now calls
// this), so no caller can pick another grid.  `count` is the plane stride B*Hq*rowstat_head_stride.
__device__ __forceinline__ void SinkInit(Sink& sink, const Params& p, int64_t row,
                                         int64_t count) {
  const float two_e =
      bm::two_e_of(p.meta[2 * count + row], static_cast<int>(p.level));
  sink.init(static_cast<int>(p.level), two_e, p.two_component_dv, p.ordered,
            bm::grid_exponent(p.rho_exponent, row, two_e), p.sink_status);
}

__device__ __forceinline__ void SinkStore(const Sink& sink, double* stats, int64_t row,
                                          int64_t count) {
  sink.store(stats, row, count);
}

// --------------------------------------------------------------- the walk

// Whether the row admits any key and, if it does, the interval it walks.  Dead
// rows (no visible key, or a non-finite logz) contribute exactly zero.
struct ChunkWalk {
  int64_t jfirst;   // first admitted key
  int64_t jlast;    // last admitted key
  int64_t jstart;   // physical kSinkTile boundary at or below jfirst
  bool live;
};

// Exactly the walk the dQ leaf performed on its own (formerly
// adv_bwd_dq.cu:114-124; it now calls this): sliding chunk
// takes [meta[4], meta[5]] from the row statistics, full causal takes the whole
// strict prefix, the last key is clamped to Nk-1, and the start snaps down to
// the physical sink tile so the ordered tile partials are the same as a walk
// from key 0.
__device__ __forceinline__ ChunkWalk WalkOf(const Params& p, int64_t i, int64_t row,
                                            int64_t count) {
  const float anchor_col = p.meta[3 * count + row];  // -1 iff the admitted set is empty
  int64_t jfirst = 0;
  int64_t jlast = i + p.offset - (p.strict ? 1 : 0);
  if (p.visibility != 0) {
    jfirst = static_cast<int64_t>(p.meta[4 * count + row]);
    jlast = static_cast<int64_t>(p.meta[5 * count + row]);
  }
  if (jlast > p.Nk - 1) jlast = p.Nk - 1;
  const int64_t jstart = jfirst / bm::kSinkTile * bm::kSinkTile;
  const bool live =
      jlast >= 0 && jfirst <= jlast && bm::row_is_live(anchor_col, p.logz[row]);
  return ChunkWalk{jfirst, jlast, jstart, live};
}

// Per-element segment test.  One definition for the family: both planes absent
// means one segment, and a lone plane is not a legal Params (host.cc and
// reader_native.cu reject it), so it is treated as no segments rather than
// dereferenced.  Ids may repeat and need not be sorted; they are compared per
// element.
__device__ __forceinline__ bool SegmentOk(const Params& p, int64_t b, int64_t i,
                                          int64_t j) {
  if (p.q_seg == nullptr || p.k_seg == nullptr) return true;
  return p.k_seg[b * p.Nk + j] == p.q_seg[b * p.Nq + i];
}

}  // namespace xattn::ops::sca_prec
