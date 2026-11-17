// Native SM103 advanced (precision level 2-4) reader backward: reduced ABI.
//
// Boundary between the public precision entries
//   _attention_precision_main_backward / _attention_precision_ordered_backward
// and a native Blackwell device leaf.  The host half (adv_bwd_host.cc) performs
// the public validation, forms this POD and calls Launch(); the leaf must not be
// reachable with an unchecked pointer.
//
// Contract of Launch()
//   - reads q, k, v, dY and every workspace plane; never writes them;
//   - writes dq, dk, dv and, at level >= 3, the statistics planes applicable at
//     that level (rho always, variance at level 3, mass and anchor probability
//     at level 4; the caller pre-zeroes the buffer, so a plane the level does
//     not use may be left untouched): every output element that is written is
//     written exactly once, by exactly one CTA;
//   - allocates nothing and keeps no static or global workspace;
//   - may enqueue more than one kernel (no single schedule is both row-owned for
//     dQ and key-owned for dK/dV).  When ordered == true no kernel in the
//     sequence may use an atomic, and each output's reduction order must be
//     fixed (row-owned dQ over ascending key blocks, key-owned dK/dV over
//     ascending query blocks and ascending query heads).  That is what makes
//     ordered mode repeat bit-exactly; Main mode may use the same schedules and
//     be deterministic as well.
//
// Units and conventions (all inherited from the public workspace, not invented
// here):
//   delta[b,h,i]   P-weighted anchored row dPsum, as the row-statistics producer
//                  writes it: sum_j P_ij (dP_ij - a_i) over the admitted set,
//                  with a_i = meta[1][i] (FP32).  It is not the unweighted
//                  sum_j dP_ij; the zero-sum identity of the sinks depends on
//                  this form.
//   logz[b,h,i]    NATURAL log of the row normalizer z of the shifted scores:
//                  z = sum_j exp(scale*s_ij - scale*m_i), so the public LSE is
//                  scale*m_i + logz_i.  The kernel needs log2(z), i.e.
//                  logz_i * log2(e); that multiply happens in the leaf, in
//                  float, at the same place the pinned kernel converts it.
//   meta planes    meta[k] at offset k*B*Hq*Nq, each plane [B,Hq,Nq]:
//                  0 = m_i (raw-score row max), 1 = a_i (anchored dP),
//                  2 = 2^{e_i} (dY row scale, exactly 1.0 below level 4),
//                  3 = anchor column index (earliest argmax over the admitted
//                      scores; -1 iff the admitted set is empty),
//                  4,5 = the min and max admitted key index of the row as the
//                        row-statistics producer records them (sentinel Nk for
//                        the min and -1 for the max on an empty row).  They are
//                        a conservative bound of A_i, never the exact predicate.
//   stats          [4,B,Hq,Nq] FP64 in Main mode.  In ordered mode the same
//                  buffer holds the int64 fixed-point sinks (identical byte
//                  count, 4*count*8) and the caller decodes it in place
//                  afterwards with rho_exponent, so the int64 sinks and the FP64
//                  planes are never live at the same time: one pointer serves
//                  both.  Plane order and grids are given below.
//   All workspace planes are [B,Hq,Nq] contiguous apart from the plane offset,
//   i.e. row stride 1 and head stride Nq (rowstat_head_stride).
//
// Scale: the entry requires a positive finite natural scale, exactly as the
// public precision entries do.  SoftDelta's zero/negative scale is normalized
// before this entry is reached (Q negated exactly, or an explicit zero fill, with
// the sign restored on dQ only); this POD carries the normalized scale and does
// not re-derive it.
//
// Algebra the leaf implements, for the admitted set
//   A_i = { j : 0 <= j < Nk, j <= i + offset - (strict ? 1 : 0),
//               and (no segment planes or k_seg[b,j] == q_seg[b,i]) }.
//   offset = Nt - Nq is the right-alignment shift of query row i, Nt = Nk (K
//   arrives with previous K/V already concatenated).  A_i is evaluated per key;
//   segment ids may be repeated and non-monotone, so no [lo,hi] interval is a
//   faithful substitute when segment planes are present.
//
//   1. P, square-root form (levels >= 2), from saved state only:
//        lz2_i  = logz_i * log2(e)                      (float)
//        P_half = exp2f(0.5f * ((s_ij - m_i) * scale_log2 - lz2_i))   = sqrt(P_ij)
//      so P_ij = P_half^2 and sum_j P_ij = 1 over A_i.
//   2. dP and dS: dP_ij = sum_c dY_ic V_jc; with da_i = meta[1][i],
//        dS_ij = P_half * (P_half * ((dP_ij * 2^{e_i} - da_i) - delta_i))
//   3. dV operand: the P operand of the dV contraction is rescaled,
//        P_op = (P_half * 2^{e_i/2})^2 = P_ij * 2^{e_i},
//      and dY was prescaled by 2^{-e_i} upstream, so dV = sum_i P_op dY_i is in
//      unscaled dY units.
//   4. Levels: at level 4 dS is carried as two element-dtype components
//      (hi = elem(dS), lo = elem(dS - hi)) for dQ and dK; when
//      two_component_dv is set the dV operand is carried as two components
//      (hi = elem(P_op), lo = elem(P_op - hi)).  two_component_dv is
//      level == 4, or level 2/3 with Nk > Nq on this total key axis (the same
//      condition the pinned dispatcher applies to p.seqlen_k > p.seqlen_q).
//   5. Contractions (scale is the natural softmax scale):
//        dQ_ic = scale * sum_j dS_ij K_jc
//        dK_jc = scale * sum_i dS_ij Q_ic        (summed over both readers by
//                                                 the caller's composition, not
//                                                 here: this entry is one reader)
//        dV_jc = sum_i (P_op)_ij dY_ic
//      The scale multiplies dQ and dK once, after the reduction, in FP32; it is
//      not folded into dS.  Sum over both element-dtype components when (4)
//      applies.
//   6. Row sinks, level >= 3, written to the four statistics planes:
//        rho_i        = sum_j dS_ij                    (both components)
//        variance_i   = sum_j dS_ij^2                  (level 3)
//        mass_i       = 2^{-e_i} sum_j (P_op hi + lo)  (level 4)
//        anchorprob_i = 2^{-e_i} sum_{j in A_i, j == meta[3][i]} (P_op hi + lo)
//      Plane order: rho at +0*count, mass at +1*count, variance at +2*count,
//      anchorprob at +3*count, count = B*Hq*Nq.  De-scale each tile partial
//      before it is summed.  Ordered mode rounds each tile partial once onto
//      the fixed-point grids rho 2^{62-e}, variance 2^{68-2e}, mass and
//      anchorprob 2^{60}, and accumulates int64 (wrapping is impossible inside
//      the documented bound); the caller's in-place decode multiplies by
//      2^{e-62}, 2^{2e-68} and 2^{-60} respectively.
//   7. Empty rows: meta[3][i] == -1 (or a non-finite logz) means A_i is empty.
//      Such a row is decided before any exp, contributes exactly zero to every
//      output, writes exactly 0 into rho/mass/variance and no anchor
//      contribution, and must never produce a NaN.
//
// The line-for-line source of items 1-6 is the pinned Hopper backward
// (csrc/attention/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp: P at 1846-1857,
// dS at 1900-1922, dV operand at 1913-1930, two components at 1936-1942 and
// 2098-2109, sinks at 1944-2050, dK postfactor at 2293 and 2351, dQ postfactor
// at csrc/attention/hopper/bwd_postprocess_kernel.h:270 and :274).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace xattn {
namespace ops {
namespace native_adv {

enum class Elem { kFp16, kBf16 };

struct Params {
  // operands (read-only)
  const void* q;
  const void* k;
  const void* v;
  const void* dy;
  // workspaces (read-only; the sinks are the exception, see stats below)
  const float* delta;
  const float* logz;          // natural log of the row normalizer
  const float* meta;          // 6 planes of [B,Hq,Nq]
  // sinks: [4,B,Hq,Nq] FP64 in Main mode, int64 fixed point in ordered mode
  // (same buffer, never live at once; the caller decodes in place).
  double* stats;
  const int32_t* rho_exponent;  // per-row e_i grid exponent, ordered levels 3/4
  // Optional sink status word; diagnostic flags do not change the reductions.
  // Null means "no channel".  When present the leaf ORs into it, atomically:
  // 1 rho grid overflow, 2 dS^2 grid overflow, 4 mass/anchor grid overflow,
  // 8 a non-finite partial reached the sink.  Nothing is stored silently:
  // a value that cannot reach its grid is flagged instead of being rounded
  // into a plausible number (a wrong grid exponent saturates and then wraps
  // in the int64 sum with no NaN and no exception).
  int* sink_status;
  // outputs (write-only, exact width)
  void* dq;
  void* dk;
  void* dv;
  // optional segment planes, [B,Nq] and [B,Nk], int64
  const int64_t* q_seg;
  const int64_t* k_seg;

  int64_t q_batch_stride, q_row_stride, q_head_stride;
  int64_t k_batch_stride, k_row_stride, k_head_stride;
  int64_t v_batch_stride, v_row_stride, v_head_stride;
  int64_t dy_batch_stride, dy_row_stride, dy_head_stride;
  int64_t dq_batch_stride, dq_row_stride, dq_head_stride;
  int64_t dk_batch_stride, dk_row_stride, dk_head_stride;
  int64_t dv_batch_stride, dv_row_stride, dv_head_stride;
  // delta/logz/meta/stats planes are [B,Hq,Nq]: row stride 1, head stride Nq.
  int64_t rowstat_head_stride;

  int64_t B, Nq, Nk, Nt, Hq, Hkv, D, Dv;
  int64_t offset;    // Nt - Nq, the right-alignment shift
  int64_t qk_ratio;  // Hq / Hkv

  // 0: full causal; 2: sliding chunk. At visibility 2 the admitted set of row
  // i is { j : meta[4][i] <= j <= meta[5][i] } (the first and last admitted key
  // the row-statistics producer recorded from the exact chunk/reset bounds),
  // intersected with the segment test; no causal bound is re-derived.
  int visibility;
  int64_t span;    // chunk size at visibility 2
  bool reset;      // chunk positions restart per segment (visibility 2)
  int level;       // 2, 3 or 4
  double softmax_scale;      // natural scale, exactly as validated by the host
  float softmax_scale_log2;  // softmax_scale * log2(e), used inside the exp2
  bool strict;               // reader 2: j <= i + offset - 1
  bool two_component_dv;     // level 4, or level 2/3 with Nt > Nq
  bool ordered;              // fixed-point sinks, no atomics, fixed order
  Elem elem;                 // operand element type (FP16 or BF16)
};

void Launch(const Params& p, cudaStream_t stream);

}  // namespace native_adv
}  // namespace ops
}  // namespace xattn
