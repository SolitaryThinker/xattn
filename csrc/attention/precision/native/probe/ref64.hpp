// Independent FP64 reference for the native advanced (level 2-4) reader
// backward contract, written from the public contract and the pinned Hopper
// algebra only.  No torch, no CUDA, no leaf source is included: this is the
// baseline side of the comparison, so it must not share code with the kernel.
//
// What it models, for one reader (the caller's composition supplies the other):
//   A_i    = { j : 0 <= j < Nk, j <= i + offset - (strict ? 1 : 0),
//                  and (no segments or k_seg[b,j] == q_seg[b,i]) }
//   s_ij   = sum_c q_ic k_jc                    (raw score, before scale)
//   m_i    = max_{j in A_i} s_ij                (meta[0])
//   z_i    = sum_{j in A_i} exp(scale*(s_ij - m_i))
//   logz_i = log z_i                            (natural log, public LSE = scale*m_i + logz_i)
//   anchor = smallest argmax over A_i          (meta[3], -1 iff A_i empty)
//   a_i    = dP at the anchor                   (meta[1])
//   delta_i= sum_j P_ij (dP_ij - a_i)
//   P_ij   = exp(scale*(s_ij - m_i)) / z_i
//   dS_ij  = P_ij * (dP_ij - a_i - delta_i)      == P * ((dP*2^e - a_i) - delta_i) * 2^-e * 2^e
//   dQ_ic  = scale * sum_j dS_ij K_jc
//   dK_jc  = scale * sum_i dS_ij Q_ic
//   dV_jc  = sum_i P_ij dY_ic                   (the 2^e dY prescale and the P*2^e
//                                               operand cancel exactly, so the
//                                               reference uses unscaled dY and P)
//   rho_i  = sum_j dS_ij, variance_i = sum_j dS_ij^2,
//   mass_i = sum_j P_ij (times 2^-e in the pin, i.e. 1 for a live row),
//   anchorprob_i = P at the anchor.
// Level 4's two-component operand carries the same real numbers through two
// element-dtype pieces, so an exact double computation is the limit it aims at.
//
// Everything is computed in double from operands that were first rounded to the
// case's element type, so the only difference from the kernel is arithmetic
// precision and its accumulation order.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <cstdio>

namespace ref64 {

// ---- element-type rounding (the kernel reads quantized operands) -------------
inline float round_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // nan/inf pass through
  uint32_t r = u + 0x7fffu + ((u >> 16) & 1u);     // round to nearest even
  r &= 0xffff0000u;
  float y;
  std::memcpy(&y, &r, 4);
  return y;
}

inline float round_fp16(float x) {
  const float inf = std::numeric_limits<float>::infinity();
  if (std::isnan(x) || std::isinf(x)) return x;
  float a = std::fabs(x);
  if (a == 0.0f) return x;
  if (a > 65504.0f) return x < 0 ? -inf : inf;
  if (a < 6.103515625e-05f) {  // subnormal range: quantize on the 2^-24 grid
    float q = std::nearbyint(x * 16777216.0f) / 16777216.0f;
    return q;
  }
  uint32_t u;
  std::memcpy(&u, &a, 4);
  uint32_t r = u + 0x0fffu + ((u >> 13) & 1u);
  r &= 0xffffe000u;
  float y;
  std::memcpy(&y, &r, 4);
  return x < 0 ? -y : y;
}

inline float round_elem(float x, int elem) { return elem == 0 ? round_fp16(x) : round_bf16(x); }

// ---- case specification ------------------------------------------------------
struct Spec {
  int B = 1, Nq = 8, Nk = 8, Hq = 2, Hkv = 1, D = 8, Dv = 8;
  int level = 2;          // 2, 3 or 4
  double scale = 0.0;     // natural scale; 0 means 1/sqrt(D)
  bool strict = false;    // reader 2
  bool segments = false;  // apply per-key segment equality
  bool anchor_scale = false;  // level 4: give some rows a non-unit 2^e
  int elem = 0;           // 0 = fp16, 1 = bf16
  uint64_t seed = 1;
  int nseg = 2;
  // Adversarial segment pattern: when > 0, every `hole_period`-th key belongs to
  // another segment, so each physical 32-key tile ends on a key this query does
  // NOT admit while keys on both sides of it ARE admitted.  That is the only
  // shape in which the pin's fold schedule (round every physical tile) and the
  // pre-fix leaf's (round once per run of tiles, because a mismatching key
  // `continue`d above the flush) produce different int64 sums.  Legal in
  // contract: per-key segment ids are arbitrary, nothing requires contiguity.
  int hole_period = 0;
};

struct Case {
  Spec s;
  int64_t Nt = 0, offset = 0, count = 0;
  double scale = 0.0;
  std::vector<double> q, k, v, dy;      // quantized operands, doubles of the rounded values
  std::vector<int64_t> qseg, kseg;      // [B*Nq], [B*Nt]; empty when unused
  // saved state (the row-statistics output, computed here in double)
  std::vector<double> delta, logz, meta;  // [count], [count], [6*count]
  std::vector<double> dq, dk, dv;         // [B*Nq*Hq*D], [B*Nt*Hkv*D], [B*Nt*Hkv*Dv]
  // Same contraction with the dS operand rounded to the reader element type, which
  // is what an element-typed MMA multiplies (the pinned dQ GEMM operand).
  std::vector<double> dq_pin;             // [B*Nq*Hq*D]
  std::vector<double> stats;
  // Same rows, but the operand the pin would use if it were not element typed:
  // exact FP32 dS and exact P.  The gap between this and `stats` is the cast
  // floor, reported rather than hidden by a matching reference.
  std::vector<double> stats_exact;              // [4*count]
  // Ordered fixed-point emulation of the sink: the int64 planes folded two
  // ways.  grid_phys folds at every PHYSICAL 32-key tile boundary (the pin's
  // fold, mainloop_bwd_sm90_tma_gmma_ws.hpp:1982-2005); grid_mask folds only
  // when the boundary key was admitted, which is what a segment-mismatch
  // `continue` placed above the flush used to do.  grid_mask_rows counts the
  // rows where the two schedules disagree at all.
  std::vector<int32_t> grid_e;  // the caller's per-row exponent, [count]
  std::vector<int64_t> grid_phys, grid_mask;
  int64_t grid_mask_rows = 0;
  double grid_mask_max_units = 0;
  int64_t dbg_pf = 0, dbg_mf = 0;   // folds per schedule, for probe self-checks
};

// deterministic xorshift so a case is reproducible from its seed alone
static inline uint64_t next_u64(uint64_t& x) {
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return x;
}
static inline double next_uniform(uint64_t& x) {
  return (double)(next_u64(x) >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
}
static inline double next_normal(uint64_t& x) {
  double u1 = next_uniform(x), u2 = next_uniform(x);
  if (u1 <= 1e-300) u1 = 1e-300;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

inline int64_t hq_of(int64_t hkv, const Case& c) { return hkv * (c.s.Hq / c.s.Hkv); }

inline bool admitted(const Case& c, int64_t i, int64_t j) {
  if (j < 0 || j >= c.Nt) return false;
  const int64_t lim = i + c.offset - (c.s.strict ? 1 : 0);
  if (j > lim) return false;
  if (!c.s.segments) return true;
  const int64_t b = 0;  // per-batch below
  (void)b;
  return true;
}

inline Case build(const Spec& s) {
  Case c;
  c.s = s;
  c.Nt = s.Nk;                 // K arrives with previous K/V already concatenated
  c.offset = c.Nt - s.Nq;
  c.count = (int64_t)s.B * s.Hq * s.Nq;
  c.scale = s.scale != 0.0 ? s.scale : 1.0 / std::sqrt((double)s.D);
  uint64_t rng = s.seed * 6364136223846793005ull + 1442695040888963407ull;

  const int64_t nq_tot = (int64_t)s.B * s.Nq * s.Hq * s.D;
  const int64_t nk_tot = (int64_t)s.B * c.Nt * s.Hkv * s.D;
  const int64_t nv_tot = (int64_t)s.B * c.Nt * s.Hkv * s.Dv;
  const int64_t ndy_tot = (int64_t)s.B * s.Nq * s.Hq * s.Dv;
  c.q.resize(nq_tot);
  c.k.resize(nk_tot);
  c.v.resize(nv_tot);
  c.dy.resize(ndy_tot);
  for (auto& x : c.q) x = (double)round_elem((float)next_normal(rng), s.elem);
  for (auto& x : c.k) x = (double)round_elem((float)next_normal(rng), s.elem);
  for (auto& x : c.v) x = (double)round_elem((float)next_normal(rng), s.elem);
  for (auto& x : c.dy) x = (double)round_elem((float)next_normal(rng), s.elem);

  if (s.segments) {
    c.qseg.assign((size_t)s.B * s.Nq, 0);
    c.kseg.assign((size_t)s.B * c.Nt, 0);
    // nondecreasing runs, ids that repeat and skip: segment r ends where the
    // generator says, so both sides agree per batch
    for (int b = 0; b < s.B; ++b) {
      int64_t qs = 0, ks = 0;
      int64_t qi = 0, ki = 0;
      while (qi < s.Nq || ki < c.Nt) {
        int64_t qrun = 1 + (int64_t)(next_uniform(rng) * 3.0);
        int64_t krun = 1 + (int64_t)(next_uniform(rng) * 3.0);
        for (int64_t t = 0; t < qrun && qi < s.Nq; ++t) c.qseg[(size_t)b * s.Nq + qi++] = qs;
        for (int64_t t = 0; t < krun && ki < c.Nt; ++t) c.kseg[(size_t)b * c.Nt + ki++] = ks;
        qs += 1 + (int64_t)(next_uniform(rng) * 2.0);
        ks = qs;  // segments align on both sides where they meet
      }
    }
  }

  if (s.hole_period > 0) {
    c.qseg.assign((size_t)s.B * s.Nq, 0);
    c.kseg.assign((size_t)s.B * c.Nt, 0);
    for (int64_t j = 0; j < c.Nt; ++j)
      c.kseg[(size_t)j] = ((j % s.hole_period) == s.hole_period - 1) ? 1 : 0;
    for (int b = 0; b < s.B; ++b)
      for (int64_t j = 0; j < c.Nt; ++j) c.kseg[(size_t)b * c.Nt + j] = c.kseg[(size_t)j];
  }

  // ---- saved state, exactly as the row-statistics producer would write it ----
  c.delta.assign((size_t)c.count, 0.0);
  c.logz.assign((size_t)c.count, 0.0);
  c.meta.assign((size_t)(6 * c.count), 0.0);
  const int64_t ratio = s.Hq / s.Hkv;
  for (int b = 0; b < s.B; ++b) {
    for (int hq = 0; hq < s.Hq; ++hq) {
      const int hkv = hq / ratio;
      for (int64_t i = 0; i < s.Nq; ++i) {
        const int64_t ci = ((int64_t)b * s.Hq + hq) * s.Nq + i;
        // admitted set
        std::vector<int64_t> adm;
        for (int64_t j = 0; j < c.Nt; ++j) {
          if (j > i + c.offset - (s.strict ? 1 : 0)) break;
          if ((s.segments || s.hole_period > 0) && c.kseg[(size_t)b * c.Nt + j] != c.qseg[(size_t)b * s.Nq + i]) continue;
          adm.push_back(j);
        }
        double* m0 = &c.meta[(size_t)0 * c.count + ci];
        double* m1 = &c.meta[(size_t)1 * c.count + ci];
        double* m2 = &c.meta[(size_t)2 * c.count + ci];
        double* m3 = &c.meta[(size_t)3 * c.count + ci];
        double* m5 = &c.meta[(size_t)5 * c.count + ci];
        double* m4 = &c.meta[(size_t)4 * c.count + ci];
        if (adm.empty()) {
          *m0 = -std::numeric_limits<double>::infinity();
          *m1 = 0.0;
          *m2 = 1.0;
          *m3 = -1.0;
          *m4 = (double)c.Nt;
          *m5 = -1.0;
          c.logz[(size_t)ci] = std::numeric_limits<double>::infinity();
          continue;
        }
        // raw scores
        std::vector<double> sc(adm.size());
        for (size_t t = 0; t < adm.size(); ++t) {
          const int64_t j = adm[t];
          double acc = 0.0;
          for (int d = 0; d < s.D; ++d)
            acc += c.q[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.D + (int64_t)hq * s.D + d] *
                   c.k[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.D + (int64_t)hkv * s.D + d];
          sc[t] = acc;
        }
        double m = -std::numeric_limits<double>::infinity();
        for (double x : sc) m = std::max(m, x);
        int64_t anchor = -1;
        for (size_t t = 0; t < adm.size(); ++t)
          if (sc[t] == m) { anchor = adm[t]; break; }
        double z = 0.0;
        for (double x : sc) z += std::exp(c.scale * (x - m));
        // anchored dP
        double dpa = 0.0;
        for (int dv = 0; dv < s.Dv; ++dv)
          dpa += c.dy[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.Dv + (int64_t)hq * s.Dv + dv] *
                 c.v[(size_t)((int64_t)b * c.Nt + anchor) * s.Hkv * s.Dv + (int64_t)hkv * s.Dv + dv];
        double del = 0.0;
        for (size_t t = 0; t < adm.size(); ++t) {
          const int64_t j = adm[t];
          double dp = 0.0;
          for (int dv = 0; dv < s.Dv; ++dv)
            dp += c.dy[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.Dv + (int64_t)hq * s.Dv + dv] *
                  c.v[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.Dv + (int64_t)hkv * s.Dv + dv];
          const double p = std::exp(c.scale * (sc[t] - m)) / z;
          del += p * (dp - dpa);
        }
        *m0 = m;
        *m1 = dpa;
        *m2 = (s.level == 4 && s.anchor_scale) ? std::ldexp(1.0, (int)(ci % 4)) : 1.0;
        *m3 = (double)anchor;
        *m4 = (double)adm.front();
        *m5 = (double)adm.back();
        c.logz[(size_t)ci] = std::log(z);
        c.delta[(size_t)ci] = del;
      }
    }
  }
  return c;
}

// ---- the reference backward --------------------------------------------------
struct Errors {
  double dq_max_abs = 0, dq_max_rel = 0;
  double dk_max_abs = 0, dk_max_rel = 0;
  double dv_max_abs = 0, dv_max_rel = 0;
  double stats_max_rel = 0;
  int64_t nonfinite = 0;
  int64_t empty_rows = 0;
  double empty_row_max_abs = 0;
};

inline void backward(Case& c) {
  const Spec& s = c.s;
  const int64_t ratio = s.Hq / s.Hkv;
  c.dq.assign((size_t)((int64_t)s.B * s.Nq * s.Hq * s.D), 0.0);
  c.dq_pin.assign((size_t)((int64_t)s.B * s.Nq * s.Hq * s.D), 0.0);
  c.dk.assign((size_t)((int64_t)s.B * c.Nt * s.Hkv * s.D), 0.0);
  c.dv.assign((size_t)((int64_t)s.B * c.Nt * s.Hkv * s.Dv), 0.0);
  c.stats.assign((size_t)(4 * c.count), 0.0);
  c.stats_exact.assign((size_t)(4 * c.count), 0.0);
  c.grid_phys.assign((size_t)(4 * c.count), 0);
  c.grid_mask.assign((size_t)(4 * c.count), 0);
  c.grid_mask_rows = 0;
  c.grid_mask_max_units = 0.0;

  for (int b = 0; b < s.B; ++b) {
    for (int hq = 0; hq < s.Hq; ++hq) {
      const int hkv = hq / ratio;
      for (int64_t i = 0; i < s.Nq; ++i) {
        const int64_t ci = ((int64_t)b * s.Hq + hq) * s.Nq + i;
        const double m = c.meta[(size_t)0 * c.count + ci];
        const double a = c.meta[(size_t)1 * c.count + ci];
        const double logz = c.logz[(size_t)ci];
        const double del = c.delta[(size_t)ci];
        const double two_e_row = c.meta[(size_t)2 * c.count + ci];  // dO prescale 2^e
        const int64_t anchor = (int64_t)c.meta[(size_t)3 * c.count + ci];
        if (anchor < 0) {  // empty row: every output exactly zero
          c.stats[0 + ci] = 0.0;
          c.stats[1 * c.count + ci] = 0.0;
          c.stats[2 * c.count + ci] = 0.0;
          c.stats[3 * c.count + ci] = 0.0;
          continue;
        }
        double rho = 0.0, var = 0.0, mass = 0.0, aprob = 0.0;
        // Ordered fixed-point emulation: per-tile partials and the two fold
        // schedules.  Rounding is the kernel's: round-half-even of the partial
        // scaled onto the plane's grid (sink_grid -> __double2ll_rn).
        // The kernel takes the grid exponent from p.rho_exponent (the caller's
        // _rho_exponent) and the de-scale from meta[2]; in this probe those are
        // deliberately different quantities, so the emulation must use the
        // caller's array for the grids and two_e for the de-scale, exactly.
        const int e_row = c.grid_e.empty() ? (int)std::lrint(std::log2(two_e_row))
                                          : (int)c.grid_e[(size_t)ci];
        int64_t ph[4] = {0, 0, 0, 0}, mk[4] = {0, 0, 0, 0};
        // Two accumulator sets, because the two fold schedules do not fold the
        // same partials.  ph = the pin: a physical 32-key tile ends for the
        // accumulator whether or not its last key is admitted, so the partial is
        // rounded and reset there unconditionally.  mk = the pre-fix leaf: a
        // mismatching key `continue`d above the flush, so its tile boundary did
        // not fold and the partial ran on into the next physical tile - rounded
        // once per run of tiles instead of once per tile.
        double pr = 0.0, pv = 0.0, pm = 0.0, pa = 0.0;   // physical schedule
        double mr = 0.0, mv = 0.0, mm = 0.0, ma = 0.0;   // mask schedule (pre-fix)
        auto fold_phys = [&]() {
          ph[0] += std::llrint(std::scalbn(pr, 62 - e_row));
          if (s.level == 3) ph[1] += std::llrint(std::scalbn(pv, 68 - 2 * e_row));
          if (s.level == 4) {
            ph[2] += std::llrint(std::scalbn(pm / two_e_row, 60));
            ph[3] += std::llrint(std::scalbn(pa / two_e_row, 60));
          }
          pr = pv = pm = pa = 0.0;
          ++c.dbg_pf;
        };
        auto fold_mask = [&]() {
          mk[0] += std::llrint(std::scalbn(mr, 62 - e_row));
          if (s.level == 3) mk[1] += std::llrint(std::scalbn(mv, 68 - 2 * e_row));
          if (s.level == 4) {
            mk[2] += std::llrint(std::scalbn(mm / two_e_row, 60));
            mk[3] += std::llrint(std::scalbn(ma / two_e_row, 60));
          }
          mr = mv = mm = ma = 0.0;
          ++c.dbg_mf;
        };
        double rho_exact = 0.0, var_exact = 0.0, mass_exact = 0.0, aprob_exact = 0.0;
        for (int64_t j = 0; j < c.Nt; ++j) {
          if (j > i + c.offset - (s.strict ? 1 : 0)) break;
          const bool boundary = ((j + 1) & 31) == 0;
          if ((s.segments || s.hole_period > 0) && c.kseg[(size_t)b * c.Nt + j] != c.qseg[(size_t)b * s.Nq + i]) {
            if (boundary) fold_phys();  // the physical tile ends here regardless
            continue;
          }
          double sc = 0.0;
          for (int d = 0; d < s.D; ++d)
            sc += c.q[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.D + (int64_t)hq * s.D + d] *
                  c.k[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.D + (int64_t)hkv * s.D + d];
          const double p = std::exp(c.scale * (sc - m) - logz);
          double dp = 0.0;
          for (int dv = 0; dv < s.Dv; ++dv)
            dp += c.dy[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.Dv + (int64_t)hq * s.Dv + dv] *
                  c.v[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.Dv + (int64_t)hkv * s.Dv + dv];
          const double ds = p * (dp - a - del);
          // The pinned sink is element typed exactly as the GEMM operands are:
          // rdS is built as Element (:1936-1937), level 3 sums h = float(hirow)
          // with no lorow (:1978-1989), and lorow is added only at level 4.  The
          // P mass operand is the level-4/two-component P*2^e split the same way,
          // in the prescaled unit, and de-scaled by 2^-e when the row is stored.
          const float g_ds = (float)ds;
          const float ds_hi = round_elem(g_ds, s.elem);
          const float ds_lo = (s.level == 4) ? round_elem(g_ds - ds_hi, s.elem) : 0.0f;
          const double ds_op = (double)ds_hi + (double)ds_lo;
          const double p_op_pre = p * two_e_row;
          const float g_p = (float)p_op_pre;
          const float p_hi = round_elem(g_p, s.elem);
          const bool two_comp = (s.level == 4) || (c.Nt > s.Nq);
          const float p_lo = two_comp ? round_elem(g_p - p_hi, s.elem) : 0.0f;
          const double p_op = (double)p_hi + (double)p_lo;
          rho += ds_op;
          var += ds_op * ds_op;
          rho_exact += ds;
          var_exact += ds * ds;
          mass += p_op;
          pr += ds_op;  mr += ds_op;
          pv += ds_op * ds_op;  mv += ds_op * ds_op;
          pm += p_op;  mm += p_op;
          if (j == anchor) { pa = p_op; ma = p_op; }
          if (boundary) { fold_phys(); fold_mask(); }
          if (j == anchor) aprob = p_op;
          mass_exact += p;
          if (j == anchor) aprob_exact = p;
          // dQ (row-owned) and dK (key-owned) in double
          for (int d = 0; d < s.D; ++d) {
            c.dq[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.D + (int64_t)hq * s.D + d] +=
                c.scale * ds * c.k[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.D + (int64_t)hkv * s.D + d];
            // The contraction multiplies the element head, plus the residual
            // as a second component at level 4 (:2082, :2102).  hi + lo is the
            // two-component operand, not the original FP32 dS.
            {
              const double op = ds_op;
              c.dq_pin[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.D + (int64_t)hq * s.D + d] +=
                  c.scale * op *
                  c.k[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.D + (int64_t)hkv * s.D + d];
            }
            c.dk[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.D + (int64_t)hkv * s.D + d] +=
                c.scale * ds * c.q[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.D + (int64_t)hq * s.D + d];
          }
          for (int dv = 0; dv < s.Dv; ++dv)
            c.dv[(size_t)((int64_t)b * c.Nt + j) * s.Hkv * s.Dv + (int64_t)hkv * s.Dv + dv] +=
                p * c.dy[(size_t)((int64_t)b * s.Nq + i) * s.Hq * s.Dv + (int64_t)hq * s.Dv + dv];
        }
        fold_phys();
        fold_mask();  // the short last tile, folded into both schedules
        bool diff = false;
        for (int pl = 0; pl < 4; ++pl) {
          c.grid_phys[(size_t)pl * c.count + ci] = ph[pl];
          c.grid_mask[(size_t)pl * c.count + ci] = mk[pl];
          const int64_t d = ph[pl] - mk[pl];
          if (d != 0) diff = true;
          c.grid_mask_max_units = std::max(c.grid_mask_max_units, (double)std::abs((long long)d));
        }
        if (diff) ++c.grid_mask_rows;
        // Plane gating is the pin's: rho at level >= 3, variance at level 3
        // only, P mass and anchor probability at level 4 only.
        c.stats[0 + ci] = (s.level >= 3) ? rho : 0.0;
        c.stats[1 * c.count + ci] = (s.level == 4) ? mass / two_e_row : 0.0;
        c.stats[2 * c.count + ci] = (s.level == 3) ? var : 0.0;
        c.stats[3 * c.count + ci] = (s.level == 4) ? aprob / two_e_row : 0.0;
        c.stats_exact[0 + ci] = (s.level >= 3) ? rho_exact : 0.0;
        // p is unscaled here, so the exact mass is already in the unit the
        // leaf's de-scaled plane is in (p_op carries 2^e and is divided by it).
        c.stats_exact[1 * c.count + ci] = (s.level == 4) ? mass_exact : 0.0;
        c.stats_exact[2 * c.count + ci] = (s.level == 3) ? var_exact : 0.0;
        c.stats_exact[3 * c.count + ci] = (s.level == 4) ? aprob_exact : 0.0;
      }
    }
  }
}

// max-abs and max-abs-over-max-abs-ref, ignoring ref entries that are exactly 0
inline void cmp(const std::vector<double>& got, const std::vector<double>& ref, double& max_abs,
                double& max_rel, int64_t& nonfinite) {
  double rmax = 0.0;
  for (double r : ref) rmax = std::max(rmax, std::fabs(r));
  max_abs = 0.0;
  max_rel = 0.0;
  for (size_t t = 0; t < ref.size(); ++t) {
    if (!std::isfinite(got[t])) { ++nonfinite; continue; }
    const double d = std::fabs(got[t] - ref[t]);
    max_abs = std::max(max_abs, d);
    if (rmax > 0.0) max_rel = std::max(max_rel, d / rmax);
  }
}

}  // namespace ref64
