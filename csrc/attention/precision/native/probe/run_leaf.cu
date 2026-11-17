// Device harness for the native advanced (L2-4) reader backward, row-owned half.
//
// One case = one reader (p.strict selects reader 2), full causal, right aligned.
// The candidate is either the tiled dQ kernel (LaunchMainDq) or the scalar
// reference schedule (LaunchMainDqScalar); the baseline is the independent FP64
// reference in ref64.hpp, which shares no code with the kernels.  Both dQ and
// the four statistics planes are compared, in Main (FP64 planes) and in ordered
// (int64 grids, decoded here exactly as the Python caller decodes them) mode.
//
// No torch, no autograd: raw cudaMalloc and the reduced ABI of adv_bwd_native.h.
// Numbers only; no timing and no speed claim.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "adv_bwd_native.h"
#include "ref64.hpp"

namespace na = ::xattn::ops::native_adv;

namespace xattn { namespace ops { namespace native_adv {
void LaunchMainDq(const Params& p, cudaStream_t stream);
void LaunchMainDqScalar(const Params& p, cudaStream_t stream);
} } }  // namespace xattn::ops::native_adv

namespace {

// Sentinel written into every statistics plane before a launch, so the
// untouched-plane contract can be measured instead of assumed.
constexpr double kStatsPoison = 12345.25;


struct HCase {
  ref64::Spec s;
  bool ordered;
  const char* name;
};

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::printf("{\"fatal\":\"%s\",\"err\":\"%s\"}\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

template <typename T, typename V>
T* upload(const std::vector<V>& v) {
  T* d = nullptr;
  std::vector<T> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = (T)v[i];
  if (v.empty()) return nullptr;
  check(cudaMalloc(&d, v.size() * sizeof(T)), "malloc");
  check(cudaMemcpy(d, h.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "copy-in");
  return d;
}

// The device reads element-dtype operands, so the host side of a case is its
// operands rounded to that dtype exactly as ref64 did before computing.
template <typename T>
T* upload_q(const std::vector<double>& v, int elem) {
  if (v.empty()) return nullptr;
  std::vector<T> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = (T)ref64::round_elem((float)v[i], elem);
  T* d = nullptr;
  check(cudaMalloc(&d, v.size() * sizeof(T)), "malloc");
  check(cudaMemcpy(d, h.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "copy-in");
  return d;
}

void run_case(const HCase& hc, const char* kernel, bool scalar, bool agg) {
  ref64::Case c = ref64::build(hc.s);

  const int64_t B = c.s.B, Nq = c.s.Nq, Nk = c.Nt, Hq = c.s.Hq, Hkv = c.s.Hkv;
  const int64_t D = c.s.D, Dv = c.s.Dv;
  const int64_t count = B * Hq * Nq;

  // ---- device operands -------------------------------------------------------
  void *q = nullptr, *k = nullptr, *v = nullptr, *dy = nullptr;
  if (c.s.elem == 0) {
    q = upload_q<__half>(c.q, 0);
    k = upload_q<__half>(c.k, 0);
    v = upload_q<__half>(c.v, 0);
    dy = upload_q<__half>(c.dy, 0);
  } else {
    q = upload_q<__nv_bfloat16>(c.q, 1);
    k = upload_q<__nv_bfloat16>(c.k, 1);
    v = upload_q<__nv_bfloat16>(c.v, 1);
    dy = upload_q<__nv_bfloat16>(c.dy, 1);
  }
  // The caller prescales dY by 2^-e at level 4 (meta plane 2 then carries 2^e,
  // and the leaf multiplies dP back up inside the bracket); the reference keeps
  // the true, unprescaled dY, so the comparison is against true units.
  {
    std::vector<double> dy_leaf = c.dy;
    if (c.s.level == 4 && c.s.anchor_scale) {
      for (int64_t ci = 0; ci < count; ++ci) {
        const int e = (int)(ci % 4);
        const int64_t i = ci % Nq, bh = ci / Nq, hq = bh % Hq, b = bh / Hq;
        const double f = std::ldexp(1.0, -e);
        for (int64_t d = 0; d < Dv; ++d) dy_leaf[((b * Nq + i) * Hq + hq) * Dv + d] *= f;
      }
      check(cudaFree(dy), "free dy");
      if (c.s.elem == 0) {
        dy = upload_q<__half>(dy_leaf, 0);
      } else {
        dy = reinterpret_cast<__half*>(upload_q<__nv_bfloat16>(dy_leaf, 1));
      }
    }
  }
  std::vector<double> dq_h((size_t)B * Nq * Hq * D, 0.0);
  float *delta = upload<float>(c.delta), *logz = upload<float>(c.logz);
  float* meta = upload<float>(c.meta);
  double* stats = nullptr;
  check(cudaMalloc(&stats, sizeof(double) * 4 * count), "malloc stats");
  check(cudaMemset(stats, 0, sizeof(double) * 4 * count), "zero stats");
  float* dq = nullptr;
  check(cudaMalloc(&dq, sizeof(float) * dq_h.size()), "malloc dq");
  check(cudaMemset(dq, 0, sizeof(float) * dq_h.size()), "zero dq");
  // dK/dV exist only in aggregate mode (the aggregate Launch chains the key-owned
  // leaf); in the leaf-only modes the pointers stay null so the leaf never sees them.
  std::vector<double> dk_h((size_t)B * c.Nt * Hkv * D, 0.0);
  std::vector<double> dv_h((size_t)B * c.Nt * Hkv * Dv, 0.0);
  float* dk = nullptr;
  float* dv = nullptr;
  if (agg) {
    // The aggregate entry must refuse before it enqueues anything when the
    // key-owned dK/dV leaf is not linked, so dQ is poisoned here: if any byte
    // changes, the refusal happened after partial side effects.
    std::vector<float> dq_poison(dq_h.size(), 1234.5f);
    check(cudaMemcpy(dq, dq_poison.data(), sizeof(float) * dq_h.size(), cudaMemcpyHostToDevice),
          "poison dq");
    check(cudaMalloc(&dk, sizeof(float) * dk_h.size()), "malloc dk");
    check(cudaMemset(dk, 0, sizeof(float) * dk_h.size()), "zero dk");
    check(cudaMalloc(&dv, sizeof(float) * dv_h.size()), "malloc dv");
    check(cudaMemset(dv, 0, sizeof(float) * dv_h.size()), "zero dv");
  }
  int64_t* qseg = c.qseg.empty() ? nullptr : upload<int64_t>(c.qseg);
  int64_t* kseg = c.kseg.empty() ? nullptr : upload<int64_t>(c.kseg);

  // Per-row grid exponent, computed the way the caller's _rho_exponent does
  // (_precision_backend.py:194-210): g_i = ||dY_i||_2 in float, w = max over all
  // rows of ||vc_row||_2 in float, bound = g*w in double, e = frexp(bound)+5,
  // clamped to +-300.  vc is the centered V that preprocess.cu builds: the first
  // key row of each batch is the shift, and a feature whose residual cannot be
  // added back exactly keeps shift 0.
  // DEVIATION FROM THE PRODUCTION PATH, stated so it is not read as more: the
  // caller computes this on the fp16-prescaled dY, and this probe has no fp16
  // prescale, so it uses the dY the case defines.  That moves e by the exact
  // power of two the prescale applies, i.e. it shifts every grid by the same
  // power of two and changes no tested arithmetic; the decode in the probe uses
  // the same e the leaf was given, so the round trip stays exact.
  std::vector<int32_t> expo((size_t)count, 0);
  if (hc.ordered && c.s.level >= 3) {
    std::vector<double> vc(c.v.size());
    for (int64_t b = 0; b < B; ++b)
      for (int64_t hk = 0; hk < Hkv; ++hk) {
        for (int64_t d = 0; d < Dv; ++d) {
          const double shift0 = c.v[((b * Nk + 0) * Hkv + hk) * Dv + d];
          bool take = true;
          for (int64_t j = 0; j < Nk && take; ++j) {
            const double orig = c.v[((b * Nk + j) * Hkv + hk) * Dv + d];
            const float residual = (float)((float)(orig - shift0));
            if ((double)residual + shift0 != orig) take = false;
          }
          const double shift = take ? shift0 : 0.0;
          for (int64_t j = 0; j < Nk; ++j)
            vc[((b * Nk + j) * Hkv + hk) * Dv + d] =
                c.v[((b * Nk + j) * Hkv + hk) * Dv + d] - shift;
        }
      }
    double w = 0.0;
    for (int64_t b = 0; b < B; ++b)
      for (int64_t hk = 0; hk < Hkv; ++hk)
        for (int64_t j = 0; j < Nk; ++j) {
          float acc = 0.0f;
          for (int64_t d = 0; d < Dv; ++d) {
            const float t = (float)vc[((b * Nk + j) * Hkv + hk) * Dv + d];
            acc += t * t;
          }
          const double row_norm = (double)std::sqrt(acc);
          if (row_norm > w) w = row_norm;
        }
    for (int64_t ci = 0; ci < count; ++ci) {
      const int64_t i = ci % Nq, bh = ci / Nq, hq = bh % Hq, b = bh / Hq;
      float acc = 0.0f;
      for (int64_t d = 0; d < Dv; ++d) {
        const float t = (float)c.dy[((b * Nq + i) * Hq + hq) * Dv + d];
        acc += t * t;
      }
      const double bound = (double)std::sqrt(acc) * w;
      int e = 0;
      if (bound > 0.0 && std::isfinite(bound)) {
        int ex = 0;
        std::frexp(bound, &ex);
        e = ex + 5;
      } else if (!std::isfinite(bound)) {
        e = 300;
      }
      if (e > 300) e = 300;
      if (e < -300) e = -300;
      expo[ci] = e;
    }
  }
  // The emulation of the ordered grids needs the caller's exponent, so the
  // reference is computed once it is known rather than at build time.
  c.grid_e = expo;
  ref64::backward(c);
  int32_t* expo_d = nullptr;
  if (hc.ordered && c.s.level >= 3) {
    check(cudaMalloc(&expo_d, sizeof(int32_t) * expo.size()), "malloc exp");
    check(cudaMemcpy(expo_d, expo.data(), sizeof(int32_t) * expo.size(), cudaMemcpyHostToDevice),
          "copy exp");
  }

  // ---- params ---------------------------------------------------------------
  na::Params p{};
  p.q = q; p.k = k; p.v = v; p.dy = dy;
  p.delta = delta; p.logz = logz; p.meta = meta;
  p.stats = stats;
  p.rho_exponent = expo_d;
  int32_t sink_status = 0;
  int32_t* status_d = nullptr;
  check(cudaMalloc(&status_d, sizeof(int32_t)), "malloc status");
  check(cudaMemcpy(status_d, &sink_status, sizeof(int32_t), cudaMemcpyHostToDevice), "copy status");
  p.sink_status = status_d;
  p.dq = dq;
  p.q_seg = qseg; p.k_seg = kseg;
  p.q_batch_stride = Nq * Hq * D; p.q_row_stride = Hq * D; p.q_head_stride = D;
  p.k_batch_stride = Nk * Hkv * D; p.k_row_stride = Hkv * D; p.k_head_stride = D;
  p.v_batch_stride = Nk * Hkv * Dv; p.v_row_stride = Hkv * Dv; p.v_head_stride = Dv;
  p.dy_batch_stride = Nq * Hq * Dv; p.dy_row_stride = Hq * Dv; p.dy_head_stride = Dv;
  p.dq_batch_stride = Nq * Hq * D; p.dq_row_stride = Hq * D; p.dq_head_stride = D;
  p.dk_batch_stride = Nk * Hkv * D; p.dk_row_stride = Hkv * D; p.dk_head_stride = D;
  p.dv_batch_stride = Nk * Hkv * Dv; p.dv_row_stride = Hkv * Dv; p.dv_head_stride = Dv;
  p.rowstat_head_stride = Nq;
  p.B = B; p.Nq = Nq; p.Nk = Nk; p.Nt = Nk; p.Hq = Hq; p.Hkv = Hkv;
  p.D = D; p.Dv = Dv; p.offset = Nk - Nq; p.qk_ratio = Hq / Hkv;
  p.visibility = 0; p.level = c.s.level;
  p.softmax_scale = c.scale;
  // The host forms this as float * float (adv_bwd_host.cc); a double product
  // rounded once is a different last bit, and this field is what the leaf uses.
  p.softmax_scale_log2 = (float)c.scale * 1.4426950408889634f;
  p.dk = dk;
  p.dv = dv;
  p.dk_batch_stride = c.Nt * Hkv * D;
  p.dk_row_stride = Hkv * D;
  p.dk_head_stride = D;
  p.dv_batch_stride = c.Nt * Hkv * Dv;
  p.dv_row_stride = Hkv * Dv;
  p.dv_head_stride = Dv;
  p.strict = c.s.strict;
  p.two_component_dv = (c.s.level == 4) || (Nk > Nq);
  p.ordered = hc.ordered;
  p.elem = (c.s.elem == 0) ? na::Elem::kFp16 : na::Elem::kBf16;

  // ---- run + repeat ---------------------------------------------------------
  bool agg_refused = false;
  std::string agg_msg;
  auto launch = [&]() {
    if (agg) {
      try {
        ::xattn::ops::native_adv::Launch(p, nullptr);
      } catch (const std::exception& e) {
        agg_refused = true;
        agg_msg = e.what();
      }
    } else if (scalar) {
      ::xattn::ops::native_adv::LaunchMainDqScalar(p, nullptr);
    } else {
      ::xattn::ops::native_adv::LaunchMainDq(p, nullptr);
    }
    check(cudaDeviceSynchronize(), "launch");
  };
  {
    std::vector<double> poison(4 * (size_t)count, kStatsPoison);
    check(cudaMemcpy(stats, poison.data(), sizeof(double) * poison.size(), cudaMemcpyHostToDevice),
          "poison stats 1");
  }
  launch();
  std::vector<float> dq1(dq_h.size());
  std::vector<float> dk1(dk_h.size()), dv1(dv_h.size());
  if (agg) {
    check(cudaMemcpy(dk1.data(), dk, sizeof(float) * dk_h.size(), cudaMemcpyDeviceToHost), "dk out");
    check(cudaMemcpy(dv1.data(), dv, sizeof(float) * dv_h.size(), cudaMemcpyDeviceToHost), "dv out");
  }
  std::vector<double> st1(4 * (size_t)count);
  check(cudaMemcpy(dq1.data(), dq, sizeof(float) * dq_h.size(), cudaMemcpyDeviceToHost), "dq out");
  if (agg) {
    int64_t touched = 0;
    for (size_t k = 0; k < dq1.size(); ++k)
      if (dq1[k] != 1234.5f) ++touched;
    const bool names_leaf = agg_msg.find("key-owned dK/dV leaf is not linked") != std::string::npos;
    std::printf(
        "{\"case\":\"%s\",\"kernel\":\"agg\",\"mode\":\"%s\",\"level\":%d,"
        "\"refused\":%d,\"names_missing_leaf\":%d,\"dq_bytes_touched\":%lld}\n",
        hc.name, hc.ordered ? "ordered" : "main", c.s.level, (int)agg_refused, (int)names_leaf,
        (long long)touched);
    cudaFree(dq); cudaFree(stats); cudaFree(dk); cudaFree(dv);
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(dy);
    cudaFree(delta); cudaFree(logz); cudaFree(meta);
    if (qseg) cudaFree(qseg);
    if (kseg) cudaFree(kseg);
    if (expo_d) cudaFree(expo_d);
    cudaFree(status_d);
    return;
  }
  check(cudaMemcpy(st1.data(), stats, sizeof(double) * st1.size(), cudaMemcpyDeviceToHost), "st out");
  check(cudaMemset(dq, 0, sizeof(float) * dq_h.size()), "zero dq");
  // Poison, not zeros: a plane the level does not produce must be left exactly
  // as the caller left it (pinned sink block :1944, :2021-2034 stores no zero).
  {
    std::vector<double> poison(st1.size(), kStatsPoison);
    check(cudaMemcpy(stats, poison.data(), sizeof(double) * st1.size(), cudaMemcpyHostToDevice),
          "poison stats");
    check(cudaMemcpy(status_d, &sink_status, sizeof(int32_t), cudaMemcpyHostToDevice), "reset status");
  }
  launch();
  std::vector<float> dq2(dq_h.size());
  std::vector<double> st2(4 * (size_t)count);
  check(cudaMemcpy(dq2.data(), dq, sizeof(float) * dq_h.size(), cudaMemcpyDeviceToHost), "dq out2");
  check(cudaMemcpy(st2.data(), stats, sizeof(double) * st2.size(), cudaMemcpyDeviceToHost), "st out2");

  // ---- decode ordered grids back to FP64, exactly as the caller does --------
  std::vector<double> st_dec = st1;
  if (hc.ordered) {
    const int64_t* g = reinterpret_cast<const int64_t*>(st1.data());
    for (int64_t ci = 0; ci < count; ++ci) {
      const double e = (double)expo[ci];
      st_dec[0 + ci] = (double)g[0 + ci] * std::ldexp(1.0, (int)(e - 62));
      st_dec[1 * count + ci] = (double)g[1 * count + ci] * std::ldexp(1.0, -60);
      st_dec[2 * count + ci] = (double)g[2 * count + ci] * std::ldexp(1.0, (int)(2 * e - 68));
      st_dec[3 * count + ci] = (double)g[3 * count + ci] * std::ldexp(1.0, -60);
    }
  }

  // ---- compare --------------------------------------------------------------
  std::vector<double> got(dq1.begin(), dq1.end());
  // Two references, both FP64: c.dq multiplies the FP32 dS, c.dq_pin the
  // element-rounded dS that the pinned GEMM operand carries.  The kernel must
  // match c.dq_pin, since its operand is element typed (:2082, :2102).
  double dq_abs = 0, dq_rel = 0;
  int64_t dq_nf = 0;
  ref64::cmp(got, c.dq, dq_abs, dq_rel, dq_nf);
  double dqp_abs = 0, dqp_rel = 0;
  int64_t dqp_nf = 0;
  ref64::cmp(got, c.dq_pin, dqp_abs, dqp_rel, dqp_nf);
  double dq_ref_max = 0.0;
  for (double r : c.dq) dq_ref_max = std::max(dq_ref_max, std::fabs(r));
  // Statistics: compare only the planes this level produces; the rest must still
  // hold the poison.
  std::vector<double> st_got, st_ref;
  int64_t untouched_bad = 0;
  double untouched_max_abs = 0.0;
  for (int pl = 0; pl < 4; ++pl) {
    const bool defined = (pl == 0 && c.s.level >= 3) || (pl == 2 && c.s.level == 3) ||
                         ((pl == 1 || pl == 3) && c.s.level == 4);
    for (int64_t ci = 0; ci < count; ++ci) {
      const double g = st_dec[pl * count + ci];
      if (defined) {
        st_got.push_back(g);
        st_ref.push_back(c.stats[pl * count + ci]);
      } else {
        // Raw plane, as the leaf left it; decoding an undefined plane would
        // reinterpret the caller's own bytes and is not part of the contract.
        const double raw = st1[(size_t)(pl * count + ci)];
        if (raw != kStatsPoison) {
          ++untouched_bad;
          untouched_max_abs = std::max(untouched_max_abs, std::fabs(raw - kStatsPoison));
        }
      }
    }
  }
  double st_abs = 0, st_rel = 0;
  int64_t st_nf = 0;
  ref64::cmp(st_got, st_ref, st_abs, st_rel, st_nf);
  // Second reference on the same rows: exact FP32 dS and exact P, i.e. what the
  // statistics would be if the sink operand were not element typed.  The gap
  // between the two residuals is the cast floor, on the record as a number.
  // Same plane gating as st_got, or the two vectors would not be the same
  // length and cmp() would read past the end.
  std::vector<double> st_exact;
  st_exact.reserve(st_got.size());
  for (int pl = 0; pl < 4; ++pl) {
    const bool defined = (pl == 0 && c.s.level >= 3) || (pl == 2 && c.s.level == 3) ||
                         ((pl == 1 || pl == 3) && c.s.level == 4);
    if (!defined) continue;
    for (int64_t ci = 0; ci < count; ++ci) st_exact.push_back(c.stats_exact[(size_t)(pl * count + ci)]);
  }
  if (st_exact.size() != st_got.size()) {
    std::printf("  probe bug: st_exact %zu vs st_got %zu\n", st_exact.size(), st_got.size());
    st_exact.resize(st_got.size(), 0.0);
  }
  double st_abs_x = 0, st_rel_x = 0;
  int64_t st_nf_x = 0;
  ref64::cmp(st_got, st_exact, st_abs_x, st_rel_x, st_nf_x);

  // Exact properties that do not depend on any tolerance.
  // Ordered-grid comparison against the two fold schedules (ref64 emulation).
  double g_phys_units = -1.0, g_mask_units = -1.0;
  int64_t g_mask_rows = -1;
  // Kernel and model value at the row where the physical-fold comparison is
  // worst, so the difference can be read in the plane's own units instead of
  // being quoted as a bare magnitude.
  int64_t g_kernel_at = 0, g_model_at = 0, g_ci_at = -1;
  int g_pl_at = -1;
  int64_t empty_rows = 0, empty_nonzero = 0;
  double empty_max_abs = 0.0;
  int64_t worst_ci = -1, worst_d = -1;
  double stats_max_abs = 0.0;
  for (int64_t ci = 0; ci < count; ++ci) {
    const bool empty = c.meta[3 * count + ci] < 0.0;
    if (!empty) continue;
    ++empty_rows;
    const int64_t i = ci % Nq, bh = ci / Nq, hq = bh % Hq, b = bh / Hq;
    for (int64_t d = 0; d < D; ++d) {
      const double x = dq1[(((b * Nq + i) * Hq + hq) * D) + d];
      if (std::fabs(x) > empty_max_abs) { empty_max_abs = std::fabs(x); worst_ci = ci; worst_d = d; }
      if (x != 0.0 || !std::isfinite(x)) ++empty_nonzero;
    }
  }
  for (double x : c.stats) stats_max_abs = std::max(stats_max_abs, std::fabs(x));
  check(cudaMemcpy(&sink_status, status_d, sizeof(int32_t), cudaMemcpyDeviceToHost), "status out");
  bool repeat_ok = (std::memcmp(dq1.data(), dq2.data(), sizeof(float) * dq1.size()) == 0) &&
                   (std::memcmp(st1.data(), st2.data(), sizeof(double) * st1.size()) == 0);

  // Per-plane diagnostics: which of the four planes disagrees, and on which row.
  {
    double pmax[4] = {0, 0, 0, 0}, premax[4] = {0, 0, 0, 0};
    int wp = -1; int64_t wci = -1; double wgot = 0, wref = 0, wabs = 0;
    // Only the planes this level produces: an unproduced plane still holds the
    // caller's pre-fill and comparing it would read as a defect that is not one.
    auto produced = [&](int pl) {
      if (pl == 0) return c.s.level >= 3;
      if (pl == 2) return c.s.level == 3;
      return c.s.level == 4;  // planes 1 and 3
    };
    if (hc.ordered && c.s.level >= 3) {
      // Every produced plane, as the raw int64 the leaf stored, against the two
      // fold schedules: physical tile boundaries (fixed) and boundary-only-if-
      // admitted (the pre-fix segment-`continue` behaviour).
      const int64_t* gk = reinterpret_cast<const int64_t*>(st1.data());
      g_phys_units = 0.0;
      g_mask_units = 0.0;
      g_mask_rows = 0;
      for (int pl = 0; pl < 4; ++pl) {
        if (!produced(pl)) continue;
        for (int64_t ci = 0; ci < count; ++ci) {
          const int64_t gv = gk[pl * count + ci];
          const int64_t pv = c.grid_phys[(size_t)pl * count + ci];
          const int64_t mv = c.grid_mask[(size_t)pl * count + ci];
          const long long d = (long long)std::llabs(gv - pv);
          if ((double)d > g_phys_units) {
            g_phys_units = (double)d;
            g_kernel_at = gv; g_model_at = pv; g_ci_at = ci; g_pl_at = pl;
          }
          g_mask_units = std::max(g_mask_units, (double)std::llabs(gv - mv));
        }
      }
      g_mask_rows = c.grid_mask_rows;
    }
    for (int pl = 0; pl < 4; ++pl)
      if (produced(pl))
      for (int64_t ci = 0; ci < count; ++ci) {
        const double g = st_dec[pl * count + ci], r = c.stats[pl * count + ci];
        pmax[pl] = std::max(pmax[pl], std::fabs(g));
        premax[pl] = std::max(premax[pl], std::fabs(r));
        const double a = std::fabs(g - r);
        if (a > wabs) { wabs = a; wp = pl; wci = ci; wgot = g; wref = r; }
      }
    if (count <= 8 && c.s.level == 4) {
      for (int64_t ci = 0; ci < count; ++ci)
        std::printf("    row ci=%lld e=%d got p1=%.9e p3=%.9e | ref p1=%.9e p3=%.9e\n",
                    (long long)ci, (int)(ci % 4), st_dec[1 * count + ci], st_dec[3 * count + ci],
                    c.stats[1 * count + ci], c.stats[3 * count + ci]);
    }
    std::printf("  plane_max |got| p0=%.6e p1=%.6e p2=%.6e p3=%.6e | |ref| p0=%.6e p1=%.6e p2=%.6e p3=%.6e | worst plane=%d ci=%lld got=%.6e ref=%.6e\n",
                pmax[0], pmax[1], pmax[2], pmax[3], premax[0], premax[1], premax[2], premax[3], wp,
                (long long)wci, wgot, wref);
  }
  std::printf(
      "{\"case\":\"%s\",\"kernel\":\"%s\",\"mode\":\"%s\",\"level\":%d,\"elem\":\"%s\","
      "\"strict\":%d,\"segments\":%d,\"B\":%lld,\"Nq\":%lld,\"Nt\":%lld,\"Hq\":%lld,"
      "\"Hkv\":%lld,\"D\":%lld,\"Dv\":%lld,\"rows\":%lld,\"empty_rows\":%lld,"
      "\"dq_max_abs\":%.6e,\"dq_max_rel\":%.6e,\"dq_pin_rel\":%.6e,\"dq_pin_abs\":%.6e,"
      "\"dq_nonfinite\":%lld,\"untouched_bad\":%lld,\"untouched_max_abs\":%.6e,\"sink_status\":%d,"
      "\"stats_max_abs\":%.6e,\"stats_max_rel\":%.6e,\"stats_nonfinite\":%lld,"
      "\"stats_ref_max\":%.6e,\"dq_ref_max\":%.6e,\"empty_dq_nonzero\":%lld,"
      "\"empty_dq_max_abs\":%.6e,\"empty_worst_ci\":%lld,\"empty_worst_d\":%lld,"
      "\"stats_exact_max_abs\":%.6e,\"stats_exact_max_rel\":%.6e,"
      "\"repeat_identical\":%s,\"grid_phys_units\":%.0f,\"grid_mask_units\":%.0f,"
      "\"grid_mask_rows\":%lld,\"grid_kernel\":%lld,\"grid_model\":%lld,\"grid_ci\":%lld,\"grid_pl\":%d}\n",
      hc.name, kernel, hc.ordered ? "ordered" : "main", c.s.level, c.s.elem == 0 ? "fp16" : "bf16",
      (int)c.s.strict, (int)c.s.segments, (long long)B, (long long)Nq, (long long)Nk, (long long)Hq,
      (long long)Hkv, (long long)D, (long long)Dv, (long long)count, (long long)empty_rows, dq_abs,
      dq_rel, dqp_rel, dqp_abs, (long long)dq_nf, (long long)untouched_bad, untouched_max_abs,
      (int)sink_status, st_abs, st_rel, (long long)st_nf, stats_max_abs,
      dq_ref_max, (long long)empty_nonzero, empty_max_abs, (long long)worst_ci, (long long)worst_d,
      st_abs_x, st_rel_x, repeat_ok ? "true" : "false", g_phys_units, g_mask_units,
      (long long)g_mask_rows, (long long)g_kernel_at, (long long)g_model_at,
      (long long)g_ci_at, g_pl_at);
  std::fflush(stdout);

  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(dy);
  cudaFree(delta); cudaFree(logz); cudaFree(meta); cudaFree(stats); cudaFree(dq);
  if (qseg) cudaFree(qseg);
  if (kseg) cudaFree(kseg);
  if (expo_d) cudaFree(expo_d);
  cudaFree(status_d);
}

}  // namespace

int main(int argc, char** argv) {
  const bool scalar = (argc > 1 && std::strcmp(argv[1], "scalar") == 0);
  const bool agg = (argc > 1 && std::strcmp(argv[1], "agg") == 0);
  const char* kernel = agg ? "agg" : (scalar ? "scalar" : "tiled");

  std::vector<HCase> cases;
  auto add = [&](const char* name, int level, int elem, bool strict, bool segments, int B, int Nq,
                 int Nk, int Hq, int Hkv, int D, int Dv, bool anchor_scale, uint64_t seed,
                 int hole_period = 0) {
    HCase h;
    h.name = name;
    h.s = ref64::Spec{};
    h.s.B = B; h.s.Nq = Nq; h.s.Nk = Nk - 0; h.s.Hq = Hq; h.s.Hkv = Hkv;
    h.s.D = D; h.s.Dv = Dv; h.s.level = level; h.s.strict = strict; h.s.segments = segments;
    h.s.anchor_scale = anchor_scale; h.s.elem = elem; h.s.seed = seed;
    h.s.nseg = segments ? 3 : 0;
    h.s.hole_period = hole_period;
    h.ordered = false;
    (void)Nk;
    cases.push_back(h);
    HCase o = h;
    o.ordered = true;
    o.name = name;
    cases.push_back(o);
  };

  // Declared cases: both readers, levels 2/3/4, both element types, GQA/MQA,
  // previous K/V (Nt > Nq), segments with empty rows, and a level-4 case with a
  // non-unit row scale.
  add("L2_square_r1", 2, 0, false, false, 1, 8, 8, 2, 1, 32, 32, false, 11);
  add("L2_square_r2", 2, 0, true, false, 1, 8, 8, 2, 1, 32, 32, false, 12);
  add("L2_prev_gqa", 2, 1, false, false, 2, 6, 10, 4, 2, 64, 64, false, 13);
  add("L3_prev_seg", 3, 0, true, true, 2, 6, 10, 4, 2, 64, 64, false, 23);
  add("L3_mqa_strict", 3, 1, true, false, 1, 16, 16, 8, 1, 128, 128, false, 24);
  add("L4_scale_r1", 4, 1, false, false, 1, 5, 5, 4, 1, 64, 64, true, 31);
  add("L4_scale_r2_seg", 4, 0, true, true, 1, 5, 9, 4, 1, 64, 64, true, 32);
  add("L4_mqa_wide", 4, 1, false, false, 1, 12, 20, 8, 1, 256, 256, true, 33);
  // One admitted key per row, four rows with 2^e = 1,2,4,8: mass and anchor
  // probability must be exactly 1 for every row, so any e-dependent factor
  // shows up directly.
  add("L4_one_key", 4, 1, false, false, 1, 1, 1, 4, 1, 32, 32, true, 41);
  // Long rows: the number of admitted keys per row crosses the 32-key sink
  // tile boundary several times, so the per-tile de-scale and ordered
  // rounding happen more than once per row.
  add("L2_long_gqa", 2, 0, false, false, 1, 24, 96, 4, 2, 32, 32, false, 51);
  add("L3_long_seg", 3, 1, true, true, 1, 40, 200, 2, 2, 64, 64, false, 52);
  add("L4_long_mqa", 4, 0, false, false, 1, 8, 160, 8, 1, 64, 64, true, 53);
  // Segment holes that land ON physical 32-key boundaries: the only shape that
  // distinguishes physical-tile rounding from mask-dependent rounding.  The
  // generator emits runs of 1-3 keys per segment id, so with Nt = 96 and a
  // query row reaching all 96 keys, key 31, 63 and 95 are mismatches for
  // essentially every segment - that is the shape the earlier `continue` above
  // the flush folded wrongly.
  add("L3_hole_96", 3, 0, false, true, 1, 24, 96, 1, 1, 64, 64, false, 61);
  add("L4_hole_96", 4, 1, true, true, 1, 24, 96, 2, 2, 64, 64, true, 62);
  // The only shapes that separate the pin's fold schedule from the pre-fix one:
  // every 32nd key belongs to another segment, so each physical tile ends on a
  // key this query rejects while keys on BOTH sides of it are admitted.  With
  // generated segments (above) a query's segment is contiguous, so a boundary
  // hole only ever truncates the tail and the two schedules coincide.
  add("L3_holephys", 3, 0, false, true, 1, 4, 256, 1, 1, 32, 32, false, 71, 32);
  add("L4_holephys", 4, 1, false, true, 1, 4, 256, 1, 1, 32, 32, true, 72, 32);
  add("L4_holephys64", 4, 0, false, true, 1, 4, 256, 1, 1, 32, 32, false, 73, 64);

  for (const HCase& h : cases) run_case(h, kernel, scalar, agg);
  std::printf("{\"done\":true,\"kernel\":\"%s\",\"cases\":%zu}\n", kernel, cases.size());
  return 0;
}
