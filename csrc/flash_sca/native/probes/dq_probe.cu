// Standalone FP64 numerics probe for the native dQ leaf.
//
// Drives `DqKernel` directly (no torch, no extension, no dispatch) against a
// host FP64 reference of the same algebra the contract states, on cases chosen
// to hit every branch this leaf has:
//   A  analytic bounds, C = N/4, single segment, GQA (group 2), the fast
//      full-tile path (TileIsFull true for every non-diagonal tile);
//   B  unequal Q/K lengths with previous K/V (one whole prev key tile) and a
//      partial query tile, i.e. ResolveKeyMap + prev row0 offset + tail TMA fill;
//   C  explicit row_lo/row_hi with a genuinely empty row (lo > hi) and a row
//      whose interval is empty only because of the segment test;
//   D  explicit segment planes (q_segment/k_segment, segment_exact == nullptr),
//      i.e. ARM B of the leaf's mask: two segments over 4 key tiles;
//   E  the p.delta == nullptr fallback, so the saved-output (dY.y) path is
//      exercised too, at level 0 (saved in the input dtype) - the path the
//      saved-plane fault-injection arms (5 and 6) need.
// Each case runs at both buckets (128/128 and 256/256) and both dtypes
// (BF16 and FP16). A non-trivial `scale` (not 1/sqrt(D)) is used in case C so a
// missing or doubled scale cannot hide behind a value near one.
//
// Reference: FP64, formed from the SAME fp16/bf16 inputs the kernel reads, in
// this order - s_raw, lse (natural log), P = exp(scale*s_raw - lse), dP, delta =
// sum_j P dP, dS = P (dP - delta), dQ = scale * sum_j dS K. That is the seed's
// algebra with no scaling anywhere before the final multiply, exactly the ABI
// the leaf obeys. The kernel's 16-bit dS pack is a rounding the reference does
// not have, so the probe reports max abs / max rel-to-peak / RMS rather than
// demanding bit equality.
//
// Build+run: `bash csrc/flash_sca/native/probes/run_dq_probe.sh` from the repo
// root (one `compute run`, gpus 8). That script and this probe are shipped
// together in the tree, so the path above is the one a reader can follow; it also
// prints the sha256 of the leaf it compiled, so an evidence row can name its source.
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "flash_sca/native/umma_bwd_dq.cuh"

using xattn::ops::flash_sca_native::Element;
using xattn::ops::flash_sca_native::Operand;
using xattn::ops::flash_sca_native::Params;
using xattn::ops::flash_sca_native::umma::LaunchUmmaDq;
using xattn::ops::flash_sca_native::umma::UmmaInputMaps;
using xattn::ops::flash_sca_native::umma::UmmaLaunchParams;

static void Ck(cudaError_t e, const char* what, int line) {
  if (e != cudaSuccess) {
    std::printf("CUDA FAILURE %s (line %d): %s\n", what, line, cudaGetErrorString(e));
    std::exit(2);
  }
}
#define CK(x) Ck((x), #x, __LINE__)

// ------------------------------------------------------- digest ledger + claims
// A bitwise claim that spans two rows ("this arm must print that arm's bits")
// needs both digests to still exist when the second one is computed, so the ledger
// keeps what every finished case printed, keyed by (dtype tag, case name). A
// digest that is printed and never compared is not evidence, it is a screenshot:
// the row that claims another row's bits has to be able to read that digest.
struct DigestSlot {
  const char* tag;
  const char* name;
  uint64_t hash;
};
static DigestSlot g_digest[128];
static int g_ndigest = 0;

static void DigestPut(const char* tag, const char* name, uint64_t hash) {
  if (g_ndigest < 128) g_digest[g_ndigest++] = DigestSlot{tag, name, hash};
}
static bool DigestGet(const char* tag, const char* name, uint64_t* out) {
  for (int i = 0; i < g_ndigest; ++i) {
    if (std::strcmp(g_digest[i].tag, tag) == 0 && std::strcmp(g_digest[i].name, name) == 0) {
      *out = g_digest[i].hash;
      return true;
    }
  }
  return false;
}

// Rows whose verdict is BIT equality with another row of the same dtype (same
// shape, same conditioning, same seed - so the digests are comparable):
//   saved-poison-supplied: the saved plane is poisoned while delta IS supplied, so
//     a correct leaf never reads it and the answer must be analytic-gqa2's bits;
//   fill-nan / fill-big: the dq allocation is garbage (quiet NaN / a large finite
//     value), and every live element must still be overwritten by the store;
//   tiles3-partial-first-again: the SAME call launched a second time in the same
//     process, so every TMEM window still holds the first launch's accumulators.
//     Identical inputs must give a bit-identical dQ, so the repeat's digest must
//     equal the first launch's - a reset that is missing, or keyed on the wrong
//     loop index, moves the second launch's bits while both runs still agree with
//     the reference to four printed digits. This is the determinism row of the
//     standalone probe; the pinned-oracle repeat belongs to the mounted
//     qualification, not here.
static const char* MustEqualRow(const char* name) {
  if (std::strcmp(name, "saved-poison-supplied") == 0) return "analytic-gqa2";
  if (std::strcmp(name, "fill-nan") == 0) return "analytic-gqa2";
  if (std::strcmp(name, "fill-big") == 0) return "analytic-gqa2";
  if (std::strcmp(name, "tiles3-partial-first-again") == 0) return "tiles3-partial-first";
  return nullptr;
}

// ---------------------------------------------------------------- TMA encoder
// The tree's `EncodeInputMap` (umma_host.h:42-88) minus its torch checks: same
// axes [column, head, row, batch], same 64x64 box, same 128 B swizzle, same
// dtype slots and the same `byte_stride = element_stride * 2`.
static PFN_cuTensorMapEncodeTiled_v12000 TensorMapEncoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* address = nullptr;
    cudaDriverEntryPointQueryResult query{};
    cudaError_t st = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &address, 12000, cudaEnableDefault, &query);
    if (st != cudaSuccess || address == nullptr) {
      std::printf("could not resolve cuTensorMapEncodeTiled\n");
      std::exit(2);
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
  }();
  return fn;
}

static CUtensorMap Encode(const Operand& src, Element el, int64_t batch, int64_t rows,
                          int64_t heads, int64_t columns) {
  CUtensorMap map{};
  if (batch == 0 || rows == 0) return map;
  std::array<cuuint64_t, 4> extent{static_cast<cuuint64_t>(columns),
                                   static_cast<cuuint64_t>(heads),
                                   static_cast<cuuint64_t>(rows),
                                   static_cast<cuuint64_t>(batch)};
  int64_t head_stride = heads == 1 ? columns : src.head_stride;
  int64_t row_stride = rows == 1 ? head_stride * heads : src.row_stride;
  int64_t batch_stride = batch == 1 ? row_stride * rows : src.batch_stride;
  std::array<int64_t, 3> element_stride{head_stride, row_stride, batch_stride};
  std::array<cuuint64_t, 3> byte_stride{};
  for (int i = 0; i < 3; ++i) byte_stride[i] = static_cast<cuuint64_t>(element_stride[i]) * 2;
  const std::array<cuuint32_t, 4> box{64, 1, 64, 1};
  const std::array<cuuint32_t, 4> step{1, 1, 1, 1};
  const auto type =
      el == Element::kHalf ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  const CUresult r = TensorMapEncoder()(
      &map, type, 4, const_cast<void*>(src.ptr), extent.data(), byte_stride.data(), box.data(),
      step.data(), CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::printf("cuTensorMapEncodeTiled failed: %d\n", static_cast<int>(r));
    std::exit(2);
  }
  return map;
}

// ---------------------------------------------------------------- case shape
struct Case {
  const char* name;
  int64_t batch, nq, nk, prev, hq, hkv, d, dv, chunk;
  bool explicit_bounds;   // row_lo/row_hi supplied
  bool segments;          // q_segment/k_segment supplied (non-exact: ARM B)
  bool empty_row;         // make row 5 of batch 0 empty in explicit mode
  bool null_delta;        // exercise the dY.y fallback (needs the saved plane)
  float scale_mult;       // scale = scale_mult / sqrt(D)
  int64_t total() const { return prev + nk; }
  int64_t group() const { return hq / hkv; }
  // (kept last so the case table can leave it out) non-finite LSE fault arm:
  // see the injection block in the runner. Never set on a case that also uses
  // empty_row, so the -inf and the +inf/NaN rows stay distinct arms.
  bool lse_sentinel = false;
  // delta / saved-plane fault arms. 0 = none; 1 delta plane -> NaN; 2 delta plane
  // -> large finite (the quantitative sibling: the model and the device are given
  // the SAME bad delta, so the row still has an error to measure); 3 delta -> NaN
  // on one row of every head; 4 delta -> NaN on one row of one head (the index and
  // head-of-the-group discriminator: every other row and head must stay correct);
  // 5 the saved plane -> NaN while delta IS supplied (a correct leaf never reads
  // it, so dq must be bit-identical to the clean run); 6 the saved plane -> NaN
  // with delta null (the positive control for 5: the fallback reads that plane, so
  // dq MUST move).
  int fault = 0;
  int64_t fault_row = 0;   // row i in batch 0, for arms 3/4/5
  int64_t fault_head = 0;  // head h in batch 0, for arms 4/5
  float fault_finite = 1.0e4f;
  // arms 1, 3, 4 corrupt delta on purpose: the model sees the same corruption, so
  // the assertion is "the same entries are NaN on both sides and every finite
  // entry still agrees" rather than a plain error bound.
  bool expect_nan = false;
  // arm 6 poisons a plane the reference does not have: its assertion is only that
  // the answer MOVES (peak garbage), never a numeric bound.
  bool expect_moved = false;
  // dq-allocation fill before the call: 0 = 0xEE (as shipped), 1 = 0x7FC00000
  // quiet NaN, 2 = large finite. Both non-default fills assert that an unwritten
  // accumulator cannot be read silently: a NaN shows up in the store, a large
  // finite value cannot be hidden by a clamp.
  int sentinel_fill = 0;
  // Arm 3: a row with no visible key must be written as EXACTLY zero, with no NaN
  // bit pattern, even when delta is corrupted around it. Checked on the bits, not
  // against a tolerance, because "no gradient contribution" is a bit-level claim.
  bool exact_zero_row = false;
  int64_t zero_row = 0;  // row i in batch 0, all heads
  // Level-1 form of the saved plane: the forward's FP32 state (`y_state`) instead of
  // the 16-bit y. The leaf's dY.y fallback reads it as float when `saved_fp32` is set,
  // so this arm is the only place that branch is executed at all.
  bool saved_fp32 = false;
};

template <class E>
static E FromF(float v);
template <>
__half FromF<__half>(float v) { return __float2half(v); }
template <>
__nv_bfloat16 FromF<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
static float ToF(__half v) { return __half2float(v); }
static float ToF(__nv_bfloat16 v) { return __bfloat162float(v); }

// The launcher is templated on the integer buckets; dispatch on the case's dims.
// The dispatch is on the BUCKET (QkBucket/ValueBucket), exactly as the host does
// it, so a case may declare a padded logical dimension (160 -> the 256 bucket) and
// still reach a real instantiation.
// Both bucket predicates come from dimension_bucket.h via the leaf's params.h
// include, so this torch-free probe needs no separate dispatch definition.

// Counts the preflight-bound violations the negative arms find. They are not case
// failures (no kernel runs), but they are verdicts: a predicate that accepts a width
// its writer cannot hold is a wrong-answer hazard, so they fold into the exit code.
static int g_preflight_fail = 0;

// The fail path, exercised rather than asserted. A run that is green everywhere
// never executes the branch that reports a refusal, so a defect in it would ship
// unexecuted. With DQ_PROBE_INJECT_REFUSAL set, `main` runs one valid case with
// `dq_accum` made non-null: the leaf serves the deterministic arm only, so its own
// preflight refuses, `RunCase` returns 2, and the summary AND the process status
// must both go red. The green table above is untouched - the fixture is one extra
// arm, counted separately in the same totals, and the runner requires its nonzero
// status as its last step.
static bool g_inject_refusal = false;
// Never dereferenced: the refusal is decided before any launch, so only its
// null-ness reaches the predicate.
static float g_refusal_sentinel = 0.f;

template <class E>
static bool LaunchFor(const UmmaLaunchParams& lp, int64_t d, int64_t dv) {
  const int bd = xattn::ops::flash_sca_native::QkBucket(d);
  const int bv = xattn::ops::flash_sca_native::ValueBucket(dv);
  if (bd == 128 && bv == 128) return LaunchUmmaDq<E, 128, 128>(lp, /*stream=*/nullptr);
  if (bd == 256 && bv == 256) return LaunchUmmaDq<E, 256, 256>(lp, /*stream=*/nullptr);
  return false;
}

template <class E>
static int RunCase(const Case& c, const char* tag) {
  const int64_t B = c.batch, Nq = c.nq, Nk = c.nk, P = c.prev, Nt = c.total();
  const int64_t Hq = c.hq, Hkv = c.hkv, D = c.d, Dv = c.dv, C = c.chunk;
  const float scale = static_cast<float>(c.scale_mult / std::sqrt(static_cast<double>(D)));
  const float sl2 = static_cast<float>(static_cast<double>(scale) * 1.4426950408889634);
  const Element el = (sizeof(E) == 2 && std::is_same<E, __half>::value) ? Element::kHalf
                                                                        : Element::kBFloat16;

  // ---- host truth, in double ------------------------------------------------
  std::mt19937_64 rng(20261104u + static_cast<unsigned long long>(B * 1000 + Nq + D));
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> qh(static_cast<size_t>(B * Nq * Hq * D));
  std::vector<double> kh(static_cast<size_t>(B * Nt * Hkv * D));
  std::vector<double> vh(static_cast<size_t>(B * Nt * Hkv * Dv));
  std::vector<double> dyh(static_cast<size_t>(B * Nq * Hq * Dv));
  for (auto& x : qh) x = nd(rng);
  for (auto& x : kh) x = nd(rng);
  for (auto& x : vh) x = 0.1 + 0.5 * nd(rng);
  for (auto& x : dyh) x = nd(rng);
  // unit-norm Q and K rows, as the oracle's conditioning does
  for (int64_t b = 0; b < B; ++b)
    for (int64_t i = 0; i < Nq; ++i)
      for (int64_t h = 0; h < Hq; ++h) {
        double* r = &qh[(((b * Nq + i) * Hq + h) * D)];
        double n = 0;
        for (int64_t d = 0; d < D; ++d) n += r[d] * r[d];
        n = std::sqrt(n);
        for (int64_t d = 0; d < D; ++d) r[d] /= n;
      }
  for (int64_t b = 0; b < B; ++b)
    for (int64_t j = 0; j < Nt; ++j)
      for (int64_t h = 0; h < Hkv; ++h) {
        double* r = &kh[(((b * Nt + j) * Hkv + h) * D)];
        double n = 0;
        for (int64_t d = 0; d < D; ++d) n += r[d] * r[d];
        n = std::sqrt(n);
        for (int64_t d = 0; d < D; ++d) r[d] /= n;
      }

  // metadata
  std::vector<int32_t> row_lo(static_cast<size_t>(B * Nq), 0), row_hi(static_cast<size_t>(B * Nq), 0);
  std::vector<int64_t> qseg(static_cast<size_t>(B * Nq), 0), kseg(static_cast<size_t>(B * Nt), 0);
  std::vector<int32_t> sexact(static_cast<size_t>(B), 0);
  for (int64_t b = 0; b < B; ++b) {
    if (c.segments) {
      const int64_t half = Nt / 2;  // two segments, split at the key midpoint
      for (int64_t i = 0; i < Nq; ++i) qseg[b * Nq + i] = (i + Nt - Nq >= half) ? 1 : 0;
      for (int64_t j = 0; j < Nt; ++j) kseg[b * Nt + j] = (j >= half) ? 1 : 0;
    }
    for (int64_t i = 0; i < Nq; ++i) {
      int64_t lo, hi;
      if (c.explicit_bounds) {
        if (c.empty_row && b == 0 && i == 5) {
          row_lo[i] = 10;
          row_hi[i] = 3;  // lo > hi: an empty row by bounds
          continue;
        }
        // the analytic interval, materialised (chunk mask on the global axis)
        const int64_t pos = i + Nt - Nq;
        lo = (pos / C - 1) * C;
        hi = pos;
        if (c.segments) {  // keep the previous chunk only inside the segment
          const int64_t seg_start = qseg[b * Nq + i] == 0 ? 0 : Nt / 2;
          if (lo < seg_start) lo = seg_start;
        }
      } else {
        const int64_t pos = i + Nt - Nq;
        lo = (pos / C - 1) * C;
        hi = pos;
      }
      lo = lo < 0 ? 0 : lo;
      hi = hi >= Nt ? Nt - 1 : hi;
      row_lo[b * Nq + i] = static_cast<int32_t>(lo);
      row_hi[b * Nq + i] = static_cast<int32_t>(hi);
    }
  }

  std::vector<double> ref_dq(static_cast<size_t>(B * Nq * Hq * D), 0.0);
  int64_t delta_corrupt = 0;  // how many (b,h,i) rows the delta fault arm hit
  std::vector<float> lse(static_cast<size_t>(B * Hq * Nq), 0.f);
  std::vector<float> delta(static_cast<size_t>(B * Hq * Nq), 0.f);
  std::vector<double> yv(static_cast<size_t>(B * Nq * Hq * Dv), 0.0);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < Hq; ++h)
      for (int64_t i = 0; i < Nq; ++i) {
        const int64_t kvh = c.group() > 0 ? h / c.group() : h;
        const double* qr = &qh[(((b * Nq + i) * Hq + h) * D)];
        int64_t lo = c.explicit_bounds ? row_lo[b * Nq + i] : ((i + Nt - Nq) / C - 1) * C;
        int64_t hi = c.explicit_bounds ? row_hi[b * Nq + i] : (i + Nt - Nq);
        lo = lo < 0 ? 0 : lo;
        hi = hi >= Nt ? Nt - 1 : hi;
        std::vector<double> sraw;
        std::vector<int64_t> jj;
        for (int64_t j = lo; j <= hi; ++j) {
          if (c.segments && kseg[b * Nt + j] != qseg[b * Nq + i]) continue;
          const double* kr = &kh[(((b * Nt + j) * Hkv + kvh) * D)];
          double s = 0;
          for (int64_t d = 0; d < D; ++d) s += qr[d] * kr[d];
          sraw.push_back(scale * s);
          jj.push_back(j);
        }
        const int64_t idx = (b * Hq + h) * Nq + i;
        if (sraw.empty()) {
          lse[idx] = -INFINITY;
          delta[idx] = 0.f;
          continue;  // dQ row stays exactly 0
        }
        double m = -INFINITY;
        for (double s : sraw) m = std::max(m, s);
        double sum = 0;
        for (double s : sraw) sum += std::exp(s - m);
        const double l = m + std::log(sum);
        lse[idx] = static_cast<float>(l);
        std::vector<double> P(sraw.size());
        for (size_t t = 0; t < sraw.size(); ++t) P[t] = std::exp(sraw[t] - l);
        const double* dyr = &dyh[(((b * Nq + i) * Hq + h) * Dv)];
        std::vector<double> dp(P.size(), 0.0);
        double d = 0;
        for (size_t t = 0; t < P.size(); ++t) {
          const double* vr = &vh[(((b * Nt + jj[t]) * Hkv + kvh) * Dv)];
          double acc = 0;
          for (int64_t e = 0; e < Dv; ++e) acc += dyr[e] * vr[e];
          dp[t] = acc;
          d += P[t] * acc;
          double* yr = &yv[(((b * Nq + i) * Hq + h) * Dv)];
          for (int64_t e = 0; e < Dv; ++e) yr[e] += P[t] * vr[e];
        }
        // ---- delta fault arms (arms 1-4) ----------------------------------
        // delta is a formal INPUT of this reference: the model consumes the plane
        // the kernel is handed instead of recomputing it, so corrupting the plane
        // corrupts both sides identically and the comparison stays quantitative.
        // Arms 3 and 4 corrupt ONE row (of every head, or of one head): every other
        // row must stay bit-identical to the clean run, which is the only way to
        // see a delta read at the wrong row or the wrong head of the group.
        if (c.fault >= 1 && c.fault <= 4) {
          const bool plane = (c.fault == 1 || c.fault == 2);
          const bool row = (c.fault == 3) && b == 0 && i == c.fault_row;
          const bool row_head = (c.fault == 4) && b == 0 && i == c.fault_row && h == c.fault_head;
          if (plane || row || row_head) {
            d = (c.fault == 2) ? static_cast<double>(c.fault_finite) : NAN;
            ++delta_corrupt;
          }
        }
        delta[idx] = static_cast<float>(d);
        double* out = &ref_dq[(((b * Nq + i) * Hq + h) * D)];
        for (size_t t = 0; t < P.size(); ++t) {
          const double dS = P[t] * (dp[t] - d);
          const double* kr = &kh[(((b * Nt + jj[t]) * Hkv + kvh) * D)];
          for (int64_t e = 0; e < D; ++e) out[e] += dS * kr[e];
        }
        for (int64_t e = 0; e < D; ++e) out[e] *= scale;  // the store's single scale
      }

  // ---- input manifest: the evidence row names its exact inputs -----------------
  // A dQ number is only evidence if the tensors that produced it are identified,
  // so every case prints a hash of the four host inputs it generated. FNV-1a over
  // the float32 values (the dtype the device receives), not over the doubles.
  {
    auto fnv = [](const std::vector<double>& v) {
      uint64_t h = 1469598103934665603ULL;
      union { float f; uint32_t u; } cvt;  // host-side bit copy (no __float_as_uint on host)
      for (double x : v) {
        cvt.f = static_cast<float>(x);
        h = (h ^ cvt.u) * 1099511628211ULL;
      }
      return h;
    };
    std::printf("  inputs   q %016llx  k %016llx  v %016llx  dy %016llx\n",
                static_cast<unsigned long long>(fnv(qh)),
                static_cast<unsigned long long>(fnv(kh)),
                static_cast<unsigned long long>(fnv(vh)),
                static_cast<unsigned long long>(fnv(dyh)));
  }

  // ---- fault injection: non-finite LSE rows ------------------------------------
  // This ABI never produces a non-finite LSE on a live row: the forward writes
  // -INFINITY only for a row with no visible key, and the explicit-empty case is
  // that arm. Anything else is a state this leaf must survive without poisoning a
  // neighbour, so the contract here is one-directional and row-local: a row whose
  // LSE is +inf or NaN contributes nothing, and every other row of the packing is
  // bit-unchanged. The leaf maps every non-finite LSE to the +inf log2 offset
  // (DqLse2) and the store writes an exact 0 for such a row, so the reference for
  // those rows is exactly 0. Injecting into BOTH the host lse and the reference
  // keeps this arm inside the ordinary comparison, and putting the two bad rows
  // in the middle of the packing is what makes the second half of the claim (no
  // neighbour is poisoned) testable: the rows around them are still compared.
  if (c.lse_sentinel) {
    // The two planes order their axes differently: lse is [B, Hq, Nq] and dq is
    // [B, Nq, Hq, Dv]. The sentinel row must be the SAME (b, h, q) in both, so both
    // indices are derived from it. Copying one index across would zero the reference
    // for a different row than the leaf zeroes, which reads as an 85 % error on a
    // correct kernel.
    const int64_t bad_q[2] = {3, 5};
    const int64_t b = 0, h = 0;
    for (int64_t t = 0; t < 2; ++t) {
      const int64_t q = bad_q[t];
      lse[static_cast<size_t>((b * Hq + h) * Nq + q)] = (t == 0) ? INFINITY : NAN;
      const int64_t row = (b * Nq + q) * Hq + h;
      for (int64_t e = 0; e < D; ++e) ref_dq[static_cast<size_t>(row * D + e)] = 0.0;
    }
  }

  // Whole-plane arms (1, 2) also poison the delta slots of rows with NO visible key:
  // those slots are the caller's to fill with anything, and arm 3's whole point is
  // that such a row must still be written as exactly zero. The model ignores those
  // slots by construction, so this cannot move the reference - it only makes the
  // device's input hostile.
  if (c.fault == 1 || c.fault == 2) {
    for (int64_t b2 = 0; b2 < B; ++b2)
      for (int64_t h2 = 0; h2 < Hq; ++h2)
        for (int64_t i2 = 0; i2 < Nq; ++i2) {
          const bool empty = c.explicit_bounds && c.empty_row && b2 == 0 && i2 == 5;
          if (!empty) continue;
          delta[static_cast<size_t>((b2 * Hq + h2) * Nq + i2)] =
              (c.fault == 2) ? c.fault_finite : static_cast<float>(NAN);
        }
  }

  // ---- device buffers ------------------------------------------------------
  auto alloc = [](size_t nbytes) {
    void* p = nullptr;
    CK(cudaMalloc(&p, nbytes));
    return p;
  };
  E* dq_dev = static_cast<E*>(alloc(sizeof(E) * B * Nq * Hq * D));
  E* dy_dev = static_cast<E*>(alloc(sizeof(E) * B * Nq * Hq * Dv));
  float* lse_dev = static_cast<float*>(alloc(sizeof(float) * B * Hq * Nq));
  float* delta_dev = static_cast<float*>(alloc(sizeof(float) * B * Hq * Nq));
  E* q_dev = static_cast<E*>(alloc(sizeof(E) * B * Nq * Hq * D));
  E* k_dev = static_cast<E*>(alloc(sizeof(E) * B * Nt * Hkv * D));
  E* v_dev = static_cast<E*>(alloc(sizeof(E) * B * Nt * Hkv * Dv));
  E* prevk_dev = nullptr;
  E* prevv_dev = nullptr;
  if (P > 0) {
    prevk_dev = static_cast<E*>(alloc(sizeof(E) * B * P * Hkv * D));
    prevv_dev = static_cast<E*>(alloc(sizeof(E) * B * P * Hkv * Dv));
  }
  E* ysave_dev = (!c.saved_fp32 && (c.null_delta || c.fault >= 5))
                     ? static_cast<E*>(alloc(sizeof(E) * B * Nq * Hq * Dv))
                     : nullptr;
  float* ystate_dev = c.saved_fp32
                          ? static_cast<float*>(alloc(sizeof(float) * B * Nq * Hq * Dv))
                          : nullptr;
  int32_t *rlo_dev = nullptr, *rhi_dev = nullptr;
  int64_t *qs_dev = nullptr, *ks_dev = nullptr;
  int32_t* sx_dev = nullptr;

  std::vector<E> tmp;
  auto upload = [&](E* dst, const std::vector<double>& src) {
    tmp.resize(src.size());
    for (size_t t = 0; t < src.size(); ++t) tmp[t] = FromF<E>(static_cast<float>(src[t]));
    CK(cudaMemcpy(dst, tmp.data(), sizeof(E) * src.size(), cudaMemcpyHostToDevice));
  };
  auto upload_raw = [&](E* dst, const E* src, size_t n) {
    CK(cudaMemcpy(dst, src, sizeof(E) * n, cudaMemcpyHostToDevice));
  };
  upload(q_dev, qh);
  upload(k_dev, kh);
  upload(v_dev, vh);
  upload(dy_dev, dyh);
  if (P > 0) {  // split the [B,Nt,...] plane into previous and current halves
    std::vector<double> pk(static_cast<size_t>(B * P * Hkv * D)), pv(static_cast<size_t>(B * P * Hkv * Dv));
    std::vector<double> ck(static_cast<size_t>(B * Nk * Hkv * D)), cv(static_cast<size_t>(B * Nk * Hkv * Dv));
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t j = 0; j < P; ++j)
        for (int64_t h = 0; h < Hkv; ++h)
          for (int64_t d = 0; d < D; ++d)
            pk[(((b * P + j) * Hkv + h) * D) + d] = kh[(((b * Nt + j) * Hkv + h) * D) + d];
      for (int64_t j = 0; j < P; ++j)
        for (int64_t h = 0; h < Hkv; ++h)
          for (int64_t e = 0; e < Dv; ++e)
            pv[(((b * P + j) * Hkv + h) * Dv) + e] = vh[(((b * Nt + j) * Hkv + h) * Dv) + e];
      for (int64_t j = 0; j < Nk; ++j)
        for (int64_t h = 0; h < Hkv; ++h)
          for (int64_t d = 0; d < D; ++d)
            ck[(((b * Nk + j) * Hkv + h) * D) + d] = kh[(((b * Nt + P + j) * Hkv + h) * D) + d];
      for (int64_t j = 0; j < Nk; ++j)
        for (int64_t h = 0; h < Hkv; ++h)
          for (int64_t e = 0; e < Dv; ++e)
            cv[(((b * Nk + j) * Hkv + h) * Dv) + e] = vh[(((b * Nt + P + j) * Hkv + h) * Dv) + e];
    }
    upload(prevk_dev, pk);
    upload(prevv_dev, pv);
    upload(k_dev, ck);
    upload(v_dev, cv);
  }
  CK(cudaMemcpy(lse_dev, lse.data(), sizeof(float) * lse.size(), cudaMemcpyHostToDevice));
  if (!c.null_delta) {
    CK(cudaMemcpy(delta_dev, delta.data(), sizeof(float) * delta.size(), cudaMemcpyHostToDevice));
  } else {
    CK(cudaMemset(delta_dev, 0, sizeof(float) * delta.size()));  // must not be read
  }
  if (ysave_dev) {
    // The saved plane holds y, because the fallback identity the leaf uses when no
    // delta is supplied is delta = dY . y. Arms 5 and 6 poison it: with delta
    // SUPPLIED a correct leaf never reads this plane, so its dq must equal the
    // clean run's bit for bit; with delta NULL the same poisoning must move dq,
    // which is what makes arm 5 mean anything.
    std::vector<double> ysave = yv;
    if (c.fault >= 5) {
      int64_t poisoned = 0;
      for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < Nq; ++i)
          for (int64_t h = 0; h < Hq; ++h) {
            if (b != 0 || i != c.fault_row) continue;
            if (c.fault == 5 && h != c.fault_head) continue;
            for (int64_t e = 0; e < Dv; ++e)
              ysave[static_cast<size_t>(((b * Nq + i) * Hq + h) * Dv + e)] = NAN;
            ++poisoned;
          }
      std::printf("  fault    saved-plane NaN on %lld (b,i,h) rows\n",
                  static_cast<long long>(poisoned));
    }
    tmp.resize(ysave.size());
    for (size_t t = 0; t < ysave.size(); ++t) tmp[t] = FromF<E>(static_cast<float>(ysave[t]));
    upload_raw(ysave_dev, tmp.data(), ysave.size());
  }
  if (ystate_dev) {
    // Level 1: the same y values, but in the forward's FP32 state plane, which is what
    // makes the leaf's saved_fp32 branch observable at all - a 16-bit reinterpretation
    // of this plane would read half-rows of garbage, and nothing else in the probe
    // reaches that branch.
    std::vector<float> ystate(static_cast<size_t>(B * Nq * Hq * Dv));
    for (size_t t = 0; t < ystate.size(); ++t) ystate[t] = static_cast<float>(yv[t]);
    CK(cudaMemcpy(ystate_dev, ystate.data(), sizeof(float) * ystate.size(),
                  cudaMemcpyHostToDevice));
  }
  if (c.explicit_bounds) {
    rlo_dev = static_cast<int32_t*>(alloc(sizeof(int32_t) * row_lo.size()));
    rhi_dev = static_cast<int32_t*>(alloc(sizeof(int32_t) * row_hi.size()));
    CK(cudaMemcpy(rlo_dev, row_lo.data(), sizeof(int32_t) * row_lo.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rhi_dev, row_hi.data(), sizeof(int32_t) * row_hi.size(), cudaMemcpyHostToDevice));
  }
  if (c.segments) {
    qs_dev = static_cast<int64_t*>(alloc(sizeof(int64_t) * qseg.size()));
    ks_dev = static_cast<int64_t*>(alloc(sizeof(int64_t) * kseg.size()));
    sx_dev = static_cast<int32_t*>(alloc(sizeof(int32_t) * sexact.size()));
    CK(cudaMemcpy(qs_dev, qseg.data(), sizeof(int64_t) * qseg.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ks_dev, kseg.data(), sizeof(int64_t) * kseg.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sx_dev, sexact.data(), sizeof(int32_t) * sexact.size(), cudaMemcpyHostToDevice));
  }
  // dq starts as garbage on purpose: the store must overwrite every element of
  // every live row. The default 0xEE byte pattern is the cheapest "not a number a
  // kernel would produce" fill; the two non-default fills sharpen it:
  // 0x7FC00000 (quiet NaN) catches an unwritten accumulator window that
  // a wrong `accumulate` flag at round 0 would otherwise leave plausible, and a
  // large finite value catches a clamp that hides such a read.
  {
    std::vector<E> fill(static_cast<size_t>(B * Nq * Hq * D));
    if (c.sentinel_fill == 0) {
      CK(cudaMemset(dq_dev, 0xEE, sizeof(E) * fill.size()));
    } else if (c.sentinel_fill == 1) {
      for (auto& x : fill) x = FromF<E>(NAN);
      CK(cudaMemcpy(dq_dev, fill.data(), sizeof(E) * fill.size(), cudaMemcpyHostToDevice));
    } else {
      for (auto& x : fill) x = FromF<E>(1.0e4f);
      CK(cudaMemcpy(dq_dev, fill.data(), sizeof(E) * fill.size(), cudaMemcpyHostToDevice));
    }
  }

  // ---- params --------------------------------------------------------------
  UmmaLaunchParams lp{};
  Params& p = lp.params;
  p.element = el;
  p.q = Operand{q_dev, Nq * Hq * D, Hq * D, D};
  p.k = Operand{k_dev, Nk * Hkv * D, Hkv * D, D};
  p.v = Operand{v_dev, Nk * Hkv * Dv, Hkv * Dv, Dv};
  if (P > 0) {
    p.prev_k = Operand{prevk_dev, P * Hkv * D, Hkv * D, D};
    p.prev_v = Operand{prevv_dev, P * Hkv * Dv, Hkv * Dv, Dv};
  }
  p.batch = B;
  p.q_length = Nq;
  p.k_length = Nk;
  p.prev_length = P;
  p.total_length = Nt;
  p.query_heads = Hq;
  p.kv_heads = Hkv;
  p.group = c.group();
  p.qk_dim = D;
  p.value_dim = Dv;
  p.chunk = C;
  p.scale = scale;
  p.scale_log2 = sl2;
  p.strict = false;
  p.row_lo = rlo_dev;
  p.row_hi = rhi_dev;
  p.q_segment = qs_dev;
  p.k_segment = ks_dev;
  p.segment_exact = sx_dev;
  p.dy = dy_dev;
  p.lse_in = lse_dev;
  p.delta = c.null_delta ? nullptr : delta_dev;
  p.saved = (!c.saved_fp32 && (c.null_delta || c.fault >= 5)) ? ysave_dev : nullptr;
  p.saved_fp32 = c.saved_fp32;  // level 0: the saved plane is the input dtype
  p.y_state = ystate_dev;       // level 1: the FP32 state plane, read only on the fallback
  p.dq = dq_dev;
  // The det arm: this leaf owns dq. The one injected refusal (the fail-path fixture,
  // see `g_inject_refusal`) hands the predicate a non-null accumulator, which is
  // exactly the call this leaf must refuse.
  p.dq_accum = g_inject_refusal ? &g_refusal_sentinel : nullptr;

  UmmaInputMaps& m = lp.maps;
  m.q = Encode(p.q, el, B, Nq, Hq, D);
  m.k = Encode(p.k, el, B, Nk, Hkv, D);
  m.v = Encode(p.v, el, B, Nk, Hkv, Dv);
  if (P > 0) {
    m.prev_k = Encode(p.prev_k, el, B, P, Hkv, D);
    m.prev_v = Encode(p.prev_v, el, B, P, Hkv, Dv);
  }
  m.dy = Encode(Operand{dy_dev, Nq * Hq * Dv, Hq * Dv, Dv}, el, B, Nq, Hq, Dv);
  m.packed_kv = false;

  // ---- preflight: the accepted width is bounded by the INSTANTIATION ----------
  // Bucket equality alone is not a bound: QkBucket saturates at 256, so 2048 has the
  // same bucket as 200, and a predicate that compared buckets only would hand a
  // 2048-wide row to a 256-wide writer that writes the first 256 columns and leaves
  // the rest of every row untouched. Every mutation below must get the verdict in
  // `accept`, and because a refusal is decided before anything is issued, the dq
  // allocation must be byte-identical across the whole block. One block per
  // (element type, bucket): the predicate is the same code for every case of that
  // bucket, so repeating it per case would add lines and evidence of nothing.
  //
  // `CanLaunchUmmaDq` returns a bare `bool` and reports no reason, so this block pins
  // the verdict, not the branch that produced it. At the 256 template `qk_dim = 257`
  // and `qk_dim = 2048` are cells the BUCKET test alone would accept - that is where
  // the order of the width bound and the bucket test becomes observable. At the 128
  // template `kD + 1 = 129` buckets to 256, so there the order is source-reading only.
  // An accepted padded width that writes nothing cannot be caught here (no launch); it
  // is caught by the `padded-dims` and `padded-prev-*` arms, which launch at padded
  // widths and compare every output element against the FP64 model.
  {
    const int bd = xattn::ops::flash_sca_native::QkBucket(D);
    const int bv = xattn::ops::flash_sca_native::ValueBucket(Dv);
    static bool done_128 = false;
    static bool done_256 = false;
    bool* done = nullptr;
    if (bd == 128 && bv == 128) {
      done = &done_128;
    } else if (bd == 256 && bv == 256) {
      done = &done_256;
    }
    if (done != nullptr && !*done) {
      *done = true;
      auto hash_prefix = [&]() {
        const size_t nbytes = sizeof(E) * static_cast<size_t>(B * Nq * Hq * D);
        const size_t take = nbytes < 8192 ? nbytes : 8192;
        std::vector<unsigned char> buf(take);
        CK(cudaMemcpy(buf.data(), dq_dev, take, cudaMemcpyDeviceToHost));
        uint64_t h = 1469598103934665603ULL;
        for (unsigned char ch : buf) h = (h ^ ch) * 1099511628211ULL;
        return h;
      };
      const uint64_t pref_before = hash_prefix();
      struct Mut {
        int64_t qk;
        int64_t vd;
        bool accept;
        const char* what;
      };
      const Mut muts[] = {
          {D, Dv, true, "unchanged"},
          {0, Dv, false, "qk_dim = 0"},
          {-8, Dv, false, "qk_dim < 0"},
          {static_cast<int64_t>(bd), Dv, true, "qk_dim at the bucket width"},
          {static_cast<int64_t>(bd) + 1, Dv, false, "qk_dim one past the bucket"},
          {2048, Dv, false, "qk_dim = 2048 (bucket 256, like 200)"},
          {4096, Dv, false, "qk_dim = 4096"},
          {D, 0, false, "value_dim = 0"},
          {D, -8, false, "value_dim < 0"},
          {D, static_cast<int64_t>(bv) + 1, false, "value_dim one past the bucket"},
          {D, 2048, false, "value_dim = 2048 (bucket 256, like 200)"},
      };
      for (const Mut& mu : muts) {
        UmmaLaunchParams lp2 = lp;
        lp2.params.qk_dim = mu.qk;
        lp2.params.value_dim = mu.vd;
        const bool got = (bd == 128 && bv == 128)
                             ? xattn::ops::flash_sca_native::umma::CanLaunchUmmaDq<E, 128, 128>(lp2)
                             : xattn::ops::flash_sca_native::umma::CanLaunchUmmaDq<E, 256, 256>(lp2);
        const bool ok = (got == mu.accept);
        if (!ok) ++g_preflight_fail;
        std::printf("  %-4s preflight %-34s D=%5lld Dv=%5lld -> %-6s %s\n", tag, mu.what,
                    static_cast<long long>(mu.qk), static_cast<long long>(mu.vd),
                    got ? "accept" : "refuse", ok ? "PASS" : "FAIL");
      }
      // The accept half of the fixture. Every cell above is a refusal except the two
      // that restate the original widths, so on its own this block cannot tell a
      // correct bound from a leaf that refuses every width - and an over-strict
      // predicate is the worse failure, because it deletes the native arm from
      // selection and looks like a fallback nobody questions. 96 is inside bucket 128
      // and 192 inside bucket 256, and neither is the template's own width, so each
      // template must accept a width it does not write.
      {
        const int64_t inner = (bd == 128 && bv == 128) ? 96 : 192;
        UmmaLaunchParams lp2 = lp;
        lp2.params.qk_dim = inner;
        lp2.params.value_dim = inner;
        const bool got = (bd == 128 && bv == 128)
                             ? xattn::ops::flash_sca_native::umma::CanLaunchUmmaDq<E, 128, 128>(lp2)
                             : xattn::ops::flash_sca_native::umma::CanLaunchUmmaDq<E, 256, 256>(lp2);
        if (!got) ++g_preflight_fail;
        std::printf("  %-4s preflight %-34s D=%5lld Dv=%5lld -> %-6s %s\n", tag,
                    "in bucket, not the width", static_cast<long long>(inner),
                    static_cast<long long>(inner), got ? "accept" : "refuse",
                    got ? "PASS" : "FAIL");
      }
      // The domain cells that the predicate's own "fail-closed on its own terms"
      // claim owes, none of which the public API is entitled to be trusted for: a
      // non-positive `chunk` is a device integer division by zero (visibility.cuh:32
      // divides by it); a broken Nt = P + Nk identity and a negative or oversized
      // `prev_length` are silent wrong answers (the kernel then reads the previous
      // plane for every tile, or shifts every current key row); a lone `q_segment` is
      // a device read of a null key plane. The predicate compares pointers only and
      // never dereferences them, so an unaligned non-null sentinel is enough for the
      // pairing cell. `chunk = 1` is the contract's minimum and must still ACCEPT:
      // the refusal classes above are refusals of a broken call, not of a small one.
      {
        // which: 0 chunk, 1 prev_length, 2 k_length, 3 segment pairing.
        struct Mut2 {
          int which;
          int64_t value;
          bool accept;
          const char* what;
        };
        const Mut2 m2[] = {
            {0, 0, false, "chunk = 0"},
            {0, -1, false, "chunk < 0"},
            {0, 1, true, "chunk = 1 (contract minimum)"},
            {1, -1, false, "prev_length < 0"},
            {1, lp.params.total_length + 1, false, "prev_length > total_length"},
            {2, 1, false, "k_length + 1 (Nt != P + Nk)"},
            {3, 0, false, "q_segment without k_segment"},
        };
        for (const Mut2& mu : m2) {
          UmmaLaunchParams lp2 = lp;
          if (mu.which == 0) lp2.params.chunk = mu.value;
          if (mu.which == 1) lp2.params.prev_length = mu.value;
          if (mu.which == 2) lp2.params.k_length = lp.params.k_length + mu.value;
          if (mu.which == 3) {
            lp2.params.q_segment = reinterpret_cast<const int64_t*>(0x10);
            lp2.params.k_segment = nullptr;
          }
          const bool got = (bd == 128 && bv == 128)
                               ? xattn::ops::flash_sca_native::umma::CanLaunchUmmaDq<E, 128, 128>(lp2)
                               : xattn::ops::flash_sca_native::umma::CanLaunchUmmaDq<E, 256, 256>(lp2);
          const bool ok2 = (got == mu.accept);
          if (!ok2) ++g_preflight_fail;
          std::printf("  %-4s preflight %-34s              -> %-6s %s\n", tag, mu.what,
                      got ? "accept" : "refuse", ok2 ? "PASS" : "FAIL");
        }
      }
      const uint64_t pref_after = hash_prefix();
      const bool same = (pref_before == pref_after);
      if (!same) ++g_preflight_fail;
      std::printf("  %-4s preflight dq bytes across the block: %016llx vs %016llx  %s\n", tag,
                  static_cast<unsigned long long>(pref_before),
                  static_cast<unsigned long long>(pref_after), same ? "PASS (unchanged)" : "FAIL");
    }
  }

  // ---- launch --------------------------------------------------------------
  const bool launched = LaunchFor<E>(lp, D, Dv);
  if (!launched) {
    std::printf("  %-4s %-16s REFUSED by the launcher\n", tag, c.name);
    return 2;  // distinct from 1 (the call ran and disagreed with the reference)
  }
  CK(cudaDeviceSynchronize());

  std::vector<E> got(static_cast<size_t>(B * Nq * Hq * D));
  CK(cudaMemcpy(got.data(), dq_dev, sizeof(E) * got.size(), cudaMemcpyDeviceToHost));
  // A bitwise claim needs a bitwise fingerprint. FNV-1a over the raw element bits,
  // printed on every row of the table: two cases of the same shape can then be
  // compared exactly instead of through four printed digits of an aggregate. Every
  // bitwise claim in this table is a row of `MustEqualRow` - the fault arms against
  // their clean twin, and the dirty-TMEM repeat against its own first launch - and
  // each of those rows compares the two digests instead of restating the claim in
  // prose.
  uint64_t dq_hash = 1469598103934665603ULL;
  for (size_t t = 0; t < got.size(); ++t) {
    const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(&got[t]);
    for (size_t q = 0; q < sizeof(E); ++q) dq_hash = (dq_hash ^ bytes[q]) * 1099511628211ULL;
  }
  DigestPut(tag, c.name, dq_hash);  // for the arms that claim another row's bits

  double max_abs = 0, peak = 0, sumsq = 0;
  int64_t n = 0;
  int64_t nan_both = 0, nan_mismatch = 0;
  for (int64_t b = 0; b < B; ++b)
    for (int64_t i = 0; i < Nq; ++i)
      for (int64_t h = 0; h < Hq; ++h)
        for (int64_t d = 0; d < D; ++d) {
          const int64_t k = (((b * Nq + i) * Hq + h) * D) + d;
          const double r = ref_dq[k];
          const double g = ToF(got[k]);
          // The corrupted-input arms produce NaN on purpose, and a NaN pair is
          // AGREEMENT, not an error: it is skipped from the magnitude statistics and
          // counted, while a NaN on one side only is a contradiction and fails.
          if (std::isnan(r) || std::isnan(g)) {
            if (std::isnan(r) != std::isnan(g)) ++nan_mismatch;
            else ++nan_both;
            continue;
          }
          max_abs = std::max(max_abs, std::fabs(g - r));
          peak = std::max(peak, std::fabs(r));
          sumsq += (g - r) * (g - r);
          ++n;
        }
  // Arm 3: a row with no visible key is "no gradient contribution", which is a claim
  // about BITS and not about a tolerance - so it is checked on the bits, sign of zero
  // included (a -0.0 is a different pattern from the +0.0 the leaf writes). This is
  // the arm that must hold while delta is corrupted all around it.
  int64_t zero_nonzero = 0, zero_nan = 0;
  if (c.exact_zero_row && c.zero_row >= 0 && c.zero_row < Nq) {
    for (int64_t h = 0; h < Hq; ++h)
      for (int64_t d = 0; d < D; ++d) {
        union { E e; uint16_t u; } cv;
        cv.e = got[static_cast<size_t>(((0 * Nq + c.zero_row) * Hq + h) * D + d)];
        if (std::isnan(ToF(cv.e))) {
          ++zero_nan;
          continue;
        }
        if (cv.u != 0u) ++zero_nonzero;
      }
  }
  const double rms = n > 0 ? std::sqrt(sumsq / static_cast<double>(n)) : 0.0;
  const double den = peak > 0 ? peak : 1.0;
  const bool within = (nan_mismatch == 0) && zero_nonzero == 0 && zero_nan == 0 &&
                      max_abs <= 0.05 * peak && rms <= 0.02 * peak;
  bool pass = within;
  const char* verdict = within ? "PASS" : "FAIL";
  if (c.expect_nan) {
    // Arms 1/3/4: delta is corrupted for this row on BOTH sides, so the required
    // result is "the same entries are NaN, and every finite entry still agrees" -
    // which is what makes a wrong-row or wrong-head read visible.
    pass = within && nan_both > 0;
    verdict = pass ? "PASS (NaN matched)" : "FAIL";
  } else if (c.expect_moved) {
    // Arm 6: the plane is poisoned and the reference has no such plane, so there is
    // no number to compare. The OLD criterion (`nan_both > 0`) was wrong: with delta
    // null the model forms delta from P.dP and never reads the saved plane, so ref_dq
    // stays finite while a correct leaf - which DOES read the poisoned plane - is NaN
    // on the device side only, i.e. exactly `nan_mismatch`. The arm therefore printed
    // UNMOVED for a correct kernel. The invariant is "the answer moved": one-sided
    // NaN, or a magnitude far past the tolerance. One-sided NaN is a PASS here and
    // only here.
    pass = nan_mismatch > 0 || max_abs > 0.05 * peak;
    verdict = pass ? "MOVED (expected: one-sided NaN or moved magnitude)" : "UNMOVED (bug)";
  } else if (const char* twin = MustEqualRow(c.name)) {
    // A BITWISE claim against another row of the same dtype: compare the digests,
    // do not restate it in prose. `within` still has to hold, so a row that is
    // "equal" the wrong way (both wrong) cannot pass on the digest alone.
    uint64_t want = 0;
    const bool have = DigestGet(tag, twin, &want);
    const bool same = have && want == dq_hash;
    pass = have && same && within;
    verdict = !have ? "FAIL (no such row)" : (same ? "PASS (bit-identical)" : "FAIL (bits differ)");
    std::printf("        digest %016llx  vs  %s %016llx\n",
                static_cast<unsigned long long>(dq_hash), twin,
                static_cast<unsigned long long>(want));
  }
  std::printf("  %-4s %-16s peak %.4g  max_abs %.4g (%.3f%%)  rms %.4g (%.3f%%)  nan %lld/%lld  %s\n",
              tag, c.name, peak, max_abs, 100.0 * max_abs / den, rms, 100.0 * rms / den,
              static_cast<long long>(nan_both), static_cast<long long>(nan_mismatch), verdict);
  if (c.exact_zero_row)
    std::printf("        exact-zero row %lld: %lld non-zero bits, %lld NaN bits\n",
                static_cast<long long>(c.zero_row), static_cast<long long>(zero_nonzero),
                static_cast<long long>(zero_nan));
  if (delta_corrupt > 0) std::printf("        delta corrupted on %lld (b,h,i) rows\n",
                                     static_cast<long long>(delta_corrupt));

  CK(cudaFree(dq_dev)); CK(cudaFree(dy_dev)); CK(cudaFree(lse_dev)); CK(cudaFree(delta_dev));
  CK(cudaFree(q_dev)); CK(cudaFree(k_dev)); CK(cudaFree(v_dev));
  if (prevk_dev) CK(cudaFree(prevk_dev));
  if (prevv_dev) CK(cudaFree(prevv_dev));
  if (ysave_dev) CK(cudaFree(ysave_dev));
  if (ystate_dev) CK(cudaFree(ystate_dev));
  if (rlo_dev) CK(cudaFree(rlo_dev));
  if (rhi_dev) CK(cudaFree(rhi_dev));
  if (qs_dev) CK(cudaFree(qs_dev));
  if (ks_dev) CK(cudaFree(ks_dev));
  if (sx_dev) CK(cudaFree(sx_dev));
  return pass ? 0 : 1;
}

int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  std::printf("device %s  cc %d.%d  smem/SM %zu\n", prop.name, prop.major, prop.minor,
              static_cast<size_t>(prop.sharedMemPerMultiprocessor));
  int fails = 0, refused = 0, arms = 0;
  std::setbuf(stdout, nullptr);  // unbuffered: if a case crashes, the lines before it survive
  // A: analytic, GQA group 2, C = N/4, every non-diagonal key tile full.
  // B: unequal + previous K/V (one whole prev tile) + a partial query tile.
  // C: explicit bounds with an empty row, and a scale that is not 1/sqrt(D).
  // D: explicit segment planes (ARM B), keys of the other segment invisible.
  // E: null delta -> the dY.y fallback through the saved plane.
  const float kOne = 1.f;
  Case cases[] = {
      {"analytic-gqa2", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne},
      {"prev-unequal", 2, 192, 128, 64, 2, 1, 256, 256, 64, false, false, false, false, kOne},
      {"explicit-empty", 1, 257, 257, 0, 2, 2, 128, 128, 96, true, false, true, false, 1.7f},
      {"segments-armB", 1, 256, 256, 0, 2, 1, 128, 128, 64, true, true, false, false, kOne},
      {"fallback-dyy", 1, 192, 192, 0, 2, 1, 128, 128, 64, false, false, false, true, kOne},
      // F: the shape every other case misses - C = 96 is NOT a
      // multiple of the 64-key tile, so tiles are misaligned with the chunk start,
      // and a deep row then sees >= 3 key tiles whose FIRST tile is partial:
      //   p = 200 -> chunk 2 (192..287), visible keys 96..200 -> tiles 1,2,3, where
      //   tile 1 holds only keys 96..127 (TileIsFull false on the first tile) and the
      //   loop runs three tiles (a per-tile reset bug is invisible with one or two).
      // Both buckets: at 256/256 kStages drops to 1, a different pipeline, and the
      // same row must give the same answer.
      {"tiles3-partial-first", 1, 768, 768, 0, 2, 1, 128, 128, 96, false, false, false, false, kOne},
      {"tiles3-partial-first-d256", 1, 768, 768, 0, 2, 1, 256, 256, 96, false, false, false, false, kOne},
      // F2 (dirty TMEM): the same call as F, launched again immediately. The
      // second launch starts with every TMEM window still holding
      // the first launch's accumulators, so a reset that is missing, or keyed on the
      // wrong loop index, returns a different answer for identical inputs - and the
      // other direction matters too: identical inputs must give a bit-identical dQ
      // on the second launch, which is a determinism row for free.
      {"tiles3-partial-first-again", 1, 768, 768, 0, 2, 1, 128, 128, 96, false, false, false, false, kOne},
      // G1: signed scale — the 1.7/sqrt(D) multiplier with its sign flipped. The
      // contract keeps masking defined for any finite scale, and this leaf has
      // exactly one scale site (the store), so what a negative scale can expose is
      // a sign-dependent path in the mask or the softmax, not a rescale. Same shape
      // as analytic-gqa2, so the two rows are directly comparable.
      {"scale-negative", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, -1.7f},
      // G2: non-finite LSE rows (+inf and NaN, injected into both the host lse and
      // the reference - see the injection block). Both must give an exact-zero dQ
      // row and must leave the neighbouring rows bit-unchanged.
      {"lse-sentinel", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne, true},
      // H: padded logical dimensions. The host dispatches on a 64/128/256 bucket,
      // so a 160-wide model arrives at the 256 instantiation with p.qk_dim = 160:
      // the descriptor clips the columns past 160 to zero and the store is masked
      // at p.qk_dim, while the TMEM slabs and the atom layout stay 256 wide. This
      // is the only arm that runs the ACCEPT side of the bucket guard (160 -> 256)
      // together with the masked store; without it, "the leaf accepts padded dims"
      // would be a claim about a path nothing executed. It is last on purpose: the
      // host buffer is allocated 160 wide, so a store that ignored p.qk_dim would
      // be an out-of-bounds write here, and a crash must not cost the earlier rows.
      {"padded-dims", 1, 192, 192, 0, 2, 1, 160, 160, 64, false, false, false, false, kOne, false, 0, 0, 0,
       1.0e4f, false, false, 0, false, 0, false},
      // P1/P2/P3: the padded dimension with the SPLIT map exercised, and the
      // slab-multiple axis separated from the bucket-padding axis. kSlab is 64
      // columns, so a logical dim inside a bucket can be a whole number of slabs or
      // not, and the two questions are different: a TMA box that reads only the real
      // columns (descriptor zero fill) versus a K-slab walk whose last slab is half
      // real. `prev = 64` is exactly one whole previous key tile, which is the shape
      // that makes ResolveKeyMap/tma_row0 compute their own row offset instead of
      // falling out of the dense walk; `nq = 129` leaves the last query tile a single
      // row. P2 (192 = 3 slabs of 64) is the CONTROL for P1 (160 = 2.5): if P2 passes
      // and P1 fails the defect is in slab-multiple handling, not in padding.
      {"padded-prev-160", 1, 129, 129, 64, 2, 1, 160, 160, 64, false, false, false, false, kOne},
      {"padded-prev-192", 1, 129, 129, 64, 2, 1, 192, 192, 64, false, false, false, false, kOne},
      {"padded-prev-96", 1, 129, 129, 64, 2, 1, 96, 96, 64, false, false, false, false, kOne},
      // ---- round-count arm (why this case exists) ---------------------------
      // Every shape above is a CORRECTNESS case and its round count is 2-4, so the
      // round-cycle instrument (`DQ_ROUND_STAT`, the `> 2 * running_min` test) has no
      // power on any of them: with nround = 2 or 3 it can flag at most once or twice,
      // and an nflag = 0 from it is a VOID field rather than a zero. The number of
      // rounds is the number of key tiles a query tile walks, i.e. about
      // (kBM + C) / kBN with kBM = 128 and kBN = 64, so C = 1024 buys
      // (128 + 1024) / 64 = 18 rounds and C = 2048 buys (128 + 2048) / 64 = 34.
      // C = 1024 gives 18 rounds - correct, passing, and below the 32-round
      // power gate of the instrument's protocol, so such a row is VOID and
      // not a zero. This row is C = 2048 to clear that gate.
      // It is a NORMAL reference-checked row like every other one - not an exempt
      // timing case - because a case with no reference is a hole somebody later reads
      // as a pass, and this probe's whole value is that every row is checked.
      {"steps-many-c2048", 1, 2304, 2304, 0, 2, 1, 128, 128, 2048, false, false, false, false, kOne},
      // ---- delta / saved-plane fault arms --------------------------------
      // All of these reuse one of the shapes above on purpose, so every row is
      // directly comparable to a clean row: an arm that changes one input and prints
      // the SAME numbers is making a bitwise claim, and an arm that must move is
      // visible against a known row. Field order after `lse_sentinel`: fault,
      // fault_row, fault_head, fault_finite, expect_nan, expect_moved, sentinel_fill.
      // 1) the quantitative form: a large FINITE delta, so the model and the device
      //    are given the same bad number and the row still has a measurable error.
      {"delta-finite", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne, false, 2},
      // 2) the whole delta plane non-finite: the same entries must be NaN on both
      //    sides and every finite entry must still agree.
      {"delta-nan", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne, false, 1,
       0, 0, 1.0e4f, true},
      // 3) one row of every head, 4) one row of one head: the index/head-of-group
      //    discriminator - every other row and head must stay correct.
      {"delta-nan-row", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne, false, 3,
       7, 0, 1.0e4f, true},
      {"delta-nan-row-head", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne, false,
       4, 9, 1, 1.0e4f, true},
      // 5) the saved plane poisoned while delta IS supplied: a correct leaf never
      //    reads it, so this row must print exactly the numbers of analytic-gqa2.
      {"saved-poison-supplied", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne,
       false, 5, 9, 1},
      // 6) the same poisoning with delta NULL: the fallback does read that plane, so
      //    dq MUST move - the positive control that gives arm 5 its meaning.
      {"saved-poison-null-delta", 1, 192, 192, 0, 2, 1, 128, 128, 64, false, false, false, true, kOne,
       false, 6, 9, 0, 1.0e4f, false, true},
      // 7) skip sensitivity, at the two rows whose window straddles a tile boundary in
      //    the C=96 packing: the FIRST row of a chunk (window = the tail of the
      //    previous chunk plus its own chunk head) and the LAST row of a chunk (window
      //    = its own chunk tail). A tile wrongly skipped makes its rows insensitive to
      //    delta, which no whole-plane corruption can see.
      {"delta-nan-chunkfirst", 1, 768, 768, 0, 2, 1, 128, 128, 96, false, false, false, false, kOne,
       false, 4, 96, 0, 1.0e4f, true},
      {"delta-nan-chunklast", 1, 768, 768, 0, 2, 1, 128, 128, 96, false, false, false, false, kOne,
       false, 4, 95, 0, 1.0e4f, true},
      // 8) allocation fills: an unwritten accumulator window must be impossible to
      //    read silently, so the garbage is a quiet NaN in one run and a large finite
      //    value in the other. Both must reproduce analytic-gqa2's numbers exactly.
      {"fill-nan", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne, false, 0, 0, 0,
       1.0e4f, false, false, 1},
      {"fill-big", 1, 256, 256, 0, 2, 1, 128, 128, 64, false, false, false, false, kOne, false, 0, 0, 0,
       1.0e4f, false, false, 2},
      // 3) the empty row under a corrupted delta: the row has no visible key, so its
      //    dq must be written as EXACTLY zero with no NaN bit pattern while every row
      //    around it is saturated with the corruption. Two forms, because a caller
      //    sends a finite delta and the non-finite one is the adversarial sibling.
      {"empty-row-delta-nan", 1, 257, 257, 0, 2, 2, 128, 128, 96, true, false, true, false, kOne,
       false, 1, 0, 0, 1.0e4f, true, false, 0, true, 5},
      {"empty-row-delta-finite", 1, 257, 257, 0, 2, 2, 128, 128, 96, true, false, true, false, kOne,
       false, 2, 0, 0, 1.0e4f, false, false, 0, true, 5},
      // The level-1 form of the same fallback: the saved plane is the forward's FP32
      // state, so the leaf's saved_fp32 branch is executed rather than assumed. Same
      // shape and seed as fallback-dyy, so the two rows are directly comparable and a
      // 16-bit reinterpretation of the FP32 plane cannot hide behind a tolerance.
      {"fallback-dyy-fp32state", 1, 192, 192, 0, 2, 1, 128, 128, 64, false, false, false, true, kOne,
       false, 0, 0, 0, 1.0e4f, false, false, 0, false, 0, true},
  };
  for (const Case& c : cases) {
    std::printf("case %s (B %lld Nq %lld Nk %lld P %lld Hq %lld Hkv %lld D %lld Dv %lld C %lld)\n",
                c.name, (long long)c.batch, (long long)c.nq, (long long)c.nk, (long long)c.prev,
                (long long)c.hq, (long long)c.hkv, (long long)c.d, (long long)c.dv, (long long)c.chunk);
    int r = RunCase<__nv_bfloat16>(c, "bf16");
    ++arms; fails += (r == 1); refused += (r == 2);
    // Both dtypes on EVERY case, including the 256 bucket (the tile-misalignment
    // row asks for it per bucket): the (256,256) instantiation is compiled for both
    // element types, so an arm that is never run is a claim about code nobody has
    // executed.
    r = RunCase<__half>(c, "fp16");
    ++arms; fails += (r == 1); refused += (r == 2);
  }
  if (g_preflight_fail != 0) {
    std::printf("\npreflight bounds: %d FAIL\n", g_preflight_fail);
    fails += g_preflight_fail;
  }
  // The fail-path fixture: one valid case, refused by the leaf's own preflight
  // because its accumulator is non-null. It must land in `refused`, and the status
  // below must then be nonzero - which is the only way this probe's failure branch
  // is ever executed instead of being a claim about unexecuted code.
  if (const char* inj = std::getenv("DQ_PROBE_INJECT_REFUSAL")) {
    if (inj[0] != '\0' && std::strcmp(inj, "0") != 0) {
      std::printf("\ninjected-refusal fixture: one valid case with dq_accum non-null\n");
      g_inject_refusal = true;
      const int r = RunCase<__nv_bfloat16>(cases[0], "bf16");
      g_inject_refusal = false;
      ++arms;
      refused += (r == 2);
      fails += (r == 1);
      std::printf("  injected refusal: RunCase returned %d (%s)\n", r,
                  r == 2 ? "refused, as required"
                         : r == 0 ? "RAN - the fixture did not refuse, verdicts are broken"
                                  : "failed, not refused - the wrong counter");
    }
  }
  // A refusal on a NUMERIC arm is a defect, not a neutral outcome: the arm was
  // supposed to launch and be compared against the FP64 model, so a run that skipped
  // it has measured nothing. The deliberate must-refuse cells are not cases at all -
  // they are preflight cells whose verdicts fold into `g_preflight_fail` (a
  // must-refuse cell that ACCEPTS is what fails there) - so they never reach this
  // counter. Both classes take the exit code, otherwise "failed or refused" is a
  // comment rather than a contract.
  std::printf("\n%d case-arms, %d failed (cases + preflight), %d refused\n", arms, fails, refused);
  return (fails == 0 && refused == 0) ? 0 : 1;
}
