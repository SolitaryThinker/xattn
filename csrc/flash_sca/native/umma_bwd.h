// Backward scheduling: select exclusive writers, bind scratch, final-check
// every selected body, initialize omitted regions, then launch delta/products.
#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

#include "flash_sca/native/backward_stage_mask.h"
#include "flash_sca/native/umma_bwd_entry.h"
#include "flash_sca/native/umma_bwd_plan.h"

namespace xattn::ops::flash_sca_native {

enum class BackwardArm { kOff, kSplit };
enum class LeafEligibility { kNotCompiled, kRefused, kAccepted };
inline const char* EligibilityName(LeafEligibility e) {
  switch (e) {
    case LeafEligibility::kNotCompiled: return "not_compiled";
    case LeafEligibility::kRefused: return "refused";
    case LeafEligibility::kAccepted: return "accepted";
  }
  return "unknown";
}

struct BackwardPhases {
  umma::BackwardPlan plan{umma::BackwardFamily::kFusedKey, umma::DqWriteMode::kDirect};
  // Capabilities and selected writers are separate: overlapping capabilities
  // never authorize overlapping producers.
  umma::BackwardSupport available{};
  umma::BackwardSupport native{};
  LeafEligibility dq_eligibility = LeafEligibility::kNotCompiled;
  LeafEligibility key_eligibility = LeafEligibility::kNotCompiled;
  BackwardArm arm = BackwardArm::kOff;
  bool deterministic = false;
  unsigned generic_stages = kBackwardAllStages;
  // Current mounted leaves write every owned row, including exact dead zeros.
  // No region-initialization protocol is implied by native key ownership.
  bool zero_prev = false;

  bool needs_dq_accumulator() const { return plan.needs_dq_accumulator(); }
  bool native_key_owner() const { return (generic_stages & kStageDkdv) == 0; }
};

#if defined(XATTN_SCA_BWD_TESTING) && XATTN_SCA_BWD_TESTING
namespace detail {
void SetBackwardSupportForTest(const umma::BackwardSupport& support);
void ClearBackwardSupportForTest();
}  // namespace detail
#endif

// A pure, pre-write decision. Inputs and *out stay unchanged on refusal. The
// bool is the user's determinism request; a forced split is recorded separately
// and must not rename the requested family. Scratch is not an eligibility input.
// Body A/B selection (force generic / require both native products) is not a
// driver input: host.cc's single XATTN_SCA_BWD_ARM parser decides it per call
// ("generic" never reaches this plan; "native" is checked on its result).
bool PlanBackwardPhases(const Params& params, const umma::UmmaLaunchParams& native,
                        at::ScalarType element, bool deterministic, BackwardPhases* out,
                        const char** reason = nullptr,
                        BackwardArm arm = BackwardArm::kOff);

// Final prepared-call check after scratch binding and before output writes.
// Includes exact dispatch-matched leaf predicates and exclusive output coverage.
bool CheckBackwardPhases(const Params& params, const umma::UmmaLaunchParams& native,
                         const BackwardPhases& phases, const char** reason = nullptr);

// Per-leg host timer (XATTN_SCA_BWD_LEGTIME), default off. One sample is one
// backward call, measured on the stream it launched on: five event records bracket
// delta, dQ, the key leaf and the generic products, in that order. host_us[i] is the
// steady_clock span from boundary i-1, so host_us[0] has no span by construction.
// All three functions return empty/0 immediately when the mode is 0: a default build
// touches no ring, allocates nothing and makes no CUDA call.
struct BwdLegSample {
  int64_t seq = -1;            // call index since the timer was enabled (per thread)
  int device = -1, thread = -1;
  float delta_ms = 0.f, dq_ms = 0.f, key_ms = 0.f, generic_ms = 0.f, whole_ms = 0.f;
  double host_us[5] = {};      // steady_clock spans between the five boundaries
  bool dq_launched = false, key_launched = false, generic_launched = false;
  bool dropped_before_read = false;
};
bool BwdLegTimerOn();
std::vector<BwdLegSample> TakeBwdLegSamples();   // drains this thread's completed samples
int BwdLegTimerDropped();                        // slots lost: device had not caught up

// Delta once, then selected native/generic products on the same stream. A
// refusal/error after commitment raises; it never replays another producer.
// The host folds scratch exactly once iff needs_dq_accumulator() is true.
void LaunchBackwardPhases(const Params& params, const umma::UmmaLaunchParams& native,
                          const BackwardPhases& phases, cudaStream_t stream);

// The native SoftDelta pair backward (kv_companion.h): the grouped read `g*` and the strict read
// `s*` over the same K/V. Eligible when both dQ leaves and the grouped key leaf are native, no
// accumulator and no generic product is needed, no previous K/V, and the companion key leaf's
// preflight accepts the grouped call. The launch: both deltas, then one key leaf for both reads
// (dK/dV of their sum, rounded once) beside the two dQ leaves.
bool SoftDeltaPairEligible(const Params& gp, const umma::UmmaLaunchParams& gnative,
                           const BackwardPhases& gph, const Params& sp,
                           const umma::UmmaLaunchParams& snative, const BackwardPhases& sph);
void LaunchSoftDeltaBackwardPhases(const Params& gp, const umma::UmmaLaunchParams& gnative,
                                   const BackwardPhases& gph, const Params& sp,
                                   const umma::UmmaLaunchParams& snative,
                                   const BackwardPhases& sph, cudaStream_t stream);

}  // namespace xattn::ops::flash_sca_native
