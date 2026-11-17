#include "flash_sca/native/umma_bwd_plan.h"

#include <cstdio>

namespace sca = xattn::ops::flash_sca_native::umma;

static_assert(sca::MakeBackwardPlan(sca::BackwardFamily::kSplit, false).separate_dq());
static_assert(!sca::MakeBackwardPlan(sca::BackwardFamily::kFusedKey, false)
                   .needs_dq_accumulator());
static_assert(sca::MakeBackwardPlan(sca::BackwardFamily::kFusedAtomicDq, false)
                  .needs_dq_accumulator());
static_assert(sca::MakeBackwardPlan(sca::BackwardFamily::kFusedAtomicDq, true)
                  .separate_dq());
// Family is requested-design identity, not the actual dQ writer.
static_assert(sca::MakeBackwardPlan(sca::BackwardFamily::kFusedAtomicDq, true).family ==
              sca::BackwardFamily::kFusedAtomicDq);

int main() {
  constexpr sca::BackwardFamily families[] = {
      sca::BackwardFamily::kSplit, sca::BackwardFamily::kFusedKey,
      sca::BackwardFamily::kFusedAtomicDq};
  unsigned cases = 0;
  unsigned failures = 0;
  for (unsigned f = 0; f < 3; ++f) {
    // The second argument is "a direct writer was selected", not the determinism flag.
    for (unsigned direct = 0; direct < 2; ++direct) {
      const auto plan = sca::MakeBackwardPlan(families[f], direct != 0);
      const bool expected_atomic = f == 2 && direct == 0;
      if (plan.family != families[f] ||
          plan.separate_dq() == expected_atomic ||
          plan.fused_key_outputs() != (f != 0) ||
          plan.needs_dq_accumulator() != expected_atomic ||
          !sca::DqAccumulatorMatches(plan, expected_atomic) ||
          sca::DqAccumulatorMatches(plan, !expected_atomic)) {
        std::fprintf(stderr, "ownership failure family=%u direct=%u\n", f, direct);
        ++failures;
      }
      // Enumerate all combinations of compiled/eligible leaves. This is an
      // independent Boolean specification, not a second call to the planner.
      for (unsigned mask = 0; mask < 32; ++mask) {
        const sca::BackwardSupport support{bool(mask & 1), bool(mask & 2), bool(mask & 4),
                                           bool(mask & 8), bool(mask & 16)};
        const unsigned required = expected_atomic ? 16u : (f == 0 ? 7u : 9u);
        const bool expected = (mask & required) == required;
        const bool actual = sca::CanLaunchBackward(plan, support);
        if (actual != expected) {
          std::fprintf(stderr, "preflight failure family=%u direct=%u mask=%u\n", f, direct,
                       mask);
          ++failures;
        }
        ++cases;
      }
    }
  }
  // Malformed internal plans fail closed even when every leaf is present.
  constexpr sca::BackwardSupport all{true, true, true, true, true};
  const sca::BackwardPlan invalid[] = {
      {sca::BackwardFamily::kSplit, sca::DqWriteMode::kAtomic},
      {sca::BackwardFamily::kFusedKey, sca::DqWriteMode::kAtomic},
      {static_cast<sca::BackwardFamily>(255), sca::DqWriteMode::kDirect},
      {sca::BackwardFamily::kFusedAtomicDq, static_cast<sca::DqWriteMode>(255)}};
  for (const auto& plan : invalid) {
    if (sca::CanLaunchBackward(plan, all)) ++failures;
    ++cases;
  }
  std::printf("{\"cases\":%u,\"failures\":%u,\"passed\":%s}\n", cases, failures,
              failures ? "false" : "true");
  return failures ? 1 : 0;
}
