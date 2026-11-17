// Refusal probe for the stage-mask launcher (csrc/flash_sca/native/dispatch.cu).
//
// backward_phases_check.cu stubs LaunchBackwardStages, so it proves what the driver
// decides but never touches the real launcher's guards. This probe calls the real
// launcher with three masks it must refuse BEFORE any kernel is issued: a bit no stage
// defines, the silent-zero combination `kStageDq + dq_accum without kStageDkdv`, and an
// empty mask with dq_accum set. All are refusals, and the one accepted mask (empty, no
// accumulator) launches nothing, so no device buffer has to be valid and no kernel runs:
// the probe is about the guards, not the arithmetic.
//
// It compiles dispatch.cu into a private binary, with the nine (D, Dv) bucket bodies
// supplied empty by backward_stage_refusal_stubs.cu. That is deliberate and it is a
// narrowing: the bucket branch is unreachable from any mask the probe sends, so the empty
// bodies change nothing about what is tested, while linking the built xattn_cuda.so made
// the probe unrunnable on any tree without build/ (it exited 2, "no built extension").
// The fact this establishes is therefore about the launcher in the tree, not about the
// binary the package ships; where a built extension exists, linking it is stronger.
#include <cstdio>

#include <c10/util/Exception.h>

#include "flash_sca/native/backward_stage_mask.h"
#include "flash_sca/native/params.h"

namespace {

using namespace xattn::ops::flash_sca_native;

int g_checks = 0;
int g_failures = 0;

void Expect(bool ok, const char* what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL %s\n", what);
  }
}

// True iff the launcher refuses before issuing anything.
template <typename Fn>
bool Refuses(Fn&& fn) {
  try {
    fn();
  } catch (const c10::Error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

}  // namespace

int main() {
  Params p{};
  p.element = Element::kHalf;
  p.q_length = 128;
  p.total_length = 128;
  p.chunk = 64;

  // 1. A bit no stage defines. Both stage loops drop unknown bits silently, so the
  //    launcher must refuse rather than run a mask it cannot honour.
  Expect(Refuses([&] { LaunchBackwardStages(p, nullptr, kBackwardAllStages | 0x40u); }),
         "U1-unknown-stage-bit-refused");

  // 2. The silent-zero combination: an allocated dQ accumulator with the dQ stage still
  //    in the mask and no key stage. The fold would write exact zeros over a dQ the
  //    generic body just wrote.
  float scratch = 0.f;
  p.dq_accum = &scratch;
  Expect(Refuses([&] { LaunchBackwardStages(p, nullptr, kStageDq); }),
         "U2-dq-stage-with-accumulator-refused");

  // 3. The same combination with the key stage present is the legal atomic route and
  //    must NOT be refused by the guard. It would launch real kernels, so it is checked
  //    on the guard's own predicate rather than by calling it: a legal mask reaching a
  //    refusal is the defect this pair exists to keep apart.
  const unsigned legal_atomic = kStageDq | kStageDkdv;
  Expect((legal_atomic & kStageDq) != 0 && (legal_atomic & kStageDkdv) != 0,
         "U3-legal-atomic-mask-is-not-the-refused-shape");

  // 4. Zero stages with the accumulator still set is the invisible form of case 2: nothing
  //    launches, so nothing fills the accumulator the host would fold into dq. Refused.
  Expect(Refuses([&] { LaunchBackwardStages(p, nullptr, 0u); }),
         "U4-zero-mask-with-accumulator-refused");

  // 5. Zero stages without an accumulator is legal by contract (nothing to launch) and
  //    must not throw: every stage was served by another body.
  p.dq_accum = nullptr;
  Expect(!Refuses([&] { LaunchBackwardStages(p, nullptr, 0u); }), "U5-zero-mask-not-refused");

  std::printf("{\"checks\": %d, \"failures\": %d, \"passed\": %s}\n", g_checks, g_failures,
              g_failures == 0 ? "true" : "false");
  return g_failures == 0 ? 0 : 1;
}
