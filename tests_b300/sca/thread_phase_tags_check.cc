// Host-only check of the per-thread dispatch tag table that the forward require guard reads
// (dispatch_tag.cuh). The guard compares two reads of the calling thread's forward
// umma -> umma/ran count around its own launch. A row recorded by another host thread must
// therefore leave that count unchanged while it still reaches the process-wide table; a row
// recorded by the calling thread moves both. The third check is the discriminating one: a
// guard over the process-wide table would pass on another thread's row, the per-thread one
// does not. No GPU and no CUDA runtime: the header is plain C++.
#include <cstdio>
#include <thread>

#include "csrc/flash_sca/native/dispatch_tag.cuh"

namespace tags = xattn::ops::flash_sca_native;
using tags::Body;
using tags::Phase;
using tags::State;

namespace {

int checks = 0;
int failures = 0;

void Expect(bool ok, const char* name) {
  ++checks;
  if (!ok) ++failures;
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
}

int64_t Mine() { return tags::ThreadPhaseCount(Phase::kForward, Body::kUmma, Body::kUmma, State::kRan); }

int64_t Process() {
  return tags::PhaseTags()
      .count[static_cast<int>(Phase::kForward)][static_cast<int>(Body::kUmma)]
            [static_cast<int>(Body::kUmma)][static_cast<int>(State::kRan)]
      .load();
}

}  // namespace

int main() {
  const int64_t mine0 = Mine();
  const int64_t process0 = Process();
  std::thread other([] { tags::RecordPhase(Phase::kForward, Body::kUmma, Body::kUmma, State::kRan); });
  other.join();
  Expect(Process() == process0 + 1, "another thread's native row reaches the process table");
  Expect(Mine() == mine0, "another thread's native row leaves this thread's count unchanged");
  Expect(Process() > process0 && !(Mine() > mine0),
         "a process-table guard would pass on it, the per-thread guard refuses it");
  tags::RecordPhase(Phase::kForward, Body::kUmma, Body::kUmma, State::kRan);
  Expect(Mine() == mine0 + 1, "this thread's native row moves this thread's count");
  tags::RecordPhase(Phase::kForward, Body::kMmaSync, Body::kMmaSync, State::kRan);
  Expect(Mine() == mine0 + 1, "a generic forward row does not move the native count");
  tags::ResetPhaseTags();
  Expect(Process() == 0, "ResetPhaseTags zeroes the process table");
  Expect(Mine() == mine0 + 1, "ResetPhaseTags leaves the per-thread count, read by difference");
  std::printf("{\"checks\": %d, \"failures\": %d, \"passed\": %s}\n", checks, failures,
              failures == 0 ? "true" : "false");
  return failures == 0 ? 0 : 1;
}
