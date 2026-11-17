// Exclusive backward producer scheduling and dispatch-matched preflight.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "flash_sca/native/dispatch_tag.cuh"
#include "flash_sca/native/umma_bwd.h"
#include "flash_sca/native/kv_companion.h"

namespace xattn::ops::flash_sca_native {

#if defined(XATTN_SCA_BWD_TESTING) && XATTN_SCA_BWD_TESTING
namespace detail {
thread_local umma::BackwardSupport g_backward_support_override{};
thread_local bool g_backward_support_override_set = false;
void SetBackwardSupportForTest(const umma::BackwardSupport& support) {
  g_backward_support_override = support;
  g_backward_support_override_set = true;
}
void ClearBackwardSupportForTest() {
  g_backward_support_override = {};
  g_backward_support_override_set = false;
}
}  // namespace detail
#endif

namespace {

bool Fail(const char** reason, const char* message) {
  if (reason) *reason = message;
  return false;
}

// XATTN_SCA_BWD_LEGTIME: per-leg host timer for the native backward launch body.
// unset/""/"0" = off (no CUDA call, no allocation, no print); "1" = record and expose
// through _flash_sca_last_backward_preparation and _flash_sca_native_bwd_legtimes;
// "2" = also print one SCA_BWD_LEG line per drained call. Resolved ONCE per process:
// this is a cost instrument, not an A/B arm, so a mid-process flip could only mix two
// protocols in one table. An unknown value raises, as every other instrument here does.
int LegTimeMode() {
  static const int mode = []() -> int {
    const char* v = std::getenv("XATTN_SCA_BWD_LEGTIME");
    if (v == nullptr || *v == '\0') return 0;
    const std::string s(v);
    if (s == "0") return 0;
    if (s == "1") return 1;
    if (s == "2") return 2;
    TORCH_CHECK(false, "XATTN_SCA_BWD_LEGTIME must be unset, '0', '1' or '2'; got '", s, "'");
    return 0;
  }();
  return mode;
}
inline bool LegTimeOn() { return LegTimeMode() != 0; }

// Five boundaries, four legs: delta, dQ, the key leaf (fused or dK/dV) and the generic
// products. Slots 2 and 3 are adjacent on the fused atomic route, which is its own label.
enum LegSlot : int {
  kLegStart = 0, kLegAfterDelta = 1, kLegAfterDq = 2, kLegAfterKey = 3, kLegEnd = 4,
  kLegSlotCount = 5
};
const char* LegSlotName(int boundary) {
  switch (boundary) {
    case kLegStart: return "start";
    case kLegAfterDelta: return "after_delta";
    case kLegAfterDq: return "after_dq";
    case kLegAfterKey: return "after_key";
    case kLegEnd: return "end";
    default: return "unknown";
  }
}
// Calls that may be in flight before a slot is reclaimed. Reuse is guarded by a query,
// never by a wait: a device that has not caught up costs one sample, not a stall.
constexpr int kLegRingCalls = 8;

struct LegRing {
  cudaEvent_t ev[kLegRingCalls][kLegSlotCount] = {};
  std::chrono::steady_clock::time_point host[kLegRingCalls][kLegSlotCount];
  int64_t seq[kLegRingCalls] = {};        // call index of the slot's last use
  int nrec[kLegRingCalls] = {};           // boundaries recorded; 0 = slot free
  bool launched[kLegRingCalls][3] = {};   // dq, key, generic: did it launch on that call
  std::vector<BwdLegSample> done;         // completed samples, drained by the caller
  int next = 0;                           // slot of the call in progress
  int dev = -1, recreated = 0, dropped = 0;
  int64_t calls = 0;
  bool ready = false;
};
LegRing& LegRingRef() { static thread_local LegRing r; return r; }

// A small stable id per host thread: two host threads keep two rings, and a bench must
// say which thread it timed.
int LegThreadIndex() {
  static std::atomic<int> next_index{0};
  static thread_local const int index = next_index.fetch_add(1, std::memory_order_relaxed);
  return index;
}

void LegRingDestroy(LegRing& ring) {
  for (int i = 0; i < kLegRingCalls; ++i) {
    for (int s = 0; s < kLegSlotCount; ++s) {
      if (ring.ev[i][s] != nullptr) (void)cudaEventDestroy(ring.ev[i][s]);
      ring.ev[i][s] = nullptr;
    }
    ring.seq[i] = -1;
    ring.nrec[i] = 0;
  }
  ring.ready = false;
}

// Events belong to the context current at creation, so they are created here: once, on
// the first enabled call, and before that call's first boundary is recorded. The ~50 us
// of creation is then outside every measured interval by construction.
void LegRingEnsure(LegRing& ring, int dev) {
  if (ring.ready && ring.dev == dev) return;
  if (ring.ready) {
    // A call on another device cannot use these events. Anything in flight is lost and
    // is counted, never reported as a zero-length leg.
    for (int i = 0; i < kLegRingCalls; ++i) {
      if (ring.nrec[i] != 0) ++ring.dropped;
    }
    LegRingDestroy(ring);
    ++ring.recreated;
  }
  for (int i = 0; i < kLegRingCalls; ++i) {
    ring.seq[i] = -1;
    ring.nrec[i] = 0;
    ring.launched[i][0] = ring.launched[i][1] = ring.launched[i][2] = false;
    for (int s = 0; s < kLegSlotCount; ++s) {
      const cudaError_t status = cudaEventCreateWithFlags(&ring.ev[i][s], cudaEventDefault);
      if (status != cudaSuccess) {
        LegRingDestroy(ring);
        TORCH_CHECK(false, "XATTN_SCA_BWD_LEGTIME: cudaEventCreate failed: ",
                    cudaGetErrorString(status));
      }
    }
  }
  ring.dev = dev;
  ring.ready = true;
}

void LegRecord(LegRing& ring, int slot, int boundary, cudaStream_t stream) {
  // The stamp sits beside the record, so span i-1 -> i is what the host spent between two
  // boundaries: the column that stops dispatch time being read as kernel time.
  ring.host[slot][boundary] = std::chrono::steady_clock::now();
  const cudaError_t status = cudaEventRecord(ring.ev[slot][boundary], stream);
  TORCH_CHECK(status == cudaSuccess, "XATTN_SCA_BWD_LEGTIME: cudaEventRecord(",
              LegSlotName(boundary), ") failed: ", cudaGetErrorString(status));
}

// One completed call, read only after its last event reported complete. All five records
// are on one stream, so device order is record order and the four legs close on `whole`
// up to event resolution; that closure is a plumbing check, not an attribution one.
void LegCollect(LegRing& ring, int slot) {
  float at[kLegSlotCount] = {};           // ms since kLegStart (at[kLegStart] == 0)
  for (int i = kLegAfterDelta; i < kLegSlotCount; ++i) {
    const cudaError_t status =
        cudaEventElapsedTime(&at[i], ring.ev[slot][kLegStart], ring.ev[slot][i]);
    TORCH_CHECK(status == cudaSuccess, "XATTN_SCA_BWD_LEGTIME: cudaEventElapsedTime failed: ",
                cudaGetErrorString(status));
  }
  BwdLegSample sample;
  sample.seq = ring.seq[slot];
  sample.device = ring.dev;
  sample.thread = LegThreadIndex();
  sample.delta_ms = at[kLegAfterDelta];
  sample.dq_ms = at[kLegAfterDq] - at[kLegAfterDelta];
  sample.key_ms = at[kLegAfterKey] - at[kLegAfterDq];
  sample.generic_ms = at[kLegEnd] - at[kLegAfterKey];
  sample.whole_ms = at[kLegEnd];
  double prev_us = 0.0;
  for (int i = 0; i < kLegSlotCount; ++i) {
    const double total_us = std::chrono::duration<double, std::micro>(
        ring.host[slot][i] - ring.host[slot][kLegStart]).count();
    sample.host_us[i] = total_us - prev_us;
    prev_us = total_us;
  }
  sample.dq_launched = ring.launched[slot][0];
  sample.key_launched = ring.launched[slot][1];
  sample.generic_launched = ring.launched[slot][2];
  sample.dropped_before_read = false;     // an unready slot is skipped, never read
  ring.done.push_back(sample);
}

// Mode 2: one line per drained call, at drain time only. A leg that did not launch prints
// "-" rather than a fabricated 0.000, and closure is (delta+dq+key+generic)/whole -- an
// identity on one stream, so it detects a mis-record and nothing else.
void LegPrint(const BwdLegSample& s) {
  char dq_ms[24] = "-", key_ms[24] = "-", generic_ms[24] = "-";
  if (s.dq_launched) std::snprintf(dq_ms, sizeof(dq_ms), "%.3f", s.dq_ms);
  if (s.key_launched) std::snprintf(key_ms, sizeof(key_ms), "%.3f", s.key_ms);
  if (s.generic_launched) std::snprintf(generic_ms, sizeof(generic_ms), "%.3f", s.generic_ms);
  const double parts = s.delta_ms + s.dq_ms + s.key_ms + s.generic_ms;
  const double closure = s.whole_ms > 0.f ? parts / s.whole_ms : 0.0;
  double host_total_us = 0.0;
  for (double span : s.host_us) host_total_us += span;
  std::printf(
      "SCA_BWD_LEG seq=%lld dev=%d thr=%d delta_ms=%.3f dq_ms=%s key_ms=%s generic_ms=%s "
      "whole_ms=%.3f closure=%.3f host_us=%.1f/%.1f/%.1f/%.1f/%.1f host_total_us=%.1f\n",
      static_cast<long long>(s.seq), s.device, s.thread, s.delta_ms, dq_ms, key_ms, generic_ms,
      s.whole_ms, closure, s.host_us[0], s.host_us[1], s.host_us[2], s.host_us[3], s.host_us[4],
      host_total_us);
  std::fflush(stdout);
}

// One boundary record on the call's own stream. kLegStart owns the ring: it creates the
// events on the first enabled call and reclaims the slot this call reuses -- query only,
// so a slot the device has not reached is counted in `dropped` and skipped, never read.
void LegBoundary(int boundary, cudaStream_t stream) {
  LegRing& ring = LegRingRef();
  if (boundary == kLegStart) {
    int dev = -1;
    const cudaError_t status = cudaGetDevice(&dev);
    TORCH_CHECK(status == cudaSuccess, "XATTN_SCA_BWD_LEGTIME: cudaGetDevice failed: ",
                cudaGetErrorString(status));
    LegRingEnsure(ring, dev);
    const int slot = static_cast<int>(ring.calls % kLegRingCalls);
    ++ring.calls;
    ring.next = slot;
    if (ring.nrec[slot] != 0) {
      if (ring.nrec[slot] != kLegSlotCount) {
        ++ring.dropped;                   // a call that raised between two boundaries
      } else {
        const cudaError_t ready = cudaEventQuery(ring.ev[slot][kLegEnd]);
        if (ready == cudaSuccess) {
          LegCollect(ring, slot);
        } else if (ready == cudaErrorNotReady) {
          ++ring.dropped;
        } else {
          TORCH_CHECK(false, "XATTN_SCA_BWD_LEGTIME: cudaEventQuery failed: ",
                      cudaGetErrorString(ready));
        }
      }
    }
    ring.seq[slot] = ring.calls - 1;
    ring.launched[slot][0] = ring.launched[slot][1] = ring.launched[slot][2] = false;
    ring.nrec[slot] = 0;
    LegRecord(ring, slot, kLegStart, stream);
    ring.nrec[slot] = 1;
    return;
  }
  const int slot = ring.next;
  if (!ring.ready || ring.nrec[slot] != boundary) return;   // a boundary out of order
  LegRecord(ring, slot, boundary, stream);
  ring.nrec[slot] = boundary + 1;
}

// The route flags come from the plan, not from an interval: a leg that did not launch
// must read "-", never a fabricated 0.000. `products` is not computed until after the
// last launch, so the flags are set here, at the end of the call, before kLegEnd.
void LegMarkLaunched(bool dq, bool key, bool generic) {
  LegRing& ring = LegRingRef();
  if (!ring.ready) return;
  ring.launched[ring.next][0] = dq;
  ring.launched[ring.next][1] = key;
  ring.launched[ring.next][2] = generic;
}

bool SameOperand(const Operand& a, const Operand& b) {
  return a.ptr == b.ptr && a.batch_stride == b.batch_stride &&
         a.row_stride == b.row_stride && a.head_stride == b.head_stride;
}

// Compare BEFORE rebinding launch.params, or this check would be vacuous.
// The sole permitted change after preparation is the planned accumulator.
bool SameBackwardCall(const Params& a, const Params& b) {
  if (!SameOperand(a.q, b.q) || !SameOperand(a.k, b.k) || !SameOperand(a.v, b.v) ||
      !SameOperand(a.prev_k, b.prev_k) || !SameOperand(a.prev_v, b.prev_v)) return false;
#define XATTN_SCA_SAME(F) if (a.F != b.F) return false;
  XATTN_SCA_SAME(element)
  XATTN_SCA_SAME(batch)
  XATTN_SCA_SAME(q_length)
  XATTN_SCA_SAME(k_length)
  XATTN_SCA_SAME(prev_length)
  XATTN_SCA_SAME(total_length)
  XATTN_SCA_SAME(query_heads)
  XATTN_SCA_SAME(kv_heads)
  XATTN_SCA_SAME(group)
  XATTN_SCA_SAME(qk_dim)
  XATTN_SCA_SAME(value_dim)
  XATTN_SCA_SAME(chunk)
  XATTN_SCA_SAME(scale)
  XATTN_SCA_SAME(scale_log2)
  XATTN_SCA_SAME(strict)
  XATTN_SCA_SAME(subtract_before_scale)
  XATTN_SCA_SAME(sorted_order)
  XATTN_SCA_SAME(order_always)
  XATTN_SCA_SAME(order_dump)
  XATTN_SCA_SAME(head_fastest)
  XATTN_SCA_SAME(row_lo)
  XATTN_SCA_SAME(row_hi)
  XATTN_SCA_SAME(q_segment)
  XATTN_SCA_SAME(k_segment)
  XATTN_SCA_SAME(segment_exact)
  XATTN_SCA_SAME(tile_bounds)
  XATTN_SCA_SAME(tile_count)
  XATTN_SCA_SAME(key_tile_range)
  XATTN_SCA_SAME(key_tile_count)
  XATTN_SCA_SAME(y)
  XATTN_SCA_SAME(y_state)
  XATTN_SCA_SAME(lse)
  XATTN_SCA_SAME(dy)
  XATTN_SCA_SAME(saved)
  XATTN_SCA_SAME(saved_fp32)
  XATTN_SCA_SAME(lse_in)
  XATTN_SCA_SAME(delta)
  XATTN_SCA_SAME(dq)
  XATTN_SCA_SAME(dk)
  XATTN_SCA_SAME(dv)
  XATTN_SCA_SAME(dprev_k)
  XATTN_SCA_SAME(dprev_v)
  XATTN_SCA_SAME(key_grads_fp32)
  XATTN_SCA_SAME(umma_maps)
#undef XATTN_SCA_SAME
  // umma_maps is a host-only FORWARD field: launch.cuh copies the forward's maps out
  // of it. A backward call carries its maps in UmmaLaunchParams::maps instead, so the
  // field must be null on both sides, not merely equal; a non-null value here would
  // be a forward map set leaking into a backward launch.
  if (a.umma_maps != nullptr || b.umma_maps != nullptr) return false;
  return true;
}

bool QueryLeaf(umma::BackwardProduct product, const umma::UmmaLaunchParams& call,
               LeafEligibility* eligibility, bool* whole) {
  const auto* entry = umma::FindBackwardLeaf(product, call.params);
  if (entry == nullptr) {
    *eligibility = LeafEligibility::kNotCompiled;
    *whole = false;
    return false;
  }
  // CUDA/API failures propagate from can(); they are not leaf refusals.
  const bool accepted = entry->can(call);
  *eligibility = accepted ? LeafEligibility::kAccepted : LeafEligibility::kRefused;
  *whole = accepted && entry->writes_whole_plane;
  return accepted;
}

umma::BackwardSupport QueryNativeSupport(const umma::UmmaLaunchParams& call,
                                         BackwardPhases* phases) {
#if defined(XATTN_SCA_BWD_TESTING) && XATTN_SCA_BWD_TESTING
  if (detail::g_backward_support_override_set) return detail::g_backward_support_override;
#endif
  umma::BackwardSupport support{};
  bool whole_dq = false;
  support.dq = QueryLeaf(umma::BackwardProduct::kDq, call, &phases->dq_eligibility, &whole_dq);
  support.dq = support.dq && whole_dq;
  support.dkdv = QueryLeaf(umma::BackwardProduct::kDkdv, call, &phases->key_eligibility,
                           &support.dkdv_writes_whole_plane);
  // This mount has no omitted-region initializer. An entry may accept the
  // operands but still not supply a complete producer for this schedule.
  support.dkdv = support.dkdv && support.dkdv_writes_whole_plane;
  return support;
}

// Available bits are capabilities, not simultaneous writers. Prefer an
// eligible requested atomic family; otherwise construct the complete mixed
// direct route, including native keys with generic row-dQ on det-off calls.
void SelectWriters(BackwardPhases* phases) {
  const auto& available = phases->available;
  // B300 is deterministic-only: the atomic-dQ key writer (kFusedAtomicDq, unordered FP32
  // red.global.add into dq_accum) is never selected, whatever phases->deterministic says.
  auto& selected = phases->native;
  selected = {};
  selected.dq = available.dq;
  selected.dkdv = available.dkdv;
  selected.dkdv_writes_whole_plane = selected.dkdv && available.dkdv_writes_whole_plane;
  phases->plan = umma::MakeBackwardPlan(umma::BackwardFamily::kFusedKey, true);
  phases->generic_stages = kStageDelta;
  if (!selected.dq && !selected.dkdv_atomic_dq) phases->generic_stages |= kStageDq;
  if (!selected.dkdv && !selected.dkdv_atomic_dq) phases->generic_stages |= kStageDkdv;
  phases->zero_prev = false;
}

bool SameSelected(const umma::BackwardSupport& a, const umma::BackwardSupport& b) {
  return a.dq == b.dq && a.dk == b.dk && a.dv == b.dv && a.dkdv == b.dkdv &&
         a.dkdv_atomic_dq == b.dkdv_atomic_dq &&
         a.dkdv_writes_whole_plane == b.dkdv_writes_whole_plane;
}

bool CheckLeaf(umma::BackwardProduct product, const umma::UmmaLaunchParams& call,
               const char** reason) {
  const auto* entry = umma::FindBackwardLeaf(product, call.params);
  if (entry == nullptr) return Fail(reason, "selected native leaf is not compiled");
  if (!entry->writes_whole_plane) return Fail(reason, "selected leaf omits output rows");
  if (!entry->can(call)) return Fail(reason, "selected native leaf refused final operands");
  return true;
}

// True when the product writes no element: dQ's plane is [B, Nq, Hq, D], the key product's
// planes are [B, Nk, Hkv, *] and [B, P, Hkv, *] with Nk + P = total_length. Such a product
// has nothing to launch, and its launcher's `true` would not mean a launch happened: the
// dK/dV launcher returns true without launching when its key grid is empty.
bool ProductHasNoWork(umma::BackwardProduct product, const Params& p) {
  if (p.batch <= 0) return true;
  if (product == umma::BackwardProduct::kDq) return p.q_length <= 0 || p.query_heads <= 0;
  return p.total_length <= 0 || p.kv_heads <= 0;
}

void LaunchLeaf(umma::BackwardProduct product, const umma::UmmaLaunchParams& call,
                cudaStream_t stream) {
  const bool dq_phase =
      product == umma::BackwardProduct::kDq || product == umma::BackwardProduct::kAtomicDq;
  const bool key_phases = product != umma::BackwardProduct::kDq;
  // A kRan row means a launch happened (dispatch_tag.cuh). With no element to write the
  // leaf is not entered and its phases are recorded as skipped, both bodies none.
  if (ProductHasNoWork(product, call.params)) {
    if (dq_phase) RecordPhase(Phase::kDq, Body::kNone, Body::kNone, State::kSkipped);
    if (key_phases) {
      RecordPhase(Phase::kDk, Body::kNone, Body::kNone, State::kSkipped);
      RecordPhase(Phase::kDv, Body::kNone, Body::kNone, State::kSkipped);
    }
    return;
  }
  const auto* entry = umma::FindBackwardLeaf(product, call.params);
  TORCH_CHECK(entry != nullptr && entry->launch(call, stream),
              "native FlashSCA backward: selected leaf refused after commitment");
  if (dq_phase) {
    RecordPhase(Phase::kDq, Body::kUmma, Body::kUmma, State::kRan);
  }
  if (key_phases) {
    RecordPhase(Phase::kDk, Body::kUmma, Body::kUmma, State::kRan);
    RecordPhase(Phase::kDv, Body::kUmma, Body::kUmma, State::kRan);
  }
}

}  // namespace

// XATTN_SCA_BWD_SERIAL: force the two native leaves back onto one stream. A probe's
// switch, read per call and off by default; no gate check sets it. It exists because the
// overlap is a scheduling change whose A/B has to be one build, two arms, one job.
bool BwdLeavesSerialForced() {
  const char* v = std::getenv("XATTN_SCA_BWD_SERIAL");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// The timer's public face: declared in umma_bwd.h so the host snapshot and the drain
// binding can read it. All three return empty/0 with the mode off, so a default build
// makes no CUDA call and never even constructs the ring.
bool BwdLegTimerOn() { return LegTimeOn(); }

int BwdLegTimerDropped() {
  if (!BwdLegTimerOn()) return 0;
  return LegRingRef().dropped;
}

// Drains this thread's completed samples. This is the one blocking read in the timer, and
// it runs outside every measured interval, only because the caller asked for the numbers:
// the per-slot reclaim in LegBoundary queries and never waits.
std::vector<BwdLegSample> TakeBwdLegSamples() {
  if (!BwdLegTimerOn()) return {};
  LegRing& ring = LegRingRef();
  if (ring.ready) {
    for (int slot = 0; slot < kLegRingCalls; ++slot) {
      if (ring.nrec[slot] != kLegSlotCount) {
        if (ring.nrec[slot] != 0) ++ring.dropped;     // a call that raised mid-way
      } else {
        const cudaError_t status = cudaEventSynchronize(ring.ev[slot][kLegEnd]);
        TORCH_CHECK(status == cudaSuccess, "XATTN_SCA_BWD_LEGTIME: cudaEventSynchronize ",
                    "failed: ", cudaGetErrorString(status));
        LegCollect(ring, slot);
      }
      ring.seq[slot] = -1;
      ring.nrec[slot] = 0;
    }
  }
  std::vector<BwdLegSample> out;
  out.swap(ring.done);
  if (LegTimeMode() == 2) {
    for (const auto& sample : out) LegPrint(sample);
  }
  return out;
}

bool PlanBackwardPhases(const Params& params, const umma::UmmaLaunchParams& native,
                        at::ScalarType element, bool deterministic, BackwardPhases* out,
                        const char** reason, BackwardArm arm) {
  if (out == nullptr) return Fail(reason, "PlanBackwardPhases: out is null");
  switch (params.element) {
    case Element::kHalf:
      if (element != at::kHalf) return Fail(reason, "PlanBackwardPhases: fp16 element mismatch");
      break;
    case Element::kBFloat16:
      if (element != at::kBFloat16) return Fail(reason, "PlanBackwardPhases: bf16 element mismatch");
      break;
    default: return Fail(reason, "PlanBackwardPhases: not a known Element value");
  }
  if (arm != BackwardArm::kOff && arm != BackwardArm::kSplit)
    return Fail(reason, "PlanBackwardPhases: unknown backward arm");
  if (!SameBackwardCall(params, native.params))
    return Fail(reason, "PlanBackwardPhases: prepared launch differs from the call");
  if (params.dq_accum != nullptr || native.params.dq_accum != nullptr)
    return Fail(reason, "PlanBackwardPhases: select writers before binding an accumulator");
  BackwardPhases phases{};
  phases.arm = arm;
  phases.deterministic = deterministic;
  phases.available = QueryNativeSupport(native, &phases);
  SelectWriters(&phases);
  *out = phases;
  if (reason) *reason = nullptr;
  return true;
}

bool CheckBackwardPhases(const Params& params, const umma::UmmaLaunchParams& native,
                         const BackwardPhases& phases, const char** reason) {
  if (params.element != Element::kHalf && params.element != Element::kBFloat16)
    return Fail(reason, "CheckBackwardPhases: unknown element");
  if (phases.arm != BackwardArm::kOff && phases.arm != BackwardArm::kSplit)
    return Fail(reason, "CheckBackwardPhases: unknown arm");
  if (!SameBackwardCall(params, native.params))
    return Fail(reason, "CheckBackwardPhases: stale prepared operands or state");
  if (native.params.dq_accum != nullptr && native.params.dq_accum != params.dq_accum)
    return Fail(reason, "CheckBackwardPhases: stale prepared accumulator");
  if (!umma::DqAccumulatorMatches(phases.plan, params.dq_accum != nullptr))
    return Fail(reason, "CheckBackwardPhases: accumulator does not match selected write mode");

  BackwardPhases expected = phases;
  SelectWriters(&expected);
  if (!SameSelected(phases.native, expected.native) ||
      phases.plan.family != expected.plan.family || phases.plan.dq_mode != expected.plan.dq_mode ||
      phases.generic_stages != expected.generic_stages || phases.zero_prev)
    return Fail(reason, "CheckBackwardPhases: nonexclusive or incomplete producer schedule");
  if ((phases.native.dkdv || phases.native.dkdv_atomic_dq) &&
      !phases.native.dkdv_writes_whole_plane)
    return Fail(reason, "CheckBackwardPhases: key leaf does not cover the whole plane");
  // Rebind only after validating the two descriptions. The map storage and its
  // retained owners still belong to native, while scratch belongs to the plan.
  umma::UmmaLaunchParams call = native;
  call.params = params;
  if (phases.native.dq && !CheckLeaf(umma::BackwardProduct::kDq, call, reason)) return false;
  if (phases.native.dkdv && !CheckLeaf(umma::BackwardProduct::kDkdv, call, reason)) return false;
  if (phases.native.dkdv_atomic_dq &&
      !CheckLeaf(umma::BackwardProduct::kAtomicDq, call, reason)) return false;
  if (reason) *reason = nullptr;
  return true;
}

void LaunchBackwardPhases(const Params& params, const umma::UmmaLaunchParams& native,
                          const BackwardPhases& phases, cudaStream_t stream) {
  const char* reason = nullptr;
  const bool legtime = LegTimeOn();          // ONE resolution per call, not per launch
  TORCH_CHECK(CheckBackwardPhases(params, native, phases, &reason),
              "native FlashSCA backward: invalid launch plan: ", reason ? reason : "unknown");
  if (legtime) LegBoundary(kLegStart, stream);
  umma::UmmaLaunchParams call = native;
  call.params = params;
  LaunchBackwardStages(params, stream, kStageDelta);
  if (legtime) LegBoundary(kLegAfterDelta, stream);
  // The two native leaves own disjoint outputs -- the dQ leaf writes only dQ, the key leaf
  // only dK/dV -- and neither writes atomically, so nothing about the result depends on
  // their order or on whether they overlap: run them on two streams and join before the
  // call returns. It is worth doing because the key leaf's CTAs are not equally heavy: at
  // the MoE rows its 1024 CTAs (6.9 waves of a 148-SM GPU) carry 2 to 128 query tiles each
  // (mean 33), so its last wave leaves SMs idle while the dQ leaf's 4096 CTAs (27.7 waves,
  // mean 17) could be using them. The key leaf is enqueued first so that its tail is the
  // part the dQ leaf fills, the event join puts both back in front of anything the caller
  // enqueues next. The leg timer measures the serial order, so it keeps it.
  const bool overlap = phases.native.dq && phases.native.dkdv &&
      !phases.native.dkdv_atomic_dq && !legtime && !BwdLeavesSerialForced();
  if (overlap) {
    const c10::cuda::CUDAStream fork =
        c10::cuda::getStreamFromPool(/*isHighPriority=*/false, c10::cuda::current_device());
    cudaEvent_t ready = nullptr;
    cudaEvent_t done = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(ready, stream));
    C10_CUDA_CHECK(cudaStreamWaitEvent(fork.stream(), ready, 0));
    LaunchLeaf(umma::BackwardProduct::kDkdv, call, stream);
    LaunchLeaf(umma::BackwardProduct::kDq, call, fork.stream());
    C10_CUDA_CHECK(cudaEventRecord(done, fork.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
    C10_CUDA_CHECK(cudaEventDestroy(ready));
    C10_CUDA_CHECK(cudaEventDestroy(done));
    if (legtime) LegBoundary(kLegAfterKey, stream);
  } else {
    if (phases.native.dq) LaunchLeaf(umma::BackwardProduct::kDq, call, stream);
    if (legtime) LegBoundary(kLegAfterDq, stream);
    if (phases.native.dkdv) LaunchLeaf(umma::BackwardProduct::kDkdv, call, stream);
    if (phases.native.dkdv_atomic_dq) LaunchLeaf(umma::BackwardProduct::kAtomicDq, call, stream);
    if (legtime) LegBoundary(kLegAfterKey, stream);
  }
  const unsigned products = phases.generic_stages & ~static_cast<unsigned>(kStageDelta);
  if (products != 0) LaunchBackwardStages(params, stream, products);
  if (legtime) {
    LegMarkLaunched(phases.native.dq, phases.native.dkdv || phases.native.dkdv_atomic_dq,
                    products != 0);
    LegBoundary(kLegEnd, stream);
  }
}

// (promotion: main <- H9 + H8 + H4, bld1)
bool SoftDeltaPairEligible(const Params& gp, const umma::UmmaLaunchParams& gnative,
                           const BackwardPhases& gph, const Params& sp,
                           const umma::UmmaLaunchParams& snative, const BackwardPhases& sph) {
  (void)gnative;
  (void)snative;
  if (!gph.native.dq || !gph.native.dkdv || gph.native.dkdv_atomic_dq) return false;
  if (!sph.native.dq || sph.native.dkdv_atomic_dq) return false;
  if (gph.needs_dq_accumulator() || sph.needs_dq_accumulator()) return false;
  if ((gph.generic_stages & ~static_cast<unsigned>(kStageDelta)) != 0) return false;
  if (gp.prev_length != 0 || sp.prev_length != 0) return false;
  umma::UmmaLaunchParams call = gnative;
  call.params = gp;
  return umma::CanLaunchUmmaBackwardKvPipeCompanion(call);
}

void LaunchSoftDeltaBackwardPhases(const Params& gp, const umma::UmmaLaunchParams& gnative,
                                   const BackwardPhases& gph, const Params& sp,
                                   const umma::UmmaLaunchParams& snative,
                                   const BackwardPhases& sph, cudaStream_t stream) {
  const char* reason = nullptr;
  TORCH_CHECK(CheckBackwardPhases(gp, gnative, gph, &reason),
              "native SoftDelta backward: invalid grouped launch plan: ", reason ? reason : "?");
  TORCH_CHECK(CheckBackwardPhases(sp, snative, sph, &reason),
              "native SoftDelta backward: invalid strict launch plan: ", reason ? reason : "?");
  TORCH_CHECK(SoftDeltaPairEligible(gp, gnative, gph, sp, snative, sph),
              "native SoftDelta backward: the pair launch was refused after commitment");
  umma::UmmaLaunchParams gcall = gnative;
  gcall.params = gp;
  umma::UmmaLaunchParams scall = snative;
  scall.params = sp;
  // Both reads' delta, tile bounds and key tile ranges first: the key leaf reads the strict
  // read's delta through the companion.
  LaunchBackwardStages(sp, stream, kStageDelta);
  LaunchBackwardStages(gp, stream, kStageDelta);
  umma::KvCompanion comp{};
  comp.q = snative.maps.q;
  comp.dy = snative.maps.dy;
  comp.lse = sp.lse_in;
  comp.delta = sp.delta;
  comp.row_lo = sp.row_lo;
  comp.row_hi = sp.row_hi;
  // The key leaf writes only dK/dV, each dQ leaf only its own dQ, none atomically: the three run
  // on two streams and join before anything the caller enqueues next.
  const c10::cuda::CUDAStream fork =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false, c10::cuda::current_device());
  cudaEvent_t ready = nullptr;
  cudaEvent_t done = nullptr;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(ready, stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(fork.stream(), ready, 0));
  if (!ProductHasNoWork(umma::BackwardProduct::kDkdv, gp)) {
    TORCH_CHECK(umma::LaunchUmmaBackwardKvPipeCompanion(gcall, comp, stream),
                "native SoftDelta backward: companion key leaf refused after commitment");
    RecordPhase(Phase::kDk, Body::kUmma, Body::kUmma, State::kRan);
    RecordPhase(Phase::kDv, Body::kUmma, Body::kUmma, State::kRan);
  } else {
    RecordPhase(Phase::kDk, Body::kNone, Body::kNone, State::kSkipped);
    RecordPhase(Phase::kDv, Body::kNone, Body::kNone, State::kSkipped);
  }
  LaunchLeaf(umma::BackwardProduct::kDq, gcall, fork.stream());
  LaunchLeaf(umma::BackwardProduct::kDq, scall, fork.stream());
  C10_CUDA_CHECK(cudaEventRecord(done, fork.stream()));
  C10_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
  C10_CUDA_CHECK(cudaEventDestroy(ready));
  C10_CUDA_CHECK(cudaEventDestroy(done));
}

}  // namespace xattn::ops::flash_sca_native
