// Linked, host-only conformance test for native/umma_bwd.{h,cu}.
//
// The registry and every launch entry are stubs in THIS translation unit. Do not
// link sca_umma_bwd_registry.cu, dispatch.cu, or a device instantiation into this test.
// The real driver chooses writers, final-checks their exact prepared arguments,
// and records successful native phase tags. Generic tags are reproduced by the
// stage-launch stub. No operand or tensor-map descriptor is dereferenced.
//
// Build with XATTN_SCA_BWD_TESTING=1 (including the driver TU). That enables only
// the planning-support override: final entry predicates must still run.
// Prints a JSON summary; a nonzero exit means at least one host contract failed.
#include <cstdio>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <string>
#include <vector>

#include "flash_sca/native/dispatch_tag.cuh"
#include "flash_sca/native/umma_bwd.h"

#if !defined(XATTN_SCA_BWD_TESTING) || !XATTN_SCA_BWD_TESTING
#error "Compile this test AND umma_bwd.cu with -DXATTN_SCA_BWD_TESTING=1"
#endif

namespace stub {
using namespace xattn::ops::flash_sca_native;

enum class Call { kGeneric, kDq, kDkdv, kAtomicDq };
struct Event {
  Call call;
  unsigned stages;
};
struct Leaf {
  bool present = true;
  bool can = true;
  bool launch = true;
  bool whole_plane = true;
  bool refuse_after_delta = false;
  int can_calls = 0;
  int launch_calls = 0;
  umma::UmmaLaunchParams last_can{};
  umma::UmmaLaunchParams last_launch{};
  cudaStream_t last_stream = nullptr;
};

Leaf dq, key, atomic;
std::vector<Event> events;
bool after_delta = false;

void Reset() {
  dq = Leaf{};
  key = Leaf{};
  atomic = Leaf{};
  // Production has no mounted atomic entry. Tests explicitly enable it only
  // for selection-table coverage, not as a claim about a device implementation.
  atomic.present = false;
  events.clear();
  after_delta = false;
  detail::ClearBackwardSupportForTest();
  ResetPhaseTags();
}

bool Can(Leaf& leaf, const umma::UmmaLaunchParams& p) {
  ++leaf.can_calls;
  leaf.last_can = p;
  return leaf.can && !(after_delta && leaf.refuse_after_delta);
}
bool Launch(Leaf& leaf, Call call, const umma::UmmaLaunchParams& p, cudaStream_t stream) {
  ++leaf.launch_calls;
  leaf.last_launch = p;
  leaf.last_stream = stream;
  // Record attempts, including false returns; a false return MUST NOT be
  // relabelled as success, nor followed by a generic replay.
  events.push_back({call, 0});
  return Can(leaf, p) && leaf.launch;
}
bool CanDq(const umma::UmmaLaunchParams& p) { return Can(dq, p); }
bool CanKey(const umma::UmmaLaunchParams& p) { return Can(key, p); }
bool CanAtomic(const umma::UmmaLaunchParams& p) { return Can(atomic, p); }
bool LaunchDq(const umma::UmmaLaunchParams& p, cudaStream_t s) {
  return Launch(dq, Call::kDq, p, s);
}
bool LaunchKey(const umma::UmmaLaunchParams& p, cudaStream_t s) {
  return Launch(key, Call::kDkdv, p, s);
}
bool LaunchAtomic(const umma::UmmaLaunchParams& p, cudaStream_t s) {
  return Launch(atomic, Call::kAtomicDq, p, s);
}
}  // namespace stub

namespace xattn::ops::flash_sca_native::umma {

// Pure registry stub: the driver queries and launches through these SAME entries.
// Entry presence, exact Can(), launch refusal, and full-plane ownership are
// independent controls. A linkable entry by itself is not an eligibility claim.
const BackwardLeafEntry* FindBackwardLeaf(BackwardProduct product, const Params&) {
  static BackwardLeafEntry dq{stub::CanDq, stub::LaunchDq, true};
  static BackwardLeafEntry key{stub::CanKey, stub::LaunchKey, true};
  static BackwardLeafEntry atomic{stub::CanAtomic, stub::LaunchAtomic, true};
  switch (product) {
    case BackwardProduct::kDq:
      dq.writes_whole_plane = stub::dq.whole_plane;
      return stub::dq.present ? &dq : nullptr;
    case BackwardProduct::kDkdv:
      key.writes_whole_plane = stub::key.whole_plane;
      return stub::key.present ? &key : nullptr;
    case BackwardProduct::kAtomicDq:
      atomic.writes_whole_plane = stub::atomic.whole_plane;
      return stub::atomic.present ? &atomic : nullptr;
  }
  return nullptr;
}
}  // namespace xattn::ops::flash_sca_native::umma

namespace xattn::ops::flash_sca_native {

// Stub for dispatch.cu, including its existing generic tag semantics. Native
// stubs deliberately emit no tags: those must come from the driver AFTER true.
void LaunchBackwardStages(const Params& p, cudaStream_t, unsigned stages) {
  stub::events.push_back({stub::Call::kGeneric, stages});
  if (stages & kStageDelta) {
    stub::after_delta = true;
    RecordPhase(Phase::kDelta, Body::kScalar, Body::kScalar, State::kRan);
  }
  if (stages & kStageDq) {
    if (p.dq_accum == nullptr) {
      RecordPhase(Phase::kDq, Body::kMmaSync, Body::kMmaSync, State::kRan);
    } else {
      RecordPhase(Phase::kDq, Body::kNone, Body::kNone, State::kSkipped);
    }
  }
  if (stages & kStageDkdv) {
    RecordPhase(Phase::kDk, Body::kMmaSync, Body::kMmaSync, State::kRan);
    RecordPhase(Phase::kDv, Body::kMmaSync, Body::kMmaSync, State::kRan);
  }
}
}  // namespace xattn::ops::flash_sca_native

namespace {
using namespace xattn::ops::flash_sca_native;
using stub::Call;
using stub::Event;

int g_checks = 0;
int g_failures = 0;
// Distinct, aligned HOST tokens only. The stubs never read/write tensor data;
// the allocations need not contain the fictional tensors described by Params.
alignas(128) unsigned char g_tokens[24][128]{};

void Expect(bool ok, const std::string& what, long long a = 0, long long b = 0) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL %s a=%lld b=%lld\n", what.c_str(), a, b);
  }
}

Params LegalCall(Element element = Element::kHalf) {
  Params p{};
  p.element = element;
  p.batch = 2;
  p.q_length = 256;
  p.k_length = 256;
  p.prev_length = 128;
  p.total_length = 384;
  p.query_heads = 4;
  p.kv_heads = 2;
  p.group = 2;
  p.qk_dim = 128;
  p.value_dim = 128;
  p.chunk = 128;
  p.scale = 0.125f;
  p.scale_log2 = 0.18033688f;
  p.q = {g_tokens[0], 256 * 4 * 128, 4 * 128, 128};
  p.k = {g_tokens[1], 256 * 2 * 128, 2 * 128, 128};
  p.v = {g_tokens[2], 256 * 2 * 128, 2 * 128, 128};
  p.prev_k = {g_tokens[3], 128 * 2 * 128, 2 * 128, 128};
  p.prev_v = {g_tokens[4], 128 * 2 * 128, 2 * 128, 128};
  p.row_lo = reinterpret_cast<const int32_t*>(g_tokens[5]);
  p.row_hi = reinterpret_cast<const int32_t*>(g_tokens[6]);
  p.q_segment = reinterpret_cast<const int64_t*>(g_tokens[7]);
  p.k_segment = reinterpret_cast<const int64_t*>(g_tokens[8]);
  p.segment_exact = reinterpret_cast<const int32_t*>(g_tokens[9]);
  p.tile_bounds = reinterpret_cast<int32_t*>(g_tokens[10]);
  p.tile_count = 4;
  p.key_tile_range = reinterpret_cast<int32_t*>(g_tokens[11]);
  p.key_tile_count = 6;
  p.y = g_tokens[12];
  p.lse = reinterpret_cast<float*>(g_tokens[13]);
  p.dy = g_tokens[14];
  p.saved = p.y;
  p.lse_in = p.lse;
  p.delta = reinterpret_cast<float*>(g_tokens[15]);
  p.dq = g_tokens[16];
  p.dk = g_tokens[17];
  p.dv = g_tokens[18];
  p.dprev_k = g_tokens[19];
  p.dprev_v = g_tokens[20];
  return p;
}

umma::UmmaLaunchParams Prepared(const Params& p) {
  umma::UmmaLaunchParams l{};
  l.params = p;  // ALL fields, not just the element enum.
  l.maps.packed_kv = true;
  // Opaque map sentinels allow the launch tests to catch a lost prepared map.
  reinterpret_cast<unsigned char*>(&l.maps.q)[0] = 0x31;
  reinterpret_cast<unsigned char*>(&l.maps.k)[0] = 0x52;
  reinterpret_cast<unsigned char*>(&l.maps.v)[0] = 0x73;
  return l;
}

at::ScalarType Scalar(Element e) { return e == Element::kHalf ? at::kHalf : at::kBFloat16; }
float* Scratch() { return reinterpret_cast<float*>(g_tokens[21]); }
float* OtherScratch() { return reinterpret_cast<float*>(g_tokens[22]); }

umma::BackwardSupport Support(bool dq = false, bool key = false, bool atomic = false,
                              bool whole = true) {
  umma::BackwardSupport s{};
  s.dq = dq;
  s.dkdv = key;
  s.dkdv_atomic_dq = atomic;
  s.dkdv_writes_whole_plane = (key || atomic) && whole;
  return s;
}

bool SameSupport(const umma::BackwardSupport& a, const umma::BackwardSupport& b) {
  return a.dq == b.dq && a.dk == b.dk && a.dv == b.dv && a.dkdv == b.dkdv &&
         a.dkdv_atomic_dq == b.dkdv_atomic_dq &&
         a.dkdv_writes_whole_plane == b.dkdv_writes_whole_plane;
}

struct SupportOverride {
  explicit SupportOverride(const umma::BackwardSupport& support) {
    detail::SetBackwardSupportForTest(support);
  }
  ~SupportOverride() { detail::ClearBackwardSupportForTest(); }
};

template <class T>
struct Snapshot {
  unsigned char bytes[sizeof(T)];
  explicit Snapshot(const T& value) { std::memcpy(bytes, &value, sizeof(T)); }
  bool unchanged(const T& value) const { return std::memcmp(bytes, &value, sizeof(T)) == 0; }
};

bool Plan(const Params& p, const umma::UmmaLaunchParams& l, bool det, BackwardPhases* out,
          BackwardArm arm = BackwardArm::kOff) {
  const char* reason = nullptr;
  return PlanBackwardPhases(p, l, Scalar(p.element), det, out, &reason, arm);
}

bool LaunchThrows(const Params& p, const umma::UmmaLaunchParams& l,
                  const BackwardPhases& phases) {
  try {
    LaunchBackwardPhases(p, l, phases, nullptr);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void ExpectEvents(std::initializer_list<Event> expected, const std::string& label) {
  // A zero mask is a legal no-op, not an attention stage. Still verify that no
  // nonzero or native attempt escaped the expected schedule.
  std::vector<Event> actual;
  for (const auto& e : stub::events) {
    if (e.call != Call::kGeneric || e.stages != 0) actual.push_back(e);
  }
  Expect(actual.size() == expected.size(), label + "-count", actual.size(), expected.size());
  size_t i = 0;
  for (const auto& e : expected) {
    if (i < actual.size()) {
      Expect(actual[i].call == e.call && actual[i].stages == e.stages,
             label + "-order-" + std::to_string(i),
             static_cast<int>(actual[i].call), actual[i].stages);
    }
    ++i;
  }
}

struct Tag {
  Phase phase;
  Body body;
  State state = State::kRan;
};
void ExpectTags(std::initializer_list<Tag> expected, const std::string& label) {
  // Exhaust the tensor of counters: wrong bodies, duplicate rows, refusal rows,
  // unexpected forward rows, and false success tags all fail this assertion.
  int64_t want[kPhaseCount][kBodyCount][kBodyCount][kStateCount]{};
  for (const auto& t : expected) {
    ++want[static_cast<int>(t.phase)][static_cast<int>(t.body)][static_cast<int>(t.body)]
          [static_cast<int>(t.state)];
  }
  bool equal = true;
  int64_t total = 0;
  for (int p = 0; p < kPhaseCount; ++p) {
    for (int s = 0; s < kBodyCount; ++s) {
      for (int l = 0; l < kBodyCount; ++l) {
        for (int t = 0; t < kStateCount; ++t) {
          const auto n = PhaseTags().count[p][s][l][t].load(std::memory_order_relaxed);
          total += n;
          equal = equal && n == want[p][s][l][t];
        }
      }
    }
  }
  Expect(equal, label, total, expected.size());
}

// 1. No native selection: force EMPTY available support even though the test
//    registry has eligible direct entries. Determinism is the user's request;
//    the requested family and actual dQ mode are different fields.
void CheckPlan() {
  stub::Reset();
  SupportOverride no_native(Support());
  Params p = LegalCall();
  auto l = Prepared(p);
  BackwardPhases det{};
  Expect(Plan(p, l, true, &det), "P1-det-accepted");
  Expect(det.plan.family == umma::BackwardFamily::kFusedKey, "P2-det-family");
  Expect(det.plan.dq_mode == umma::DqWriteMode::kDirect, "P3-det-direct");
  Expect(!det.needs_dq_accumulator(), "P4-det-no-accumulator");
  Expect(!det.zero_prev, "P5-det-omits-nothing");
  Expect(det.generic_stages == kBackwardAllStages, "P6-det-all-generic");
  Expect(SameSupport(det.native, Support()) && SameSupport(det.available, Support()),
         "P7-det-no-leaf");
  Expect(!det.native_key_owner(), "P8-det-generic-key-owner");
  Expect(det.deterministic && det.arm == BackwardArm::kOff, "P8b-request-recorded");

  p = LegalCall(Element::kBFloat16);
  l = Prepared(p);
  BackwardPhases fast{};
  Expect(Plan(p, l, false, &fast), "P9-fast-accepted");
  Expect(fast.plan.family == umma::BackwardFamily::kFusedAtomicDq, "P10-fast-family");
  Expect(fast.plan.dq_mode == umma::DqWriteMode::kAtomic, "P11-fast-atomic");
  Expect(fast.needs_dq_accumulator(), "P12-fast-accumulator");
  Expect(fast.generic_stages == kBackwardAllStages, "P13-fast-all-generic");
  Expect(!fast.deterministic && fast.arm == BackwardArm::kOff, "P14-user-det-not-rewritten");
}

// 2. Planning refusal is pure: neither the inputs nor *out may change.
void CheckPlanRefusals() {
  stub::Reset();
  SupportOverride no_native(Support());
  const Params p = LegalCall();
  const auto l = Prepared(p);
  const char* reason = nullptr;
  Expect(!PlanBackwardPhases(p, l, at::kHalf, true, nullptr, &reason), "R1-null-out-refused");
  Expect(reason != nullptr, "R2-null-out-has-reason");

  auto refuse = [](const Params& call, const umma::UmmaLaunchParams& prepared,
                   at::ScalarType scalar, const char* name) {
    BackwardPhases out{};
    out.generic_stages = 0x5au;
    out.zero_prev = true;
    out.arm = BackwardArm::kSplit;
    out.available = Support(true, true, true);
    const Snapshot<Params> before_call(call);
    const Snapshot<umma::UmmaLaunchParams> before_prepared(prepared);
    const Snapshot<BackwardPhases> before_out(out);
    const char* why = nullptr;
    Expect(!PlanBackwardPhases(call, prepared, scalar, true, &out, &why), name);
    Expect(why != nullptr, std::string(name) + "-reason");
    Expect(before_out.unchanged(out), std::string(name) + "-out-unchanged");
    Expect(before_call.unchanged(call) && before_prepared.unchanged(prepared),
           std::string(name) + "-inputs-unchanged");
  };
  refuse(p, l, at::kBFloat16, "R3-element-arg-mismatch-refused");
  auto stale = l;
  stale.params.element = Element::kBFloat16;
  refuse(p, stale, at::kHalf, "R5-prepared-element-mismatch-refused");
  Params unknown = p;
  unknown.element = static_cast<Element>(7);
  refuse(unknown, Prepared(unknown), at::kHalf, "R6-unknown-element-refused");
  BackwardPhases out{};
  reason = nullptr;
  PlanBackwardPhases(unknown, Prepared(unknown), at::kHalf, true, &out, &reason);
  Expect(reason != nullptr && std::string(reason).find("known Element") != std::string::npos,
         "R7-unknown-element-names-itself");
  Expect(stub::events.empty(), "R8-planning-never-launches");
  ExpectTags({}, "R9-planning-never-tags");
}

// 3. The final preflight owns the scratch/coverage and prepared-call checks.
void CheckFinalPreflight() {
  stub::Reset();
  SupportOverride no_native(Support());
  Params p = LegalCall();
  auto l = Prepared(p);
  BackwardPhases direct{};
  const char* reason = nullptr;
  Expect(Plan(p, l, true, &direct), "F1-plan");
  Expect(CheckBackwardPhases(p, l, direct, &reason), "F2-legal-call-passes");

  auto forged = direct;
  forged.native.dq = true;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F3-forged-selected-dq-refused");
  Expect(reason != nullptr, "F4-forged-native-has-reason");
  forged = direct;
  forged.native.dkdv_atomic_dq = true;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F5-forged-selected-atomic-refused");

  BackwardPhases atomic{};
  Expect(Plan(p, l, false, &atomic), "F6-generic-atomic-plan");
  Expect(!CheckBackwardPhases(p, l, atomic, &reason), "F7-missing-accumulator-refused");
  Params bound = p;
  bound.dq_accum = Scratch();
  // The sole legal prepared/Params mismatch: bind a newly allocated accumulator
  // when the preparation snapshot still has a null accumulator.
  Expect(CheckBackwardPhases(bound, l, atomic, &reason), "F8-new-scratch-binding-passes");
  Expect(CheckBackwardPhases(bound, Prepared(bound), atomic, &reason),
         "F8b-matching-prebound-scratch-passes");
  Expect(!CheckBackwardPhases(bound, l, direct, &reason), "F9-direct-with-scratch-refused");
  auto stale = Prepared(bound);
  stale.params.dq_accum = OtherScratch();
  Expect(!CheckBackwardPhases(bound, stale, atomic, &reason), "F10-stale-nonnull-scratch-refused");
  Expect(!CheckBackwardPhases(p, Prepared(bound), direct, &reason),
         "F11-prepared-only-scratch-refused");

  forged = direct;
  forged.zero_prev = true;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F12-no-omitted-region-protocol");
  forged = direct;
  forged.plan.family = umma::BackwardFamily::kFusedAtomicDq;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F13-family-must-match-user-det");
  forged = direct;
  forged.plan.family = umma::BackwardFamily::kSplit;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F14-arm-must-not-rename-family");
  forged = direct;
  forged.arm = static_cast<BackwardArm>(7);
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F15-unknown-arm-refused");
  forged = direct;
  forged.plan.dq_mode = static_cast<umma::DqWriteMode>(7);
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F16-unknown-mode-refused");
  forged = atomic;
  forged.plan.dq_mode = umma::DqWriteMode::kDirect;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F17-mode-must-match-selection");
  forged = direct;
  forged.native.dk = forged.available.dk = true;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F18-unmounted-separate-dk-refused");
  forged = direct;
  forged.native.dv = forged.available.dv = true;
  Expect(!CheckBackwardPhases(p, l, forged, &reason), "F19-unmounted-separate-dv-refused");
  Expect(stub::events.empty(), "F20-preflight-never-launches");
  ExpectTags({}, "F21-preflight-never-tags");
}

// 4. Generic / one-native / both-native schedules: delta once, before all
//    products; each phase has exactly one honest post-success tag.
void CheckLaunchOrder() {
  const struct {
    const char* name;
    bool dq, key, det;
  } rows[] = {{"L-generic-direct", false, false, true},
              {"L-generic-atomic", false, false, false},
              {"L-native-dq-only", true, false, false},
              {"L-native-key-only-detoff", false, true, false},
              {"L-both-native", true, true, false}};
  for (const auto& row : rows) {
    stub::Reset();
    SupportOverride available(Support(row.dq, row.key));
    Params p = LegalCall(Element::kBFloat16);
    const auto l = Prepared(p);
    BackwardPhases phases{};
    Expect(Plan(p, l, row.det, &phases), std::string(row.name) + "-plans");
    if (phases.needs_dq_accumulator()) p.dq_accum = Scratch();
    const char* reason = nullptr;
    Expect(CheckBackwardPhases(p, l, phases, &reason), std::string(row.name) + "-final-check");
    // A non-null stream token also must survive the driver; never dereferenced.
    const auto stream = reinterpret_cast<cudaStream_t>(g_tokens[23]);
    bool threw = false;
    try {
      LaunchBackwardPhases(p, l, phases, stream);
    } catch (const std::exception&) {
      threw = true;
    }
    Expect(!threw, std::string(row.name) + "-launch-succeeds");
    if (!row.dq && !row.key) {
      ExpectEvents({{Call::kGeneric, kStageDelta}, {Call::kGeneric, kStageDq | kStageDkdv}},
                   row.name);
    } else if (row.dq && !row.key) {
      ExpectEvents({{Call::kGeneric, kStageDelta}, {Call::kDq, 0},
                    {Call::kGeneric, kStageDkdv}}, row.name);
    } else if (!row.dq && row.key) {
      ExpectEvents({{Call::kGeneric, kStageDelta}, {Call::kDkdv, 0},
                    {Call::kGeneric, kStageDq}}, row.name);
    } else {
      ExpectEvents({{Call::kGeneric, kStageDelta}, {Call::kDq, 0}, {Call::kDkdv, 0}}, row.name);
    }
    const Body dq_body = row.dq ? Body::kUmma :
                         phases.needs_dq_accumulator() ? Body::kNone : Body::kMmaSync;
    const State dq_state = phases.needs_dq_accumulator() ? State::kSkipped : State::kRan;
    const Body key_body = row.key ? Body::kUmma : Body::kMmaSync;
    ExpectTags({{Phase::kDelta, Body::kScalar}, {Phase::kDq, dq_body, dq_state},
                {Phase::kDk, key_body}, {Phase::kDv, key_body}},
               std::string(row.name) + "-exact-tags");
    for (auto* leaf : {&stub::dq, &stub::key}) {
      const bool selected = leaf == &stub::dq ? row.dq : row.key;
      Expect(leaf->launch_calls == (selected ? 1 : 0), std::string(row.name) + "-launch-count");
      if (selected) {
        Expect(leaf->last_launch.params.dq == p.dq && leaf->last_launch.params.dk == p.dk &&
                   leaf->last_launch.params.dq_accum == p.dq_accum &&
                   leaf->last_launch.params.prev_k.ptr == p.prev_k.ptr,
               std::string(row.name) + "-exact-bound-params");
        Expect(leaf->last_stream == stream, std::string(row.name) + "-same-stream");
        Expect(leaf->last_launch.maps.packed_kv == l.maps.packed_kv &&
                   std::memcmp(&leaf->last_launch.maps.q, &l.maps.q, sizeof(l.maps.q)) == 0 &&
                   std::memcmp(&leaf->last_launch.maps.k, &l.maps.k, sizeof(l.maps.k)) == 0 &&
                   std::memcmp(&leaf->last_launch.maps.v, &l.maps.v, sizeof(l.maps.v)) == 0,
               std::string(row.name) + "-prepared-maps-preserved");
      }
    }
  }
}

// 5. The launch backstop rechecks BEFORE delta. Once delta/products commit,
//    false or a fresh entry refusal throws; it must never replay a fallback.
void CheckLaunchBackstop() {
  stub::Reset();
  Params p = LegalCall();
  const auto l = Prepared(p);
  {
    SupportOverride no_native(Support());
    BackwardPhases forged{};
    Expect(Plan(p, l, true, &forged), "B0-plan");
    forged.native.dkdv = true;
    Expect(LaunchThrows(p, l, forged), "B1-forged-claim-throws");
    Expect(stub::events.empty(), "B2-nothing-before-forged-claim-throw");
    ExpectTags({}, "B3-no-tags-on-prewrite-refusal");
  }

  // Override lies at planning time; it MUST NOT bypass final Can() checks.
  for (bool refuse_key : {false, true}) {
    stub::Reset();
    SupportOverride available(Support(true, true));
    auto& leaf = refuse_key ? stub::key : stub::dq;
    leaf.can = false;
    BackwardPhases phases{};
    Expect(Plan(p, l, false, &phases), "B4-override-can-plan-claimed-support");
    const char* reason = nullptr;
    Expect(!CheckBackwardPhases(p, l, phases, &reason), "B5-exact-final-can-refuses");
    Expect(reason != nullptr && leaf.can_calls > 0, "B6-final-predicate-not-bypassed");
    Expect(LaunchThrows(p, l, phases), "B7-launch-rechecks-can-before-delta");
    Expect(stub::events.empty(), "B8-final-can-refusal-is-prewrite");
    ExpectTags({}, "B9-final-can-refusal-no-tags");
  }

  // An entry disappears between planning and commitment: also no writes.
  stub::Reset();
  {
    SupportOverride available(Support(true, true));
    BackwardPhases phases{};
    Expect(Plan(p, l, false, &phases), "B10-plan-before-entry-disappears");
    stub::key.present = false;
    Expect(LaunchThrows(p, l, phases), "B11-missing-selected-entry-throws");
    Expect(stub::events.empty(), "B12-missing-selected-entry-prewrite");
  }

  for (bool refuse_key : {false, true}) {
    stub::Reset();
    SupportOverride available(Support(true, true));
    BackwardPhases phases{};
    Expect(Plan(p, l, false, &phases), "B13-plan-before-launch-false");
    (refuse_key ? stub::key : stub::dq).launch = false;
    Expect(LaunchThrows(p, l, phases), "B14-launch-false-raises-no-fallback");
    if (refuse_key) {
      ExpectEvents({{Call::kGeneric, kStageDelta}, {Call::kDq, 0}, {Call::kDkdv, 0}},
                   "B15-key-false-no-generic-replay");
      ExpectTags({{Phase::kDelta, Body::kScalar}, {Phase::kDq, Body::kUmma}},
                 "B16-key-false-no-key-success-tags");
    } else {
      ExpectEvents({{Call::kGeneric, kStageDelta}, {Call::kDq, 0}},
                   "B17-dq-false-no-key-or-generic-replay");
      ExpectTags({{Phase::kDelta, Body::kScalar}}, "B18-dq-false-no-product-success-tags");
    }
  }

  stub::Reset();
  {
    SupportOverride available(Support(true, true));
    BackwardPhases phases{};
    Expect(Plan(p, l, false, &phases), "B19-plan-before-post-delta-refusal");
    stub::key.refuse_after_delta = true;
    const char* reason = nullptr;
    Expect(CheckBackwardPhases(p, l, phases, &reason), "B20-can-accepts-before-delta");
    Expect(LaunchThrows(p, l, phases), "B21-refusal-after-delta-throws");
    ExpectEvents({{Call::kGeneric, kStageDelta}, {Call::kDq, 0}, {Call::kDkdv, 0}},
                 "B22-refusal-after-delta-no-fallback");
    ExpectTags({{Phase::kDelta, Body::kScalar}, {Phase::kDq, Body::kUmma}},
               "B23-refusal-after-delta-no-false-success");
  }
}

// 6. Capabilities are not selected writers. Atomic (when available and allowed)
//    wins exclusively; otherwise select direct entries and preserve user family.
void CheckDecisionTable() {
  const struct {
    const char* name;
    bool dq, key, atomic, det;
    BackwardArm arm;
    bool selected_dq, selected_key, selected_atomic, direct;
    unsigned mask;
  } rows[] = {
      {"T-generic-det", false, false, false, true, BackwardArm::kOff,
       false, false, false, true, kBackwardAllStages},
      {"T-generic-fast", false, false, false, false, BackwardArm::kOff,
       false, false, false, false, kBackwardAllStages},
      {"T-dq-fast", true, false, false, false, BackwardArm::kOff,
       true, false, false, true, kStageDelta | kStageDkdv},
      {"T-key-only-detoff-legal", false, true, false, false, BackwardArm::kOff,
       false, true, false, true, kStageDelta | kStageDq},
      {"T-atomic-fast", false, false, true, false, BackwardArm::kOff,
       false, false, true, false, kStageDelta},
      {"T-atomic-ineligible-det", false, false, true, true, BackwardArm::kOff,
       false, false, false, true, kBackwardAllStages},
      {"T-dq-det", true, false, false, true, BackwardArm::kOff,
       true, false, false, true, kStageDelta | kStageDkdv},
      {"T-key-det", false, true, false, true, BackwardArm::kOff,
       false, true, false, true, kStageDelta | kStageDq},
      {"T-both-det", true, true, false, true, BackwardArm::kOff,
       true, true, false, true, kStageDelta},
      {"T-both-fast", true, true, false, false, BackwardArm::kOff,
       true, true, false, true, kStageDelta},
      {"T-all-fast-atomic-exclusive", true, true, true, false, BackwardArm::kOff,
       false, false, true, false, kStageDelta},
      {"T-dq-and-atomic-exclusive", true, false, true, false, BackwardArm::kOff,
       false, false, true, false, kStageDelta},
      {"T-key-and-atomic-exclusive", false, true, true, false, BackwardArm::kOff,
       false, false, true, false, kStageDelta},
      {"T-all-det-direct", true, true, true, true, BackwardArm::kOff,
       true, true, false, true, kStageDelta},
      {"T-all-forced-split", true, true, true, false, BackwardArm::kSplit,
       true, true, false, true, kStageDelta},
      {"T-atomic-only-forced-split", false, false, true, false, BackwardArm::kSplit,
       false, false, false, true, kBackwardAllStages},
      {"T-generic-forced-split", false, false, false, false, BackwardArm::kSplit,
       false, false, false, true, kBackwardAllStages},
      {"T-det-forced-split", false, false, false, true, BackwardArm::kSplit,
       false, false, false, true, kBackwardAllStages},
  };
  for (const auto& row : rows) {
    stub::Reset();
    stub::atomic.present = row.atomic;
    const auto support = Support(row.dq, row.key, row.atomic);
    SupportOverride available(support);
    Params p = LegalCall();
    auto l = Prepared(p);
    BackwardPhases phases{};
    const std::string name(row.name);
    Expect(Plan(p, l, row.det, &phases, row.arm), name + "-plans");
    Expect(SameSupport(phases.available, support), name + "-available-preserved");
    Expect(phases.native.dq == row.selected_dq && phases.native.dkdv == row.selected_key &&
               phases.native.dkdv_atomic_dq == row.selected_atomic &&
               !phases.native.dk && !phases.native.dv,
           name + "-selected-exclusive");
    Expect(phases.generic_stages == row.mask, name + "-mask", phases.generic_stages, row.mask);
    Expect(phases.plan.family == (row.det ? umma::BackwardFamily::kFusedKey :
                                           umma::BackwardFamily::kFusedAtomicDq),
           name + "-family-is-user-request");
    Expect(phases.deterministic == row.det && phases.arm == row.arm, name + "-arm-independent");
    Expect(phases.plan.dq_mode == (row.direct ? umma::DqWriteMode::kDirect :
                                              umma::DqWriteMode::kAtomic), name + "-actual-mode");
    Expect(phases.needs_dq_accumulator() == !row.direct, name + "-scratch");
    Expect(phases.native_key_owner() == (row.selected_key || row.selected_atomic),
           name + "-key-owner");
    Expect(!phases.zero_prev, name + "-no-implicit-zero-protocol");
    if (row.selected_key) {
      Expect(phases.native.dkdv_writes_whole_plane, name + "-selected-key-whole-plane");
    }
    if (!row.selected_atomic) {
      if (!row.direct) p.dq_accum = Scratch();
      const char* reason = nullptr;
      Expect(CheckBackwardPhases(p, l, phases, &reason), name + "-final-check");
    }
    // A native atomic body is not mounted in production. These rows test the
    // exclusive PLAN only; final Check may explicitly refuse atomic selection.
    Expect(stub::events.empty(), name + "-plan-and-check-prewrite");
  }
}

// 7. Coverage is exact, not merely a subset test. Delta cannot be omitted,
//    neither output may have zero or two producers, and no unknown mask bit runs.
void CheckStageMask() {
  stub::Reset();
  Expect(kBackwardAllStages == (kStageDelta | kStageDq | kStageDkdv), "M1-three-stage-mask");
  Expect((kStageDelta & kStageDq) == 0 && (kStageDq & kStageDkdv) == 0 &&
             (kStageDelta & kStageDkdv) == 0, "M2-distinct-mask-bits");
  Params p = LegalCall();
  auto l = Prepared(p);
  const char* reason = nullptr;
  {
    SupportOverride no_native(Support());
    BackwardPhases phases{};
    Expect(Plan(p, l, true, &phases), "M2b-plan");
    auto forged = phases;
    forged.generic_stages = kStageDelta | kStageDkdv;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M3-cleared-dq-without-writer-refused");
    Expect(reason != nullptr, "M4-cleared-dq-has-reason");
    forged = phases;
    forged.generic_stages = kStageDelta | kStageDq;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M5-cleared-key-without-writer-refused");
    forged = phases;
    forged.generic_stages = kStageDq | kStageDkdv;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M6-missing-delta-refused");
    Expect(reason != nullptr, "M7-missing-delta-has-reason");
    forged = phases;
    forged.generic_stages |= 0x40u;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M8-unknown-mask-bit-refused");
  }
  {
    SupportOverride available(Support(true, true));
    BackwardPhases phases{};
    Expect(Plan(p, l, false, &phases), "M9-native-plan");
    auto forged = phases;
    forged.generic_stages |= kStageDq;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M10-duplicate-dq-writers-refused");
    forged = phases;
    forged.generic_stages |= kStageDkdv;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M11-duplicate-key-writers-refused");
    forged = phases;
    forged.available.dq = false;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M12-selected-dq-not-available-refused");
    forged = phases;
    forged.available.dkdv = false;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M13-selected-key-not-available-refused");
    forged = phases;
    forged.available.dkdv_writes_whole_plane = false;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M14-available-key-partial-plane-refused");
    forged = phases;
    forged.native.dkdv_writes_whole_plane = false;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M15-selected-key-partial-plane-refused");
    forged = phases;
    forged.available.dkdv_writes_whole_plane = false;
    forged.native.dkdv_writes_whole_plane = false;
    Expect(!CheckBackwardPhases(p, l, forged, &reason), "M16-matching-partial-plane-still-refused");
    // A stale support record also cannot upgrade an entry's actual ownership.
    stub::key.whole_plane = false;
    Expect(!CheckBackwardPhases(p, l, phases, &reason), "M17-entry-partial-plane-refused");
  }
}

// 8. The primary mounted det-off row retains the requested family, uses direct
//    dQ, and leaves only delta generic. Keep this row named, not just tabulated.
void CheckPrimaryMountedRow() {
  stub::Reset();
  SupportOverride available(Support(true, true));
  Params p = LegalCall(Element::kBFloat16);
  const auto l = Prepared(p);
  BackwardPhases phases{};
  Expect(Plan(p, l, false, &phases), "N1-primary-mount-plans");
  Expect(phases.plan.family == umma::BackwardFamily::kFusedAtomicDq, "N2-family-is-the-request");
  Expect(phases.plan.dq_mode == umma::DqWriteMode::kDirect, "N3-dq-mode-degrades-to-direct");
  Expect(phases.generic_stages == kStageDelta, "N4-mask-is-delta-alone");
  Expect(!phases.needs_dq_accumulator(), "N5-no-accumulator");
  Expect(phases.native_key_owner(), "N6-native-owns-the-key-products");
  Expect(!phases.zero_prev, "N7-whole-plane-leaf-omits-nothing");
}

// 9. With NO override, actual registry presence and Can() produce available
//    support. Clearing an override must restore this path, not leave stale bits.
void CheckRegistrySupport() {
  stub::Reset();
  Params p = LegalCall();
  const auto l = Prepared(p);
  BackwardPhases phases{};
  Expect(Plan(p, l, false, &phases), "Q1-registered-direct-entries-plan");
  Expect(SameSupport(phases.available, Support(true, true)), "Q2-registry-support-not-empty");
  Expect(phases.dq_eligibility == LeafEligibility::kAccepted &&
             phases.key_eligibility == LeafEligibility::kAccepted,
         "Q3-accepted-entry-classification");
  Expect(stub::dq.can_calls > 0 && stub::key.can_calls > 0, "Q4-real-registry-predicates-called");
  {
    SupportOverride no_native(Support());
    Expect(Plan(p, l, true, &phases) && SameSupport(phases.available, Support()),
           "Q5-empty-override-replaces-registered-capabilities");
  }
  Expect(Plan(p, l, false, &phases) && SameSupport(phases.available, Support(true, true)),
         "Q6-clear-restores-registry");
  stub::key.can = false;
  Expect(Plan(p, l, false, &phases), "Q7-ineligible-key-plans-mixed");
  Expect(phases.available.dq && !phases.available.dkdv &&
             phases.key_eligibility == LeafEligibility::kRefused,
         "Q8-linked-but-refused-is-not-available");
  stub::dq.present = false;
  Expect(Plan(p, l, true, &phases), "Q9-unmounted-dq-plans-generic");
  Expect(!phases.available.dq && phases.dq_eligibility == LeafEligibility::kNotCompiled,
         "Q10-missing-entry-is-not-compiled");
  Expect(phases.generic_stages == kBackwardAllStages, "Q11-no-selected-native-without-support");
  Expect(std::string(EligibilityName(LeafEligibility::kNotCompiled)) == "not_compiled" &&
             std::string(EligibilityName(LeafEligibility::kRefused)) == "refused" &&
             std::string(EligibilityName(LeafEligibility::kAccepted)) == "accepted",
         "Q12-eligibility-labels");
  Expect(stub::events.empty(), "Q13-query-never-launches");

  stub::Reset();
  stub::key.whole_plane = false;
  Expect(Plan(p, l, false, &phases), "Q14-partial-key-plans-complete-mixed-route");
  Expect(phases.available.dq && !phases.available.dkdv && phases.native.dq &&
             !phases.native.dkdv && phases.generic_stages == (kStageDelta | kStageDkdv),
         "Q15-partial-key-never-selected");
  const char* reason = nullptr;
  Expect(CheckBackwardPhases(p, l, phases, &reason), "Q16-mixed-route-final-check");
  stub::dq.whole_plane = false;
  Expect(Plan(p, l, true, &phases), "Q17-partial-products-plan-generic");
  Expect(!phases.native.dq && !phases.native.dkdv &&
             phases.generic_stages == kBackwardAllStages,
         "Q18-partial-products-not-writers");
  Expect(CheckBackwardPhases(p, l, phases, &reason), "Q19-generic-route-final-check");
}

// 10. Every Params field is part of the preparation identity. Probe each field
//     individually so stale input/output/metadata pointers cannot hide behind a
//     matching dtype. dq_accum's one-way binding exception was tested in section 3.
void CheckPreparedIdentity() {
  stub::Reset();
  SupportOverride no_native(Support());
  const Params p = LegalCall();
  const auto l = Prepared(p);
  BackwardPhases phases{};
  Expect(Plan(p, l, true, &phases), "I1-identity-plan");
  const auto mismatch = [&](const umma::UmmaLaunchParams& stale, const std::string& name) {
    const char* reason = nullptr;
    Expect(!CheckBackwardPhases(p, stale, phases, &reason), name + "-refused");
    Expect(reason != nullptr, name + "-reason");
  };
  auto stale = l;
  stale.params.element = Element::kBFloat16;
  mismatch(stale, "I-element");
  Operand Params::* operands[] = {&Params::q, &Params::k, &Params::v,
                                 &Params::prev_k, &Params::prev_v};
  int64_t Operand::* strides[] = {&Operand::batch_stride, &Operand::row_stride,
                                  &Operand::head_stride};
  for (size_t i = 0; i < sizeof(operands) / sizeof(operands[0]); ++i) {
    stale = l;
    (stale.params.*operands[i]).ptr = g_tokens[22];
    mismatch(stale, "I-operand-pointer-" + std::to_string(i));
    for (size_t j = 0; j < sizeof(strides) / sizeof(strides[0]); ++j) {
      stale = l;
      (stale.params.*operands[i]).*strides[j] += 8;
      mismatch(stale, "I-operand-stride-" + std::to_string(i) + "-" + std::to_string(j));
    }
  }
  int64_t Params::* sizes[] = {&Params::batch, &Params::q_length, &Params::k_length,
      &Params::prev_length, &Params::total_length, &Params::query_heads, &Params::kv_heads,
      &Params::group, &Params::qk_dim, &Params::value_dim, &Params::chunk,
      &Params::tile_count, &Params::key_tile_count};
  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
    stale = l;
    stale.params.*sizes[i] += 1;
    mismatch(stale, "I-dimension-" + std::to_string(i));
  }
  float Params::* scales[] = {&Params::scale, &Params::scale_log2};
  for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); ++i) {
    stale = l;
    stale.params.*scales[i] += 0.25f;
    mismatch(stale, "I-scale-" + std::to_string(i));
  }
  bool Params::* flags[] = {&Params::strict, &Params::subtract_before_scale, &Params::sorted_order,
      &Params::order_always, &Params::order_dump, &Params::head_fastest, &Params::saved_fp32};
  for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); ++i) {
    stale = l;
    stale.params.*flags[i] = !(stale.params.*flags[i]);
    mismatch(stale, "I-flag-" + std::to_string(i));
  }
  const int32_t* Params::* metadata32[] = {&Params::row_lo, &Params::row_hi, &Params::segment_exact};
  for (size_t i = 0; i < sizeof(metadata32) / sizeof(metadata32[0]); ++i) {
    stale = l;
    stale.params.*metadata32[i] = reinterpret_cast<const int32_t*>(g_tokens[22]);
    mismatch(stale, "I-metadata32-" + std::to_string(i));
  }
  const int64_t* Params::* metadata64[] = {&Params::q_segment, &Params::k_segment};
  for (size_t i = 0; i < sizeof(metadata64) / sizeof(metadata64[0]); ++i) {
    stale = l;
    stale.params.*metadata64[i] = reinterpret_cast<const int64_t*>(g_tokens[22]);
    mismatch(stale, "I-metadata64-" + std::to_string(i));
  }
  int32_t* Params::* bounds[] = {&Params::tile_bounds, &Params::key_tile_range};
  for (size_t i = 0; i < sizeof(bounds) / sizeof(bounds[0]); ++i) {
    stale = l;
    stale.params.*bounds[i] = reinterpret_cast<int32_t*>(g_tokens[22]);
    mismatch(stale, "I-tile-metadata-" + std::to_string(i));
  }
  void* Params::* outputs[] = {&Params::y, &Params::dq, &Params::dk, &Params::dv,
                               &Params::dprev_k, &Params::dprev_v};
  for (size_t i = 0; i < sizeof(outputs) / sizeof(outputs[0]); ++i) {
    stale = l;
    stale.params.*outputs[i] = g_tokens[22];
    mismatch(stale, "I-output-pointer-" + std::to_string(i));
  }
  const void* Params::* inputs[] = {&Params::dy, &Params::saved};
  for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    stale = l;
    stale.params.*inputs[i] = g_tokens[22];
    mismatch(stale, "I-backward-input-" + std::to_string(i));
  }
  float* Params::* states[] = {&Params::y_state, &Params::lse, &Params::delta};
  for (size_t i = 0; i < sizeof(states) / sizeof(states[0]); ++i) {
    stale = l;
    stale.params.*states[i] = OtherScratch();
    mismatch(stale, "I-state-pointer-" + std::to_string(i));
  }
  stale = l;
  stale.params.lse_in = OtherScratch();
  mismatch(stale, "I-lse-input");
  Expect(LaunchThrows(p, stale, phases), "I2-stale-prepared-launch-throws-before-delta");
  Expect(stub::events.empty(), "I3-stale-prepared-no-launches");
  ExpectTags({}, "I4-stale-prepared-no-tags");
}

}  // namespace

int main() {
  CheckPlan();
  CheckPlanRefusals();
  CheckFinalPreflight();
  CheckLaunchOrder();
  CheckLaunchBackstop();
  CheckDecisionTable();
  CheckStageMask();
  CheckPrimaryMountedRow();
  CheckRegistrySupport();
  CheckPreparedIdentity();
  xattn::ops::flash_sca_native::detail::ClearBackwardSupportForTest();
  std::printf("{\"checks\": %d, \"failures\": %d, \"passed\": %s}\n", g_checks, g_failures,
              g_failures == 0 ? "true" : "false");
  return g_failures == 0 ? 0 : 1;
}
