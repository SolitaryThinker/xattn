// Host boundary of the native Blackwell (sm_103a) FlashSCA kernels.
//
// Validates every operand before any pointer reaches a kernel, normalizes
// layouts the kernels cannot read directly (non-unit last stride, strides that
// are not a multiple of 8 elements, bases that are not 16-byte aligned) with a
// copy, allocates outputs and launches on Q's current stream. Visibility
// metadata (row bounds, segment planes) is computed by the Python boundary
// (xattn/ops/_sca_native.py); see params.h for its meaning.
//
// Registered as xattn_cuda.ops._flash_sca_native_* by DefineNativeFlashSCAOps,
// only in builds whose XATTN_HAS_NATIVE_SCA group is complete.
//
// ---------------------------------------------------------------- the switch surface
//
// Three kinds of XATTN_* name live under this subtree and this block enumerates all
// three: the run-time switches (read with std::getenv, each on the call that uses it
// unless its row says otherwise), the compile-time knobs that change a body or its
// admission, and the instantiation macros that decide which cubins exist. The run-time
// list is CLOSED OVER THE SHIPPED SOURCES: a name absent from it is read at run time by no
// host or device source under csrc/flash_sca/native/ outside probes/, the directory no
// shipped build compiles (named at the end). Each row
// names the reader, because the reader owns the accepted spellings and the behaviour on
// every other value - or, where it has no parser, the exact names it matches or the presence
// it alone takes. The list is also a property of THIS TREE (main plus the pipelined
// forward and its instantiations): a tree without that forward reads two fewer names,
// XATTN_SCA_FWD_PIPE and XATTN_FWD_PROBE, so a read set is discovered from the source of
// the tree it is quoted for and never from memory.
//
//   XATTN_SCA_ORDER           host.cc               OrderArmOptIn()             scheduling instrument, generic forward
//   XATTN_SCA_LAYOUT          host.cc               LayoutOptIn()               grid layout instrument, generic forward
//   XATTN_SCA_UMMA_FWD        host.cc               UmmaForwardArmOptIn()       forward body: unset | "0" | "1"
//   XATTN_SCA_FWD_PIPE        umma_fwd_pipe_host.h  PipeArmValue()              forward body: the pipelined body
//   XATTN_FWD_PROBE           umma_fwd_pipe_host.h  ProbeHook()                 probe builds only: exact-name hook, refusal/fault injection
//   XATTN_SCA_BWD_ARM         host.cc               BackwardArmOptIn()          backward plan arm: fused|split|generic|native
//   XATTN_SCA_BWD             host.cc               BackwardArmOptIn()          retired; any non-empty value raises
//   XATTN_SCA_BWD_DQ_PIPE     bwd_pipe_host.h       DqPipeArmValue()            dQ body arm
//   XATTN_SCA_BWD_KEY_ARM     umma_bwd_dkdv.cuh     KeyArmUnsetRequested() /    key body arm: unset/empty/"pair128p", or
//                                                   KeyArmPair128pRequested()   any other value forces the fused leaf
//   XATTN_SCA_BWD_LEGTIME     umma_bwd.cu           LegTimeMode()               measurement only; once per process
//   XATTN_SCA_BWD_SERIAL      umma_bwd.cu           BwdLeavesSerialForced()     probe only: forces both native leaves onto one stream; per call, default off, no check sets it
//   XATTN_SCA_BWD_DIAGNOSTIC  host.cc               BackwardDiagnosticRequested()  record only, changes no number; once per process
//
// Refusal is the rule for the strict parsers, and stating it of every reader would be
// false. OrderArmOptIn, LayoutOptIn, UmmaForwardArmOptIn, PipeArmValue, BackwardArmOptIn
// (for both of its names), DqPipeArmValue and LegTimeMode accept a fixed set of spellings
// and refuse every other value (LegTimeMode's set is unset, "", "0", "1" and "2"), so an
// arm label cannot silently time a body it does not name.
// XATTN_SCA_BWD_KEY_ARM is deliberately NOT strict, and what it does instead is worth
// stating exactly, because an unknown label there is neither an error nor the default.
// Its only readers are KeyArmUnsetRequested and KeyArmPair128pRequested
// (umma_bwd_dkdv.cuh:1370-1379), and DecideInSituKeyPair (ibid.:1420-1438) turns them
// into a body, per bucket. At (256,256) at level 0, unset/empty takes the pair route (the
// dV-only leaf, then the dK-only leaf) and "pair128p" takes that route or refuses,
// while EVERY other string falls through to the fused M64 leaf; unset/empty falls back to
// the same fused leaf if either of the pair's two preflights refuses
// (UmmaBackwardDkdvCanLaunch<E, 256, 256, 2, 128, 0> and CanLaunchUmmaBackwardDkPipe). At
// (128,128) at level 0, unset/empty takes the pipelined KV leaf only when that leaf's own
// preflight accepts (CanLaunchUmmaBackwardKvPipe, which refuses an unpacked previous plane
// whose length is not a multiple of 128; a packed plane, or an unpacked one whose length is a
// multiple of 128, does not refuse on that ground) and otherwise the fused route, while every
// other string, "pair128p" included, takes the fused route. Above level 0, and in every other
// bucket, every value, unset included, takes the fused route. So a misspelt label is accepted
// rather than refused and forces the fused route: at (256,256) at level 0 with the pair
// preflight accepting that is a body neither unset/empty nor "pair128p" selects; at (128,128)
// at level 0 it is the route "pair128p" always reaches and the route unset/empty reaches only
// when the KV leaf's preflight refuses, not when it accepts; and above level 0, or in any other
// bucket, it is the route every label reaches.
// XATTN_FWD_PROBE is NOT a presence reader and must not be grouped with one. It is an
// exact-name hook selector that a shipped build does not read at all: its definition and all
// three of its call sites sit inside XATTN_FWD_PROBE_HOOKS (umma_fwd_pipe_host.h:144-157,
// the getenv itself at :154), a macro no source of this package defines. In a probe build its
// three recognised values deliberately leave the healthy path, and they are why the sentence
// above is about the reader and not about a spelling set: "pipe-refuse" returns
// TryLaunchPipe's pre-launch refusal (umma_fwd_pipe.cuh:731), so the caller's two-row
// fallback runs and the call executes a DIFFERENT body; "pipe-posterr" (the empty-grid arm of
// that same function) and "leaf-posterr" (umma_fwd.cuh:727, the empty-grid arm of
// LaunchUmmaForward) make the launch call report an error, so the post-attempt raise runs
// where the fallback otherwise would. The first changes routing and the other two change
// failure; none of them refuses a value. An unrecognised value is simply inactive, because
// ProbeHook compares exact names and has no parser.
// XATTN_SCA_BWD_DIAGNOSTIC is the presence reader that sentence is about:
// BackwardDiagnosticRequested (host.cc) reads the name once per process, takes set, non-empty
// and not "0" as the request, has no spelling to refuse, and changes no number - the published
// record reports it and nothing else does.
//
// By kind: five select a body (XATTN_SCA_UMMA_FWD, XATTN_SCA_FWD_PIPE, XATTN_SCA_BWD_ARM and
// the two leaf arms under it, XATTN_SCA_BWD_DQ_PIPE and XATTN_SCA_BWD_KEY_ARM); two schedule
// the generic forward body and keep every native body out of the call while they are live
// (XATTN_SCA_ORDER, XATTN_SCA_LAYOUT); two measure or record without changing a number
// (XATTN_SCA_BWD_LEGTIME, XATTN_SCA_BWD_DIAGNOSTIC); one is retired and refuses rather than
// being ignored (XATTN_SCA_BWD); one exists only in a probe build and there injects a refusal
// or a fault rather than selecting a body (XATTN_FWD_PROBE).
//
// COMPILE-TIME KNOBS, defined by a build flag and never read at run time. They change a
// body or its admission, so they belong on this page even though getenv does not see them:
// XATTN_TARGET_SM (the build's target SMs - set by setup.py and read by the build, not
// by these sources, so it is a build input and never an arm), XATTN_UMMA_DEVICE_ASM (the device-asm guard),
// XATTN_FWD_PROBE_HOOKS (defines the probe hooks; defined in no source and by no build flag),
// XATTN_BWD_MIN_BLOCKS (umma_bwd_dkdv.cuh:440 - the key leaf's __launch_bounds__ minimum
// CTAs per SM; one is the measured form and two is measured slower, see the comment there),
// XATTN_SCA_BWD_TESTING (umma_bwd.h:47, umma_bwd.cu:17,342 - enables the backward test
// hooks), XATTN_DQ_ROUND_STAT (umma_bwd_dq.cuh:532 onwards - enables the dQ round
// instrumentation) and XATTN_UMMA_PREPARE_FORCE_BUCKET (umma_prepare.h:44,176-180 - bypasses
// the compiled-bucket admission test so a test can reach a bucket the build did not
// instantiate).
//
// INSTANTIATION MACROS, which decide which cubins exist and never which body a call runs:
// XATTN_SCA_NATIVE_INSTANTIATE, XATTN_UMMA_FWD_INSTANTIATE, XATTN_UMMA_BWD_DKDV_INSTANTIATE,
// XATTN_UMMA_BWD_DKDV_M128_INSTANTIATE, XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE,
// XATTN_UMMA_BWD_KEY_INSTANTIATE, XATTN_UMMA_DQ_INSTANTIATE, XATTN_UMMA_DQ_PIPE_INSTANTIATE
// and XATTN_FWD_PIPE_INSTANTIATE.
//
// Every other XATTN_* identifier under the subtree is a structural macro and not a switch:
// include guards (XATTN_UMMA_COMMON_LOCAL_INCLUDED, XATTN_UMMA_COMMON_SHARED_INCLUDED), table
// and label macros (XATTN_SCA_ENTRY, XATTN_SCA_BUCKET, XATTN_SCA_MATCH_BUCKET,
// XATTN_SCA_NATIVE_CASE, XATTN_SCA_BACKWARD_BUCKETS, XATTN_SCA_SAME, XATTN_DQP_BAR,
// XATTN_TMA_CHECK, XATTN_TMA_TRAP, XATTN_UMMA_CHECK, XATTN_UMMA_HD), the build-group name
// XATTN_HAS_NATIVE_SCA, and one stderr tag that is printed rather than read
// (XATTN_FWD_PIPE_RESCALE_EVENTS, umma_fwd_pipe.cuh:784). The receipt for that sentence is
// the census script's own identifier list, not a reading of this comment.
//
// Outside the subtree, named so a reader can see the boundary: XATTN_ATTENTION_VARLEN_SCHEDULER
// and XATTN_FLASH_SCA_VARLEN_SCHEDULER are read in csrc/attention/hopper/tile_scheduler.hpp,
// which no sm_103a build of this subtree compiles. Excluded from the closed set and named
// here for the same reason: DQ_PROBE_INJECT_REFUSAL is read at
// csrc/flash_sca/native/probes/dq_probe.cu:1107, a probe translation unit that no shipped
// build compiles.
#include "bindings.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/python_variable.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "flash_sca/native/bwd_pipe_host.h"
#include "flash_sca/native/dispatch_tag.cuh"
#include "flash_sca/native/params.h"
#include "flash_sca/native/registration.h"
#include "flash_sca/native/row_bounds.h"
#include "flash_sca/native/umma_bwd.h"
#include "flash_sca/native/row_bounds.h"
#include "flash_sca/native/umma_fwd_pipe_host.h"
#include "flash_sca/native/umma_prepare.h"

namespace xattn {
namespace ops {
namespace flash_sca_native {

// The backward preparation lives beside the operands it builds; the seam reads it
// under this shorter name, and the phase driver above it under its own.
namespace bwd_host = umma::bwd_host;

namespace {

std::atomic<int64_t> g_forward_launches{0};
std::atomic<int64_t> g_backward_launches{0};
// One sequence number per Backward() entry, from 1, in entry order over every host thread
// of the process. It is the stamp that ties a published record to the call that made it.
std::atomic<int64_t> g_backward_calls{0};

// The diagnostic record of one backward call, not a synchronization or completion
// guarantee. Backward() fills the calling thread's working copy (g_last_backward below)
// and, on every exit including a raise, publishes it as the one process-wide record that
// LastBackwardPreparation reads (PublishBackwardRecord). Under autograd the backward runs
// on the engine's thread and the snapshot is read on the caller's, so a record is
// attributed by its stamps (seq, recording thread, launch counters), never by the
// thread that reads it. Product phase counters are recorded at the actual launch sites.
//
// The record holds host-side values only. This build performs no device readback for the
// record: no field is copied from device memory, and no CUDA call fills, publishes or reads
// it.
struct BackwardDiagnostic {
  const char* refusal = "not_called";
  BackwardPhases phases{};
  bool prepared = false;
  bool packed = false;
  bool launched = false;
  bool has_accumulator = false;
  int qk_bucket = 0;
  int value_bucket = 0;
  // The key-tile height (M) of the key body this call RAN: overwritten after the launches
  // from umma::InSituLaunchedKeyRows(), which the in-situ launch pair sets to the tile it
  // actually launched. The pre-launch value read from the registry row's
  // BackwardLeafEntry::key_rows is the arm the BUCKET PREFERS and can overstate it - at
  // (128,128) the M128 arm refuses an unpacked previous plane whose length is not a
  // multiple of 128 and that call runs the M64 body - so read this field, never the row's,
  // and read it from a record whose stamps match the call it is quoted for.
  int key_rows = 0;
  // The clipped-bounds proof, read from the params the backward LAUNCHED with: whether both
  // row bound planes were present (Python `_sca_native.py:visibility()`'s run_lo/run_hi,
  // built for bos_mask/segment calls) and whether the 64-row key-tile table was built and
  // handed on (host.cc builds it iff row_lo is set, so False + row_bounds True is the leaf's
  // "every query tile is a candidate" envelope of QueryTileEnvelope, not a table walk).
  bool row_bounds = false;
  bool key_tile_range = false;
  // The table itself is device memory that only the kernels write and read. No entry of it
  // is copied back, so the record has no field for one; the snapshot's key_tile_range_first*
  // and key_tile_sample* keys are fixed placeholders (LastBackwardPreparation).
  int64_t prev_length = 0;
  bool force_generic = false;
  bool require_native = false;
  // Owned copy of the XATTN_SCA_BWD_ARM value this call parsed (nullopt = unset);
  // never getenv's pointer, which a later setenv in the process may invalidate.
  std::optional<std::string> raw_arm;
  // Owned copy of the retired XATTN_SCA_BWD value this call read (nullopt = unset,
  // "" = set but empty). Additive: the older key's meaning is unchanged.
  std::optional<std::string> raw_retired_bwd;
  // Attribution. `seq` is set when the call enters Backward (0 = no backward call has been
  // recorded in this process); the other three are stamped when the record is published:
  // a hash of the recording host thread's id, and g_forward_launches / g_backward_launches
  // at that moment (the call's own backward launch included). A caller proves a record is
  // its own by these stamps, e.g. by matching the launch counters it read around its call.
  int64_t seq = 0;
  uint64_t recording_thread = 0;
  // The recording thread's id itself: the hash above is only the printable form, and a hash
  // can collide, so recorded_on_this_thread compares ids.
  std::thread::id recording_thread_id{};
  int64_t forward_launches = 0;
  int64_t backward_launches = 0;
};
// The calling thread's working record for the backward it is running. The snapshot never
// reads it: PublishBackwardRecord moves it out when the call leaves Backward.
thread_local BackwardDiagnostic g_last_backward{};

uint64_t ThisThreadStamp() {
  return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

// XATTN_SCA_BWD_DIAGNOSTIC: set, non-empty and not "0" requests the backward diagnostic.
// Resolved once per process. This build performs no device readback for the record, so a
// request changes nothing a call does or records: the snapshot only reports it, as
// `diagnostic_requested`, and its key-tile sample keys read the same fixed values either way.
bool BackwardDiagnosticRequested() {
  static const bool requested = [] {
    const char* v = std::getenv("XATTN_SCA_BWD_DIAGNOSTIC");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  }();
  return requested;
}

// The published record, in host memory and guarded by this one mutex.
std::mutex g_backward_record_mutex;
BackwardDiagnostic g_published_backward;  // guarded by g_backward_record_mutex

// Publish the calling thread's working record as the process-wide one, stamped with the
// recording thread and the launch counters. Runs when Backward exits, by return or by a
// raise (BackwardRecordPublisher), so a refused call is readable from any thread too. The
// working copy is moved, not copied: nothing reads it again before the next call resets it.
// Moving its strings hands their buffers over within one allocator, so taking the lock is
// the step expected to throw; the publisher's try covers every step regardless.
void PublishBackwardRecord() {
  BackwardDiagnostic d = std::move(g_last_backward);
  d.recording_thread = ThisThreadStamp();
  d.recording_thread_id = std::this_thread::get_id();
  d.forward_launches = g_forward_launches.load(std::memory_order_relaxed);
  d.backward_launches = g_backward_launches.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_backward_record_mutex);
  g_published_backward = std::move(d);
}

// Guards host memory only: the record and its mutex, no device resource. Destructors are
// noexcept, and no exception escapes this one: a failure to publish leaves the previous
// record in place rather than terminating the process or replacing the exception a refused
// call is already raising. A reader tells a stale record by its stamps.
struct BackwardRecordPublisher {
  ~BackwardRecordPublisher() {
    try {
      PublishBackwardRecord();
    } catch (...) {
    }
  }
};

// Record, from the params the backward launched with, whether this call carried segment-
// clipped row bounds and the key-tile table. Two pointer tests on host memory: no CUDA call,
// no lock, nothing read from the device.
void RecordRowBounds(const Params& launched) {
  BackwardDiagnostic& d = g_last_backward;
  d.row_bounds = launched.row_lo != nullptr && launched.row_hi != nullptr;
  d.key_tile_range = launched.key_tile_range != nullptr;
}

// XATTN_SCA_ORDER: unset = today's map; "sdesc" = the permutation on shallow
// grids only (kOrderMaxCtas); "sdesc-all" = on every grid; "dump" =
// sdesc-all plus the device-side rank map. Any other value refuses the call:
// an instrument whose whole purpose is to label an A/B arm must not fall back
// silently to a third path. Read per call (no caching), so one process can
// time both arms in one session.
enum class OrderArm { kNone, kGated, kAlways, kDump };

OrderArm OrderArmOptIn() {
  const char* value = std::getenv("XATTN_SCA_ORDER");
  if (value == nullptr) return OrderArm::kNone;
  const std::string arm(value);
  if (arm == "sdesc") return OrderArm::kGated;
  if (arm == "sdesc-all") return OrderArm::kAlways;
  if (arm == "dump") return OrderArm::kDump;
  TORCH_CHECK(false,
              "XATTN_SCA_ORDER must be unset, 'sdesc', 'sdesc-all' or 'dump'; got '", arm, "'");
  return OrderArm::kNone;
}

// XATTN_SCA_LAYOUT: unset or "tf" = the base grid, "hf" = head-fastest
// grid(Hq, tiles, B); any other value refuses the call, same reason.
bool LayoutOptIn() {
  const char* value = std::getenv("XATTN_SCA_LAYOUT");
  if (value == nullptr) return false;
  const std::string layout(value);
  if (layout == "hf") return true;
  if (layout == "tf") return false;
  TORCH_CHECK(false, "XATTN_SCA_LAYOUT must be unset, 'tf' or 'hf'; got '", layout, "'");
  return false;
}

// Head-fastest layout (opt-in, default off): XATTN_SCA_LAYOUT=hf launches the
// forward grid head-fastest, grid(Hq, tiles, B), so the Hq copies of one query
// tile are consecutive CTAs. Default "tf" (and unset) is the base grid. The
// exact-value form above is the only reader; a lenient reader that accepted
// any value starting with 'h' would create a
// silent third arm in an instrument built to label arms.

// XATTN_SCA_BWD_ARM is read once per call. Unset, set-but-empty and "fused" select
// eligible producers; "split" additionally requires direct dQ. "generic" skips native
// preparation and uses the generic bodies. "native" requires BOTH native
// products; a mixed plan is legal for the default but refused by this strict
// measurement arm. User determinism is recorded separately from actual dQ
// write mode: a det-off call with direct native producers needs no accumulator.
// Unknown values raise rather than silently timing a different arm.
// The separate body arm XATTN_SCA_BWD ("0" = force generic, "1" = require native)
// is folded into this one parser as "generic" / "native". A set XATTN_SCA_BWD
// raises instead of being ignored, so a script written for it cannot time the
// default arm under that arm's label. Both variables treat the empty string as
// unset; the raw value is still recorded, so the snapshot shows "" rather than None.
struct BackwardRequest {
  BackwardArm arm = BackwardArm::kOff;
  bool force_generic = false;
  bool require_native = false;
  // The raw XATTN_SCA_BWD_ARM value, COPIED. getenv's pointer may be invalidated by a
  // later setenv/putenv in the same process (an A/B harness flips the variable between
  // calls), so the request and the per-call diagnostic own their bytes. nullopt = unset.
  std::optional<std::string> raw_arm;
};

BackwardRequest BackwardArmOptIn() {
  // Both raw values are stored AT THE READ, above BOTH refusals (the
  // retired name's, then the unknown arm's), so a refused call can still say what
  // the environment said. A refused call that cannot distinguish unset from
  // "native" cannot tell a script which arm it asked for. Each store is
  // null-guarded: getenv's null is nullopt, never std::string(nullptr), which is
  // undefined behaviour. One store per raw value, and no second writer in the call.
  const char* retired = std::getenv("XATTN_SCA_BWD");
  const char* arm_value = std::getenv("XATTN_SCA_BWD_ARM");
  g_last_backward.raw_retired_bwd = retired == nullptr
      ? std::optional<std::string>{}
      : std::optional<std::string>(std::string(retired));
  g_last_backward.raw_arm = arm_value == nullptr
      ? std::optional<std::string>{}
      : std::optional<std::string>(std::string(arm_value));
  TORCH_CHECK(retired == nullptr || *retired == '\0',
              "XATTN_SCA_BWD is retired: use XATTN_SCA_BWD_ARM=generic (was XATTN_SCA_BWD=0) or "
              "XATTN_SCA_BWD_ARM=native (was XATTN_SCA_BWD=1); got XATTN_SCA_BWD='",
              std::string(retired == nullptr ? "" : retired), "'");
  const char* value = arm_value;
  if (value == nullptr || *value == '\0') return {};
  const std::string arm(value);
  if (arm == "fused") return {BackwardArm::kOff, false, false, arm};
  if (arm == "split") return {BackwardArm::kSplit, false, false, arm};
  if (arm == "generic") return {BackwardArm::kOff, true, false, arm};
  if (arm == "native") return {BackwardArm::kOff, false, true, arm};
  TORCH_CHECK(false,
              "XATTN_SCA_BWD_ARM must be unset, 'fused', 'split', 'generic' or 'native'; got '",
              arm, "'");
  return {};
}

using OptionalTensor = std::optional<at::Tensor>;

bool Readable(const at::Tensor& t) {
  if (t.numel() == 0) return true;
  if (t.stride(-1) != 1) return false;
  for (int64_t d = 0; d + 1 < t.dim(); ++d) {
    // `stride <= 0` is not redundant with `% 8`: a broadcast view (stride 0 on
    // an axis of size > 1) passes `% 8`. Without this term Prepare() returned
    // such a view unchanged and the backward's TMA preparation, which refuses
    // any normalised stride <= 0 (MirrorEncodeChecks), answered kMapSource, so
    // the call silently lost the native arm instead of being materialised here.
    if (t.size(d) > 1 && (t.stride(d) <= 0 || t.stride(d) % 8 != 0)) return false;
  }
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

// Same values, in a layout the kernels read with 16-byte copies.
at::Tensor Prepare(const at::Tensor& t) {
  if (Readable(t)) return t;
  at::Tensor packed = t.contiguous();
  if (!Readable(packed)) packed = packed.clone(at::MemoryFormat::Contiguous);
  TORCH_CHECK(Readable(packed), "native FlashSCA could not align an operand");
  return packed;
}

at::Tensor PrepareContiguous(const at::Tensor& t) {
  at::Tensor packed = t.contiguous();
  if (packed.numel() && reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 != 0) {
    packed = packed.clone(at::MemoryFormat::Contiguous);
  }
  return packed;
}

Operand MakeOperand(const at::Tensor& t) {
  Operand o;
  o.ptr = t.data_ptr();
  o.batch_stride = t.stride(0);
  o.row_stride = t.stride(1);
  o.head_stride = t.stride(2);
  return o;
}

void CheckDevice(const at::Tensor& t, const at::Tensor& q, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == q.device(), "native FlashSCA ", name,
              " must be a CUDA tensor on Q's device");
}

struct Prepared {
  Params params;
  at::Tensor q, k, v, prev_k, prev_v, q_segment, k_segment, segment_exact, row_lo, row_hi;
  at::Tensor tile_bounds;
};

// Shared validation of the forward and backward operands and metadata.
Prepared PrepareCommon(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const OptionalTensor& prev_k, const OptionalTensor& prev_v,
                       int64_t chunk_size, double scale, const OptionalTensor& q_segment,
                       const OptionalTensor& k_segment, const OptionalTensor& segment_exact,
                       const OptionalTensor& row_lo, const OptionalTensor& row_hi, bool strict) {
  TORCH_CHECK(q.is_cuda(), "native FlashSCA requires CUDA tensors");
  CheckDevice(k, q, "k");
  CheckDevice(v, q, "v");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "native FlashSCA expects Q/K/V of shape [B,L,H,D]");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "native FlashSCA supports float16 and bfloat16 inputs");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "native FlashSCA Q/K/V must share a dtype");
  const int64_t B = q.size(0), Nq = q.size(1), Hq = q.size(2), D = q.size(3);
  const int64_t Nk = k.size(1), Hkv = k.size(2), Dv = v.size(3);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == Nk && v.size(2) == Hkv &&
                  k.size(3) == D,
              "native FlashSCA Q/K/V shapes are inconsistent");
  TORCH_CHECK(Hq > 0 && Hkv > 0 && Hq % Hkv == 0, "native FlashSCA requires Hq % Hkv == 0");
  TORCH_CHECK(D > 0 && D <= 256 && D % 8 == 0 && Dv > 0 && Dv <= 256 && Dv % 8 == 0,
              "native FlashSCA requires D and Dv to be multiples of 8 in [8, 256]; got D=", D,
              ", Dv=", Dv);
  TORCH_CHECK(chunk_size > 0, "native FlashSCA chunk_size must be positive");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "native FlashSCA requires a positive finite scale (the Python boundary "
              "normalizes zero and negative scales)");
  TORCH_CHECK(prev_k.has_value() == prev_v.has_value(),
              "native FlashSCA prev_k and prev_v must be given together");
  int64_t P = 0;
  if (prev_k) {
    CheckDevice(*prev_k, q, "prev_k");
    CheckDevice(*prev_v, q, "prev_v");
    TORCH_CHECK(prev_k->dim() == 4 && prev_v->dim() == 4 &&
                    prev_k->scalar_type() == q.scalar_type() &&
                    prev_v->scalar_type() == q.scalar_type(),
                "native FlashSCA previous K/V must be 4D tensors of Q's dtype");
    P = prev_k->size(1);
    TORCH_CHECK(prev_k->size(0) == B && prev_k->size(2) == Hkv && prev_k->size(3) == D &&
                    prev_v->size(0) == B && prev_v->size(1) == P && prev_v->size(2) == Hkv &&
                    prev_v->size(3) == Dv,
                "native FlashSCA previous K/V shapes are inconsistent");
  }
  const int64_t Nt = P + Nk;
  TORCH_CHECK(Nq > 0 && Nq <= Nt, "native FlashSCA requires 0 < Nq <= total K length");
  TORCH_CHECK(Nt < (int64_t(1) << 31) - 64 && B <= 65535 && Hq <= 65535,
              "native FlashSCA problem exceeds the launch limits");
  TORCH_CHECK(q_segment.has_value() == k_segment.has_value(),
              "native FlashSCA segment planes must be given together");
  TORCH_CHECK(row_lo.has_value() == row_hi.has_value(),
              "native FlashSCA row bounds must be given together");

  Prepared out;
  out.q = Prepare(q);
  out.k = Prepare(k);
  out.v = Prepare(v);
  Params& p = out.params;
  p.element = q.scalar_type() == at::kHalf ? Element::kHalf : Element::kBFloat16;
  p.q = MakeOperand(out.q);
  p.k = MakeOperand(out.k);
  p.v = MakeOperand(out.v);
  if (prev_k) {
    out.prev_k = Prepare(*prev_k);
    out.prev_v = Prepare(*prev_v);
    p.prev_k = MakeOperand(out.prev_k);
    p.prev_v = MakeOperand(out.prev_v);
  }
  if (q_segment) {
    for (const auto* t : {&*q_segment, &*k_segment}) {
      CheckDevice(*t, q, "segment plane");
      TORCH_CHECK(t->scalar_type() == at::kLong && t->dim() == 2 && t->size(0) == B,
                  "native FlashSCA segment planes must be int64 [B, L]");
    }
    TORCH_CHECK(q_segment->size(1) == Nq && k_segment->size(1) == Nt,
                "native FlashSCA segment planes must be [B,Nq] and [B,Nt]");
    out.q_segment = q_segment->contiguous();
    out.k_segment = k_segment->contiguous();
    p.q_segment = out.q_segment.data_ptr<int64_t>();
    p.k_segment = out.k_segment.data_ptr<int64_t>();
    if (segment_exact) {
      CheckDevice(*segment_exact, q, "segment_exact");
      TORCH_CHECK(segment_exact->scalar_type() == at::kInt && segment_exact->numel() == B,
                  "native FlashSCA segment_exact must be int32 [B]");
      out.segment_exact = segment_exact->contiguous();
      p.segment_exact = out.segment_exact.data_ptr<int32_t>();
    }
  } else {
    TORCH_CHECK(!segment_exact.has_value(), "native FlashSCA segment_exact needs segment planes");
  }
  if (row_lo) {
    for (const auto* t : {&*row_lo, &*row_hi}) {
      CheckDevice(*t, q, "row bounds");
      TORCH_CHECK(t->scalar_type() == at::kInt && t->dim() == 2 && t->size(0) == B &&
                      t->size(1) == Nq,
                  "native FlashSCA row bounds must be int32 [B, Nq]");
    }
    out.row_lo = row_lo->contiguous();
    out.row_hi = row_hi->contiguous();
    p.row_lo = out.row_lo.data_ptr<int32_t>();
    p.row_hi = out.row_hi.data_ptr<int32_t>();
  }

  p.batch = B;
  p.q_length = Nq;
  p.k_length = Nk;
  p.prev_length = P;
  p.total_length = Nt;
  p.query_heads = Hq;
  p.kv_heads = Hkv;
  p.group = Hq / Hkv;
  p.qk_dim = D;
  p.value_dim = Dv;
  p.chunk = chunk_size;
  p.scale = static_cast<float>(scale);
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  p.strict = strict;
  p.tile_count = (Nq + 63) / 64;
  // Params::tile_bounds is NOT built here: the tile-bounds table is read only by the bodies
  // that walk 64-row key tiles through it (the generic mma.sync forward, the older native
  // leaf, umma_tile_mask.h, and the backward's key leaves), and the pipelined forward derives
  // its tile walk from the per-row bounds in-kernel (umma_fwd_pipe.cuh: RowBounds -> min_lo /
  // max_hi), so it reads no table. Each caller therefore builds the table for the bodies it
  // resolved to, right before the call that consumes it: host.cc Forward does so only for a
  // body other than kPipe (and keeps it under the scheduling instruments, which resolve
  // kGeneric), Backward builds it for every schedule, and a call whose body reads no table
  // pays neither the allocation nor the LaunchTileBounds launch.
  return out;
}

// XATTN_SCA_UMMA_FWD: the forward-body switch, and the only instrument that
// makes a same-binary forward A/B possible. MAIN'S BODY IS THE DEFAULT: unset and
// "0" both run the generic `mma.sync` body, and they leave the SAME phase row
// behind -- the tag table has no arm field, so neither is distinguishable from
// the other in a retained artifact. "1" =
// require a native leaf and refuse the call when this build has none for the
// bucket, so the require arm can never fall back and pass as if it had served.
// Any other value refuses: an instrument whose purpose is to label an arm must not
// silently time a third path. Read per call (no caching), so one process times
// both arms in one session.
//
// The native leaf is therefore reachable ONLY through "1". That is deliberate: a
// default that selects the leaf would put the native forward's cost on every
// `backend='auto'` call, traced or not, so the default stays main's generic
// body. `kAuto` stays in the enum to keep the parser's three spellings
// distinct at the source level; it selects no body, and it leaves no tag that
// would distinguish it from "0" either.
//
// One body precedes this switch's default: the pipelined level-0 body at (256,256) and (128,128)
// (XATTN_SCA_FWD_PIPE, umma_fwd_pipe_host.h) serves the calls it accepts when BOTH
// switches are unset or empty. An explicit value of this switch keeps the meaning
// above - "0" the generic body, "1" the leaf, never the pipelined body - so a
// control row taken with it keeps its referent, and XATTN_SCA_FWD_PIPE=1 together
// with an explicit value here raises. The precedence is ResolveForwardBody below.
enum class UmmaForwardArm { kAuto, kOff, kRequire };

UmmaForwardArm UmmaForwardArmOptIn() {
  const char* value = std::getenv("XATTN_SCA_UMMA_FWD");
  if (value == nullptr || *value == '\0') return UmmaForwardArm::kAuto;
  const std::string arm(value);
  if (arm == "0") return UmmaForwardArm::kOff;
  if (arm == "1") return UmmaForwardArm::kRequire;
  TORCH_CHECK(false, "XATTN_SCA_UMMA_FWD must be unset, '0' or '1'; got '", arm, "'");
  return UmmaForwardArm::kAuto;
}

// The host needs the answer before it prepares a call (the preparation
// allocates), and `launch.cuh` cannot be included here without pulling every
// kernel body into this translation unit, so both read the one list in params.h.
bool HasUmmaForwardLeaf(int64_t qk_dim, int64_t value_dim) {
  return IsUmmaForwardDims(qk_dim, value_dim);
}

// The two forward arms of one call, XATTN_SCA_FWD_PIPE (umma_fwd_pipe_host.h) and
// XATTN_SCA_UMMA_FWD (above), parsed ONCE per call and before anything is allocated.
// Raises on a value that names no arm and on two explicit choices that name different
// bodies, so no arm label can time a body it does not name.
struct ForwardArms {
  umma::fwdpipe::PipeArm pipe;
  UmmaForwardArm umma;
};

ForwardArms ParseForwardArms() {
  const UmmaForwardArm umma_arm = UmmaForwardArmOptIn();
  const char* const pipe_value = umma::fwdpipe::PipeArmValue();
  const umma::fwdpipe::PipeArm pipe_arm = umma::fwdpipe::ParsePipeArm(pipe_value);
  TORCH_CHECK(pipe_arm != umma::fwdpipe::PipeArm::kInvalid,
              "XATTN_SCA_FWD_PIPE must be unset, empty, '0' or '1'; got '",
              std::string(pipe_value != nullptr ? pipe_value : ""), "'");
  TORCH_CHECK(!(pipe_arm == umma::fwdpipe::PipeArm::kOn && umma_arm != UmmaForwardArm::kAuto),
              "XATTN_SCA_FWD_PIPE=1 conflicts with XATTN_SCA_UMMA_FWD=",
              umma_arm == UmmaForwardArm::kRequire ? "1" : "0",
              ": both name a forward body. Unset one of them; XATTN_SCA_FWD_PIPE=0 gives "
              "main's forward, in which XATTN_SCA_UMMA_FWD decides");
  return {pipe_arm, umma_arm};
}

// The body one forward call runs, resolved ONCE from both arms; host.cc's map
// preparation and the launch site both read this one value. The precedence:
//   PIPE unset/empty, UMMA_FWD unset/empty -> kPipe where PipeAccepts admits the call and
//                                             no scheduling instrument is live, else kGeneric
//                                             (main's default forward)
//   PIPE unset/empty, UMMA_FWD "0"         -> kGeneric (main's "0")
//   PIPE unset/empty, UMMA_FWD "1"         -> kLeaf    (main's "1"; never the pipelined body)
//   PIPE "1",         UMMA_FWD unset/empty -> as the first row: "1" is the explicit spelling
//                                             of on, and a call PipeAccepts refuses runs
//                                             main's default forward
//   PIPE "1",         UMMA_FWD "0" or "1"  -> raised in ParseForwardArms (conflict)
//   PIPE "0"                               -> main's forward: kLeaf under UMMA_FWD "1", else
//                                             kGeneric
//   PIPE any other value, "2" included     -> raised in ParseForwardArms
umma::fwdpipe::ForwardBody ResolveForwardBody(const ForwardArms& arms, bool pipe_accepts,
                                              bool order_instrument) {
  using umma::fwdpipe::ForwardBody;
  if (arms.pipe == umma::fwdpipe::PipeArm::kOff || arms.umma != UmmaForwardArm::kAuto) {
    return arms.umma == UmmaForwardArm::kRequire ? ForwardBody::kLeaf : ForwardBody::kGeneric;
  }
  return pipe_accepts && !order_instrument ? ForwardBody::kPipe : ForwardBody::kGeneric;
}

// Defined below, next to RowBounds: the mask-side metadata boundary (row_bounds.h).
std::tuple<at::Tensor, at::Tensor> RowBoundsFromMask(const at::Tensor& mask, int64_t q_length,
                                                     int64_t total, int64_t chunk_size,
                                                     bool strict);
void CheckRowBoundsMaskArgs(const at::Tensor& mask, int64_t q_length, int64_t total,
                            int64_t chunk_size);
void LaunchRowBoundsMaskInto(const at::Tensor& mask_c, int64_t q_length, int64_t total,
                             int64_t chunk_size, bool strict, int32_t* lo, int32_t* hi,
                             cudaStream_t stream, int32_t* tile_bounds_out = nullptr,
                             bool tile128 = false);

// A contiguous, uninitialized CUDA tensor from the CUDA caching allocator: what at::empty makes
// for these options, without the operator dispatch at::empty goes through first.
at::Tensor EmptyCuda(at::IntArrayRef size, at::ScalarType dtype, at::Device device) {
  return at::detail::empty_cuda(size, dtype, device, std::nullopt);
}

std::tuple<at::Tensor, at::Tensor> Forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const OptionalTensor& prev_k,
    const OptionalTensor& prev_v, int64_t chunk_size, double scale,
    const OptionalTensor& q_segment, const OptionalTensor& k_segment,
    const OptionalTensor& segment_exact, const OptionalTensor& row_lo,
    const OptionalTensor& row_hi, bool strict, bool subtract_before_scale,
    const OptionalTensor& output_state, const OptionalTensor& bos_mask) {
  c10::cuda::CUDAGuard guard(q.device());
  // Both forward arms, parsed ONCE for this call and first: every refusal of an arm value, or of
  // a combination of the two, raises here, before anything is allocated (ParseForwardArms).
  const ForwardArms arms = ParseForwardArms();
  // A BOS mask given here is the whole metadata boundary: this call derives the row bounds from
  // it in one metadata launch, so the caller passes no bounds and makes no metadata call.
  //
  // The mask route's three int32 metadata arrays - row lo [B, Nq], row hi [B, Nq] and the
  // tile-bounds table [B, ceil(Nq/64), 4] - are one allocation, addressed by raw offsets (each
  // segment starts on a 256-byte boundary; no ATen view is made of it). The row bounds are
  // written by the same launch as before; the checks and their order are RowBoundsFromMask's
  // and PrepareCommon's (the row-bounds device check runs after PrepareCommon's, where its
  // check of row bound tensors ran).
  at::Tensor mask_meta;
  at::Tensor mask_c;
  int32_t* meta_lo = nullptr;
  int32_t* meta_hi = nullptr;
  int32_t* meta_tiles = nullptr;
  int32_t* fwd_ljf_ptr = nullptr;
  unsigned int* fwd_cnt_ptr = nullptr;
  int64_t mask_total = 0;
  int64_t total_q128 = 0;
  bool use_tile128_meta = false;
  if (bos_mask) {
    TORCH_CHECK(!row_lo.has_value() && !row_hi.has_value(),
                "native FlashSCA takes row bounds or a BOS mask, not both");
    TORCH_CHECK(bos_mask->dim() == 2 && bos_mask->size(0) == q.size(0) &&
                    bos_mask->size(1) >= q.size(1),
                "native FlashSCA bos_mask must be [B, Nt] with Nt >= Nq; got ",
                bos_mask->sizes(), " for q of ", q.sizes());
    const int64_t Nq = q.size(1), total = bos_mask->size(1);
    mask_total = total;
    CheckRowBoundsMaskArgs(*bos_mask, Nq, total, chunk_size);
    const int64_t B = bos_mask->size(0);
    constexpr int64_t kAlignInts = 64;  // 256 bytes
    const int64_t row_span = (B * Nq + kAlignInts - 1) / kAlignInts * kAlignInts;
    const int64_t tile_ints = B * ((Nq + 63) / 64) * 4;
    const int64_t nqt128 = (Nq + 127) / 128;
    total_q128 = B * nqt128;
    const int64_t ljf_need = total_q128 * 4 + total_q128 + 4;
    const int64_t extra_ints = tile_ints > ljf_need ? tile_ints : ljf_need;
    c10::cuda::CUDAGuard mask_guard(bos_mask->device());
    mask_c = bos_mask->contiguous();
    mask_meta = EmptyCuda({2 * row_span + extra_ints}, at::kInt, bos_mask->device());
    int32_t* const base = mask_meta.data_ptr<int32_t>();
    meta_lo = base;
    meta_hi = base + row_span;
    meta_tiles = base + 2 * row_span;
    use_tile128_meta = (B > 0 && total_q128 > 0 && !strict);
    if (use_tile128_meta && total_q128 <= 2048) {
      fwd_ljf_ptr = meta_tiles + total_q128 * 4;
      fwd_cnt_ptr = reinterpret_cast<unsigned int*>(fwd_ljf_ptr + total_q128);
    }
  }
  Prepared prepared = PrepareCommon(q, k, v, prev_k, prev_v, chunk_size, scale, q_segment,
                                    k_segment, segment_exact, row_lo, row_hi, strict);
  Params& p = prepared.params;
  if (mask_meta.defined()) {
    CheckDevice(mask_meta, q, "row bounds");
    prepared.row_lo = mask_meta;  // holds the storage of the three arrays to the end of the call
    p.row_lo = meta_lo;
    p.row_hi = meta_hi;
  }
  p.subtract_before_scale = subtract_before_scale;
  const OrderArm order_arm = OrderArmOptIn();
  p.sorted_order = order_arm != OrderArm::kNone;
  p.order_always = order_arm == OrderArm::kAlways || order_arm == OrderArm::kDump;
  p.order_dump = order_arm == OrderArm::kDump;
  p.head_fastest = LayoutOptIn();
  // Both of those are scheduling instruments of the generic forward body; the
  // native leaf (`launch.cuh::ForwardCase`) reads neither. The policy is the
  // backward's (umma_prepare.h, Refusal::kOrderArm): a live instrument keeps the
  // native body out of the way. With XATTN_SCA_UMMA_FWD unset the call runs the
  // generic body the instrument applies to, and its phase tag says so;
  // XATTN_SCA_UMMA_FWD=1 refuses here, before anything is allocated, rather than
  // timing a native body under an instrument it ignores.
  const bool order_instrument = p.sorted_order || p.head_fastest;
  const UmmaForwardArm fwd_arm = arms.umma;
  if (fwd_arm == UmmaForwardArm::kRequire && !HasUmmaForwardLeaf(p.qk_dim, p.value_dim)) {
    TORCH_CHECK(false,
                "XATTN_SCA_UMMA_FWD=1 requires a native forward leaf, and this build has none "
                "for D=", p.qk_dim, ", Dv=", p.value_dim,
                " (the native forward serves D and Dv in 8..256, multiples of 8)");
  }
  TORCH_CHECK(!(fwd_arm == UmmaForwardArm::kRequire && order_instrument),
              "XATTN_SCA_UMMA_FWD=1 cannot be combined with XATTN_SCA_ORDER or "
              "XATTN_SCA_LAYOUT=hf: they instrument the generic forward body, which the "
              "native leaf does not run");
  if (p.sorted_order) {
    // Refuse the arm on a grid the in-kernel table cannot hold (kOrderTiles in
    // kernels.cuh) instead of silently running the in-order map: the forward
    // tile is 128 rows, so 768 tiles is q_length <= 98,304.
    const int64_t order_tiles = (p.q_length + 127) / 128;
    TORCH_CHECK(order_tiles <= 768,
                "XATTN_SCA_ORDER supports at most 768 query tiles per batch row (q_length <= "
                "98304); got q_length=", p.q_length);
  }
  auto y = EmptyCuda({p.batch, p.q_length, p.query_heads, p.value_dim}, q.scalar_type(),
                     q.device());
  auto lse = EmptyCuda({p.batch, p.query_heads, p.q_length}, at::kFloat, q.device());
  if (output_state) {
    CheckDevice(*output_state, q, "output_state");
    TORCH_CHECK(output_state->scalar_type() == at::kFloat && output_state->is_contiguous() &&
                    output_state->sizes() == y.sizes(),
                "native FlashSCA output_state must be contiguous FP32 [B,Nq,Hq,Dv]");
    p.y_state = output_state->data_ptr<float>();
  }
  p.y = y.data_ptr();
  p.lse = lse.data_ptr<float>();
  if (p.batch == 0) return {y, lse};
  // Native Blackwell forward: the host prepares the TMA input maps once per call
  // and the launch template (`launch.cuh::ForwardCase`) consumes them, so the
  // leaf is selected by one prepared pointer rather than by a second predicate
  // that could disagree with `kUmmaForwardLeaf`. The preparation only describes
  // the caller's bytes to the copy engine; it changes no number, and it allocates
  // only when a tensor cannot be described to TMA.
  // The pipelined body consumes the same prepared maps (the (256,256) and (128,128) launchers try
  // it first). Which native body, if any, consumes them is resolved ONCE here, after the last
  // field PipeAccepts reads (y_state) is set, and that one value drives both the preparation
  // below and the launch site (ScopedForwardBody), so the two cannot disagree. A live
  // scheduling instrument resolves the generic body, as above.
  const umma::fwdpipe::ForwardBody body =
      ResolveForwardBody(arms, umma::fwdpipe::PipeAccepts(p), order_instrument);
  // When the caller passed precomputed row_lo/row_hi (e.g., xllm's cached visibility route
  // without bos_mask) for a persistent-eligible (128,128) kPipe forward, allocate the tiny
  // tile128_bounds + LJF worklist buffer and populate it in one launch right before LaunchForward
  // so F2 Persistent LJF Forward activates on both bos_mask and precomputed row_bounds routes.
  const bool row_bounds_persist_meta =
      !mask_meta.defined() && body == umma::fwdpipe::ForwardBody::kPipe &&
      p.row_lo != nullptr && p.q_segment == nullptr && !p.strict &&
      QkBucket(p.qk_dim) == 128 && ValueBucket(p.value_dim) == 128;
  if (mask_meta.defined()) {
    p.tile_bounds = meta_tiles;
  } else if (row_bounds_persist_meta) {
    const int64_t nqt128 = (p.q_length + 127) / 128;
    total_q128 = p.batch * nqt128;
    const int64_t tile_ints = p.batch * p.tile_count * 4;
    const int64_t ljf_need = total_q128 * 4 + total_q128 + 4;
    const int64_t extra_ints = tile_ints > ljf_need ? tile_ints : ljf_need;
    prepared.tile_bounds = EmptyCuda({extra_ints}, at::kInt, q.device());
    meta_tiles = prepared.tile_bounds.data_ptr<int32_t>();
    p.tile_bounds = meta_tiles;
    use_tile128_meta = (p.batch > 0 && total_q128 > 0);
    if (use_tile128_meta && total_q128 <= 2048) {
      fwd_ljf_ptr = meta_tiles + total_q128 * 4;
      fwd_cnt_ptr = reinterpret_cast<unsigned int*>(fwd_ljf_ptr + total_q128);
    }
  } else {
    prepared.tile_bounds = EmptyCuda({p.batch, p.tile_count, 4}, at::kInt, q.device());
    p.tile_bounds = prepared.tile_bounds.data_ptr<int32_t>();
  }

  umma::fwd_host::PreparedUmmaCall umma_call;
  if (body != umma::fwdpipe::ForwardBody::kGeneric && !order_instrument &&
      HasUmmaForwardLeaf(p.qk_dim, p.value_dim)) {
    umma_call = umma::fwd_host::PrepareUmmaCall(
        p, prepared.q, prepared.k, prepared.v,
        prev_k ? OptionalTensor(prepared.prev_k) : OptionalTensor{},
        prev_v ? OptionalTensor(prepared.prev_v) : OptionalTensor{}, std::nullopt);
    // The y map the epilogue's staged TMA store writes through, encoded only for the pipelined
    // body that stores through it - the non-pipelined native leaf (`kLeaf`) shares these maps but
    // never reads this one, and must not pay the driver call. y is allocated here, contiguous
    // [B, Nq, Hq, Dv] and 256 B aligned, so it needs no canonicalization; the map's extent is the
    // logical one, so a box past Nq or past Dv is clipped by the TMA.
    if (body == umma::fwdpipe::ForwardBody::kPipe) {
      const int dev = static_cast<int>(q.get_device());
      umma_call.launch.maps.y =
          umma::CachedInputMap(umma::bwd_host::MakeOperand(y), p.element, p.batch, p.q_length,
                               p.query_heads, p.value_dim, dev);
      umma_call.launch.maps.tile128_bounds_ready = use_tile128_meta;
      umma_call.launch.maps.fwd_ljf_order = fwd_ljf_ptr;
      umma_call.launch.maps.fwd_counter = fwd_cnt_ptr;
    }
    p.umma_maps = &umma_call.launch.maps;
  }
  // This thread's own count of forward umma -> umma/ran rows, not the process
  // table's: only a launch this call enqueues can move it before the check below.
  const auto umma_ran = [] {
    return ThreadPhaseCount(Phase::kForward, Body::kUmma, Body::kUmma, State::kRan);
  };
  const int64_t umma_ran_before = umma_ran();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  // Launch fast 1-wave RowBoundsKernel<2> (+ tile128_bounds) & BuildFwdLjf right before
  // LaunchForward (after host TMA descriptor lookup) so there is zero host bubble.
  if (bos_mask) {
    const bool emit_tile128 = (body == umma::fwdpipe::ForwardBody::kPipe && use_tile128_meta);
    LaunchRowBoundsMaskInto(mask_c, p.q_length, mask_total, chunk_size, strict, meta_lo, meta_hi,
                            stream, emit_tile128 ? meta_tiles : nullptr, /*tile128=*/emit_tile128);
    if (body == umma::fwdpipe::ForwardBody::kPipe && fwd_ljf_ptr != nullptr) {
      LaunchBuildFwdLjf(meta_tiles, fwd_ljf_ptr, fwd_cnt_ptr, p.batch, p.q_length,
                        p.total_length, stream);
    }
  } else if (row_bounds_persist_meta && use_tile128_meta) {
    LaunchTile128BoundsAndFwdLjf(p, meta_tiles, fwd_ljf_ptr, fwd_cnt_ptr, stream);
  }
  if (body != umma::fwdpipe::ForwardBody::kPipe) LaunchTileBounds(p, stream);
  {
    // The resolved body, visible to the launch site for exactly this call's launches.
    const umma::fwdpipe::ScopedForwardBody resolved(body);
    LaunchForward(p, stream);
  }
  // Null the host-only handle on `p` at once. `umma_call` itself lives to the
  // end of this function and must: it owns the canonicalized inputs until the
  // launches are enqueued. What the null-out buys is that no copy of `p` taken
  // after this line can carry a host address into a launch's argument build.
  p.umma_maps = nullptr;
  if (fwd_arm == UmmaForwardArm::kRequire) {
    // The switch is the arm label; this assert is what keeps the label honest.
    // It cannot pass unless a launch site recorded `umma -> umma -> ran` for this
    // call, so a require arm that quietly fell back to the generic body fails here
    // instead of timing the control arm twice. The count is this thread's, so a
    // native launch by a concurrent call on another thread cannot satisfy it.
    TORCH_CHECK(umma_ran() > umma_ran_before,
                "XATTN_SCA_UMMA_FWD=1 prepared a native forward call for D=", p.qk_dim,
                ", Dv=", p.value_dim, " but no launch site recorded forward:umma/ran");
  }
  g_forward_launches.fetch_add(1, std::memory_order_relaxed);
  return {y, lse};
}

// The BOS-mask training route of Forward: the mask is the whole metadata boundary, so the fields
// the Python boundary would otherwise hand over are null here - no row bounds, no segment planes,
// no output state, not strict, level 0. Same preparation, launches, numbers and counters.
std::tuple<at::Tensor, at::Tensor> ForwardMask(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const OptionalTensor& prev_k,
    const OptionalTensor& prev_v, int64_t chunk_size, double scale, const at::Tensor& bos_mask) {
  return Forward(q, k, v, prev_k, prev_v, chunk_size, scale, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, std::nullopt, /*strict=*/false,
                 /*subtract_before_scale=*/false, std::nullopt, OptionalTensor(bos_mask));
}

// ---------------------------------------------------------------- repeat-call key
// The public flash_sca_fwd's memo of the BOS-mask level-0 training route, built and checked
// here, inside the one pybind crossing of the call, from the raw Python arguments. A key holds
// every value the public entry's checks (precision level, scale, dimensions, the non-positive
// scale branch, `selected`) and _sca_native._mask_forward_route read: each operand's exact
// type (a torch.Tensor, not a subclass), sizes, dtype and device; the mask's contiguity; the
// absence of segment arguments; and every scalar argument with its exact Python type and value.
// Those checks are pure functions of the key and of the process, so a key the full Python path
// once ran to the mask route's extension call (it is learned there, by
// _flash_sca_native_fwd_fast_learn) runs there again; any other call is a miss, returns None
// and takes the full path, with its records and its errors. Nothing that depends on tensor
// contents or data pointers is decided here: ForwardMask validates, allocates and prepares
// every call. Bounded (cleared whole when full); read and written under the GIL.
struct FastTensorMeta {
  int dim = -1;
  int64_t sizes[4] = {0, 0, 0, 0};
  int dtype = -1;
  int dev_type = -1;
  int dev_index = -1;
  bool operator==(const FastTensorMeta& o) const {
    return dim == o.dim && dtype == o.dtype && dev_type == o.dev_type &&
           dev_index == o.dev_index && sizes[0] == o.sizes[0] && sizes[1] == o.sizes[1] &&
           sizes[2] == o.sizes[2] && sizes[3] == o.sizes[3];
  }
};

struct FastKey {
  FastTensorMeta q, k, v, prev_k, prev_v, mask;
  bool mask_contiguous = false;
  int64_t chunk = 0;
  int scale_kind = 0;  // 0 None, 1 float, 2 int
  double scale_float = 0.0;
  int64_t scale_int = 0;
  int64_t level = 0;
  bool reset = false;
  std::string backend, attn_method;
};

constexpr size_t kFastKeysMax = 64;
std::vector<FastKey> g_fast_keys;

bool FastMetaOf(PyObject* o, FastTensorMeta* m) {
  if (!THPVariable_CheckExact(o)) return false;
  const at::Tensor& t = THPVariable_Unpack(o);
  const int64_t d = t.dim();
  if (d > 4) return false;
  m->dim = static_cast<int>(d);
  for (int64_t i = 0; i < d; ++i) m->sizes[i] = t.size(i);
  m->dtype = static_cast<int>(t.scalar_type());
  m->dev_type = static_cast<int>(t.device().type());
  m->dev_index = static_cast<int>(t.device().index());
  return true;
}

bool FastInt(PyObject* o, int64_t* out) {
  if (!PyLong_CheckExact(o)) return false;
  int overflow = 0;
  const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
  if (overflow != 0 || (v == -1 && PyErr_Occurred())) {
    PyErr_Clear();
    return false;
  }
  *out = static_cast<int64_t>(v);
  return true;
}

bool FastStr(PyObject* o, const char** s, Py_ssize_t* n) {
  if (!PyUnicode_CheckExact(o)) return false;
  *s = PyUnicode_AsUTF8AndSize(o, n);
  if (*s == nullptr) {
    PyErr_Clear();
    return false;
  }
  return true;
}

// The call's key, or false when the call is never memoised (and so always takes the full path).
// The strings are compared in place (backend_s / attn_s point into the Python objects).
struct FastCall {
  FastKey key;  // backend / attn_method left empty; compared through the views below
  const char* backend_s = nullptr;
  Py_ssize_t backend_n = 0;
  const char* attn_s = nullptr;
  Py_ssize_t attn_n = 0;
};

bool BuildFastCall(py::handle q, py::handle k, py::handle v, py::handle chunk, py::handle scale,
                   py::handle prev_k, py::handle prev_v, py::handle q_segment,
                   py::handle k_segment, py::handle segment, py::handle mask, py::handle backend,
                   py::handle reset, py::handle attn_method, py::handle level, FastCall* c) {
  FastKey& key = c->key;
  if (mask.is_none() || !q_segment.is_none() || !k_segment.is_none() || !segment.is_none()) {
    return false;
  }
  if (!FastInt(chunk.ptr(), &key.chunk) || !FastInt(level.ptr(), &key.level)) return false;
  if (scale.is_none()) {
    key.scale_kind = 0;
  } else if (PyFloat_CheckExact(scale.ptr())) {
    key.scale_kind = 1;
    key.scale_float = PyFloat_AS_DOUBLE(scale.ptr());
  } else if (FastInt(scale.ptr(), &key.scale_int)) {
    key.scale_kind = 2;
  } else {
    return false;
  }
  if (reset.ptr() == Py_True) {
    key.reset = true;
  } else if (reset.ptr() == Py_False) {
    key.reset = false;
  } else {
    return false;
  }
  if (!FastStr(backend.ptr(), &c->backend_s, &c->backend_n) ||
      !FastStr(attn_method.ptr(), &c->attn_s, &c->attn_n)) {
    return false;
  }
  if (!FastMetaOf(q.ptr(), &key.q) || !FastMetaOf(k.ptr(), &key.k) ||
      !FastMetaOf(v.ptr(), &key.v) || !FastMetaOf(mask.ptr(), &key.mask)) {
    return false;
  }
  if (!prev_k.is_none() || !prev_v.is_none()) {
    if (!FastMetaOf(prev_k.ptr(), &key.prev_k) || !FastMetaOf(prev_v.ptr(), &key.prev_v)) {
      return false;
    }
  }
  key.mask_contiguous = THPVariable_Unpack(mask.ptr()).is_contiguous();
  return true;
}

bool FastSame(const FastCall& c, const FastKey& s) {
  const FastKey& a = c.key;
  return a.q == s.q && a.k == s.k && a.v == s.v && a.mask == s.mask && a.prev_k == s.prev_k &&
         a.prev_v == s.prev_v && a.mask_contiguous == s.mask_contiguous && a.chunk == s.chunk &&
         a.scale_kind == s.scale_kind &&
         (a.scale_kind != 1 || a.scale_float == s.scale_float) &&  // NaN never matches
         (a.scale_kind != 2 || a.scale_int == s.scale_int) && a.level == s.level &&
         a.reset == s.reset && static_cast<size_t>(c.backend_n) == s.backend.size() &&
         std::memcmp(c.backend_s, s.backend.data(), s.backend.size()) == 0 &&
         static_cast<size_t>(c.attn_n) == s.attn_method.size() &&
         std::memcmp(c.attn_s, s.attn_method.data(), s.attn_method.size()) == 0;
}

bool FastKnown(const FastCall& c) {
  for (const FastKey& s : g_fast_keys) {
    if (FastSame(c, s)) return true;
  }
  return false;
}

#define XATTN_FAST_ARGS                                                                      \
  py::handle q, py::handle k, py::handle v, py::handle chunk, py::handle scale,              \
      py::handle prev_k, py::handle prev_v, py::handle q_segment, py::handle k_segment,      \
      py::handle segment, py::handle mask, py::handle backend, py::handle reset,             \
      py::handle attn_method, py::handle level
#define XATTN_FAST_PASS                                                                     \
  q, k, v, chunk, scale, prev_k, prev_v, q_segment, k_segment, segment, mask, backend, reset, \
      attn_method, level

// A memoised call: ForwardMask's (y, lse), under no_grad as the full path runs it; else None.
py::object ForwardFast(XATTN_FAST_ARGS) {
  FastCall c;
  if (!BuildFastCall(XATTN_FAST_PASS, &c) || !FastKnown(c)) return py::none();
  const at::Tensor& qt = THPVariable_Unpack(q.ptr());
  // _sca_native._scale: 1.0 / q.shape[-1] ** 0.5 for None (Python's float pow is the C pow),
  // else float(scale).
  const double s = c.key.scale_kind == 0
                       ? 1.0 / std::pow(static_cast<double>(qt.size(-1)), 0.5)
                       : (c.key.scale_kind == 1 ? c.key.scale_float
                                                : static_cast<double>(c.key.scale_int));
  OptionalTensor pk, pv;
  if (!prev_k.is_none()) {
    pk = THPVariable_Unpack(prev_k.ptr());
    pv = THPVariable_Unpack(prev_v.ptr());
  }
  at::Tensor y, lse;
  {
    at::NoGradGuard no_grad;
    std::tie(y, lse) = ForwardMask(qt, THPVariable_Unpack(k.ptr()), THPVariable_Unpack(v.ptr()),
                                   pk, pv, c.key.chunk, s, THPVariable_Unpack(mask.ptr()));
  }
  return py::make_tuple(y, lse);
}

bool ForwardFastHit(XATTN_FAST_ARGS) {
  FastCall c;
  return BuildFastCall(XATTN_FAST_PASS, &c) && FastKnown(c);
}

void ForwardFastLearn(XATTN_FAST_ARGS) {
  FastCall c;
  if (!BuildFastCall(XATTN_FAST_PASS, &c) || FastKnown(c)) return;
  if (g_fast_keys.size() >= kFastKeysMax) g_fast_keys.clear();
  FastKey key = c.key;
  key.backend.assign(c.backend_s, static_cast<size_t>(c.backend_n));
  key.attn_method.assign(c.attn_s, static_cast<size_t>(c.attn_n));
  g_fast_keys.push_back(std::move(key));
}
#undef XATTN_FAST_ARGS
#undef XATTN_FAST_PASS

struct CachedBwdSync {
  int dev = -1;
  cudaEvent_t start = nullptr;
  cudaEvent_t meta_done = nullptr;
  cudaStream_t meta_stream = nullptr;
  cudaStream_t dq_stream = nullptr;
  ~CachedBwdSync() {
    if (start) (void)cudaEventDestroy(start);
    if (meta_done) (void)cudaEventDestroy(meta_done);
    if (meta_stream) (void)cudaStreamDestroy(meta_stream);
    if (dq_stream) (void)cudaStreamDestroy(dq_stream);
  }
  void Ensure(int device) {
    if (dev == device && start != nullptr && meta_done != nullptr &&
        meta_stream != nullptr && dq_stream != nullptr) {
      return;
    }
    if (start) (void)cudaEventDestroy(start);
    if (meta_done) (void)cudaEventDestroy(meta_done);
    if (meta_stream) (void)cudaStreamDestroy(meta_stream);
    if (dq_stream) (void)cudaStreamDestroy(dq_stream);
    start = meta_done = nullptr;
    meta_stream = dq_stream = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&meta_done, cudaEventDisableTiming));
    int least_prio = 0, greatest_prio = 0;
    C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(&meta_stream, cudaStreamNonBlocking, greatest_prio));
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&dq_stream, cudaStreamNonBlocking));
    dev = device;
  }
};

CachedBwdSync& BwdSyncRef() {
  static thread_local CachedBwdSync sync;
  return sync;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, OptionalTensor, OptionalTensor> Backward(
    const at::Tensor& dy, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const OptionalTensor& prev_k, const OptionalTensor& prev_v, const at::Tensor& saved,
    const at::Tensor& lse, int64_t chunk_size, double scale, const OptionalTensor& q_segment,
    const OptionalTensor& k_segment, const OptionalTensor& segment_exact,
    const OptionalTensor& row_lo, const OptionalTensor& row_hi, bool strict,
    bool requested_deterministic, bool fp32_kv_grads, const OptionalTensor& bos_mask) {
  // B300 supports only deterministic SCA: every backward takes the fixed-order schedule
  // (kFusedKey family, row-owned direct dQ, no FP32 atomic dQ accumulator) whatever the
  // caller's flag says. The flag stays in the signature for API compatibility only.
  (void)requested_deterministic;
  constexpr bool deterministic = true;
  g_last_backward = {};
  g_last_backward.seq = g_backward_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  // The launched-tile slot is per thread and only a launch writes it, so clear it per call:
  // a call whose key leaf does not launch then cannot report the previous call's tile.
  umma::InSituLaunchedKeyRows() = umma::kInSituKeyRowsDefault;
  // Published on every exit from here on, a refusal's raise included.
  BackwardRecordPublisher publish_on_exit;
  // XATTN_SCA_BWD_DQ_PIPE (bwd_pipe_host.h), read on every call: a value that names no arm is
  // refused here, before anything is allocated. The (256,256) dQ launch reads it again.
  const char* const dq_pipe_value = umma::dqpipe::DqPipeArmValue();
  TORCH_CHECK(umma::dqpipe::ParseDqPipeArm(dq_pipe_value) != umma::dqpipe::DqPipeArm::kInvalid,
              "XATTN_SCA_BWD_DQ_PIPE must be unset, empty, '0' or '1'; got '",
              std::string(dq_pipe_value != nullptr ? dq_pipe_value : ""), "'");
  c10::cuda::CUDAGuard guard(q.device());
  const int dev = static_cast<int>(q.get_device());
  // A BOS mask given here is the whole metadata boundary: fused RowBounds(+TileBounds) +
  // BuildKeyRangeAndLjf run on a high-priority metadata stream in parallel with Delta.
  at::Tensor mask_c;
  if (bos_mask) {
    TORCH_CHECK(!row_lo.has_value() && !row_hi.has_value(),
                "native FlashSCA takes row bounds or a BOS mask, not both");
    TORCH_CHECK(bos_mask->dim() == 2 && bos_mask->size(0) == q.size(0) &&
                    bos_mask->size(1) >= q.size(1),
                "native FlashSCA bos_mask must be [B, Nt] with Nt >= Nq; got ",
                bos_mask->sizes(), " for q of ", q.sizes());
    CheckRowBoundsMaskArgs(*bos_mask, q.size(1), bos_mask->size(1), chunk_size);
    c10::cuda::CUDAGuard mask_guard(bos_mask->device());
    mask_c = bos_mask->contiguous();
  }
  Prepared prepared = PrepareCommon(q, k, v, prev_k, prev_v, chunk_size, scale, q_segment,
                                    k_segment, segment_exact, row_lo, row_hi, strict);
  Params& p = prepared.params;
  if (bos_mask) {
    CheckDevice(mask_c, q, "row bounds");
  }
  CheckDevice(dy, q, "y_grad");
  CheckDevice(saved, q, "saved output");
  CheckDevice(lse, q, "lse");
  const std::array<int64_t, 4> out_shape{p.batch, p.q_length, p.query_heads, p.value_dim};
  const std::array<int64_t, 3> lse_shape{p.batch, p.query_heads, p.q_length};
  TORCH_CHECK(dy.sizes() == at::IntArrayRef(out_shape) && dy.scalar_type() == q.scalar_type(),
              "native FlashSCA y_grad must be [B,Nq,Hq,Dv] in Q's dtype");
  TORCH_CHECK(saved.sizes() == at::IntArrayRef(out_shape) &&
                  (saved.scalar_type() == q.scalar_type() || saved.scalar_type() == at::kFloat),
              "native FlashSCA saved output must be [B,Nq,Hq,Dv] in Q's dtype or FP32");
  TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.sizes() == at::IntArrayRef(lse_shape),
              "native FlashSCA lse must be FP32 [B,Hq,Nq]");
  at::Tensor dy_c = PrepareContiguous(dy);
  at::Tensor saved_c = PrepareContiguous(saved);
  at::Tensor lse_c = lse.contiguous();
  p.dy = dy_c.data_ptr();
  p.saved = saved_c.data_ptr();
  p.saved_fp32 = saved.scalar_type() == at::kFloat;
  p.lse_in = lse_c.data_ptr<float>();

  // FP32 key gradients (upstream's fp32_kv_grads, the composed SoftDelta merge input): the
  // values every key writer would round to the input dtype, so they exist only where a key
  // row sums several query heads.
  TORCH_CHECK(!fp32_kv_grads || p.group > 1,
              "FlashSCA FP32 dK / dV gradients need grouped query heads");
  p.key_grads_fp32 = fp32_kv_grads;
  const auto key_dtype = fp32_kv_grads ? at::kFloat : q.scalar_type();
  auto dq = EmptyCuda({p.batch, p.q_length, p.query_heads, p.qk_dim}, q.scalar_type(), q.device());
  auto dk = EmptyCuda({p.batch, p.k_length, p.kv_heads, p.qk_dim}, key_dtype, q.device());
  auto dv = EmptyCuda({p.batch, p.k_length, p.kv_heads, p.value_dim}, key_dtype, q.device());
  OptionalTensor dprev_k, dprev_v;
  if (prev_k) {
    dprev_k = EmptyCuda({p.batch, p.prev_length, p.kv_heads, p.qk_dim}, key_dtype, q.device());
    dprev_v = EmptyCuda({p.batch, p.prev_length, p.kv_heads, p.value_dim}, key_dtype, q.device());
    p.dprev_k = dprev_k->data_ptr();
    p.dprev_v = dprev_v->data_ptr();
  }
  p.dq = dq.data_ptr();
  p.dk = dk.data_ptr();
  p.dv = dv.data_ptr();
  if (p.batch == 0) return {dq, dk, dv, dprev_k, dprev_v};

  auto delta = EmptyCuda({p.batch, p.query_heads, p.q_length}, at::kFloat, q.device());
  p.delta = delta.data_ptr<float>();

  // The arm decides whether dQ is reduced inside the dK/dV kernels. Hoisted above metadata
  // allocation so any invalid/retired env raises before device allocations.
  const BackwardRequest request = BackwardArmOptIn();
  const BackwardArm arm = request.arm;
  const bool force_generic = request.force_generic;
  const bool require_native = request.require_native;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

  // Single combined int32 metadata allocation: [row_lo | row_hi | tile_bounds | key_tile_range | worklist].
  constexpr int64_t kAlignInts = 64;
  auto align_up = [](int64_t n) -> int64_t {
    return (n + kAlignInts - 1) / kAlignInts * kAlignInts;
  };
  const bool has_mask = bos_mask.has_value();
  const bool has_bounds = has_mask || (p.row_lo != nullptr);
  const int64_t row_span = has_mask ? align_up(p.batch * p.q_length) : 0;
  const int64_t tb_span = align_up(p.batch * p.tile_count * 4);
  const int64_t kt_count = has_bounds ? (p.total_length + 63) / 64 : 0;
  const int64_t kt_span = has_bounds ? align_up(p.batch * kt_count * 2) : 0;
  const int64_t nkt = (p.total_length + 127) / 128;
  const int64_t kv_pairs = p.batch * nkt;
  const int64_t total_kv_tiles = kv_pairs * p.kv_heads;
  const int64_t q_pairs = p.batch * ((p.q_length + 127) / 128);
  const bool use_kv_sched = (!p.key_grads_fp32 && kv_pairs > 0 && kv_pairs <= 2048 &&
                             p.batch <= 64 && p.kv_heads <= 256 && nkt <= 65535 &&
                             total_kv_tiles <= 65536);
  const bool use_q_ljf = (q_pairs > 0 && q_pairs <= 2048);
  const int64_t kv_wl_ints = use_kv_sched ? total_kv_tiles : 0;
  const int64_t q_wl_ints = use_q_ljf ? q_pairs : 0;
  const int64_t wl_ints = kv_wl_ints + q_wl_ints + 2;
  const int64_t total_meta_ints = 2 * row_span + tb_span + kt_span + align_up(wl_ints);
  at::Tensor bwd_meta = EmptyCuda({total_meta_ints}, at::kInt, q.device());
  int32_t* cursor = bwd_meta.data_ptr<int32_t>();
  int32_t* mask_lo_ptr = nullptr;
  int32_t* mask_hi_ptr = nullptr;
  if (has_mask) {
    mask_lo_ptr = cursor;
    cursor += row_span;
    mask_hi_ptr = cursor;
    cursor += row_span;
    p.row_lo = mask_lo_ptr;
    p.row_hi = mask_hi_ptr;
  }
  p.tile_bounds = cursor;
  cursor += tb_span;
  if (has_bounds) {
    p.key_tile_count = kt_count;
    p.key_tile_range = cursor;
    cursor += kt_span;
  }
  int32_t* wl_base = cursor;

  bwd_host::BackwardUmma umma;
  if (!force_generic) {
    umma = bwd_host::PrepareBackwardUmma(
        prepared.q, prepared.k, prepared.v,
        prev_k ? &prepared.prev_k : nullptr, prev_v ? &prepared.prev_v : nullptr,
        dy_c, saved_c, lse_c, delta, dq, dk, dv,
        dprev_k ? &*dprev_k : nullptr, dprev_v ? &*dprev_v : nullptr,
        at::Tensor(), deterministic, p);
  }
  BackwardPhases phases;
  const char* reason = nullptr;
  const bool use_phase_driver = !force_generic && umma.ok();
  if (use_phase_driver) {
    TORCH_CHECK(PlanBackwardPhases(p, umma.launch, q.scalar_type(), deterministic, &phases,
                                   &reason, arm),
                "native FlashSCA backward: invalid prepared call: ", reason ? reason : "unknown");
  } else {
    phases.arm = arm;
    phases.deterministic = deterministic;
    phases.plan = umma::MakeBackwardPlan(
        umma::BackwardFamily::kFusedKey, true);  // always direct: B300 is deterministic-only
  }
  const bool native_products = use_phase_driver &&
      ((phases.native.dq && phases.native.dkdv) || phases.native.dkdv_atomic_dq);
  g_last_backward.refusal = force_generic ? "forced_generic" : bwd_host::RefusalName(umma.refusal);
  g_last_backward.prepared = use_phase_driver;
  g_last_backward.phases = phases;
  g_last_backward.packed = use_phase_driver && umma.launch.maps.packed_kv;
  g_last_backward.qk_bucket = QkBucket(p.qk_dim);
  g_last_backward.value_bucket = ValueBucket(p.value_dim);
  {
    const auto* key_row = umma::FindBackwardLeaf(umma::BackwardProduct::kDkdv, p);
    g_last_backward.key_rows = key_row ? key_row->key_rows : umma::kInSituKeyRowsDefault;
  }
  g_last_backward.prev_length = p.prev_length;
  g_last_backward.force_generic = force_generic;
  g_last_backward.require_native = require_native;
  TORCH_CHECK(!require_native || native_products,
              "XATTN_SCA_BWD_ARM=native requires both native backward products; preparation=",
              bwd_host::RefusalName(umma.refusal), ", dq=", EligibilityName(phases.dq_eligibility),
              ", key=", EligibilityName(phases.key_eligibility));
  if (!use_phase_driver) {
    at::Tensor dq_accum;
    if (has_mask) {
      LaunchRowBoundsMaskInto(mask_c, p.q_length, mask_c.size(1), chunk_size, strict,
                              mask_lo_ptr, mask_hi_ptr, stream);
    }
    LaunchBackward(p, stream);
    if (dq_accum.defined()) {
      dq.copy_(dq_accum.mul_(p.scale));
    }
    RecordRowBounds(p);
    g_last_backward.has_accumulator = dq_accum.defined();
    g_last_backward.launched = true;
    g_backward_launches.fetch_add(1, std::memory_order_relaxed);
    return {dq, dk, dv, dprev_k, dprev_v};
  }

  at::Tensor dq_accum;
  if (phases.needs_dq_accumulator()) {
    dq_accum = at::zeros(dq.sizes(), q.options().dtype(at::kFloat));
    p.dq_accum = dq_accum.data_ptr<float>();
  }
  int32_t* kv_ljf_ptr = use_kv_sched ? wl_base : nullptr;
  int32_t* q_ljf_ptr = use_q_ljf ? (wl_base + kv_wl_ints) : nullptr;
  unsigned int* kv_cnt_ptr = reinterpret_cast<unsigned int*>(wl_base + kv_wl_ints + q_wl_ints);
  unsigned int* dq_cnt_ptr = kv_cnt_ptr + 1;
  umma.launch.kv_ljf_order = kv_ljf_ptr;
  umma.launch.q_ljf_order = q_ljf_ptr;
  umma.launch.kv_counter = kv_cnt_ptr;
  umma.launch.dq_counter = dq_cnt_ptr;

  TORCH_CHECK(CheckBackwardPhases(p, umma.launch, phases, &reason),
              "native FlashSCA backward: phases do not cover this call: ",
              reason ? reason : "?");
  if (phases.zero_prev && dprev_k && dprev_k->defined()) {
    dprev_k->zero_();
    dprev_v->zero_();
  }

  // Run fused RowBounds(+TileBounds) + BuildKeyRangeAndLjf on a cached HIGH-PRIORITY metadata
  // stream so tiny metadata CTAs are scheduled immediately alongside DeltaVectorKernel, and
  // reuse a dedicated cached dq_stream so DqPipeTileCounter hits its cache with zero churn.
  CachedBwdSync& bsync = BwdSyncRef();
  bsync.Ensure(dev);
  C10_CUDA_CHECK(cudaEventRecord(bsync.start, stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(bsync.meta_stream, bsync.start, 0));
  if (has_mask) {
    LaunchRowBoundsMaskInto(mask_c, p.q_length, mask_c.size(1), chunk_size, strict,
                            mask_lo_ptr, mask_hi_ptr, bsync.meta_stream, p.tile_bounds);
  }
  LaunchKeyRangeAndLjf(p, /*tile_bounds_ready=*/has_mask, kv_ljf_ptr, q_ljf_ptr,
                       kv_cnt_ptr, dq_cnt_ptr, bsync.meta_stream);
  C10_CUDA_CHECK(cudaEventRecord(bsync.meta_done, bsync.meta_stream));

  LaunchBackwardPhases(p, umma.launch, phases, stream, /*tables_ready_on_fork=*/true,
                       bsync.dq_stream, bsync.meta_done, /*skip_recheck=*/true);
  if (phases.native.dkdv) g_last_backward.key_rows = umma::InSituLaunchedKeyRows();
  if (phases.needs_dq_accumulator()) {
    dq.copy_(dq_accum.mul_(p.scale));
  }
  RecordRowBounds(p);
  g_last_backward.has_accumulator = dq_accum.defined();
  g_last_backward.launched = true;
  g_backward_launches.fetch_add(1, std::memory_order_relaxed);
  return {dq, dk, dv, dprev_k, dprev_v};
}

// One read of the SoftDelta pair backward, prepared as Backward() prepares a call (bounds, Params,
// delta, outputs, tile tables, maps, phase plan) and not launched. `ok` is false when the read has
// no native plan; nothing has run then.
struct PairRead {
  Prepared prepared;
  at::Tensor mask_lo, mask_hi, dy_c, saved_c, lse_c, delta, dq, dk, dv, key_tile_range;
  bwd_host::BackwardUmma umma;
  BackwardPhases phases;
  bool ok = false;
};

void PreparePairRead(PairRead& r, const at::Tensor& dy, const at::Tensor& q, const at::Tensor& k,
                     const at::Tensor& v, const OptionalTensor& saved, const at::Tensor& lse,
                     int64_t chunk_size, double scale, const OptionalTensor& q_segment,
                     const OptionalTensor& k_segment, const OptionalTensor& segment_exact,
                     const OptionalTensor& row_lo, const OptionalTensor& row_hi, bool strict,
                     const OptionalTensor& bos_mask, const OptionalTensor& given_delta) {
  OptionalTensor lo_arg = row_lo;
  OptionalTensor hi_arg = row_hi;
  if (bos_mask) {
    TORCH_CHECK(!row_lo.has_value() && !row_hi.has_value(),
                "native FlashSCA takes row bounds or a BOS mask, not both");
    TORCH_CHECK(bos_mask->dim() == 2 && bos_mask->size(0) == q.size(0) &&
                    bos_mask->size(1) >= q.size(1),
                "native FlashSCA bos_mask must be [B, Nt] with Nt >= Nq");
    std::tie(r.mask_lo, r.mask_hi) =
        RowBoundsFromMask(*bos_mask, q.size(1), bos_mask->size(1), chunk_size, strict);
    lo_arg = r.mask_lo;
    hi_arg = r.mask_hi;
  }
  r.prepared = PrepareCommon(q, k, v, OptionalTensor(), OptionalTensor(), chunk_size, scale,
                             q_segment, k_segment, segment_exact, lo_arg, hi_arg, strict);
  Params& p = r.prepared.params;
  CheckDevice(dy, q, "y_grad");
  CheckDevice(lse, q, "lse");
  const std::vector<int64_t> out_shape{p.batch, p.q_length, p.query_heads, p.value_dim};
  const std::vector<int64_t> stat_shape{p.batch, p.query_heads, p.q_length};
  TORCH_CHECK(dy.sizes() == out_shape && dy.scalar_type() == q.scalar_type(),
              "native SoftDelta backward: y_grad must be [B,Nq,Hq,Dv] in Q's dtype");
  // Exactly one of the saved output and a precomputed delta: with the delta (the SoftDelta gate
  // backward's, in the preprocessing's summation order) the read itself is never needed, the
  // preprocessing keeps only its tile tables, and no restored read is materialized.
  TORCH_CHECK(saved.has_value() != given_delta.has_value(),
              "native SoftDelta backward: give the saved output or the delta, exactly one");
  if (saved) {
    CheckDevice(*saved, q, "saved output");
    TORCH_CHECK(saved->sizes() == out_shape && saved->scalar_type() == q.scalar_type(),
                "native SoftDelta backward: saved output must be [B,Nq,Hq,Dv] in Q's dtype");
  } else {
    CheckDevice(*given_delta, q, "delta");
    TORCH_CHECK(given_delta->scalar_type() == at::kFloat && given_delta->sizes() == stat_shape,
                "native SoftDelta backward: delta must be FP32 [B,Hq,Nq]");
  }
  TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.sizes() == stat_shape,
              "native SoftDelta backward: lse must be FP32 [B,Hq,Nq]");
  r.dy_c = PrepareContiguous(dy);
  // Without a saved plane the prepared launch is described with dy in its place (same shape,
  // dtype and alignment, which is all the preparation checks) and `saved` is cleared below:
  // nothing reads it once delta is given.
  r.saved_c = saved ? PrepareContiguous(*saved) : r.dy_c;
  r.lse_c = lse.contiguous();
  p.dy = r.dy_c.data_ptr();
  p.saved = r.saved_c.data_ptr();
  p.saved_fp32 = false;
  p.lse_in = r.lse_c.data_ptr<float>();
  if (given_delta) {
    r.delta = PrepareContiguous(*given_delta);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(r.delta.data_ptr()) % 16 == 0,
                "native SoftDelta backward: delta must be 16-byte aligned");
  } else {
    r.delta = at::empty(stat_shape, q.options().dtype(at::kFloat));
  }
  p.delta = r.delta.data_ptr<float>();
  p.key_grads_fp32 = false;
  r.dq = at::empty({p.batch, p.q_length, p.query_heads, p.qk_dim}, q.options());
  r.dk = at::empty({p.batch, p.k_length, p.kv_heads, p.qk_dim}, q.options());
  r.dv = at::empty({p.batch, p.k_length, p.kv_heads, p.value_dim}, q.options());
  p.dq = r.dq.data_ptr();
  p.dk = r.dk.data_ptr();
  p.dv = r.dv.data_ptr();
  if (p.batch == 0) return;
  r.prepared.tile_bounds = at::empty({p.batch, p.tile_count, 4}, q.options().dtype(at::kInt));
  p.tile_bounds = r.prepared.tile_bounds.data_ptr<int32_t>();
  if (p.row_lo != nullptr) {
    p.key_tile_count = (p.total_length + 63) / 64;
    r.key_tile_range = at::empty({p.batch, p.key_tile_count, 2}, q.options().dtype(at::kInt));
    p.key_tile_range = r.key_tile_range.data_ptr<int32_t>();
  }
  r.umma = bwd_host::PrepareBackwardUmma(r.prepared.q, r.prepared.k, r.prepared.v, nullptr,
                                         nullptr, r.dy_c, r.saved_c, r.lse_c, r.delta, r.dq, r.dk,
                                         r.dv, nullptr, nullptr, at::Tensor(), true, p);
  if (!r.umma.ok()) return;
  if (!saved) {
    p.saved = nullptr;
    r.umma.launch.params.saved = nullptr;
  }
  const char* reason = nullptr;
  if (!PlanBackwardPhases(p, r.umma.launch, q.scalar_type(), true, &r.phases, &reason,
                          BackwardArm::kOff)) {
    return;
  }
  r.ok = true;
}

// The native SoftDelta pair backward: the grouped read (q, dy, y, lse; inclusive) and the strict
// read (s_q = the read keys, s_dy, s_saved = the negative read, s_lse; strict past) over the same
// K/V, with one key leaf for both (kv_companion.h). Returns [dq, dk, dv, dk_read] with dk/dv the
// sum of both reads rounded once, or an empty list when the pair is not eligible, in which case
// nothing has been launched and the caller runs the two reads and the merge.
std::vector<at::Tensor> SoftDeltaPairBackward(
    const at::Tensor& dy, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const OptionalTensor& saved, const at::Tensor& lse, const at::Tensor& s_dy,
    const at::Tensor& s_q, const OptionalTensor& s_saved, const at::Tensor& s_lse,
    int64_t chunk_size, double scale, const OptionalTensor& q_segment,
    const OptionalTensor& k_segment, const OptionalTensor& segment_exact,
    const OptionalTensor& row_lo, const OptionalTensor& row_hi, const OptionalTensor& s_row_lo,
    const OptionalTensor& s_row_hi, const OptionalTensor& bos_mask, const OptionalTensor& delta,
    const OptionalTensor& s_delta) {
  c10::cuda::CUDAGuard guard(q.device());
  PairRead g;
  PreparePairRead(g, dy, q, k, v, saved, lse, chunk_size, scale, q_segment, k_segment,
                  segment_exact, row_lo, row_hi, false, bos_mask, delta);
  if (!g.ok) return {};
  PairRead s;
  PreparePairRead(s, s_dy, s_q, k, v, s_saved, s_lse, chunk_size, scale, q_segment, k_segment,
                  segment_exact, s_row_lo, s_row_hi, true, bos_mask, s_delta);
  if (!s.ok) return {};
  if (!SoftDeltaPairEligible(g.prepared.params, g.umma.launch, g.phases, s.prepared.params,
                             s.umma.launch, s.phases)) {
    return {};
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  LaunchSoftDeltaBackwardPhases(g.prepared.params, g.umma.launch, g.phases, s.prepared.params,
                                s.umma.launch, s.phases, stream);
  g_backward_launches.fetch_add(1, std::memory_order_relaxed);
  return {g.dq, g.dk, g.dv, s.dq};
}

std::tuple<at::Tensor, at::Tensor> RowBounds(const at::Tensor& ids, int64_t q_length,
                                             int64_t total, int64_t chunk_size, bool strict) {
  // The fused visible-bounds reduction of the metadata boundary (row_bounds.h). Its input is
  // the caller's key segment id plane [B, Nt], nondecreasing per row - the BOS mask's cumsum,
  // which is what xattn/ops/_sca_native.py builds - and the two integers the Python route
  // applies to the query positions. The outputs are the int32 [B, Nq] row bounds the forward
  // and the backward take, so the Python boundary can answer with ONE launch instead of its
  // ~14-kernel chain (cumsum, arange, div, clamp, two searchsorted, max, min, two int32 casts,
  // contiguous). Integers only and the sorted branch's formula verbatim: bit for bit the same
  // bounds. No float, no reduction order, no atomics - nothing here can differ between runs.
  TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && ids.scalar_type() == at::kLong,
              "native FlashSCA row bounds need int64 [B, Nt] key segment ids");
  TORCH_CHECK(total == ids.size(1), "native FlashSCA row bounds: ids must be [B, total]; got Nt=",
              ids.size(1), ", total=", total);
  TORCH_CHECK(chunk_size > 0, "native FlashSCA row bounds chunk_size must be positive");
  TORCH_CHECK(q_length > 0 && q_length <= total,
              "native FlashSCA row bounds need 0 < q_length <= total; got q_length=", q_length,
              ", total=", total);
  const int64_t B = ids.size(0);
  c10::cuda::CUDAGuard guard(ids.device());
  at::Tensor ids_c = ids.contiguous();
  auto lo = at::empty({B, q_length}, ids.options().dtype(at::kInt));
  auto hi = at::empty({B, q_length}, ids.options().dtype(at::kInt));
  if (B == 0) return {lo, hi};
  RowBoundsArgs args;
  args.ids = ids_c.data_ptr<int64_t>();
  args.row_lo = lo.data_ptr<int32_t>();
  args.row_hi = hi.data_ptr<int32_t>();
  args.batch = B;
  args.q_length = q_length;
  args.total = total;
  args.chunk = chunk_size;
  args.strict = strict ? 1 : 0;
  LaunchRowBounds(args, at::cuda::getCurrentCUDAStream(ids.get_device()));
  return {lo, hi};
}

void CheckRowBoundsMaskArgs(const at::Tensor& mask, int64_t q_length, int64_t total,
                            int64_t chunk_size) {
  // RowBoundsFromMask's argument checks, in its order (the forward's mask route makes them too).
  TORCH_CHECK(mask.is_cuda() && mask.dim() == 2 && mask.scalar_type() == at::kBool,
              "native FlashSCA derives row bounds from a bool [B, Nt] BOS mask");
  TORCH_CHECK(total == mask.size(1),
              "native FlashSCA row bounds: the BOS mask must be [B, total]; got Nt=",
              mask.size(1), ", total=", total);
  TORCH_CHECK(chunk_size > 0, "native FlashSCA chunk_size must be positive");
  TORCH_CHECK(q_length > 0 && q_length <= total,
              "native FlashSCA row bounds need 0 < q_length <= total; got q_length=", q_length,
              ", total=", total);
  TORCH_CHECK(total <= kMaskMaxKeys,
              "native FlashSCA derives row bounds from a BOS mask only up to ", kMaskMaxKeys,
              " keys per row; got total=", total);
}

void LaunchRowBoundsMaskInto(const at::Tensor& mask_c, int64_t q_length, int64_t total,
                             int64_t chunk_size, bool strict, int32_t* lo, int32_t* hi,
                             cudaStream_t stream, int32_t* tile_bounds_out, bool tile128) {
  const int64_t B = mask_c.size(0);
  if (B == 0) return;
  const int32_t clamped_chunk =
      static_cast<int32_t>(std::min<int64_t>(chunk_size, total + 1));
  LaunchRowBoundsFromBos(reinterpret_cast<const unsigned char*>(mask_c.data_ptr<bool>()),
                         lo, hi, B, total, q_length, total - q_length, clamped_chunk, strict,
                         /*reset=*/false, stream, tile_bounds_out, tile128);
}

std::tuple<at::Tensor, at::Tensor> RowBoundsFromMask(const at::Tensor& mask, int64_t q_length,
                                                     int64_t total, int64_t chunk_size,
                                                     bool strict) {
  CheckRowBoundsMaskArgs(mask, q_length, total, chunk_size);
  const int64_t B = mask.size(0);
  c10::cuda::CUDAGuard guard(mask.device());
  at::Tensor mask_c = mask.contiguous();
  auto lo = at::empty({B, q_length}, mask.options().dtype(at::kInt));
  auto hi = at::empty({B, q_length}, mask.options().dtype(at::kInt));
  LaunchRowBoundsMaskInto(mask_c, q_length, total, chunk_size, strict,
                          lo.data_ptr<int32_t>(), hi.data_ptr<int32_t>(),
                          at::cuda::getCurrentCUDAStream(mask_c.get_device()));
  return {lo, hi};
}

// The sm_103a cubin runs on compute capability 10.3 only.
bool Available(const at::Tensor& q) {
  if (!q.is_cuda()) return false;
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(static_cast<c10::DeviceIndex>(q.get_device()));
  return properties != nullptr && properties->major == 10 && properties->minor == 3;
}

// Completed host dispatches, including generic fallback. These counters are
// not native-body evidence; use per-call phase tags and registered symbols.
std::tuple<int64_t, int64_t> LaunchCounts() {
  return {g_forward_launches.load(), g_backward_launches.load()};
}

// One per-leg backward sample as a dict, built by hand for the same reason the phase-tag
// rows are: bindings.h pulls no STL caster, so a std::vector<BwdLegSample> has no
// automatic conversion.
py::dict BwdLegSampleDict(const BwdLegSample& s) {
  py::dict d;
  d["seq"] = s.seq;
  d["device"] = s.device;
  d["thread"] = s.thread;
  d["delta_ms"] = s.delta_ms;
  d["dq_ms"] = s.dq_ms;
  d["key_ms"] = s.key_ms;
  d["generic_ms"] = s.generic_ms;
  d["whole_ms"] = s.whole_ms;
  d["dq_launched"] = s.dq_launched;
  d["key_launched"] = s.key_launched;
  d["generic_launched"] = s.generic_launched;
  d["dropped"] = s.dropped_before_read;
  py::list host_us;
  for (double span : s.host_us) host_us.append(span);
  d["host_us"] = host_us;
  return d;
}

py::dict LastBackwardPreparation() {
  // The process-wide record of the last backward call that left Backward, on whichever host
  // thread it ran, copied under the lock. It holds host-side values only (no device readback,
  // so the copy waits on nothing). Which call the record describes is in its stamps, not in
  // the thread that reads it.
  BackwardDiagnostic d;
  {
    std::lock_guard<std::mutex> lock(g_backward_record_mutex);
    d = g_published_backward;
  }
  const auto& p = d.phases;
  py::dict out;
  out["record_seq"] = d.seq;
  out["record_thread"] = d.recording_thread;
  out["record_launch_counts"] = py::make_tuple(d.forward_launches, d.backward_launches);
  out["recorded_on_this_thread"] = d.seq != 0 && d.recording_thread_id == std::this_thread::get_id();
  out["refusal"] = d.refusal;
  out["prepared"] = d.prepared;
  out["packed_kv"] = d.packed;
  out["launched"] = d.launched;
  // The schedule constraint is not the body-selection request. In particular,
  // "off" below means no forced split, not that native products are disabled.
  out["arm"] = p.arm == BackwardArm::kSplit ? "split" : "off";
  if (d.raw_arm) {
    out["arm_env"] = *d.raw_arm;
  } else {
    out["arm_env"] = py::none();
  }
  // Additive: the retired name's own value, kept only so a refused call
  // can be read back. Absent maps to None and set-but-empty to "" - the two are
  // different answers, and collapsing them would re-create the defect above.
  if (d.raw_retired_bwd) {
    out["retired_env"] = *d.raw_retired_bwd;
  } else {
    out["retired_env"] = py::none();
  }
  out["body_arm"] = d.force_generic ? "forced_generic" :
                    d.require_native ? "require_native" :
                    p.arm == BackwardArm::kSplit ? "split" : "auto";
  out["deterministic"] = p.deterministic;
  out["family"] = p.plan.family == umma::BackwardFamily::kFusedAtomicDq ? "fused_atomic_dq" :
                   p.plan.family == umma::BackwardFamily::kFusedKey ? "fused_key" : "split";
  out["dq_mode"] = p.plan.dq_mode == umma::DqWriteMode::kDirect ? "direct" : "atomic";
  out["generic_mask"] = p.generic_stages;
  // Structural policy, not observed zero coverage: this mount never initializes
  // dPrev separately. Only whole-plane key producers may be selected.
  out["zero_prev"] = p.zero_prev;
  out["zero_prev_policy"] = "always_false_whole_plane_writers_required";
  out["key_writes_whole_plane"] = p.native.dkdv_writes_whole_plane;
  out["has_accumulator"] = d.has_accumulator;
  out["native_dq"] = p.native.dq;
  out["native_key"] = p.native.dkdv || p.native.dkdv_atomic_dq;
  out["native_atomic_dq"] = p.native.dkdv_atomic_dq;
  out["dq_eligibility"] = EligibilityName(p.dq_eligibility);
  out["key_eligibility"] = EligibilityName(p.key_eligibility);
  out["qk_bucket"] = d.qk_bucket;
  out["value_bucket"] = d.value_bucket;
  // The key-tile height (M) of the key body the call launched: 128 on the (128,128) in-situ
  // arm and on the (256,256) single-product M128 pair (that bucket's level-0 default), 64 on
  // the fused M64 body (every other bucket, and (256,256) at level 1, under
  // XATTN_SCA_BWD_KEY_ARM=m64 or when the pair refuses the call). A result row that claims an
  // M128 key-tile arm must be gated on this field and not on the bucket - two rows can share a
  // bucket and differ only here - and on a record whose record_seq / record_launch_counts tie
  // it to the call the row describes.
  //
  // TWO FACTS SHARE THE NAME `key_rows`, and a reader must not read one as the other:
  // `BackwardLeafEntry::key_rows` (the registry header) is what the BUCKET PREFERS. This
  // snapshot field is what the CALL LAUNCHED, taken from InSituLaunchedKeyRows() after the
  // launch, ONLY when a native key product ran (`native_key` true). On a call that took the
  // generic branch it keeps the preference set at preparation, and no key tile ran at all,
  // so read this field only beside `native_key: true`. Among native calls the two differ on
  // exactly two kinds of call: those the M64 fallback serves at (128,128) - an unpacked
  // previous plane that is not a multiple of 128 - where 64 beside a header that says 128 is a
  // correct run, and those the M128 pair serves at (256,256), where 128 beside a header that
  // says 64 is a correct run.
  out["key_rows"] = d.key_rows;
  // The clipped-bounds proof: whether the LAUNCHED params carried the boundary's row bounds
  // and the key-tile table. False row_bounds is the analytic branch (no segment/BOS metadata
  // on the call); row_bounds True with key_tile_range False is the leaf's "every query tile
  // is a candidate" envelope, not a table walk (umma_bwd_dkdv.cuh:QueryTileEnvelope).
  out["row_bounds"] = d.row_bounds;
  out["key_tile_range"] = d.key_tile_range;
  // This build performs no device readback for the record: no entry of the key-tile table is
  // copied to the host. The four keys below stay for API stability with fixed values, set or
  // unset XATTN_SCA_BWD_DIAGNOSTIC alike: both entries -1, key_tile_sample "off" and an empty
  // key_tile_sample_error. diagnostic_requested says whether that variable asked.
  out["key_tile_range_first0"] = -1;
  out["key_tile_range_first1"] = -1;
  out["key_tile_sample"] = "off";
  out["key_tile_sample_error"] = "";
  out["diagnostic_requested"] = BackwardDiagnosticRequested();
  out["prev_length"] = d.prev_length;
  out["force_generic"] = d.force_generic;
  out["require_native"] = d.require_native;
  // Per-leg CUDA-event timer, host-only and default off; every key below is inert with
  // XATTN_SCA_BWD_LEGTIME unset. Draining here is deliberate: the snapshot is the last
  // per-call state, and the binding below exists for a probe that would rather drain
  // after its loop. Both read the same thread-local ring - unlike the record above, the
  // ring is the CALLING thread's, so leg samples of a backward that ran on the autograd
  // engine's thread are not drained here.
  out["leg_timer"] = BwdLegTimerOn();
  py::list legs;
  for (const auto& s : TakeBwdLegSamples()) legs.append(BwdLegSampleDict(s));
  out["legs"] = legs;
  out["legs_dropped"] = BwdLegTimerDropped();
  return out;
}

// Per-phase dispatch tags: which kernel body the host selected and launched for
// each contraction of a call, as (phase, selected_body, launched_body, state,
// count) rows, one row per non-zero counter. Rows are keyed by that tuple and not
// by order. These rows name actual producers, not every eligibility query.
// Refused native candidates are distinguished from uncompiled ones in the
// per-call preparation snapshot. A no-op row-dQ stage on the generic atomic
// route is skipped/none; the key body and final host fold own that dQ route.
std::vector<std::tuple<std::string, std::string, std::string, std::string, int64_t>>
PhaseTagRowsStd() {
  std::vector<std::tuple<std::string, std::string, std::string, std::string, int64_t>> rows;
  PhaseTagCounters& t = PhaseTags();
  for (int p = 0; p < kPhaseCount; ++p) {
    for (int s = 0; s < kBodyCount; ++s) {
      for (int l = 0; l < kBodyCount; ++l) {
        for (int st = 0; st < kStateCount; ++st) {
          const int64_t count = t.count[p][s][l][st].load(std::memory_order_relaxed);
          if (count == 0) continue;
          rows.emplace_back(PhaseName(static_cast<Phase>(p)), BodyName(static_cast<Body>(s)),
                            BodyName(static_cast<Body>(l)), StateName(static_cast<State>(st)),
                            count);
        }
      }
    }
  }
  return rows;
}

// The Python-facing form: bindings.h does not pull pybind11/stl.h, so the rows
// are converted here rather than relying on an STL caster.
py::list PhaseTagRows() {
  py::list rows;
  for (const auto& row : PhaseTagRowsStd()) {
    rows.append(py::make_tuple(std::get<0>(row), std::get<1>(row), std::get<2>(row),
                               std::get<3>(row), std::get<4>(row)));
  }
  return rows;
}

// Drain for a probe that runs a loop of calls: the per-call snapshot above would show only
// the last one. One dict per completed backward call. Empty unless
// XATTN_SCA_BWD_LEGTIME is 1 or 2.
py::list BwdLegTimeRows() {
  py::list rows;
  for (const auto& s : TakeBwdLegSamples()) rows.append(BwdLegSampleDict(s));
  return rows;
}

// Row bounds of a packed-sequence BOS mask, in one launch (row_bounds.cu).
// Called from `xattn/ops/_sca_native.py` where the route used to reduce the mask
// with a chain of torch ops; the values are the same ones, bit for bit.
std::vector<at::Tensor> RowBoundsFromBos(const at::Tensor& bos_mask, int64_t q_length,
                                         int64_t total, int64_t chunk_size, bool strict_past,
                                         bool reset) {
  TORCH_CHECK(bos_mask.is_cuda(), "native FlashSCA row bounds require a CUDA bos_mask");
  TORCH_CHECK(bos_mask.scalar_type() == at::kBool && bos_mask.dim() == 2,
              "native FlashSCA row bounds need a bool [B, total] bos_mask");
  TORCH_CHECK(bos_mask.is_contiguous(), "native FlashSCA row bounds need a contiguous bos_mask");
  TORCH_CHECK(q_length > 0 && q_length <= total, "native FlashSCA row bounds need 0 < Nq <= T");
  TORCH_CHECK(chunk_size > 0, "native FlashSCA row bounds need a positive chunk_size");
  TORCH_CHECK(total < (int64_t(1) << 31) - 64 && bos_mask.size(0) <= 65535,
              "native FlashSCA row bounds: total key length or batch out of range");
  TORCH_CHECK(bos_mask.size(1) == total, "native FlashSCA row bounds: bos_mask must be [B, T]");
  c10::cuda::CUDAGuard guard(bos_mask.device());
  const int64_t batch = bos_mask.size(0);
  auto options = bos_mask.options().dtype(at::kInt);
  std::vector<at::Tensor> out{at::empty({batch, q_length}, options),
                              at::empty({batch, q_length}, options)};
  if (batch == 0) return out;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(bos_mask.get_device());
  LaunchRowBoundsFromBos(reinterpret_cast<const unsigned char*>(bos_mask.data_ptr<bool>()),
                         out[0].data_ptr<int32_t>(), out[1].data_ptr<int32_t>(), batch, total,
                         q_length, total - q_length, static_cast<int32_t>(chunk_size),
                         strict_past, reset, stream);
  return out;
}

// H25: segment ids of a packed-sequence BOS mask, one launch (row_bounds.cu). Called from
// `xattn/ops/softdelta_attention.py` where `_prepare_segments` used to build them with
// `bos_mask.cumsum(dim=1, dtype=torch.int64)`; the values are the same, bit for bit (an integer
// prefix sum has one correct result).
at::Tensor SegmentIdsFromBos(const at::Tensor& bos_mask) {
  TORCH_CHECK(bos_mask.is_cuda(), "native FlashSCA segment ids require a CUDA bos_mask");
  TORCH_CHECK(bos_mask.scalar_type() == at::kBool && bos_mask.dim() == 2,
              "native FlashSCA segment ids need a bool [B, total] bos_mask");
  TORCH_CHECK(bos_mask.is_contiguous(), "native FlashSCA segment ids need a contiguous bos_mask");
  c10::cuda::CUDAGuard guard(bos_mask.device());
  const int64_t batch = bos_mask.size(0);
  const int64_t total = bos_mask.size(1);
  at::Tensor out = at::empty({batch, total}, bos_mask.options().dtype(at::kLong));
  if (batch == 0 || total == 0) return out;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(bos_mask.get_device());
  LaunchSegmentIdsFromBos(reinterpret_cast<const unsigned char*>(bos_mask.data_ptr<bool>()),
                          out.data_ptr<int64_t>(), batch, total, stream);
  return out;
}

}  // namespace
}  // namespace flash_sca_native

void DefineNativeFlashSCAOps(py::module& m) {
  using namespace flash_sca_native;
  m.def("_flash_sca_native_available", &Available,
        "Whether the native Blackwell FlashSCA kernels can run on q's device "
        "(compute capability 10.3)",
        py::arg("q"))
      .def("_flash_sca_native_launch_counts", &LaunchCounts,
           "(forward, backward) host dispatches completed in this process, generic fallback "
           "included; a call counter, not evidence of which body ran (read the phase tags)")
      .def("_flash_sca_last_backward_preparation", &LastBackwardPreparation,
           "Last level-0/1 backward preparation/plan published in this process by any host "
           "thread (each call publishes on exit, a refusal included); launched means enqueued. "
           "It is the latest record, not a per-caller history: record_seq numbers backward "
           "calls in the order they started, record_thread names the publishing host thread "
           "and record_launch_counts are the process-wide counters at publication, so a "
           "caller running beside other threads compares them with its own before treating "
           "the record as its call's. Host-side values only: no device readback, so "
           "key_tile_sample is always 'off'")
      .def("_flash_sca_native_bwd_legtimes", &BwdLegTimeRows,
           "Drain the per-leg backward timing samples recorded since the last drain; empty "
           "unless XATTN_SCA_BWD_LEGTIME is 1 or 2 (host-only, default off)")
      .def("_flash_sca_native_phase_tags", &PhaseTagRows,
           "Per-phase dispatch tags as (phase, selected_body, launched_body, state, count) "
           "rows, keyed by that tuple rather than by order. Covers precision levels 0 and 1 "
           "through the producers below, and precision levels 2-4 through the advanced reader "
           "(native_adv LaunchMainDq / LaunchDkdv), which records dq, dk and dv one row each "
           "as (phase, scalar, scalar, ran) after its launch returned and passed its launch "
           "check: those three rows are the levels 2-4 dispatch proof, and scalar is the "
           "point - the advanced leaves hold no matrix instruction, so a level >= 2 row never "
           "names a tensor-core body. launched_body means the launch call returned and passed the "
           "launch check, not that it succeeded and not that the device executed that body: "
           "pair it with a cubin/SASS check for device evidence. state is ran, refused (the "
           "selected body refused before launching and launched_body carried the phase; "
           "only the native forward emits it - the leaf on a leaf-bucket call whose prepared "
           "maps do not fit its column budget, the pipelined body (at (256,256) or (128,128)) "
           "when a check made "
           "before its launch call fails (a launch call that reports an error raises instead, "
           "and no row and no other body follow it) - and such a call writes TWO rows - this "
           "decision "
           "row then the generic body's own ran row - so count decisions from refused rows "
           "and executed bodies from ran rows, never one row per call; backward refusals are "
           "reported in _flash_sca_last_backward_preparation instead) or skipped (nothing was "
           "attempted, both body fields none). On the forward selected is umma for a call "
           "the pipelined level-0 body serves at (256,256) or (128,128) (XATTN_SCA_FWD_PIPE "
           "unset, empty or 1 "
           "while XATTN_SCA_UMMA_FWD is unset or empty), and otherwise the XATTN_SCA_UMMA_FWD "
           "decision: 1 requires a native leaf and selects umma on every bucket this build "
           "compiles one for; unset and 0 select mma_sync. Unset and 0 reach the same body and "
           "record the same row - the table has no arm field - so a tag cannot tell them apart; "
           "the pipelined body and the leaf both record umma, so a forward row names its body "
           "only with _flash_sca_native_fwd_pipe_launches and _flash_sca_native_fwd_leaf_launches "
           "read by difference around ONE call, the XATTN_SCA_FWD_PIPE and XATTN_SCA_UMMA_FWD "
           "values and the extension's sha256 (dispatch_tag.cuh). Native backward products "
           "record umma only after their launch returns; delta is scalar. Per-call "
           "eligibility and planned ownership are available in "
           "_flash_sca_last_backward_preparation. Not covered, so not "
           "reconcilable from these rows: LaunchTileBounds and LaunchKeyTileRange (scalar "
           "metadata kernels) and the torch elementwise kernels of the nondeterministic dQ "
           "route (at::zeros then mul_/copy_). A batch-0 call records nothing")
      .def("_flash_sca_native_phase_tags_reset", &ResetPhaseTags,
           "Zero the process-wide per-phase dispatch tags (read them before and after a call "
           "instead). The per-thread copies that the forward require guard reads are not "
           "reset; they are read by difference across a call")
      .def("_flash_sca_native_fwd_pipe_launches",
           [] {
             return flash_sca_native::umma::fwdpipe::PipeLaunchCount().load(
                 std::memory_order_relaxed);
           },
           "Launches of the pipelined level-0 native forward enqueued in this process, at "
           "both of its buckets, (256,256) and (128,128), in one count (the launch call "
           "returned without an error; the call's D and Dv name the bucket); never reset, read it by "
           "difference across a call. The forward phase tag records this body and the older "
           "native leaf alike as umma -> umma/ran, so this count and "
           "_flash_sca_native_fwd_leaf_launches are what tell them apart. A host launch "
           "witness, not device evidence: join its delta with a synchronized, successful "
           "completion of the same call and its numerical result")
      .def("_flash_sca_native_fwd_leaf_launches",
           [] {
             return flash_sca_native::umma::fwdpipe::LeafLaunchCount().load(
                 std::memory_order_relaxed);
           },
           "Launches of the older native forward leaf (umma::ForwardKernel, every bucket it is "
           "compiled for) enqueued in this process (the launch call returned without an "
           "error); never reset, read it by difference across a call. Beside "
           "_flash_sca_native_fwd_pipe_launches it names which native forward body ran; a host "
           "launch witness, not device evidence")
      .def("_flash_sca_native_kv_pipe_launches",
           [] {
             return flash_sca_native::umma::kvpipe::KvPipeLaunchCount().load(
                 std::memory_order_relaxed);
           },
           "Launches of the pipelined (128,128) fused M128 native key leaf enqueued in this "
           "process (the level-0 default at that bucket while XATTN_SCA_BWD_KEY_ARM is unset or "
           "empty); never reset, read it by difference across a call. The backward record's "
           "key_rows is 128 for this leaf and for the two-stage fused M128 leaf alike, so this "
           "count is what tells them apart")
      .def("_flash_sca_native_dq_pipe_launches",
           [] {
             return flash_sca_native::umma::dqpipe::DqPipeLaunchCount().load(
                 std::memory_order_relaxed);
           },
           "Launches of the pipelined (256,256) native dQ leaf enqueued in this process (the "
           "level-0 default at that bucket while XATTN_SCA_BWD_DQ_PIPE is unset, empty or 1); "
           "never reset, read it by difference across a call. The phase tags record it and "
           "DqKernel alike as umma, so this count is what tells them apart")
      .def("_flash_sca_row_bounds", &RowBoundsFromBos,
           "Row bounds [row_lo, row_hi] (int32 [B, Nq]) of a packed-sequence BOS mask, one "
           "launch; the same values the torch reduction produced",
           py::arg("bos_mask"), py::arg("q_length"), py::arg("total"), py::arg("chunk_size"),
           py::arg("strict_past"), py::arg("reset"))
      .def("_flash_sca_segment_ids_from_bos", &SegmentIdsFromBos,
           "Segment ids (int64 [B, total]) of a packed-sequence BOS mask, one launch; the same "
           "values bos_mask.cumsum(dim=1, dtype=torch.int64) produces",
           py::arg("bos_mask"))
      .def("_flash_sca_native_fwd", &Forward, "Native Blackwell FlashSCA forward",
           py::arg("q"), py::arg("k"), py::arg("v"), py::arg("prev_k"), py::arg("prev_v"),
           py::arg("chunk_size"), py::arg("scale"), py::arg("q_segment_idx"),
           py::arg("k_segment_idx"), py::arg("segment_exact"), py::arg("row_lo"),
           py::arg("row_hi"), py::arg("strict_past"), py::arg("subtract_before_scale"),
           py::arg("output_state"), py::arg("bos_mask") = py::none())
      .def("_flash_sca_native_fwd_mask", &ForwardMask,
           "The BOS-mask level-0 training route of _flash_sca_native_fwd, with the mask as the "
           "whole metadata boundary: identical to _flash_sca_native_fwd(q, k, v, prev_k, prev_v, "
           "chunk_size, scale, None, None, None, None, None, False, False, None, "
           "bos_mask=mask) - same preparation, launches, numbers and counters - with eight "
           "arguments instead of sixteen",
           py::arg("q"), py::arg("k"), py::arg("v"), py::arg("prev_k"), py::arg("prev_v"),
           py::arg("chunk_size"), py::arg("scale"), py::arg("bos_mask"))
      .def("_flash_sca_native_fwd_fast", &ForwardFast,
           "The public flash_sca_fwd's repeat-call route: from the call's raw arguments "
           "(q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx, "
           "segment_idx, bos_mask, backend, reset_chunk_pos_per_seq, attn_method, "
           "high_precision_level), the BOS-mask level-0 route's (y, lse) when its metadata key "
           "was learned by _flash_sca_native_fwd_fast_learn - the same _flash_sca_native_fwd_mask "
           "call the full path makes, under no_grad - else None (the caller takes the full path)")
      .def("_flash_sca_native_fwd_fast_hit", &ForwardFastHit,
           "Whether _flash_sca_native_fwd_fast would serve these arguments (same arguments); "
           "launches nothing")
      .def("_flash_sca_native_fwd_fast_learn", &ForwardFastLearn,
           "Remember the metadata key of a call (same arguments) that the full path ran to the "
           "BOS-mask level-0 extension call; a call that is never memoised is ignored. At most "
           "64 keys, cleared whole when full")
      .def("_flash_sca_native_fwd_fast_clear", [] { g_fast_keys.clear(); },
           "Forget every learned repeat-call key")
      .def("_flash_sca_native_fwd_fast_size",
           [] { return static_cast<int64_t>(g_fast_keys.size()); },
           "How many repeat-call keys are learned")
      .def("_flash_sca_native_row_bounds", &RowBounds,
           "Per-query-row visible key bounds (row_lo, row_hi) as int32 [B, Nq] from key "
           "segment ids [B, Nt] that are nondecreasing per row (a BOS mask's cumsum), in ONE "
           "launch: the sorted integer formula of xattn.ops._sca_native.visibility (chunk "
           "bounds of the right-aligned query position intersected with the row's own "
           "key-segment run; strict_past drops the query's own key). The values are integers, "
           "so this agrees with the Python chain bit for bit; a row with row_lo > row_hi has no "
           "visible key (zero output, zero gradient)",
           py::arg("ids"), py::arg("q_length"), py::arg("total"), py::arg("chunk_size"),
           py::arg("strict_past"))
      .def("_flash_sca_native_softdelta_pair_bwd", &SoftDeltaPairBackward,
           "Native SoftDelta backward of both reads over the same K/V, the strict read as one "
           "more reader of the grouped read's key leaf: [dq, dk, dv, dk_read] (dk/dv the sum of "
           "both reads, rounded once), or [] when the pair is not eligible (nothing launched)",
           py::arg("y_grad"), py::arg("q"), py::arg("k"), py::arg("v"),
           py::arg("saved_output"), py::arg("lse"), py::arg("strict_y_grad"),
           py::arg("strict_q"), py::arg("strict_saved_output"), py::arg("strict_lse"),
           py::arg("chunk_size"), py::arg("scale"), py::arg("q_segment_idx"),
           py::arg("k_segment_idx"), py::arg("segment_exact"), py::arg("row_lo"),
           py::arg("row_hi"), py::arg("strict_row_lo"), py::arg("strict_row_hi"),
           py::arg("bos_mask") = py::none(), py::arg("delta") = py::none(),
           py::arg("strict_delta") = py::none())
      .def("_flash_sca_native_bwd", &Backward,
           "Native Blackwell FlashSCA backward; fp32_kv_grads (grouped query heads) returns "
           "dk/dv/dprev_k/dprev_v in FP32, the values the 16-bit outputs are rounded from",
           py::arg("y_grad"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("prev_k"),
           py::arg("prev_v"), py::arg("saved_output"), py::arg("lse"), py::arg("chunk_size"),
           py::arg("scale"), py::arg("q_segment_idx"), py::arg("k_segment_idx"),
           py::arg("segment_exact"), py::arg("row_lo"), py::arg("row_hi"),
           py::arg("strict_past"), py::arg("deterministic"), py::arg("fp32_kv_grads") = false,
           py::arg("bos_mask") = py::none());
}

}  // namespace ops
}  // namespace xattn
