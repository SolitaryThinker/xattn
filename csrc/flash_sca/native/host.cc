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
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

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
#include "flash_sca/native/umma_bwd.h"
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
  out.tile_bounds = at::empty({B, p.tile_count, 4}, q.options().dtype(at::kInt));
  p.tile_bounds = out.tile_bounds.data_ptr<int32_t>();
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
  return IsUmmaForwardLeafBucket(qk_dim, value_dim);
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

std::tuple<at::Tensor, at::Tensor> Forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const OptionalTensor& prev_k,
    const OptionalTensor& prev_v, int64_t chunk_size, double scale,
    const OptionalTensor& q_segment, const OptionalTensor& k_segment,
    const OptionalTensor& segment_exact, const OptionalTensor& row_lo,
    const OptionalTensor& row_hi, bool strict, bool subtract_before_scale,
    const OptionalTensor& output_state) {
  c10::cuda::CUDAGuard guard(q.device());
  // Both forward arms, parsed ONCE for this call and first: every refusal of an arm value, or of
  // a combination of the two, raises here, before anything is allocated (ParseForwardArms).
  const ForwardArms arms = ParseForwardArms();
  Prepared prepared = PrepareCommon(q, k, v, prev_k, prev_v, chunk_size, scale, q_segment,
                                    k_segment, segment_exact, row_lo, row_hi, strict);
  Params& p = prepared.params;
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
                " (the buckets with a leaf are (128,128) and (256,256))");
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
  auto y = at::empty({p.batch, p.q_length, p.query_heads, p.value_dim}, q.options());
  auto lse = at::empty({p.batch, p.query_heads, p.q_length}, q.options().dtype(at::kFloat));
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
  umma::fwd_host::PreparedUmmaCall umma_call;
  if (body != umma::fwdpipe::ForwardBody::kGeneric && !order_instrument &&
      HasUmmaForwardLeaf(p.qk_dim, p.value_dim)) {
    umma_call = umma::fwd_host::PrepareUmmaCall(
        p, prepared.q, prepared.k, prepared.v,
        prev_k ? OptionalTensor(prepared.prev_k) : OptionalTensor{},
        prev_v ? OptionalTensor(prepared.prev_v) : OptionalTensor{}, std::nullopt);
    p.umma_maps = &umma_call.launch.maps;
  }
  // This thread's own count of forward umma -> umma/ran rows, not the process
  // table's: only a launch this call enqueues can move it before the check below.
  const auto umma_ran = [] {
    return ThreadPhaseCount(Phase::kForward, Body::kUmma, Body::kUmma, State::kRan);
  };
  const int64_t umma_ran_before = umma_ran();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  LaunchTileBounds(p, stream);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, OptionalTensor, OptionalTensor> Backward(
    const at::Tensor& dy, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const OptionalTensor& prev_k, const OptionalTensor& prev_v, const at::Tensor& saved,
    const at::Tensor& lse, int64_t chunk_size, double scale, const OptionalTensor& q_segment,
    const OptionalTensor& k_segment, const OptionalTensor& segment_exact,
    const OptionalTensor& row_lo, const OptionalTensor& row_hi, bool strict,
    bool requested_deterministic, bool fp32_kv_grads) {
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
  Prepared prepared = PrepareCommon(q, k, v, prev_k, prev_v, chunk_size, scale, q_segment,
                                    k_segment, segment_exact, row_lo, row_hi, strict);
  Params& p = prepared.params;
  CheckDevice(dy, q, "y_grad");
  CheckDevice(saved, q, "saved output");
  CheckDevice(lse, q, "lse");
  const std::vector<int64_t> out_shape{p.batch, p.q_length, p.query_heads, p.value_dim};
  TORCH_CHECK(dy.sizes() == out_shape && dy.scalar_type() == q.scalar_type(),
              "native FlashSCA y_grad must be [B,Nq,Hq,Dv] in Q's dtype");
  TORCH_CHECK(saved.sizes() == out_shape &&
                  (saved.scalar_type() == q.scalar_type() || saved.scalar_type() == at::kFloat),
              "native FlashSCA saved output must be [B,Nq,Hq,Dv] in Q's dtype or FP32");
  TORCH_CHECK(lse.scalar_type() == at::kFloat &&
                  lse.sizes() == std::vector<int64_t>({p.batch, p.query_heads, p.q_length}),
              "native FlashSCA lse must be FP32 [B,Hq,Nq]");
  at::Tensor dy_c = PrepareContiguous(dy);
  at::Tensor saved_c = PrepareContiguous(saved);
  at::Tensor lse_c = lse.contiguous();
  p.dy = dy_c.data_ptr();
  p.saved = saved_c.data_ptr();
  p.saved_fp32 = saved.scalar_type() == at::kFloat;
  p.lse_in = lse_c.data_ptr<float>();
  auto delta = at::empty({p.batch, p.query_heads, p.q_length}, q.options().dtype(at::kFloat));
  p.delta = delta.data_ptr<float>();

  // FP32 key gradients (upstream's fp32_kv_grads, the composed SoftDelta merge input): the
  // values every key writer would round to the input dtype, so they exist only where a key
  // row sums several query heads.
  TORCH_CHECK(!fp32_kv_grads || p.group > 1,
              "FlashSCA FP32 dK / dV gradients need grouped query heads");
  p.key_grads_fp32 = fp32_kv_grads;
  const auto key_options = fp32_kv_grads ? q.options().dtype(at::kFloat) : q.options();
  auto dq = at::empty({p.batch, p.q_length, p.query_heads, p.qk_dim}, q.options());
  auto dk = at::empty({p.batch, p.k_length, p.kv_heads, p.qk_dim}, key_options);
  auto dv = at::empty({p.batch, p.k_length, p.kv_heads, p.value_dim}, key_options);
  OptionalTensor dprev_k, dprev_v;
  if (prev_k) {
    dprev_k = at::empty({p.batch, p.prev_length, p.kv_heads, p.qk_dim}, key_options);
    dprev_v = at::empty({p.batch, p.prev_length, p.kv_heads, p.value_dim}, key_options);
    p.dprev_k = dprev_k->data_ptr();
    p.dprev_v = dprev_v->data_ptr();
  }
  p.dq = dq.data_ptr();
  p.dk = dk.data_ptr();
  p.dv = dv.data_ptr();
  if (p.batch == 0) return {dq, dk, dv, dprev_k, dprev_v};
  // The arm decides whether dQ is reduced inside the dK/dV kernels (the branch's own
  // allocation and fold both read `!deterministic && !split`). Hoisted above the plan
  // call so `deterministic || split` is its exact complement, and read ONCE: the opt-in
  // is env-backed, so a second read in one call is a divergence waiting to happen.
  const BackwardRequest request = BackwardArmOptIn();
  // The raw arm value was already stored inside the parser, above its refusals -
  // the only place a refused call can still be read back from, since a refusal
  // raises before this line. A second store here would be a second writer for one
  // value, and it would never run for the calls that matter.
  const BackwardArm arm = request.arm;
  const bool split = arm == BackwardArm::kSplit;
  const bool force_generic = request.force_generic;
  const bool require_native = request.require_native;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  at::Tensor key_tile_range;
  if (p.row_lo != nullptr) {
    p.key_tile_count = (p.total_length + 63) / 64;
    key_tile_range = at::empty({p.batch, p.key_tile_count, 2}, q.options().dtype(at::kInt));
    p.key_tile_range = key_tile_range.data_ptr<int32_t>();
  }

  // Decide before allocating. The family plan, not the user's determinism flag,
  // owns dQ: `deterministic=false` permits a direct row-owned writer, so an
  // accumulator allocated from the flag alone would be a 1.07 GB allocation of
  // which a native writer knows nothing.
  //
  // The driver is reached only on a preparation that succeeded. The preparation
  // takes `Params` by const reference and fills its own copy (`out.launch.params`),
  // so the refusal branch below keeps the caller's normalized operands. The
  // generic atomic route retains its usual reduction-order nondeterminism.

  // Refusal is a STATUS, not an exception (umma_prepare.h) and not a silent skip.
  // When no native leaf serves the call, the seed's mma.sync body runs exactly as it
  // does without a native leaf, including the seed's own dQ protocol. The preparation
  // takes `p` by const reference and fills its own copy, so `p` here is the Params
  // the generic branch uses.
  // Preparation may decline before commitment. Once preparation succeeds the
  // planner always has a complete generic schedule for valid calls; a plan
  // failure is therefore an invariant error, not permission to hide the defect
  // behind a fallback. A selected leaf that fails its final check also raises.
  // The single per-call arm parser runs before preparation. 'generic' avoids
  // TMA encoding/packing entirely; its preprocessing cost is therefore not
  // the cost of preparing a native call and then falling back. 'native' is a
  // strict A/B instrument: it requires BOTH native product writers, not just
  // an accepted mixed or all-generic plan. Default dispatch may use those
  // complete mixed plans; the require arm raises instead of timing a fallback.
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
    // Ask the same table the launch asks; a bucket with no row reports the default.
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
    // No atomic dQ accumulator on B300 (deterministic-only): p.dq_accum stays null, so the
    // generic body writes dQ row-owned in a fixed order.
    at::Tensor dq_accum;
    LaunchTileBounds(p, stream);
    if (p.row_lo != nullptr) LaunchKeyTileRange(p, stream);
    LaunchBackward(p, stream);
    if (dq_accum.defined()) {
      dq.copy_(dq_accum.mul_(p.scale));
    }
    // After every launch of this branch: `p` is the params LaunchBackward was handed, so the
    // bounds/table booleans are what the generic body launched with.
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
  // Final pointer/coverage and exact leaf predicates run after scratch binding,
  // before metadata/product launches. All selected key writers cover their
  // complete output plane; zero_prev is structurally false in this mount.
  TORCH_CHECK(CheckBackwardPhases(p, umma.launch, phases, &reason),
              "native FlashSCA backward: phases do not cover this call: ",
              reason ? reason : "?");
  // Reserved omitted-region initialization: the validated schedule currently
  // requires zero_prev == false and whole-plane key writers. Key ownership
  // alone never implies that previous gradients need initialization.
  if (phases.zero_prev && dprev_k && dprev_k->defined()) {
    dprev_k->zero_();
    dprev_v->zero_();
  }
  LaunchTileBounds(p, stream);
  if (p.row_lo != nullptr) LaunchKeyTileRange(p, stream);
  LaunchBackwardPhases(p, umma.launch, phases, stream);
  // Which key tile ACTUALLY launched, per call, after the launch. `key_rows` was set above
  // from the row's static value, which is the arm the BUCKET prefers; at (128,128) the M128
  // arm refuses an unpacked previous plane whose length is not a multiple of 128 and the same
  // call then runs the M64 body in the same TU (umma_bwd_dkdv.cuh, InSituLaunchedKeyRows), so
  // the static value would name an arm that did not run. Only a native key product may
  // overwrite it: on the generic branch below the field keeps the prepared value and the
  // phase tag already says `mma_sync`.
  // `dkdv` ALONE is the right guard here, and it is right by a fact that lives in three other
  // files, so it is written down rather than relied on: the key product has no atomic-dQ family
  // on this tree. `QueryNativeSupport` (umma_bwd.cu) fills only `dq`, `dkdv` and
  // `dkdv_writes_whole_plane`, and the registry (`sca_umma_bwd_registry.cu`, kEntries) emits rows
  // for kDq and kDkdv only, so `FindBackwardLeaf(kAtomicDq, ...)` returns nullptr and
  // `CheckBackwardPhases` fails the plan with "selected native leaf is not compiled" instead of
  // labelling a generic body native. The day a kAtomicDq row is registered, both of those facts
  // change at once and THIS GUARD MUST BE WIDENED to
  // `dkdv || dkdv_atomic_dq` - widening it today would be a no-op that hides the dependency.
  if (phases.native.dkdv) g_last_backward.key_rows = umma::InSituLaunchedKeyRows();
  // Fold only an actual atomic plan; never overwrite a direct producer. Keep
  // the generic accumulator's scale-then-copy form, including the output cast.
  if (phases.needs_dq_accumulator()) {
    dq.copy_(dq_accum.mul_(p.scale));
  }
  // After every launch of this branch. LaunchBackwardPhases rebinds its own copy with
  // `call.params = params` before launching any leaf, and CheckBackwardPhases has just
  // proved SameBackwardCall(params, native.params) - which compares row_lo, row_hi,
  // key_tile_range and key_tile_count (umma_bwd.cu:SameBackwardCall) - so `p` here IS the
  // params the leaves were launched with, not a second description of them.
  RecordRowBounds(p);
  g_last_backward.has_accumulator = dq_accum.defined();
  g_last_backward.launched = true;
  g_backward_launches.fetch_add(1, std::memory_order_relaxed);
  return {dq, dk, dv, dprev_k, dprev_v};
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
      .def("_flash_sca_native_fwd", &Forward, "Native Blackwell FlashSCA forward",
           py::arg("q"), py::arg("k"), py::arg("v"), py::arg("prev_k"), py::arg("prev_v"),
           py::arg("chunk_size"), py::arg("scale"), py::arg("q_segment_idx"),
           py::arg("k_segment_idx"), py::arg("segment_exact"), py::arg("row_lo"),
           py::arg("row_hi"), py::arg("strict_past"), py::arg("subtract_before_scale"),
           py::arg("output_state"))
      .def("_flash_sca_native_bwd", &Backward,
           "Native Blackwell FlashSCA backward; fp32_kv_grads (grouped query heads) returns "
           "dk/dv/dprev_k/dprev_v in FP32, the values the 16-bit outputs are rounded from",
           py::arg("y_grad"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("prev_k"),
           py::arg("prev_v"), py::arg("saved_output"), py::arg("lse"), py::arg("chunk_size"),
           py::arg("scale"), py::arg("q_segment_idx"), py::arg("k_segment_idx"),
           py::arg("segment_exact"), py::arg("row_lo"), py::arg("row_hi"),
           py::arg("strict_past"), py::arg("deterministic"), py::arg("fp32_kv_grads") = false);
}

}  // namespace ops
}  // namespace xattn
