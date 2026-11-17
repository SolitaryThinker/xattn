// Host-side half of the pipelined native forward (umma_fwd_pipe.cuh): the XATTN_SCA_FWD_PIPE arm
// and the body one call resolves to together with XATTN_SCA_UMMA_FWD, which calls the pipelined
// body serves, the launch counts of the two native forward bodies, and the error a pipelined
// launch reports after its launch call. Host-only and free of device code, so that host.cc (which
// resolves the call's body ONCE and prepares its TMA maps) and the launch site in umma_fwd.cuh /
// umma_fwd_pipe.cuh (which consumes them) read the same rule and the same resolved value.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include "flash_sca/native/params.h"

namespace xattn::ops::flash_sca_native::umma::fwdpipe {

// XATTN_SCA_FWD_PIPE, read once per forward call by host.cc (no caching):
//   unset or empty  kDefault: the pipelined body serves every call PipeAccepts admits, but only
//                   while XATTN_SCA_UMMA_FWD is unset or empty too. An explicit XATTN_SCA_UMMA_FWD
//                   keeps the meaning it has on main ("0" the generic mma.sync body, "1" the older
//                   native leaf), and the pipelined body then serves nothing.
//   "1"             kOn: the same calls as kDefault, spelled explicitly (1 is on, as in every other
//                   switch of this tree). With an explicit XATTN_SCA_UMMA_FWD ("0" or "1") it names
//                   two bodies at once, and host.cc raises.
//   "0"             kOff: the pipelined body serves nothing and the forward is exactly main's
//                   (XATTN_SCA_UMMA_FWD decides as it does on main): the control arm of an A/B.
//   anything else   kInvalid, "2" included: host.cc raises before anything is allocated, so an
//                   arm label can never time another body.
// Under kDefault and kOn alike, a call PipeAccepts does not admit, or one under a live scheduling
// instrument (XATTN_SCA_ORDER, XATTN_SCA_LAYOUT=hf), runs main's default forward: the generic
// mma.sync body. The full precedence is host.cc's ResolveForwardBody.
enum class PipeArm { kDefault, kOn, kOff, kInvalid };

inline const char* PipeArmValue() { return std::getenv("XATTN_SCA_FWD_PIPE"); }

inline PipeArm ParsePipeArm(const char* value) {
  if (value == nullptr || *value == '\0') return PipeArm::kDefault;
  if (std::strcmp(value, "1") == 0) return PipeArm::kOn;
  if (std::strcmp(value, "0") == 0) return PipeArm::kOff;
  return PipeArm::kInvalid;
}

// The forward body of one call. host.cc resolves it ONCE per call from both arms
// (ResolveForwardBody) and hands the one value to both consumers, so they cannot disagree: the
// TMA-map preparation (maps are prepared for kLeaf and kPipe only) and the launch site
// (TryLaunchPipe launches for kPipe only; a kLeaf call reaches ForwardKernel unchanged; a kGeneric
// call prepares no maps and never reaches either native body).
enum class ForwardBody { kGeneric, kLeaf, kPipe };

// How the resolved body reaches the launch site. The launch chain (LaunchForward -> ForwardCase ->
// LaunchUmmaForward) carries Params, whose layout is a kernel argument, so the choice rides in this
// per-thread slot instead: host.cc sets it for exactly the span of one call's launches
// (ScopedForwardBody) and the launch site only reads it. A host thread enqueues one call at a
// time, so the slot holds the choice of the call being enqueued. Outside that span it is kGeneric:
// a caller that resolved no body gets no pipelined launch.
inline ForwardBody& ForwardBodySlot() {
  thread_local ForwardBody slot = ForwardBody::kGeneric;
  return slot;
}

class ScopedForwardBody {
 public:
  explicit ScopedForwardBody(ForwardBody body) : saved_(ForwardBodySlot()) {
    ForwardBodySlot() = body;
  }
  ~ScopedForwardBody() { ForwardBodySlot() = saved_; }
  ScopedForwardBody(const ScopedForwardBody&) = delete;
  ScopedForwardBody& operator=(const ScopedForwardBody&) = delete;

 private:
  ForwardBody saved_;
};

// The calls the pipelined body serves: level 0 (the plain softmax form, no FP32 state), every
// (D, Dv) the native forward's maps describe (params.h IsUmmaForwardDims: each in 8..256 and a
// multiple of 8; a dim below its 64/128/256 bucket is zero-filled by the TMA and its stores keep
// the logical extent; every bucket pair has an instantiation, picked by the launcher's bucket) -
// and at least one query row. FP16 and BF16 are the only
// element types Params carries. Levels 1-4 keep the forward they had. Previous K/V is served (the
// split and packed maps of the leaf), so the traced prefill calls at both widths take this body.
// Not in this predicate: the ORDER/LAYOUT exclusion. XATTN_SCA_ORDER and XATTN_SCA_LAYOUT=hf
// instrument the generic body, which the pipelined body is not, and host.cc's rule keeps every
// native body out of the way under them (ResolveForwardBody: a live instrument resolves
// kGeneric). A second caller of LaunchUmmaForward<E, 256, 256> or <E, 128, 128> must keep that
// rule: this predicate alone would let the pipelined body serve an instrumented call silently.
// Such a caller that resolves no body leaves ForwardBodySlot() at kGeneric, and the pipelined body
// does not launch.
inline bool PipeAccepts(const Params& p) {
  const bool bucket = IsUmmaForwardDims(p.qk_dim, p.value_dim);
  return !p.subtract_before_scale && p.y_state == nullptr && bucket && p.q_length > 0;
}

// Launches enqueued in this process, one count per native forward body: the pipelined body
// (PipeForwardKernel, umma_fwd_pipe.cuh) and the older leaf (ForwardKernel, umma_fwd.cuh, every
// bucket it is compiled for). A count moves when the launch call returned without an error. The
// forward phase tag records both bodies as umma -> umma/ran, so these counts, read by difference
// around ONE call, are what tell them apart. They are host launch witnesses, not device evidence:
// a row joins them with a synchronized, successful completion of the same call and its numerical
// result (dispatch_tag.cuh). One object each across the translation units of the extension:
// static locals of inline functions, like dispatch_tag.cuh's PhaseTags().
inline std::atomic<int64_t>& PipeLaunchCount() {
  static std::atomic<int64_t> count{0};
  return count;
}

inline std::atomic<int64_t>& LeafLaunchCount() {
  static std::atomic<int64_t> count{0};
  return count;
}

// A native forward launch whose launch call was made and then reported a CUDA error. That status
// is no proof that nothing was enqueued (a launch call's status can also carry an earlier
// asynchronous error), so it is never treated as a refusal: the launch site stores the error here
// and raises (ThrowPostAttemptError), and no other body is launched over the same outputs. Both
// native forward launch sites follow this rule - TryLaunchPipe (umma_fwd_pipe.cuh) and the older
// leaf's tail in LaunchUmmaForward (umma_fwd.cuh) - and both refuse (return a refusal status) only
// from checks made before their launch call.
inline cudaError_t& PostAttemptErrorSlot() {
  thread_local cudaError_t slot = cudaSuccess;
  return slot;
}

inline std::string PostAttemptErrorText(const char* body) {
  const cudaError_t error = PostAttemptErrorSlot();
  char text[640];
  std::snprintf(text, sizeof(text),
                "flash_sca forward: %s reported %s (cudaError_t %d: %s) after its launch call, so "
                "it is not established that it enqueued nothing; the call is refused here rather "
                "than falling back to a second producer over the same outputs",
                body, cudaGetErrorName(error), static_cast<int>(error), cudaGetErrorString(error));
  return std::string(text);
}

[[noreturn]] inline void ThrowPostAttemptError(const char* body) {
  throw std::runtime_error(PostAttemptErrorText(body));
}

#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
// Probe builds only. XATTN_FWD_PROBE_HOOKS is defined in no source and by no build flag of this
// package: a probe job prepends it to copies of the three forward instantiation TUs
// (sca_umma_fwd_pipe_{bf16,fp16}.cu, sca_umma_fwd_d256_v256.cu) to observe paths a healthy sm_103
// device never takes. In such a build the pipelined body counts its lazy-rescale and clear
// branches (umma_fwd_pipe.cuh), and XATTN_FWD_PROBE selects one hook per process:
//   "pipe-refuse"   TryLaunchPipe refuses before its launch call (the two-row fallback)
//   "pipe-posterr"  the pipelined launch call is made with an empty grid and reports an error
//   "leaf-posterr"  the same at the older leaf's launch call (LaunchUmmaForward)
inline bool ProbeHook(const char* name) {
  const char* value = std::getenv("XATTN_FWD_PROBE");
  return value != nullptr && std::strcmp(value, name) == 0;
}
#endif

}  // namespace xattn::ops::flash_sca_native::umma::fwdpipe
