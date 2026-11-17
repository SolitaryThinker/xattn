// Host-only pieces of the two pipelined native backward leaves, kept free of device code so that
// host.cc can include them: the launch counters of the pipelined fused M128 key leaf at (128,128)
// (umma_bwd_kv_pipe.cuh) and of the pipelined dQ leaf at (256,256) and (128,128)
// (umma_bwd_dq_pipe.cuh), and the parse of XATTN_SCA_BWD_DQ_PIPE.
//
// XATTN_SCA_BWD_DQ_PIPE selects the dQ body of a call at (256,256) and at (128,128), read on
// every call:
//   unset, empty or "1"  the pipelined dQ leaf for every level-0 call its preflight accepts (BF16
//                        and FP16), DqKernel for every other call - the default ("1" names it);
//   "0"                  DqKernel for every call - the control arm;
//   any other value is refused: host.cc raises before the backward allocates anything.
// (The key body of the (128,128) bucket is selected by XATTN_SCA_BWD_KEY_ARM; umma_bwd_dkdv.cuh
// documents it.)
//
// The counters: one process-wide object each (a static local of an inline function is a single
// object across the translation units of the extension, as dispatch_tag.cuh's PhaseTags()), one
// increment per launch enqueued, never reset - read them by difference across a call. host.cc
// exposes them as xattn_cuda.ops._flash_sca_native_kv_pipe_launches() and
// _flash_sca_native_dq_pipe_launches(). The phase tags record both leaves as umma, like the
// leaves they replace, and the backward record's key_rows is 128 for both M128 key bodies, so
// these counts are what tells a pipelined body from the body it replaces.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace xattn::ops::flash_sca_native::umma {

namespace kvpipe {
inline std::atomic<int64_t>& KvPipeLaunchCount() {
  static std::atomic<int64_t> count{0};
  return count;
}
}  // namespace kvpipe

namespace dqpipe {
inline std::atomic<int64_t>& DqPipeLaunchCount() {
  static std::atomic<int64_t> count{0};
  return count;
}

enum class DqPipeArm { kOn, kOff, kInvalid };

inline const char* DqPipeArmValue() { return std::getenv("XATTN_SCA_BWD_DQ_PIPE"); }

inline DqPipeArm ParseDqPipeArm(const char* value) {
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "1") == 0) return DqPipeArm::kOn;
  if (std::strcmp(value, "0") == 0) return DqPipeArm::kOff;
  return DqPipeArm::kInvalid;
}
}  // namespace dqpipe

}  // namespace xattn::ops::flash_sca_native::umma
