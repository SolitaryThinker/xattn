// Native Blackwell (sm_103a) SCA backward at the head-dimension buckets with no other tcgen05
// leaf: (64,64), (64,128) and (128,64). Registry entries (instantiations/sca_umma_bwd_registry.cu,
// XATTN_SCA_BACKWARD_PIPE_BUCKETS in umma_bwd_entry.h) for the two backward products:
//   dQ   the pipelined dQ leaf (umma_bwd_dq_pipe.cuh, DqPipeKernel<E, D, Dv>), one CTA per
//        (batch, query head, 128-query tile), row-owned dQ;
//   dK/dV the pipelined fused M128 key leaf (umma_bwd_kv_pipe.cuh, BackwardKvPipeKernel<E, S,
//        fp32, D, Dv>), one CTA per (batch, KV head, 128-key tile), row-owned dK/dV/dPrevK/dPrevV.
// Both are the leaves that serve (128,128), templated on the slab counts D / 64 and Dv / 64. One
// writer per output row, no atomics, a fixed accumulation order: every call repeats bit-exactly.
//
// Scope: precision level 0 (16-bit saved output, no subtract-before-scale) and every call the
// leaves' preflights accept; XATTN_SCA_BWD_DQ_PIPE=0 refuses the dQ product. A refused product
// runs the generic body (kernels.cuh) exactly as before these entries existed.
#pragma once

#include "flash_sca/native/umma_bwd_dkdv.cuh"
#include "flash_sca/native/umma_bwd_kv_pipe.cuh"
#include "flash_sca/native/umma_bwd_dq_pipe.cuh"

namespace xattn::ops::flash_sca_native::umma {

template <class E, int kD, int kDv>
bool CanLaunchUmmaDqPipeAt(const UmmaLaunchParams& lp) {
  static_assert(!(kD == 128 && kDv == 128) && !(kD == 256 && kDv == 256),
                "(128,128) and (256,256) reach the dQ pipe through LaunchUmmaDq");
  const Params& p = lp.params;
  if (p.saved_fp32 || p.subtract_before_scale) return false;
  const dqpipe::DqPipeArm arm = dqpipe::ParseDqPipeArm(dqpipe::DqPipeArmValue());
  if (arm != dqpipe::DqPipeArm::kOn) return false;
  if (!CanLaunchUmmaDq<E, kD, kDv>(lp)) return false;
  return dqpipe::CanLaunchUmmaDqPipe<E, kD, kDv>(lp);
}

template <class E, int kD, int kDv>
bool LaunchUmmaDqPipeAt(const UmmaLaunchParams& lp, cudaStream_t stream) {
  if (!CanLaunchUmmaDqPipeAt<E, kD, kDv>(lp)) return false;
  dqpipe::LaunchDqPipeKernel<E, kD, kDv>(lp, stream);
  dqpipe::DqPipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
  return true;
}

}  // namespace xattn::ops::flash_sca_native::umma

#define XATTN_UMMA_BWD_PIPE_SMALL_INSTANTIATE(E, D, DV)                                          \
  template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaDqPipeAt<E, D, DV>(            \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);                              \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaDqPipeAt<E, D, DV>(               \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);                \
  template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardKvPipeAt<E, D, DV>(    \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);                              \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardKvPipeAt<E, D, DV>(       \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// The buckets where only the dQ leg has a pipelined leaf: the fused M128 key leaf's map
// holds dK's D and dV's Dv in TMEM columns 0..255 (umma_bwd_kv_pipe.cuh, `kD + kDv <= 256`), so it
// refuses every pair above that, and these buckets' key product keeps the generic body while
// their dQ product runs the pipelined leaf. The dQ leaf's map is derived from (D, Dv) alone, so
// the same entry serves them; only the key half of the macro above is left out.
#define XATTN_UMMA_BWD_DQ_PIPE_ONLY_INSTANTIATE(E, D, DV)                                        \
  template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaDqPipeAt<E, D, DV>(            \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);                              \
  template bool xattn::ops::flash_sca_native::umma::LaunchUmmaDqPipeAt<E, D, DV>(               \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
