// Native Blackwell FlashSCA backward dK/dV leaf (tcgen05/UMMA), D bucket 128,
// Dv bucket 128 - the traced Gekko 4B shape and the contract's (128,128) rows.
// Instantiated here so the leaf joins the XATTN_HAS_NATIVE_SCA build group and its
// sm_103a cubin comes from the ordinary package build, not only from a probe.
#include "flash_sca/native/umma_bwd_dkdv.cuh"
#include "flash_sca/native/umma_bwd_kv_pipe.cuh"
#include "flash_sca/native/umma_bwd_kv_r128.cuh"

XATTN_UMMA_BWD_DKDV_INSTANTIATE(__nv_bfloat16, 128, 128)
XATTN_UMMA_BWD_DKDV_INSTANTIATE(__half, 128, 128)

// The in-situ arm this bucket selects (umma_bwd_entry.h:InSituKeyRowsFor -> 128): the
// same fused leaf with a 128-row key tile, which is a pure regrouping of the M64 MMA
// (umma_bwd_dkdv.cuh, the in-situ commentary block) and the only bucket whose TMEM map
// holds it. Instantiated here, in the same TU as the M64 form, so one .so carries both
// and the registry row has a definition to bind to.
XATTN_UMMA_BWD_DKDV_M128_INSTANTIATE(__nv_bfloat16)
XATTN_UMMA_BWD_DKDV_M128_INSTANTIATE(__half)

// The pipelined fused M128 key leaf (umma_bwd_kv_pipe.cuh), which the in-situ launch above runs
// at level 0, BF16 and FP16, for every call its preflight accepts while XATTN_SCA_BWD_KEY_ARM is
// unset or empty (umma_bwd_dkdv.cuh, DecideInSituKeyPair).
template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardKvPipe<__nv_bfloat16>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardKvPipe<__nv_bfloat16>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
// H2 r1: the plain 128-query-round launch LaunchUmmaBackwardKvPipe defaults to (16-bit key grads).
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardKvR128PlainT<__nv_bfloat16>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardKvR128PlainT<__half>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardKvPipe<__half>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardKvPipe<__half>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

namespace xattn::ops::flash_sca_native::umma {

// The (128,128) half of InSituKeyPairWouldLaunch (umma_bwd_entry.h): the launch's own decision,
// DecideInSituKeyPair, at this bucket's two element types, so the answer and the body that
// launches cannot disagree.
bool InSituKvPipeWouldLaunch(const UmmaLaunchParams& lp) {
  const InSituKeyPairDecision decision =
      lp.params.element == Element::kBFloat16 ? DecideInSituKeyPair<__nv_bfloat16, 128, 128>(lp)
                                              : DecideInSituKeyPair<__half, 128, 128>(lp);
  return decision == InSituKeyPairDecision::kKvPipe;
}

// The companion key leaf (kv_companion.h): the strict read as one more reader of the grouped
// read's key pass, at this bucket's two element types.
bool CanLaunchUmmaBackwardKvPipeCompanion(const UmmaLaunchParams& lp) {
  if (lp.params.element == Element::kBFloat16) {
    return CanLaunchUmmaBackwardKvPipeCompanionT<__nv_bfloat16>(lp);
  }
  if (lp.params.element == Element::kHalf) return CanLaunchUmmaBackwardKvPipeCompanionT<__half>(lp);
  return false;
}

bool LaunchUmmaBackwardKvPipeCompanion(const UmmaLaunchParams& lp, const KvCompanion& comp,
                                       cudaStream_t stream) {
#if XATTN_KV_R128
  // H29: the companion key leaf in 128-query rounds (umma_bwd_kv_r128.cuh).
  if (lp.params.element == Element::kBFloat16) {
    return LaunchUmmaBackwardKvR128CompanionT<__nv_bfloat16>(lp, comp, stream);
  }
  if (lp.params.element == Element::kHalf) {
    return LaunchUmmaBackwardKvR128CompanionT<__half>(lp, comp, stream);
  }
#else
  if (lp.params.element == Element::kBFloat16) {
    return LaunchUmmaBackwardKvPipeCompanionT<__nv_bfloat16>(lp, comp, stream);
  }
  if (lp.params.element == Element::kHalf) {
    return LaunchUmmaBackwardKvPipeCompanionT<__half>(lp, comp, stream);
  }
#endif
  return false;
}

}  // namespace xattn::ops::flash_sca_native::umma
