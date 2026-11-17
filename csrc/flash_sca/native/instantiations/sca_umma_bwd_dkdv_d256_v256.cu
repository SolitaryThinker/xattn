// Native Blackwell FlashSCA backward dK/dV leaf (tcgen05/UMMA), D bucket 256,
// Dv bucket 256 - the traced Gekko 0.9B shape, where the two 64-key accumulators
// and the two 128-column score windows fill all 512 TMEM columns.
#include "flash_sca/native/umma_bwd_dkdv.cuh"
#include "flash_sca/native/umma_bwd_dv_pipe.cuh"
#include "flash_sca/native/umma_bwd_dk_pipe.cuh"

XATTN_UMMA_BWD_DKDV_INSTANTIATE(__nv_bfloat16, 256, 256)
XATTN_UMMA_BWD_DKDV_INSTANTIATE(__half, 256, 256)

// The in-situ entry at THIS bucket, and it is not a stub: InSituKeyRowsFor(256, 256) is
// kInSituKeyRowsDefault (64), so this row's FUSED body is the M64 datapath, and the header's
// static_assert is what keeps it there (a fused M128 key tile at (256,256) would need 1024
// TMEM columns against the 512 a CTA may allocate - 0.9B refuses that arm by construction).
// At level 0 the entry runs the single-product M128 pair instantiated below instead, and the
// fused M64 body for every call the pair refuses (umma_bwd_dkdv.cuh,
// LaunchUmmaBackwardInSituDkdv; XATTN_SCA_BWD_KEY_ARM=m64 forces the fused body).
// Instantiated because the host's in-situ selection is compiled for every backward bucket,
// not only for the one that takes the M128 arm; without it the .so still LINKS, because a
// shared object may carry an unresolved reference, and the failure surfaces only at import,
// after a full build, as `undefined symbol ...InSituDkdv<half,256,256>`.
XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(__nv_bfloat16, 256, 256)
XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(__half, 256, 256)

// The unpipelined dV-only leaf at M128 (kWrite 2, 128-key tiles), BF16. The in-situ launch
// does not select it: its preflight is the one the pair's pipelined dV-only leaf uses, and its
// kernel stays compiled as the unpipelined form of that leaf.
template bool xattn::ops::flash_sca_native::umma::UmmaBackwardDkdvCanLaunch<__nv_bfloat16, 256,
    256, 2, 128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkdv<__nv_bfloat16, 256,
    256, 2, 128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// The (256,256) pair, which the in-situ launch above runs at level 0 instead of the fused M64
// leaf whenever both leaves accept the call. First the dV-only M128 leaf, warp-specialized and
// pipelined (umma_bwd_dv_pipe.cuh), BF16:
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDvPipe<__nv_bfloat16>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// Then the dK-only M128 leaf, warp-specialized and pipelined over 64-query rounds
// (umma_bwd_dk_pipe.cuh), BF16:
template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardDkPipe<__nv_bfloat16>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkPipe<__nv_bfloat16>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// The same pair for FP16: the dV leaf's preflight (the BF16 one is instantiated above), the
// dV leaf, then the dK leaf.
template bool xattn::ops::flash_sca_native::umma::UmmaBackwardDkdvCanLaunch<__half, 256, 256, 2,
    128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDvPipe<__half>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardDkPipe<__half>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkPipe<__half>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

namespace xattn::ops::flash_sca_native::umma {

// The host-side answer to "does a pipelined key body replace the fused one for this call"
// (umma_bwd_entry.h). It is the launch's own decision, DecideInSituKeyPair, at this bucket's two
// element types - and at (128,128) the same decision in that bucket's TU
// (InSituKvPipeWouldLaunch) - so the answer and the body that launches cannot disagree.
bool InSituKeyPairWouldLaunch(const UmmaLaunchParams& lp) {
  const int qk = QkBucket(lp.params.qk_dim), value = ValueBucket(lp.params.value_dim);
  if (qk == 128 && value == 128) return InSituKvPipeWouldLaunch(lp);
  if (qk != 256 || value != 256) return false;
  const InSituKeyPairDecision decision =
      lp.params.element == Element::kBFloat16 ? DecideInSituKeyPair<__nv_bfloat16, 256, 256>(lp)
                                              : DecideInSituKeyPair<__half, 256, 256>(lp);
  return decision == InSituKeyPairDecision::kPair;
}

}  // namespace xattn::ops::flash_sca_native::umma
