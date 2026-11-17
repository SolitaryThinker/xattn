// Native Blackwell FlashSCA backward dK/dV leaf pair (tcgen05/UMMA), D bucket 256, Dv bucket
// 64: one of the four unequal buckets with D + Dv > 256 and D, Dv <= 256
// (kDkdvPairFits, umma_bwd_dkdv.cuh), which the fused M128 key leaf refuses (its TMEM map holds
// dK's D and dV's Dv columns together and needs D + Dv <= 256) and which the fused M64 leaf
// serves at level 0 only when the pair itself refuses (e.g. an unpacked previous plane not a
// multiple of 128 rows). At level 0 the pair below - the pipelined dV-only M128 leaf, then the
// pipelined dK-only M128 leaf - runs instead, same shape as (256,256)'s pair
// (sca_umma_bwd_dkdv_d256_v256.cu).
#include "flash_sca/native/umma_bwd_dkdv.cuh"
#include "flash_sca/native/umma_bwd_dv_pipe.cuh"
#include "flash_sca/native/umma_bwd_dk_pipe.cuh"

// The fused M64 leaf (kWrite 3, the "wide" fallback LaunchUmmaBackwardInSituDkdv takes when the
// pair's own preflight refuses, or at level >= 1 where the pair never runs).
XATTN_UMMA_BWD_DKDV_INSTANTIATE(__nv_bfloat16, 256, 64)
XATTN_UMMA_BWD_DKDV_INSTANTIATE(__half, 256, 64)

// The in-situ entry at this bucket: the registry's kDkdv row (sca_umma_bwd_registry.cu) names
// it directly, so it must exist as a strong symbol even though the key product was previously
// served only by the generic body (no row at all).
XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(__nv_bfloat16, 256, 64)
XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(__half, 256, 64)

// The unpipelined dV-only leaf at M128 (kWrite 2, 128-key tiles): never launched here (the
// pipelined dV-only leaf below runs instead), but its preflight is what that pipelined leaf's
// own preflight reuses, so both CanLaunch and Launch are kept compiled, same as (256,256).
template bool xattn::ops::flash_sca_native::umma::UmmaBackwardDkdvCanLaunch<__nv_bfloat16, 256,
    64, 2, 128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkdv<__nv_bfloat16, 256,
    64, 2, 128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// The pair itself: first the dV-only M128 leaf, pipelined (umma_bwd_dv_pipe.cuh), BF16:
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDvPipe<__nv_bfloat16, 256,
    64>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// Then the dK-only M128 leaf, pipelined (umma_bwd_dk_pipe.cuh), BF16:
template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardDkPipe<__nv_bfloat16, 256,
    64>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkPipe<__nv_bfloat16, 256,
    64>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

// The same pair for FP16.
template bool xattn::ops::flash_sca_native::umma::UmmaBackwardDkdvCanLaunch<__half, 256, 64, 2,
    128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkdv<__half, 256, 64, 2, 128,
    0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDvPipe<__half, 256, 64>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardDkPipe<__half, 256, 64>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkPipe<__half, 256, 64>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
