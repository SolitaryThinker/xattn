// Native Blackwell FlashSCA backward dK/dV leaf pair (tcgen05/UMMA), D bucket 256, Dv bucket
// 128. See sca_umma_bwd_dkdv_pair_d256_v64.cu for the bucket's general rationale
// (kDkdvPairFits, umma_bwd_dkdv.cuh); the structure here is identical with Dv = 128.
#include "flash_sca/native/umma_bwd_dkdv.cuh"
#include "flash_sca/native/umma_bwd_dv_pipe.cuh"
#include "flash_sca/native/umma_bwd_dk_pipe.cuh"

XATTN_UMMA_BWD_DKDV_INSTANTIATE(__nv_bfloat16, 256, 128)
XATTN_UMMA_BWD_DKDV_INSTANTIATE(__half, 256, 128)

XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(__nv_bfloat16, 256, 128)
XATTN_UMMA_BWD_DKDV_IN_SITU_INSTANTIATE(__half, 256, 128)

template bool xattn::ops::flash_sca_native::umma::UmmaBackwardDkdvCanLaunch<__nv_bfloat16, 256,
    128, 2, 128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkdv<__nv_bfloat16, 256,
    128, 2, 128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDvPipe<__nv_bfloat16, 256,
    128>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardDkPipe<__nv_bfloat16, 256,
    128>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkPipe<__nv_bfloat16, 256,
    128>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);

template bool xattn::ops::flash_sca_native::umma::UmmaBackwardDkdvCanLaunch<__half, 256, 128, 2,
    128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkdv<__half, 256, 128, 2,
    128, 0>(const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDvPipe<__half, 256, 128>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
template bool xattn::ops::flash_sca_native::umma::CanLaunchUmmaBackwardDkPipe<__half, 256, 128>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&);
template bool xattn::ops::flash_sca_native::umma::LaunchUmmaBackwardDkPipe<__half, 256, 128>(
    const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
