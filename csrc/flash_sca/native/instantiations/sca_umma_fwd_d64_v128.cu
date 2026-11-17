// Native Blackwell FlashSCA forward, D bucket 64, Dv bucket 128 (the mixed-64-128 guard shape):
// the leaf launcher LaunchUmmaForward<E, 64, 128> and the pipelined body it tries first
// (fwdpipe::TryLaunchPipe<E, 64, 128>), both element types. One TU per bucket pair: launch.cuh's
// ForwardCase calls LaunchUmmaForward<E, D, Dv> by declaration, and the definitions are
// instantiated only here.
#include "flash_sca/native/umma_fwd_pipe.cuh"

XATTN_UMMA_FWD_INSTANTIATE(__nv_bfloat16, 64, 128)
XATTN_UMMA_FWD_INSTANTIATE(__half, 64, 128)
XATTN_FWD_PIPE_INSTANTIATE(__nv_bfloat16, 64, 128)
XATTN_FWD_PIPE_INSTANTIATE(__half, 64, 128)
