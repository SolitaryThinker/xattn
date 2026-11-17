// Native Blackwell FlashSCA forward, D bucket 256, Dv bucket 128: the leaf launcher
// LaunchUmmaForward<E, 256, 128> and the pipelined body it tries first
// (fwdpipe::TryLaunchPipe<E, 256, 128>), both element types.
#include "flash_sca/native/umma_fwd_pipe.cuh"

XATTN_UMMA_FWD_INSTANTIATE(__nv_bfloat16, 256, 128)
XATTN_UMMA_FWD_INSTANTIATE(__half, 256, 128)
XATTN_FWD_PIPE_INSTANTIATE(__nv_bfloat16, 256, 128)
XATTN_FWD_PIPE_INSTANTIATE(__half, 256, 128)
