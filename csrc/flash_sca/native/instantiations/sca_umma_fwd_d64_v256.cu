// Native Blackwell FlashSCA forward, D bucket 64, Dv bucket 256: the leaf launcher
// LaunchUmmaForward<E, 64, 256> and the pipelined body it tries first
// (fwdpipe::TryLaunchPipe<E, 64, 256>), both element types.
#include "flash_sca/native/umma_fwd_pipe.cuh"

XATTN_UMMA_FWD_INSTANTIATE(__nv_bfloat16, 64, 256)
XATTN_UMMA_FWD_INSTANTIATE(__half, 64, 256)
XATTN_FWD_PIPE_INSTANTIATE(__nv_bfloat16, 64, 256)
XATTN_FWD_PIPE_INSTANTIATE(__half, 64, 256)
