// Native Blackwell FlashSCA forward, D bucket 64, Dv bucket 64 (the ladder head dim): the leaf
// launcher LaunchUmmaForward<E, 64, 64> and the pipelined body it tries first
// (fwdpipe::TryLaunchPipe<E, 64, 64>), both element types.
#include "flash_sca/native/umma_fwd_pipe.cuh"

XATTN_UMMA_FWD_INSTANTIATE(__nv_bfloat16, 64, 64)
XATTN_UMMA_FWD_INSTANTIATE(__half, 64, 64)
XATTN_FWD_PIPE_INSTANTIATE(__nv_bfloat16, 64, 64)
XATTN_FWD_PIPE_INSTANTIATE(__half, 64, 64)
