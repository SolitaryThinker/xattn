// Native Blackwell FlashSCA forward, D bucket 128, Dv bucket 128 (the traced Gekko 4B shape):
// the leaf launcher LaunchUmmaForward<E, 128, 128> and the pipelined body it tries first
// (fwdpipe::TryLaunchPipe<E, 128, 128>), both element types, compiled by the package build at
// sm_103a with real device code in the cubin.
#include "flash_sca/native/umma_fwd_pipe.cuh"

XATTN_UMMA_FWD_INSTANTIATE(__nv_bfloat16, 128, 128)
XATTN_UMMA_FWD_INSTANTIATE(__half, 128, 128)
XATTN_FWD_PIPE_INSTANTIATE(__nv_bfloat16, 128, 128)
XATTN_FWD_PIPE_INSTANTIATE(__half, 128, 128)
