// Native Blackwell FlashSCA forward, D bucket 256, Dv bucket 256 (the traced Gekko 0.9B shape):
// the leaf launcher LaunchUmmaForward<E, 256, 256> and the pipelined body it tries first
// (fwdpipe::TryLaunchPipe<E, 256, 256>), both element types, compiled by the package build at
// sm_103a with real device code in the cubin.
#include "flash_sca/native/umma_fwd_pipe.cuh"

XATTN_UMMA_FWD_INSTANTIATE(__nv_bfloat16, 256, 256)
XATTN_UMMA_FWD_INSTANTIATE(__half, 256, 256)
XATTN_FWD_PIPE_INSTANTIATE(__nv_bfloat16, 256, 256)
XATTN_FWD_PIPE_INSTANTIATE(__half, 256, 256)
