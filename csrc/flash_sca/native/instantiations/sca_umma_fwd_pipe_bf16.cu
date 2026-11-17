// Native Blackwell FlashSCA pipelined forward (tcgen05/UMMA, warp-specialized), D = Dv = 256 and
// D = Dv = 128, level 0, BF16: fwdpipe::TryLaunchPipe<__nv_bfloat16> and its two kernels (one per
// bucket), called by LaunchUmmaForward<__nv_bfloat16, 256, 256> and <__nv_bfloat16, 128, 128>
// (umma_fwd.cuh). One element type per file so the build compiles the two in parallel.
#include "flash_sca/native/umma_fwd_pipe.cuh"

XATTN_FWD_PIPE_INSTANTIATE(__nv_bfloat16)
