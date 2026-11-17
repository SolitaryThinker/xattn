// Native Blackwell FlashSCA backward at bucket (128,64): the pipelined dQ and fused M128 key
// leaves (umma_bwd_pipe_small.cuh), BF16 and FP16.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flash_sca/native/umma_bwd_pipe_small.cuh"

XATTN_UMMA_BWD_PIPE_SMALL_INSTANTIATE(__nv_bfloat16, 128, 64)
XATTN_UMMA_BWD_PIPE_SMALL_INSTANTIATE(__half, 128, 64)
