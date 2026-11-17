// Native Blackwell FlashSCA backward: the pipelined dQ leaf at (256,256), FP16
// (dqpipe::TryLaunchDqPipe<E>, declared in umma_bwd_dq.cuh and tried first by
// LaunchUmmaDq<E, 256, 256>; see umma_bwd_dq_pipe.cuh).
#include "flash_sca/native/umma_bwd_dq_pipe.cuh"

XATTN_UMMA_DQ_PIPE_INSTANTIATE(__half)
