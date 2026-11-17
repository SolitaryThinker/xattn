// Native Blackwell FlashSCA backward: the pipelined dQ leaf at (128,128), FP16
// (dqpipe::TryLaunchDqPipe<E, 128, 128>, declared in umma_bwd_dq.cuh and tried first by
// LaunchUmmaDq<E, 128, 128>; see umma_bwd_dq_pipe.cuh).
#include "flash_sca/native/umma_bwd_dq_pipe.cuh"

XATTN_UMMA_DQ_PIPE_INSTANTIATE_BUCKET(__half, 128, 128)
