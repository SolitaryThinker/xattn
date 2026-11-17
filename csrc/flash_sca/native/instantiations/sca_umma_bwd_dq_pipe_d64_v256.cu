// Native Blackwell FlashSCA backward at bucket (64,256): the pipelined dQ leaf only, BF16
// and FP16. The fused M128 key leaf holds dK's D and dV's Dv in TMEM columns 0..255
// (umma_bwd_kv_pipe.cuh, `kD + kDv <= 256`), so it refuses this pair and the key product keeps the
// generic body; the dQ leaf's map comes from (D, Dv) alone (umma_bwd_dq_pipe.cuh:Plan), and its
// 256-column small map is the same one it picks below D = 128 - here with one CTA, since two of
// its 135,168 B do not fit the SM's 233,472 B.
#include "flash_sca/native/umma_bwd_pipe_small.cuh"

XATTN_UMMA_BWD_DQ_PIPE_ONLY_INSTANTIATE(__nv_bfloat16, 64, 256)
XATTN_UMMA_BWD_DQ_PIPE_ONLY_INSTANTIATE(__half, 64, 256)