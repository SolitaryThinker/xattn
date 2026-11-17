// Native Blackwell (sm_103a) FlashSCA backward dQ leaf: D = Dv = 128 bucket.
// Registered by setup.py's `instantiations/*.cu` glob; nothing else to add.
#include "flash_sca/native/umma_bwd_dq.cuh"

XATTN_UMMA_DQ_INSTANTIATE(__nv_bfloat16, 128, 128)
XATTN_UMMA_DQ_INSTANTIATE(__half, 128, 128)
