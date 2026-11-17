// Native Blackwell (sm_103a) FlashSCA backward dQ leaf: D = Dv = 256 bucket.
// Registered by setup.py's `instantiations/*.cu` glob; nothing else to add.
#include "flash_sca/native/umma_bwd_dq.cuh"

XATTN_UMMA_DQ_INSTANTIATE(__nv_bfloat16, 256, 256)
XATTN_UMMA_DQ_INSTANTIATE(__half, 256, 256)
