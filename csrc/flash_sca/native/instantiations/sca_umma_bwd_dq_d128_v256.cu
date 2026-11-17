// Native Blackwell (sm_103a) FlashSCA backward dQ leaf: D = 128, Dv = 256 bucket.
// Registered by setup.py's `instantiations/*.cu` glob; nothing else to add.
#include "flash_sca/native/umma_bwd_dq.cuh"

static_assert(xattn::ops::flash_sca_native::umma::DqPlan<128, 256>::kAllocCols ==
                  xattn::ops::flash_sca_native::umma::DqTmemColumns(128, 256) &&
              xattn::ops::flash_sca_native::umma::DqPlan<128, 256>::kAllocCols <= 512,
              "plan TMEM columns must agree with the host audit and fit the 512-column file");

XATTN_UMMA_DQ_INSTANTIATE(__nv_bfloat16, 128, 256)
XATTN_UMMA_DQ_INSTANTIATE(__half, 128, 256)
