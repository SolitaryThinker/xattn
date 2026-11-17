// Native Blackwell (sm_103a) FlashSCA backward dQ leaf: D = Dv = 64 bucket.
// Registered by setup.py's `instantiations/*.cu` glob; nothing else to add.
#include "flash_sca/native/umma_bwd_dq.cuh"

// The TU's own check rather than decoration: if the leaf's plan is ever reshaped,
// this file fails to compile instead of the host audit quietly disagreeing with
// the kernel it launches.
static_assert(xattn::ops::flash_sca_native::umma::DqPlan<64, 64>::kAllocCols ==
                  xattn::ops::flash_sca_native::umma::DqTmemColumns(64, 64) &&
              xattn::ops::flash_sca_native::umma::DqPlan<64, 64>::kAllocCols <= 512,
              "plan TMEM columns must agree with the host audit and fit the 512-column file");

XATTN_UMMA_DQ_INSTANTIATE(__nv_bfloat16, 64, 64)
XATTN_UMMA_DQ_INSTANTIATE(__half, 64, 64)
