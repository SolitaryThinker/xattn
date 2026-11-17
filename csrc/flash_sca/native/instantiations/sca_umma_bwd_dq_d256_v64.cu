// Native Blackwell (sm_103a) FlashSCA backward dQ leaf: D = 256, Dv = 64 bucket.
// Registered by setup.py's `instantiations/*.cu` glob; nothing else to add.
//
// The last residue cell of the nine: (192 < qk_dim <= 256, value_dim <= 64). It is the
// widest qk axis against the narrowest value axis, so its TMEM use is the largest of
// the three residue cells - which is exactly why the ceiling half of the assert below
// is worth carrying here rather than trusting the header alone.
#include "flash_sca/native/umma_bwd_dq.cuh"

// The TU's own check rather than decoration: if the leaf's plan is ever reshaped,
// this file fails to compile instead of the host audit quietly disagreeing with
// the kernel it launches. The second half pins the allocation to the physical
// 512-column TMEM file, so the pair of facts does not rest on one header assertion.
static_assert(xattn::ops::flash_sca_native::umma::DqPlan<256, 64>::kAllocCols ==
                      xattn::ops::flash_sca_native::umma::DqTmemColumns(256, 64) &&
                  xattn::ops::flash_sca_native::umma::DqPlan<256, 64>::kAllocCols <= 512,
              "plan TMEM columns must agree with the host audit and fit the 512-column file");

XATTN_UMMA_DQ_INSTANTIATE(__nv_bfloat16, 256, 64)
XATTN_UMMA_DQ_INSTANTIATE(__half, 256, 64)
