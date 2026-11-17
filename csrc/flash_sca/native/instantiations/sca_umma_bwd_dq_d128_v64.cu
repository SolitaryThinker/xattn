// Native Blackwell (sm_103a) FlashSCA backward dQ leaf: D = 128, Dv = 64 bucket.
// Registered by setup.py's `instantiations/*.cu` glob; nothing else to add.
//
// A residue cell of the nine the host's dimension buckets can reach (see
// sca_umma_bwd_dq_d64_v128.cu for the argument): (64 < qk_dim <= 128, value_dim <= 64)
// is a reachable call that had no native dQ body. This is the mirror of the other
// padding cell - here the HIGH side of the pair carries the real dimension and the
// value axis pads - which is why both are instantiated rather than one standing in
// for the other: the epilogue's two width guards are separate expressions over
// different tensors.
#include "flash_sca/native/umma_bwd_dq.cuh"

// The TU's own check rather than decoration: if the leaf's plan is ever reshaped,
// this file fails to compile instead of the host audit quietly disagreeing with
// the kernel it launches. The second half pins the allocation to the physical
// 512-column TMEM file, so the pair of facts does not rest on one header assertion.
static_assert(xattn::ops::flash_sca_native::umma::DqPlan<128, 64>::kAllocCols ==
                      xattn::ops::flash_sca_native::umma::DqTmemColumns(128, 64) &&
                  xattn::ops::flash_sca_native::umma::DqPlan<128, 64>::kAllocCols <= 512,
              "plan TMEM columns must agree with the host audit and fit the 512-column file");

XATTN_UMMA_DQ_INSTANTIATE(__nv_bfloat16, 128, 64)
XATTN_UMMA_DQ_INSTANTIATE(__half, 128, 64)
