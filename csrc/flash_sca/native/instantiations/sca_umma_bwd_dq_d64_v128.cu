// Native Blackwell (sm_103a) FlashSCA backward dQ leaf: D = 64, Dv = 128 bucket.
// Registered by setup.py's `instantiations/*.cu` glob; nothing else to add.
//
// A residue cell of the nine the host's dimension buckets can reach: the contract's
// real dimensions are D in {32,40,64,96,128,160,192,256} and Dv in
// {56,64,96,104,128,160,192}, so (qk_dim <= 64, 64 < value_dim <= 128) is a reachable
// call - and until this file existed it had no native dQ body and fell to the generic
// mma.sync kernel. A call that lands here has real dimensions that PAD the bucket, so
// the epilogue's `col >= p.qk_dim` / `col >= p.value_dim` guards are what keep the
// padding columns out of memory; the store loop is unchanged by this instantiation.
#include "flash_sca/native/umma_bwd_dq.cuh"

// The TU's own check rather than decoration: if the leaf's plan is ever reshaped,
// this file fails to compile instead of the host audit quietly disagreeing with
// the kernel it launches. The second half pins the allocation to the physical
// 512-column TMEM file, so the pair of facts does not rest on one header assertion.
static_assert(xattn::ops::flash_sca_native::umma::DqPlan<64, 128>::kAllocCols ==
                      xattn::ops::flash_sca_native::umma::DqTmemColumns(64, 128) &&
                  xattn::ops::flash_sca_native::umma::DqPlan<64, 128>::kAllocCols <= 512,
              "plan TMEM columns must agree with the host audit and fit the 512-column file");

XATTN_UMMA_DQ_INSTANTIATE(__nv_bfloat16, 64, 128)
XATTN_UMMA_DQ_INSTANTIATE(__half, 64, 128)
