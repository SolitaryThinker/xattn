// Native Blackwell FlashSCA forward leaf (tcgen05/UMMA), D bucket 256,
// Dv bucket 256 - the traced Gekko 0.9B shape. Same reasoning as
// sca_umma_fwd_d128_v128.cu: the leaf must be compiled by the package build at
// sm_103a, with real device code in the cubin.
#include "flash_sca/native/umma_fwd.cuh"

XATTN_UMMA_FWD_INSTANTIATE(__nv_bfloat16, 256, 256)
XATTN_UMMA_FWD_INSTANTIATE(__half, 256, 256)
