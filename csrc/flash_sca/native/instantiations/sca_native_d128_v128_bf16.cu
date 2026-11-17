// Native Blackwell FlashSCA kernels for D bucket 128, Dv bucket 128, bf16.
#include "flash_sca/native/launch.cuh"

XATTN_SCA_NATIVE_INSTANTIATE(__nv_bfloat16, 128, 128)
