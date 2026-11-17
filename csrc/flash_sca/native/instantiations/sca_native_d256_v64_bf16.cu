// Native Blackwell FlashSCA kernels for D bucket 256, Dv bucket 64, bf16.
#include "flash_sca/native/launch.cuh"

XATTN_SCA_NATIVE_INSTANTIATE(__nv_bfloat16, 256, 64)
