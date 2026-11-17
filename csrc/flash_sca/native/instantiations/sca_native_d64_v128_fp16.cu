// Native Blackwell FlashSCA kernels for D bucket 64, Dv bucket 128, fp16.
#include "flash_sca/native/launch.cuh"

XATTN_SCA_NATIVE_INSTANTIATE(__half, 64, 128)
