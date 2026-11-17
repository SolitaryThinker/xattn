// Native Blackwell FlashSCA kernels for D bucket 256, Dv bucket 64, fp16.
#include "flash_sca/native/launch.cuh"

XATTN_SCA_NATIVE_INSTANTIATE(__half, 256, 64)
