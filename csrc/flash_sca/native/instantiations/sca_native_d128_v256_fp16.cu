// Native Blackwell FlashSCA kernels for D bucket 128, Dv bucket 256, fp16.
#include "flash_sca/native/launch.cuh"

XATTN_SCA_NATIVE_INSTANTIATE(__half, 128, 256)
