// Native Blackwell FlashSCA kernels for D bucket 128, Dv bucket 128, fp16.
#include "flash_sca/native/launch.cuh"

XATTN_SCA_NATIVE_INSTANTIATE(__half, 128, 128)
XATTN_SCA_NATIVE_INSTANTIATE_ELEMENT(__half)
