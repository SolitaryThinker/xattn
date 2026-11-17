// Compile check for the backward launch preparation (native/umma_prepare.h),
// which is a host-only header: the driver that calls it is a separate change,
// so without this translation unit the header would be text no compiler had
// seen. It lives in the instantiations group because that group is what the
// package build compiles for the XATTN_HAS_NATIVE_SCA configuration, and the
// header belongs to that group.
//
// It defines nothing and links nothing. If it stops compiling, the preparation
// has drifted from the ABI in params.h / umma_params.h.
#include "flash_sca/native/umma_prepare.h"
