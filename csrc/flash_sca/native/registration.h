// Registration of the native Blackwell FlashSCA ops. Defined only when the
// complete XATTN_HAS_NATIVE_SCA source group is linked (see setup.py).
#pragma once

#include <pybind11/pybind11.h>

namespace xattn::ops {

void DefineNativeFlashSCAOps(pybind11::module_& m);

}  // namespace xattn::ops
