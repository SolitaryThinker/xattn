#pragma once

#include <pybind11/pybind11.h>

#if defined(XATTN_HAS_NATIVE_PRECISION) && !defined(XATTN_HAS_NATIVE_READER)
#error "Native precision requires native reader entry points."
#endif

namespace xattn::ops {

// Registered only when the native host boundary and real CUDA readers are
// linked. Precision support additionally requires the complete native
// statistics, main/ordered contractions and repair helpers.
void DefineNativeAttentionOps(pybind11::module_& m);

}  // namespace xattn::ops
