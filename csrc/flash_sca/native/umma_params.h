// Launch ABI for the SM100-family UMMA sliding-chunk attention kernels.
// This header deliberately contains no CuTe types: host preparation depends
// on runtime tensor strides, while operand layouts are kernel-template choices.
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "flash_sca/native/params.h"

namespace xattn::ops::flash_sca_native::umma {

// One descriptor serves every phase. Larger tiles issue several 64-by-64
// transfers. The 64 contiguous 16-bit elements fit the 128-byte TMA swizzle.
inline constexpr int kTmaRows = 64;
inline constexpr int kTmaColumns = 64;
// A split previous/current map must not cut a key tile in any initial kernel.
inline constexpr int kKeyMapAlignment = 128;

struct UmmaInputMaps {
  CUtensorMap q{};
  CUtensorMap k{};
  CUtensorMap v{};
  CUtensorMap prev_k{};
  CUtensorMap prev_v{};
  CUtensorMap dy{};

  // False: k/v describe current K/V and prev_k/prev_v describe the previous
  // chunk when present. True: k/v describe the packed total [previous;current]
  // key axis and prev_k/prev_v must not be read. Params::prev_length always
  // retains its logical value for masks and the gradient output split.
  bool packed_kv = false;
};

struct UmmaLaunchParams {
  Params params;
  UmmaInputMaps maps;
};

// Host launch wrappers accept a reference; the CUDA kernels take a const
// __grid_constant__ UmmaLaunchParams value to keep the descriptors immutable
// and accessible to TMA for the duration of the launch.
//
// Returns true when the requested work was launched, false when preflight
// refused it before any output was modified. A multi-stage backward must check
// every required leaf before starting its first output-producing stage. After
// a stage starts, errors must propagate; returning false and replaying a
// fallback over partially written gradients is not permitted.
// On false, the caller must select another implementation rather than consume
// unwritten outputs. A linkable refusal stub is not a native implementation.
template <typename E, int kD, int kDv>
bool LaunchUmmaForward(const UmmaLaunchParams& p, cudaStream_t stream);

template <typename E, int kD, int kDv>
bool LaunchUmmaBackward(const UmmaLaunchParams& p, cudaStream_t stream);

}  // namespace xattn::ops::flash_sca_native::umma
