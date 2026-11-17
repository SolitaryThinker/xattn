// Host-side TMA descriptor construction for the UMMA SCA launch ABI.
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <array>
#include <cstdint>

#include "flash_sca/native/umma_params.h"

namespace xattn::ops::flash_sca_native::umma {

// Resolve through the CUDA runtime rather than adding an implicit libcuda
// link dependency to the extension. The function address is process-local;
// descriptors themselves are rebuilt for each call's actual tensor storage.
inline PFN_cuTensorMapEncodeTiled_v12000 TensorMapEncoder() {
  static const auto encode = [] {
    void* address = nullptr;
    cudaDriverEntryPointQueryResult query{};
#if CUDART_VERSION >= 13000
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &address, 12000, cudaEnableDefault, &query));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint(
        "cuTensorMapEncodeTiled", &address, cudaEnableDefault, &query));
#endif
    TORCH_CHECK(query == cudaDriverEntryPointSuccess && address != nullptr,
                "UMMA FlashSCA could not resolve cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
  }();
  return encode;
}

// All source data are 16-bit. Extents retain the logical width, so a final
// partial 64-column box is zero-filled without reading the adjacent head.
// The map axes are [column, head, row, batch]; a tile starts at [d,h,r,b].
inline CUtensorMap EncodeInputMap(const Operand& source, Element element,
                                  int64_t batch, int64_t rows,
                                  int64_t heads, int64_t columns) {
  CUtensorMap map{};
  if (batch == 0 || rows == 0) return map;  // an absent map is never consumed
  TORCH_CHECK(source.ptr != nullptr && batch > 0 && rows > 0 && heads > 0 &&
                  columns > 0 && columns <= 256 && columns % 8 == 0,
              "UMMA FlashSCA received invalid TMA source extents");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(source.ptr) % 16 == 0,
              "UMMA FlashSCA TMA sources must be 16-byte aligned");

  std::array<cuuint64_t, 4> extent{static_cast<cuuint64_t>(columns),
                                  static_cast<cuuint64_t>(heads),
                                  static_cast<cuuint64_t>(rows),
                                  static_cast<cuuint64_t>(batch)};
  // Strides on singleton axes never affect an address. Normalize those axes
  // too: tensor views may report zero there, but TMA still validates strides.
  int64_t head_stride = heads == 1 ? columns : source.head_stride;
  int64_t row_stride = rows == 1 ? head_stride * heads : source.row_stride;
  int64_t batch_stride = batch == 1 ? row_stride * rows : source.batch_stride;
  std::array<int64_t, 3> element_stride{head_stride, row_stride, batch_stride};
  std::array<cuuint64_t, 3> byte_stride{};
  for (int i = 0; i < 3; ++i) {
    TORCH_CHECK(element_stride[i] > 0 && element_stride[i] % 8 == 0 &&
                    element_stride[i] < (int64_t(1) << 39),
                "UMMA FlashSCA requires positive TMA strides aligned to 16 bytes");
    byte_stride[i] = static_cast<cuuint64_t>(element_stride[i]) * 2;
  }
  const std::array<cuuint32_t, 4> box{kTmaColumns, 1, kTmaRows, 1};
  const std::array<cuuint32_t, 4> step{1, 1, 1, 1};
  const auto type = element == Element::kHalf ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                             : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  // The encoder is a driver-API call and needs a context current on this host
  // thread. The runtime binds the primary context lazily, at a thread's first
  // runtime call that needs it, and PyTorch's device guard skips cudaSetDevice
  // when the thread's device already matches. So on a thread whose first CUDA
  // work is this call (the autograd engine's device thread running a native
  // backward straight off flash_sca, or a recompute there) no context is
  // current and the encode fails with CUDA_ERROR_INVALID_CONTEXT (201).
  // cudaSetDevice makes the device's primary context current at once.
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaSetDevice(device));
  const CUresult result = TensorMapEncoder()(
      &map, type, 4, const_cast<void*>(source.ptr), extent.data(),
      byte_stride.data(), box.data(), step.data(), CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "UMMA FlashSCA could not encode TMA source: CUDA driver error ",
              static_cast<int>(result), "; shape=[", batch, ",", rows, ",", heads, ",",
              columns, "]");
  return map;
}

}  // namespace xattn::ops::flash_sca_native::umma
