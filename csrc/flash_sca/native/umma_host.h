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
  // thread. Bind the device's primary context once per (thread, device).
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  static thread_local int bound_device = -1;
  if (bound_device != device) {
    C10_CUDA_CHECK(cudaSetDevice(device));
    bound_device = device;
  }
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

// Per-thread cache of encoded 128-byte TMA descriptors keyed by every field EncodeInputMap reads.
// Shared across forward (q, k, v, prev_k, prev_v, y) and backward (q, dy, k, v, prev_k, prev_v).
struct InputMapKey {
  int device = -1;
  const void* ptr = nullptr;
  int64_t batch_stride = 0, row_stride = 0, head_stride = 0;
  int element = 0;
  int64_t batch = 0, rows = 0, heads = 0, columns = 0;
  bool operator==(const InputMapKey& o) const {
    return device == o.device && ptr == o.ptr && batch_stride == o.batch_stride &&
           row_stride == o.row_stride && head_stride == o.head_stride && element == o.element &&
           batch == o.batch && rows == o.rows && heads == o.heads && columns == o.columns;
  }
};

inline CUtensorMap CachedInputMap(const Operand& source, Element element, int64_t batch,
                                  int64_t rows, int64_t heads, int64_t columns, int device) {
  constexpr int kEntries = 32;
  struct Entry {
    InputMapKey key;
    CUtensorMap map;
    bool valid = false;
  };
  thread_local Entry entries[kEntries];
  thread_local int next = 0;
  InputMapKey key;
  key.device = device;
  key.ptr = source.ptr;
  key.batch_stride = source.batch_stride;
  key.row_stride = source.row_stride;
  key.head_stride = source.head_stride;
  key.element = static_cast<int>(element);
  key.batch = batch;
  key.rows = rows;
  key.heads = heads;
  key.columns = columns;
  for (const Entry& e : entries) {
    if (e.valid && e.key == key) return e.map;
  }
  const CUtensorMap map = EncodeInputMap(source, element, batch, rows, heads, columns);
  Entry& slot = entries[next];
  next = (next + 1) & (kEntries - 1);
  slot.key = key;
  slot.map = map;
  slot.valid = true;
  return map;
}

inline CUtensorMap CachedInputMap(const Operand& source, Element element, int64_t batch,
                                  int64_t rows, int64_t heads, int64_t columns) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  return CachedInputMap(source, element, batch, rows, heads, columns, device);
}

}  // namespace xattn::ops::flash_sca_native::umma
