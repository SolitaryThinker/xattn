// Head-dimension bucket selection shared by dispatch and leaf preflight.
// This header has no CUDA or tensor-library dependencies.
#pragma once

#include <cstdint>

namespace xattn::ops::flash_sca_native {

// Select the compiled ceiling bucket, with the largest bucket saturated at 256.
// This is NOT dimension validation: nonpositive inputs still map to 64 and
// dimensions above 256 still map to 256. The public boundary and each leaf's
// preflight enforce their own supported range. Logical extents remain unchanged;
// padded input columns are zero-filled and output stores must use those extents.
inline constexpr int QkBucket(int64_t dim) {
  return dim <= 64 ? 64 : dim <= 128 ? 128 : 256;
}

inline constexpr int ValueBucket(int64_t dim) { return QkBucket(dim); }

}  // namespace xattn::ops::flash_sca_native
