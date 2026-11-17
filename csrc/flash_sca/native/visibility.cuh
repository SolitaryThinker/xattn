// Shared sliding-chunk visibility for the Blackwell SCA kernel families.
// The bounds and table ABI are defined in params.h. Keep these helpers
// independent of the tensor-core implementation so every kernel evaluates
// the same mask, including empty rows and unsorted segment identifiers.
#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "flash_sca/native/params.h"

namespace xattn::ops::flash_sca_native {
namespace detail {

constexpr int kTileRows = 64;  // granularity of Params::tile_bounds

struct Bounds {
  int lo;
  int hi;
};

__device__ __forceinline__ Bounds RowBounds(const Params& p, int64_t b, int64_t i) {
  if (i < 0 || i >= p.q_length) return {INT_MAX, -1};
  int64_t lo, hi;
  if (p.row_lo != nullptr) {
    lo = p.row_lo[b * p.q_length + i];
    hi = p.row_hi[b * p.q_length + i];
  } else {
    const int64_t position = i + p.total_length - p.q_length;
    lo = (position / p.chunk - 1) * p.chunk;
    hi = position - (p.strict ? 1 : 0);
  }
  lo = lo < 0 ? 0 : lo;
  hi = hi >= p.total_length ? p.total_length - 1 : hi;
  if (lo > hi) return {INT_MAX, -1};
  return {static_cast<int>(lo), static_cast<int>(hi)};
}

__device__ __forceinline__ bool SegmentTest(const Params& p, int64_t b) {
  return p.q_segment != nullptr && !(p.segment_exact != nullptr && p.segment_exact[b] != 0);
}

// Bounds of query rows [row_begin, row_end) from the 64-row tile table:
// min lo / max hi over live rows, max lo / min hi over every row.
struct TileRange {
  int min_lo, max_hi, max_lo, min_hi;
};

__device__ __forceinline__ TileRange ReadTileRange(const Params& p, int64_t b, int64_t first_tile,
                                                   int tiles) {
  TileRange r{INT_MAX, -1, INT_MIN, INT_MAX};
  for (int t = 0; t < tiles; ++t) {
    const int64_t tile = first_tile + t;
    if (tile >= p.tile_count) {
      r.max_lo = INT_MAX;  // rows beyond Nq are dead
      r.min_hi = -1;
      continue;
    }
    const int32_t* e = p.tile_bounds + (b * p.tile_count + tile) * 4;
    r.min_lo = min(r.min_lo, e[0]);
    r.max_hi = max(r.max_hi, e[1]);
    r.max_lo = max(r.max_lo, e[2]);
    r.min_hi = min(r.min_hi, e[3]);
  }
  return r;
}

}  // namespace detail
}  // namespace xattn::ops::flash_sca_native
