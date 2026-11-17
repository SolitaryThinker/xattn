// The per-query-row visibility bounds of the native FlashSCA metadata boundary, computed on
// the device in ONE launch instead of by the Python route's ~14-kernel chain
// (xattn/ops/_sca_native.py:visibility).
//
// What it computes, for a batch row whose key segment ids are nondecreasing (a BOS mask's
// cumsum is, by construction), is exactly the Python sorted branch, integer for integer:
//   p        = i + (total - q_length)          (Q right-aligned to the keys)
//   lo       = max(0, (p / chunk - 1) * chunk)
//   hi       = p - strict
//   first    = lower_bound(ids, ids[p])         (torch.searchsorted(ids, q_ids), left)
//   last     = upper_bound(ids, ids[p]) - 1     (torch.searchsorted(ids, q_ids, right=True) - 1)
//   row_lo   = max(lo, first)
//   row_hi   = min(hi, last)
// so a row with row_lo > row_hi is dead (zero output, exactly zero gradient) as before, and
// no float is involved: the values are integers and the two routes agree bit for bit.
//
// The kernel is a metadata kernel like TileBoundsKernel: it belongs to no (D, Dv) bucket and
// serves every body that reads per-row bounds (the pipelined forward, the generic mma.sync
// forward, and the backward's leaves), in both element types.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace xattn::ops::flash_sca_native {

struct RowBoundsArgs {
  const int64_t* ids = nullptr;  // [batch, total], nondecreasing per row
  int32_t* row_lo = nullptr;     // [batch, q_length]
  int32_t* row_hi = nullptr;     // [batch, q_length]
  int64_t batch = 0;
  int64_t q_length = 0;
  int64_t total = 0;
  int64_t chunk = 0;
  // 1 drops each query's own key (SoftDelta's strict past); 0 keeps it.
  int strict = 0;
};

// The same bounds, one step earlier: straight from the BOS mask, with no cumsum and neither
// the bool->int64 cast nor the ids plane built at all. A "start" of the mask is a segment
// start, so the ids route's lower_bound/upper_bound over the cumsum are, integer for integer,
//   first = the last start at or before p   (0 if the row has no start at all)
//   last  = the next start after p - 1      (total - 1 if the row has no start after p)
// which needs the mask's set bytes only. Every value is an integer and no reduction over the
// mask is order-dependent (the per-chunk first/last are a min/max, which commute), so this and
// the ids route agree bit for bit, and both agree with the Python chain.
//
// Rows above kMaskMaxKeys keys keep the ids route: the per-chunk start table is sized by the
// row length (about 17 KB of shared memory at the cap), and above the cap the caller builds
// the cumsum and uses RowBoundsKernel instead. The two routes compute the same integers, so
// where the switch happens changes nothing but the cost.
constexpr int64_t kMaskMaxKeys = 131072;

struct RowBoundsMaskArgs {
  const uint8_t* mask = nullptr;  // [batch, total], bool: 1 starts a segment
  int32_t* row_lo = nullptr;      // [batch, q_length]
  int32_t* row_hi = nullptr;      // [batch, q_length]
  int64_t batch = 0;
  int64_t q_length = 0;
  int64_t total = 0;  // <= kMaskMaxKeys
  int64_t chunk = 0;
  // 1 drops each query's own key (SoftDelta's strict past); 0 keeps it.
  int strict = 0;
};

void LaunchRowBounds(const RowBoundsArgs& args, cudaStream_t stream);
void LaunchRowBoundsMask(const RowBoundsMaskArgs& args, cudaStream_t stream);

}  // namespace xattn::ops::flash_sca_native