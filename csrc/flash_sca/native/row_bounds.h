// Row bounds of the native FlashSCA metadata boundary, on the device, in one launch: both live
// SCA runs added this header independently, and the combined tree (COMBINED.md) keeps both
// declarations as each main has them. They compute the same integers.
//   - The forward main's ids and mask routes, LaunchRowBounds / LaunchRowBoundsMask (defined in
//     dispatch.cu): the forward's host route (host.cc Forward/ForwardMask, and Backward when a
//     caller hands it the mask) and _flash_sca_native_row_bounds.
//   - The backward main's BOS route, LaunchRowBoundsFromBos (defined in row_bounds.cu):
//     _flash_sca_row_bounds, which the public backward's metadata takes
//     (xattn/ops/_sca_native.py:visibility with bos_in_call=False).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace xattn::ops::flash_sca_native {

// ---- The forward main's routes ----
//
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

// ---- The backward main's route ----
//
// Row bounds of a BOS mask, on the device, in one launch.
//
// The public FlashSCA API accepts `bos_mask` (one `True` at the first token of
// every document of a packed sequence). Everything the kernels need from it is,
// per query row, the intersection of the analytic chunk window with the key run
// of that row's segment (see `RowBounds` in visibility.cuh):
//
//   seg_start(p) = position of the last BOS at or before key position p, else 0
//   lo = max(chunk_base(p), seg_start(p))               (reset: seg_start + local
//                                                        chunk base inside the run)
//   hi = p - strict
//
// where the query row at index i sits at key position p = i + (total - q_length).
// The BOS ids are the cumsum of the mask, so a row's segment is exactly this run
// and the whole rule is that one scan value: the segment's last key is at or
// after p, so the visibility interval can only be cut short by its start.
//
// Before this kernel the route reduced the mask with ~10 torch ops per call
// (cumsum over [B, total], two `searchsorted` over the packed key axis, ...),
// measured at 135-157 us/call on every objective shape; that is one launch here.

// Row bounds of one batch of BOS masks. `bos` is a bool (one byte, 0/1) array of
// shape [batch, total]; `lo`/`hi` are written as [batch, q_length] int32.
// `shift` is total - q_length (the query axis is right-aligned to the keys).
void LaunchRowBoundsFromBos(const unsigned char* bos, int32_t* lo, int32_t* hi,
                            int64_t batch, int64_t total, int64_t q_length, int64_t shift,
                            int32_t chunk_size, bool strict_past, bool reset,
                            cudaStream_t stream);

// H25: the segment ids a packed-sequence BOS mask names (one int64 per key, the count of BOS
// positions at or before it: the same values as `bos_mask.cumsum(dim=1, dtype=torch.int64)`,
// bit for bit -- an inclusive integer prefix sum, so there is only one correct result and every
// sound scan gives it). SoftDelta's composed backward passes these ids (not the mask) to its
// private reads, which read a per-key segment id on chip (SegmentTest / `p.q_segment`,
// `p.k_segment`); building them with torch's generic scan measured 53-58 us/call (one launch,
// shared by both reads since D7/H9), independent of the mask's visible-key share. One block per
// batch row, a two-pass block scan (every thread's own contiguous run, then a block-wide
// Hillis-Steele scan of the per-thread totals): batch is at most a few dozen for any shape the
// API accepts, so a few blocks finish in microseconds.
void LaunchSegmentIdsFromBos(const unsigned char* bos, int64_t* ids, int64_t batch,
                             int64_t total, cudaStream_t stream);
// (sha marker)

}  // namespace xattn::ops::flash_sca_native
