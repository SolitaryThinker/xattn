// Segment-aware tile eligibility for the native (tcgen05/UMMA) SCA kernels.
//
// One rule, consumed the same way by every native SCA family: a query tile may
// skip a key tile only when MaySee() says it cannot see any key of the range.
// MaySee() == false is the ONLY skip licence; FullTile() is the only unmasked
// licence and it is narrower (see its contract). Nothing here enumerates tiles,
// launches anything, or adds a field to Params.
//
// Two consumers, two questions:
//   * FromTable() reads Params::tile_bounds (the ABI dispatch.cu:TileBoundsKernel
//     writes) and answers with exactly the numbers detail::ReadTileRange answers,
//     so this descriptor and KeyTileRangeKernel never disagree on a tile.
//   * ReduceFromRows()/LookupFromRows() read the per-row bounds of one query tile;
//     ReduceFromRows is the exact live hull and is the form to use where a padded
//     tail must not disable the unmasked path (see kPaaddedTail note).
//
// Producer contract (the one assumption): row_lo/row_hi enclose the visible key
// interval of every live query row. xattn/ops/_sca_native.py:visibility() supplies
// run-clipped bounds where the segment ids prove the window and the plain chunk
// window on rows with segment_exact[b] == 0. On those rows the bounds are a strict
// superset and the per-key k_segment == q_segment test must stay in the kernel:
// nothing here is allowed to tighten them, and no count from this header is a
// count of visible pairs.
#pragma once

#include <climits>
#include <cstdint>

#include "flash_sca/native/params.h"
#include "flash_sca/native/visibility.cuh"

namespace xattn::ops::flash_sca_native {
namespace umma {

// Query tile = 128 rows = two 64-row tile_bounds entries; a 64-row kernel head
// passes halves = 1. Key tiles are 64 or 128 keys on the GLOBAL key axis.
constexpr int kQueryTileRows = 128;
constexpr int kKeyTileRows64 = 64;
constexpr int kKeyTileRows128 = 128;
static_assert(kQueryTileRows == 2 * detail::kTileRows,
              "a 128-row query descriptor must be exactly two tile_bounds entries");

enum QueryTileMaskFlag : uint32_t {
  kNoLiveRow = 1u << 0,     // every row of the tile is dead: MaySee is always false
  kHalfDead = 1u << 1,      // an in-range half holds a dead or absent row: FullTile is false
  kFlagLookupRoute = 1u << 2,  // produced by LookupFromRows, valid only under RowBoundsMonotone
  kSegmentTest = 1u << 3,   // this batch row keeps the per-key segment test (AnnotateSegmentTest)
};

// Descriptor of one query tile against the key axis [0, total_length): previous
// K/V rows 0..P-1, current K/V rows P..P+Nk-1.
struct QueryTileMask {
  int lo = INT_MAX;       // min lo over the tile's live rows (exact envelope)
  int hi = -1;            // max hi over the tile's live rows
  int full_lo = INT_MAX;  // window every live row contains whole (max lo, min hi)
  int full_hi = -1;
  uint32_t flags = static_cast<uint32_t>(kNoLiveRow);
};

__host__ __device__ __forceinline__ bool IsEmpty(const QueryTileMask& m) { return m.lo > m.hi; }

// True iff a visible pair may exist between a live row of the tile and a key of
// [key_begin, key_end). False => no visible pair, so the tile may be skipped.
// True says nothing about which keys are visible: with lo and hi taken from
// different rows the envelope can cover a gap, so tiles strictly inside
// [first, last] may still fail this test.
__host__ __device__ __forceinline__ bool MaySee(const QueryTileMask& m, int64_t key_begin,
                                                int64_t key_end) {
  if (key_end <= key_begin || IsEmpty(m)) return false;
  return key_begin <= m.hi && key_end > m.lo;
}

// Is [key_begin, key_end) wholly inside every live row's chunk window? This is a
// window claim, not a visibility claim: while SegmentTest() holds for the batch
// row the window is the plain chunk band, so a key of [key_begin, key_end) can
// belong to another segment. In that case this licenses skipping the MASK, never
// the per-key segment test; use FullTileExact() (device) when that test is on.
__host__ __device__ __forceinline__ bool FullTile(const QueryTileMask& m, int64_t key_begin,
                                                  int64_t key_end) {
  if (key_end <= key_begin || IsEmpty(m)) return false;
  return key_begin >= m.full_lo && key_end - 1 <= m.full_hi;
}

// The numbers of detail::ReadTileRange on the same table, mapped to a descriptor:
//   lo/hi            = min_lo/max_hi over live rows of the tile,
//   full_lo/full_hi  = max_lo/min_hi over EVERY row of the tile (padding included,
//                      as TileBoundsKernel writes them), collapsed when that pair
//                      is not a valid interval.
// Consequences a consumer must know:
//   * kHalfDead is set whenever an in-range half holds a dead row OR the tile is a
//     padded tail (q_length % 64 != 0). FullTile is then false: a dead or absent
//     row cannot see the window whole. This was measured on the shipped
//     kernel; the traced shapes (4096/16384/65536) are multiples of 64, so they
//     keep the elision, while contract-shaped calls (Nq <= 517) do not.
//   * Use ReduceFromRows() when a padded tail must not disable the unmasked path:
//     it skips dead and absent rows and returns the true live hull.
__host__ __device__ __forceinline__ QueryTileMask FromTable(const Params& p, int64_t b,
                                                            int64_t first_half, int halves) {
  QueryTileMask m;
  m.lo = INT_MAX;
  m.hi = -1;
  m.full_lo = INT_MIN;
  m.full_hi = INT_MAX;
  m.flags = 0;
  for (int h = 0; h < halves; ++h) {
    const int64_t tile = first_half + h;
    if (tile >= p.tile_count) {
      m.full_lo = INT_MAX;  // rows beyond q_length are dead, same as ReadTileRange
      m.full_hi = -1;
      continue;
    }
    const int32_t* e = p.tile_bounds + (b * p.tile_count + tile) * 4;
    m.lo = e[0] < m.lo ? e[0] : m.lo;
    m.hi = e[1] > m.hi ? e[1] : m.hi;
    m.full_lo = e[2] > m.full_lo ? e[2] : m.full_lo;
    m.full_hi = e[3] < m.full_hi ? e[3] : m.full_hi;
  }
  m.flags = 0;
  if (m.lo > m.hi) {
    m.flags |= static_cast<uint32_t>(kNoLiveRow);
    m.lo = INT_MAX;
    m.hi = -1;
  }
  if (m.full_lo > m.full_hi) {
    m.flags |= static_cast<uint32_t>(kHalfDead);
    m.full_lo = INT_MAX;
    m.full_hi = -1;
  }
  return m;
}

// Exact descriptor of query rows [first_row, first_row + rows) read row by row.
// Skips dead rows and rows past q_length, so a padded tail does not set kHalfDead
// and full_lo/full_hi is the window every live row really contains. O(rows).
__host__ __device__ __forceinline__ QueryTileMask ReduceFromRows(const int32_t* row_lo,
                                                                 const int32_t* row_hi,
                                                                 int64_t q_length, int64_t b,
                                                                 int64_t first_row, int rows) {
  QueryTileMask m;
  for (int64_t x = 0; x < rows; ++x) {
    const int64_t i = first_row + x;
    if (i >= q_length) break;
    const int lo = row_lo[b * q_length + i];
    const int hi = row_hi[b * q_length + i];
    if (lo > hi) continue;
    if (m.lo > m.hi) m.flags &= ~static_cast<uint32_t>(kNoLiveRow);
    m.lo = min(m.lo, lo);
    m.hi = max(m.hi, hi);
    m.full_lo = (m.full_lo == INT_MAX) ? lo : (m.full_lo > lo ? m.full_lo : lo);
    m.full_hi = (m.full_hi < 0) ? hi : (m.full_hi < hi ? m.full_hi : hi);
  }
  if (m.lo > m.hi) {
    m.flags |= static_cast<uint32_t>(kNoLiveRow);
    m.lo = INT_MAX;
    m.hi = -1;
  }
  return m;
}

// Endpoint form of ReduceFromRows for rows whose lo/hi are nondecreasing and whose
// dead rows are a leading run (the lookup route): lo = lo[first live],
// hi = hi[last live], full window = [lo[last live], hi[first live]].
// Equal to ReduceFromRows for every tile of the row when RowBoundsMonotone() holds.
__host__ __device__ __forceinline__ QueryTileMask LookupFromRows(const int32_t* row_lo,
                                                                 const int32_t* row_hi,
                                                                 int64_t q_length, int64_t b,
                                                                 int64_t first_row, int rows) {
  int64_t i0 = -1, i1 = -1;
  for (int64_t x = 0; x < rows; ++x) {
    const int64_t i = first_row + x;
    if (i >= q_length) break;
    if (row_lo[b * q_length + i] <= row_hi[b * q_length + i]) {
      if (i0 < 0) i0 = i;
      i1 = i;
    }
  }
  QueryTileMask m;
  if (i0 < 0) return m;
  m.lo = row_lo[b * q_length + i0];
  m.hi = row_hi[b * q_length + i1];
  m.full_lo = row_lo[b * q_length + i1];
  m.full_hi = row_hi[b * q_length + i0];
  m.flags = static_cast<uint32_t>(kFlagLookupRoute);
  // full_lo > full_hi is NOT an empty tile: it means no key is visible to every
  // live row (routine when chunk < 64). Never collapse the envelope here -- that
  // would turn a live tile into a skip. FullTile already refuses such a window.
  return m;
}

// Host-only route predicate, O(q_length): true iff lo and hi are nondecreasing
// along the live rows and no dead row sits after the first live one (a dead row
// may only precede the live run, so the live tiles are a suffix). An all-dead row
// returns true and yields no live tile. True => LookupFromRows == ReduceFromRows
// for every query tile of that batch row, and only then may a caller take full_*
// from the lookup route: on a non-monotone row its endpoint identity under-visits
// (e.g. (5,20),(0,10),(5,20),(0,10) gives full [0,20] against the true [5,10]).
// Never call it per query tile on device.
inline bool RowBoundsMonotone(const int32_t* row_lo, const int32_t* row_hi, int64_t q_length,
                              int64_t b) {
  bool seen = false;
  int prev_lo = INT_MIN, prev_hi = INT_MIN;
  for (int64_t i = 0; i < q_length; ++i) {
    const int lo = row_lo[b * q_length + i];
    const int hi = row_hi[b * q_length + i];
    if (lo > hi) {
      if (seen) return false;  // a dead row after a live row, or a gap in the live run
      continue;
    }
    if (seen && (lo < prev_lo || hi < prev_hi)) return false;
    prev_lo = lo;
    prev_hi = hi;
    seen = true;
  }
  return true;
}

enum RowRoute : int32_t { kRouteReduce = 0, kRouteLookup = 1 };

// The route array is supplied by the caller and lives outside this header: no
// field of Params carries it today, so every current call site passes nullptr and
// gets kRouteReduce, which needs no per-row precondition. If a route array is ever
// added it belongs in the advanced (UmmaLaunchParams) struct, not in the base
// launch ABI. kRouteLookup is informational: it tells a kernel the row satisfies
// RowBoundsMonotone, and it is never a licence to skip anything by itself.
__host__ __device__ __forceinline__ RowRoute RowRouteAt(const int32_t* route, int64_t b) {
  if (route == nullptr) return kRouteReduce;
  return route[b] == static_cast<int32_t>(kRouteLookup) ? kRouteLookup : kRouteReduce;
}

__host__ __device__ __forceinline__ void AnnotateRoute(QueryTileMask& m, RowRoute route) {
  if (route == kRouteLookup) m.flags |= static_cast<uint32_t>(kFlagLookupRoute);
}

// Device-only: verbatim detail::SegmentTest (visibility.cuh), no fork. This is the
// flag's only producer, so a kernel that branches on kSegmentTest must call it
// once per batch row: AnnotateSegmentTest(m, p, b).
__device__ __forceinline__ bool NeedsSegmentTest(const Params& p, int64_t b) {
  return detail::SegmentTest(p, b);
}

__device__ __forceinline__ void AnnotateSegmentTest(QueryTileMask& m, const Params& p,
                                                    int64_t b) {
  if (NeedsSegmentTest(p, b)) m.flags |= static_cast<uint32_t>(kSegmentTest);
}

__device__ __forceinline__ bool FullTileExact(const QueryTileMask& m, const Params& p, int64_t b,
                                              int64_t key_begin, int64_t key_end) {
  if (NeedsSegmentTest(p, b)) return false;
  return FullTile(m, key_begin, key_end);
}

// Key-owned loops: the BOUNDING interval of query tiles that may see the key range.
// Empty iff first > last (also when key_end <= key_begin), written as
// *first = 0, *last = -1 -- test emptiness with first > last, never by comparing
// first against a sentinel such as INT_MAX. A tile inside the interval may still
// fail MaySee: a caller that needs the exact set must test each tile, and it may
// leave the interval early only on MaySee() == false.
__host__ __device__ __forceinline__ void QueryTilesForKeyScan(const QueryTileMask* masks,
                                                              int64_t q_tiles, int64_t key_begin,
                                                              int64_t key_end, int* first,
                                                              int* last) {
  *first = 0;
  *last = -1;
  if (key_end <= key_begin) return;
  int f = -1, l = -1;
  for (int64_t t = 0; t < q_tiles; ++t) {
    if (!MaySee(masks[t], key_begin, key_end)) continue;
    if (f < 0) f = static_cast<int>(t);
    l = static_cast<int>(t);
  }
  if (f < 0) return;
  *first = f;
  *last = l;
}

// Same answer as QueryTilesForKeyScan, O(log) per bound. Preconditions: the row's
// dead rows form a leading run, so the live tiles are a suffix [first_live,
// q_tiles); on that suffix lo and hi are nondecreasing (RowBoundsMonotone). Both
// searches run inside the suffix, which is what keeps a leading empty tile from
// yielding a too-small last (an empty tile has lo = INT_MAX and would fail every
// search predicate).
__host__ __device__ __forceinline__ void QueryTilesForKeyBisect(const QueryTileMask* masks,
                                                                int64_t q_tiles,
                                                                int64_t key_begin, int64_t key_end,
                                                                int* first, int* last) {
  *first = 0;
  *last = -1;
  if (key_end <= key_begin) return;
  int64_t first_live = q_tiles;
  for (int64_t t = 0; t < q_tiles; ++t) {
    if (!IsEmpty(masks[t])) {
      first_live = t;
      break;
    }
  }
  if (first_live >= q_tiles) return;
  // first tile with hi >= key_begin
  int64_t lo = first_live, hi = q_tiles;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (masks[mid].hi >= key_begin) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  const int64_t f = lo;
  if (f >= q_tiles || masks[f].lo > key_end - 1) return;
  // last tile with lo <= key_end - 1, searched inside [f, q_tiles)
  lo = f;
  hi = q_tiles;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (masks[mid].lo <= key_end - 1) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  *first = static_cast<int>(f);
  *last = static_cast<int>(lo) - 1;
}

// Where a global key tile lives. Preconditions the caller owns: global_tile *
// tile_rows < total_length (a tile past the end returns a valid current map), and
// every key range handed to MaySee is on the GLOBAL key axis (0..P-1 previous,
// P..P+Nk-1 current), which is what makes the split-map case safe.
struct KeyTilePlacement {
  enum Map : int32_t { kCurrentOrPacked = 0, kPrevious = 1 };
  int32_t map;
  int64_t local_row;
  bool straddles;
};

// packed: the host materialised previous K/V in front of current K/V, so global row
// r is buffer row r under one map. Unpacked (P % 128 == 0 under the host rule, so
// P % tile_rows == 0 for 64- and 128-key tiles) every tile is wholly in one map.
// straddles is a tripwire for a broken host rule, never a licence.
__host__ __device__ __forceinline__ KeyTilePlacement PlaceKeyTile(int64_t prev_length,
                                                                  int64_t total_length,
                                                                  bool packed, int tile_rows,
                                                                  int64_t global_tile) {
  KeyTilePlacement out;
  out.straddles = false;
  const int64_t begin = global_tile * tile_rows;
  const int64_t end = begin + tile_rows < total_length ? begin + tile_rows : total_length;
  if (prev_length == 0 || packed) {
    out.map = KeyTilePlacement::kCurrentOrPacked;
    out.local_row = begin;
    return out;
  }
  if (end <= prev_length) {
    out.map = KeyTilePlacement::kPrevious;
    out.local_row = begin;
    return out;
  }
  if (begin >= prev_length) {
    out.map = KeyTilePlacement::kCurrentOrPacked;
    out.local_row = begin - prev_length;
    return out;
  }
  out.map = KeyTilePlacement::kCurrentOrPacked;
  out.local_row = 0;
  out.straddles = true;
  return out;
}

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
