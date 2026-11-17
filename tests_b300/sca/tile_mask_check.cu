// Conformance test for csrc/flash_sca/native/umma_tile_mask.cuh.
//
// Torch-free, host-only: builds Params exactly as the host boundary does (64-row
// bound table from per-row [lo,hi], TileBoundsKernel's four ints per entry), then
// checks the header's contract on generated metadata of every route:
//
//  A. FromTable == the documented reduction: lo/hi = min live lo / max hi over the
//     tile's halves; full_lo/full_hi = max lo / min hi over EVERY row of the halves
//     (the ABI TileBoundsKernel writes), collapsed when that is not an interval, and
//     kHalfDead exactly when a half holds a dead row or lies past tile_count.
//  B. FromTable's lo/hi == ReduceFromRows' lo/hi: the table path and the row path
//     agree on the envelope (they may differ on the full window by design).
//  C. ReduceFromRows == the exact live hull (dead and absent rows skipped, never
//     collapsed), so a padded tail does not disable FullTile for that form.
//  D. MaySee is sound in ONE direction: a tile that holds a visible pair passes it
//     (see => MaySee). Equality is checked only where bounds are monotone, where the
//     envelope has no gap by construction.
//  E. FullTile => every live row's window contains the key range whole.
//  F. RowBoundsMonotone true => LookupFromRows == ReduceFromRows on all four fields
//     for every query tile, and QueryTilesForKeyBisect == QueryTilesForKeyScan.
//  G. Leading empty tiles on the lookup route: scan and bisect agree (the search
//     must run inside the live suffix), and key_begin == key_end is empty in both.
//  H. Mutants: widened-lo and narrowed-hi bound sequences are rejected by
//     RowBoundsMonotone, and forcing the bisect on them differs from the scan --
//     the test fails if a mutant slips through, so the arm cannot pass vacuously.
//  I. Odd-Nq tails: a half past tile_count counts as poison (INT_MAX/-1), so the
//     descriptor equals the one built from a table padded with those sentinels.
//  J. PlaceKeyTile: a tile straddles iff P % tile_rows != 0; aligned boundaries split
//     cleanly (previous below P, current from P with local_row = begin - P); the host
//     rule "pack when P % 128 != 0" is checked separately and never demanded of a
//     straddle expectation.
//
// Prints one JSON object; exit code 0 iff every check passed.
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "flash_sca/native/umma_tile_mask.cuh"

using namespace xattn::ops::flash_sca_native;
using namespace xattn::ops::flash_sca_native::umma;

namespace {

struct Counters {
  long long checks = 0, failures = 0, mutants_caught = 0, leading_dead_cases = 0;
  long long interior_dead_cases = 0;
  std::map<std::string, long long> by_name;
  std::map<std::string, long long> case_hits;  // how often each arm RAN, not failed
  std::map<std::string, int> shown;
  void hit(const char* what) { ++case_hits[what]; }
  void fail(const char* what, long long a, long long b, long long c, long long d) {
    ++failures;
    ++by_name[what];
    if (shown[what]++ < 2)  // two witnesses per distinct failure name
      std::printf("FAIL %s a=%lld b=%lld c=%lld d=%lld\n", what, a, b, c, d);
  }
};

// One batch row's bounds plus a Params carrying the matching 64-row table.
struct Rows {
  int64_t q_length = 0, total_length = 0;
  std::vector<int32_t> lo, hi;  // [q_length]
  int32_t at(int64_t i, bool high) const { return (high ? hi : lo)[i]; }
};

// TileBoundsKernel: [min live lo, max live hi, max lo over every row (dead = INT_MAX),
// min hi over every row (dead = -1)]; rows past q_length are dead.
std::vector<int32_t> BuildTable(const Rows& r) {
  const int64_t tiles = (r.q_length + detail::kTileRows - 1) / detail::kTileRows;
  std::vector<int32_t> t(static_cast<size_t>(tiles * 4));
  for (int64_t k = 0; k < tiles; ++k) {
    int mn = INT_MAX, mx = -1, mxlo = INT_MIN, mnhi = INT_MAX;
    for (int64_t x = 0; x < detail::kTileRows; ++x) {
      const int64_t i = k * detail::kTileRows + x;
      int lo = INT_MAX, hi = -1;
      if (i < r.q_length) {
        lo = r.at(i, false);
        hi = r.at(i, true);
        if (lo > hi) {
          lo = INT_MAX;
          hi = -1;
        }
      }
      if (lo <= hi) {
        mn = std::min(mn, lo);
        mx = std::max(mx, hi);
      }
      mxlo = std::max(mxlo, lo);
      mnhi = std::min(mnhi, hi);
    }
    int32_t* e = &t[static_cast<size_t>(k * 4)];
    e[0] = mn;
    e[1] = mx;
    e[2] = mxlo;
    e[3] = mnhi;
  }
  return t;
}

Params MakeParams(const Rows& r, std::vector<int32_t>& table) {
  Params p{};
  p.q_length = r.q_length;
  p.total_length = r.total_length;
  p.tile_count = (r.q_length + detail::kTileRows - 1) / detail::kTileRows;
  p.tile_bounds = table.data();
  p.row_lo = r.lo.data();
  p.row_hi = r.hi.data();
  return p;
}

bool MaySeeBrute(const Rows& r, int64_t first_row, int rows, int64_t kb, int64_t ke) {
  if (ke <= kb) return false;
  for (int x = 0; x < rows; ++x) {
    const int64_t i = first_row + x;
    if (i >= r.q_length) break;
    const int lo = r.at(i, false), hi = r.at(i, true);
    if (lo > hi) continue;
    if (lo <= ke - 1 && hi >= kb) return true;
  }
  return false;
}

bool WindowBrute(const Rows& r, int64_t first_row, int rows, int64_t kb, int64_t ke) {
  if (ke <= kb) return false;
  bool any = false;
  for (int x = 0; x < rows; ++x) {
    const int64_t i = first_row + x;
    if (i >= r.q_length) break;
    const int lo = r.at(i, false), hi = r.at(i, true);
    if (lo > hi) continue;
    any = true;
    if (!(kb >= lo && ke - 1 <= hi)) return false;
  }
  return any;
}

void CheckRow(const Rows& r, long long tag, Counters& c, int expect_monotone = 0) {
  std::vector<int32_t> table = BuildTable(r);
  Params p = MakeParams(r, table);
  const int64_t q_tiles64 = (r.q_length + 63) / 64;
  const int64_t q_tiles128 = (q_tiles64 + 1) / 2;
  const int64_t key_tiles64 = (r.total_length + 63) / 64;
  const int64_t key_tiles128 = (r.total_length + 127) / 128;
  c.hit("checkrow_calls");
  const bool monotone = RowBoundsMonotone(r.lo.data(), r.hi.data(), r.q_length, 0);
  if (expect_monotone) {
    ++c.checks;
    if (!monotone) c.fail("R5c-route-not-monotone", tag, r.q_length, 0, 0);
  } else if (!monotone) {
    ++c.interior_dead_cases;  // strict Runs puts a dead row at an interior
                              // segment start; legal metadata, no failure
  }

  for (int halves = 1; halves <= 2; ++halves) {
    const int64_t qt = halves == 1 ? q_tiles64 : q_tiles128;
    std::vector<QueryTileMask> from_table(static_cast<size_t>(qt));
    std::vector<QueryTileMask> from_rows(static_cast<size_t>(qt));
    for (int64_t t = 0; t < qt; ++t) {
      from_table[t] = FromTable(p, 0, t * halves, halves);
      from_rows[t] = ReduceFromRows(r.lo.data(), r.hi.data(), r.q_length, 0, t * halves * 64,
                                    halves * 64);
    }
    for (int64_t t = 0; t < qt; ++t) {
      const int64_t first_row = t * halves * 64;
      // A: FromTable against the documented reduction over the same halves
      int mn = INT_MAX, mx = -1, mxlo = INT_MIN, mnhi = INT_MAX;
      for (int h = 0; h < halves; ++h) {
        const int64_t tile = t * halves + h;
        if (tile >= p.tile_count) {
          mxlo = INT_MAX;
          mnhi = -1;
          continue;
        }
        for (int64_t x = 0; x < 64; ++x) {
          const int64_t i = tile * 64 + x;
          int lo = INT_MAX, hi = -1;
          if (i < r.q_length) {
            lo = r.at(i, false);
            hi = r.at(i, true);
            if (lo > hi) {
              lo = INT_MAX;
              hi = -1;
            }
          }
          if (lo <= hi) {
            mn = std::min(mn, lo);
            mx = std::max(mx, hi);
          }
          mxlo = std::max(mxlo, lo);
          mnhi = std::min(mnhi, hi);
        }
      }
      const bool dead_or_absent = mxlo > mnhi;
      ++c.checks;
      if (from_table[t].lo != (mn > mx ? INT_MAX : mn) || from_table[t].hi != (mn > mx ? -1 : mx))
        c.fail("A-fromtable-envelope", tag, t, from_table[t].lo, from_table[t].hi);
      ++c.checks;
      if (dead_or_absent) {
        if (!(from_table[t].flags & kHalfDead) || from_table[t].full_lo != INT_MAX ||
            from_table[t].full_hi != -1)
          c.fail("A-halfdead", tag, t, from_table[t].flags, from_table[t].full_lo);
      } else if (from_table[t].full_lo != mxlo || from_table[t].full_hi != mnhi) {
        c.fail("A-fullwindow", tag, t, from_table[t].full_lo, from_table[t].full_hi);
      }
      // B: the two producers agree on the envelope
      ++c.checks;
      if (from_table[t].lo != from_rows[t].lo || from_table[t].hi != from_rows[t].hi)
        c.fail("B-envelope-mismatch", tag, t, from_table[t].lo, from_rows[t].lo);
      // C: ReduceFromRows is the exact live hull (two checks: envelope, window)
      // rflo accumulates a MAX, so its identity is INT_MIN, not INT_MAX: with INT_MAX
      // every tile would "disagree" and the arm would be red on a correct header.
      int rmn = INT_MAX, rmx = -1, rflo = INT_MIN, rfhi = INT_MAX;
      bool rany = false;
      for (int x = 0; x < halves * 64; ++x) {
        const int64_t i = first_row + x;
        if (i >= r.q_length) break;
        const int lo = r.at(i, false), hi = r.at(i, true);
        if (lo > hi) continue;
        rany = true;
        rmn = std::min(rmn, lo);
        rmx = std::max(rmx, hi);
        rflo = std::max(rflo, lo);
        rfhi = std::min(rfhi, hi);
      }
      ++c.checks;
      if (rany) {
        if (from_rows[t].lo != rmn || from_rows[t].hi != rmx)
          c.fail("C-envelope", tag * 8 + halves, t, from_rows[t].lo, rmn);
        ++c.checks;
        if (from_rows[t].full_lo != rflo || from_rows[t].full_hi != rfhi) {
          c.fail("C-full", tag * 8 + halves, t, from_rows[t].full_lo, rflo);
          if (c.shown["C-full"] <= 3) {
            std::printf(
                "  C-full q_length=%lld total=%lld chunk=%lld halves=%d t=%lld rows=%d "
                "hdr=%d/%d/%d/%d brute=%d/%d/%d/%d\n",
                static_cast<long long>(r.q_length), static_cast<long long>(r.total_length),
                static_cast<long long>(p.chunk), halves, static_cast<long long>(t), halves * 64,
                from_rows[t].lo, from_rows[t].hi, from_rows[t].full_lo, from_rows[t].full_hi, rmn,
                rmx, rflo, rfhi);
            for (int64_t x = 0; x < halves * 64 && x < 6; ++x) {
              const int64_t i = first_row + x;
              if (i >= r.q_length) break;
              std::printf("    row %lld = %d/%d\n", static_cast<long long>(i), r.at(i, false),
                          r.at(i, true));
            }
          }
        }
      } else if (!(from_rows[t].flags & kNoLiveRow) || !IsEmpty(from_rows[t])) {
        c.fail("C-reduce-empty", tag, t, from_rows[t].flags, from_rows[t].lo);
      }
      // D/E: predicate direction, on both producers and both key granularities
      for (int which = 0; which < 2; ++which) {
        const QueryTileMask& m = which == 0 ? from_table[t] : from_rows[t];
        for (int ks = 0; ks < 2; ++ks) {
          const int tile_rows = ks == 0 ? 64 : 128;
          const int64_t kt = tile_rows == 64 ? key_tiles64 : key_tiles128;
          for (int64_t k = 0; k < kt; ++k) {
            const int64_t kb = k * tile_rows;
            const int64_t ke = std::min<int64_t>(kb + tile_rows, r.total_length);
            const bool see = MaySeeBrute(r, first_row, halves * 64, kb, ke);
            const bool full = WindowBrute(r, first_row, halves * 64, kb, ke);
            const bool ft = FullTile(m, kb, ke);
            c.hit("tile_predicates");
            if (see) c.hit("maysee_true");
            if (ft) c.hit("fulltile_true");
            ++c.checks;
            if (see && !MaySee(m, kb, ke)) c.fail("D-maysee-unsound", tag, t, k, which);
            ++c.checks;
            if (ft && !full) c.fail("E-full-not-whole", tag, t, k, which);
          }
        }
      }
    }
    // F/G: key-owned ranges
    for (int ks = 0; ks < 2; ++ks) {
      const int tile_rows = ks == 0 ? 64 : 128;
      const int64_t kt = tile_rows == 64 ? key_tiles64 : key_tiles128;
      for (int64_t k = 0; k < kt + 1; ++k) {
        const int64_t kb = k * tile_rows;
        const int64_t ke = std::min<int64_t>(kb + tile_rows, r.total_length);
        int f = 0, l = 0;
        QueryTilesForKeyScan(from_table.data(), qt, kb, ke, &f, &l);
        int bf = INT_MAX, bl = -1;
        for (int64_t t = 0; t < qt; ++t) {
          if (MaySee(from_table[t], kb, ke)) {
            if (bf == INT_MAX) bf = static_cast<int>(t);
            bl = static_cast<int>(t);
          }
        }
        ++c.checks;
        // both forms answer with (0,-1) when empty: compare emptiness, then endpoints
        const bool empty_scan = f > l;
        const bool empty_brute = bf > bl;
        c.hit("key_ranges");
        if (empty_scan) c.hit("scan_empty"); else c.hit("scan_nonempty");
        if (empty_scan != empty_brute || (!empty_scan && (f != bf || l != bl)))
          c.fail("F-scan", tag * 8 + ks, k, f, l);
        if (kb >= ke) {
          ++c.checks;
          if (l >= f) c.fail("G-empty-range", tag, k, f, l);
        }
        if (monotone) {
          int f2 = 0, l2 = 0;
          QueryTilesForKeyBisect(from_table.data(), qt, kb, ke, &f2, &l2);
          c.hit("bisect_cases");
          ++c.checks;
          if (f2 != f || l2 != l) c.fail("F-bisect", tag * 10 + ks, k, f2, l2);
        }
      }
    }
  }
  if (monotone) {
    for (int64_t t = 0; t < q_tiles128; ++t) {
      c.hit("lookup_vs_reduce");
      const QueryTileMask a = LookupFromRows(r.lo.data(), r.hi.data(), r.q_length, 0, t * 128, 128);
      const QueryTileMask d = ReduceFromRows(r.lo.data(), r.hi.data(), r.q_length, 0, t * 128, 128);
      ++c.checks;
      if (a.lo != d.lo || a.hi != d.hi || a.full_lo != d.full_lo || a.full_hi != d.full_hi)
        c.fail("F-lookup-vs-reduce", tag, t, a.lo, d.lo);
    }
  }
}

Rows Dense(int64_t q_length, int64_t total, int64_t chunk, bool strict) {
  Rows r;
  r.q_length = q_length;
  r.total_length = total;
  r.lo.resize(static_cast<size_t>(q_length));
  r.hi.resize(static_cast<size_t>(q_length));
  for (int64_t i = 0; i < q_length; ++i) {
    const int64_t pos = i + total - q_length;
    int64_t lo = (pos / chunk - 1) * chunk;
    lo = std::max<int64_t>(lo, 0);
    const int64_t hi = pos - (strict ? 1 : 0);
    if (lo > hi) {
      r.lo[i] = INT_MAX;
      r.hi[i] = -1;
    } else {
      r.lo[i] = static_cast<int32_t>(lo);
      r.hi[i] = static_cast<int32_t>(hi);
    }
  }
  return r;
}

// Segment runs on the global key axis, run-clipped bounds (the BOS route).
Rows Runs(int64_t q_length, int64_t total, int64_t chunk, std::mt19937& g, bool strict,
          int64_t leading_dead = 0) {
  std::uniform_int_distribution<int> gap(1, 40);
  std::vector<int64_t> start{0};
  while (start.back() < total) {
    const int64_t s = start.back() + gap(g);
    if (s >= total) break;
    start.push_back(s);
  }
  start.push_back(total);
  auto seg_of = [&](int64_t j) {
    int64_t s = 0;
    while (s + 1 < static_cast<int64_t>(start.size()) && start[s + 1] <= j) ++s;
    return s;
  };
  Rows r;
  r.q_length = q_length;
  r.total_length = total;
  r.lo.resize(static_cast<size_t>(q_length));
  r.hi.resize(static_cast<size_t>(q_length));
  for (int64_t i = 0; i < q_length; ++i) {
    if (i < leading_dead) {
      r.lo[i] = INT_MAX;
      r.hi[i] = -1;
      continue;
    }
    const int64_t pos = i + total - q_length;
    const int64_t sg = seg_of(pos);
    const int64_t kstart = start[sg], kend = start[sg + 1] - 1;
    int64_t lo = (pos / chunk - 1) * chunk;
    lo = std::max<int64_t>(lo, 0);
    int64_t hi = pos - (strict ? 1 : 0);
    lo = std::max(lo, kstart);
    hi = std::min(hi, kend);
    if (lo > hi) {
      r.lo[i] = INT_MAX;
      r.hi[i] = -1;
    } else {
      r.lo[i] = static_cast<int32_t>(lo);
      r.hi[i] = static_cast<int32_t>(hi);
    }
  }
  return r;
}

// Arbitrary intervals: the envelope-gap class that MaySee must survive.
Rows Random(int64_t q_length, int64_t total, std::mt19937& g, int dead_pct) {
  Rows r;
  r.q_length = q_length;
  r.total_length = total;
  r.lo.resize(static_cast<size_t>(q_length));
  r.hi.resize(static_cast<size_t>(q_length));
  std::uniform_int_distribution<int> pos(0, static_cast<int>(2 * total));
  std::uniform_int_distribution<int> pick(0, 99);
  for (int64_t i = 0; i < q_length; ++i) {
    if (pick(g) < dead_pct) {
      r.lo[i] = INT_MAX;
      r.hi[i] = -1;
      continue;
    }
    const int a = pos(g), b = pos(g);
    r.lo[i] = std::min(a, b);
    r.hi[i] = std::max(a, b);
  }
  return r;
}

void CheckOddTail(Counters& c) {
  // I: a half past tile_count is poison, the same convention ReadTileRange uses.
  for (int64_t q_length : {1, 2, 63, 64, 65, 127, 129, 191, 257, 517}) {
    c.hit("odd_tail_cases");
    const Rows r = Dense(q_length, q_length + 5, 8, false);
    std::vector<int32_t> table = BuildTable(r);
    Params p = MakeParams(r, table);
    std::vector<int32_t> padded2 = table;
    for (int k = 0; k < 4; ++k) {
      // poison entry: max lo = INT_MAX, min hi = -1, live pair absent
      padded2.push_back(INT_MAX);
      padded2.push_back(-1);
      padded2.push_back(INT_MAX);
      padded2.push_back(-1);
    }
    Params pp = p;
    pp.tile_bounds = padded2.data();
    // pp.tile_count must cover the padded rectangle, or FromTable takes the
    // `tile >= tile_count` branch for the second half and never reads padded2 at
    // index `tiles`: b would be the same computation as a and the arm
    // vacuous.
    pp.tile_count = p.tile_count + 4;
    const int64_t tiles = p.tile_count;
    const QueryTileMask a = FromTable(p, 0, tiles - 1, 2);
    const QueryTileMask b = FromTable(pp, 0, tiles - 1, 2);
    ++c.checks;
    if (a.lo != b.lo || a.hi != b.hi || a.full_lo != b.full_lo || a.full_hi != b.full_hi ||
        (a.flags & (kNoLiveRow | kHalfDead)) != (b.flags & (kNoLiveRow | kHalfDead)))
      c.fail("I-odd-tail-poison", q_length, tiles, a.full_lo, b.full_lo);
    ++c.checks;
    if (q_length % 64 != 0 && !(b.flags & kHalfDead)) c.fail("I-tail-collapse", q_length, tiles, b.flags, 0);
  }
}

// H: a mutant bound sequence must be rejected by the route predicate AND must make
// the endpoint lookup differ from the exact reduction (a false "monotone" would
// silently under-visit). The witness is a flat live row set, so the mutated row is
// the only source of either property.
void CheckMutants(Counters& c) {
  for (int64_t q_length : {128, 256, 517}) {
    for (int mut = 0; mut < 2; ++mut) {
      Rows r;
      r.q_length = q_length;
      r.total_length = q_length;
      r.lo.assign(static_cast<size_t>(q_length), 100);
      r.hi.assign(static_cast<size_t>(q_length), 200);
      const int64_t i0 = q_length / 2 + 17;
      if (mut == 0) {
        r.lo[i0] = 50;  // widened lo in the middle of a tile
      } else {
        r.hi[i0] = 150;  // narrowed hi in the middle of a tile
      }
      int caught = 0;
      c.hit("mutant_cases");
      if (!RowBoundsMonotone(r.lo.data(), r.hi.data(), r.q_length, 0)) {
        ++caught;
      } else {
        c.fail("H-mutant-accepted", q_length, mut, 0, 0);
      }
      int differ_tiles = 0;
      for (int64_t t = 0; t * 128 < q_length; ++t) {
        const QueryTileMask red =
            ReduceFromRows(r.lo.data(), r.hi.data(), q_length, 0, t * 128, 128);
        const QueryTileMask loo =
            LookupFromRows(r.lo.data(), r.hi.data(), q_length, 0, t * 128, 128);
        if (red.lo != loo.lo || red.hi != loo.hi || red.full_lo != loo.full_lo ||
            red.full_hi != loo.full_hi)
          ++differ_tiles;
      }
      ++c.checks;
      if (differ_tiles > 0) {
        ++caught;
      } else {
        c.fail("H-mutant-not-discriminated", q_length, mut, caught, differ_tiles);
      }
      ++c.checks;
      c.mutants_caught += caught == 2 ? 1 : 0;
      if (caught != 2) c.fail("H-mutant-not-caught", q_length, mut, caught, 0);
    }
  }
}

void CheckHeaderMisc(Counters& c) {
  // R5e: the host-callable parts of the interface, each at least once. The
  // segment-test path (NeedsSegmentTest / AnnotateSegmentTest / FullTileExact) is
  // __device__-only by design -- an inclusion in a native TU compiles it for
  // sm_103a, and a device-side driver exercises it there.
  QueryTileMask m;
  c.hit("misc_R5e");
  ++c.checks;
  if (!IsEmpty(m) || (m.flags & kNoLiveRow) == 0) c.fail("R5e-default-mask", 0, 0, 0, 0);
  ++c.checks;
  if (MaySee(m, 0, 64) || FullTile(m, 0, 64)) c.fail("R5e-empty-predicates", 0, 0, 0, 0);
  m.lo = 0;
  m.hi = 100;
  m.full_lo = 10;
  m.full_hi = 20;
  m.flags = 0;
  ++c.checks;
  if (IsEmpty(m) || !MaySee(m, 20, 40) || MaySee(m, 101, 200)) c.fail("R5e-envelope", 0, 0, 0, 0);
  ++c.checks;
  // full_hi is the last visible key index (inclusive), so keys 12..20 fill the
  // window [10,20] whole while 12..21 does not
  if (!FullTile(m, 12, 21) || FullTile(m, 9, 21) || FullTile(m, 12, 22))
    c.fail("R5e-full-window", 0, 0, 0, 0);
  ++c.checks;
  if (RowRouteAt(nullptr, 3) != kRouteReduce) c.fail("R5e-route-null", 0, 0, 0, 0);
  const int32_t route[2] = {1, 0};
  ++c.checks;
  if (RowRouteAt(route, 0) != kRouteLookup || RowRouteAt(route, 1) != kRouteReduce)
    c.fail("R5e-route-array", 0, 0, 0, 0);
  QueryTileMask a;
  AnnotateRoute(a, kRouteLookup);
  ++c.checks;
  if ((a.flags & kFlagLookupRoute) == 0) c.fail("R5e-annotate-route", 0, 0, 0, 0);
  // an empty key range is empty in both forms (R7)
  QueryTileMask one;
  one.lo = 0;
  one.hi = 200;
  one.full_lo = 0;
  one.full_hi = 200;
  one.flags = 0;
  int f = 0, l = 0;
  QueryTilesForKeyBisect(&one, 1, 100, 100, &f, &l);
  ++c.checks;
  if (l >= f) c.fail("R7-bisect-empty-range", f, l, 0, 0);
  QueryTilesForKeyScan(&one, 1, 100, 100, &f, &l);
  ++c.checks;
  if (l >= f) c.fail("R7-scan-empty-range", f, l, 0, 0);
}

void CheckPlacement(Counters& c) {
  const int64_t prevs[] = {0, 64, 128, 192, 256, 4096};
  const int64_t lengths[] = {1, 64, 128, 256, 4096, 16384};
  for (int64_t P : prevs) {
    c.hit("placement_P_values");
    for (int64_t N : lengths) {
      const int64_t total = P + N;
      for (int tile_rows : {64, 128}) {
        const int64_t tiles = (total + tile_rows - 1) / tile_rows;
        // unpacked: legal only when P % tile_rows == 0 (the host packs otherwise)
        const bool aligned = P % tile_rows == 0;
        for (int64_t t = 0; t < tiles; ++t) {
          const KeyTilePlacement pl = PlaceKeyTile(P, total, false, tile_rows, t);
          const int64_t begin = t * tile_rows;
          ++c.checks;
          if (aligned) {
            if (pl.straddles) c.fail("J-unexpected-straddle", P, t, tile_rows, 0);
            if (P > 0 && begin + tile_rows <= P) {
              if (pl.map != KeyTilePlacement::kPrevious || pl.local_row != begin)
                c.fail("J-prev-map", P, t, tile_rows, pl.local_row);
              // previous map over rows [0,P): identity local row
            } else if (begin >= P) {
              if (pl.map != KeyTilePlacement::kCurrentOrPacked || pl.local_row != begin - P)
                c.fail("J-current-map", P, t, tile_rows, pl.local_row);
            }
            if (P == 0 && (pl.map != KeyTilePlacement::kCurrentOrPacked || pl.local_row != begin))
              c.fail("J-no-prev-map", P, t, tile_rows, pl.local_row);
          } else {
            const bool crosses = begin < P && begin + tile_rows > P;
            if (pl.straddles != crosses)
              c.fail("J-straddle-tripwire", P, t, tile_rows, crosses);
          }
        }
        // packed: one map, identity rows, never straddles
        for (int64_t t = 0; t < tiles; ++t) {
          const KeyTilePlacement pl = PlaceKeyTile(P, total, true, tile_rows, t);
          ++c.checks;
          if (pl.straddles || pl.map != KeyTilePlacement::kCurrentOrPacked ||
              pl.local_row != t * tile_rows)
            c.fail("J-packed", P, t, tile_rows, pl.local_row);
        }
      }
    }
  }
}

}  // namespace

int main() {
  Counters c;
  std::mt19937 g(20261102u);
  for (int rep = 0; rep < 40; ++rep) {
    const int64_t total = 1 + (g() % 600);
    const int64_t q_length = 1 + (g() % total);
    const int64_t chunk = 1 + (g() % 128);
    Rows r = Dense(q_length, total, chunk, rep % 3 == 0);
    CheckRow(r, rep, c, 1);
    // strict Runs has a dead row at every interior segment start (legal, and not
    // monotone), so monotonicity is asserted only for the non-strict ones
    r = Runs(q_length, total, chunk, g, rep % 3 == 0);
    CheckRow(r, 100 + rep, c, rep % 3 == 0 ? 0 : 1);
    // leading dead run: the lookup route with empty tiles in front (G),
    // deterministic lengths so a fully dead 128- and 256-row tile is in every
    // run
    const int64_t lead = 64 << (rep % 3);
    if (lead < q_length) {
      Rows s = Runs(q_length, total, chunk, g, rep % 3 == 0, lead);
      ++c.leading_dead_cases;
      CheckRow(s, 200 + rep, c, rep % 3 == 0 ? 0 : 1);
    }
    // odd tail: q_length % 128 != 0 and % 64 != 0 (I)
    Rows t = Runs(q_length / 2 + 3, total, chunk, g, false);
    CheckRow(t, 300 + rep, c, 1);
  }
  for (int rep = 0; rep < 120; ++rep) {
    const int64_t total = 1 + (g() % 400);
    const int64_t q_length = 1 + (g() % total);
    CheckRow(Random(q_length, total, g, rep % 40), 1000 + rep, c);
  }
  CheckOddTail(c);
  CheckMutants(c);
  CheckHeaderMisc(c);
  CheckPlacement(c);
  std::printf("\"by_name\": {");
  bool first = true;
  for (const auto& kv : c.by_name) {
    std::printf("%s\"%s\": %lld", first ? "" : ", ", kv.first.c_str(), kv.second);
    first = false;
  }
  std::printf("}, ");
  // case_hits: how often each arm RAN. by_name only counts failures, so on a green
  // run it is empty and a reader cannot tell an absent arm from one that agreed.
  std::printf("\"case_hits\": {");
  first = true;
  for (const auto& kv : c.case_hits) {
    std::printf("%s\"%s\": %lld", first ? "" : ", ", kv.first.c_str(), kv.second);
    first = false;
  }
  std::printf("}, ");
  std::printf(
      "{\"checks\": %lld, \"failures\": %lld, \"mutants_caught\": %lld, "
      "\"leading_dead_cases\": %lld, \"interior_dead_cases\": %lld, \"passed\": %s}\n",
      c.checks, c.failures, c.mutants_caught, c.leading_dead_cases, c.interior_dead_cases,
      c.failures == 0 ? "true" : "false");
  return c.failures == 0 ? 0 : 1;
}
