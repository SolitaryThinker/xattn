// H29: the (128,128) grouped key leaf in 128-query rounds (CUTLASS example 77_2bwd's dKdV round
// structure, 1 CTA, M = 128 keys), beside BackwardKvPipeKernel (umma_bwd_kv_pipe.cuh), and run
// where the companion launch runs (LaunchUmmaBackwardKvPipeCompanionT, the SoftDelta pair route).
//
// What changes against KvPipe:
// - A round is 128 queries of one (tile, head): KvPipe's two 64-query halves (t, gh, h = 0, 1)
//   as one round. The score MMAs are SS M128 N128 (full rate) instead of N64.
// - TMEM is single-buffered: dK 0..127 | dV 128..255 | S^T(r), then P^T(r) packed over it,
//   256..383 | dP^T(r), then dS^T(r) packed over it, 384..511.
// - One MMA issuer, in order: S(0), dP(0); per round r: [p_full(r)] dV(r); S(r+1);
//   [ds_full(r)] dK(r); dP(r+1). The write of S(r+1) over P^T(r) (and of dP(r+1) over dS^T(r)) is
//   ordered behind dV(r) (dK(r)) by the one issuer's in-order tcgen05 pipe.
// - The pack is two stages per round on the same eight warps: stage P (wait s_full: S^T -> P,
//   P^T packed, arrive p_full; P kept in registers) and stage dS (wait dp_full: dS = P (dP -
//   delta), dS^T packed, arrive ds_full). Stage P of round r+1 then overlaps dK(r) and dP(r+1).
//
// Bits: each dK/dV element receives the same K = 16 query steps (queries ascending inside a tile,
// heads ascending, tiles by NextTile) with the same operands as KvPipe's halves, and P / dS are
// the same per-element expressions (PackBlockFull / PackBlockMasked split in two), so dK and dV
// are KvPipe's bits. The launch is one CTA per 128-key tile (no persistent loop yet).
#pragma once

#include "flash_sca/native/umma_bwd_kv_pipe.cuh"

#ifndef XATTN_KV_R128
#define XATTN_KV_R128 1
#endif

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace kvr128 {

using kvpipe::kKeyRows;
using kvpipe::kPackThreads;
using kvpipe::kPackWarps;
using kvpipe::kProducerWarp;
using kvpipe::kMmaWarp;
inline constexpr int kRoundQueries = 128;         // one round = one 128-query tile of one head
inline constexpr int kSteps = kRoundQueries / 16;  // 16-query K steps of dV / dK
inline constexpr int kStages = 2;                  // Q/dO SMEM stages (64 KB each)
inline constexpr int kThreads = 32 * kvpipe::kBaseWarps;  // 352 (warp 10 idles)
static_assert(kRoundQueries == kBwdQueryRows, "one round per query tile and head");

struct Bars {
  static constexpr int kKv = 0;        // K and V landed
  static constexpr int kLd = 1;        // + st: Q(r), dO(r) landed in stage st
  static constexpr int kSFull = 3;     // S^T(r) complete
  static constexpr int kPFull = 4;     // P^T(r) packed (256 arrivals)
  static constexpr int kDpFull = 5;    // dP^T(r) complete
  static constexpr int kDsFull = 6;    // dS^T(r) packed (256 arrivals)
  static constexpr int kAcc = 7;       // + st: dK(r) (and everything before it) complete
  static constexpr int kCount = 9;
};

struct Plan {
  static constexpr int kSlabBytes = 128 * 128;              // 128 rows x 64 dims x 2 B
  static constexpr int kQBytes = 2 * kSlabBytes;            // 32 KB
  static constexpr int kGBytes = 2 * kSlabBytes;            // 32 KB
  static constexpr int kStageBytes = kQBytes + kGBytes;     // 64 KB
  static constexpr int kKOff = kStages * kStageBytes;
  static constexpr int kKBytes = 2 * kSlabBytes;
  static constexpr int kVOff = kKOff + kKBytes;
  static constexpr int kVBytes = 2 * kSlabBytes;
  static constexpr int kBarOff = kVOff + kVBytes;
  static constexpr int kSlotOff = kBarOff + 256;
  static constexpr int kRedOff = kSlotOff + 16;
  static constexpr int kMetaOff = kRedOff + 32;
  // Two round buffers of 128 columns = KvPipe's 4-slot table of 64 columns (slot 2 buf + half).
  using Table = kvpipe::ColumnTable<4>;
  static constexpr int kEnd = kMetaOff + Table::kBytes;
  static constexpr int kSmemBytes = ((kEnd + 1023) / 1024) * 1024 + 1024;
  static constexpr int kdKCol = 0;
  static constexpr int kdVCol = 128;
  static constexpr int kSCol = 256;
  static constexpr int kdPCol = 384;
  static constexpr int kAllocCols = 512;
  static_assert(kSmemBytes <= 232448, "the sm_103a opt-in dynamic SMEM ceiling");
  static_assert(kMetaOff % 16 == 0, "the column table is read 16 bytes at a time");
};

// Packed operand column of K step kl (16 queries) in a window: block kl >> 1 (32 queries) packs
// its 16 words at the start of its own 32 columns (KvPipe's PackedCol at buffer 0).
__host__ __device__ constexpr uint32_t PackedCol(int window, int kl) {
  return static_cast<uint32_t>(window + 32 * (kl >> 1) + 8 * (kl & 1));
}

// Round sequence: (tile, head), heads (and the companion, kComp) ascending inside a tile.
template <bool kComp>
__device__ __forceinline__ void NextR(const Params& p, int64_t b, int64_t kb, int64_t ke,
                                      int64_t last, dkpipe::Round* c) {
  c->h = 0;
  if (++c->gh < p.group + (kComp ? 1 : 0)) return;
  c->gh = 0;
  c->t = dvpipe::NextTile(p, b, kb, ke, c->t + 1, last, &c->m);
}

// Stage P of PackBlockFull / PackBlockMasked: the same P expressions, P kept for stage dS.
template <class E, bool kMasked, bool kSegs>
__device__ __forceinline__ void PStage(const float (&s)[pt::kBlock], const float* l2,
                                       const int32_t* lo, const int32_t* hi, const int64_t* sg,
                                       int32_t j32, int64_t key_segment, float scale_log2,
                                       float (&pe)[pt::kBlock], uint32_t& vm, uint32_t* out_p) {
  vm = 0xffffffffu;
  const float2 sc = make_float2(scale_log2, scale_log2);
#pragma unroll
  for (int q = 0; q < pt::kBlock / 4; ++q) {
    const float4 L = *reinterpret_cast<const float4*>(l2 + 4 * q);
    const float lv[4] = {L.x, L.y, L.z, L.w};
    bool vis[4] = {true, true, true, true};
    if constexpr (kMasked) {
      const int4 LO = *reinterpret_cast<const int4*>(lo + 4 * q);
      const int4 HI = *reinterpret_cast<const int4*>(hi + 4 * q);
      const int32_t lov[4] = {LO.x, LO.y, LO.z, LO.w};
      const int32_t hiv[4] = {HI.x, HI.y, HI.z, HI.w};
      int64_t sv[4] = {0, 0, 0, 0};
      if constexpr (kSegs) {
        const longlong2 A = *reinterpret_cast<const longlong2*>(sg + 4 * q);
        const longlong2 B = *reinterpret_cast<const longlong2*>(sg + 4 * q + 2);
        sv[0] = A.x;
        sv[1] = A.y;
        sv[2] = B.x;
        sv[3] = B.y;
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        vis[e] = (j32 >= lov[e]) && (j32 <= hiv[e]);
        if constexpr (kSegs) vis[e] = vis[e] && (key_segment == sv[e]);
        if (!vis[e]) vm &= ~(1u << (4 * q + e));
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 x = pt::FmaX2(make_float2(s[4 * q + 2 * h], s[4 * q + 2 * h + 1]), sc,
                                 make_float2(-lv[2 * h], -lv[2 * h + 1]));
      if constexpr (kMasked) {
        pe[4 * q + 2 * h] = vis[2 * h] ? kvpipe::KvExp2(x.x) : 0.f;
        pe[4 * q + 2 * h + 1] = vis[2 * h + 1] ? kvpipe::KvExp2(x.y) : 0.f;
      } else {
        pe[4 * q + 2 * h] = kvpipe::KvExp2(x.x);
        pe[4 * q + 2 * h + 1] = kvpipe::KvExp2(x.y);
      }
    }
    out_p[2 * q] = kvpipe::PackPair<E>(pe[4 * q], pe[4 * q + 1]);
    out_p[2 * q + 1] = kvpipe::PackPair<E>(pe[4 * q + 2], pe[4 * q + 3]);
  }
}

// Stage dS: dS = P (dP - delta), zero where the key is not visible (the masked form's select).
template <class E, bool kMasked>
__device__ __forceinline__ void DsStage(const float (&pe)[pt::kBlock],
                                        const float (&dp)[pt::kBlock], const float* dl,
                                        uint32_t vm, uint32_t* out_ds) {
#pragma unroll
  for (int q = 0; q < pt::kBlock / 4; ++q) {
    const float4 D = *reinterpret_cast<const float4*>(dl + 4 * q);
    const float dv[4] = {D.x, D.y, D.z, D.w};
    float ge[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 gv = pt::MulX2(make_float2(pe[4 * q + 2 * h], pe[4 * q + 2 * h + 1]),
                                  pt::SubX2(make_float2(dp[4 * q + 2 * h], dp[4 * q + 2 * h + 1]),
                                            make_float2(dv[2 * h], dv[2 * h + 1])));
      ge[2 * h] = gv.x;
      ge[2 * h + 1] = gv.y;
    }
    if constexpr (kMasked) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        if (!((vm >> (4 * q + e)) & 1u)) ge[e] = 0.f;
      }
    }
    out_ds[2 * q] = kvpipe::PackPair<E>(ge[0], ge[1]);
    out_ds[2 * q + 1] = kvpipe::PackPair<E>(ge[2], ge[3]);
  }
}

template <typename E, bool kComp>
__global__ void __launch_bounds__(kThreads, 1) BackwardKvR128Kernel(
    const __grid_constant__ KvPipeLaunch L) {
  const UmmaLaunchParams& lp = L.lp;
  using Pl = Plan;
  using Br = Bars;
  using ColumnTable = Pl::Table;
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const smem = smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);

  unsigned char* const sk = smem + Pl::kKOff;
  unsigned char* const sv = smem + Pl::kVOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Pl::kRedOff);
  const uint32_t bar_kv = bar0 + 8u * Br::kKv;
  auto bar = [&](int i) { return bar0 + 8u * static_cast<uint32_t>(i); };

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hk = blockIdx.y;
  const int64_t kt = blockIdx.x;
  const int64_t kb = kt * kKeyRows;
  const int64_t ke = kb + kKeyRows;
  const int64_t nt_total = p.total_length;

  if (tid == 0) {
    for (int i = 0; i < Br::kCount; ++i) {
      const int count = (i == Br::kPFull || i == Br::kDsFull) ? kPackThreads : 1;
      MbarInit(bar(i), count);
    }
  }
  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) MbarInitFence();

  const bool seg = detail::SegmentTest(p, b);
  if (tid == 0) {
    const CUtensorMap* kmap;
    const CUtensorMap* vmap;
    int row0;
    ResolveKeyMap(lp, kb, &kmap, &vmap, &row0);
    MbarArriveExpectTx(bar_kv, Pl::kKBytes + Pl::kVBytes);
    for (int s = 0; s < 2; ++s) {
      for (int r = 0; r < kKeyRows; r += kTmaRows) {
        TmaLoad4D(sk + s * Pl::kSlabBytes + r * 128, kmap, bar_kv, s * 64,
                  static_cast<int>(hk), row0 + r, static_cast<int>(b));
      }
    }
    for (int s = 0; s < 2; ++s) {
      for (int r = 0; r < kKeyRows; r += kTmaRows) {
        TmaLoad4D(sv + s * Pl::kSlabBytes + r * 128, vmap, bar_kv, s * 64,
                  static_cast<int>(hk), row0 + r, static_cast<int>(b));
      }
    }
  }

  int64_t qt_first, qt_last;
  if (tid == 0) {
    QueryTileEnvelope<kKeyRows>(p, b, kt, kb, ke - 1, &qt_first, &qt_last);
    red[0] = static_cast<int32_t>(qt_first);
    red[1] = static_cast<int32_t>(qt_last);
  }
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  qt_first = red[0];
  qt_last = red[1];

  const uint32_t tbase = *tmem_slot & 0x1FFu;
  const int elem_fmt = BwdTok<E>::kFormat;

  if (warp == kProducerWarp) {
    // ---------------------------------------------------------------- TMA producer
    if (lane == 0) {
      dkpipe::Round c;
      dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
      uint32_t r = 0;
      while (c.t <= qt_last) {
        const uint32_t st = r % kStages;
        if (r >= static_cast<uint32_t>(kStages)) {
          MbarWait(bar(Br::kAcc + st), ((r - kStages) / kStages) & 1u);
        }
        unsigned char* const qs = smem + st * Pl::kStageBytes;
        unsigned char* const gs = qs + Pl::kQBytes;
        const uint32_t bl = bar(Br::kLd + st);
        const bool cr = kComp && c.gh == p.group;
        const CUtensorMap* const qmap = cr ? &L.comp.q : &lp.maps.q;
        const CUtensorMap* const gmap = cr ? &L.comp.dy : &lp.maps.dy;
        const int hq = cr ? static_cast<int>(hk) : static_cast<int>(hk * p.group + c.gh);
        const int i0 = static_cast<int>(c.t * kBwdQueryRows);
        MbarArriveExpectTx(bl, Pl::kStageBytes);
        for (int s = 0; s < 2; ++s) {
          for (int rr = 0; rr < kRoundQueries; rr += kTmaRows) {
            TmaLoad4D(qs + s * Pl::kSlabBytes + rr * 128, qmap, bl, s * 64, hq, i0 + rr,
                      static_cast<int>(b));
          }
        }
        for (int s = 0; s < 2; ++s) {
          for (int rr = 0; rr < kRoundQueries; rr += kTmaRows) {
            TmaLoad4D(gs + s * Pl::kSlabBytes + rr * 128, gmap, bl, s * 64, hq, i0 + rr,
                      static_cast<int>(b));
          }
        }
        ++r;
        NextR<kComp>(p, b, kb, ke, qt_last, &c);
      }
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    // ---------------------------------------------------------------- MMA issuer
    MbarWait(bar_kv, 0);
    dkpipe::Round c;
    dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    // The whole warp runs the issue code, as in KvPipe (the MMA helpers elect one lane).
    if (c.t <= qt_last) {
      const uint32_t s_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kRoundQueries, 0, 0);
      const uint32_t v_idesc = MakeInstrDesc(elem_fmt, kKeyRows, 128, 0, 1);
      const uint32_t k_idesc = MakeInstrDesc(elem_fmt, kKeyRows, 128, 0, 1);
      constexpr uint32_t kSlabLbo = static_cast<uint32_t>(Pl::kSlabBytes / 16);
      static_assert(kSlabLbo < (1u << 14), "LBO field");
      const uint32_t s_base = SmemAddr(smem);
      auto issue_s = [&](uint32_t r) {
        const uint32_t qs = s_base + (r % kStages) * Pl::kStageBytes;
        for (int kk = 0; kk < 128 / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(tbase + Pl::kSCol, MakeSmemDesc(SmemAddr(sk) + slab * Pl::kSlabBytes + within * 32),
                MakeSmemDesc(qs + slab * Pl::kSlabBytes + within * 32), s_idesc, kk == 0 ? 0u : 1u);
        }
      };
      auto issue_dp = [&](uint32_t r) {
        const uint32_t gs = s_base + (r % kStages) * Pl::kStageBytes + Pl::kQBytes;
        for (int kk = 0; kk < 128 / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(tbase + Pl::kdPCol, MakeSmemDesc(SmemAddr(sv) + slab * Pl::kSlabBytes + within * 32),
                MakeSmemDesc(gs + slab * Pl::kSlabBytes + within * 32), s_idesc, kk == 0 ? 0u : 1u);
        }
      };
      auto issue_dv = [&](uint32_t r) {
        const uint32_t gs = s_base + (r % kStages) * Pl::kStageBytes + Pl::kQBytes;
        for (int kl = 0; kl < kSteps; ++kl) {
          MmaTs(tbase + Pl::kdVCol, tbase + PackedCol(Pl::kSCol, kl),
                MakeSmemDesc(gs + kl * 16 * 128, kSlabLbo, 64), v_idesc,
                (r == 0 && kl == 0) ? 0u : 1u);
        }
      };
      auto issue_dk = [&](uint32_t r) {
        const uint32_t qs = s_base + (r % kStages) * Pl::kStageBytes;
        for (int kl = 0; kl < kSteps; ++kl) {
          MmaTs(tbase + Pl::kdKCol, tbase + PackedCol(Pl::kdPCol, kl),
                MakeSmemDesc(qs + kl * 16 * 128, kSlabLbo, 64), k_idesc,
                (r == 0 && kl == 0) ? 0u : 1u);
        }
      };
      MbarWait(bar(Br::kLd + 0), 0u);
      TmemFenceAfterSync();
      __syncwarp();
      issue_s(0u);
      MmaCommit(bar(Br::kSFull));
      issue_dp(0u);
      MmaCommit(bar(Br::kDpFull));
      for (uint32_t r = 0;; ++r) {
        dkpipe::Round n = c;
        NextR<kComp>(p, b, kb, ke, qt_last, &n);
        const bool more = n.t <= qt_last;
        const uint32_t r1 = r + 1;
        MbarWait(bar(Br::kPFull), r & 1u);
        TmemFenceAfterSync();
        __syncwarp();
        issue_dv(r);
        if (more) {
          MbarWait(bar(Br::kLd + r1 % kStages), (r1 / kStages) & 1u);
          TmemFenceAfterSync();
          __syncwarp();
          issue_s(r1);  // over P^T(r): ordered behind dV(r) by this one issuer
          MmaCommit(bar(Br::kSFull));
        }
        MbarWait(bar(Br::kDsFull), r & 1u);
        TmemFenceAfterSync();
        __syncwarp();
        issue_dk(r);
        MmaCommit(bar(Br::kAcc + r % kStages));
        if (!more) break;
        issue_dp(r1);  // over dS^T(r): ordered behind dK(r)
        MmaCommit(bar(Br::kDpFull));
        c = n;
      }
    }
    __syncwarp();
  } else if (warp < kPackWarps) {
    // ---------------------------------------------------------------- pack warps 0..7
    // Warp w: TMEM lane quarter w & 3 (key rows 32 (w & 3) ..), query columns 64 (w >> 2) ..
    // as two 32-column blocks.
    const int w4 = warp & 3;
    const int ch = warp >> 2;
    const uint32_t lane_base = static_cast<uint32_t>(32 * w4) << 16;
    const int64_t j = kb + w4 * 32 + lane;
    const int64_t ks = (seg && j < nt_total) ? p.k_segment[b * nt_total + j] : 0;
    const int32_t j32 = static_cast<int32_t>(j);
    unsigned char* const tb = smem + Pl::kMetaOff;
    dkpipe::Round c;
    dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    pt::ColumnMeta cm;
    if (c.t <= qt_last && tid < kRoundQueries) {
      cm = kvpipe::RoundColumnOf<kComp>(L, b, hk, c, c.t * kBwdQueryRows + tid, seg);
    }
    uint32_t r = 0;
    while (c.t <= qt_last) {
      dkpipe::Round n = c;
      NextR<kComp>(p, b, kb, ke, qt_last, &n);
      const bool full = !seg && FullTile(c.m, kb, ke) && !(kComp && c.gh == p.group);
      const uint32_t buf = r & 1u;
      // Buffer buf's last readers (round r - 2's stages) finished before the barrier of round
      // r - 1, which this writer passed.
      if (tid < kRoundQueries) ColumnTable::Store(tb, 2u * buf, tid, cm);
      dvpipe::PackBarrierSync();
      if (n.t <= qt_last && tid < kRoundQueries) {
        cm = kvpipe::RoundColumnOf<kComp>(L, b, hk, n, n.t * kBwdQueryRows + tid, seg);
      }
      const int obase = static_cast<int>(2u * buf) * kvpipe::kRoundQueries + 64 * ch;
      // Stage P.
      float pe[2][pt::kBlock];
      uint32_t vm[2];
      MbarWait(bar(Br::kSFull), r & 1u);
      TmemFenceAfterSync();
#pragma unroll
      for (int blk = 0; blk < 2; ++blk) {
        const int c0 = 64 * ch + 32 * blk;
        const uint32_t scol = tbase + Pl::kSCol + c0;
        float s[pt::kBlock];
        TmemLoad32(lane_base | scol, s);
        TmemWaitLd();
        const int o = obase + 32 * blk;
        const float* const l2 = reinterpret_cast<const float*>(tb + ColumnTable::kL2) + o;
        const int32_t* const lo = reinterpret_cast<const int32_t*>(tb + ColumnTable::kLo) + o;
        const int32_t* const hi = reinterpret_cast<const int32_t*>(tb + ColumnTable::kHi) + o;
        const int64_t* const sg = reinterpret_cast<const int64_t*>(tb + ColumnTable::kSg) + o;
        uint32_t pv[pt::kBlock / 2];
        if (full) {
          PStage<E, false, false>(s, l2, lo, hi, sg, j32, ks, p.scale_log2, pe[blk], vm[blk], pv);
        } else if (seg) {
          PStage<E, true, true>(s, l2, lo, hi, sg, j32, ks, p.scale_log2, pe[blk], vm[blk], pv);
        } else {
          PStage<E, true, false>(s, l2, lo, hi, sg, j32, ks, p.scale_log2, pe[blk], vm[blk], pv);
        }
        dkpipe::TmemStore16(lane_base | scol, pv);
      }
      TmemWaitSt();
      TmemFenceBeforeSync();
      dvpipe::MbarArrive(bar(Br::kPFull));
      // Stage dS.
      MbarWait(bar(Br::kDpFull), r & 1u);
      TmemFenceAfterSync();
#pragma unroll
      for (int blk = 0; blk < 2; ++blk) {
        const int c0 = 64 * ch + 32 * blk;
        const uint32_t dcol = tbase + Pl::kdPCol + c0;
        float dp[pt::kBlock];
        TmemLoad32(lane_base | dcol, dp);
        TmemWaitLd();
        const float* const dl =
            reinterpret_cast<const float*>(tb + ColumnTable::kDl) + obase + 32 * blk;
        uint32_t dsv[pt::kBlock / 2];
        if (full) {
          DsStage<E, false>(pe[blk], dp, dl, vm[blk], dsv);
        } else {
          DsStage<E, true>(pe[blk], dp, dl, vm[blk], dsv);
        }
        dkpipe::TmemStore16(lane_base | dcol, dsv);
      }
      TmemWaitSt();
      TmemFenceBeforeSync();
      dvpipe::MbarArrive(bar(Br::kDsFull));
      ++r;
      c = n;
    }
    const bool any = r > 0;

    // ---------------------------------------------------------------- epilogue (KvPipe's)
    if (any) {
      const uint32_t last = r - 1;
      MbarWait(bar(Br::kAcc + last % kStages), (last / kStages) & 1u);
      TmemFenceAfterSync();
    }
    const bool is_k = (ch == 0);
    const int64_t width = is_k ? p.qk_dim : p.value_dim;
    const float mul = is_k ? p.scale : 1.f;
    void* row_out = nullptr;
    if (j < p.prev_length) {
      row_out = KeyGradRow<E, false>(is_k ? p.dprev_k : p.dprev_v,
                                     (b * p.prev_length + j) * p.kv_heads + hk, width);
    } else if (j >= p.prev_length && j < nt_total) {
      const int64_t lk = j - p.prev_length;
      row_out = KeyGradRow<E, false>(is_k ? p.dk : p.dv, (b * p.k_length + lk) * p.kv_heads + hk,
                                     width);
    }
    const uint32_t acc_col = is_k ? Pl::kdKCol : Pl::kdVCol;
    for (int cc = 0; cc < 128; cc += 32) {
      float v[32];
      if (any) {
        TmemLoad32(lane_base | (tbase + acc_col + cc), v);
        TmemWaitLd();
      } else {
#pragma unroll
        for (int x = 0; x < 32; ++x) v[x] = 0.f;
      }
      if (row_out != nullptr) StoreKeyGrad32<E, false>(row_out, cc, width, v, mul);
    }
  }

  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
}

}  // namespace kvr128

// The companion launch in 128-query rounds: KvPipe's companion preflight, then this plan's SMEM.
template <typename E>
bool LaunchUmmaBackwardKvR128CompanionT(const UmmaLaunchParams& lp, const KvCompanion& comp,
                                        cudaStream_t stream) {
  using Pl = kvr128::Plan;
  const Params& p = lp.params;
  if (!CanLaunchUmmaBackwardKvPipeCompanionT<E>(lp)) return false;
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: cudaGetDevice", q);
  int smem_cap = 0;
  q = cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: shared memory", q);
  if (static_cast<int>(Pl::kSmemBytes) > smem_cap) {
    return LaunchUmmaBackwardKvPipeCompanionT<E>(lp, comp, stream);
  }
  const auto kernel = kvr128::BackwardKvR128Kernel<E, true>;
  const cudaError_t attr =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: cudaFuncSetAttribute", attr);
  }
  const unsigned gx =
      static_cast<unsigned>((p.total_length + kvpipe::kKeyRows - 1) / kvpipe::kKeyRows);
  if (gx == 0) return true;
  const dim3 grid(gx, static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
  KvPipeLaunch launch{};
  launch.lp = lp;
  launch.comp = comp;
  kernel<<<grid, kvr128::kThreads, Pl::kSmemBytes, stream>>>(launch);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: kernel launch", launched);
  }
  kvpipe::KvPipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
  InSituLaunchedKeyRows() = kvpipe::kKeyRows;
  return true;
}

// H2 r1: the plain call (no companion read) in 128-query rounds, the default of
// LaunchUmmaBackwardKvPipe for 16-bit key gradients. The caller has run KvPipe's preflight.
// Returns false, enqueuing nothing, when the device's opt-in SMEM cannot hold this plan.
template <typename E>
bool LaunchUmmaBackwardKvR128PlainT(const UmmaLaunchParams& lp, cudaStream_t stream) {
  using Pl = kvr128::Plan;
  const Params& p = lp.params;
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: cudaGetDevice", q);
  int smem_cap = 0;
  q = cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: shared memory", q);
  if (static_cast<int>(Pl::kSmemBytes) > smem_cap) return false;
  const auto kernel = kvr128::BackwardKvR128Kernel<E, false>;
  const cudaError_t attr =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: cudaFuncSetAttribute", attr);
  }
  const unsigned gx =
      static_cast<unsigned>((p.total_length + kvpipe::kKeyRows - 1) / kvpipe::kKeyRows);
  if (gx == 0) return true;  // no work (see LaunchUmmaBackwardDkdv)
  const dim3 grid(gx, static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
  KvPipeLaunch launch{};
  launch.lp = lp;
  kernel<<<grid, kvr128::kThreads, Pl::kSmemBytes, stream>>>(launch);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: kernel launch", launched);
  }
  kvpipe::KvPipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
  return true;
}

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
