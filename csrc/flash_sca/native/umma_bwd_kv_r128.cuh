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
inline constexpr int kThreads = 32 * (kPackWarps + 2);  // 320 (8 pack + producer + MMA)
static_assert(kRoundQueries == kBwdQueryRows, "one round per query tile and head");

struct Bars {
  static constexpr int kK = 0;         // K landed
  static constexpr int kV = 1;         // V landed
  static constexpr int kLdQ = 2;       // + st: Q(r) landed in stage st
  static constexpr int kLdG = 4;       // + st: dO(r) landed in stage st
  static constexpr int kSFull = 6;     // S^T(r) complete
  static constexpr int kPFull = 7;     // P^T(r) packed (256 arrivals)
  static constexpr int kDpFull = 8;    // dP^T(r) complete
  static constexpr int kDsFull = 9;    // dS^T(r) packed (256 arrivals)
  static constexpr int kDvDone = 10;   // + st: dV(r) complete -> dO stage st free
  static constexpr int kAcc = 12;      // + st: dK(r) complete -> Q stage st free
  static constexpr int kCount = 14;
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

// Per-head stats (negated lse2, delta) loaded every round; bounds loaded only when tile changes.
struct RoundStat {
  float nlse2 = -INFINITY;  // -lse * log2(e), or -INFINITY when non-finite (ex2(-inf) = 0)
  float delta = 0.f;
};
struct RoundBounds {
  int32_t lo = 1;
  int32_t hi = 0;
  int64_t seg = 0;
};

template <bool kComp>
__device__ __forceinline__ RoundStat LoadRoundStat(const KvPipeLaunch& L, int64_t b, int64_t hk,
                                                   const dkpipe::Round& c, int64_t ic) {
  const Params& p = L.lp.params;
  RoundStat st;
  if (ic >= p.q_length) return st;
  if (kComp && c.gh == p.group) {
    const int64_t idx = (b * p.kv_heads + hk) * p.q_length + ic;
    const float lse = L.comp.lse != nullptr ? L.comp.lse[idx] : -INFINITY;
    st.nlse2 = isfinite(lse) ? -lse * pt::kLog2e : -INFINITY;
    st.delta = L.comp.delta != nullptr ? L.comp.delta[idx] : 0.f;
  } else {
    const int64_t hq = hk * p.group + c.gh;
    const int64_t idx = (b * p.query_heads + hq) * p.q_length + ic;
    const float lse = p.lse_in != nullptr ? p.lse_in[idx] : -INFINITY;
    st.nlse2 = isfinite(lse) ? -lse * pt::kLog2e : -INFINITY;
    st.delta = p.delta != nullptr ? p.delta[idx] : 0.f;
  }
  return st;
}

template <bool kComp>
__device__ __forceinline__ RoundBounds LoadRoundBounds(const KvPipeLaunch& L, int64_t b,
                                                       const dkpipe::Round& c, int64_t ic,
                                                       bool seg) {
  const Params& p = L.lp.params;
  RoundBounds bd;
  if (ic >= p.q_length) return bd;
  if (kComp && c.gh == p.group) {
    int64_t lo, hi;
    if (L.comp.row_lo != nullptr) {
      lo = L.comp.row_lo[b * p.q_length + ic];
      hi = L.comp.row_hi[b * p.q_length + ic];
    } else {
      const int64_t pos = ic + p.total_length - p.q_length;
      lo = (pos / p.chunk - 1) * p.chunk;
      hi = pos - 1;
    }
    lo = lo < 0 ? 0 : lo;
    hi = hi >= p.total_length ? p.total_length - 1 : hi;
    if (lo > hi) {
      bd.lo = INT_MAX;
      bd.hi = -1;
    } else {
      bd.lo = static_cast<int32_t>(lo);
      bd.hi = static_cast<int32_t>(hi);
    }
  } else {
    const detail::Bounds rb = detail::RowBounds(p, b, ic);
    bd.lo = rb.lo;
    bd.hi = rb.hi;
  }
  bd.seg = seg ? p.q_segment[b * p.q_length + ic] : 0;
  return bd;
}

// Stage P unmasked in-place on `pe[0..31]` (holds S on entry, P on exit); `nl2` holds -lse2.
template <class E>
__device__ __forceinline__ void PStageFullInPlace(float (&pe)[pt::kBlock], const float* nl2,
                                                  float scale_log2, uint32_t* out_p) {
  const float2 sc = make_float2(scale_log2, scale_log2);
#pragma unroll
  for (int q = 0; q < pt::kBlock / 4; ++q) {
    const float4 NL = *reinterpret_cast<const float4*>(nl2 + 4 * q);
    const float2 x0 =
        pt::FmaX2(make_float2(pe[4 * q + 0], pe[4 * q + 1]), sc, make_float2(NL.x, NL.y));
    const float2 x1 =
        pt::FmaX2(make_float2(pe[4 * q + 2], pe[4 * q + 3]), sc, make_float2(NL.z, NL.w));
    pe[4 * q + 0] = kvpipe::KvExp2(x0.x);
    pe[4 * q + 1] = kvpipe::KvExp2(x0.y);
    pe[4 * q + 2] = kvpipe::KvExp2(x1.x);
    pe[4 * q + 3] = kvpipe::KvExp2(x1.y);
    out_p[2 * q] = kvpipe::PackPair<E>(pe[4 * q + 0], pe[4 * q + 1]);
    out_p[2 * q + 1] = kvpipe::PackPair<E>(pe[4 * q + 2], pe[4 * q + 3]);
  }
}

// Stage P masked in-place on `pe[0..31]` using precomputed 32-bit visibility mask `vm`.
template <class E>
__device__ __forceinline__ void PStageVmInPlace(float (&pe)[pt::kBlock], const float* nl2,
                                                uint32_t vm, float scale_log2, uint32_t* out_p) {
  const float2 sc = make_float2(scale_log2, scale_log2);
#pragma unroll
  for (int q = 0; q < pt::kBlock / 4; ++q) {
    const float4 NL = *reinterpret_cast<const float4*>(nl2 + 4 * q);
    const int base = 4 * q;
    const float2 x0 =
        pt::FmaX2(make_float2(pe[base + 0], pe[base + 1]), sc, make_float2(NL.x, NL.y));
    const float2 x1 =
        pt::FmaX2(make_float2(pe[base + 2], pe[base + 3]), sc, make_float2(NL.z, NL.w));
    pe[base + 0] = ((vm >> (base + 0)) & 1u) ? kvpipe::KvExp2(x0.x) : 0.f;
    pe[base + 1] = ((vm >> (base + 1)) & 1u) ? kvpipe::KvExp2(x0.y) : 0.f;
    pe[base + 2] = ((vm >> (base + 2)) & 1u) ? kvpipe::KvExp2(x1.x) : 0.f;
    pe[base + 3] = ((vm >> (base + 3)) & 1u) ? kvpipe::KvExp2(x1.y) : 0.f;
    out_p[2 * q] = kvpipe::PackPair<E>(pe[base + 0], pe[base + 1]);
    out_p[2 * q + 1] = kvpipe::PackPair<E>(pe[base + 2], pe[base + 3]);
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
  const uint32_t bar_k = bar0 + 8u * Br::kK;
  const uint32_t bar_v = bar0 + 8u * Br::kV;
  auto bar = [&](int i) { return bar0 + 8u * static_cast<uint32_t>(i); };

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  int64_t b, hk, kt;
  if (!kComp && lp.kv_ljf_order != nullptr) {
    const uint32_t packed = static_cast<uint32_t>(lp.kv_ljf_order[blockIdx.x]);
    b = static_cast<int64_t>(packed >> 24);
    hk = static_cast<int64_t>((packed >> 16) & 0xffu);
    kt = static_cast<int64_t>(packed & 0xffffu);
  } else {
    b = blockIdx.z;
    hk = blockIdx.y;
    kt = blockIdx.x;
  }
  const int64_t kb = kt * kKeyRows;
  const int64_t ke = kb + kKeyRows;
  const int64_t nt_total = p.total_length;

  int64_t qt_first, qt_last;
  if (tid == 0) {
    QueryTileEnvelope<kKeyRows>(p, b, kt, kb, ke - 1, &qt_first, &qt_last);
    red[0] = static_cast<int32_t>(qt_first);
    red[1] = static_cast<int32_t>(qt_last);
    if (qt_first <= qt_last) {
      for (int i = 0; i < Br::kCount; ++i) {
        const int count = (i == Br::kPFull || i == Br::kDsFull) ? kPackThreads : 1;
        MbarInit(bar(i), count);
      }
      MbarInitFence();
      const CUtensorMap* kmap;
      const CUtensorMap* vmap;
      int row0;
      ResolveKeyMap(lp, kb, &kmap, &vmap, &row0);
      MbarArriveExpectTx(bar_k, Pl::kKBytes);
#pragma unroll
      for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int r = 0; r < kKeyRows; r += kTmaRows) {
          TmaLoad4D(sk + s * Pl::kSlabBytes + r * 128, kmap, bar_k, s * 64,
                    static_cast<int>(hk), row0 + r, static_cast<int>(b));
        }
      }
      MbarArriveExpectTx(bar_v, Pl::kVBytes);
#pragma unroll
      for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int r = 0; r < kKeyRows; r += kTmaRows) {
          TmaLoad4D(sv + s * Pl::kSlabBytes + r * 128, vmap, bar_v, s * 64,
                    static_cast<int>(hk), row0 + r, static_cast<int>(b));
        }
      }
    }
  }
  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  const bool seg = detail::SegmentTest(p, b);
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
        unsigned char* const qs = smem + st * Pl::kStageBytes;
        unsigned char* const gs = qs + Pl::kQBytes;
        const uint32_t blq = bar(Br::kLdQ + st);
        const uint32_t blg = bar(Br::kLdG + st);
        const bool cr = kComp && c.gh == p.group;
        const CUtensorMap* const qmap = cr ? &L.comp.q : &lp.maps.q;
        const CUtensorMap* const gmap = cr ? &L.comp.dy : &lp.maps.dy;
        const int hq = cr ? static_cast<int>(hk) : static_cast<int>(hk * p.group + c.gh);
        const int i0 = static_cast<int>(c.t * kBwdQueryRows);
        auto load_q = [&]() {
          MbarArriveExpectTx(blq, Pl::kQBytes);
#pragma unroll
          for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int rr = 0; rr < kRoundQueries; rr += kTmaRows) {
              TmaLoad4D(qs + s * Pl::kSlabBytes + rr * 128, qmap, blq, s * 64, hq, i0 + rr,
                        static_cast<int>(b));
            }
          }
        };
        auto load_g = [&]() {
          MbarArriveExpectTx(blg, Pl::kGBytes);
#pragma unroll
          for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int rr = 0; rr < kRoundQueries; rr += kTmaRows) {
              TmaLoad4D(gs + s * Pl::kSlabBytes + rr * 128, gmap, blg, s * 64, hq, i0 + rr,
                        static_cast<int>(b));
            }
          }
        };
        if (r < static_cast<uint32_t>(kStages)) {
          // Prologue: issue Q first so S(0) starts immediately, then dO.
          load_q();
          load_g();
        } else {
          // Steady state: dV(r - kStages) frees dO ~1024 cycles before dK frees Q; load dO early!
          const uint32_t prev_ph = ((r - kStages) / kStages) & 1u;
          MbarWait(bar(Br::kDvDone + st), prev_ph);
          load_g();
          MbarWait(bar(Br::kAcc + st), prev_ph);
          load_q();
        }
        ++r;
        NextR<kComp>(p, b, kb, ke, qt_last, &c);
      }
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    // ---------------------------------------------------------------- MMA issuer
    dkpipe::Round c;
    dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    if (c.t <= qt_last) {
      const uint32_t s_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kRoundQueries, 0, 0);
      const uint32_t v_idesc = MakeInstrDesc(elem_fmt, kKeyRows, 128, 0, 1);
      const uint32_t k_idesc = MakeInstrDesc(elem_fmt, kKeyRows, 128, 0, 1);
      constexpr uint32_t kSlabLbo = static_cast<uint32_t>(Pl::kSlabBytes / 16);
      static_assert(kSlabLbo < (1u << 14), "LBO field");
      const uint32_t s_base = SmemAddr(smem);
      const uint64_t k_desc0 = MakeSmemDesc(SmemAddr(sk)).value;
      const uint64_t v_desc0 = MakeSmemDesc(SmemAddr(sv)).value;
      const uint64_t q_desc0 = MakeSmemDesc(s_base).value;
      const uint64_t g_desc0 = MakeSmemDesc(s_base + Pl::kQBytes).value;
      const uint64_t dk_q_desc0 = MakeSmemDesc(s_base, kSlabLbo, 64).value;
      const uint64_t dv_g_desc0 = MakeSmemDesc(s_base + Pl::kQBytes, kSlabLbo, 64).value;
      auto issue_s_raw = [&](uint32_t r) {
        const uint64_t qs =
            q_desc0 + static_cast<uint64_t>((r % kStages) * (Pl::kStageBytes >> 4));
#pragma unroll
        for (int kk = 0; kk < 128 / 16; ++kk) {
          const uint64_t off =
              static_cast<uint64_t>(((kk >> 2) * Pl::kSlabBytes + (kk & 3) * 32) >> 4);
          MmaSsRaw(tbase + Pl::kSCol, k_desc0 + off, qs + off, s_idesc, kk == 0 ? 0u : 1u);
        }
      };
      auto issue_dp_raw = [&](uint32_t r) {
        const uint64_t gs =
            g_desc0 + static_cast<uint64_t>((r % kStages) * (Pl::kStageBytes >> 4));
#pragma unroll
        for (int kk = 0; kk < 128 / 16; ++kk) {
          const uint64_t off =
              static_cast<uint64_t>(((kk >> 2) * Pl::kSlabBytes + (kk & 3) * 32) >> 4);
          MmaSsRaw(tbase + Pl::kdPCol, v_desc0 + off, gs + off, s_idesc, kk == 0 ? 0u : 1u);
        }
      };
      auto issue_dv_raw = [&](uint32_t r) {
        const uint64_t gs =
            dv_g_desc0 + static_cast<uint64_t>((r % kStages) * (Pl::kStageBytes >> 4));
#pragma unroll
        for (int kl = 0; kl < kSteps; ++kl) {
          MmaTsRaw(tbase + Pl::kdVCol, tbase + PackedCol(Pl::kSCol, kl),
                   gs + static_cast<uint64_t>((kl * 16 * 128) >> 4), v_idesc,
                   (r == 0 && kl == 0) ? 0u : 1u);
        }
      };
      auto issue_dk_raw = [&](uint32_t r) {
        const uint64_t qs =
            dk_q_desc0 + static_cast<uint64_t>((r % kStages) * (Pl::kStageBytes >> 4));
#pragma unroll
        for (int kl = 0; kl < kSteps; ++kl) {
          MmaTsRaw(tbase + Pl::kdKCol, tbase + PackedCol(Pl::kdPCol, kl),
                   qs + static_cast<uint64_t>((kl * 16 * 128) >> 4), k_idesc,
                   (r == 0 && kl == 0) ? 0u : 1u);
        }
      };
      // S(0) starts as soon as K and Q(0) land; V and dO(0) arrive while S(0) runs!
      MbarWait(bar_k, 0);
      MbarWait(bar(Br::kLdQ + 0), 0u);
      TmemFenceAfterSync();
      __syncwarp();
      if (ElectOne()) {
        issue_s_raw(0u);
        MmaCommitRaw(bar(Br::kSFull));
      }
      MbarWait(bar_v, 0);
      MbarWait(bar(Br::kLdG + 0), 0u);
      TmemFenceAfterSync();
      __syncwarp();
      if (ElectOne()) {
        issue_dp_raw(0u);
        MmaCommitRaw(bar(Br::kDpFull));
      }
      for (uint32_t r = 0;; ++r) {
        dkpipe::Round n = c;
        NextR<kComp>(p, b, kb, ke, qt_last, &n);
        const bool more = n.t <= qt_last;
        const uint32_t r1 = r + 1;
        MbarWait(bar(Br::kPFull), r & 1u);
        TmemFenceAfterSync();
        __syncwarp();
        if (ElectOne()) {
          issue_dv_raw(r);
          MmaCommitRaw(bar(Br::kDvDone + r % kStages));
        }
        if (more) {
          MbarWait(bar(Br::kLdQ + r1 % kStages), (r1 / kStages) & 1u);
          TmemFenceAfterSync();
          __syncwarp();
          if (ElectOne()) {
            issue_s_raw(r1);  // over P^T(r): ordered behind dV(r) by this one issuer
            MmaCommitRaw(bar(Br::kSFull));
          }
        }
        MbarWait(bar(Br::kDsFull), r & 1u);
        TmemFenceAfterSync();
        __syncwarp();
        if (ElectOne()) {
          issue_dk_raw(r);
          MmaCommitRaw(bar(Br::kAcc + r % kStages));
        }
        if (!more) break;
        MbarWait(bar(Br::kLdG + r1 % kStages), (r1 / kStages) & 1u);
        TmemFenceAfterSync();
        __syncwarp();
        if (ElectOne()) {
          issue_dp_raw(r1);  // over dS^T(r): ordered behind dK(r)
          MmaCommitRaw(bar(Br::kDpFull));
        }
        c = n;
      }
    }
    __syncwarp();
  } else if (warp < kPackWarps) {
    // ---------------------------------------------------------------- pack warps 0..7
    const int w4 = warp & 3;
    const int ch = warp >> 2;
    const uint32_t lane_base = static_cast<uint32_t>(32 * w4) << 16;
    const int64_t j = kb + w4 * 32 + lane;
    const int64_t ks = (seg && j < nt_total) ? p.k_segment[b * nt_total + j] : 0;
    const int32_t j32 = static_cast<int32_t>(j);
    unsigned char* const tb = smem + Pl::kMetaOff;
    dkpipe::Round c;
    dkpipe::FirstRound(p, b, kb, ke, qt_first, qt_last, &c);
    RoundStat cm_st;
    RoundBounds cm_bd;
    int64_t loaded_bnd_t = -1;
    bool loaded_bnd_comp = false;
    int64_t vm_cached_t = -1;
    bool vm_cached_comp = false;
    uint32_t vm[2] = {0xffffffffu, 0xffffffffu};
    if (c.t <= qt_last && tid < kRoundQueries) {
      cm_st = LoadRoundStat<kComp>(L, b, hk, c, c.t * kBwdQueryRows + tid);
      const bool full0 = !seg && FullTile(c.m, kb, ke) && !(kComp && c.gh == p.group);
      if (!full0) {
        cm_bd = LoadRoundBounds<kComp>(L, b, c, c.t * kBwdQueryRows + tid, seg);
        loaded_bnd_t = c.t;
        loaded_bnd_comp = (kComp && c.gh == p.group);
      }
    }
    auto compute_vm = [&](int o) -> uint32_t {
      const int32_t* const lo = reinterpret_cast<const int32_t*>(tb + ColumnTable::kLo) + o;
      const int32_t* const hi = reinterpret_cast<const int32_t*>(tb + ColumnTable::kHi) + o;
      const int64_t* const sg = reinterpret_cast<const int64_t*>(tb + ColumnTable::kSg) + o;
      uint32_t mask = 0u;
#pragma unroll
      for (int q = 0; q < pt::kBlock / 4; ++q) {
        const int4 LO = *reinterpret_cast<const int4*>(lo + 4 * q);
        const int4 HI = *reinterpret_cast<const int4*>(hi + 4 * q);
        const int32_t lov[4] = {LO.x, LO.y, LO.z, LO.w};
        const int32_t hiv[4] = {HI.x, HI.y, HI.z, HI.w};
        int64_t sv[4] = {0, 0, 0, 0};
        if (seg) {
          const longlong2 A = *reinterpret_cast<const longlong2*>(sg + 4 * q);
          const longlong2 B = *reinterpret_cast<const longlong2*>(sg + 4 * q + 2);
          sv[0] = A.x;
          sv[1] = A.y;
          sv[2] = B.x;
          sv[3] = B.y;
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          bool v = (j32 >= lov[e]) && (j32 <= hiv[e]);
          if (seg) v = v && (ks == sv[e]);
          if (v) mask |= (1u << (4 * q + e));
        }
      }
      return mask;
    };
    uint32_t r = 0;
    while (c.t <= qt_last) {
      dkpipe::Round n = c;
      NextR<kComp>(p, b, kb, ke, qt_last, &n);
      const bool cur_comp = (kComp && c.gh == p.group);
      const bool full = !seg && FullTile(c.m, kb, ke) && !cur_comp;
      const bool need_vm = !full && (c.t != vm_cached_t || cur_comp != vm_cached_comp);
      const uint32_t buf = r & 1u;
      if (tid < kRoundQueries) {
        const int o = static_cast<int>(2u * buf) * kvpipe::kRoundQueries + tid;
        reinterpret_cast<float*>(tb + ColumnTable::kL2)[o] = cm_st.nlse2;
        reinterpret_cast<float*>(tb + ColumnTable::kDl)[o] = cm_st.delta;
        if (need_vm) {
          reinterpret_cast<int32_t*>(tb + ColumnTable::kLo)[o] = cm_bd.lo;
          reinterpret_cast<int32_t*>(tb + ColumnTable::kHi)[o] = cm_bd.hi;
          if (seg) reinterpret_cast<int64_t*>(tb + ColumnTable::kSg)[o] = cm_bd.seg;
        }
      }
      dvpipe::PackBarrierSync();
      // Prefetch next round's per-head stats; only load bounds if next round's tile/comp changes.
      if (n.t <= qt_last && tid < kRoundQueries) {
        cm_st = LoadRoundStat<kComp>(L, b, hk, n, n.t * kBwdQueryRows + tid);
        const bool next_comp = (kComp && n.gh == p.group);
        const bool next_full = !seg && FullTile(n.m, kb, ke) && !next_comp;
        if (!next_full && (n.t != loaded_bnd_t || next_comp != loaded_bnd_comp)) {
          cm_bd = LoadRoundBounds<kComp>(L, b, n, n.t * kBwdQueryRows + tid, seg);
          loaded_bnd_t = n.t;
          loaded_bnd_comp = next_comp;
        }
      }
      const int obase = static_cast<int>(2u * buf) * kvpipe::kRoundQueries + 64 * ch;
      // Compute visibility masks BEFORE waiting for S^T(r) so mask evaluation overlaps MMA!
      if (need_vm) {
        vm[0] = compute_vm(obase);
        vm[1] = compute_vm(obase + 32);
        vm_cached_t = c.t;
        vm_cached_comp = cur_comp;
      }
      // Stage P: in-place TmemLoad32 into pe[blk] (saves 32 regs, zero code duplication, starts
      // MUFU.EX2 immediately after the first 32 columns arrive).
      float pe[2][pt::kBlock];
      MbarWait(bar(Br::kSFull), r & 1u);
      TmemFenceAfterSync();
#pragma unroll
      for (int blk = 0; blk < 2; ++blk) {
        const int c0 = 64 * ch + 32 * blk;
        const uint32_t scol = tbase + Pl::kSCol + c0;
        if (!full && __all_sync(0xffffffffu, vm[blk] == 0u)) {
          uint32_t z[pt::kBlock / 2];
#pragma unroll
          for (int i = 0; i < pt::kBlock / 2; ++i) z[i] = 0u;
          dkpipe::TmemStore16(lane_base | scol, z);
          continue;
        }
        TmemLoad32(lane_base | scol, pe[blk]);
        TmemWaitLd();
        const float* const nl2 =
            reinterpret_cast<const float*>(tb + ColumnTable::kL2) + obase + 32 * blk;
        uint32_t pv[pt::kBlock / 2];
        if (full || __all_sync(0xffffffffu, vm[blk] == 0xffffffffu)) {
          PStageFullInPlace<E>(pe[blk], nl2, p.scale_log2, pv);
        } else {
          PStageVmInPlace<E>(pe[blk], nl2, vm[blk], p.scale_log2, pv);
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
        if (!full && __all_sync(0xffffffffu, vm[blk] == 0u)) {
          uint32_t z[pt::kBlock / 2];
#pragma unroll
          for (int i = 0; i < pt::kBlock / 2; ++i) z[i] = 0u;
          dkpipe::TmemStore16(lane_base | dcol, z);
          continue;
        }
        float dp[pt::kBlock];
        TmemLoad32(lane_base | dcol, dp);
        TmemWaitLd();
        const float* const dl =
            reinterpret_cast<const float*>(tb + ColumnTable::kDl) + obase + 32 * blk;
        uint32_t dsv[pt::kBlock / 2];
        if (full || __all_sync(0xffffffffu, vm[blk] == 0xffffffffu)) {
          DsStage<E, false>(pe[blk], dp, dl, 0xffffffffu, dsv);
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

    // ---------------------------------------------------------------- epilogue
    if (any) {
      const uint32_t last = r - 1;
      MbarWait(bar(Br::kAcc + last % kStages), (last / kStages) & 1u);
      TmemFenceAfterSync();
    } else if (qt_first <= qt_last) {
      // Ensure initial K/V TMA loads into sk/sv have completed before reusing sk/sv for TMA stores.
      MbarWait(bar_k, 0);
      MbarWait(bar_v, 0);
    }
    const bool is_k = (ch == 0);
    const int64_t width = is_k ? p.qk_dim : p.value_dim;
    const float mul = is_k ? p.scale : 1.f;
    const uint32_t acc_col = is_k ? Pl::kdKCol : Pl::kdVCol;
    const bool use_prev_tma = (ke <= p.prev_length) && lp.maps.has_dprev_kv_map;
    const bool use_curr_tma = (kb >= p.prev_length) && lp.maps.has_dkv_map;
    if (use_prev_tma || use_curr_tma) {
      unsigned char* const s_out = is_k ? sk : sv;
      const int krow = w4 * 32 + lane;
      const int swz = krow & 7;
      const float2 mul2 = make_float2(mul, mul);
#pragma unroll
      for (int cc = 0; cc < 128; cc += 32) {
        float v[32];
        if (any) {
          TmemLoad32(lane_base | (tbase + acc_col + cc), v);
          TmemWaitLd();
        }
        if (j < nt_total) {
          uint4* const chunks =
              reinterpret_cast<uint4*>(s_out + (cc >> 6) * Pl::kSlabBytes + krow * 128);
          const int chunk0 = (cc & 63) >> 3;
          if (any) {
            union {
              uint4 u4[4];
              uint32_t w[16];
            } pk;
#pragma unroll
            for (int x = 0; x < 16; ++x) {
              const float2 m = pt::MulX2(make_float2(v[2 * x], v[2 * x + 1]), mul2);
              pk.w[x] = pt::Pack2Fast<E>::Of(m.x, m.y);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) chunks[(chunk0 + i) ^ swz] = pk.u4[i];
          } else {
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int i = 0; i < 4; ++i) chunks[(chunk0 + i) ^ swz] = z;
          }
        }
      }
      FenceProxyAsyncShared();
      dvpipe::PackBarrierSync();
      if (tid == 0) {
        const CUtensorMap* const dk_map = use_prev_tma ? &lp.maps.dprev_k : &lp.maps.dk;
        const CUtensorMap* const dv_map = use_prev_tma ? &lp.maps.dprev_v : &lp.maps.dv;
        const int tma_row0 = static_cast<int>(use_prev_tma ? kb : (kb - p.prev_length));
        const uint32_t dk_sbase = SmemAddr(sk);
        const uint32_t dv_sbase = SmemAddr(sv);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
          for (int rr = 0; rr < kKeyRows; rr += kTmaRows) {
            TmaStoreAtom(dk_map, dk_sbase + static_cast<uint32_t>(s * Pl::kSlabBytes + rr * 128),
                         s * 64, static_cast<int>(hk), tma_row0 + rr, static_cast<int>(b));
            TmaStoreAtom(dv_map, dv_sbase + static_cast<uint32_t>(s * Pl::kSlabBytes + rr * 128),
                         s * 64, static_cast<int>(hk), tma_row0 + rr, static_cast<int>(b));
          }
        }
        TmaStoreCommit();
        TmaStoreWaitRead();
      }
    } else {
      void* row_out = nullptr;
      if (j < p.prev_length) {
        row_out = KeyGradRow<E, false>(is_k ? p.dprev_k : p.dprev_v,
                                       (b * p.prev_length + j) * p.kv_heads + hk, width);
      } else if (j >= p.prev_length && j < nt_total) {
        const int64_t lk = j - p.prev_length;
        row_out = KeyGradRow<E, false>(is_k ? p.dk : p.dv, (b * p.k_length + lk) * p.kv_heads + hk,
                                       width);
      }
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
  static int cached_dev = -1;
  static int cached_smem_cap = 0;
  int device = 0;
  cudaError_t q = cudaGetDevice(&device);
  if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: cudaGetDevice", q);
  if (device != cached_dev) {
    q = cudaDeviceGetAttribute(&cached_smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (q != cudaSuccess) dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: shared memory", q);
    cached_dev = device;
  }
  if (static_cast<int>(Pl::kSmemBytes) > cached_smem_cap) return false;
  const auto kernel = kvr128::BackwardKvR128Kernel<E, false>;
  const cudaError_t attr =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd kv r128: cudaFuncSetAttribute", attr);
  }
  const int64_t nkt = (p.total_length + kvpipe::kKeyRows - 1) / kvpipe::kKeyRows;
  if (nkt == 0) return true;  // no work (see LaunchUmmaBackwardDkdv)
  const int64_t total_tiles64 = nkt * p.kv_heads * p.batch;
  const dim3 grid = (lp.kv_ljf_order != nullptr)
                        ? dim3(static_cast<unsigned>(total_tiles64))
                        : dim3(static_cast<unsigned>(nkt), static_cast<unsigned>(p.kv_heads),
                               static_cast<unsigned>(p.batch));
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
