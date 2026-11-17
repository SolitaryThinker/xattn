// Native Blackwell (sm_103a) SCA backward: the dV-only M128 key leaf at (256,256),
// warp-specialized and pipelined across rounds. It is the first leaf of the (256,256) key pair
// that the in-situ launch runs at level 0 (umma_bwd_dkdv.cuh, LaunchUmmaBackwardInSituDkdv);
// the pipelined dK-only leaf (umma_bwd_dk_pipe.cuh) follows it on the stream and writes dK and
// dPrevK.
//
// It computes exactly what the unpipelined dV-only leaf computes
// (BackwardDkdvKernel<E, 256, 256, 2, 128, 0>): dV and dPrevV, while dK and dPrevK are NOT
// written. One CTA owns one (batch, KV head, 128-key tile); one round is one admitted
// 128-query tile and one query head:
//   S^T  = K Q^T        tcgen05.mma SS, M128 N128, K = D, into an S^T buffer
//   P^T  = exp2(S^T * scale_log2 - lse2), masked, packed to 16 bits IN PLACE over S^T
//   dV  += P^T dO       tcgen05.mma TS, A = P^T in TMEM, B = dO (MN-major), 4 slabs x 8 k-steps
// The 256 pack threads (warps 0..7) keep the unpipelined leaf's lane/column mapping and its
// arithmetic line for line (same helpers, same ColumnMeta table, same in-place WAR barrier),
// and the MMA k-order and the round order are unchanged, so dV is expected bitwise equal to
// the unpipelined dV-only leaf. The round sequence (query tiles ascending inside the key
// tile's envelope, MaySee skipping, heads ascending inside a tile) is evaluated by each role
// with the same table reads, so all roles agree on it.
//
// Roles, 320 threads (__launch_bounds__(320, 1)):
//   warps 0..7  pack: wait S^T(r) -> pack P^T(r) in its own buffer -> tcgen05.wait::st,
//               fence::before_thread_sync, arrive pack_done(r) (256 arrivals); afterwards the
//               epilogue (drain dV once the last dV MMA has completed, store dV / dPrevV);
//   warp 8      TMA producer, ONE thread polling two independent streams with test_wait:
//               Q(r+1) as soon as S^T(r) has completed (bar_s), dO(r+1) as soon as dV(r) has
//               completed (bar_dv). There is ONE Q stage and ONE dO stage - a second 64 KB stage
//               does not fit (199,680 B of 232,448 B) - each released right after its last
//               reader;
//   warp 9      MMA issuer: S^T(0); then for each r: S^T(r+1) into buffer (r+1)%2 once Q(r+1)
//               has landed and dV(r-1) has completed (the last reader of that buffer, a
//               conservative release), WITHOUT waiting for pack(r); then dV(r) once pack(r) is
//               done and dO(r) has landed.
// Every __syncthreads() of the unpipelined pack is a named barrier over the 256 pack threads
// here (bar.sync 1, 256): a CTA-wide barrier would deadlock against the producer and MMA warps.
// The CTA-wide barriers that remain are the prologue's and the tile exit's, which all 320
// threads reach exactly once.
//
// TMEM (512 columns allocated): dV accumulator cols 0..255; S^T buffer b at 256 + 128 b (buffer
// 0: 256..383, buffer 1: 384..511), its packed P^T in place at 256 + 128 b .. +63.
// SMEM: the dv128 image unchanged (Q 64 KB at 0, dO 64 KB at 64 KB, K 64 KB at 128 KB, no V),
// then 9 mbarriers in a 16-slot area, the TMEM slot, the envelope words and the 128 key
// segments: the same 199,680 B of dynamic SMEM as the dv128 leaf (+3,072 B static ColumnMeta).
//
// Barrier phases. Round r uses buffer r & 1. bar_q and bar_g complete once per round (parity
// r & 1); bar_s[b], bar_dv[b] and bar_pack[b] complete once per round of their buffer (parity
// (r >> 1) & 1). A parity wait is exact only if the previous phase of its barrier has completed
// before the wait starts and the next phase cannot complete before the wait returns; each wait
// below has both properties:
//   MMA   waits bar_q(r+1): Q(r+1) is issued after S(r), which was issued after Q(r) landed;
//         Q(r+2) needs S(r+1), issued after this wait.
//   MMA   waits bar_dv[(r-1)&1] (dV(r-1)): it observed dV(r-3) before issuing S(r-1); dV(r+1)
//         is issued by itself later.
//   MMA   waits bar_pack[r&1] (pack(r)): pack(r-2) observed earlier; pack(r+2) needs S(r+2),
//         which it issues only after this wait. (One shared pack barrier would NOT be safe:
//         pack(r+1) only needs S(r+1), which is issued BEFORE this wait.)
//   MMA   waits bar_g(r): dO(r-1) observed before dV(r-1); dO(r+1) needs dV(r).
//   pack  waits bar_s[r&1] (S(r)): S(r-2) observed by pack(r-2); S(r+2) needs dV(r), which
//         needs pack(r) to have arrived, i.e. every pack thread to have passed this wait.
//   prod  tests bar_s[(r-1)&1] / bar_dv[(r-1)&1] for round r-1 in round order: rounds r-3 were
//         observed before Q(r-2) / dO(r-2) were issued; S(r+1) needs Q(r+1) and dV(r+1) needs
//         dO(r+1), both issued only after this test succeeded.
//   pack  (epilogue) waits bar_dv of the last round L: dV(L-2) completed before S(L) was
//         issued, and nothing is committed to that barrier after dV(L).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/umma_bwd_dkdv.cuh"

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace dvpipe {

inline constexpr int kPackWarps = 8;
inline constexpr int kPackThreads = 32 * kPackWarps;    // 256: the unpipelined leaf's CTA
inline constexpr int kProducerWarp = kPackWarps;        // warp 8
inline constexpr int kMmaWarp = kPackWarps + 1;         // warp 9
inline constexpr int kThreads = 32 * (kPackWarps + 2);  // 320
inline constexpr int kPackBarrier = 1;  // bar.sync id of the 256 pack threads (0 = __syncthreads)
inline constexpr int kKeyRows = 128;
static_assert(kPackThreads == kBwdThreads,
              "the pack warps are the unpipelined leaf's 256 threads, with its mapping");
static_assert(kPackThreads == 2 * kBwdQueryRows,
              "the pack splits the 128 query columns across two groups of four warps");

// mbarrier slots (8 B each) after the operand image.
inline constexpr int kBarKv = 0;    // K tile landed (once per CTA)
inline constexpr int kBarQ = 1;     // Q(r) landed (TMA, expect_tx)
inline constexpr int kBarG = 2;     // dO(r) landed (TMA, expect_tx)
inline constexpr int kBarS = 3;     // + (r & 1): S^T(r) complete (tcgen05.commit)
inline constexpr int kBarDv = 5;    // + (r & 1): dV(r) complete (tcgen05.commit)
inline constexpr int kBarPack = 7;  // + (r & 1): P^T(r) packed (kPackThreads arrivals)
inline constexpr int kBarCount = 9;

template <int kD, int kDv>
struct Plan {
  // The dv128 plan: operand image, slab geometry and TMEM columns are taken from it.
  using Base = BackwardDkdvPlan<kD, kDv, kKeyRows, 1, 2>;
  // General by capacity, not by bucket name: symmetric to dkpipe::Plan's note above. dV needs
  // kDv <= 256 TMEM columns (kdVColBase + kDv <= kSCol, asserted below); kD only sets how much
  // K this leaf stages and carries no TMEM-column requirement of its own.
  static_assert(Base::kDvOnlyM128 && kD % 64 == 0 && kDv % 64 == 0 && kD > 0 && kD <= 256 &&
                    kDv > 0 && kDv <= 256,
                "the pipelined dV-only leaf fits every (D, Dv) with D, Dv <= 256, each a "
                "multiple of 64");
  static_assert(Base::kVBytes == 0, "the dV-only leaf stages no V");
  static constexpr int kSlabsK = Base::kSlabsK;
  static constexpr int kSlabsV = Base::kSlabsV;
  static constexpr int kKeySlabBytes = Base::kKeySlabBytes;
  static constexpr int kQuerySlabBytes = Base::kQuerySlabBytes;
  static constexpr int kKBytes = Base::kKBytes;
  static constexpr int kQBytes = Base::kQBytes;
  static constexpr int kGBytes = Base::kGBytes;
  static constexpr int kQOff = Base::kQOff;
  static constexpr int kGOff = Base::kGOff;
  static constexpr int kKOff = Base::kKOff;
  static constexpr int kdVColBase = Base::kdVColBase;  // 0
  static constexpr int kSCol = Base::kSCol;            // 256: S^T buffer 0
  static constexpr int kSBufCols = kBwdQueryRows;      // 128: S^T buffer 1 at 384
  static constexpr int kAllocCols = Base::kAllocCols;  // 512
  static_assert(kdVColBase == 0 && kdVColBase + kDv <= kSCol, "dV sits below both S^T buffers");
  static_assert(kSCol + 2 * kSBufCols <= kAllocCols, "two S^T buffers fit the allocation");
  static constexpr int kBarOff = kKOff + kKBytes;
  static constexpr int kSlotOff = kBarOff + 128;  // 16 barrier slots, kBarCount used
  static constexpr int kRedOff = kSlotOff + 16;
  static constexpr int kSegOff = kRedOff + 32;
  static constexpr int kSmemBytes = ((kSegOff + kKeyRows * 8 + 1023) / 1024) * 1024 + 1024;
  static_assert(kBarCount * 8 <= kSlotOff - kBarOff, "barrier area");
  static_assert(kQOff % 1024 == 0 && kGOff % 1024 == 0 && kKOff % 1024 == 0,
                "descriptor tile base must stay 1024-aligned");
  static_assert(kSegOff % 8 == 0, "key-segment table must stay 8-aligned");
  static_assert(kSmemBytes <= Base::kSmemBytes,
                "no more dynamic SMEM than the dv128 leaf, whose capacity test the launch reuses");
};

// Non-blocking phase test (try_wait may suspend the thread; the producer polls two barriers).
__device__ __forceinline__ bool MbarTestWait(uint32_t mbar, uint32_t phase) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t ok;
  asm volatile("{\n\t.reg .pred P;\n\t"
               "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, P;\n\t}\n"
               : "=r"(ok)
               : "r"(mbar), "r"(phase)
               : "memory");
  return ok != 0;
#else
  (void)mbar;
  (void)phase;
  return true;
#endif
}

// One arrival (release, CTA scope) on a count-kPackThreads barrier.
__device__ __forceinline__ void MbarArrive(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mbar) : "memory");
#endif
}

// The pack warps' CTA barrier: the 256 pack threads only.
__device__ __forceinline__ void PackBarrierSync() {
  asm volatile("bar.sync %0, %1;" ::"n"(kPackBarrier), "n"(kPackThreads) : "memory");
}

// First query tile t' >= t of [.., last] that MaySee the key tile [kb, ke), or last + 1.
// The unpipelined round loop's own skip test (FromTable + MaySee per tile), as a cursor.
__device__ __forceinline__ int64_t NextTile(const Params& p, int64_t b, int64_t kb, int64_t ke,
                                            int64_t t, int64_t last, QueryTileMask* m) {
  for (; t <= last; ++t) {
    *m = FromTable(p, b, 2 * t, 2);
    if (MaySee(*m, kb, ke)) return t;
  }
  return last + 1;
}

// kFp32Keys: dV/dPrevV in FP32 (Params::key_grads_fp32), a separate instantiation so that the
// 16-bit one keeps its own store code.
template <typename E, int kD, int kDv, bool kFp32Keys = false>
__global__ void __launch_bounds__(kThreads, 1) BackwardDvPipeKernel(
    const __grid_constant__ UmmaLaunchParams lp) {
  using Pl = Plan<kD, kDv>;
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  // Same 1024-byte rebase as the unpipelined leaf (descriptor base_offset = 0).
  unsigned char* const smem = smem_raw + ((1024u - (SmemAddr(smem_raw) & 1023u)) & 1023u);

  unsigned char* const sq = smem + Pl::kQOff;
  unsigned char* const sg = smem + Pl::kGOff;
  unsigned char* const sk = smem + Pl::kKOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Pl::kRedOff);
  int64_t* const kseg = reinterpret_cast<int64_t*>(smem + Pl::kSegOff);
  const uint32_t bar_kv = bar0 + 8 * kBarKv;
  const uint32_t bar_q = bar0 + 8 * kBarQ;
  const uint32_t bar_g = bar0 + 8 * kBarG;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hk = blockIdx.y;
  const int64_t kt = blockIdx.x;
  const int64_t kb = kt * kKeyRows;  // first global key of this tile
  const int64_t ke = kb + kKeyRows;  // exclusive
  const int64_t nt_total = p.total_length;

  if (tid == 0) {
    for (int i = 0; i < kBarCount; ++i) {
      MbarInit(bar0 + 8 * i, (i == kBarPack || i == kBarPack + 1) ? kPackThreads : 1);
    }
  }
  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }
  if (tid == 0) MbarInitFence();

  // ---- the key tile, once: K into SMEM, and the key segments if needed
  const bool seg = detail::SegmentTest(p, b);
  if (seg && tid < kKeyRows) {
    const int64_t j = kb + tid;
    kseg[tid] = j < nt_total ? p.k_segment[b * nt_total + j] : 0;
  }
  if (tid == 0) {
    const CUtensorMap* kmap;
    const CUtensorMap* vmap;
    int row0;
    ResolveKeyMap(lp, kb, &kmap, &vmap, &row0);
    MbarArriveExpectTx(bar_kv, Pl::kKBytes);
    for (int s = 0; s < Pl::kSlabsK; ++s) {
      for (int r = 0; r < kKeyRows; r += kTmaRows) {
        TmaLoad4D(sk + s * Pl::kKeySlabBytes + r * 128, kmap, bar_kv, s * 64,
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

  const uint32_t tbase = *tmem_slot & 0x1FFu;  // 1-CTA allocation: lane 0, column base
  const int elem_fmt = BwdTok<E>::kFormat;

  if (warp == kProducerWarp) {
    // ------------------------------------------------------------- TMA producer
    // One thread, two cursors over the same round sequence. Q(rq) may be issued once S^T(rq-1)
    // has completed (the only reader of the single Q stage), dO(rg) once dV(rg-1) has completed
    // (the only reader of the single dO stage). Neither stream waits for the other.
    if (lane == 0) {
      QueryTileMask m;
      int64_t qt = NextTile(p, b, kb, ke, qt_first, qt_last, &m);
      int64_t gt = qt;
      int64_t qgh = 0, ggh = 0;
      uint32_t rq = 0, rg = 0;
      uint32_t idle = 0;
      while (qt <= qt_last || gt <= qt_last) {
        bool moved = false;
        if (qt <= qt_last &&
            (rq == 0 ||
             MbarTestWait(bar0 + 8 * (kBarS + ((rq - 1) & 1)), ((rq - 1) >> 1) & 1))) {
          const int hq = static_cast<int>(hk * p.group + qgh);
          const int64_t i0 = qt * kBwdQueryRows;
          MbarArriveExpectTx(bar_q, Pl::kQBytes);
          for (int s = 0; s < Pl::kSlabsK; ++s) {
            for (int r = 0; r < kBwdQueryRows; r += kTmaRows) {
              TmaLoad4D(sq + s * Pl::kQuerySlabBytes + r * 128, &lp.maps.q, bar_q, s * 64, hq,
                        static_cast<int>(i0 + r), static_cast<int>(b));
            }
          }
          ++rq;
          if (++qgh == p.group) {
            qgh = 0;
            qt = NextTile(p, b, kb, ke, qt + 1, qt_last, &m);
          }
          moved = true;
        }
        if (gt <= qt_last &&
            (rg == 0 ||
             MbarTestWait(bar0 + 8 * (kBarDv + ((rg - 1) & 1)), ((rg - 1) >> 1) & 1))) {
          const int hq = static_cast<int>(hk * p.group + ggh);
          const int64_t i0 = gt * kBwdQueryRows;
          MbarArriveExpectTx(bar_g, Pl::kGBytes);
          for (int s = 0; s < Pl::kSlabsV; ++s) {
            for (int r = 0; r < kBwdQueryRows; r += kTmaRows) {
              TmaLoad4D(sg + s * Pl::kQuerySlabBytes + r * 128, &lp.maps.dy, bar_g, s * 64, hq,
                        static_cast<int>(i0 + r), static_cast<int>(b));
            }
          }
          ++rg;
          if (++ggh == p.group) {
            ggh = 0;
            gt = NextTile(p, b, kb, ke, gt + 1, qt_last, &m);
          }
          moved = true;
        }
        if (moved) {
          idle = 0;
        } else {
          // Bounded like MbarWait: a stream that never becomes issuable is a kernel bug, and a
          // trap is the report; a live pipeline never comes near the bound.
          if (++idle > (1u << 22)) __trap();
          __nanosleep(20);
        }
      }
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    // ------------------------------------------------------------- MMA issuer
    // K is waited for unconditionally: a CTA with no round must not exit while its K load is
    // still writing shared memory.
    MbarWait(bar_kv, 0);
    QueryTileMask m;
    int64_t t = NextTile(p, b, kb, ke, qt_first, qt_last, &m);
    if (t <= qt_last) {
      const uint32_t s_idesc = MakeInstrDesc(elem_fmt, kKeyRows, kBwdQueryRows, /*a=*/0, /*b=*/0);
      const uint32_t v_idesc = MakeInstrDesc(elem_fmt, kKeyRows, 64, /*a=*/0, /*b=*/1);
      // S^T = K Q^T into buffer `buf`: the unpipelined leaf's k-loop, at the buffer's columns.
      auto issue_s = [&](uint32_t buf) {
        for (int kk = 0; kk < kD / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(tbase + Pl::kSCol + buf * Pl::kSBufCols,
                MakeSmemDesc(SmemAddr(sk) + slab * Pl::kKeySlabBytes + within * 32),
                MakeSmemDesc(SmemAddr(sq) + slab * Pl::kQuerySlabBytes + within * 32), s_idesc,
                kk == 0 ? 0u : 1u);
        }
        MmaCommit(bar0 + 8 * (kBarS + buf));
      };
      int64_t gh = 0;
      uint32_t r = 0;
      MbarWait(bar_q, 0u);
      __syncwarp();
      issue_s(0u);
      for (;;) {
        int64_t nt = t, ngh = gh + 1;
        if (ngh == p.group) {
          ngh = 0;
          nt = NextTile(p, b, kb, ke, t + 1, qt_last, &m);
        }
        const bool more = nt <= qt_last;
        if (more) {
          // S^T(r+1) into buffer (r+1)&1: Q(r+1) landed, and that buffer's last reader - the
          // dV MMA of round r-1 - has completed. pack(r) is NOT waited for.
          MbarWait(bar_q, (r + 1) & 1);
          if (r >= 1) MbarWait(bar0 + 8 * (kBarDv + ((r - 1) & 1)), ((r - 1) >> 1) & 1);
          TmemFenceAfterSync();
          __syncwarp();
          issue_s((r + 1) & 1);
        }
        // dV(r) += P^T(r) dO(r): P^T(r) packed (pack_done) and dO(r) landed.
        MbarWait(bar0 + 8 * (kBarPack + (r & 1)), (r >> 1) & 1);
        TmemFenceAfterSync();
        MbarWait(bar_g, r & 1);
        __syncwarp();
        const uint32_t pcol = Pl::kSCol + (r & 1) * Pl::kSBufCols;
        for (int sl = 0; sl < Pl::kSlabsV; ++sl) {
          for (int kk = 0; kk < kBwdQueryRows / 16; ++kk) {
            MmaTs(tbase + Pl::kdVColBase + sl * 64, tbase + pcol + kk * 8,
                  MakeSmemDesc(SmemAddr(sg) + sl * Pl::kQuerySlabBytes + kk * 16 * 128, 0u, 64),
                  v_idesc, (r == 0 && kk == 0) ? 0u : 1u);
          }
        }
        MmaCommit(bar0 + 8 * (kBarDv + (r & 1)));
        if (!more) break;
        t = nt;
        gh = ngh;
        ++r;
      }
    }
  } else {
    // ------------------------------------------------------------- pack warps 0..7
    // TMEM lane block of this warp: warps 4..7 address the SAME 128 key rows as warps 0..3 and
    // differ only in the query columns they pack (the unpipelined leaf's mapping).
    const uint32_t lane_base = static_cast<uint32_t>(32 * (warp & 3)) << 16;
    __shared__ pt::ColumnMeta smeta[kBwdQueryRows];
    // lse2 as its own contiguous SoA plane (+512 B static SMEM), read by the pack core
    // instead of gathered from smeta[c0].lse2; smeta itself (lo/hi/segment) is unchanged and
    // still feeds BlockIsSuffix/Visible below.
    __shared__ float smeta_lse2[kBwdQueryRows];
    __shared__ int32_t smeta_blk[kBwdQueryRows / pt::kBlock];  // block passed BlockIsSuffix
    bool any = false;
    uint32_t r = 0;
    QueryTileMask m;
    for (int64_t t = NextTile(p, b, kb, ke, qt_first, qt_last, &m); t <= qt_last;
         t = NextTile(p, b, kb, ke, t + 1, qt_last, &m)) {
      const int64_t i0 = t * kBwdQueryRows;
      const bool full = FullTileExact(m, p, b, kb, ke);
      for (int64_t gh = 0; gh < p.group; ++gh) {
        const int64_t hq = hk * p.group + gh;
        const uint32_t buf = r & 1;
        const uint32_t scol = Pl::kSCol + buf * Pl::kSBufCols;
        MbarWait(bar0 + 8 * (kBarS + buf), (r >> 1) & 1);
        TmemFenceAfterSync();

        // The unpipelined M128 dV-only pack, line for line; only the S^T window (`scol`) and
        // the two CTA barriers (now the 256-thread named barrier) differ.
        const int w4 = warp & 3;
        const int64_t row_in_tile = static_cast<int64_t>(w4 * 32 + lane);
        const int64_t j = kb + row_in_tile;
        const int64_t ks = seg ? kseg[row_in_tile] : 0;
        // Suffix-mask precondition (same as dk_pipe, and the same reason): the block must not
        // straddle a chunk boundary, i.e. the chunk and the query-to-key position shift must be
        // multiples of the block. Otherwise `lo` jumps inside the block and the mask would admit
        // keys those columns must not see.
        const bool suffix_ok = !seg && p.row_lo == nullptr && p.chunk % pt::kBlock == 0 &&
            (p.total_length - p.q_length) % pt::kBlock == 0;
        uint32_t pv[32];  // P^T pairs only: dV += P^T dO needs no dS^T, so none is produced
        // smeta is rewritten here only after every pack thread passed the previous round's
        // in-loop barrier, which follows its last read of smeta (rounds are processed in order).
        for (int sc = tid; sc < kBwdQueryRows; sc += kPackThreads) {
          const int64_t ic = i0 + sc;
          const pt::ColumnMeta cm = pt::ColumnOf(p, b, hq, ic, seg, ic < p.q_length);
          smeta[sc] = cm;
          smeta_lse2[sc] = cm.lse2;
          // Warps 0..3 hold the round's four 32-column blocks, one column per lane.
          const bool blk_ok = pt::BlockIsSuffix(cm.lo, cm.hi, cm.segment, seg);
          if (lane == 0) smeta_blk[sc / pt::kBlock] = blk_ok ? 1 : 0;
        }
        PackBarrierSync();
        const int blk0 = (tid >= kBwdQueryRows) ? kBwdQueryRows / pt::kBlock / 2 : 0;
        for (int blk = blk0; blk < blk0 + kBwdQueryRows / pt::kBlock / 2; ++blk) {
          const int c0 = blk * pt::kBlock;
          float s[pt::kBlock];
          TmemLoadCols(lane_base | (tbase + scol + c0), s, pt::kBlock);
          uint32_t* pvp = pv + 16 * (blk & 1);
          if (full) {
            pt::MakePtBlockFullPSoA<E, pt::kBlock>(s, &smeta_lse2[c0], p.scale_log2, pvp);
          } else if ((suffix_ok && smeta[c0].lo <= smeta[c0].hi) || smeta_blk[blk] != 0) {
            // For key row j the visible query columns are a suffix of the 32-column block
            // (lo constant, hi slope 1, one segment: pt::SuffixMask, pt::BlockIsSuffix), for
            // the whole launch by the analytic guard or for this block by its metadata. The
            // analytic guard also needs the block's first column live (an empty row is
            // {INT_MAX, -1}: a strict-past read's first query), as in the dK leaf.
            const uint32_t mask = pt::BlockSuffixMask(j, smeta[c0].hi, smeta[c0].lo,
                                                      smeta[c0].segment, ks, seg, pt::kBlock);
            pt::MakePtBlockSuffixPSoA<E, pt::kBlock>(s, &smeta_lse2[c0], mask, p.scale_log2, pvp);
          } else {
            pt::MakePtBlockFastPSoA<E, pt::kBlock>(s, &smeta_lse2[c0], &smeta[c0], j, ks, seg,
                                                   p.scale_log2, pvp);
          }
          if (blk & 1) {
            const int pair = blk >> 1;
            // In-place WAR: the second group's store lands on columns the first group read as
            // its block 1 (see the unpipelined leaf); both groups reach this barrier after
            // their second load and before their store.
            TmemFenceBeforeSync();
            PackBarrierSync();
            TmemFenceAfterSync();
            uint32_t pvx[32];
#pragma unroll
            for (int x = 0; x < 32; ++x) pvx[x] = pv[x];
            TmemStore32(lane_base | (tbase + pt::PackedColumn(static_cast<int>(scol), 2 * pair)),
                        pvx);
          }
        }
        // Hand P^T(r) to the MMA warp.
        TmemWaitSt();
        TmemFenceBeforeSync();
        MbarArrive(bar0 + 8 * (kBarPack + buf));
        any = true;
        ++r;
      }
    }

    // ----------------------------------------------------------------- epilogue
    // The unpipelined leaf's dV sweep (kKeyRows 128, kWrite 2): lane L of warps 0..3 holds key
    // row 32 (warp & 3) + L; warps 4..7 read the same rows and store nothing.
    if (any) {
      MbarWait(bar0 + 8 * (kBarDv + ((r - 1) & 1)), ((r - 1) >> 1) & 1);
      TmemFenceAfterSync();
    }
    const int rr = 32 * (warp & 3) + lane;
    const int64_t j = kb + rr;
    // This leaf writes dV only, whose accumulator holds exactly kDv TMEM columns
    // (kdVColBase..kdVColBase+kDv-1): at kD != kDv (the unequal buckets) max(kD, kDv) here
    // read columns past the accumulator instead of the dV width alone.
    const int cols = kDv;
    const int64_t width = p.value_dim;
    const float mul = 1.f;
    void* row_out = nullptr;
    if (j < p.prev_length) {
      row_out = KeyGradRow<E, kFp32Keys>(p.dprev_v, (b * p.prev_length + j) * p.kv_heads + hk, width);
    } else if (j >= p.prev_length && j < nt_total) {
      const int64_t lk = j - p.prev_length;
      row_out = KeyGradRow<E, kFp32Keys>(p.dv, (b * p.k_length + lk) * p.kv_heads + hk, width);
    }
    if (warp >= 4) row_out = nullptr;
    for (int c = 0; c < cols; c += 32) {
      float v[32];
      if (any) {
        TmemLoad32(lane_base | (tbase + Pl::kdVColBase + c), v);
        TmemWaitLd();
      } else {
#pragma unroll
        for (int x = 0; x < 32; ++x) v[x] = 0.f;
      }
      if (row_out != nullptr) StoreKeyGrad32<E, kFp32Keys>(row_out, c, width, v, mul);
    }
  }

  // ------------------------------------------------------------------ tile exit
  // All 320 threads: the pack warps after the epilogue (which waited for the last dV), the
  // producer after its last issue, the MMA warp after its last commit.
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
}

}  // namespace dvpipe

// Launch entry of the pipelined dV-only leaf (declared in umma_bwd_dkdv.cuh, instantiated for
// BF16 and FP16 at every (D, Dv) the pair serves: (256,256) in
// instantiations/sca_umma_bwd_dkdv_d256_v256.cu, the four unequal buckets in
// instantiations/sca_umma_bwd_dkdv_pair_d*.cu). Its refusal set is the unpipelined dV-only
// leaf's (UmmaBackwardDkdvCanLaunch<E, kD, kDv, 2, 128, 0>: accepted set, bucket and real
// dimensions, element, device 10.3, SMEM capacity - this plan asks no more dynamic SMEM than
// that one, asserted above); `false` means nothing was written, and after it every driver or
// launch error raises, exactly like LaunchUmmaBackwardDkdv.
template <typename E, int kD, int kDv>
bool LaunchUmmaBackwardDvPipe(const UmmaLaunchParams& lp, cudaStream_t stream) {
  using Pl = dvpipe::Plan<kD, kDv>;
  const Params& p = lp.params;
  if (!UmmaBackwardDkdvCanLaunch<E, kD, kDv, 2, dvpipe::kKeyRows, 0>(lp)) return false;
  const auto kernel = p.key_grads_fp32 ? dvpipe::BackwardDvPipeKernel<E, kD, kDv, true>
                                       : dvpipe::BackwardDvPipeKernel<E, kD, kDv, false>;
  const cudaError_t attr =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pl::kSmemBytes);
  if (attr != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dV pipe: cudaFuncSetAttribute", attr);
  }
  const unsigned gx =
      static_cast<unsigned>((p.total_length + dvpipe::kKeyRows - 1) / dvpipe::kKeyRows);
  if (gx == 0) return true;  // no work (see LaunchUmmaBackwardDkdv)
  const dim3 grid(gx, static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
  kernel<<<grid, dvpipe::kThreads, Pl::kSmemBytes, stream>>>(lp);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    dkdv_detail::RaiseCuda("flash_sca native bwd dV pipe: kernel launch", launched);
  }
  return true;
}

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
