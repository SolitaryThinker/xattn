// The native Blackwell (sm_103a) FlashSCA forward at
// (D, Dv) = (256, 256) and (128, 128), level 0, warp-specialized and pipelined across key tiles.
//
// Same work decomposition, tile walk and per-element semantics as umma_fwd.cuh's
// ForwardKernel (one CTA per (batch, query head, 128-query tile); key tiles of 64 from
// min_lo / 64 to max_hi / 64 over the CTA's live rows; TileIsFull on max_lo / min_hi over ALL
// rows; per-element row bounds and, when SegmentTest holds, the per-key segment test; empty
// row y = 0 and LSE = -inf; the mask is a select on the exponent's argument, so zero and
// negative scales stay defined). What changes is the schedule:
//
//   warps 0 .. kSmWarps-1  softmax (4: thread t owns query row t and the 64 columns of a tile;
//                          8: warps w and w + 4 share row 32 (w & 3) + lane, columns 0..31 and
//                          32..63, row max and row sum combined through SMEM per tile)
//   warp kSmWarps          TMA producer (one thread): Q once, then K(j) and V(j) into separate
//                          rings, each stage released by its own reader's tcgen05.commit
//   warp kSmWarps + 1      MMA issuer: S(0); then per tile j: S(j+1) = Q K(j+1)^T into S buffer
//                          (j+1) % kSB, then O += P(j) V(j) once softmax(j) has arrived
//
// TMEM (512 columns): O FP32 at 0..255 (lane = query row); S buffer b at 256 + 64 b; the
// softmax packs P(j) (16-bit, round to nearest even) IN PLACE over its own S(j) columns, so the
// PV MMA reads A from TMEM (kind::f16 TS form, N = 256 per instruction, B = V MN-major with the
// 64-row slab hop as LBO, the backward dK pipe's form).
//
// Online softmax with a LAZY rescale (log2 domain, level 0 form P = exp2(s * scale_log2 - m)):
// a row keeps the maximum it last rescaled to (m_used); a tile whose maximum exceeds m_used by
// more than `rescale_thr` (8, so P <= 2^8) waits for the previous PV, rescales the row's
// O and l by exp2(m_used - m_new) (a select to exact 0 when that underflows) and moves m_used.
// The first visible tile of a row only sets m_used (its O row is exactly zero). LSE =
// ln2 * m_used + ln(l), which is the same quantity ForwardKernel writes.
// A row with no visible key (dead by its bounds, or live by its bounds with no key of its segment)
// keeps m_used = -inf; every P it packs is exp2(-inf + 0) = 0, so its O row and its l stay exactly
// 0, and the epilogue writes y = 0 (inv = 0 when l == 0; a dead row writes zeros through
// write_dead) and LSE = -inf (l > 0 is the only finite branch): never 0/0 and never NaN, in a CTA
// with live rows as in a CTA without any.
// Rounding: a FULL tile forms the exponent's argument as fmaf(s, scale_log2, -m) (one rounding); a
// masked tile scales first (s * scale_log2, which the -inf select needs) and then adds -m (two
// roundings). The two forms of one score can differ in the last FP32 ulp, so P and y are not
// bit-equal between them, and this body is bit-equal neither to the older leaf nor to the generic
// body; it is held to the contract's tolerances, not to their bits.
//
// Barrier phases (tile j uses K stage j % kKS, V stage j % 2, S buffer j % kSB; each barrier
// completes once per use of its slot, so a wait for use u is on parity u & 1, and every wait
// below sees the previous phase of its barrier completed and cannot be overtaken by the next):
//   k_full / v_full   producer expect_tx + TMA bytes; waited by the MMA warp
//   k_empty / v_empty commit after S(j) / PV(j), only when a later load reuses the stage;
//                     waited by the producer before K(j + kKS) / V(j + 2)
//   s[b]              commit after S(j); waited by the softmax warps
//   p[b]              kSmThreads arrivals after P(j) is stored (and O rescaled); waited by MMA
//   pv[b]             commit after PV(j); waited by MMA before S(j + kSB) (last reader of the
//                     buffer), by a rescaling softmax warp in tile j + 1, and by the epilogue
// The last commit the MMA warp issues is pv of the last tile, which the epilogue waits for, so
// no commit can arrive on this CTA's shared memory after it exits.
//
// Compiled configuration: one kernel per element type and bucket, PipeForwardKernel<E, 8, 2, 2,
// 256> and <E, 8, 2, 2, 128> (8 softmax warps, 2 K stages, 2 S buffers; the last argument is D =
// Dv and every width-dependent constant of the plan derives from it; TryLaunchPipe below). Which
// calls it serves, the XATTN_SCA_FWD_PIPE arm and the per-call body slot are in
// umma_fwd_pipe_host.h, the precedence of the two forward arms is host.cc's ResolveForwardBody,
// and the launch sites are LaunchUmmaForward<E, 256, 256> and <E, 128, 128> (umma_fwd.cuh), after
// those launchers' own refusals.
#pragma once

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash_sca/native/umma_fwd.cuh"
#include "flash_sca/native/umma_fwd_pipe_host.h"

namespace xattn::ops::flash_sca_native {
namespace umma {
namespace fwdpipe {

inline constexpr int kPipeRows = 128;  // query rows per CTA (M128: TMEM lane = row)
inline constexpr int kPipeKeys = 64;   // keys per tile (one TMA box per 64-column slab)
// The D = Dv bucket each instantiation is built for: (256,256) and (128,128) exist. The plan's
// last template argument is the width and every constant that depends on it - the 64-column slab
// count, the Q and KV tile bytes, S's TMEM column and the O columns per softmax thread - derives
// from it, so a second bucket costs a second body and changes no arithmetic for the first.
inline constexpr int kPipeWidthMax = 256;
inline constexpr int kSmBarrierId = 5;  // named barrier of all softmax threads (seg staging)

template <int kSmWarps_, int kKS_, int kSB_, int kW_>
struct Plan {
  static constexpr int kSmWarps = kSmWarps_;
  static constexpr int kKS = kKS_;  // K stages
  static constexpr int kVS = 2;     // V stages
  static constexpr int kSB = kSB_;  // S buffers in TMEM
  static constexpr int kPipeWidth = kW_;       // D = Dv of the bucket
  static constexpr int kPipeSlabs = kW_ / 64;  // 64-column TMA slabs per row
  static constexpr int kSmThreads = 32 * kSmWarps;
  static constexpr int kProducerWarp = kSmWarps;
  static constexpr int kMmaWarp = kSmWarps + 1;
  static constexpr int kThreads = 32 * (kSmWarps + 2);
  static constexpr int kQSlabBytes = kPipeRows * 128;   // 16,384
  static constexpr int kKvSlabBytes = kPipeKeys * 128;  // 8,192
  static constexpr int kQBytes = kPipeSlabs * kQSlabBytes;     // 65,536
  static constexpr int kTileBytes = kPipeSlabs * kKvSlabBytes; // 32,768
  static constexpr int kQOff = 0;
  static constexpr int kKOff = kQOff + kQBytes;
  static constexpr int kVOff = kKOff + kKS * kTileBytes;
  static constexpr int kBarOff = kVOff + kVS * kTileBytes;
  static constexpr int kBarQ = 0;
  static constexpr int kBarKFull = 1;
  static constexpr int kBarKEmpty = kBarKFull + kKS;
  static constexpr int kBarVFull = kBarKEmpty + kKS;
  static constexpr int kBarVEmpty = kBarVFull + kVS;
  static constexpr int kBarS = kBarVEmpty + kVS;
  static constexpr int kBarP = kBarS + kSB;
  static constexpr int kBarPv = kBarP + kSB;
  static constexpr int kBarCount = kBarPv + kSB;
  static constexpr int kSlotOff = kBarOff + ((kBarCount * 8 + 15) / 16) * 16;
  static constexpr int kRedOff = kSlotOff + 16;
  static constexpr int kKsegOff = kRedOff + 4 * 16;          // 4 row warps x 4 int32
  static constexpr int kXchOff = kKsegOff + 2 * kPipeKeys * 8;  // kseg[2][64] int64
  static constexpr int kXchBytes = kSmWarps == 8 ? 2 * 2 * kPipeRows * 4 : 0;
  static constexpr int kSmemBytes = kXchOff + kXchBytes;
  // TMEM columns.
  static constexpr int kOCol = 0;
  static constexpr int kSCol = kPipeWidth;
  static constexpr int kBufCols = kPipeKeys;
  // The whole TMEM of an SM (O and the two S buffers use 384 of its 512 columns). That is safe only
  // because exactly one CTA of this kernel is resident per SM: kSmemBytes (199,888 B of the SM's
  // 228 KiB) and __launch_bounds__(..., 1) force it. A second resident CTA (a variant with a
  // smaller SMEM footprint, or another TMEM user on the same SM) would wait in tcgen05.alloc until
  // this CTA frees its columns, so this body cannot be tuned by raising occupancy: lower the
  // allocation to what is used before any change that lets two CTAs share an SM.
  static constexpr int kAllocCols = 512;
  // Only <E, 8, 2, 2> is instantiated (TryLaunchPipe), and the asserts admit only that: the
  // 4-softmax-warp form and 3 K stages / 3 S buffers were tuning arms of an earlier probe that nothing
  // here compiles or validates. The 4-warp branches stay in the source; widening an assert needs
  // its own validation.
  static_assert(kSmWarps == 8, "8 softmax warps (the compiled configuration)");
  static_assert(kKS == 2, "2 K stages (the compiled configuration)");
  static_assert(kSB == 2, "2 S buffers (the compiled configuration)");
  static_assert(kPipeWidth == 128 || kPipeWidth == 256,
                "the buckets with an instantiation: (256,256) and (128,128)");
  static_assert(kSCol + kSB * kBufCols <= kAllocCols, "TMEM column budget");
  static_assert(kQOff % 1024 == 0 && kKOff % 1024 == 0 && kVOff % 1024 == 0 &&
                    kTileBytes % 1024 == 0 && kKvSlabBytes % 1024 == 0,
                "descriptor tile bases must stay 1024-aligned");
  static_assert(kKsegOff % 8 == 0 && kXchOff % 8 == 0, "SMEM tables stay 8-aligned");
  static_assert(kSmemBytes <= 232448, "the sm_103a opt-in dynamic SMEM ceiling");
};

// ---------------------------------------------------------------- device primitives
template <class E>
struct PackOf;
template <>
struct PackOf<__nv_bfloat16> {
  __device__ __forceinline__ static uint32_t Pack2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
  }
  __device__ __forceinline__ static __nv_bfloat16 From(float x) { return __float2bfloat16_rn(x); }
};
template <>
struct PackOf<__half> {
  __device__ __forceinline__ static uint32_t Pack2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
  }
  __device__ __forceinline__ static __half From(float x) { return __float2half_rn(x); }
};

__device__ __forceinline__ float Ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void Arrive(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mbar) : "memory");
#endif
}

__device__ __forceinline__ void NamedSync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// 32 lanes x 64 columns, then tcgen05.wait::ld, in ONE asm statement: no use of the loaded
// registers can be scheduled between the load and its wait.
__device__ __forceinline__ void Ld64(uint32_t addr, float (&v)[64]) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t r[64];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]),
        "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]),
        "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]),
        "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]),
        "=r"(r[62]), "=r"(r[63])
      : "r"(addr)
      : "memory");
#pragma unroll
  for (int x = 0; x < 64; ++x) v[x] = __uint_as_float(r[x]);
#else
  (void)addr;
#pragma unroll
  for (int x = 0; x < 64; ++x) v[x] = 0.f;
#endif
}

__device__ __forceinline__ void Ld32(uint32_t addr, float (&v)[32]) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(addr)
      : "memory");
#pragma unroll
  for (int x = 0; x < 32; ++x) v[x] = __uint_as_float(r[x]);
#else
  (void)addr;
#pragma unroll
  for (int x = 0; x < 32; ++x) v[x] = 0.f;
#endif
}

// 32 lanes x 16 columns, warp-collective (32x32b.x16), no wait.
__device__ __forceinline__ void St16(uint32_t addr, const uint32_t (&v)[16]) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};" ::"r"(v[0]),
      "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]),
      "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]),
      "r"(addr)
      : "memory");
#else
  (void)addr;
  (void)v;
#endif
}

// TMEM column of the packed P(j) for the 16-key k-step kk of buffer column `scol`: 4 softmax
// warps write 32 words at the buffer start; 8 write 16 words at the start of each 32-column half.
template <int kSmWarps>
__device__ __forceinline__ uint32_t PCol(uint32_t scol, int kk) {
  return kSmWarps == 4 ? scol + 8u * static_cast<uint32_t>(kk)
                       : scol + 32u * static_cast<uint32_t>(kk >> 1) + 8u * static_cast<uint32_t>(kk & 1);
}

#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
// Probe builds only (XATTN_FWD_PROBE_HOOKS, see umma_fwd_pipe_host.h; never in a shipped build):
// lazy-rescale events and, among them, the events whose factor underflowed to exactly 0 (the clear
// branch), one per softmax thread that takes the branch (both threads of a row in the 8-warp
// form), read back by TryLaunchPipe after every launch of a probe build.
static __device__ unsigned long long g_pipe_rescale_events = 0;
static __device__ unsigned long long g_pipe_clear_events = 0;
#endif

// ---------------------------------------------------------------- kernel
template <class E, int kSmWarps, int kKS, int kSB, int kW>
__global__ void __launch_bounds__(32 * (kSmWarps + 2), 1)
    PipeForwardKernel(const __grid_constant__ UmmaLaunchParams lp, const float rescale_thr) {
  using Pl = Plan<kSmWarps, kKS, kSB, kW>;
  constexpr int kHalves = kSmWarps / 4;
  constexpr int kCols = kPipeKeys / kHalves;        // S columns per softmax thread
  constexpr int kOCols = Pl::kPipeWidth / kHalves;  // O columns per softmax thread
  const Params& p = lp.params;
  extern __shared__ __align__(1024) unsigned char smem[];

  unsigned char* const sq = smem + Pl::kQOff;
  unsigned char* const sk = smem + Pl::kKOff;
  unsigned char* const sv = smem + Pl::kVOff;
  const uint32_t bar0 = SmemAddr(smem + Pl::kBarOff);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + Pl::kSlotOff);
  int32_t* const red = reinterpret_cast<int32_t*>(smem + Pl::kRedOff);
  int64_t* const kseg = reinterpret_cast<int64_t*>(smem + Pl::kKsegOff);
  float* const xch = reinterpret_cast<float*>(smem + Pl::kXchOff);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t b = blockIdx.z;
  const int64_t hq = blockIdx.y;
  const int qt = blockIdx.x;
  const int64_t kvh = p.group > 0 ? hq / p.group : hq;
  auto bar = [&](int idx) -> uint32_t { return bar0 + 8u * static_cast<uint32_t>(idx); };

  if (tid == 0) {
    // The descriptors encode base_offset = 0: every operand tile must sit on a 1024 B boundary.
    if ((SmemAddr(smem) & 1023u) != 0u) __trap();
    for (int x = 0; x < Pl::kBarCount; ++x) {
      const bool is_p = x >= Pl::kBarP && x < Pl::kBarP + kSB;
      MbarInit(bar(x), is_p ? Pl::kSmThreads : 1);
    }
    MbarInitFence();
  }
  if (warp == 0) {
    TmemAlloc(Pl::kAllocCols, tmem_slot);
    TmemRelinquish();
  }

  // Row ownership: softmax warp w holds row 32 (w & 3) + lane (TMEM lane group w & 3).
  const bool is_sm = warp < kSmWarps;
  const int g = warp & 3;
  const int h = (warp >> 2) & 1;
  const int row = 32 * g + lane;
  const int64_t i = static_cast<int64_t>(qt) * kPipeRows + row;
  const bool row_in_range = is_sm && i < p.q_length;
  const detail::Bounds bd = is_sm ? detail::RowBounds(p, b, i) : detail::Bounds{INT_MAX, -1};
  const bool live = row_in_range && bd.lo <= bd.hi;
  if (warp < 4) {
    // ForwardKernel's reduction, verbatim: min_lo/max_hi over live rows drive the walk,
    // max_lo/min_hi over ALL rows drive the FULL test (a dead row never allows it).
    const int w_lo = WarpMin(live ? bd.lo : INT_MAX);
    const int w_hi = WarpMax(live ? bd.hi : -1);
    const int w_mlo = WarpMax(live ? bd.lo : INT_MAX);
    const int w_mhi = WarpMin(live ? bd.hi : INT_MIN);
    if (lane == 0) {
      red[warp * 4 + 0] = w_lo;
      red[warp * 4 + 1] = w_hi;
      red[warp * 4 + 2] = w_mlo;
      red[warp * 4 + 3] = w_mhi;
    }
  }
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  int min_lo = red[0], max_hi = red[1], max_lo = red[2], min_hi = red[3];
  for (int w = 1; w < 4; ++w) {
    min_lo = min(min_lo, red[w * 4 + 0]);
    max_hi = max(max_hi, red[w * 4 + 1]);
    max_lo = max(max_lo, red[w * 4 + 2]);
    min_hi = min(min_hi, red[w * 4 + 3]);
  }
  const uint32_t tbase = *tmem_slot & 0x1FFu;  // 1-CTA allocation: lane 0, column base

  const int ocol0 = h * kOCols;
  auto write_dead = [&](bool with_lse) {
    // ForwardKernel's empty-row convention: y = 0, LSE = -inf (level 0 writes no state).
    if (!row_in_range) return;
    E* const yrow =
        reinterpret_cast<E*>(p.y) + ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim;
    if (p.value_dim == Pl::kPipeWidth) {
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      for (int c = ocol0; c < ocol0 + kOCols; c += 8) *reinterpret_cast<uint4*>(yrow + c) = z;
    } else {
      for (int64_t c = ocol0; c < ocol0 + kOCols && c < p.value_dim; ++c) {
        yrow[c] = PackOf<E>::From(0.f);
      }
    }
    if (with_lse && p.lse != nullptr) p.lse[(b * p.query_heads + hq) * p.q_length + i] = -INFINITY;
  };

  if (min_lo > max_hi) {
    // No live row in this CTA: nothing is loaded or issued.
    if (is_sm) write_dead(h == 0);
  } else {
    const int t0 = min_lo / kPipeKeys;
    const int t1 = max_hi / kPipeKeys;
    const int n = t1 - t0 + 1;

    if (warp == Pl::kProducerWarp) {
      // ------------------------------------------------------------ TMA producer
      if (lane == 0) {
        MbarArriveExpectTx(bar(Pl::kBarQ), Pl::kQBytes);
        for (int s = 0; s < Pl::kPipeSlabs; ++s) {
          for (int r = 0; r < kPipeRows; r += kTmaRows) {
            TmaLoad4D(sq + s * Pl::kQSlabBytes + r * 128, &lp.maps.q, bar(Pl::kBarQ), s * 64,
                      static_cast<int>(hq), qt * kPipeRows + r, static_cast<int>(b));
          }
        }
        auto load = [&](int j, bool is_k) {
          const int stages = is_k ? kKS : Pl::kVS;
          const int st = j % stages;
          const int full_bar = (is_k ? Pl::kBarKFull : Pl::kBarVFull) + st;
          const int empty_bar = (is_k ? Pl::kBarKEmpty : Pl::kBarVEmpty) + st;
          // The stage's previous reader (S(j - kKS) for K, PV(j - 2) for V) has completed.
          if (j >= stages) MbarWait(bar(empty_bar), static_cast<uint32_t>(((j - stages) / stages) & 1));
          const CUtensorMap* kmap;
          const CUtensorMap* vmap;
          int row0;
          ResolveKeyMap(lp, static_cast<int64_t>(t0 + j) * kPipeKeys, &kmap, &vmap, &row0);
          unsigned char* const dst = (is_k ? sk : sv) + st * Pl::kTileBytes;
          MbarArriveExpectTx(bar(full_bar), Pl::kTileBytes);
          for (int s = 0; s < Pl::kPipeSlabs; ++s) {
            TmaLoad4D(dst + s * Pl::kKvSlabBytes, is_k ? kmap : vmap, bar(full_bar), s * 64,
                      static_cast<int>(kvh), row0, static_cast<int>(b));
          }
        };
        // Release order of the stages: S(j - 1) completes before PV(j - 2), so K runs kKS - 1
        // tiles ahead of V and every blocking wait below is met in the order it happens.
        int jk = 0;
        for (; jk < n && jk < kKS - 1; ++jk) load(jk, true);
        for (int j = 0; j < n; ++j) {
          if (jk < n) {
            load(jk, true);
            ++jk;
          }
          load(j, false);
        }
      }
      __syncwarp();
    } else if (warp == Pl::kMmaWarp) {
      // ------------------------------------------------------------ MMA issuer
      const int fmt = ElemTraits<E>::kFormat;
      const uint32_t s_idesc = MakeInstrDesc(fmt, kPipeRows, kPipeKeys, /*a_major=*/0, /*b_major=*/0);
      const uint32_t pv_idesc = MakeInstrDesc(fmt, kPipeRows, Pl::kPipeWidth, /*a_major=*/0, /*b_major=*/1);
      MbarWait(bar(Pl::kBarQ), 0u);
      auto issue_s = [&](int j) {
        const int st = j % kKS;
        const int buf = j % kSB;
        MbarWait(bar(Pl::kBarKFull + st), static_cast<uint32_t>((j / kKS) & 1));
        // PV(j - kSB) read P(j - kSB) from this buffer: it must have completed.
        if (j >= kSB) MbarWait(bar(Pl::kBarPv + buf), static_cast<uint32_t>(((j - kSB) / kSB) & 1));
        TmemFenceAfterSync();
        __syncwarp();
        const uint32_t qa = SmemAddr(sq);
        const uint32_t ka = SmemAddr(sk + st * Pl::kTileBytes);
        const uint32_t dst = tbase + Pl::kSCol + buf * Pl::kBufCols;
#pragma unroll
        for (int kk = 0; kk < Pl::kPipeWidth / 16; ++kk) {
          const int slab = kk >> 2, within = kk & 3;
          MmaSs(dst, MakeSmemDesc(qa + slab * Pl::kQSlabBytes + within * 32),
                MakeSmemDesc(ka + slab * Pl::kKvSlabBytes + within * 32), s_idesc,
                kk == 0 ? 0u : 1u);
        }
        if (j + kKS < n) MmaCommit(bar(Pl::kBarKEmpty + st));
        MmaCommit(bar(Pl::kBarS + buf));
      };
      auto issue_pv = [&](int j) {
        const int st = j % Pl::kVS;
        const int buf = j % kSB;
        MbarWait(bar(Pl::kBarP + buf), static_cast<uint32_t>((j / kSB) & 1));
        MbarWait(bar(Pl::kBarVFull + st), static_cast<uint32_t>((j / Pl::kVS) & 1));
        TmemFenceAfterSync();
        __syncwarp();
        const uint32_t va = SmemAddr(sv + st * Pl::kTileBytes);
        const uint32_t scol = tbase + Pl::kSCol + buf * Pl::kBufCols;
#pragma unroll
        for (int kk = 0; kk < kPipeKeys / 16; ++kk) {
          // B = V(j) MN-major, N = 256 in one instruction: the next 64-column group is one
          // 64-row slab (8 KB = 512 x 16 B) further, the backward dK pipe's form.
          MmaTs(tbase + Pl::kOCol, PCol<kSmWarps>(scol, kk),
                MakeSmemDesc(va + kk * 16 * 128, 512u, 64u), pv_idesc,
                (j == 0 && kk == 0) ? 0u : 1u);
        }
        if (j + Pl::kVS < n) MmaCommit(bar(Pl::kBarVEmpty + st));
        MmaCommit(bar(Pl::kBarPv + buf));
      };
      issue_s(0);
      for (int j = 0; j < n; ++j) {
        if (j + 1 < n) issue_s(j + 1);
        issue_pv(j);
      }
    } else {
      // ------------------------------------------------------------ softmax warps
      const int col0 = h * kCols;
      const uint32_t lane_base = static_cast<uint32_t>(32 * g) << 16;
      const float sl2 = p.scale_log2;
      const bool neg_scale = sl2 < 0.f;
      const bool seg = detail::SegmentTest(p, b);
      // q_segment is a dense [B, Nq] plane (host.cc makes it contiguous), as in ForwardKernel.
      const int64_t qseg = (seg && row_in_range) ? p.q_segment[b * p.q_length + i] : 0;
      const int64_t nt_total = p.total_length;
      float m_used = -INFINITY;
      float lsum = 0.f;

      for (int j = 0; j < n; ++j) {
        const int buf = j % kSB;
        const int kb = (t0 + j) * kPipeKeys;
        const bool full = TileIsFull(seg, kb, max_lo, min_hi);
        if (seg) {
          // The tile's key segments, staged once for all softmax threads (double-buffered by
          // tile parity: one barrier per tile orders every write after the last read).
          if (tid < kPipeKeys) {
            const int64_t key = static_cast<int64_t>(kb) + tid;
            kseg[(j & 1) * kPipeKeys + tid] =
                key < nt_total ? p.k_segment[b * nt_total + key] : INT64_MIN;
          }
          NamedSync(kSmBarrierId, Pl::kSmThreads);
        }
        const uint32_t scol = tbase + Pl::kSCol + buf * Pl::kBufCols;
        MbarWait(bar(Pl::kBarS + buf), static_cast<uint32_t>((j / kSB) & 1));
        TmemFenceAfterSync();
        float s[kCols];
        if constexpr (kCols == 64) {
          Ld64(lane_base | scol, s);
        } else {
          Ld32(lane_base | (scol + col0), s);
        }

        // Tile maximum in log2-scaled units over this thread's visible columns. FULL: every row
        // of the CTA is live and sees all 64 keys; the extreme of the raw scores times the scale
        // equals the maximum of the scaled scores exactly (rounding is monotonic), for a
        // negative scale through the minimum. Otherwise the mask is a select on the scaled
        // score: masked entries become -inf and never enter any arithmetic.
        float tmax = -INFINITY;
        if (full) {
          float e = s[0];
          if (!neg_scale) {
#pragma unroll
            for (int c = 1; c < kCols; ++c) e = fmaxf(e, s[c]);
          } else {
#pragma unroll
            for (int c = 1; c < kCols; ++c) e = fminf(e, s[c]);
          }
          tmax = e * sl2;
        } else {
          const int lo_c = live ? bd.lo - kb - col0 : 1;
          const int hi_c = live ? bd.hi - kb - col0 : 0;
          const int64_t* const ks = kseg + (j & 1) * kPipeKeys + col0;
#pragma unroll
          for (int c = 0; c < kCols; ++c) {
            bool ok = c >= lo_c && c <= hi_c;
            if (seg) ok = ok && ks[c] == qseg;
            const float x = s[c] * sl2;
            s[c] = ok ? x : -INFINITY;
            tmax = fmaxf(tmax, s[c]);
          }
        }
        if constexpr (kHalves == 2) {
          float* const xb = xch + (j & 1) * 2 * kPipeRows;
          xb[h * kPipeRows + row] = tmax;
          NamedSync(1 + g, 64);
          tmax = fmaxf(tmax, xb[(1 - h) * kPipeRows + row]);
        }

        // Lazy rescale (identical decision in both threads of a row in the 8-warp form).
        float alpha = 1.f;
        bool need = false;
        if (tmax > -INFINITY) {
          if (m_used == -INFINITY) {
            m_used = tmax;  // first visible key of the row: its O row is exactly zero
          } else if (tmax > m_used + rescale_thr) {
            alpha = Ex2(m_used - tmax);
            m_used = tmax;
            need = true;
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
            atomicAdd(&g_pipe_rescale_events, 1ull);  // probe builds only (see TryLaunchPipe)
            if (alpha == 0.f) atomicAdd(&g_pipe_clear_events, 1ull);
#endif
          }
        }
        // A row with no visible key in this tile gets P = 0: its scores are -inf (masked path),
        // and m is then 0 so that -inf - m never meets -inf.
        const float mneg = tmax > -INFINITY ? -m_used : 0.f;
        uint32_t pk[kCols / 2];
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        // FULL rounds the exponent's argument once (fmaf), the masked path twice (s was scaled
        // above, then + mneg): not bit-equal for one score (see the header's Rounding note).
        if (full) {
#pragma unroll
          for (int c = 0; c < kCols; c += 4) {
            const float e0 = Ex2(fmaf(s[c + 0], sl2, mneg));
            const float e1 = Ex2(fmaf(s[c + 1], sl2, mneg));
            const float e2 = Ex2(fmaf(s[c + 2], sl2, mneg));
            const float e3 = Ex2(fmaf(s[c + 3], sl2, mneg));
            a0 += e0;
            a1 += e1;
            a2 += e2;
            a3 += e3;
            pk[c / 2] = PackOf<E>::Pack2(e0, e1);
            pk[c / 2 + 1] = PackOf<E>::Pack2(e2, e3);
          }
        } else {
#pragma unroll
          for (int c = 0; c < kCols; c += 4) {
            const float e0 = Ex2(s[c + 0] + mneg);
            const float e1 = Ex2(s[c + 1] + mneg);
            const float e2 = Ex2(s[c + 2] + mneg);
            const float e3 = Ex2(s[c + 3] + mneg);
            a0 += e0;
            a1 += e1;
            a2 += e2;
            a3 += e3;
            pk[c / 2] = PackOf<E>::Pack2(e0, e1);
            pk[c / 2 + 1] = PackOf<E>::Pack2(e2, e3);
          }
        }
        lsum = lsum * alpha + ((a0 + a1) + (a2 + a3));
        // P(j) in place over this thread's own (already loaded) S(j) columns.
        if constexpr (kCols == 64) {
          TmemStore32(lane_base | scol, pk);
        } else {
          St16(lane_base | (scol + col0), pk);
        }
        // O rescale: tcgen05.ld/st are warp-collective, so the warp runs it when any of its rows
        // needs it, multiplying by 1 elsewhere. O was last written by PV(j - 1); need implies an
        // earlier visible tile, so j >= 1.
        if (__any_sync(0xffffffffu, need)) {
          MbarWait(bar(Pl::kBarPv + (j - 1) % kSB), static_cast<uint32_t>(((j - 1) / kSB) & 1));
          TmemFenceAfterSync();
          const float a = need ? alpha : 1.f;
          const bool clear = need && alpha == 0.f;
#pragma unroll 1
          for (int c = 0; c < kOCols; c += 64) {
            float o[64];
            Ld64(lane_base | (tbase + Pl::kOCol + ocol0 + c), o);
            uint32_t w0[32], w1[32];
#pragma unroll
            for (int x = 0; x < 32; ++x) {
              w0[x] = __float_as_uint(clear ? 0.f : o[x] * a);
              w1[x] = __float_as_uint(clear ? 0.f : o[32 + x] * a);
            }
            TmemStore32(lane_base | (tbase + Pl::kOCol + ocol0 + c), w0);
            TmemStore32(lane_base | (tbase + Pl::kOCol + ocol0 + c + 32), w1);
          }
        }
        TmemWaitSt();
        TmemFenceBeforeSync();
        Arrive(bar(Pl::kBarP + buf));
      }

      // -------------------------------------------------------------- epilogue
      MbarWait(bar(Pl::kBarPv + (n - 1) % kSB), static_cast<uint32_t>(((n - 1) / kSB) & 1));
      TmemFenceAfterSync();
      if constexpr (kHalves == 2) {
        // Buffer n & 1 was last read in tile n - 2, before both threads passed tile n - 1's
        // barrier.
        float* const xb = xch + (n & 1) * 2 * kPipeRows;
        xb[h * kPipeRows + row] = lsum;
        NamedSync(1 + g, 64);
        lsum = lsum + xb[(1 - h) * kPipeRows + row];
      }
      const float inv = lsum > 0.f ? 1.f / lsum : 0.f;
      E* const yrow = live ? reinterpret_cast<E*>(p.y) +
                                 ((b * p.q_length + i) * p.query_heads + hq) * p.value_dim
                           : nullptr;
      const bool vec = p.value_dim == Pl::kPipeWidth;
#pragma unroll 1
      for (int c = 0; c < kOCols; c += 32) {
        float o[32];
        Ld32(lane_base | (tbase + Pl::kOCol + ocol0 + c), o);
        if (yrow != nullptr) {
          const int cc = ocol0 + c;
          if (vec) {
            union {
              uint4 u4[4];
              uint32_t w[16];
            } pk;
#pragma unroll
            for (int x = 0; x < 16; ++x) pk.w[x] = PackOf<E>::Pack2(o[2 * x] * inv, o[2 * x + 1] * inv);
            uint4* const dst = reinterpret_cast<uint4*>(yrow + cc);
            dst[0] = pk.u4[0];
            dst[1] = pk.u4[1];
            dst[2] = pk.u4[2];
            dst[3] = pk.u4[3];
          } else {
            for (int x = 0; x < 32; ++x) {
              if (cc + x >= p.value_dim) break;
              yrow[cc + x] = PackOf<E>::From(o[x] * inv);
            }
          }
        }
      }
      if (live) {
        if (h == 0 && p.lse != nullptr) {
          p.lse[(b * p.query_heads + hq) * p.q_length + i] =
              lsum > 0.f ? 0.6931471805599453f * m_used + __logf(lsum) : -INFINITY;
        }
      } else {
        write_dead(h == 0);
      }
    }
  }

  // ------------------------------------------------------------------ exit
  TmemFenceBeforeSync();
  __syncthreads();
  TmemFenceAfterSync();
  if (warp == 0) TmemDealloc(tbase, Pl::kAllocCols);
}

// ---------------------------------------------------------------- host
// The compiled configuration: 8 softmax warps (two threads per query row, 32 S columns and
// 128 O columns each), 2 K stages, 2 S buffers.
inline constexpr int kSmWarpsShipped = 8;
inline constexpr int kKStagesShipped = 2;
inline constexpr int kSBuffersShipped = 2;
// The lazy-rescale threshold in log2 units: a row's O and l are rescaled only when a tile's
// maximum exceeds the maximum they were last scaled to by more than this, so P <= 2^8.
inline constexpr float kRescaleThreshold = 8.f;

// Called by LaunchUmmaForward<E, 256, 256> and <E, 128, 128> after their own refusals (split-map alignment, device
// capability), for the body host.cc resolved for this call (ForwardBodySlot, set once per call;
// the environment is not read here). Returns
//   0  not taken: the call resolved the older leaf (XATTN_SCA_UMMA_FWD=1), and the caller launches
//      ForwardKernel unchanged;
//   1  launched (PipeLaunchCount moved);
//   2  refused BEFORE the launch call, with nothing enqueued: a call PipeAccepts does not admit
//      (host.cc resolves none such to kPipe, so this is a guard), a shared-memory opt-in the device
//      does not grant, or a failed attribute call. The caller records the refusal and the generic
//      body carries the call (dispatch_tag.cuh's two-row fallback).
// A launch call that reports an error does not return: it raises (ThrowPostAttemptError) with the
// CUDA error's name, code and text, and no other body runs over the call's outputs. One
// precondition is checked on the device and nowhere on the host: the 1024-byte alignment of the
// dynamic SMEM base that the UMMA descriptors (base_offset 0) need. The host cannot see the address
// a launch will get, so the kernel traps on it - a device fault that poisons the context, not a
// refusal and not a fallback.
template <class E>
int TryLaunchPipe(const UmmaLaunchParams& lp, cudaStream_t stream) {
  PostAttemptErrorSlot() = cudaSuccess;
  if (ForwardBodySlot() != ForwardBody::kPipe) return 0;
  const Params& hp = lp.params;
  if (!PipeAccepts(hp)) return 2;
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
  // Probe builds only: a pre-launch refusal, so that the caller's refusal row and generic fallback
  // can be observed (a healthy sm_103 device grants every check below).
  if (ProbeHook("pipe-refuse")) return 2;
#endif
  // The bucket decides which body runs: PipeAccepts admits exactly D = Dv = 256 and D = Dv = 128,
  // and the width is a compile-time constant of the plan, so this is the one place the choice is
  // made. Both bodies are instantiated here, which is what makes the launcher width-general with
  // no second launch site; every other constant (slab count, tile bytes, SMEM, threads) follows.
  const bool wide = hp.qk_dim == 256;
  using Pl = Plan<kSmWarpsShipped, kKStagesShipped, kSBuffersShipped, 256>;
  using Pl128 = Plan<kSmWarpsShipped, kKStagesShipped, kSBuffersShipped, 128>;
  const int smem_bytes = wide ? Pl::kSmemBytes : Pl128::kSmemBytes;
  const int threads = wide ? Pl::kThreads : Pl128::kThreads;
  int device = 0, cap = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) {
    (void)cudaGetLastError();
    return 2;
  }
  if (smem_bytes > cap) return 2;
  const auto kernel = wide
      ? &PipeForwardKernel<E, kSmWarpsShipped, kKStagesShipped, kSBuffersShipped, 256>
      : &PipeForwardKernel<E, kSmWarpsShipped, kKStagesShipped, kSBuffersShipped, 128>;
  if (cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) != cudaSuccess) {
    (void)cudaGetLastError();
    return 2;
  }
  dim3 grid(static_cast<unsigned>((hp.q_length + kPipeRows - 1) / kPipeRows),
            static_cast<unsigned>(hp.query_heads), static_cast<unsigned>(hp.batch));
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
  // Fault control, probe builds only: an empty grid, which the runtime rejects AT the launch call
  // (cudaErrorInvalidValue under CUDA 13.4) without enqueueing work or faulting the device, so the
  // post-attempt path below runs on a healthy context.
  if (ProbeHook("pipe-posterr")) grid.x = 0u;
#endif
  // Drop any error latched before this point, so the status read after the launch call is that
  // launch's own (LaunchUmmaForward drains at its entry too; the attribute calls consume theirs).
  (void)cudaGetLastError();
  kernel<<<grid, threads, smem_bytes, stream>>>(lp, kRescaleThreshold);
  const cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    PostAttemptErrorSlot() = launch_err;
    ThrowPostAttemptError(wide ? "the pipelined (256,256) body (fwdpipe::PipeForwardKernel)"
                               : "the pipelined (128,128) body (fwdpipe::PipeForwardKernel)");
  }
  PipeLaunchCount().fetch_add(1, std::memory_order_relaxed);
#if defined(XATTN_FWD_PROBE_HOOKS) && XATTN_FWD_PROBE_HOOKS
  {
    // Probe builds only: the cumulative lazy-rescale and clear counts of this element type after
    // the launch.
    unsigned long long events = 0, clears = 0;
    if (cudaStreamSynchronize(stream) == cudaSuccess &&
        cudaMemcpyFromSymbol(&events, g_pipe_rescale_events, sizeof(events)) == cudaSuccess &&
        cudaMemcpyFromSymbol(&clears, g_pipe_clear_events, sizeof(clears)) == cudaSuccess) {
      std::fprintf(stderr, "XATTN_FWD_PIPE_RESCALE_EVENTS %llu CLEAR_EVENTS %llu\n", events, clears);
    } else {
      (void)cudaGetLastError();
      std::fprintf(stderr, "XATTN_FWD_PIPE_RESCALE_EVENTS unreadable\n");
    }
  }
#endif
  return 1;
}

}  // namespace fwdpipe
}  // namespace umma
}  // namespace xattn::ops::flash_sca_native

#define XATTN_FWD_PIPE_INSTANTIATE(E)                                                   \
  template int xattn::ops::flash_sca_native::umma::fwdpipe::TryLaunchPipe<E>(           \
      const xattn::ops::flash_sca_native::umma::UmmaLaunchParams&, cudaStream_t);
