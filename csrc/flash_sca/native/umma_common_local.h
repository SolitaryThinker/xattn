// TEMPORARY pinned interface copy of the shared UMMA/TMEM primitives.
//
// The canonical file is `flash_sca/native/umma_common.cuh`. It is not in
// this tree yet, so `umma_fwd.cuh` includes this pinned copy EXPLICITLY - one
// unconditional include, no `__has_include` and no automatic switch-over: this file has
// no way to notice that the shared header has landed. The port is an include swap plus
// deleting this file in the same commit (the shared header takes this
// file's 4-D load body verbatim; the 4-D and 5-D spellings in play assemble
// to the same UTMALDG.4D, so no encoder is owed), and until then the guard below makes a
// TU that sees both primitive sets fail to compile rather than silently pick one.
// Nothing here may drift from the shared header: if a signature or a descriptor constant
// changes there, it changes here in the same commit, and this file is deleted.
//
// Every asm is arch-conditional: below sm_100 each body is empty, so the header
// parses in any compilation pass and the host dispatch decides reachability
// (capability 10.3 only). No host code, no CuTe, no CUTLASS include.
#pragma once

// Mutual exclusion with the shared set (`flash_sca/native/umma_common.cuh`).
// Both headers define the same primitives with different bodies and signatures, so a TU
// that sees both gets one definition per name from whichever header came first - a
// silent switch umma_fwd.cuh must not have. Fail the build instead of picking.
// The reciprocal guard belongs in the shared header: it must
// define XATTN_UMMA_COMMON_SHARED_INCLUDED and #error on XATTN_UMMA_COMMON_LOCAL_INCLUDED.
#if defined(XATTN_UMMA_COMMON_SHARED_INCLUDED)
#error "umma_common_local.h and umma_common.cuh are mutually exclusive: include exactly one primitive set"
#endif
#define XATTN_UMMA_COMMON_LOCAL_INCLUDED 1

#include <cstdint>

#include <cuda_runtime_api.h>

// Arch guard for every tcgen05 / mbarrier / TMA / fence asm in this pair of
// headers. Two reasons it is a 0/1 macro defined here and not the expression
// `(defined(__CUDA_ARCH__) && (__CUDA_ARCH__) >= 1000)` pasted into each `#if`:
//   (1) `defined` produced by macro expansion is undefined behaviour in the
//       preprocessor - it belongs in the `#if` line itself, never inside a
//       macro's body that an `#if` then expands;
//   (2) the family bound matters. sm_120 (consumer Blackwell, __CUDA_ARCH__ 1200)
//       satisfies `>= 1000` and has no tcgen05 at all, so an open upper bound
//       arms this asm on an arch where it cannot assemble.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) >= 1000 && (__CUDA_ARCH__) < 1200
#define XATTN_UMMA_DEVICE_ASM 1
#else
#define XATTN_UMMA_DEVICE_ASM 0
#endif

namespace xattn::ops::flash_sca_native {
namespace umma {

__device__ __forceinline__ uint32_t SmemAddr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- descriptors
// SMEM operand descriptor. Units are 16 B; layout_type 2 is SWIZZLE_128B.
struct SmemDesc {
  uint64_t value;
};

__device__ __forceinline__ SmemDesc MakeSmemDesc(uint32_t smem_addr, uint32_t lbo_u128 = 1,
                                                 uint32_t sbo_u128 = 64, uint32_t layout_type = 2,
                                                 uint32_t base_offset = 0) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFFu);
  d |= static_cast<uint64_t>(lbo_u128 & 0x3FFFu) << 16;
  d |= static_cast<uint64_t>(sbo_u128 & 0x3FFFu) << 32;
  d |= static_cast<uint64_t>(base_offset & 0x7u) << 49;
  d |= static_cast<uint64_t>(1) << 46;  // version 1
  d |= static_cast<uint64_t>(layout_type & 0x7u) << 61;
  return SmemDesc{d};
}

__device__ __forceinline__ SmemDesc AdvanceDesc(SmemDesc d, uint32_t u128_offset) {
  const uint32_t start = static_cast<uint32_t>(d.value & 0x3FFFu) + u128_offset;
  d.value &= ~static_cast<uint64_t>(0x3FFFu);
  d.value |= static_cast<uint64_t>(start & 0x3FFFu);
  return d;
}

// Instruction descriptor. kElemBf16 = 1, kElemF16 = 0.
__device__ __forceinline__ uint32_t MakeInstrDesc(int elem, int m, int n, int a_major,
                                                  int b_major) {
  uint32_t d = 0;
  d |= 1u << 4;                                // c_format: f32
  d |= static_cast<uint32_t>(elem) << 7;        // a_format
  d |= static_cast<uint32_t>(elem) << 10;       // b_format
  d |= static_cast<uint32_t>(a_major) << 15;
  d |= static_cast<uint32_t>(b_major) << 16;
  d |= static_cast<uint32_t>((n >> 3) & 0x3F) << 17;
  d |= static_cast<uint32_t>((m >> 4) & 0x1F) << 24;
  return d;
}

// ---------------------------------------------------------------- barriers
__device__ __forceinline__ void MbarInit(uint32_t mbar, int count) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count) : "memory");
#endif
}

__device__ __forceinline__ void MbarInitFence() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

__device__ __forceinline__ void MbarArriveExpectTx(uint32_t mbar, int bytes) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes)
               : "memory");
#endif
}

__device__ __forceinline__ bool MbarTryWait(uint32_t mbar, uint32_t phase) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t ok;
  asm volatile("{\n\t.reg .pred P;\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
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

// Hardware-suspended parity wait: mbarrier.try_wait.parity suspends the warp in hardware and wakes
// immediately on the exact cycle the mbarrier phase flips, avoiding nanosleep overshoot and spin
// counter registers on hot warp-specialized UMMA loops.
__device__ __forceinline__ void MbarWait(uint32_t mbar, uint32_t phase) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "{\n\t"
      ".reg .pred P;\n\t"
      "LAB_MBAR_WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra.uni LAB_MBAR_WAIT_%=;\n\t"
      "}\n"
      :: "r"(mbar), "r"(phase)
      : "memory");
#else
  (void)mbar;
  (void)phase;
#endif
}

// Needed only when generic stores wrote SMEM that an MMA reads (P/dS in SMEM).
__device__ __forceinline__ void FenceAsyncShared() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

// ---------------------------------------------------------------- UMMA
__device__ __forceinline__ bool ElectOne() {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t pred = 0;
  asm volatile("{\n\t.reg .pred P;\nelect.sync _|P, 0xffffffff;\nselp.b32 %0, 1, 0, P;\n}\n"
               : "=r"(pred));
  return pred != 0;
#else
  return false;
#endif
}

// tcgen05.mma is a single-thread instruction: one elected lane of one warp.
__device__ __forceinline__ void MmaSs(uint32_t dtmem, SmemDesc a, SmemDesc b, uint32_t idesc,
                                      uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  if (!ElectOne()) return;
  uint32_t z = 0;
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n" ::"r"(dtmem),
      "l"(a.value), "l"(b.value), "r"(idesc), "r"(accumulate), "r"(z), "r"(z), "r"(z), "r"(z));
#endif
}

__device__ __forceinline__ void MmaTs(uint32_t dtmem, uint32_t atmem, SmemDesc b, uint32_t idesc,
                                      uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  if (!ElectOne()) return;
  uint32_t z = 0;
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n\t}\n" ::"r"(dtmem),
      "r"(atmem), "l"(b.value), "r"(idesc), "r"(accumulate), "r"(z), "r"(z), "r"(z), "r"(z));
#endif
}

// Completion of the preceding tcgen05.mma of this CTA into `mbar`.
__device__ __forceinline__ void MmaCommit(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  if (ElectOne()) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar)
        : "memory");
  }
#endif
}

// Raw single-thread UMMA / TMEM copy / commit helpers (caller already elected one lane;
// omits the optional all-zero disable-output-lane vector and takes precomputed 64-bit SMEM
// descriptors directly).
__device__ __forceinline__ void MmaSsRaw(uint32_t dtmem, uint64_t a, uint64_t b, uint32_t idesc,
                                         uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(dtmem),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate));
#else
  (void)dtmem;
  (void)a;
  (void)b;
  (void)idesc;
  (void)accumulate;
#endif
}

__device__ __forceinline__ void MmaTsRaw(uint32_t dtmem, uint32_t atmem, uint64_t b,
                                         uint32_t idesc, uint32_t accumulate) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(dtmem),
      "r"(atmem), "l"(b), "r"(idesc), "r"(accumulate));
#else
  (void)dtmem;
  (void)atmem;
  (void)b;
  (void)idesc;
  (void)accumulate;
#endif
}

__device__ __forceinline__ void MmaCommitRaw(uint32_t mbar) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar)
      : "memory");
#else
  (void)mbar;
#endif
}

__device__ __forceinline__ void TmemCopy128x256bRaw(uint32_t taddr, uint64_t src_desc) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" ::"r"(taddr), "l"(src_desc)
               : "memory");
#else
  (void)taddr;
  (void)src_desc;
#endif
}

// ---------------------------------------------------------------- TMEM
// TMEM address: upper 16 bits = 128 DP lanes, lower 16 bits = 512 columns.
__device__ __forceinline__ uint32_t TmemAddr(uint32_t base, uint32_t lane, uint32_t col) {
  return base + (lane << 16) + col;
}

__device__ __forceinline__ void TmemAlloc(uint32_t cols, uint32_t* dst_smem) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                   SmemAddr(dst_smem)),
               "r"(cols)
               : "memory");
#endif
}

__device__ __forceinline__ void TmemRelinquish() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
#endif
}

// Warp-wide, by all 32 threads of the warp that allocated.
__device__ __forceinline__ void TmemDealloc(uint32_t taddr, uint32_t cols) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols)
               : "memory");
#endif
}

__device__ __forceinline__ void TmemWaitSt() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void TmemWaitLd() {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

// 32dp32b.x32: thread laneid reads accumulator row (32*warp + laneid), 32
// consecutive columns from the address column field. Address is warp-uniform.
__device__ __forceinline__ void TmemLoad32(uint32_t addr, float (&v)[32]) {
#if XATTN_UMMA_DEVICE_ASM
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
        "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(addr));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
#else
  (void)addr;
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = 0.f;
#endif
}

__device__ __forceinline__ void TmemStore32(uint32_t addr, const uint32_t (&v)[32]) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};" ::"r"(v[0]), "r"(v[1]),
      "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]),
      "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]),
      "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]),
      "r"(v[26]), "r"(v[27]), "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31]), "r"(addr));
#else
  (void)addr;
  (void)v;
#endif
}

// ---------------------------------------------------------------- TMA
// One box of the 4D map: (column, head, row, batch). Destination must be 128 B
// aligned; the box width is kTmaColumns 16-bit elements under SWIZZLE_128B.
__device__ __forceinline__ void TmaLoad4D(void* dst, const CUtensorMap* map, uint32_t mbar, int c0,
                                          int c1, int c2, int c3) {
#if XATTN_UMMA_DEVICE_ASM
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];" ::"r"(SmemAddr(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(mbar), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
#endif
}

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
