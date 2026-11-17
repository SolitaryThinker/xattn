// Sliding-chunk attention on sm_103a: TMA operand loads, shared-memory operand
// layout and UMMA (tcgen05) shared-memory descriptors.
//
// Device side only: the host builds the CUtensorMap (umma_host.h, EncodeInputMap)
// with the fold [logicalD, H, L, B], box [64, 1, box_rows, 1], SWIZZLE_128B.
// This header says which bytes such a load writes where, and the descriptor
// fields that make those bytes a UMMA operand.
//
// What is asserted here and nothing more: the byte image of one 64-column
// 128B-swizzled atom; that a (rows x 64) atom is both a K-major and an MN-major
// operand with the descriptor fields below; that the descriptors equal what the
// pinned CUTLASS UMMA descriptor builder produces for the K-major case, rows 64
// and 128. Past a logical D the map zero-fills (measured at D = 40 and D = 200),
// so a padded atom is exact; the key extent is never a fill and stays the
// caller's predicate. Not asserted, and not to be relied on by a caller: the
// MN-major descriptor above a single 64-MN group, and any bandwidth number.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda.h>  // CUtensorMap lives in the driver header
#endif

#if defined(__CUDACC__)
#define XATTN_UMMA_HD __host__ __device__
#else
#define XATTN_UMMA_HD
#endif

// A contract check that survives the shipped build. setup.py compiles the
// extension with -DNDEBUG, so assert() is dead code in the .so the gate tests;
// this one traps in every configuration and costs one predicate per tile, on a
// path that runs once per load. The trap must be the legal one for the pass that
// compiles it: __builtin_trap() is a host builtin and nvcc rejects it inside a
// __device__ function, __trap() is device-only. __CUDA_ARCH__ is defined only in
// the device pass, and every use site here is device-visible, so the split
// covers both.
//
// The names carry the TMA prefix on purpose: a #define is
// not namespaced, umma_common.cuh defines XATTN_UMMA_CHECK with a different
// arity, and a leaf must include both headers - the loads live here and the
// MMA/TMEM primitives there. Same name, two arities, is a macro redefinition in
// exactly the TU that matters. The successor shape is that header's
// UmmaCheck(ok, code, mailbox): a trap with a code and a host-mapped mailbox
// says which check fired and keeps the buffers a host would have read, and a
// leaf then has one check mechanism rather than two. That moves with the adapter
// swap, not before it, so that no TU sees both forms.
#if defined(__CUDA_ARCH__)
#define XATTN_TMA_TRAP() __trap()
#else
#define XATTN_TMA_TRAP() __builtin_trap()
#endif
#define XATTN_TMA_CHECK(cond) \
  do {                        \
    if (!(cond)) XATTN_TMA_TRAP(); \
  } while (0)

namespace xattn::ops::flash_sca_native::umma {

// TEMPORARY ADAPTER - DELETE ON MERGE OF THE SHARED PRIMITIVE HEADER.
//
// The barrier and descriptor functions further down (MBarrierInit,
// MBarrierInitPublish, MBarrierInitSingleArrival, MBarrierInvalidate,
// MBarrierArriveExpectTx, MBarrierTryWaitParity, MBarrierWait,
// MBarrierWaitBounded, FenceAsyncProxy, MakeSmemDescriptor, SmemLayoutType)
// duplicate what csrc/flash_sca/native/umma_common.cuh owns. They are here
// because that file is not in main yet, and because a leaf must see exactly one
// definition of each: do not include both headers in one translation unit.
// The swap, once umma_common.cuh is merged, is mechanical:
//
//   MBarrierInit(a, c)            -> MbarInit(a, c)
//   MBarrierArriveExpectTx(a, b)  -> MbarArriveExpectTx(a, b)
//   MBarrierTryWaitParity/Wait    -> MbarWait(a, phase)
//   MakeSmemDescriptor(...)       -> MakeSmemDesc(smem, lbo, sbo, layout, 0).value
//   CheckOperandBase stays here   (the operand-base obligation is this file's)
// and AtomDescKMajor/AtomDescMNMajor stay as the one-line wrappers they are.

// ---------------------------------------------------------------------------
// Physical tiling of one operand tile in shared memory.
//
// A tile is (rows x Dpad) 16-bit elements, made of Dpad/64 atoms of
// (rows x 64) elements. One TMA box writes exactly one atom: 64 columns is the
// largest inner box a 128B swizzle accepts (the driver rejects boxDim[0] = 128
// and 256 under SWIZZLE_128B with CUDA_ERROR_INVALID_VALUE), and the atom row
// is exactly one swizzle span (128 B).
//
//   atom bytes            = rows * 128
//   element (row, col)    = atom*rows*128 + swizzle(row*128 + col*2)
//   swizzle(b)            = b ^ (((b >> 7) & 7) << 4)   // 16B chunk index x row
//
// so column 0..63 of a row stays inside the atom's 128B row and the D loop of
// an MMA walks atoms, not columns.
//
// Two loops have to be kept apart by a consumer: the contraction inside one
// atom is K = 64, which a 16-bit tcgen05.mma covers only 16 columns at a time,
// so an MMA steps the shared-memory descriptor by +2 (16 B units = 32 B) per
// k-step, four steps per atom with the first not accumulating; the step to the
// next atom along D is +rows*8 u128 (+rows*128 B) and is a different
// descriptor, not a stride.
//
// Caller obligation, and it is absolute rather than relative: the 128 B swizzle
// permutes by the absolute shared-memory address, so the base of every operand
// tile must be 1024 B aligned. A stage ring therefore advances the tile base by
// multiples of 1024 B (rows a multiple of 8 at 128 B rows), not by the atom size
// alone. MakeSmemDescriptor checks the weaker of the two conditions - 128 B, the
// granularity its start_address field can express - so passing this header does
// not by itself prove the stage base is right.
// ---------------------------------------------------------------------------
constexpr int kSwizzleBytes = 128;   // SWIZZLE_128B
constexpr int kAtomCols = 64;        // 64 x 16-bit = 128 B = one swizzle span
constexpr int kElementBytes = 2;     // fp16 / bf16 only on this path
constexpr int kU128Bytes = 16;       // descriptor offsets drop 4 LSB == 16 B units
constexpr int kBoxRows = 64;         // rows per TMA box: EncodeInputMap's kTmaRows

// Smallest of {64, 128, 192, 256} that holds d. The tile is padded to it; the
// map's dim0 stays the logical d, so the box reaches past the logical extent and
// the map zero-fills there (measured at D = 40 and D = 200), which is what makes
// a padded atom exact. The key extent is never a fill and stays the caller's
// predicate.
XATTN_UMMA_HD constexpr int PadDim(int64_t d) {
  return d <= 64 ? 64 : d <= 128 ? 128 : d <= 192 ? 192 : 256;
}
XATTN_UMMA_HD constexpr int DAtoms(int dpad) { return dpad / kAtomCols; }
XATTN_UMMA_HD constexpr int AtomBytes(int rows) { return rows * kSwizzleBytes; }
XATTN_UMMA_HD constexpr int TileBytes(int rows, int dpad) { return AtomBytes(rows) * DAtoms(dpad); }
// Bytes one TmaLoadTile of (rows x datoms atoms) transmits: the value the caller
// arrives with. Box-rounded, because the loader issues whole boxes, so it is
// exact for every row count and equals AtomBytes(rows)*datoms whenever rows is a
// multiple of box_rows. Defined for ANY row count on purpose - it is the
// arithmetic a rank-4 caller re-deriving the sum needs - but TmaLoadTile itself
// accepts only multiples of box_rows: a partial box would both write past its
// atom and make the count something no partial sum of the box deliveries
// reaches, and that never completes (measured: 64512 of 65536 spins to the
// bounded timeout in 68 ms). So `TileTxBytes(72, 4) == 65536` in the static
// asserts below is a statement about this function, NOT support for a 72-row
// tile.
XATTN_UMMA_HD constexpr int TileTxBytes(int rows, int datoms, int box_rows = kBoxRows) {
  return ((rows + box_rows - 1) / box_rows) * box_rows * kSwizzleBytes * datoms;
}
// Smallest row count >= rows that a whole number of TMA boxes covers, so a caller
// sizing a stage from a logical BN never lands between two boxes.
XATTN_UMMA_HD constexpr int RowsForBoxes(int rows, int box_rows = kBoxRows) {
  return ((rows + box_rows - 1) / box_rows) * box_rows;
}
// Rows a tile actually needs: ceil(rows/8)*8 so that the last 8-row UMMA core
// matrix is complete (the tail is zero filled by TMA / masked by the predicate).
XATTN_UMMA_HD constexpr int RowsCeil(int rows) { return (rows + 7) / 8 * 8; }

#if defined(__CUDACC__)

// Byte offset of element (row, col) of atom `atom` in a (rows x Dpad) tile.
XATTN_UMMA_HD __forceinline__ int SmemElementOffset(int atom, int rows, int row, int col) {
  const int in_atom = row * kSwizzleBytes + (col & (kAtomCols - 1)) * kElementBytes;
  return atom * AtomBytes(rows) + (in_atom ^ (((in_atom >> 7) & 7) << 4));
}

__device__ __forceinline__ uint32_t SmemU32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------
// mbarrier used by the TMA loads (arrive+expect_tx on the issuing thread,
// parity wait on the consumers).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void MBarrierInit(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(count));
}
// The init fence: this one instruction, after MBarrierInit and before the first
// arrive.expect_tx or TMA that targets the barrier. FenceAsyncProxy() is not a
// substitute - it orders the generic and async proxies over memory, it does not
// publish the barrier's initial state, and omitting the publish gives a kernel
// that hangs with no output on one code path and passes on another.
__device__ __forceinline__ void MBarrierInitPublish() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
// A tile loader deposits a whole tile into one barrier phase: exactly one
// arrival per phase, and the initial state must be published. Folding the
// publish in here is deliberate - this is the form every tile loader uses, so it
// cannot be forgotten; a barrier with a different arrival count calls
// MBarrierInit and MBarrierInitPublish itself. A per-atom arrival against this
// count hangs the phase.
__device__ __forceinline__ void MBarrierInitSingleArrival(uint32_t addr) {
  MBarrierInit(addr, 1);
  MBarrierInitPublish();
}
// Before the storage of an invalidated barrier is reused for anything else.
// Harmless while a kernel exits after its last wait; undefined as soon as a
// stage buffer doubles as an operand buffer.
__device__ __forceinline__ void MBarrierInvalidate(uint32_t addr) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(addr) : "memory");
}
// Required before an MMA consumes shared memory that threads (not TMA) wrote,
// e.g. a P tile staged through smem in the backward; not needed after a TMA
// load, whose completion is already visible through the mbarrier.
__device__ __forceinline__ void FenceAsyncProxy() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void MBarrierArriveExpectTx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr),
               "r"(bytes));
}
__device__ __forceinline__ bool MBarrierTryWaitParity(uint32_t addr, uint32_t phase) {
  uint32_t done;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.b32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(done)
      : "r"(addr), "r"(phase));
  return done != 0;
}
__device__ __forceinline__ void MBarrierWait(uint32_t addr, uint32_t phase) {
  while (!MBarrierTryWaitParity(addr, phase)) {
  }
}
// Same wait with a bound, for bring-up: returns false if the phase did not
// complete within `spins` tries. A parity or arrive-count mistake otherwise
// looks like a hang with no information at all.
__device__ __forceinline__ bool MBarrierWaitBounded(uint32_t addr, uint32_t phase, int spins) {
  for (int i = 0; i < spins; ++i)
    if (MBarrierTryWaitParity(addr, phase)) return true;
  return false;
}

// ---------------------------------------------------------------------------
// TMA load of one atom: cp.async.bulk.tensor.4d, coordinates (d0, h, row0, b)
// into the map built by EncodeInputMap. Completion is by byte count, so the
// caller arrives once with AtomBytes(rows)*datoms for a whole tile.
// `map` must reach this function as the __grid_constant__ kernel parameter (or
// from constant/global memory). Passed by value it is copied to the thread's
// stack frame and the load is issued from that copy - 15 x STL.128 with
// STACK:368 in SASS instead of LDCU.64 from c[0x0] - and a map outside
// const/global memory is not a legal descriptor operand. `cuobjdump -res-usage`
// (STACK > 0 plus STL next to the load) tells the two apart on a cubin.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void TmaLoadAtom(const CUtensorMap* map, uint32_t dst_smem,
                                            uint32_t mbar, int d0, int h, int row0, int b) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];\n" ::"r"(dst_smem),
      "l"(map), "r"(mbar), "r"(d0), "r"(h), "r"(row0), "r"(b)
      : "memory");
}

// ---------------------------------------------------------------------------
// TMA store of one atom: cp.async.bulk.tensor.4d.global.shared::cta.bulk_group, coordinates
// (d0, h, row0, b) through the same map EncodeInputMap builds (a tensor map is direction
// agnostic; the extent is the logical one, so a box that reaches past it is clipped by the TMA -
// rows past q_length and padded Dv columns are simply not written). The SMEM source must be in
// the map's declared 128 B swizzle (SwizzleBytes below), and the caller must have fenced the
// generic writes with FenceProxyAsyncShared before the barrier that precedes this call.
//
// Completion is by bulk group, not by mbarrier: TmaStoreCommit closes a group and
// TmaStoreWaitRead waits until the SMEM it reads may be reused. Writing to global is complete for
// the kernel's consumer when the kernel ends, so the epilogue waits on `.read`, not on the write.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void TmaStoreAtom(const CUtensorMap* map, uint32_t src_smem, int d0,
                                             int h, int row0, int b) {
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];\n" ::
          "l"(map),
      "r"(src_smem), "r"(d0), "r"(h), "r"(row0), "r"(b)
      : "memory");
}

__device__ __forceinline__ void TmaStoreCommit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Waits until every issued bulk-group read of shared memory is complete, i.e. until the staged
// bytes may be overwritten or the CTA may exit without the copy engine reading a dead stage.
__device__ __forceinline__ void TmaStoreWaitRead() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Generic-proxy shared writes made visible to the async proxy (the TMA engine), then ordered
// against the barrier that follows: this is the one fence a staged copy needs on the producer side
// and it must precede the named barrier, not follow it.
__device__ __forceinline__ void FenceProxyAsyncShared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// A stage that carries more than one tile into one barrier arrives once with the
// sum of their TileTxBytes (the dK leaf stages P and dO that way).
//
// Whole tile: Dpad/64 atoms of (rows x 64), atom a at a*AtomBytes(rows), and
// rows/box_rows boxes per atom (EncodeInputMap fixes box_rows = kTmaRows = 64,
// so a 128-row resident tile is two boxes). One arrive+expect_tx per tile, with
// exactly TileTxBytes(rows, datoms, box_rows) bytes, one mbarrier phase.
//
// Tail policy: the map's extent is the logical one, so a box may reach past it.
// Fill is not masking - the caller excludes keys >= Nt in the softmax predicate
// and predicates gradient writes, whatever lands there. The D padding columns
// are a separate question that this header does not answer: a padded atom is
// exact only if those columns are zero, so a caller must either confirm that or
// zero them itself. rows % box_rows == 0 is required (resident rows 64 or 128).
//
// For a caller that issues rank 4 itself instead of calling this function, the
// two things to reproduce are:
//  - the column coordinate is per slab, d0 = slab * 64 (the box is 64 columns
//    wide, so slab s of a Dpad-wide operand starts at 64*s, and rows always
//    start at the same row0 for every slab);
//  - the barrier accounting is per stage, in bytes, not per box: one
//    arrive.expect_tx with TileTxBytes(rows, datoms, box_rows), i.e. exactly the
//    sum of every box the stage will deliver, issued before the first load.
//
// The three ways the count and the issued boxes can disagree do NOT fail the
// same way, and only one of them is quiet - keep them apart when reading a log:
//   expected > delivered  -> the phase never completes (measured: an expectation of
//                            66560 against 65536 delivered spins to a bounded
//                            timeout; the 64512-of-65536 arm behaved the same);
//   expected < delivered  -> the barrier releases the consumer once the expected
//                            bytes have landed WHILE the surplus boxes are still
//                            writing that stage, and the surplus bytes are
//                            charged to the next phase: a silent race that shows
//                            up as a wrong tile, not as a hang;
//   expected == delivered -> one completion per phase, the only accepted case.
// "One box too many" above means the second row: boxes ISSUED beyond the
// expectation, not an expectation set too high. TileTxBytes removes the case for
// callers of TmaLoadTileAndArrive, because the count and the issue loop are
// computed from the same three arguments and the box height is pinned to the
// encoder's kBoxRows; it cannot remove it for a caller that issues its own
// boxes, which is why the rule is written here.
__device__ __forceinline__ void TmaLoadTile(const CUtensorMap* map, uint32_t tile_smem,
                                           uint32_t mbar, int h, int row0, int b, int rows,
                                           int datoms, int box_rows) {
  // A partial box both writes past its atom and makes expect_tx a count that is
  // not a partial sum of the box deliveries, which never completes. Traps rather
  // than asserts: the shipped build is compiled with -DNDEBUG.
  XATTN_TMA_CHECK(box_rows > 0 && rows % box_rows == 0);
  // box_rows is also pinned to the encoder's box height here, because the two
  // have to be the same number and nothing else ties them: EncodeInputMap fixes
  // boxDim[2] = kBoxRows = 64, so a caller that asks for box_rows = 128 issues
  // one 64-row box per atom (half the bytes for the box) against a doubled
  // expect_tx - and rows % box_rows == 0 does not fire for it, so the stage
  // simply never completes. This is the one guard the row parity cannot cover.
  XATTN_TMA_CHECK(box_rows == kBoxRows);
  for (int a = 0; a < datoms; ++a) {
    const uint32_t atom = tile_smem + (uint32_t)(a * AtomBytes(rows));
    for (int r = 0; r < rows; r += box_rows)
      TmaLoadAtom(map, atom + (uint32_t)(r * kSwizzleBytes), mbar, a * kAtomCols, h, row0 + r, b);
  }
}

// The form to use unless a stage carries several tiles into ONE barrier: arrive
// with the byte count the same arguments imply, then issue. Count and issue
// cannot drift apart because neither is a parameter. Leader form below.
__device__ __forceinline__ void TmaLoadTileAndArrive(const CUtensorMap* map, uint32_t tile_smem,
                                                    uint32_t mbar, int h, int row0, int b,
                                                    int rows, int datoms, int box_rows) {
  MBarrierArriveExpectTx(mbar, (uint32_t)TileTxBytes(rows, datoms, box_rows));
  TmaLoadTile(map, tile_smem, mbar, h, row0, b, rows, datoms, box_rows);
}

// ---------------------------------------------------------------------------
// UMMA shared-memory descriptor (mma_sm100_desc.hpp field layout). The K-major
// numbers below reproduce cute::UMMA::make_umma_desc<Major::K> bit for bit on
// the pinned CUTLASS for rows 64 and 128; the MN-major stacking is derived and
// agrees with an independent derivation, and is confirmed by a PV product, not
// by this comment.
//
//   canonical form for a 128B swizzle:
//
//   K-major  ((8,n),2):((8,SBO),1)      -> LBO = 1,  SBO = 8 rows * 128B / 16
//   MN-major ((8,n),(8,k)):((1,LBO),(8,SBO)) -> LBO = next atom / 16, SBO = 8 rows * 128B / 16
//
// i.e. both are the same physical atom: the D-contiguous (rows x 64) tile our
// TMA writes is a legal K-major operand (contraction along the 64 columns) and
// a legal MN-major operand (contraction along rows, e.g. P^T for dV).
//
// start_address: 14 bits, 4 LSB dropped (16 B units)
// leading_byte_offset [16,30), stride_byte_offset [32,46), base_offset [49,52),
// lbo_mode [52], layout_type [61,64), version [46,48) = 1 on Blackwell.
// ---------------------------------------------------------------------------
// Leader-only form of the tile load. TmaLoadTile is per-thread and deposits
// completion bytes into one barrier, so calling it from every thread multiplies
// the transmitted bytes against a count-1 barrier.
__device__ __forceinline__ void TmaLoadTileLeader(const CUtensorMap* map, uint32_t tile_smem,
                                                  uint32_t mbar, int h, int row0, int b, int rows,
                                                  int datoms, int box_rows) {
  if (threadIdx.x == 0) TmaLoadTile(map, tile_smem, mbar, h, row0, b, rows, datoms, box_rows);
}

// Same, arriving with the byte count: the one a stage of one tile should call.
// One thread both arrives and issues - an arrive.expect_tx from every thread of
// a warp is a different arrival count and breaks a single-arrival barrier.
__device__ __forceinline__ void TmaLoadTileLeaderAndArrive(
    const CUtensorMap* map, uint32_t tile_smem, uint32_t mbar, int h, int row0, int b, int rows,
    int datoms, int box_rows) {
  if (threadIdx.x == 0)
    TmaLoadTileAndArrive(map, tile_smem, mbar, h, row0, b, rows, datoms, box_rows);
}

// Fences, in one place because the asymmetry is easy to get backwards:
//  - init: mbarrier.init, then fence.mbarrier_init.release.cluster, then
//    __syncthreads(), before the first arrive.expect_tx or TMA issue.
//    FenceAsyncProxy() is a different fence and does not publish the barrier -
//    the missing publish does not announce itself, it hangs with no output;
//  - completion: after the parity wait no fence is needed for a generic read of
//    what the TMA wrote - the mbarrier completion releases and the wait
//    acquires;
//  - the case that does need FenceAsyncProxy() is an MMA reading smem that
//    threads (not the TMA) wrote, e.g. a staged P tile.

// Provisional with respect to umma_common.cuh: the reusable UMMA primitives
// (barriers, descriptor builder, MMA/TMEM) live in that file. The barrier and
// descriptor functions below are the loader's local copies so that this header
// can be read and compiled on its own, and they are to be replaced by that
// header's definitions rather than kept beside them.
enum class SmemLayoutType : uint8_t {
  kSwizzleNone = 0,  // CUTLASS SWIZZLE_NONE; the 128 B permutations are the only ones used here
  k128BBase32B = 1,
  k128B = 2,
  k64B = 4,
  k32B = 6,
};

// The operand-base check, and the one place it belongs: the base is a
// planning-time fact about where a tile sits, so it is checked once per atom
// (below), not on every descriptor build. `start_address` drops the low 4 bits,
// so a 16 B aligned but not 128 B aligned base would otherwise give a
// valid-looking descriptor pointing at the wrong bytes.
__device__ __forceinline__ void CheckOperandBase(uint32_t base_smem) {
  XATTN_TMA_CHECK((base_smem & 127) == 0);
}

// The obligation base_offset = 0 actually imposes, as a separate check a caller
// may take. Kept OUT of the atom builders on purpose: widening the
// builders' check to 1024 B would trap a legal non-zero-base_offset window on a
// path that works today, so the strong check is opt-in and the caller that wants
// it calls it once per stage. E.g. a dQ leaf that stages P/dS into a
// __align__(128) buffer and reads it through a descriptor with base_offset = 0
// needs this one, and would otherwise get a valid-looking descriptor over the
// wrong 128 B phase.
__device__ __forceinline__ void CheckOperandBaseFull(uint32_t base_smem) {
  XATTN_TMA_CHECK((base_smem & 1023) == 0);
}

// Pure bit folding: no check, so it stays usable inside a k-step or atom walk
// where the base was already validated. Call CheckOperandBase (or one of the
// atom builders) first.
__device__ __forceinline__ uint64_t MakeSmemDescriptor(uint32_t base_smem, int lbo_u128,
                                                       int sbo_u128, SmemLayoutType layout) {
  uint64_t d = 0;
  d |= uint64_t((base_smem >> 4) & 0x3FFFu);              // start_address
  d |= uint64_t(lbo_u128 & 0x3FFFu) << 16;                // leading_byte_offset
  d |= uint64_t(sbo_u128 & 0x3FFFu) << 32;                // stride_byte_offset
  d |= uint64_t(1) << 46;                                 // version = 1 (Blackwell)
  d |= uint64_t(0) << 49;                                 // base_offset = 0
  d |= uint64_t(0) << 52;                                 // lbo_mode = 0 (legacy)
  d |= uint64_t(uint8_t(layout)) << 61;                   // layout_type
  return d;
}

// K-major operand over one (rows x 64) atom: contraction along the columns.
__device__ __forceinline__ uint64_t AtomDescKMajor(uint32_t atom_smem, int rows) {
  CheckOperandBase(atom_smem);
  const int sbo = (8 * kSwizzleBytes) / kU128Bytes;  // 8 rows of 128 B = 1024 B
  (void)rows;                                        // rows only matters for MN-major
  return MakeSmemDescriptor(atom_smem, /*lbo=*/1, sbo, SmemLayoutType::k128B);
}

// MN-major operand over one 64-wide MN group of a (image_rows x 64) atom:
// contraction along the rows (e.g. P^T for dV). Two of its numbers are
// properties of the IMAGE the atom belongs to and not of the operand's own
// extent - using the operand's rows is a silent-wrongness class, which is why
// the parameter carries the image's name:
//
//  - lbo is the distance to the NEXT 64-wide MN group. An image lays its atoms
//    one after another along D and each atom is image_rows*128 B, so
//    lbo = image_rows*8 u128: 512 for a 64-row image, 1024 for a 128-row image.
//    The operand's own M does not enter it: take a 64-key operand out of a
//    128-key image and the lbo is still 1024. Measured on the device exactly so
//    (device word 0x4000404004000c40, 0 of 32768 wrong) while a `rows*8` rule
//    gives 512 there - a legal stride for some image, so nothing traps.
//    The same number measured from the other side: 1024 at 128 keys, the
//    512 arm wrong in 24430 of 32768 elements outside block 0.
//  - lbo = 0 where the MMA reads a single 64-wide group and never walks
//    (n_dim <= 64): the field is unused there, which is why both 0 and the
//    image value are exact at N = 64 (measured on four paired cases
//    - byte-identical tiles read through both descriptors, both exact).
//
// `n_dim` is the operand's MN extent that this MMA consumes - >64 iff more than
// one 64-wide group is read - and NOT the instruction descriptor's per-MMA N
// field. `image_rows` is the row count of the stage tile the atom was cut from
// (64 or 128 resident rows), never the operand's own M extent.
//
// THE ROW STEP IS THE OTHER HALF OF THE PAIR (measured on the device): moving
// along the contraction is not K-major's +2 u128 per 16-column k-step. On a deep
// image (one 128-row group) the MN-major operand steps +128 u128 per 16 rows; on
// a stacked two-64-row-block image the step between the two 64-row blocks is
// +2048 u128 (11 of 11 arms matching the prediction, 0/32768 wrong at
// 7.99e-07 and 7.30e-07; the controls deep-with-512, deep-with-2048 and
// stacked-with-0 each failed about 23k of 32768 at O(1), and only in groups
// >= 2). No descriptor carries this step: whoever walks rows must add it, and it
// has the same image dependence as lbo.
//
// Precondition of base_offset = 0: the operand's byte address
// (tile_base + row_offset*128) is 1024 B aligned. Slab starts are
// row_alignment*128 B, so from a 1024 B aligned tile every row offset that is a
// multiple of 8 keeps base_offset = 0 (row 8 -> 1024 B, row 16 -> 2048 B);
// offsets 1..7 are the cases that carry a non-zero phase, and base_offset is not
// exposed here.
__device__ __forceinline__ uint64_t AtomDescMNMajor(uint32_t atom_smem, int image_rows,
                                                    int n_dim) {
  CheckOperandBase(atom_smem);
  const int sbo = (8 * kSwizzleBytes) / kU128Bytes;  // 8 rows of 128 B
  // Distance to the next 64-wide MN group: one image atom, image_rows*128 B.
  // Measured on the device (1024 for a 64-key operand out of a 128-key
  // image; 1024 at 128 keys), and 0 for a single group, where the
  // field is not consulted. n_dim has no default on purpose - guessing it gives
  // a plausible, silently wrong descriptor.
  const int lbo = (n_dim > 64) ? (image_rows * kSwizzleBytes) / kU128Bytes : 0;
  return MakeSmemDescriptor(atom_smem, lbo, sbo, SmemLayoutType::k128B);
}

static_assert(PadDim(256) == 256 && PadDim(128) == 128 && PadDim(96) == 128 &&
                  PadDim(40) == 64 && PadDim(1) == 64,
              "PadDim");
static_assert(DAtoms(PadDim(256)) == 4, "D=256 tile is 4 atoms");
static_assert(AtomBytes(64) == 8192, "RB=64 atom is 8 KiB");
static_assert(TileBytes(64, PadDim(256)) == 32768, "RB=64 D=256 tile is 32 KiB");
static_assert(TileTxBytes(64, 4) == 32768, "64 rows: tile bytes == transmitted bytes");
static_assert(TileTxBytes(72, 4) == 65536, "72 rows: two boxes per atom, 128 rows transmitted");
static_assert(RowsForBoxes(100) == 128 && RowsForBoxes(128) == 128, "box-rounded rows");
static_assert((8 * kSwizzleBytes) / kU128Bytes == 64, "K-major SBO is 64 x 16B");

#endif  // __CUDACC__

}  // namespace xattn::ops::flash_sca_native::umma
