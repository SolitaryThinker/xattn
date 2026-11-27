// Launch preparation for the SM100-family UMMA sliding-chunk attention
// kernels: both directions, in one header and one vocabulary.
//
// The backward half (`bwd_host`) does exactly two things and nothing else:
//   1. it decides, before any output is touched, whether a native backward
//      leaf can serve this call (`Refusal`), and
//   2. when it can, it completes `Params` with the backward operands and
//      builds the TMA maps an UMMA leaf reads (`UmmaLaunchParams`).
//
// It performs no arithmetic on the tensors' contents, writes no output and
// launches nothing: the only allocation it may make is the packed
// [previous; current] key/value plane, which is a copy of inputs. The row/tile
// tables (`tile_bounds`, `key_tile_range`) are *not* built here: they are prep
// stages the driver launches for either arm, and a leaf that walks analytic row
// bounds needs neither.
//
// Refusal is a status, not an exception. Every condition `EncodeInputMap`
// would raise on is replicated here as a refusal first, so a layout the TMA
// cannot describe falls back to the generic arm instead of aborting the call
// in the middle of a backward. Every check on a source that could refuse runs
// before the packed copy is enqueued; the packed plane's own geometry is then
// arithmetic on a tensor this header just made contiguous, so it cannot refuse
// and is checked where it is built. A refusal leaves the caller's tensors
// exactly as they arrived.
//
// A field a leaf needs belongs in `UmmaLaunchParams` / `UmmaInputMaps`, which
// is the shared ABI, never in a second struct here.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#include <array>
#include <cstdint>
#include <optional>

#include "flash_sca/native/params.h"
#include "flash_sca/native/umma_host.h"
#include "flash_sca/native/umma_bwd_entry.h"
#include "flash_sca/native/umma_params.h"

// Test-only switch, default off: see step 2 of PrepareBackwardUmma.
#ifndef XATTN_UMMA_PREPARE_FORCE_BUCKET
#define XATTN_UMMA_PREPARE_FORCE_BUCKET 0
#endif

namespace xattn::ops::flash_sca_native::umma {
namespace bwd_host {

// Why the native arm refused a call. The driver records this and runs the
// generic arm; it must never be read as "the call is invalid".
enum class Refusal {
  kNone = 0,
  kCapability,  // not compute capability (10, 3)
  kBucket,      // no leaf compiled for this (D, Dv) bucket
  kState,       // saved/lse/delta state the leaf cannot consume
  kTables,      // a prep table is present but not the length the kernel indexes
  kLayout,      // an output operand is not contiguous as the leaf writes it
  kMapSource,   // a TMA source the encoder cannot describe (see MirrorEncodeChecks)
  kOrderArm,    // a grid-order instrument is set: never switch body under it
  kDqAccum,     // an accumulator was handed in with nothing to fold it into
};

inline const char* RefusalName(Refusal r) {
  switch (r) {
    case Refusal::kNone: return "none";
    case Refusal::kCapability: return "capability";
    case Refusal::kBucket: return "bucket";
    case Refusal::kState: return "state";
    case Refusal::kTables: return "tables";
    case Refusal::kLayout: return "layout";
    case Refusal::kMapSource: return "map_source";
    case Refusal::kOrderArm: return "order_arm";
    case Refusal::kDqAccum: return "dq_accum";
  }
  return "unknown";
}

// Preparation is useful if at least one product has a compiled entry. Each
// stage's exact call eligibility is decided later by that entry's predicate;
// a supported bucket alone never selects a writer. All nine public bucket
// pairs remain legal, including those without a mounted native product.
inline bool BackwardLeafCompiled(int64_t d, int64_t dv) {
  return umma::BackwardLeafCompiled(BackwardProduct::kDq, d, dv) ||
         umma::BackwardLeafCompiled(BackwardProduct::kDkdv, d, dv);
}

inline Operand MakeOperand(const at::Tensor& t) {
  Operand o;
  o.ptr = t.data_ptr();
  o.batch_stride = t.stride(0);
  o.row_stride = t.stride(1);
  o.head_stride = t.stride(2);
  return o;
}

// The exact constraints of EncodeInputMap (umma_host.h), as a predicate, in
// the order the encoder raises them and with the same singleton-axis
// normalisation: a stride on a singleton axis never moves an address, but the
// TMA still validates it, so it is normalised rather than trusted.
inline bool MirrorEncodeChecks(const Operand& source, int64_t batch, int64_t rows,
                               int64_t heads, int64_t columns) {
  if (batch == 0 || rows == 0) return true;  // absent map, never consumed
  if (source.ptr == nullptr || batch <= 0 || rows <= 0 || heads <= 0) return false;
  if (columns <= 0 || columns > 256 || columns % 8 != 0) return false;
  if (reinterpret_cast<uintptr_t>(source.ptr) % 16 != 0) return false;
  const int64_t head_stride = heads == 1 ? columns : source.head_stride;
  const int64_t row_stride = rows == 1 ? head_stride * heads : source.row_stride;
  const int64_t batch_stride = batch == 1 ? row_stride * rows : source.batch_stride;
  const int64_t strides[3] = {head_stride, row_stride, batch_stride};
  for (const int64_t s : strides) {
    if (s <= 0 || s % 8 != 0 || s >= (int64_t(1) << 39)) return false;
  }
  return true;
}

struct BackwardUmma {
  UmmaLaunchParams launch{};
  // Owners of the packed [previous; current] K/V planes. Empty unless the call
  // had a previous chunk whose length is not a multiple of kKeyMapAlignment.
  // Lifetime, normative: they must outlive the LAST LAUNCH ENQUEUED from
  // `launch` on the consuming stream, not merely the enqueue - same-stream
  // ordering plus the caching allocator's stream-ordered reuse make holding
  // them past the enqueue sufficient today only because every launch is on that
  // one stream. A driver that launches on a second stream must keep them until
  // that stream has passed the kernels, or record an event on it.
  at::Tensor packed_k, packed_v;
  Refusal refusal = Refusal::kNone;
  bool ok() const { return refusal == Refusal::kNone; }
};

// Completes an already-validated `Params` (host.cc PrepareCommon) with the
// backward operands and builds the maps. `common` is taken by value because the
// result is a different launch, not a mutation of the caller's preparation.
//
// Inputs the caller owns and has already normalised with host.cc's Prepare /
// PrepareContiguous (16-byte base alignment, strides as the public contract
// allows):
//   q, k, v, prev_k, prev_v   the forward operands, same meaning as forward
//   dy_c                      [B,Nq,Hq,Dv] Q's dtype, the incoming gradient
//   saved_c                   the forward output the backward reads: level 0
//                             [B,Nq,Hq,Dv] Q's dtype, levels 1-4 FP32
//   lse_c                     [B,Hq,Nq] FP32
//   delta                     [B,Hq,Nq] FP32 scratch, row sums of dO*O
//   dq, dk, dv, dprev_k, dprev_v  freshly allocated, contiguous, Q's dtype
//   dq_accum                  FP32 [B,Nq,Hq,D] when the chosen family plan
//                             reduces dQ atomically, an empty tensor when it
//                             writes dQ row-owned; never a function of the
//                             user's `deterministic` flag (see step 6)
// `deterministic` is accepted for the driver's convenience and is not read:
// who owns dQ is the family plan's decision, and the plan is the driver's.
inline BackwardUmma PrepareBackwardUmma(const at::Tensor& q, const at::Tensor& k,
                                        const at::Tensor& v, const at::Tensor* prev_k,
                                        const at::Tensor* prev_v, const at::Tensor& dy_c,
                                        const at::Tensor& saved_c, const at::Tensor& lse_c,
                                        const at::Tensor& delta, at::Tensor& dq, at::Tensor& dk,
                                        at::Tensor& dv, const at::Tensor* dprev_k,
                                        const at::Tensor* dprev_v, const at::Tensor& dq_accum,
                                        bool deterministic, const Params& common) {
  (void)deterministic;  // dQ ownership is the family plan's, not this header's
  BackwardUmma out;
  Params& p = out.launch.params;
  p = common;

  // 1. Capability. Same test as Available(): the sm_103a cubin runs on 10.3 only.
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(static_cast<c10::DeviceIndex>(q.get_device()));
  if (properties == nullptr || properties->major != 10 || properties->minor != 3) {
    out.refusal = Refusal::kCapability;
    return out;
  }

  // 2. Bucket. Refuse before anything is described or allocated.
  //
  // XATTN_UMMA_PREPARE_FORCE_BUCKET exists so that a test can reach the legs
  // below while no leaf is compiled: with it set the bucket predicate is
  // bypassed and every later refusal becomes observable. It defaults to 0 and
  // the build never defines it, so the shipped behaviour is the predicate.
  if (!XATTN_UMMA_PREPARE_FORCE_BUCKET && !BackwardLeafCompiled(p.qk_dim, p.value_dim)) {
    out.refusal = Refusal::kBucket;
    return out;
  }

  // 3. The grid-order instruments select an order and a layout for the
  // mma.sync body. A body switch under a live instrument would report one arm's
  // numbers as the other's, so the native arm stays out of the way.
  if (p.sorted_order || p.head_fastest) {
    out.refusal = Refusal::kOrderArm;
    return out;
  }

  // 4. Saved state the leaf consumes, described exactly. A leaf reads these
  // three planes as dense row-major [B,Hq,Nq] / [B,Nq,Hq,Dv], so a view that
  // only happens to have the right numel is a wrong answer, not a slow one:
  // sizes and dtype are exact, contiguity is required, and the saved pointer
  // must carry the 16-byte alignment its vector loads assume.
  const std::array<int64_t, 3> stat_shape{p.batch, p.query_heads, p.q_length};
  const std::array<int64_t, 4> saved_shape{p.batch, p.q_length, p.query_heads, p.value_dim};
  if (lse_c.scalar_type() != at::kFloat || lse_c.numel() == 0 || !lse_c.is_contiguous() ||
      lse_c.sizes() != at::IntArrayRef(stat_shape) ||
      reinterpret_cast<uintptr_t>(lse_c.data_ptr()) % 16 != 0) {
    out.refusal = Refusal::kState;
    return out;
  }
  if (delta.scalar_type() != at::kFloat || !delta.is_contiguous() ||
      delta.sizes() != at::IntArrayRef(stat_shape) ||
      reinterpret_cast<uintptr_t>(delta.data_ptr()) % 16 != 0) {
    out.refusal = Refusal::kState;
    return out;
  }
  if ((saved_c.scalar_type() != q.scalar_type() && saved_c.scalar_type() != at::kFloat) ||
      !saved_c.is_contiguous() || saved_c.sizes() != at::IntArrayRef(saved_shape) ||
      reinterpret_cast<uintptr_t>(saved_c.data_ptr()) % 16 != 0) {
    out.refusal = Refusal::kState;
    return out;
  }
  // dy is the fourth plane the leaf reads as given, and the only one that
  // becomes a TMA source: its map is encoded below from Q's element type and
  // from these extents, so a wrong dtype encodes a 16-bit map over elements of
  // another width and a wrong extent indexes past the tensor's end. The driver
  // checks both before it gets here; a seam that is fail-closed does not rely
  // on that.
  //
  // Contiguity is required as well, because `p.dy` is published to a leaf as a
  // raw row-major `const void*` beside the map: a strided-but-describable dy
  // (a head-sliced view) encodes a lawful map and then reads as dense rows, i.e.
  // a silently wrong gradient in any leaf that uses the raw pointer.
  if (dy_c.scalar_type() != q.scalar_type() || dy_c.sizes() != at::IntArrayRef(saved_shape) ||
      !dy_c.is_contiguous()) {
    out.refusal = Refusal::kState;
    return out;
  }

  // 5. Prep tables. A table that *is* present must have the length the kernel
  // indexes it with: a wrong count is a garbage read, not a slow path. Absence
  // is not a refusal, because a leaf may consume analytic row bounds instead.
  const int64_t key_tiles = (p.total_length + 63) / 64;
  if (p.tile_bounds != nullptr && p.tile_count != (p.q_length + 63) / 64) {
    out.refusal = Refusal::kTables;
    return out;
  }
  if (p.key_tile_range != nullptr && p.key_tile_count != key_tiles) {
    out.refusal = Refusal::kTables;
    return out;
  }

  // 6. dQ reduction. `dq_accum` is a consequence of the family plan the driver
  // selected, not the pointer the driver reads to decide which body owns dQ:
  // deterministic=false permits a direct row-owned dQ writer, so it must not
  // force an atomic accumulator (nor its allocation) on every call. What this
  // header can check without knowing the family is coherence: an accumulator
  // needs a fold target, because the caller's finalize writes dq from it.
  if (dq_accum.defined() && dq_accum.numel() != 0) {
    if (dq.data_ptr() == nullptr || dq_accum.scalar_type() != at::kFloat ||
        !dq_accum.is_contiguous() || dq_accum.sizes() != dq.sizes() ||
        reinterpret_cast<uintptr_t>(dq_accum.data_ptr()) % 16 != 0) {
      out.refusal = Refusal::kDqAccum;
      return out;
    }
    p.dq_accum = dq_accum.data_ptr<float>();
  } else {
    p.dq_accum = nullptr;
  }

  // 7. Output layouts and extents, checked before a single byte is written: a
  // contiguous, right-typed, wrong-shaped output is an out-of-bounds write from
  // the leaf, which is the failure class this preparation exists to make
  // impossible.
  //
  // Pair rule and coverage home — file-level, never coordinates, because a line
  // number in a shipped comment goes stale the next time the file is touched:
  // this preparation intentionally does not refuse `has_prev` with both
  // `dprev_*` null. Home of the rule: the public boundary's dprev allocation
  // site in `host.cc` (it allocates the pair together and only when the caller
  // supplied a previous chunk, so a half pair never reaches a leaf from the
  // public API), plus the plan's `zero_prev` and the driver's
  // producer-per-output check in `umma_bwd.h`. A hand-built call that passes
  // prev K/V with both gradients absent is served correctly by the generic body
  // today, and a refusal here would duplicate one rule across two layers and
  // fail-closed on a call that works. What is checked here is the layout of
  // each gradient that *is* present.
  const std::array<int64_t, 4> dq_shape{p.batch, p.q_length, p.query_heads, p.qk_dim};
  const std::array<int64_t, 4> dk_shape{p.batch, p.k_length, p.kv_heads, p.qk_dim};
  const std::array<int64_t, 4> dv_shape{p.batch, p.k_length, p.kv_heads, p.value_dim};
  // The key gradients are FP32 when the call asked for them (Params::key_grads_fp32).
  const at::ScalarType key_type = p.key_grads_fp32 ? at::kFloat : q.scalar_type();
  if (!dq.is_contiguous() || dq.scalar_type() != q.scalar_type() ||
      dq.sizes() != at::IntArrayRef(dq_shape) ||
      !dk.is_contiguous() || dk.scalar_type() != key_type ||
      dk.sizes() != at::IntArrayRef(dk_shape) ||
      !dv.is_contiguous() || dv.scalar_type() != key_type ||
      dv.sizes() != at::IntArrayRef(dv_shape)) {
    out.refusal = Refusal::kLayout;
    return out;
  }
  if (dprev_k != nullptr || dprev_v != nullptr) {
    const std::array<int64_t, 4> dprev_k_shape{p.batch, p.prev_length, p.kv_heads, p.qk_dim};
    const std::array<int64_t, 4> dprev_v_shape{p.batch, p.prev_length, p.kv_heads, p.value_dim};
    if (dprev_k == nullptr || dprev_v == nullptr ||
        !dprev_k->is_contiguous() || dprev_k->scalar_type() != key_type ||
        dprev_k->sizes() != at::IntArrayRef(dprev_k_shape) || !dprev_v->is_contiguous() ||
        dprev_v->scalar_type() != key_type ||
        dprev_v->sizes() != at::IntArrayRef(dprev_v_shape)) {
      out.refusal = Refusal::kLayout;
      return out;
    }
  }

  // 7b. The INPUT pair rule, which is not the output pair rule above. A half
  // previous K/V pair is refused in both directions: `has_prev` in step 8 is a
  // conjunction, so a call with prev_k and no prev_v would be prepared as a call
  // with no previous chunk at all - and the masks, whose metadata still covers the
  // previous key positions, would then skip keys the caller believes are there.
  // Nothing downstream can tell the two calls apart, so the refusal has to happen
  // here, before any operand is built and before `has_prev` can collapse them.
  if ((prev_k != nullptr) != (prev_v != nullptr)) {
    out.refusal = Refusal::kLayout;
    return out;
  }

  // Operand carries no last-dimension stride. Never encode a map over a view
  // whose columns are not contiguous, even if its outer strides pass the TMA
  // checks. The public host passes the normalized, retained input owners.
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 ||
      q.stride(-1) != 1 || k.stride(-1) != 1 || v.stride(-1) != 1 ||
      (prev_k != nullptr && (prev_k->dim() != 4 || prev_v->dim() != 4 ||
                            prev_k->stride(-1) != 1 || prev_v->stride(-1) != 1))) {
    out.refusal = Refusal::kMapSource;
    return out;
  }

  // 8. Operands, fully populated before anything is encoded or checked, so that
  // no path leaves the caller with a half-filled Params. k/v/prev_* stay the
  // *logical* views: a leaf that wants the packed plane below reads it through
  // maps.k/maps.v with maps.packed_kv set, while Params keeps the operands the
  // masks and the dK/dV -> dPrevK/dPrevV split are written against.
  const int64_t B = p.batch, Nq = p.q_length, Nk = p.k_length, Nt = p.total_length;
  const int64_t Hq = p.query_heads, Hkv = p.kv_heads, D = p.qk_dim, Dv = p.value_dim;
  const bool has_prev = prev_k != nullptr && prev_v != nullptr;
  const bool pack = has_prev && (p.prev_length % kKeyMapAlignment) != 0;

  p.q = MakeOperand(q);
  p.k = MakeOperand(k);
  p.v = MakeOperand(v);
  p.prev_k = has_prev ? MakeOperand(*prev_k) : Operand{};
  p.prev_v = has_prev ? MakeOperand(*prev_v) : Operand{};
  p.dy = dy_c.data_ptr();
  p.saved = saved_c.data_ptr();
  p.saved_fp32 = saved_c.scalar_type() == at::kFloat;
  // `saved_fp32` is level-blind by construction: it says what dtype the plane
  // has, not which precision level produced it, and level >= 2 is routed to the
  // precision backend before the native route is considered, so it must not be
  // read as a precision selector. `saved` has no entry in UmmaInputMaps, and
  // cannot have one yet: the only
  // encoder in the tree (umma_host.h:EncodeInputMap) describes 2-byte elements
  // with a 64-element box, while an FP32 plane at SWIZZLE_128B needs a
  // 32-element box (boxDim[0] * elemsize <= 128 B), i.e. 8 slabs at Dv=256. A
  // leaf therefore reads `saved` with plain global loads (as the mma.sync body
  // already does; the pointer is 16-byte aligned, so float4 where the row
  // allows). Adding a 4-byte map later is a change to the shared maps struct
  // and to the encoder, with a 64 KiB SMEM tile at Dv=256 to budget; no leaf
  // at levels 0-1 needs it.
  p.lse_in = lse_c.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  p.dq = dq.data_ptr();
  p.dk = dk.data_ptr();
  p.dv = dv.data_ptr();
  p.dprev_k = (has_prev && dprev_k != nullptr) ? dprev_k->data_ptr() : nullptr;
  p.dprev_v = (has_prev && dprev_v != nullptr) ? dprev_v->data_ptr() : nullptr;

  // 9. Every source is checked against the encoder's constraints *before* any
  // device work: a source the TMA cannot describe is a refusal, and a call that
  // will be refused must not have paid for a copy first. `p.dy` is a raw
  // `const void*` in Params, so its operand form lives here: it feeds the check
  // and the map and is never stored.
  const Operand dy_op = MakeOperand(dy_c);
  if (!MirrorEncodeChecks(p.q, B, Nq, Hq, D) || !MirrorEncodeChecks(dy_op, B, Nq, Hq, Dv) ||
      !MirrorEncodeChecks(p.k, B, Nk, Hkv, D) || !MirrorEncodeChecks(p.v, B, Nk, Hkv, Dv)) {
    out.refusal = Refusal::kMapSource;
    return out;
  }
  // The previous plane is checked even when packing will densify it. The
  // packed copy would rescue a previous plane the map walk cannot describe, and
  // then the backward arm would accept a call the forward arm refuses for the
  // same input - two bodies disagreeing about what they serve, and the cat
  // hiding a layout neither of them was written for. So the source predicate is
  // the same on both arms and runs before the copy (see step 10).
  if (has_prev) {
    // Extents and element type first: the mirror checks below validate the
    // strides a shape *implies*, so a footprint-short contiguous previous plane
    // with lawful %8 strides would pass them and the map would then describe
    // `p.prev_length` rows of a shorter tensor - an out-of-bounds read on the
    // non-packed path and a TORCH_CHECK out of `at::cat`, not a refusal, on the
    // packed one. `kLayout`, because an extent is a layout fact and
    // describability is `kMapSource`.
    const std::array<int64_t, 4> prev_k_shape{p.batch, p.prev_length, p.kv_heads, p.qk_dim};
    const std::array<int64_t, 4> prev_v_shape{p.batch, p.prev_length, p.kv_heads, p.value_dim};
    if (prev_k->scalar_type() != q.scalar_type() ||
        prev_k->sizes() != at::IntArrayRef(prev_k_shape) ||
        prev_v->scalar_type() != q.scalar_type() ||
        prev_v->sizes() != at::IntArrayRef(prev_v_shape)) {
      out.refusal = Refusal::kLayout;
      return out;
    }
    if (!MirrorEncodeChecks(p.prev_k, B, p.prev_length, Hkv, D) ||
        !MirrorEncodeChecks(p.prev_v, B, p.prev_length, Hkv, Dv)) {
      out.refusal = Refusal::kMapSource;
      return out;
    }
  }

  // 10. Key axis. The forward leaf refuses a previous chunk whose length is not
  // a multiple of the key-map alignment unless the two planes are packed into
  // one map; the backward leaf applies the *same* predicate, or the two bodies
  // would disagree about which key tile means what. Params::prev_length keeps
  // its logical value either way: masks and the dK/dV -> dPrevK/dPrevV split
  // read it, not the map.
  Operand src_k = p.k, src_v = p.v;
  if (pack) {
    // One contiguous [B, Nt, Hkv, *] plane per operand. This is the only
    // allocation the preparation performs, and it happens only on the unaligned
    // path: the traced prefill's previous chunk is 4096 rows, a multiple of the
    // key-map alignment, so no traced call takes it. When a call does take it,
    // the copy is the previous plane of the packed extent - at the traced
    // 4096/16384 prefill geometry a quarter of the current plane and a fifth of
    // the packed one - not a rounding.
    out.packed_k = at::cat({*prev_k, k}, /*dim=*/1).contiguous();
    out.packed_v = at::cat({*prev_v, v}, /*dim=*/1).contiguous();
    src_k = MakeOperand(out.packed_k);
    src_v = MakeOperand(out.packed_v);
    if (!MirrorEncodeChecks(src_k, B, Nt, Hkv, D) ||
        !MirrorEncodeChecks(src_v, B, Nt, Hkv, Dv)) {
      out.refusal = Refusal::kMapSource;
      return out;
    }
  }

  // 11. Maps. Axes are [column, head, row, batch]; a tile starts at [d,h,r,b].
  const int dev = static_cast<int>(q.get_device());
  UmmaInputMaps& m = out.launch.maps;
  m.q = CachedInputMap(p.q, p.element, B, Nq, Hq, D, dev);
  m.dy = CachedInputMap(dy_op, p.element, B, Nq, Hq, Dv, dev);
  m.k = CachedInputMap(src_k, p.element, B, pack ? Nt : Nk, Hkv, D, dev);
  m.v = CachedInputMap(src_v, p.element, B, pack ? Nt : Nk, Hkv, Dv, dev);
  if (has_prev && !pack) {
    m.prev_k = CachedInputMap(p.prev_k, p.element, B, p.prev_length, Hkv, D, dev);
    m.prev_v = CachedInputMap(p.prev_v, p.element, B, p.prev_length, Hkv, Dv, dev);
  }
  // Backward output TMA store maps (same 64x64 SWIZZLE_128B layout as inputs; TMA clips OOB).
  if (B > 0 && Nq > 0) {
    const Operand dq_op = MakeOperand(dq);
    if (MirrorEncodeChecks(dq_op, B, Nq, Hq, D)) {
      m.dq = CachedInputMap(dq_op, p.element, B, Nq, Hq, D, dev);
      m.has_dq_map = true;
    }
  }
  if (!p.key_grads_fp32 && B > 0 && Nk > 0) {
    const Operand dk_op = MakeOperand(dk);
    const Operand dv_op = MakeOperand(dv);
    if (MirrorEncodeChecks(dk_op, B, Nk, Hkv, D) && MirrorEncodeChecks(dv_op, B, Nk, Hkv, Dv)) {
      m.dk = CachedInputMap(dk_op, p.element, B, Nk, Hkv, D, dev);
      m.dv = CachedInputMap(dv_op, p.element, B, Nk, Hkv, Dv, dev);
      m.has_dkv_map = true;
    }
  }
  if (!p.key_grads_fp32 && has_prev && B > 0 && p.prev_length > 0 &&
      dprev_k != nullptr && dprev_v != nullptr) {
    const Operand dpk_op = MakeOperand(*dprev_k);
    const Operand dpv_op = MakeOperand(*dprev_v);
    if (MirrorEncodeChecks(dpk_op, B, p.prev_length, Hkv, D) &&
        MirrorEncodeChecks(dpv_op, B, p.prev_length, Hkv, Dv)) {
      m.dprev_k = CachedInputMap(dpk_op, p.element, B, p.prev_length, Hkv, D, dev);
      m.dprev_v = CachedInputMap(dpv_op, p.element, B, p.prev_length, Hkv, Dv, dev);
      m.has_dprev_kv_map = true;
    }
  }
  // Invariant, normative: with packed_kv set, key/value are consumed through
  // the maps only; Params::prev_k / Params::prev_v describe the call's logical
  // previous plane and must not be used to address the packed storage, and
  // Params::k / Params::v describe the current plane, whose rows start at
  // offset prev_length in the packed map.
  m.packed_kv = pack;
  // Stream order, normative: the packed planes above were produced on the
  // current stream of the device at the time of this call, while a launch reads
  // them on the stream the driver passes. A driver whose stream differs must
  // record an event after the pack and make its stream wait on it; with one
  // stream (the current one) nothing more is needed. This is an ordering
  // requirement, separate from the survival requirement on BackwardUmma.

  // 12. This preparation writes no gradient output. The driver requires each
  // selected key leaf to cover the whole current/previous plane, including
  // exact zeros for keys no query can see. Partial-plane leaves need a future
  // explicit omitted-region protocol; key ownership alone never requests zeroing.

  out.refusal = Refusal::kNone;
  return out;
}

}  // namespace bwd_host

// ---------------------------------------------------------------------------
// Forward launch preparation.
//
// The forward half of the same ABI, and deliberately a different policy on the
// one question the two directions answer differently: an input the copy engine
// cannot describe. The backward refuses it to the generic arm (`kMapSource`),
// because a body switch in the middle of a backward must never change which
// bytes a gradient is computed from. The forward *materializes* it instead: the
// forward arm already has its own predicate for whether a leaf may serve the
// call (`HasUmmaForwardLeaf` plus the column budget in host.cc), a copy of one
// input plane is cheaper than sending a traced training call to the slower body,
// and the copy is a copy of the caller's own bytes, not a rounding. Neither half
// changes a number: both only decide how the same bytes are described.
//
// What it guarantees to the leaf, so the kernel does not re-check any of it:
//   * every map's base is 16-byte aligned and its last-dimension stride is 1;
//   * every outer stride is positive, a multiple of 8 elements (16 bytes) and
//     inside TMA's range;
//   * `maps.packed_kv` is set exactly when the previous K/V length is not a
//     multiple of `kKeyMapAlignment`, and in that case `maps.k/v` address a
//     contiguous [previous; current] copy while `params.prev_length` keeps its
//     logical value (the mask reads the logical value);
//   * absent inputs leave their map zeroed; an absent map is never consumed.
namespace fwd_host {

// Singleton strides are normalized by EncodeInputMap. All other outer strides
// must be positive, 16-byte aligned, and representable by TMA.
inline bool IsTmaReadable(const at::Tensor& tensor) {
  if (tensor.numel() == 0) return true;
  if (tensor.stride(-1) != 1 || reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 != 0) {
    return false;
  }
  // Keep the zero-copy path conservatively non-overlapping in the public
  // [B,L,H,D] axis order. Other positive-stride layouts still work via
  // materialization; no assumption about overlap support leaks into TMA maps.
  int64_t span = tensor.size(-1);
  for (int64_t axis = tensor.dim() - 2; axis >= 0; --axis) {
    if (tensor.size(axis) <= 1) continue;
    const int64_t stride = tensor.stride(axis);
    if (stride < span || stride % 8 != 0 || stride >= (int64_t(1) << 39)) return false;
    if (tensor.size(axis) - 1 > ((int64_t(1) << 62) - span) / stride) return false;
    span += (tensor.size(axis) - 1) * stride;
  }
  return true;
}

inline at::Tensor CanonicalizeTmaInput(const at::Tensor& tensor) {
  if (IsTmaReadable(tensor)) return tensor;
  at::Tensor result = tensor.contiguous();
  // contiguous() may retain an already-contiguous view with an unaligned
  // storage offset. A fresh allocation is required in that case.
  if (!IsTmaReadable(result)) result = result.clone(at::MemoryFormat::Contiguous);
  TORCH_CHECK(IsTmaReadable(result), "UMMA FlashSCA could not prepare a TMA input");
  return result;
}

// Strong references remain live through launch submission on the caller's
// current stream. Tensor maps contain addresses, not ownership of storage.
struct PreparedUmmaCall {
  UmmaLaunchParams launch;
  at::Tensor q, k, v, prev_k, prev_v, dy;
  at::Tensor packed_k, packed_v;
};

// The public host boundary has already validated shapes, dtype and device and
// has installed the current CUDA device/stream. `Params` keeps the logical
// current/previous split even when the input maps address one packed key axis:
// the mask reads the logical value.
inline PreparedUmmaCall PrepareUmmaCall(
    const Params& semantic, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const std::optional<at::Tensor>& prev_k, const std::optional<at::Tensor>& prev_v,
    const std::optional<at::Tensor>& dy = std::nullopt) {
  TORCH_CHECK(prev_k.has_value() == prev_v.has_value(),
              "UMMA FlashSCA previous K/V must be supplied together");
  TORCH_CHECK((semantic.prev_length > 0) == prev_k.has_value(),
              "UMMA FlashSCA previous K/V do not match the logical key axis");
  PreparedUmmaCall result;
  result.launch.params = semantic;
  // The host handle points back at this object's own maps; the launched Params
  // copy has no use for it and must not carry a pointer into the caller's frame.
  result.launch.params.umma_maps = nullptr;
  Params& p = result.launch.params;
  UmmaInputMaps& maps = result.launch.maps;

  result.q = CanonicalizeTmaInput(q);
  result.k = CanonicalizeTmaInput(k);
  result.v = CanonicalizeTmaInput(v);
  p.q = bwd_host::MakeOperand(result.q);
  p.k = bwd_host::MakeOperand(result.k);
  p.v = bwd_host::MakeOperand(result.v);
  if (prev_k) {
    result.prev_k = CanonicalizeTmaInput(*prev_k);
    result.prev_v = CanonicalizeTmaInput(*prev_v);
    p.prev_k = bwd_host::MakeOperand(result.prev_k);
    p.prev_v = bwd_host::MakeOperand(result.prev_v);
  }
  if (dy) {
    // Params::dy is also read by the base delta reduction, whose ABI is dense.
    result.dy = CanonicalizeTmaInput(dy->contiguous());
    p.dy = result.dy.data_ptr();
  }
  if (p.batch == 0) return result;

  const int dev = static_cast<int>(q.get_device());
  maps.q = CachedInputMap(p.q, p.element, p.batch, p.q_length, p.query_heads, p.qk_dim, dev);
  maps.packed_kv = p.prev_length != 0 && p.prev_length % kKeyMapAlignment != 0;
  if (maps.packed_kv) {
    result.packed_k = at::cat({result.prev_k, result.k}, 1);
    result.packed_v = at::cat({result.prev_v, result.v}, 1);
    maps.k = CachedInputMap(bwd_host::MakeOperand(result.packed_k), p.element, p.batch,
                            p.total_length, p.kv_heads, p.qk_dim, dev);
    maps.v = CachedInputMap(bwd_host::MakeOperand(result.packed_v), p.element, p.batch,
                            p.total_length, p.kv_heads, p.value_dim, dev);
  } else {
    maps.k = CachedInputMap(p.k, p.element, p.batch, p.k_length, p.kv_heads, p.qk_dim, dev);
    maps.v = CachedInputMap(p.v, p.element, p.batch, p.k_length, p.kv_heads, p.value_dim, dev);
    if (p.prev_length != 0) {
      maps.prev_k =
          CachedInputMap(p.prev_k, p.element, p.batch, p.prev_length, p.kv_heads, p.qk_dim, dev);
      maps.prev_v =
          CachedInputMap(p.prev_v, p.element, p.batch, p.prev_length, p.kv_heads, p.value_dim, dev);
    }
  }
  if (dy) {
    maps.dy = CachedInputMap(bwd_host::MakeOperand(result.dy), p.element, p.batch, p.q_length,
                             p.query_heads, p.value_dim, dev);
  }
  return result;
}

}  // namespace fwd_host
}  // namespace xattn::ops::flash_sca_native::umma
