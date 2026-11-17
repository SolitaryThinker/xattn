// Backward product ownership and native preflight for SM100-family FlashSCA.
// This host-only header has no CUDA or tensor-library dependencies.
//
// Scope: the level 0-1 product families. Their shared delta prerequisite is
// always the scalar preprocessing stage, completed once before any product
// consumes it. An alternative delta producer requires explicit exclusive
// selection in the stage driver; it is not a product-support bit here.
// Levels 2-4 use a separate state/repair pipeline outside this planner. They
// are a shared native-port requirement, not a fourth value of BackwardFamily.
#pragma once

#include <cstdint>

namespace xattn::ops::flash_sca_native::umma {

enum class BackwardFamily : uint8_t {
  kSplit,          // Independent row-dQ, key-dK and key-dV passes.
  kFusedKey,       // Row-dQ followed by a fused key-owned dK/dV pass.
  kFusedAtomicDq,  // Key-owned dK/dV with FP32 atomic dQ contributions.
};

// A direct writer covers every row it owns, including exact zero stores for
// empty rows. Deterministic requests also require a fixed key/reduction order;
// absence of atomics alone is not a proof of bit-exact repetition.
enum class DqWriteMode : uint8_t { kDirect, kAtomic };

struct BackwardPlan {
  BackwardFamily family;
  DqWriteMode dq_mode;

  constexpr bool separate_dq() const { return dq_mode == DqWriteMode::kDirect; }
  constexpr bool fused_key_outputs() const { return family != BackwardFamily::kSplit; }
  constexpr bool needs_dq_accumulator() const { return dq_mode == DqWriteMode::kAtomic; }
};

// `direct` is true when the stage driver has selected a direct writer: a
// deterministic or split-arm request forces one, and so does any selected native
// row-dQ or key leaf (SelectWriters in umma_bwd.cu). It is NOT the determinism
// flag. A deterministic request is always direct, so it never selects unordered
// dQ updates; a deterministic=false call is direct whenever a direct writer is
// available, so it does not require atomics. Only a kFusedAtomicDq request with
// no direct writer yields kAtomic. The family identifies the requested design;
// dq_mode identifies the actual dQ producer. The atomic family on a direct call
// retains its family identity but uses fused-key plus row-dQ bodies. Dispatch
// and phase records must consume both fields (or the accessors), never switch
// on family alone.
constexpr BackwardPlan MakeBackwardPlan(BackwardFamily family, bool direct) {
  const bool atomic = family == BackwardFamily::kFusedAtomicDq && !direct;
  return {family, atomic ? DqWriteMode::kAtomic : DqWriteMode::kDirect};
}

// Results of native-leaf preflight on the exact prepared call, produced by the
// stage driver's per-leaf eligibility queries. Each bit means a real native
// leaf is compiled AND accepts its operands, maps, dimensions, state and output
// layouts, not merely that a wrapper symbol is linkable. The driver retains
// each refusal reason and selected body separately for diagnostics/phase tags;
// these Boolean requirements do not replace that record.
struct BackwardSupport {
  bool dq = false;
  bool dk = false;
  bool dv = false;
  bool dkdv = false;
  bool dkdv_atomic_dq = false;
  // A key owner must cover all current and previous rows, including zeros.
  // Partial-plane leaves need an explicit omitted-region protocol before use.
  bool dkdv_writes_whole_plane = false;
};

// True means the COMPLETE native product family is supported. False does not
// prohibit a transitional mixed native/generic schedule: that schedule is
// constructed and preflighted by the stage driver, with each output assigned
// exactly once and each actual producer recorded. Generic availability must
// not be encoded as a true native-support bit. In particular a fused generic
// dK/dV fallback cannot replace just one of split dK and dV while leaving the
// other native writer active; that would write the latter output twice.
constexpr bool CanLaunchBackward(const BackwardPlan& plan,
                                 const BackwardSupport& support) {
  switch (plan.family) {
    case BackwardFamily::kSplit:
      return plan.dq_mode == DqWriteMode::kDirect && support.dq && support.dk && support.dv;
    case BackwardFamily::kFusedKey:
      return plan.dq_mode == DqWriteMode::kDirect && support.dq && support.dkdv;
    case BackwardFamily::kFusedAtomicDq:
      if (plan.dq_mode == DqWriteMode::kAtomic) return support.dkdv_atomic_dq;
      return plan.dq_mode == DqWriteMode::kDirect && support.dq && support.dkdv;
  }
  return false;
}

// Checks presence ONLY: it cannot inspect dtype, shape, zeroing or lifetime.
// A mismatch is a driver invariant violation, not a leaf-refusal reason. The
// host separately validates FP32 dtype and exact dQ shape, and zero-initializes
// an atomic accumulator before its first writer. A direct writer must not have
// an accumulator finalized over its output.
constexpr bool DqAccumulatorMatches(const BackwardPlan& plan, bool has_accumulator) {
  return plan.needs_dq_accumulator() == has_accumulator;
}

// Driver/host protocol (including transitional mixed schedules):
//  * The stage driver selects the requested family and calls MakeBackwardPlan
//    before the host allocates dq_accum. Selection must not infer ownership
//    from a scratch pointer. After allocation/binding, the driver preflights
//    every selected producer on the final prepared operands before any output
//    writer starts. A refusal may select another schedule only at this point.
//  * Every output has a named coverage owner. A leaf normally writes all its
//    output rows, including zeros for invisible keys. A leaf that omits regions
//    must declare those regions to the driver, which initializes them after
//    acceptance and before the first output writer. No unconditional zeroing
//    is added to generic outputs whose own epilogues already cover all rows.
//  * Run delta once, then all products on the same CUDA stream. Atomic dQ
//    contributions are unscaled; finalize scale/conversion exactly once on that
//    stream after the last atomic writer. Direct dQ/dK scale their FP32 sums
//    once; dV is never scaled. Never finalize scratch over direct dQ output.
//  * Propagate launch errors. Once a writer launches, do not return a refusal
//    and replay another family over partially written outputs.
// Before preflight succeeds, no output may be modified. Packed TMA staging
// owners survive the last launch enqueued on the consuming stream; if producer
// and consumer streams differ, the driver must provide both execution ordering
// and allocator lifetime protection on the consumer stream. Maps must preserve
// the logical current/previous operand split and previous-key length, including
// a key tile that straddles an unaligned previous/current boundary in a packed
// map: storage alignment never changes the mask or the gradient output split.

}  // namespace xattn::ops::flash_sca_native::umma
