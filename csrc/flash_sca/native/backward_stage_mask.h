// Which stages of the generic flash_sca backward body still have to run.
//
// The backward is a driver over three stages -- delta, dQ, one fused dK+dV -- in
// the order the tree has always used (launch.cuh:BackwardStages). A stage leaves
// the mask when another body has served it, so one call can run a native dQ
// beside the generic dK/dV while the leaves land. kBackwardAllStages is the
// behaviour every existing caller had, and the only mask in use while no other
// body is registered.
//
// This header carries no state and defines nothing: it exists so that the driver
// (launch.cuh) and the bucket dispatch (dispatch.cu) name the same mask instead of
// each carrying its own. The torch-facing host does not consume it yet -- it still
// calls `LaunchBackward`, that is, kBackwardAllStages -- and will include it when
// the driver call replaces that line. If the shared ABI folds it into params.h,
// this file goes away and the includes do not.
#pragma once

#include <cuda_runtime.h>

#include "flash_sca/native/params.h"

namespace xattn::ops::flash_sca_native {

enum BackwardStageMask : unsigned {
  kStageDelta = 1u << 0,
  kStageDq = 1u << 1,
  kStageDkdv = 1u << 2,
  kBackwardAllStages = kStageDelta | kStageDq | kStageDkdv,
};

// Runs the stages still set in `stages`, in the usual order and with the same
// kernels, tags and buffers. `LaunchBackward(p, stream)` is this with
// kBackwardAllStages.
//
// Contract:
//   * Unknown bits are refused. stages == 0 without an accumulator is legal and
//     means "every stage is served by another body"; the caller owns the claim
//     that every output was written. stages == 0 with p.dq_accum set is refused
//     (see the dQ item below).
//   * A cleared stage writes nothing here and records no tag here: the body that
//     served it writes every element of its outputs and records its own phase
//     row. The readout invariant is one row per phase of a backward -- scalar for
//     delta, mma_sync or umma for dQ/dK/dV, and none/skipped only for dQ on the
//     atomics route. No row means a lost phase; two rows mean a double record.
//   * Delta is an input to everything after it: every later stage reads
//     p.delta. Clearing kStageDelta is legal only when another body wrote it.
//   * dQ ownership on the atomics route: when p.dq_accum is non-null the fused
//     dK+dV stage reduces dQ as part of its own work, so kStageDq and kStageDkdv
//     are not independent there. Clearing kStageDq does not stop the fused stage
//     from writing the accumulator. The fold the host runs after the call
//     (dq <- dq_accum * scale) is the SOLE writer of dq on that route: a body that
//     serves dQ there must accumulate into p.dq_accum, never write dq directly,
//     because a direct write is overwritten by the fold and is invisible to any
//     numeric check. A mask that clears both kStageDq and kStageDkdv while an
//     accumulator is set leaves filling it to another body or to a later call of
//     this function, and nothing here can check that: an unfilled accumulator
//     folds exact zeros into dq. LaunchBackwardPhases relies on the later call
//     (kStageDelta first, then the products). Two forms of the mistake are visible
//     here and refused: kStageDq with an accumulator but without kStageDkdv, and
//     stages == 0 with an accumulator, which launches nothing and which no caller
//     passes.
void LaunchBackwardStages(const Params& params, cudaStream_t stream, unsigned stages);

}  // namespace xattn::ops::flash_sca_native
