// Host-only dispatch registry for the mounted level 0-1 backward leaves.
#pragma once

#include "flash_sca/native/dimension_bucket.h"
#include "flash_sca/native/umma_params.h"

namespace xattn::ops::flash_sca_native::umma {

enum class BackwardProduct { kDq, kDkdv, kAtomicDq };

// Keep the compiled domain independent of a particular call's eligibility.
// Real dimensions, formats, operands, maps and device resources are checked
// by the selected entry's complete predicate, never by this bucket test.
#define XATTN_SCA_BACKWARD_BUCKETS(M) M(128, 128) M(256, 256)
// The buckets served by the pipelined dQ and fused M128 key leaves templated on (D, Dv)
// (umma_bwd_pipe_small.cuh), with no fused in-situ body: their registry rows name those leaves'
// level-0 preflights directly, and every call they refuse runs the generic body.
#define XATTN_SCA_BACKWARD_PIPE_BUCKETS(M) M(64, 64) M(64, 128) M(128, 64)
// The buckets at the four unequal (D, Dv) pairs the public API accepts with D + Dv > 256 and
// D, Dv <= 256 (the dQ leg first, then the key leg). The fused M128 key leaf's
// map holds dK's D and dV's Dv in TMEM columns 0..255 and refuses every one of these (D + Dv >
// 256), so their key product is instead the dV-only/dK-only M128 pair (umma_bwd_dkdv.cuh:
// kDkdvPairFits, LaunchUmmaBackwardInSituDkdv), a capacity-rule leaf the registry (below) names
// with a kDkdv row at these buckets. Their dQ product takes the pipelined dQ-only leaf, whose
// map comes from (D, Dv) alone (umma_bwd_dq_pipe.cuh:Plan), through the dQ-only entry of the
// same header.
#define XATTN_SCA_BACKWARD_DQ_PIPE_BUCKETS(M) M(256, 64) M(256, 128) M(128, 256) M(64, 256)

inline bool BackwardLeafCompiled(BackwardProduct product, int64_t d, int64_t dv) {
  if (product != BackwardProduct::kDq && product != BackwardProduct::kDkdv) return false;
  const int kd = QkBucket(d), kv = ValueBucket(dv);
#define XATTN_SCA_MATCH_BUCKET(D, V) if (kd == D && kv == V) return true;
  XATTN_SCA_BACKWARD_BUCKETS(XATTN_SCA_MATCH_BUCKET)
  XATTN_SCA_BACKWARD_PIPE_BUCKETS(XATTN_SCA_MATCH_BUCKET)
  // Both products are compiled at these buckets now: the dQ-only pipelined leaf and the
  // dV-only/dK-only M128 pair, named by the same registry row's two function pointers
  // (sca_umma_bwd_registry.cu:XATTN_SCA_DQ_PIPE_ENTRY). This test must change in the same round
  // as that registry row: reporting a product compiled here without a matching row promises a
  // leaf the link does not have.
  XATTN_SCA_BACKWARD_DQ_PIPE_BUCKETS(XATTN_SCA_MATCH_BUCKET)
#undef XATTN_SCA_MATCH_BUCKET
  return false;
}

// The key-tile height M the FUSED in-situ key body enumerates K/V in, per dimension
// bucket: 128 at (128,128) - the one bucket whose TMEM map holds both M128 dK/dV
// accumulators (dK cols 0..127 + dV cols 128..255 + S^T 256..383 + dP^T 384..511 = 512 =
// kAllocCols; umma_bwd_dkdv.cuh:BackwardDkdvPlan) - and 64 (kBwdKeyRows, the M64
// datapath, the default) at every other bucket. 0.9B's (256,256) bucket CANNOT take a
// fused M128 key tile: the same map would be 256 (dK) + 256 (dV) + 256 (S^T) + 256 (dP^T) =
// 1024 columns against the 512 one CTA may allocate, so that bucket refuses the arm
// (BackwardDkdvPlan's static_assert, at compile time) instead of running a 64-row body
// under an M128 label. At level 0 its in-situ launch runs a pair of single-product M128
// leaves instead, dV then dK, each with its own 512-column map (umma_bwd_dkdv.cuh,
// LaunchUmmaBackwardInSituDkdv), and keeps the fused M64 body for every other call.
//
// This is the ONE definition of the fused rule: the in-situ launch template takes the fused
// body's kKeyRows argument from it (umma_bwd_dkdv.cuh:InSituKeyRows) and a registry row
// reports it, so the fused body a call prepares and the one it launches cannot disagree.
// The tile a call actually launched, the (256,256) pair included, is InSituLaunchedKeyRows().
constexpr int kInSituKeyRowsDefault = 64;  // == umma::kBwdKeyRows (device header)

constexpr int InSituKeyRowsFor(int qk_bucket, int value_bucket) {
  return (qk_bucket == 128 && value_bucket == 128) ? 128 : kInSituKeyRowsDefault;
}

// The key-tile height of the body that actually LAUNCHED, per call, per host thread.
//
// `InSituKeyRowsFor` is the arm a bucket's FUSED body PREFERS; it is not always the arm
// that runs. At (128,128) the M128 arm refuses an unpacked previous plane whose length is
// not a multiple of 128 (Accepted's alignment test is the tile height's, umma_bwd_dkdv.cuh),
// and that call must still be served natively - the M64 body serves it - so the
// in-situ launch tries M128 first and falls back to the M64 body in the same TU rather
// than dropping the whole key product to the generic body. At (256,256) the level-0 launch
// runs the single-product M128 pair (128) instead of the fused M64 body the bucket's row
// names. A row's static `key_rows` therefore cannot answer "which tile ran"; this does, and
// it is the field a snapshot must print.
//
// Default-constructed to the M64 body: a call that never launched any key row has run
// nothing, and 64 is the only value that can never be read as a claim about M128. The
// in-situ launch pair sets it on every successful launch (umma_bwd_dkdv.cuh).
inline int& InSituLaunchedKeyRows() {
  static thread_local int value = kInSituKeyRowsDefault;
  return value;
}

// Whether the in-situ key launch runs a pipelined key body instead of the fused key body for this
// call: the (256,256) pair of pipelined M128 single-product leaves (dV-only, then dK-only), or the
// (128,128) pipelined fused M128 leaf (umma_bwd_kv_pipe.cuh). It evaluates the SAME decision the
// launch acts on (umma_bwd_dkdv.cuh:DecideInSituKeyPair) for the call's element type, with the
// XATTN_SCA_BWD_KEY_ARM value of the moment, and is false at every other bucket and for every
// call those bodies refuse. They write dK, dV, dPrevK and dPrevV and nothing else, so host code
// that relies on a by-product of the fused key body (a stored dS^T plane, say) must ask this
// before the launch and must not expect that by-product from a call for which it answers true.
// Defined in instantiations/sca_umma_bwd_dkdv_d256_v256.cu; its (128,128) half,
// InSituKvPipeWouldLaunch, in instantiations/sca_umma_bwd_dkdv_d128_v128.cu.
bool InSituKeyPairWouldLaunch(const UmmaLaunchParams& lp);

struct BackwardLeafEntry {
  bool (*can)(const UmmaLaunchParams&);
  bool (*launch)(const UmmaLaunchParams&, cudaStream_t);
  bool writes_whole_plane;
  // The key-tile height (M) this row's FUSED key body PREFERS, so a preparation snapshot can
  // name the arm the bucket selects instead of assuming it. 64 for every M64 row; 128 on the
  // (128,128) key row. A dQ row enumerates no key tile and keeps the default. This is NOT
  // the arm a given call ran: at (128,128) the M128 arm refuses some calls and the same
  // call falls back to the M64 body, and at (256,256) the level-0 launch runs the
  // single-product M128 pair instead of the fused M64 body, so a snapshot must print
  // InSituLaunchedKeyRows(), which the launch pair sets to the tile it actually launched.
  int key_rows = kInSituKeyRowsDefault;
};

// Query and launch use the same entry. No second switch may silently select
// a different instantiation after final preflight.
const BackwardLeafEntry* FindBackwardLeaf(BackwardProduct product, const Params& params);

}  // namespace xattn::ops::flash_sca_native::umma
