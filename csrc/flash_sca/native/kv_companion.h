// The strict-past companion reader of the fused dK/dV key leaf (BackwardKvPipeKernel, kComp).
//
// The native SoftDelta backward reads K/V twice: the grouped read (Hq query heads over Hkv KV
// heads, inclusive mask) and the strict read (one read key per KV head, strict-past mask), with
// one dK/dV that is the sum of both. With a companion, the grouped call's key leaf takes the read
// key of KV head hk as one more reader of its CTA's key tile: after the group's heads of a query
// tile, one more (tile, head) whose Q is the read key, whose dO is the negative read's gradient
// and whose column table holds the strict read's LSE, delta and row bounds. Both reads then
// accumulate in the same TMEM accumulators, one issuer, in round order (a fixed order), and the
// key gradients are rounded once, with no FP32 round trip and no merge.
#pragma once

#include <cuda.h>

#include <cstdint>

#include "flash_sca/native/umma_params.h"

namespace xattn::ops::flash_sca_native {
namespace umma {

struct KvCompanion {
  CUtensorMap q{};                  // the read keys [B, Nq, Hkv, D] (the strict read's Q map)
  CUtensorMap dy{};                 // its output gradient [B, Nq, Hkv, Dv] (its dO map)
  const float* lse = nullptr;       // [B, Hkv, Nq] FP32, natural log
  const float* delta = nullptr;     // [B, Hkv, Nq] FP32, rowsum(dO * O) of the strict read
  const int32_t* row_lo = nullptr;  // [B, Nq] strict row bounds; null: the plain chunk rule
  const int32_t* row_hi = nullptr;
};

// The key leaf's kernel parameter: the launch plus the companion (unused when kComp is false).
struct KvPipeLaunch {
  UmmaLaunchParams lp;
  KvCompanion comp;
};

// (128,128) grouped calls at level 0 with 16-bit key gradients: the preflight of the companion
// launch, and the launch. `false` means refused, nothing written.
bool CanLaunchUmmaBackwardKvPipeCompanion(const UmmaLaunchParams& lp);
bool LaunchUmmaBackwardKvPipeCompanion(const UmmaLaunchParams& lp, const KvCompanion& comp,
                                       cudaStream_t stream);

}  // namespace umma
}  // namespace xattn::ops::flash_sca_native
