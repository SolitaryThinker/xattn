// Full-causal and sliding-chunk native advanced backward on the caller's current stream.
// Each invocation handles one reader. Its row-owned dQ/statistics and
// key-owned dK/dV have disjoint outputs and deterministic reduction order.
#include "attention/precision/native/adv_bwd_native.h"
#include "attention/precision/native/adv_bwd_dkdv.h"
#include "flash_sca/native/dispatch_tag.cuh"
#include <c10/util/Exception.h>

namespace xattn::ops::native_adv {
void LaunchMainDq(const Params& p, cudaStream_t stream);

namespace tags = ::xattn::ops::flash_sca_native;

void Launch(const Params& p, cudaStream_t stream) {
  TORCH_CHECK(p.visibility == 0 || p.visibility == 2,
              "native advanced backward implements full-causal and sliding-chunk visibility");
  TORCH_CHECK(p.level >= 2 && p.level <= 4, "native advanced backward implements levels 2-4");
  TORCH_CHECK(p.B >= 0 && p.Nq >= 0 && p.Nt >= 0 && p.Nk == p.Nt &&
                  p.D > 0 && p.D <= 256 && p.Dv > 0 && p.Dv <= 256,
              "native advanced backward: inconsistent B/N/D/Nt");
  TORCH_CHECK(p.Hq > 0 && p.Hkv > 0 && p.Hq % p.Hkv == 0 &&
                  p.qk_ratio == p.Hq / p.Hkv,
              "native advanced backward: invalid GQA mapping");
  TORCH_CHECK(p.B <= 65535 && p.Hq <= 65535 &&
                  p.Nq <= 2147483647LL && p.Nt <= 2147483647LL,
              "native advanced backward launch dimensions exceed CUDA limits");
  // Strong references are intentional: a missing gradient leaf must fail the
  // link, not leave an apparently complete precision feature at runtime.
  //
  // Phase rows for this reader, per dispatch_tag.cuh's contract: a row names the
  // body that carried the phase and is written only after its launcher returned,
  // so a throw inside a launch leaves no success row and absence is not evidence
  // of another body. The advanced leaves hold no matrix instruction -- their
  // arithmetic is plain fmaf over warp-private rows -- so the body class is
  // kScalar for the selection and for the body that ran. A batch-empty call
  // records nothing, as the header requires.
  const bool record = p.B > 0 && p.Nq > 0 && p.Nt > 0;
  LaunchMainDq(p, stream);
  if (record) {
    tags::RecordPhase(tags::Phase::kDq, tags::Body::kScalar, tags::Body::kScalar,
                      tags::State::kRan);
  }
  LaunchDkdv(p, stream);
  if (record) {
    // One fused key launch owns two contractions; the header's rule is one row
    // per phase, so dK and dV each get their own row from this single launch.
    tags::RecordPhase(tags::Phase::kDk, tags::Body::kScalar, tags::Body::kScalar,
                      tags::State::kRan);
    tags::RecordPhase(tags::Phase::kDv, tags::Body::kScalar, tags::Body::kScalar,
                      tags::State::kRan);
  }
}
}  // namespace xattn::ops::native_adv
