// Empty bucket bodies for the stage-mask refusal probe.
//
// LaunchBackwardStages (csrc/flash_sca/native/dispatch.cu) validates its stage mask in the
// TORCH_CHECKs at its top and only then branches into nine (D, Dv) buckets per element.
// The probe reaches those guards and nothing else, so every bucket body may be empty: no
// kernel runs, no device buffer has to be valid, and no extension has to be built.
//
// That is the whole point of this file. Linking the shipped xattn_cuda.so instead needs a
// build that takes minutes, and on a tree without build/ such a runner can only exit 2 with
// "no built extension", leaving the launcher's guards unchecked there. With these bodies
// the probe compiles against the launcher as it stands in the tree in seconds.
//
// What this does NOT do, stated where the next reader will look for it: it is a
// SOURCE-level fact about the launcher in this tree, not a fact about the binary the
// package ships. Where a built extension exists, linking it is the stronger check and the
// two can disagree only if the tree and the build differ.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flash_sca/native/params.h"

namespace xattn::ops::flash_sca_native {

// Primary templates, defined here and nowhere else: dispatch.cu declares them without a
// body and calls them, so the explicit instantiations below satisfy its references at
// link time with the same mangled names.
template <typename E, int kD, int kDv>
void ForwardCase(const Params&, cudaStream_t) {}

template <typename E, int kD, int kDv>
void BackwardStages(const Params&, cudaStream_t, unsigned) {}

// The nine bucket pairs of DispatchBuckets / DispatchBackwardBuckets, for both elements.
// ForwardCase is stubbed too because LaunchForward lives in the same translation unit and
// its references are live whether or not the probe calls it.
#define XATTN_SCA_REFUSAL_STUB(E, D, DV)                                       \
  template void ForwardCase<E, D, DV>(const Params&, cudaStream_t);            \
  template void BackwardStages<E, D, DV>(const Params&, cudaStream_t, unsigned);

#define XATTN_SCA_REFUSAL_STUB_ELEMENT(E) \
  XATTN_SCA_REFUSAL_STUB(E, 64, 64)       \
  XATTN_SCA_REFUSAL_STUB(E, 64, 128)      \
  XATTN_SCA_REFUSAL_STUB(E, 64, 256)      \
  XATTN_SCA_REFUSAL_STUB(E, 128, 64)      \
  XATTN_SCA_REFUSAL_STUB(E, 128, 128)     \
  XATTN_SCA_REFUSAL_STUB(E, 128, 256)     \
  XATTN_SCA_REFUSAL_STUB(E, 256, 64)      \
  XATTN_SCA_REFUSAL_STUB(E, 256, 128)     \
  XATTN_SCA_REFUSAL_STUB(E, 256, 256)

XATTN_SCA_REFUSAL_STUB_ELEMENT(__half)
XATTN_SCA_REFUSAL_STUB_ELEMENT(__nv_bfloat16)

#undef XATTN_SCA_REFUSAL_STUB_ELEMENT
#undef XATTN_SCA_REFUSAL_STUB

}  // namespace xattn::ops::flash_sca_native
