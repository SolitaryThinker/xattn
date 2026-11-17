// Type-erased host entries refer to the existing per-bucket instantiations.
// Keeping device definitions out of this TU avoids recompiling the kernels
// just to select one of them.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flash_sca/native/umma_bwd_entry.h"

namespace xattn::ops::flash_sca_native::umma {

template <class E, int D, int V>
bool CanLaunchUmmaDq(const UmmaLaunchParams&);
template <class E, int D, int V>
bool LaunchUmmaDq(const UmmaLaunchParams&, cudaStream_t);
// The DEFINITION of these two lives in umma_bwd_dkdv.cuh:1003-1005 / :1052-1054 and is
//   template <typename E, int kD, int kDv, int kWrite = 3, int kKeyRows = kBwdKeyRows, int ABL = 0>
// so a declaration here must carry EVERY parameter. A shorter form names a DIFFERENT template that
// no translation unit defines: the reference survives the link, and the .so then fails at import
// with `undefined symbol: ...UmmaBackwardDkdvCanLaunchI6__halfLi128ELi128ELi3EEE...` while every
// instantiation TU has emitted `...Li3ELi64ELi0EE...`. kBwdKeyRows is 64; it is spelled here rather
// than included because this TU deliberately holds no device code (see the header comment above).
template <class E, int D, int V, int W = 3, int K = 64, int A = 0>
bool UmmaBackwardDkdvCanLaunch(const UmmaLaunchParams&);
template <class E, int D, int V, int W = 3, int K = 64, int A = 0>
bool LaunchUmmaBackwardDkdv(const UmmaLaunchParams&, cudaStream_t);

// The IN-SITU pair the kDkdv rows below name: the same fused leaf at the bucket's own
// key-tile height - `InSituKeyRowsFor` in umma_bwd_entry.h, 128 at (128,128) and 64
// (kBwdKeyRows, the M64 default) at every other bucket, 0.9B's (256,256) included, which
// cannot hold both M128 accumulators in TMEM (dK 0..255 + dV 256..511 + S^T + dP^T = 1024
// columns against 512) and therefore must refuse that arm rather than run 64 rows under
// an M128 label. (At (256,256) and level 0 the launch runs a pair of single-product M128
// leaves instead of the fused body; see umma_bwd_dkdv.cuh, LaunchUmmaBackwardInSituDkdv.)
// Declared with exactly THREE parameters and NO defaults, so this TU has
// no shorter form of these names to get wrong: the M is a template argument of the
// DEFINITION (umma_bwd_dkdv.cuh, InSituKeyRows<kD, kDv>::value) and never a value spelled
// here. A green link still proves nothing on its own - the (128,128) TU must emit these
// explicit instantiations (XATTN_UMMA_BWD_DKDV_M128_INSTANTIATE) or the .so fails at
// import, exactly as the dkdv 4-parameter mismatch above once did.
template <class E, int D, int V>
bool CanLaunchUmmaBackwardInSituDkdv(const UmmaLaunchParams&);
template <class E, int D, int V>
bool LaunchUmmaBackwardInSituDkdv(const UmmaLaunchParams&, cudaStream_t);

namespace {
struct Row {
  BackwardProduct product;
  Element element;
  int d, dv;
  BackwardLeafEntry entry;
};
// The key row's fourth field is the key-tile height its FUSED body enumerates K/V in: the
// SAME `InSituKeyRowsFor(D, V)` the in-situ launch takes that body's template argument from,
// so the reported arm and the launched fused body are one decision, not two. Per bucket, not
// per call: (128,128) -> 128, (256,256) -> 64 (M64, the default). The tile a call actually
// launched - the (128,128) M64 fallback and the (256,256) level-0 M128 pair included - is
// umma::InSituLaunchedKeyRows(), which the backward record reports after the launch.
#define XATTN_SCA_ENTRY(E, T, D, V) \
  {BackwardProduct::kDq, E, D, V, \
   {CanLaunchUmmaDq<T, D, V>, LaunchUmmaDq<T, D, V>, true}}, \
  {BackwardProduct::kDkdv, E, D, V, \
   {CanLaunchUmmaBackwardInSituDkdv<T, D, V>, LaunchUmmaBackwardInSituDkdv<T, D, V>, true, \
    InSituKeyRowsFor(D, V)}},
#define XATTN_SCA_BUCKET(D, V) \
  XATTN_SCA_ENTRY(Element::kHalf, __half, D, V) \
  XATTN_SCA_ENTRY(Element::kBFloat16, __nv_bfloat16, D, V)
const Row kEntries[] = {XATTN_SCA_BACKWARD_BUCKETS(XATTN_SCA_BUCKET)};
#undef XATTN_SCA_BUCKET
#undef XATTN_SCA_ENTRY
}  // namespace

const BackwardLeafEntry* FindBackwardLeaf(BackwardProduct product, const Params& p) {
  const int d = QkBucket(p.qk_dim), dv = ValueBucket(p.value_dim);
  for (const Row& row : kEntries) {
    if (row.product == product && row.element == p.element && row.d == d && row.dv == dv) {
      return &row.entry;
    }
  }
  return nullptr;
}

}  // namespace xattn::ops::flash_sca_native::umma
