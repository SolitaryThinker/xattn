// Launch templates of the native Blackwell FlashSCA kernels for one
// (element, D bucket, Dv bucket) case. Each instantiation file under
// instantiations/ compiles exactly one case so the build parallelizes.
#pragma once

#include <c10/cuda/CUDAException.h>

#include "flash_sca/native/backward_stage_mask.h"
#include "flash_sca/native/dispatch_tag.cuh"
#include "flash_sca/native/kernels.cuh"
#include "flash_sca/native/umma_params.h"

namespace xattn::ops::flash_sca_native {

// Buckets with an explicitly instantiated tcgen05/UMMA forward leaf, read from
// the one list in params.h that host.cc's `HasUmmaForwardLeaf` also reads: the
// host decides whether to prepare the maps for the call, this decides whether to
// consume them. A bucket without a leaf keeps the generic `mma.sync` body exactly
// as before; a prepared call arriving at a bucket that has no leaf is a host bug
// and falls back rather than launching a leaf of the wrong shape.
template <int kD, int kDv>
inline constexpr bool kUmmaForwardLeaf = IsUmmaForwardLeafBucket(kD, kDv);

template <typename E, int kD, int kDv>
void ForwardCase(const Params& p, cudaStream_t stream) {
  // Native body first when the host prepared a call for this bucket. The leaf
  // reports a refusal by returning false before anything is launched
  // (dispatch_tag.cuh's contract for every UMMA entry point), and only then does
  // the generic body run, so a `refused` row can never describe a partial launch.
  if constexpr (kUmmaForwardLeaf<kD, kDv>) {
    if (p.umma_maps != nullptr) {
      umma::UmmaLaunchParams lp;
      lp.params = p;
      lp.params.umma_maps = nullptr;  // host-only field, never a kernel input
      lp.maps = *static_cast<const umma::UmmaInputMaps*>(p.umma_maps);
      if (umma::LaunchUmmaForward<E, kD, kDv>(lp, stream)) {
        RecordPhase(Phase::kForward, Body::kUmma, Body::kUmma, State::kRan);
        return;
      }
      RecordPhase(Phase::kForward, Body::kUmma, Body::kMmaSync, State::kRefused);
    }
  }
  using Config = detail::ForwardConfig<kD, kDv>;
  auto kernel = p.subtract_before_scale ? detail::ForwardKernel<E, kD, kDv, true>
                                        : detail::ForwardKernel<E, kD, kDv, false>;
  // The launch-order scratch (kOrderBytes) rides on the dynamic shared memory
  // tail and is asked for only when the permutation will actually run: the same
  // CTA gate the kernel applies below (p.order_always, or gridDim product <=
  // kOrderMaxCtas), evaluated on the host because the host knows the grid. So
  // the default launch, and the plain `sdesc` arm on a deep grid (the traced
  // train shapes, 16,384 CTAs), request exactly what base requests.
  const unsigned tiles = static_cast<unsigned>((p.q_length + Config::kBM - 1) / Config::kBM);
  const unsigned heads = static_cast<unsigned>(p.query_heads);
  const unsigned ctas = tiles * heads * static_cast<unsigned>(p.batch);
  const bool order_active = p.sorted_order && (p.order_always || ctas <= detail::kOrderMaxCtas);
  const int smem = Config::kSmemBytes + (order_active ? Config::kOrderBytes : 0);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if (p.head_fastest) {
    // Under hf the tile axis is gridDim.y, which CUDA caps at 65535; no traced
    // shape approaches it (512 at N=65536), but a short-chunk contract shape
    // could, and the failure would otherwise surface as a launch error.
    TORCH_CHECK(tiles <= 65535u,
                "XATTN_SCA_LAYOUT=hf needs at most 65535 forward tiles; got ", tiles);
  }
  // Head-fastest enumeration (opt-in). Default (tf) is byte-for-byte the
  // original grid; hf swaps x and y so that the Hq copies of one query tile
  // launch consecutively, which with a rank-ordered x is the LPT schedule of
  // CTA space instead of a per-head replay of it.
  const dim3 grid(p.head_fastest ? heads : tiles, p.head_fastest ? tiles : heads,
                  static_cast<unsigned>(p.batch));
  kernel<<<grid, Config::kThreads, smem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  RecordPhase(Phase::kForward, Body::kMmaSync, Body::kMmaSync, State::kRan);
}

// The backward body as three independently selectable stages, so a native arm
// can serve one phase while the generic body still runs the others. The order
// and every launch below is the seed's: delta, then dQ (row-owned kernel when
// dq_accum is null, else the phase is skipped because the fused dK+dV pass
// reduces dQ with atomics), then one fused dK+dV launch. A mask that clears a
// stage means that stage already ran, in another body; nothing here records a
// tag for it.
// One launch for the backward's native preprocessing: delta, the per-64-row
// query tile bounds and the per-64-key tile range (PreprocessBackwardKernel,
// kernels.cuh). Grid (row tiles, Hq, B) with kLanes threads per query row, so no
// per-thread int64 div/mod. The key range needs its own batch's tile bounds; one
// block per batch computes both in shared memory, so no cross-block dependency
// and no atomics. A second key-range launch runs only when the tile table is too
// large for shared memory.
template <typename E, int kLanes>
inline void LaunchPreprocess(const Params& p, cudaStream_t stream) {
  constexpr int kRowsPerBlock = detail::kPreprocessThreads / kLanes;
  const unsigned row_tiles =
      static_cast<unsigned>((p.q_length + kRowsPerBlock - 1) / kRowsPerBlock);
  const dim3 grid(row_tiles, static_cast<unsigned>(p.query_heads),
                  static_cast<unsigned>(p.batch));
  if (p.key_tile_range != nullptr && p.tile_count <= detail::kPreprocessMaxTileCount) {
    const int smem = static_cast<int>(p.tile_count * 4 * sizeof(int));
    detail::PreprocessBackwardKernel<E, kLanes, true>
        <<<grid, detail::kPreprocessThreads, smem, stream>>>(p);
  } else {
    detail::PreprocessBackwardKernel<E, kLanes, false>
        <<<grid, detail::kPreprocessThreads, 0, stream>>>(p);
    if (p.key_tile_range != nullptr) {
      Params mutable_p = p;
      LaunchKeyTileRange(mutable_p, stream);  // tile bounds were written above
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename E>
void BackwardDeltaStage(const Params& p, cudaStream_t stream) {
  const int64_t rows = p.batch * p.q_length * p.query_heads;
  // Vector path: 16-B loads need dy and saved contiguous, 16-B aligned and a
  // value dim that is a multiple of 8 (every bucket dim is). Otherwise the seed's
  // scalar delta kernel runs, with the two table kernels beside it.
  const bool vectorized = rows > 0 && !p.saved_fp32 && p.dy != nullptr && p.saved != nullptr &&
                          p.delta != nullptr && (p.value_dim % 8) == 0 && p.value_dim >= 8 &&
                          (reinterpret_cast<unsigned long long>(p.dy) % 16) == 0 &&
                          (reinterpret_cast<unsigned long long>(p.saved) % 16) == 0;
  if (vectorized) {
    const int64_t nvec = p.value_dim >> 3;
    if (nvec >= 32) {
      LaunchPreprocess<E, 32>(p, stream);
    } else if (nvec >= 16) {
      LaunchPreprocess<E, 16>(p, stream);
    } else {
      LaunchPreprocess<E, 8>(p, stream);
    }
  } else if (rows > 0) {
    Params mutable_p = p;
    LaunchTileBounds(mutable_p, stream);
    if (p.row_lo != nullptr) LaunchKeyTileRange(mutable_p, stream);
    constexpr int kRowsPerBlock = 8;
    detail::DeltaKernel<E><<<static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock),
                             kRowsPerBlock * 32, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // The delta body is a scalar dot-product/reduction with no T::Mma, so its body
  // is kScalar: labelling it with a tensor-core body would put a false row in the
  // first phase every backward records.
  RecordPhase(Phase::kDelta, Body::kScalar, Body::kScalar, State::kRan);
}

template <typename E, int kD, int kDv>
void BackwardDqStage(const Params& p, cudaStream_t stream) {
  if (p.dq_accum == nullptr) {
    using Config = detail::DqConfig<kD, kDv>;
    auto kernel = detail::DqKernel<E, kD, kDv>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Config::kSmemBytes));
    const dim3 grid(static_cast<unsigned>((p.q_length + Config::kBM - 1) / Config::kBM),
                    static_cast<unsigned>(p.query_heads), static_cast<unsigned>(p.batch));
    kernel<<<grid, Config::kThreads, Config::kSmemBytes, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    RecordPhase(Phase::kDq, Body::kMmaSync, Body::kMmaSync, State::kRan);
  } else {
    // The phase had nothing to do: dQ was accumulated by the caller. A skip is
    // recorded so a reader never mistakes its absence for a refusal or a lost
    // phase, and it names no body -- no selection happened.
    RecordPhase(Phase::kDq, Body::kNone, Body::kNone, State::kSkipped);
  }
}

template <typename E, int kD, int kDv>
void BackwardDkdvStage(const Params& p, cudaStream_t stream) {
  if constexpr (kD + kDv <= 256) {
    using Config = detail::DkdvRegConfig<kD, kDv>;
    auto kernel = detail::DkdvRegKernel<E, kD, kDv>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Config::kSmemBytes));
    const dim3 grid(static_cast<unsigned>((p.total_length + Config::kBN - 1) / Config::kBN),
                    static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
    kernel<<<grid, Config::kThreads, Config::kSmemBytes, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    RecordPhase(Phase::kDk, Body::kMmaSync, Body::kMmaSync, State::kRan);
    RecordPhase(Phase::kDv, Body::kMmaSync, Body::kMmaSync, State::kRan);
  } else {
    using Config = detail::DkdvConfig<kD, kDv>;
    auto kernel = detail::DkdvKernel<E, kD, kDv>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Config::kSmemBytes));
    const dim3 grid(static_cast<unsigned>((p.total_length + Config::kBN - 1) / Config::kBN),
                    static_cast<unsigned>(p.kv_heads), static_cast<unsigned>(p.batch));
    kernel<<<grid, Config::kThreads, Config::kSmemBytes, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    RecordPhase(Phase::kDk, Body::kMmaSync, Body::kMmaSync, State::kRan);
    RecordPhase(Phase::kDv, Body::kMmaSync, Body::kMmaSync, State::kRan);
  }
}

template <typename E, int kD, int kDv>
void BackwardStages(const Params& p, cudaStream_t stream, unsigned stages) {
  if (stages & kStageDelta) BackwardDeltaStage<E>(p, stream);
  if (stages & kStageDq) BackwardDqStage<E, kD, kDv>(p, stream);
  if (stages & kStageDkdv) BackwardDkdvStage<E, kD, kDv>(p, stream);
}

template <typename E, int kD, int kDv>
void BackwardCase(const Params& p, cudaStream_t stream) {
  BackwardStages<E, kD, kDv>(p, stream, kBackwardAllStages);
}

}  // namespace xattn::ops::flash_sca_native

#define XATTN_SCA_NATIVE_INSTANTIATE(ELEMENT, D, DV)                                         \
  namespace xattn::ops::flash_sca_native {                                                   \
  template void ForwardCase<ELEMENT, D, DV>(const Params&, cudaStream_t);                    \
  template void BackwardCase<ELEMENT, D, DV>(const Params&, cudaStream_t);                   \
  template void BackwardStages<ELEMENT, D, DV>(const Params&, cudaStream_t, unsigned);       \
  }
