// Hopper GMMA row-statistics contraction, split out of row_statistics.cu so that
// an SM100/SM103 build never compiles it: row_statistics_sm90.cuh builds
// cute::GMMA tiled MMAs (wgmma), and below sm_90a cute lowers those to an
// invalid-control-path trap. Also note the warpgroup mapping in that header needs
// __CUDA_ARCH_FEAT_SM90_ALL, so this object belongs in the SM90 source list only.
//
// The kernel, its shared-memory layout and the tile selection below are the seed
// ones, unchanged; only the operand plumbing is type-erased so the arch-neutral
// caller can reach it through row_statistics_sm90_launch.h.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include "attention/precision/row_statistics_sm90.cuh"
#include "attention/precision/row_statistics_sm90_launch.h"

namespace xattn::ops::attention::precision {

void RowStatisticsSm90Launch(const void* q_ptr, const void* k_ptr, const void* v_ptr,
                             const void* g_ptr, const int* left, const int* right,
                             const int64_t* qs, const int64_t* ks, const int* exact,
                             float* delta, float* logz, float* meta, int nq, int nk, int h,
                             int hk, int d, int dv, int batch, float scale, int dtype,
                             cudaStream_t stream) {
  const auto type = dtype == 0 ? at::kHalf : at::kBFloat16;
  AT_DISPATCH_REDUCED_FLOATING_TYPES(type, "precision_row_statistics_sm90", [&] {
    using E = std::conditional_t<std::is_same_v<scalar_t, c10::Half>, cutlass::half_t, cutlass::bfloat16_t>;
    auto rectangular = [&](auto bucket, auto vbucket, auto tag) {
      constexpr int D = decltype(bucket)::value, DV = decltype(vbucket)::value;
      constexpr bool segments = decltype(tag)::value;
      // One K/V stage when it raises CTAs per SM (D+DV between 128 and 512);
      // 64/64 and 256/256 keep the double-buffered ring.
      constexpr int Stages = D + DV > 128 && D + DV < 512 ? 1 : 2;
      // Two warpgroups share each K/V tile (128 query rows per CTA) when two such
      // CTAs still fit per SM; each row's arithmetic is independent of Groups.
      constexpr int Groups = sizeof(StatisticsShared<E, D, DV, Stages, 2>) <= 113 * 1024 ? 2 : 1;
      auto kernel = RowStatisticsSm90<E, D, DV, segments, Stages, Groups>;
      constexpr int smem = sizeof(StatisticsShared<E, D, DV, Stages, Groups>);
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      // 1-D grid; the kernel orders query blocks heaviest first.
      const dim3 grid(unsigned(((nq + 64 * Groups - 1) / (64 * Groups)) * batch * h));
      kernel<<<grid, 128 * Groups, smem, stream>>>(
          reinterpret_cast<const E*>(q_ptr), reinterpret_cast<const E*>(k_ptr),
          reinterpret_cast<const E*>(v_ptr), reinterpret_cast<const E*>(g_ptr), left, right,
          qs, ks, exact, delta, logz, meta, nq, nk, h, hk, d, dv, batch, scale);
    };
    auto segments = [&](auto bucket, auto vbucket) {
      if (qs)
        rectangular(bucket, vbucket, std::true_type{});
      else
        rectangular(bucket, vbucket, std::false_type{});
    };
    auto values = [&](auto bucket) {
      if (dv <= 64)
        segments(bucket, std::integral_constant<int, 64>{});
      else if (dv <= 128)
        segments(bucket, std::integral_constant<int, 128>{});
      else
        segments(bucket, std::integral_constant<int, 256>{});
    };
    if (d <= 64)
      values(std::integral_constant<int, 64>{});
    else if (d <= 128)
      values(std::integral_constant<int, 128>{});
    else
      values(std::integral_constant<int, 256>{});
  });
}

}  // namespace xattn::ops::attention::precision
