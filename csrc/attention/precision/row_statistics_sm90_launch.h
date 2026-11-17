#pragma once
// Type-erased entry point of the Hopper GMMA row-statistics contraction.
//
// It is defined in row_statistics_sm90_launch.cu, which a Blackwell-only build
// does not compile. The declaration is weak, so where that object is absent the
// address is null and the caller falls back to the portable native contraction
// instead of failing to link.
#include <cstdint>
#include <cuda_runtime_api.h>

namespace xattn::ops::attention::precision {

// dtype: 0 = FP16, 1 = BF16 (the only types the precision path accepts).
__attribute__((weak)) void RowStatisticsSm90Launch(
    const void* q, const void* k, const void* v, const void* g, const int* left,
    const int* right, const int64_t* qs, const int64_t* ks, const int* exact,
    float* delta, float* logz, float* meta, int nq, int nk, int h, int hk, int d,
    int dv, int batch, float scale, int dtype, cudaStream_t stream);

}  // namespace xattn::ops::attention::precision
