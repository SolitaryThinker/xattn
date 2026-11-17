#include "attention/native_cuda/reader_params.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace xattn::ops::attention::native_cuda {
namespace {

constexpr int kTile = 16;
constexpr int kWidth = 256;
constexpr int kThreads = 32;

// Every CTA has one warp. The matrix products and their register ownership
// are identical in forward, row-normalizer reconstruction and both gradients.
template <typename E>
struct alignas(32) TileStorage {
  alignas(32) E a[kTile * kWidth];
  alignas(32) E b[kTile * kWidth];
  alignas(32) float score[kTile * kTile];
  alignas(32) float dp[kTile * kTile];
  float probability[kTile * kTile];
  float ds[kTile * kTile];
  float accum0[kTile * kWidth];
  float accum1[kTile * kWidth];
  float maximum[kTile];
  float normalizer[kTile];
  float rescale[kTile];
};

template <typename E>
__device__ __forceinline__ float ToFloat(E value);
template <>
__device__ __forceinline__ float ToFloat(__half value) {
  return __half2float(value);
}
template <>
__device__ __forceinline__ float ToFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename E>
__device__ __forceinline__ E FromFloat(float value);
template <>
__device__ __forceinline__ __half FromFloat(float value) {
  return __float2half_rn(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 FromFloat(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float Accumulate(float a, float b, float c) {
  float result;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;"
               : "=f"(result) : "f"(a), "f"(b), "f"(c));
  return result;
}

__device__ __forceinline__ int64_t Offset(
    int64_t batch, int64_t row, int64_t head, int64_t length,
    int64_t heads, int64_t width) {
  return ((batch * length + row) * heads + head) * width;
}

__device__ __forceinline__ int64_t RowOffset(
    const ReaderShape& s, int64_t batch, int64_t head, int64_t row) {
  return (batch * s.query_heads + head) * s.q_length + row;
}

__device__ __forceinline__ bool Admitted(
    const ReaderInputs& in, int64_t batch, int64_t row, int64_t key) {
  const auto& s = in.shape;
  if (row >= s.q_length || key >= s.k_length || key < 0 ||
      key > row + s.k_length - s.q_length - int64_t(in.strict_past)) {
    return false;
  }
  return !in.q_segment_idx ||
      in.q_segment_idx[batch * s.q_length + row] ==
          in.k_segment_idx[batch * s.k_length + key];
}

// The difference is nonpositive for the stable softmax. Keep scale in double:
// a finite nonzero public scale need not be representable as a nonzero float.
__device__ __forceinline__ float RelativeWeight(
    float score, float maximum, double scale) {
  return static_cast<float>(exp((double(score) - double(maximum)) * scale));
}

template <typename E>
__device__ __forceinline__ void DotTile(
    TileStorage<E>& sm, const E* a, const E* b, int64_t batch,
    int64_t row_a, int64_t row_b, int64_t head_a, int64_t head_b,
    int64_t length_a, int64_t length_b, int64_t heads_a,
    int64_t heads_b, int width, float* result) {
  for (int x = threadIdx.x; x < kTile * kWidth; x += kThreads) {
    const int row = x / kWidth;
    const int col = x % kWidth;
    sm.a[x] = row_a + row < length_a && col < width
        ? a[Offset(batch, row_a + row, head_a, length_a, heads_a, width) + col]
        : FromFloat<E>(0.0f);
    sm.b[x] = row_b + row < length_b && col < width
        ? b[Offset(batch, row_b + row, head_b, length_b, heads_b, width) + col]
        : FromFloat<E>(0.0f);
  }
  __syncwarp();
  namespace wmma = nvcuda::wmma;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, E, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, E, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
  wmma::fill_fragment(fc, 0.0f);
  for (int col = 0; col < width; col += 16) {
    wmma::load_matrix_sync(fa, sm.a + col, kWidth);
    wmma::load_matrix_sync(fb, sm.b + col, kWidth);
    wmma::mma_sync(fc, fa, fb, fc);
  }
  wmma::store_matrix_sync(result, fc, kTile, wmma::mem_row_major);
  __syncwarp();
}

template <typename E>
__device__ __forceinline__ void InitNormalizer(TileStorage<E>& sm) {
  if (threadIdx.x < kTile) {
    sm.maximum[threadIdx.x] = -CUDART_INF_F;
    sm.normalizer[threadIdx.x] = 0.0f;
    sm.rescale[threadIdx.x] = 0.0f;
  }
  __syncwarp();
}

template <typename E>
__device__ __forceinline__ void UpdateNormalizer(
    TileStorage<E>& sm, const ReaderInputs& in, int64_t batch,
    int64_t query0, int64_t key0) {
  const int row = threadIdx.x;
  if (row < kTile) {
    float maximum = sm.maximum[row];
    bool any = sm.normalizer[row] > 0.0f;
    for (int j = 0; j < kTile; ++j) {
      if (Admitted(in, batch, query0 + row, key0 + j)) {
        maximum = any ? fmaxf(maximum, sm.score[row * kTile + j])
                      : sm.score[row * kTile + j];
        any = true;
      }
    }
    const float rescale = sm.normalizer[row] > 0.0f
        ? RelativeWeight(sm.maximum[row], maximum, in.scale) : 0.0f;
    float normalizer = sm.normalizer[row] * rescale;
    for (int j = 0; j < kTile; ++j) {
      const float p = Admitted(in, batch, query0 + row, key0 + j)
          ? RelativeWeight(sm.score[row * kTile + j], maximum, in.scale)
          : 0.0f;
      sm.probability[row * kTile + j] = p;
      normalizer += p;
    }
    sm.maximum[row] = maximum;
    sm.normalizer[row] = normalizer;
    sm.rescale[row] = rescale;
  }
  __syncwarp();
}

template <typename E>
__global__ void ReaderForwardKernel(ReaderFwdParams p) {
  extern __shared__ __align__(32) unsigned char bytes[];
  auto& sm = *reinterpret_cast<TileStorage<E>*>(bytes);
  const auto& in = p.inputs;
  const auto& s = in.shape;
  const int64_t tiles = (s.q_length + kTile - 1) / kTile;
  const int64_t query0 = (int64_t(blockIdx.x) % tiles) * kTile;
  const int64_t head = (int64_t(blockIdx.x) / tiles) % s.query_heads;
  const int64_t batch = int64_t(blockIdx.x) / (tiles * s.query_heads);
  const int64_t kh = head / (s.query_heads / s.kv_heads);
  const auto* q = static_cast<const E*>(in.q);
  const auto* k = static_cast<const E*>(in.k);
  const auto* v = static_cast<const E*>(in.v);
  InitNormalizer(sm);
  for (int x = threadIdx.x; x < kTile * s.value_dim; x += kThreads)
    sm.accum0[(x / s.value_dim) * kWidth + x % s.value_dim] = 0.0f;
  __syncwarp();
  for (int64_t key0 = 0; key0 < s.k_length; key0 += kTile) {
    DotTile(sm, q, k, batch, query0, key0, head, kh,
            s.q_length, s.k_length, s.query_heads, s.kv_heads, int(s.qk_dim), sm.score);
    UpdateNormalizer(sm, in, batch, query0, key0);
    for (int x = threadIdx.x; x < kTile * s.value_dim; x += kThreads) {
      const int row = x / s.value_dim;
      const int col = x % s.value_dim;
      float value = sm.accum0[row * kWidth + col] * sm.rescale[row];
      for (int j = 0; j < kTile; ++j) {
        const float probability = sm.probability[row * kTile + j];
        if (probability != 0.0f) {
          value = Accumulate(probability,
              ToFloat(v[Offset(batch, key0 + j, kh, s.k_length, s.kv_heads, s.value_dim) + col]),
              value);
        }
      }
      sm.accum0[row * kWidth + col] = value;
    }
    __syncwarp();
  }
  for (int x = threadIdx.x; x < kTile * s.value_dim; x += kThreads) {
    const int row = x / s.value_dim;
    const int col = x % s.value_dim;
    if (query0 + row < s.q_length) {
      const float value = sm.normalizer[row] > 0.0f
          ? sm.accum0[row * kWidth + col] / sm.normalizer[row] : 0.0f;
      const int64_t dst = Offset(batch, query0 + row, head, s.q_length, s.query_heads, s.value_dim) + col;
      static_cast<E*>(p.output)[dst] = FromFloat<E>(value);
      if (p.output_state) p.output_state[dst] = value;
    }
  }
  if (threadIdx.x < kTile && query0 + threadIdx.x < s.q_length) {
    const int row = threadIdx.x;
    p.lse[RowOffset(s, batch, head, query0 + row)] = sm.normalizer[row] > 0.0f
        ? static_cast<float>(double(sm.maximum[row]) * in.scale +
                             log(double(sm.normalizer[row])))
        : -CUDART_INF_F;
  }
}

template <typename E>
__global__ void ReaderBackwardRowsKernel(ReaderBwdParams p) {
  extern __shared__ __align__(32) unsigned char bytes[];
  auto& sm = *reinterpret_cast<TileStorage<E>*>(bytes);
  const auto& in = p.inputs;
  const auto& s = in.shape;
  const int64_t tiles = (s.q_length + kTile - 1) / kTile;
  const int64_t query0 = (int64_t(blockIdx.x) % tiles) * kTile;
  const int64_t head = (int64_t(blockIdx.x) / tiles) % s.query_heads;
  const int64_t batch = int64_t(blockIdx.x) / (tiles * s.query_heads);
  const int64_t kh = head / (s.query_heads / s.kv_heads);
  InitNormalizer(sm);
  for (int64_t key0 = 0; key0 < s.k_length; key0 += kTile) {
    DotTile(sm, static_cast<const E*>(in.q), static_cast<const E*>(in.k),
            batch, query0, key0, head, kh, s.q_length, s.k_length, s.query_heads, s.kv_heads,
            int(s.qk_dim), sm.score);
    UpdateNormalizer(sm, in, batch, query0, key0);
  }
  if (threadIdx.x < kTile && query0 + threadIdx.x < s.q_length) {
    const int row = threadIdx.x;
    const int64_t idx = RowOffset(s, batch, head, query0 + row);
    float delta = 0.0f;
    const bool live = sm.normalizer[row] > 0.0f;
    if (live) {
      const int64_t base = Offset(batch, query0 + row, head, s.q_length, s.query_heads, s.value_dim);
      for (int col = 0; col < s.value_dim; ++col) {
        const float state = in.high_precision_level == 0
            ? ToFloat(static_cast<const E*>(p.saved_output)[base + col])
            : static_cast<const float*>(p.saved_output)[base + col];
        delta = Accumulate(ToFloat(static_cast<const E*>(p.dy)[base + col]), state, delta);
      }
    }
    p.delta[idx] = delta;
    p.row_max[idx] = live ? sm.maximum[row] : 0.0f;
    p.inv_normalizer[idx] = live ? 1.0f / sm.normalizer[row] : 0.0f;
  }
}

template <typename E>
__device__ __forceinline__ void BackwardTile(
    TileStorage<E>& sm, const ReaderBwdParams& p, int64_t batch,
    int64_t head, int64_t kh, int64_t query0, int64_t key0) {
  const auto& in = p.inputs;
  const auto& s = in.shape;
  DotTile(sm, static_cast<const E*>(in.q), static_cast<const E*>(in.k),
          batch, query0, key0, head, kh, s.q_length, s.k_length, s.query_heads, s.kv_heads,
          int(s.qk_dim), sm.score);
  DotTile(sm, static_cast<const E*>(p.dy), static_cast<const E*>(in.v),
          batch, query0, key0, head, kh, s.q_length, s.k_length, s.query_heads, s.kv_heads,
          int(s.value_dim), sm.dp);
  for (int x = threadIdx.x; x < kTile * kTile; x += kThreads) {
    const int row = x / kTile;
    const int col = x % kTile;
    float probability = 0.0f;
    float ds = 0.0f;
    if (Admitted(in, batch, query0 + row, key0 + col)) {
      const int64_t idx = RowOffset(s, batch, head, query0 + row);
      if (p.inv_normalizer[idx] > 0.0f) {
        probability = RelativeWeight(sm.score[x], p.row_max[idx], in.scale) *
                      p.inv_normalizer[idx];
        ds = probability * (sm.dp[x] - p.delta[idx]);
      }
    }
    sm.probability[x] = probability;
    sm.ds[x] = ds;
  }
  __syncwarp();
}

template <typename E>
__global__ void ReaderQueryGradientKernel(ReaderBwdParams p) {
  extern __shared__ __align__(32) unsigned char bytes[];
  auto& sm = *reinterpret_cast<TileStorage<E>*>(bytes);
  const auto& in = p.inputs;
  const auto& s = in.shape;
  const int64_t tiles = (s.q_length + kTile - 1) / kTile;
  const int64_t query0 = (int64_t(blockIdx.x) % tiles) * kTile;
  const int64_t head = (int64_t(blockIdx.x) / tiles) % s.query_heads;
  const int64_t batch = int64_t(blockIdx.x) / (tiles * s.query_heads);
  const int64_t kh = head / (s.query_heads / s.kv_heads);
  const auto* k = static_cast<const E*>(in.k);
  for (int x = threadIdx.x; x < kTile * s.qk_dim; x += kThreads)
    sm.accum0[(x / s.qk_dim) * kWidth + x % s.qk_dim] = 0.0f;
  __syncwarp();
  for (int64_t key0 = 0; key0 < s.k_length; key0 += kTile) {
    BackwardTile(sm, p, batch, head, kh, query0, key0);
    for (int x = threadIdx.x; x < kTile * s.qk_dim; x += kThreads) {
      const int row = x / s.qk_dim;
      const int col = x % s.qk_dim;
      float value = sm.accum0[row * kWidth + col];
      for (int j = 0; j < kTile; ++j) {
        const float ds = sm.ds[row * kTile + j];
        if (ds != 0.0f)
          value = Accumulate(ds,
              ToFloat(k[Offset(batch, key0 + j, kh, s.k_length, s.kv_heads, s.qk_dim) + col]), value);
      }
      sm.accum0[row * kWidth + col] = value;
    }
    __syncwarp();
  }
  for (int x = threadIdx.x; x < kTile * s.qk_dim; x += kThreads) {
    const int row = x / s.qk_dim;
    const int col = x % s.qk_dim;
    if (query0 + row < s.q_length)
      static_cast<E*>(p.dq)[Offset(batch, query0 + row, head, s.q_length, s.query_heads, s.qk_dim) + col] =
          FromFloat<E>(static_cast<float>(double(sm.accum0[row * kWidth + col]) * in.scale));
  }
}

template <typename E>
__global__ void ReaderKeyValueGradientKernel(ReaderBwdParams p) {
  extern __shared__ __align__(32) unsigned char bytes[];
  auto& sm = *reinterpret_cast<TileStorage<E>*>(bytes);
  const auto& in = p.inputs;
  const auto& s = in.shape;
  const int64_t tiles = (s.k_length + kTile - 1) / kTile;
  const int64_t key0 = (int64_t(blockIdx.x) % tiles) * kTile;
  const int64_t kh = (int64_t(blockIdx.x) / tiles) % s.kv_heads;
  const int64_t batch = int64_t(blockIdx.x) / (tiles * s.kv_heads);
  const int64_t ratio = s.query_heads / s.kv_heads;
  const auto* q = static_cast<const E*>(in.q);
  const auto* dy = static_cast<const E*>(p.dy);
  for (int x = threadIdx.x; x < kTile * s.qk_dim; x += kThreads)
    sm.accum0[(x / s.qk_dim) * kWidth + x % s.qk_dim] = 0.0f;
  for (int x = threadIdx.x; x < kTile * s.value_dim; x += kThreads)
    sm.accum1[(x / s.value_dim) * kWidth + x % s.value_dim] = 0.0f;
  __syncwarp();
  for (int64_t head = kh * ratio; head < (kh + 1) * ratio; ++head) {
    for (int64_t query0 = 0; query0 < s.q_length; query0 += kTile) {
      BackwardTile(sm, p, batch, head, kh, query0, key0);
      for (int x = threadIdx.x; x < kTile * s.qk_dim; x += kThreads) {
        const int key = x / s.qk_dim;
        const int col = x % s.qk_dim;
        float value = sm.accum0[key * kWidth + col];
        for (int row = 0; row < kTile; ++row) {
          const float ds = sm.ds[row * kTile + key];
          if (ds != 0.0f)
            value = Accumulate(ds,
                ToFloat(q[Offset(batch, query0 + row, head, s.q_length, s.query_heads, s.qk_dim) + col]), value);
        }
        sm.accum0[key * kWidth + col] = value;
      }
      for (int x = threadIdx.x; x < kTile * s.value_dim; x += kThreads) {
        const int key = x / s.value_dim;
        const int col = x % s.value_dim;
        float value = sm.accum1[key * kWidth + col];
        for (int row = 0; row < kTile; ++row) {
          const float probability = sm.probability[row * kTile + key];
          if (probability != 0.0f)
            value = Accumulate(probability,
                ToFloat(dy[Offset(batch, query0 + row, head, s.q_length, s.query_heads, s.value_dim) + col]), value);
        }
        sm.accum1[key * kWidth + col] = value;
      }
      __syncwarp();
    }
  }
  for (int x = threadIdx.x; x < kTile * s.qk_dim; x += kThreads) {
    const int key = x / s.qk_dim;
    const int col = x % s.qk_dim;
    if (key0 + key < s.k_length)
      static_cast<E*>(p.dk)[Offset(batch, key0 + key, kh, s.k_length, s.kv_heads, s.qk_dim) + col] =
          FromFloat<E>(static_cast<float>(double(sm.accum0[key * kWidth + col]) * in.scale));
  }
  for (int x = threadIdx.x; x < kTile * s.value_dim; x += kThreads) {
    const int key = x / s.value_dim;
    const int col = x % s.value_dim;
    if (key0 + key < s.k_length)
      static_cast<E*>(p.dv)[Offset(batch, key0 + key, kh, s.k_length, s.kv_heads, s.value_dim) + col] =
          FromFloat<E>(sm.accum1[key * kWidth + col]);
  }
}

void CheckCuda(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string("native reader CUDA: ") + cudaGetErrorString(result));
}

int GridSize(int64_t batch, int64_t heads, int64_t length) {
  const int64_t tiles = (length + kTile - 1) / kTile;
  if (batch <= 0 || heads <= 0 || tiles <= 0 ||
      batch > std::numeric_limits<int>::max() / heads ||
      batch * heads > std::numeric_limits<int>::max() / tiles)
    throw std::invalid_argument("native reader launch grid is outside the supported range");
  return int(batch * heads * tiles);
}

void CheckInputs(const ReaderInputs& in) {
  const auto& s = in.shape;
  if (s.qk_dim <= 0 || s.qk_dim > kWidth || s.value_dim <= 0 || s.value_dim > kWidth ||
      s.qk_dim % 8 || s.value_dim % 8 || s.kv_heads <= 0 || s.query_heads <= 0 || s.query_heads % s.kv_heads ||
      !(in.scale > 0.0) || !std::isfinite(in.scale) ||
      in.high_precision_level < 0 || in.high_precision_level > 4)
    throw std::invalid_argument("unsupported native reader dimensions, scale or precision level");
  // Admitted() dereferences k_segment_idx for every candidate key as soon as
  // q_segment_idx is non-null (the causal bound is checked first, so the id
  // load happens on the very first admitted key). A half-supplied pair would
  // therefore be an illegal address rather than a refused call; reject it here
  // so a leaf launched without the host boundary cannot be driven into it.
  if ((in.q_segment_idx == nullptr) != (in.k_segment_idx == nullptr))
    throw std::invalid_argument(
        "native reader q_segment_idx and k_segment_idx must be both present or both null");
}

template <typename E>
void Forward(const ReaderFwdParams& p, cudaStream_t stream) {
  CheckCuda(cudaFuncSetAttribute(ReaderForwardKernel<E>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(TileStorage<E>))));
  ReaderForwardKernel<E><<<GridSize(p.inputs.shape.batch, p.inputs.shape.query_heads, p.inputs.shape.q_length),
      kThreads, sizeof(TileStorage<E>), stream>>>(p);
  CheckCuda(cudaGetLastError());
}

template <typename E>
void Backward(const ReaderBwdParams& p, cudaStream_t stream) {
  const auto& s = p.inputs.shape;
  const int qgrid = GridSize(s.batch, s.query_heads, s.q_length);
  const int kgrid = GridSize(s.batch, s.kv_heads, s.k_length);
  CheckCuda(cudaFuncSetAttribute(ReaderBackwardRowsKernel<E>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(TileStorage<E>))));
  CheckCuda(cudaFuncSetAttribute(ReaderQueryGradientKernel<E>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(TileStorage<E>))));
  CheckCuda(cudaFuncSetAttribute(ReaderKeyValueGradientKernel<E>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(TileStorage<E>))));
  ReaderBackwardRowsKernel<E><<<qgrid, kThreads, sizeof(TileStorage<E>), stream>>>(p);
  CheckCuda(cudaGetLastError());
  ReaderQueryGradientKernel<E><<<qgrid, kThreads, sizeof(TileStorage<E>), stream>>>(p);
  CheckCuda(cudaGetLastError());
  ReaderKeyValueGradientKernel<E><<<kgrid, kThreads, sizeof(TileStorage<E>), stream>>>(p);
  CheckCuda(cudaGetLastError());
}

}  // namespace

void LaunchReaderFwd(const ReaderFwdParams& p, cudaStream_t stream) {
  CheckInputs(p.inputs);
  // The POD contract is explicit: output_state is absent at level 0 and is the
  // FP32 accumulator state at levels 1..4. Enforce it here so a leaf launched
  // without the host boundary cannot write an unrequested FP32 state at level 0
  // or return a level 1 call with no state for the paired backward to read. The
  // saved state's dtype IS the level, so a mismatched pair is silent: backward
  // takes saved_output as an untyped pointer and cannot tell FP32 from input
  // dtype, which is why the forward side is the only place this can be refused.
  if ((p.inputs.high_precision_level == 0) != (p.output_state == nullptr))
    throw std::invalid_argument(
        "native reader high_precision_level 0 must not be given an output_state, and levels 1..4 must be given one: "
        "the saved state's dtype is the precision level, so a mismatched pair cannot be caught downstream");
  if (p.inputs.scalar_type == ReaderScalarType::kHalf) Forward<__half>(p, stream);
  else if (p.inputs.scalar_type == ReaderScalarType::kBFloat16) Forward<__nv_bfloat16>(p, stream);
  else throw std::invalid_argument("unsupported native reader element type");
}

void LaunchReaderBwd(const ReaderBwdParams& p, cudaStream_t stream) {
  CheckInputs(p.inputs);
  if (p.inputs.high_precision_level > 1)
    throw std::invalid_argument("basic native reader backward supports levels 0 and 1 only");
  if (!p.delta || !p.row_max || !p.inv_normalizer)
    throw std::invalid_argument("native reader backward requires row scratch");
  if (p.inputs.scalar_type == ReaderScalarType::kHalf) Backward<__half>(p, stream);
  else if (p.inputs.scalar_type == ReaderScalarType::kBFloat16) Backward<__nv_bfloat16>(p, stream);
  else throw std::invalid_argument("unsupported native reader element type");
}

}  // namespace xattn::ops::attention::native_cuda
