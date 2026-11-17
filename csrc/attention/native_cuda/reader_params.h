#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace xattn::ops::attention::native_cuda {

enum class ReaderScalarType : int32_t { kHalf, kBFloat16 };

// Logical extents of aligned, contiguous rank-four reader operands. K/V
// include any previous sequence. Q/K have width qk_dim; V has value_dim.
struct ReaderShape {
  int64_t batch;
  int64_t q_length;
  int64_t k_length;
  int64_t query_heads;
  int64_t kv_heads;
  int64_t qk_dim;
  int64_t value_dim;
};

struct ReaderInputs {
  ReaderShape shape;
  ReaderScalarType scalar_type;
  const void* q;
  const void* k;
  const void* v;
  // Both absent, or contiguous [B,Nq]/[B,Nt] int64 IDs. Equality is checked
  // independently of the causal bound; IDs need not be monotone.
  const int64_t* q_segment_idx;
  const int64_t* k_segment_idx;
  // Positive finite scale after exact host Q sign/zero normalization.
  double scale;
  bool strict_past;
  int32_t high_precision_level;
};

struct ReaderFwdParams {
  ReaderInputs inputs;
  // Contiguous [B,Nq,Hq,Dv] in the input dtype.
  void* output;
  // Absent at level 0; actual FP32 accumulator state at levels 1..4.
  float* output_state;
  // Contiguous [B,Hq,Nq], natural log; empty rows store -infinity.
  float* lse;
};

struct ReaderBwdParams {
  ReaderInputs inputs;
  // dy is contiguous [B,Nq,Hq,Dv] in the input dtype. saved_output has
  // the same shape and input dtype at level 0, FP32 at level 1.
  const void* dy;
  const void* saved_output;
  const float* lse;
  bool deterministic;
  // Basic level-0/1 gradients are in the input dtype, contiguous, with
  // the respective Q/Ktotal/Vtotal shapes. dK/dV sum every mapped query
  // head's contributions; their head dimension remains Hkv. The two readers
  // use distinct output buffers; the caller sums their dK/dV in FP32 before
  // the final public input-dtype cast.
  void* dq;
  void* dk;
  void* dv;
  // Three host-allocated FP32 scratch planes, each [B,Hq,Nq]. They are
  // initialized on the supplied stream before either gradient contraction.
  // delta is sum_c dy_c * saved_output_c in FP32; the saved state retains
  // its level-specific dtype. The raw maximum and inverse normalizer are
  // recomputed using the forward contraction and visibility, not recovered
  // by subtracting a rounded LSE. Empty rows store delta=0, max=0, inv=0;
  // consumers test inv=0 before exponentiation. These are not the anchored
  // metadata/statistics of the separate advanced-precision backward.
  float* delta;
  float* row_max;
  float* inv_normalizer;
};

// CUDA-only leaf entry points. They allocate no Torch tensors, do not
// synchronize with the host, and use only the explicitly supplied stream.
// Host validation/packing/allocation and dtype dispatch precede them.
// FWD accepts levels 0..4; basic BWD accepts only 0/1. All empty-row output
// and gradient semantics are defined in reader.h. No Nq*Nt workspace.
void LaunchReaderFwd(const ReaderFwdParams& params, cudaStream_t stream);
void LaunchReaderBwd(const ReaderBwdParams& params, cudaStream_t stream);

}  // namespace xattn::ops::attention::native_cuda
