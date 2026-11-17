#pragma once

#include "attention/native_cuda/reader.h"

namespace xattn::ops::attention::native_cuda {

using CudaBackwardResult =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

// Internal launch boundary after host validation and packing. All operands
// are contiguous, aligned to 16 bytes, and on Q's device; K/V contain the
// concatenated previous and current sequences. Metadata is optional as a
// pair: contiguous int64 [B,Nq] and [B,Nt] on the same device. It retains
// exact IDs (including nonmonotone/repeated IDs), not inferred row ranges.
//
// Scale is positive and finite. Host normalization uses exact Q negation
// for negative scales and Q zeroing with scale=1 for zero scale; BWD restores
// the original chain rule. This leaves masked -infinity safe under scaling.
// Output state, when present, is contiguous FP32 [B,Nq,Hq,Dv].
//
// The full-causal mask, state semantics and empty-row convention in reader.h
// apply unchanged. Implementations launch on Q's current CUDA stream and
// may allocate linear-size workspaces, but no Nq*Nt attention matrix.
ForwardResult ReaderFwdCuda(
    const torch::Tensor& q, const torch::Tensor& k_total,
    const torch::Tensor& v_total, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past, int64_t high_precision_level,
    const c10::optional<torch::Tensor>& output_state);

// Levels 0/1 only; output gradients are contiguous tensors in Q's dtype.
// saved_output and LSE are packed without numerical conversion: state dtype
// is Q's dtype at level 0, FP32 at level 1; LSE is FP32 [B,Hq,Nq].
// Returned dK/dV cover the total key sequence. The host boundary splits at
// Nprev and restores the original-scale chain rule before returning them.
CudaBackwardResult ReaderBwdCuda(
    const torch::Tensor& dy, const torch::Tensor& q,
    const torch::Tensor& k_total, const torch::Tensor& v_total,
    const torch::Tensor& saved_output, const torch::Tensor& lse,
    double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past, int64_t high_precision_level, bool deterministic);

}  // namespace xattn::ops::attention::native_cuda
