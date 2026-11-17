#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn::ops::attention::native_cuda {

using ForwardResult = std::tuple<torch::Tensor, torch::Tensor>;
using BackwardResult = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>;

// Full-causal, single-reader host boundary. Q is [B,Nq,Hq,D], K is
// [B,Nk,Hkv,D], and V is [B,Nk,Hkv,Dv]. Value groups are flattened here;
// their interpretation belongs to the SoftDelta gate, not to attention.
// Hq must be a positive multiple of Hkv; query head h maps to KV head
// h / (Hq / Hkv). Public metadata validation is unchanged; this internal
// boundary can also represent arbitrary IDs without changing equality.
// Previous K/V, when present, precede current K/V in the logical key space.
// With Nt = Nprev + Nk, key j is visible to row i exactly when
//   0 <= j < Nt && j <= i + Nt - Nq - int(strict_past)
// and, when segments are present, q_segment_idx[b,i] == k_segment_idx[b,j].
// Segment IDs need not be monotone or describe a single run.
//
// These entry points validate before accessing device pointers. Supported
// strided inputs may be packed; contiguous storage does not imply pointer
// alignment. Packing, launches, and gradient assembly use Q's current CUDA
// stream under a guard for Q's device. No default-stream or host-sync contract
// is introduced by this interface.
//
// FWD returns O in Q's dtype and natural-log LSE, FP32 [B,Hq,Nq]. Empty rows
// have O=0 and LSE=-infinity. Level 0 has no output_state; levels 1..4 require
// FP32 output_state [B,Nq,Hq,Dv], written from the FP32 accumulator before
// converting O to Q's dtype. The gate consumes input-dtype O, not the FP32
// state. Its backward rounds O1 to input dtype at every level and emits
// input-dtype dO1 = round(-sigmoid(g) * dY); dGate reduces channels within
// each value group, never across groups. Levels 1..4 use subtract-before-scale softmax.
// This is the common forward algorithm for those levels, not a claim that
// their backward algorithms are interchangeable. Scale is any finite value.
ForwardResult ReaderFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past, int64_t high_precision_level,
    const c10::optional<torch::Tensor>& output_state);

// Basic reader BWD accepts levels 0 and 1 only. Saved state is O in Q's dtype
// for level 0, and the actual FP32 output_state for level 1. The caller must
// pair FWD/BWD options, including scale, metadata, strictness and level.
// dy has Q's dtype and shape [B,Nq,Hq,Dv]. LSE is as returned by FWD.
// Returns dQ,dK,dV,dPrevK,dPrevV in Q's dtype with the corresponding input
// shapes; absent previous inputs produce absent previous gradients.
//
// Empty rows contribute exactly zero to every gradient. Validity must be
// checked before evaluating a softmax expression; masking a NaN afterwards
// is insufficient. For original scale=0, dQ and dK (including dPrevK) are
// explicitly zero; dV and the separately computed dGate still use the
// uniform-softmax readers and are not zeroed. Requested deterministic
// execution is bit-identical across repeated launches with identical inputs
// and has a fixed reduction order for every output, including GQA accumulation.
//
// This function rejects levels 2..4. Those levels use separate precision
// RowStatistics/Main/Ordered entry points, whose FP32 branch gradients are
// combined before the public dtype cast. FP32 state dtype is not permission
// to treat levels 2..4 as level 1.
BackwardResult ReaderBwd(
    const torch::Tensor& dy, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& saved_output, const torch::Tensor& lse,
    double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past, int64_t high_precision_level, bool deterministic);

}  // namespace xattn::ops::attention::native_cuda
