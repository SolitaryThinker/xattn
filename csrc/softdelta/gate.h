#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

// y[b, t, h, c, :] = read[b, t, h, c, :] - sigmoid(gate[b, t, h, c]) * neg_read[b, t, h / R, c, :]
// read [B, L, H, C, Dg] and neg_read [B, L, Hkv, C, Dg] (the gate dtype, or both FP32),
// gate [B, L, H, C, 1] contiguous. read, neg_read and output may be head slices of the
// readers [B, L, Hkv + H, C, Dg] (packed heads, unit column stride); output, in the gate
// dtype, may alias read (the in-place gate). Without output a contiguous y is returned.
torch::Tensor SoftDeltaGateFwd(
    const torch::Tensor& read,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate,
    const c10::optional<torch::Tensor>& output);

// The read recovered from the gated output: y + sigmoid(gate) * neg_read, contiguous
// [B, L, H, C, Dg] in the gate dtype (operands as in SoftDeltaGateFwd). ``precomputed``, when
// given, is returned unchanged with no kernel launch: SoftDeltaGateBwd's fused call (given its
// own ``y``) already wrote this value in the same pass as the gate backward, over the same
// neg_read and gate; the two-call sequence (gate_bwd then gate_restore) stays as callers, route
// traces and launch counts see it, with the second call's cost now one Python/C++ crossing.
torch::Tensor SoftDeltaGateRestore(
    const torch::Tensor& y,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate,
    const c10::optional<torch::Tensor>& precomputed = c10::nullopt);

// (neg_read_grad [B, L, Hkv, C, Dg], gate_grad [B, L, H, C, 1], read) in the gate dtype: the read
// gradient is dy itself. neg_read_grad sums the R query heads of each KV head in a fixed order.
// dy is contiguous; neg_read (gate dtype or FP32) may be a head slice of the readers. With ``y``
// (the query reads' gated output, dy's shape, maybe a head slice of the readers) the call also
// recovers the restored read (SoftDeltaGateRestore's value, from the same neg_read and gate this
// call already reads for the gate gradient) as the third return; without ``y`` the third return
// is an empty tensor and the call is exactly the unfused gate backward.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SoftDeltaGateBwd(
    const torch::Tensor& dy,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate,
    const c10::optional<torch::Tensor>& y = c10::nullopt);

// SoftDeltaGateBwd's (neg_read_grad, gate_grad) and, instead of the restored read, the two reads'
// backward deltas: delta_read [B, H, L] = rowsum(dy * read) and delta_neg [B, Hkv, L] =
// rowsum(neg_read_grad * neg_read) in FP32, bit-equal to what the native backward preprocessing
// computes from the read SoftDeltaGateBwd would restore and from neg_read_grad (gate.cu,
// SoftDeltaGateBwdDeltaKernel). Takes one value group of V = 64, 128 or 256 with 16-bit, 16-byte
// aligned rows; returns four undefined tensors (and runs nothing) otherwise.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> SoftDeltaGateBwdDeltas(
    const torch::Tensor& dy,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate,
    const torch::Tensor& y);

// Sums the two readers' K/V gradients in FP32, then rounds once:
//   key_grad = read_key_grad + neg_read_key_grad (+ dneg_q on the last Lq rows)
//   value_grad = read_value_grad + neg_read_value_grad
// All [B, Lk, Hkv, *]; dneg_q [B, Lq, Hkv, D] or None. 16-bit read gradients are updated in place;
// FP32 read accumulators give new gradients in the negative read's dtype.
std::tuple<torch::Tensor, torch::Tensor> SoftDeltaMergeGradients(
    torch::Tensor read_key_grad,
    const torch::Tensor& neg_read_key_grad,
    const c10::optional<torch::Tensor>& dneg_q,
    torch::Tensor read_value_grad,
    const torch::Tensor& neg_read_value_grad);

}  // namespace ops
}  // namespace xattn
