"""Upstream xattn main b112365's FP64 oracle for SoftDelta, verbatim: tests/_reference.py (the masks, the
reads and the shared tolerances ATOL / GRAD_ATOL / RTOL) followed by tests/softdelta/reference.py
(softdelta_fp64_oracle). Copied so that the frozen B300 suite needs nothing from the tree under test; the
tolerances are unchanged."""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

ATOL = {
    torch.float16: 2e-4,
    torch.bfloat16: 1e-3,
}

GRAD_ATOL = {
    torch.float16: 2e-4,
    torch.bfloat16: 2e-3,
}

RTOL = {
    torch.float16: 1e-3,
    torch.bfloat16: 1e-2,
}


def _contiguous_runs(values) -> Dict[int, Tuple[int, int]]:
    result = {}
    start = 0
    for position in range(1, len(values) + 1):
        if position == len(values) or values[position] != values[start]:
            segment = values[start]
            assert segment not in result
            result[segment] = (start, position)
            start = position
    return result


def _allowed_masks(
    q_length: int,
    k_length: int,
    batch: int,
    device: torch.device,
    visibility: str,
    span: Optional[int],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    reset_position_per_segment: bool,
) -> Tuple[Tensor, Tensor]:
    if q_segment_idx is None and not reset_position_per_segment:
        q_position = torch.arange(q_length, device=device)[:, None]
        q_position = q_position + k_length - q_length
        k_position = torch.arange(k_length, device=device)[None, :]
        if visibility == "full":
            left = torch.ones((q_length, k_length), device=device, dtype=torch.bool)
        elif visibility == "window":
            assert span is not None
            left = k_position >= q_position - span
        elif visibility == "chunk":
            assert span is not None
            left = k_position >= (q_position // span) * span - span
        else:
            raise AssertionError(visibility)
        return (
            (left & (k_position <= q_position)).expand(batch, -1, -1),
            (left & (k_position < q_position)).expand(batch, -1, -1),
        )

    # Keep the independent literal mask oracle, but build its bits on the CPU.
    # Per-element CUDA assignments turn a small reference into thousands of
    # kernel launches and dominate comprehensive correctness sweeps.
    count = batch * q_length * k_length
    if count == 0:
        empty = torch.zeros(batch, q_length, k_length, dtype=torch.bool, device=device)
        return empty, torch.zeros_like(empty)
    inclusive_bits = bytearray(count)
    strict_bits = bytearray(count)

    for batch_idx in range(batch):
        if q_segment_idx is None:
            q_segments = [batch_idx] * q_length
            k_segments = [batch_idx] * k_length
        else:
            assert k_segment_idx is not None
            q_segments = q_segment_idx[batch_idx].detach().cpu().tolist()
            k_segments = k_segment_idx[batch_idx].detach().cpu().tolist()

        if reset_position_per_segment:
            q_runs = _contiguous_runs(q_segments)
            k_runs = _contiguous_runs(k_segments)
        else:
            q_runs = k_runs = {}

        for q_index in range(q_length):
            q_segment = q_segments[q_index]
            if reset_position_per_segment:
                q_start, q_end = q_runs[q_segment]
                k_start, k_end = k_runs.get(q_segment, (0, 0))
                q_position = (
                    q_index
                    - q_start
                    + (k_end - k_start)
                    - (q_end - q_start)
                )
            else:
                q_position = q_index + k_length - q_length

            for k_index in range(k_length):
                if k_segments[k_index] != q_segment:
                    continue
                if reset_position_per_segment:
                    k_position = k_index - k_runs[q_segment][0]
                else:
                    k_position = k_index

                if visibility == "full":
                    in_left_range = True
                elif visibility == "window":
                    assert span is not None
                    in_left_range = k_position >= q_position - span
                elif visibility == "chunk":
                    assert span is not None
                    chunk_begin = (q_position // span) * span
                    in_left_range = k_position >= chunk_begin - span
                else:
                    raise AssertionError(visibility)

                if in_left_range and k_position <= q_position:
                    inclusive_bits[(batch_idx * q_length + q_index) * k_length + k_index] = 1
                if in_left_range and k_position < q_position:
                    strict_bits[(batch_idx * q_length + q_index) * k_length + k_index] = 1
    return (
        torch.frombuffer(inclusive_bits, dtype=torch.bool).reshape(batch, q_length, k_length).to(device),
        torch.frombuffer(strict_bits, dtype=torch.bool).reshape(batch, q_length, k_length).to(device),
    )


def _attention_from_mask(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed: Tensor,
    scale: float,
) -> Tensor:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)

    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    allowed_heads = allowed.unsqueeze(1)
    masked_scores = scores.masked_fill(~allowed_heads, float("-inf"))
    nonempty = allowed.any(dim=-1, keepdim=True).unsqueeze(1)
    scores = torch.where(nonempty, masked_scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = probabilities.masked_fill(~allowed_heads, 0.0)
    return torch.einsum("bhqk,bkhv->bqhv", probabilities, v)


# ---- tests/softdelta/reference.py ----
def softdelta_fp64_oracle(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    k_read: Optional[Tensor] = None,
    *,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    visibility: str = "full",
    span: Optional[int] = None,
    reset_position_per_segment: bool = False,
    return_aux: bool = False,
):
    """Independent FP64 oracle over caller-provided quantized values.

    The negative read uses ``k_read`` (default: the aligned current keys) as a
    per-KV-head query over the strict past; every query head of the group
    subtracts its gated copy. ``visibility`` ("full", "window" or "chunk") and
    ``span`` select the SoftDelta oracle's support for both reads.

    Scalar gates (the public layout: ``v`` ``[B, L, Hkv, V]``, ``g`` ``[B, Lq, H]``)
    or the internal value groups (``v`` ``[B, L, Hkv, C, Dg]``, ``g``
    ``[B, Lq, H, C, 1]``); the outputs follow the input layout.
    """
    tensors = (q, k, v, g) if k_read is None else (q, k, v, g, k_read)
    assert all(t.dtype == torch.float64 for t in tensors)
    assert (prev_k is None) == (prev_v is None)
    scalar = v.dim() == 4
    if scalar:
        v, g = v.unsqueeze(3), g[..., None, None]
        prev_v = None if prev_v is None else prev_v.unsqueeze(3)
    batch, length, q_heads, _ = q.shape
    if k_read is None:
        k_read = k[:, k.shape[1] - length:]
    if prev_k is not None:
        assert prev_v is not None
        k = torch.cat((prev_k, k), dim=1)
        v = torch.cat((prev_v, v), dim=1)
    kv_heads = k.shape[2]
    groups, group_dim = v.shape[3:]
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    inclusive, strict_past = _allowed_masks(
        length, k.shape[1], batch, q.device, visibility, span,
        q_segment_idx, k_segment_idx, reset_position_per_segment,
    )
    assert bool(inclusive.any(dim=-1).all())
    flat_v = v.flatten(3)
    read = _attention_from_mask(q, k, flat_v, inclusive, resolved_scale)
    neg_read = _attention_from_mask(k_read, k, flat_v, strict_past, resolved_scale)
    rep = q_heads // kv_heads
    read = read.reshape(batch, length, kv_heads, rep, groups, group_dim)
    neg_read = neg_read.reshape(batch, length, kv_heads, 1, groups, group_dim)
    gate = torch.sigmoid(g).reshape(batch, length, kv_heads, rep, groups, 1)
    y = (read - gate * neg_read).reshape(batch, length, q_heads, groups, group_dim)
    read = read.reshape(batch, length, q_heads, groups, group_dim)
    neg_read = neg_read.reshape(batch, length, kv_heads, groups, group_dim)
    if scalar:
        y, read, neg_read = y.flatten(3), read.flatten(3), neg_read.flatten(3)
    return (y, read, neg_read) if return_aux else y
