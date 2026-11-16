"""Shared SoftDelta cases on B300.

Upstream xattn main b112365's helpers of tests/softdelta/test_softdelta_attention.py, verbatim (inputs,
segment ids and BOS masks, the composed-output bound, the gradient checks and their relative-norm bound,
the D/V tables); the B300 gate; the level-1 refusal that replaces upstream's level-1 parametrizations
(precision level 1 has no native Blackwell route and must raise before anything runs); and the
determinism trio every case runs on B300, where SoftDelta is deterministic whatever ``deterministic``
says: the forward repeats bit-exactly, the backward repeats bit-exactly from the same forward and from a
fresh forward, and the other ``deterministic`` flag gives the same bits.
"""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from softdelta_main_reference import ATOL, GRAD_ATOL, RTOL, softdelta_fp64_oracle

DTYPES = (torch.float16, torch.bfloat16)
# Refusals the current test asserted (read by conftest's native-route proof, which resets it).
REFUSALS = []
LEVEL1_REFUSAL = 'high_precision_level=1 has no native Blackwell route'
WINDOW_REFUSAL = 'no native Blackwell route for a sliding window'


def require_native_b300():
    """B300 gate: a compute capability 10.3 GPU with the native Blackwell FlashSCA kernels and the
    SoftDelta gate kernels linked (replaces upstream's SM90 gate)."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if importlib.util.find_spec('xattn_cuda') is None:
        pytest.skip('xattn CUDA extension is not built')
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip('B300 (compute capability 10.3) required')
    import xattn_cuda
    assert hasattr(xattn_cuda.ops, '_flash_sca_native_fwd'), 'native FlashSCA kernels are not linked'
    assert hasattr(xattn_cuda.ops, '_softdelta_gate_fwd'), 'SoftDelta gate kernels are not linked'


def expect_refusal(call, match=LEVEL1_REFUSAL, error=NotImplementedError):
    """``call`` must raise ``error`` (matching ``match``) before anything runs; the proof fixture then
    requires that no kernel was launched by a test whose only calls were refused."""
    with pytest.raises(error, match=match):
        call()
    REFUSALS.append(match)


def _bits(t):
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def assert_same_bits(first, second, names):
    for name, a, b in zip(names, first, second, strict=True):
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(_bits(a), _bits(b)), f'{name} is not bit-identical'


def autograd_determinism(call, leaves, y_grad, deterministic, y, grads):
    """The trio of an autograd call ``call(deterministic) -> y`` over ``leaves`` whose first forward gave
    ``y`` and whose backward with ``y_grad`` gave ``grads``."""
    names = list(leaves)
    again = call(deterministic)
    assert_same_bits((y,), (again,), ('y (forward repeat)',))
    fresh = torch.autograd.grad(again, list(leaves.values()), y_grad, retain_graph=True)
    assert_same_bits(grads, fresh, [f'{n} (backward after a fresh forward)' for n in names])
    same = torch.autograd.grad(again, list(leaves.values()), y_grad)
    assert_same_bits(fresh, same, [f'{n} (backward repeat from the same forward)' for n in names])
    flipped = call(not deterministic)
    assert_same_bits((y,), (flipped,), (f'y (deterministic={not deterministic})',))
    assert_same_bits(grads, torch.autograd.grad(flipped, list(leaves.values()), y_grad),
                     [f'{n} (deterministic={not deterministic})' for n in names])


def standalone_determinism(forward, backward, deterministic, state, gradients):
    """The trio of a standalone pair ``forward(deterministic) -> (y, neg_read, lse)`` and
    ``backward(state, deterministic) -> gradients``, whose first calls gave ``state`` and ``gradients``."""
    names = [f'grad{i}' for i in range(len(gradients))]
    again = forward(deterministic)
    assert_same_bits(state, again, ('y', 'neg_read', 'lse'))
    assert_same_bits(gradients, backward(state, deterministic), [f'{n} (same forward)' for n in names])
    assert_same_bits(gradients, backward(again, deterministic), [f'{n} (fresh forward)' for n in names])
    flipped = forward(not deterministic)
    assert_same_bits(state, flipped, ('y', 'neg_read', 'lse'))
    assert_same_bits(gradients, backward(flipped, not deterministic),
                     [f'{n} (deterministic={not deterministic})' for n in names])


# ---- upstream tests/softdelta/test_softdelta_attention.py helpers, verbatim ----------------------------

def _inputs(*, batch, length, q_heads, kv_heads, qk_dim, value_dim, dtype,
            device="cuda", with_k_read=False, q_length=None, prev_length=0):
    q_length = length if q_length is None else q_length
    q = torch.randn(batch, q_length, q_heads, qk_dim, dtype=dtype, device=device)
    k = torch.randn(batch, length, kv_heads, qk_dim, dtype=dtype, device=device)
    v = torch.randn(batch, length, kv_heads, value_dim, dtype=dtype, device=device)
    g = torch.randn(batch, q_length, q_heads, dtype=dtype, device=device)
    k_read = torch.randn(batch, q_length, kv_heads, qk_dim, dtype=dtype, device=device) if with_k_read else None
    with torch.no_grad():
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.silu(v) + 0.1
        g = g * 0.75
        k_read = None if k_read is None else F.normalize(F.silu(k_read), dim=-1)
    prev_k = prev_v = None
    if prev_length:
        prev_k = F.normalize(torch.randn(batch, prev_length, kv_heads, qk_dim, dtype=dtype, device=device), dim=-1)
        prev_v = F.silu(torch.randn(batch, prev_length, kv_heads, value_dim, dtype=dtype, device=device)) + 0.1
    return dict(q=q, k=k, v=v, g=g, k_read=k_read, prev_k=prev_k, prev_v=prev_v)


def _segments(batch, length, lengths, device="cuda"):
    """Contiguous segment ids from a run-length list, repeated to cover ``length``."""
    rows = []
    for b in range(batch):
        ids, sid, i = [], 0, b
        while len(ids) < length:
            ids += [sid] * lengths[i % len(lengths)]
            sid += 1
            i += 1
        rows.append(ids[:length])
    return torch.tensor(rows, dtype=torch.int64, device=device)


def _assert_composed_close(actual, expected, read, neg_read, g, dtype):
    q_heads, kv_heads = read.shape[2], neg_read.shape[2]
    neg_read = neg_read.repeat_interleave(q_heads // kv_heads, dim=2)
    gate = torch.sigmoid(g.double())
    gate = gate if gate.dim() == read.dim() else gate[..., None]  # scalar gates [B, L, H]
    difference = (actual.double() - expected).abs()
    bound = (
        ATOL[dtype] + RTOL[dtype] * read.abs()
        + gate * (ATOL[dtype] + RTOL[dtype] * neg_read.abs())
        + ATOL[dtype] + RTOL[dtype] * expected.abs()
    )
    normalized = difference / bound
    assert bool((normalized <= 1).all().item()), (
        f"maximum composed normalized error={normalized.max().item()} "
        f"maximum absolute error={difference.max().item()}"
    )


def _oracle_metadata(metadata, q_length):
    """Segment ids of the call's metadata for the oracle."""
    oracle_metadata = dict(metadata)
    if "segment_idx" in oracle_metadata:
        segment_idx = oracle_metadata.pop("segment_idx")
        oracle_metadata["k_segment_idx"] = segment_idx
        oracle_metadata["q_segment_idx"] = segment_idx[:, -q_length:]
    if "bos_mask" in oracle_metadata:
        bos_mask = oracle_metadata.pop("bos_mask")
        segment_idx = bos_mask.cumsum(dim=1)
        oracle_metadata["k_segment_idx"] = segment_idx
        oracle_metadata["q_segment_idx"] = segment_idx[:, -q_length:]
    return oracle_metadata


def _oracle(tensors, metadata):
    """(expected, read, neg_read, FP64 leaves) over the call's already-quantized tensors."""
    leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = softdelta_fp64_oracle(
        **leaves, return_aux=True, **_oracle_metadata(metadata, tensors["q"].shape[1]))
    return expected, read, neg_read, leaves


def _assert_output(actual, expected, read, neg_read, g, dtype):
    assert actual.shape == expected.shape and actual.dtype == dtype
    assert bool(torch.isfinite(actual).all().item())
    _assert_composed_close(actual, expected, read, neg_read, g, dtype)


def _assert_gradients(names, actual_grads, expected_grads, dtype):
    for name, a, e in zip(names, actual_grads, expected_grads):
        assert a.dtype == dtype and a.shape == e.shape, name
        torch.testing.assert_close(a, e.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype], msg=lambda m: f"{name}: {m}")
        # Elementwise tolerances alone can hide a missing gradient term; bound the relative norm error too.
        floor = GRAD_ATOL[dtype] * e.numel() ** 0.5  # exact-zero gradients fall back to the atol check
        relative = ((a.double() - e).norm() / e.norm().clamp_min(floor)).item()
        assert relative < GRAD_REL_NORM[dtype], f"{name}: relative norm error {relative}"


GRAD_REL_NORM = {torch.float16: 5e-3, torch.bfloat16: 3e-2}
NEG_IDS = ("neg-key", "neg-given")
# (deterministic, high_precision_level): every backward algorithm.
ALL_MODES = tuple((det, level) for level in range(2) for det in (False, True))
MODE_IDS = [f"{'det' if det else 'nondet'}-level{level}" for det, level in ALL_MODES]


def _bos_mask(batch, length, lengths):
    ids = _segments(batch, length, lengths)
    return ids.diff(dim=1, prepend=torch.full((batch, 1), -1, dtype=torch.int64, device="cuda")) != 0


# (D, V) across the fused buckets: exact buckets, dims rounded up to a bucket, the
# dense forward full-span tiles ((32, 64), (192, 128), (256, 32)) and the dense backward square remaps
# ((64, 96) -> (96, 96); (160, 128), (152, 136) -> (160, 160), (192, 128), (192, 160) -> (192, 192);
# (128, 256), (256, 128) -> (256, 256)).
FUSED_DIMS = {
    "d32v32": (32, 32),
    "d32v64": (32, 64),
    "d40v56": (40, 56),
    "d64v96": (64, 96),
    "d80v72": (80, 72),
    "d128v64": (128, 64),
    "d152v136": (152, 136),
    "d160v128": (160, 128),
    "d192v128": (192, 128),
    "d192v160": (192, 160),
    "d192v192": (192, 192),
    "d128v256": (128, 256),
    "d256v32": (256, 32),
    "d256v128": (256, 128),
    "d256v256": (256, 256),
    "d64v256": (64, 256),
}


DEFAULT_DIMS = {"d32v32", "d40v56", "d64v96", "d256v128"}
DIM_CASES = [
    pytest.param(name, id=name, marks=() if name in DEFAULT_DIMS else pytest.mark.full_matrix)
    for name in FUSED_DIMS
]


def _rope(x, base=500000.0):
    """Adjacent-pair RoPE (2m, 2m + 1) over positions 0..L-1 in FP32 (FP64 for FP64 inputs)."""
    compute = torch.float64 if x.dtype == torch.float64 else torch.float32
    length, dim = x.shape[1], x.shape[-1]
    inverse = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float64) / dim)
    angle = torch.arange(length, device=x.device, dtype=torch.float64)[:, None] * inverse
    cos, sin = (f(angle).to(compute)[None, :, None, :] for f in (torch.cos, torch.sin))
    even, odd = x[..., 0::2].to(compute), x[..., 1::2].to(compute)
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)


def _negative_query_transforms(source, leaves):
    """(q, k, k_read) from the leaves: post-RoPE queries and keys, and the negative-branch query the
    caller builds (None: the keys; the keys passed explicitly; rope(silu(k)); a second query)."""
    q, k = _rope(leaves["q_pre"]), _rope(leaves["k_pre"])
    k_read = {
        "key": None,
        "explicit-key": k,
        "silu-key": _rope(F.silu(leaves["k_pre"])),
        "second-query": _rope(leaves["q2_pre"]) if "q2_pre" in leaves else None,
    }[source]
    return q, k, k_read
