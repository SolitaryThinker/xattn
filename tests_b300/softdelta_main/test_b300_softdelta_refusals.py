"""What SoftDelta refuses on B300, and what it serves there in place of upstream's SM90 routes.

Refused before anything runs (the proof fixture checks that no kernel was launched): precision level 1
(no native FP32 output state or strict-past precision-repair backward), a sliding window that does not
cover every earlier key (no native sliding-window read), and backend='sm90' (not an SM90 device). A
covering window is causal support, served natively like softdelta_attention.
"""
import pytest
import torch

from xattn import (
    softdelta_attention,
    softdelta_attention_bwd,
    softdelta_attention_fwd,
    sliding_chunk_softdelta_attention,
    sliding_chunk_softdelta_attention_bwd,
    sliding_chunk_softdelta_attention_fwd,
    sliding_window_softdelta_attention,
    sliding_window_softdelta_attention_bwd,
    sliding_window_softdelta_attention_fwd,
)

from softdelta_main_cases import (
    LEVEL1_REFUSAL,
    REFUSALS,
    WINDOW_REFUSAL,
    _inputs,
    _segments,
    expect_refusal,
    require_native_b300,
)

APIS = {
    "causal": (softdelta_attention, softdelta_attention_fwd, softdelta_attention_bwd, {}),
    "chunk": (sliding_chunk_softdelta_attention, sliding_chunk_softdelta_attention_fwd,
              sliding_chunk_softdelta_attention_bwd, {"chunk_size": 32}),
    "window": (sliding_window_softdelta_attention, sliding_window_softdelta_attention_fwd,
               sliding_window_softdelta_attention_bwd, {"window_size": 32}),
}


def _tensors(q_heads=8, kv_heads=2, length=150):
    torch.manual_seed(7)
    inputs = _inputs(batch=2, length=length, q_heads=q_heads, kv_heads=kv_heads, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16, with_k_read=True)
    return {n: inputs[n] for n in ("q", "k", "v", "g", "k_read")}


def _state(tensors, level=0):
    batch, length, heads, _ = tensors["q"].shape
    kv_heads, value_dim = tensors["k"].shape[2], tensors["v"].shape[-1]
    dtype = torch.float32 if level else tensors["q"].dtype
    return dict(y_grad=torch.zeros(batch, length, heads, value_dim, dtype=tensors["q"].dtype, device="cuda"),
                y=torch.zeros(batch, length, heads, value_dim, dtype=tensors["q"].dtype, device="cuda"),
                neg_read=torch.zeros(batch, length, kv_heads, value_dim, dtype=dtype, device="cuda"),
                lse=torch.zeros(batch, kv_heads + heads, length, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("api", ("causal", "chunk"))
def test_level1_refused_before_anything_runs(api):
    require_native_b300()
    autograd, forward, backward, span = APIS[api]
    tensors = _tensors()
    leaves = {n: t.clone().requires_grad_(True) for n, t in tensors.items()}
    expect_refusal(lambda: autograd(**leaves, **span, high_precision_level=1))
    expect_refusal(lambda: forward(**tensors, **span, high_precision_level=1))
    expect_refusal(lambda: backward(**tensors, **span, **_state(tensors, 1), high_precision_level=1))
    for deterministic in (False, True):
        expect_refusal(lambda: autograd(**leaves, **span, high_precision_level=1, deterministic=deterministic))
    assert len(REFUSALS) == 5


def test_window_refused_before_anything_runs():
    require_native_b300()
    autograd, forward, backward, span = APIS["window"]
    tensors = _tensors()
    leaves = {n: t.clone().requires_grad_(True) for n, t in tensors.items()}
    segments = dict(segment_idx=_segments(2, 150, (1, 2, 40, 3, 64, 7)))
    expect_refusal(lambda: autograd(**leaves, **span), WINDOW_REFUSAL)
    expect_refusal(lambda: forward(**tensors, **span, **segments), WINDOW_REFUSAL)
    expect_refusal(lambda: backward(**tensors, **span, **_state(tensors)), WINDOW_REFUSAL)
    expect_refusal(lambda: autograd(**leaves, window_size=0), WINDOW_REFUSAL)
    # Level 1 is refused first, whatever the support.
    expect_refusal(lambda: autograd(**leaves, **span, high_precision_level=1), LEVEL1_REFUSAL)


@pytest.mark.parametrize("api", ("causal", "chunk", "window"))
def test_sm90_backend_refused(api):
    require_native_b300()
    autograd, forward, _, span = APIS[api]
    tensors = _tensors()
    expect_refusal(lambda: autograd(**tensors, **span, backend="sm90"), "requires an SM90 CUDA tensor", RuntimeError)
    expect_refusal(lambda: forward(**tensors, **span, backend="sm90"), "requires an SM90 CUDA tensor", RuntimeError)


@pytest.mark.parametrize("heads", ((8, 2), (4, 4)), ids=("h8kv2", "h4kv4"))
def test_covering_window_runs_as_causal(heads):
    """A window covering every earlier key is causal support: the native composition, bitwise the causal
    API's output and gradients."""
    require_native_b300()
    tensors = _tensors(*heads)
    segments = dict(segment_idx=_segments(2, 150, (1, 2, 40, 3, 64, 7)))
    results = []
    for call in (lambda t: sliding_window_softdelta_attention(**t, window_size=149, **segments),
                 lambda t: softdelta_attention(**t, **segments),
                 lambda t: sliding_chunk_softdelta_attention(**t, chunk_size=150, **segments)):
        leaves = {n: t.clone().requires_grad_(True) for n, t in tensors.items()}
        y = call(leaves)
        results.append((y, *torch.autograd.grad(y, list(leaves.values()), torch.ones_like(y))))
    for other in results[1:]:
        for a, b in zip(results[0], other):
            assert torch.equal(a, b)
