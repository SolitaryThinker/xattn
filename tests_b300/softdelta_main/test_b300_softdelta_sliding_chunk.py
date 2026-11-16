"""Sliding chunk SoftDelta against the FP64 oracle on B300.

The sliding-chunk cases of upstream xattn main b112365's tests/softdelta/test_local_softdelta_attention.py,
ported so that nothing skips on B300: the SM90 gate becomes the B300 gate, the calls take backend 'auto'
(the native composition), tolerances and oracle are upstream's. Every level-0 parametrization runs on the
native route and adds the determinism trio (softdelta_main_cases); every level-1 parametrization asserts
the refusal (precision level 1 has no native Blackwell route), nothing having run. The window variant of
each test is not ported (no native sliding-window read; test_b300_softdelta_refusals checks it raises),
and the SM90-only checks have B300 counterparts: test_native_composition_is_used replaces
test_fused_path_is_used.
"""
import importlib

import pytest
import torch

from xattn import (
    softdelta_attention,
    sliding_chunk_softdelta_attention,
    sliding_chunk_softdelta_attention_bwd,
    sliding_chunk_softdelta_attention_fwd,
    sliding_window_softdelta_attention,
    sliding_window_softdelta_attention_fwd,
)

from softdelta_main_cases import (
    ALL_MODES,
    DIM_CASES,
    FUSED_DIMS,
    MODE_IDS,
    NEG_IDS,
    REFUSALS,
    _assert_gradients,
    _assert_output,
    _bos_mask,
    _inputs,
    _oracle_metadata,
    _segments,
    autograd_determinism,
    expect_refusal,
    require_native_b300,
    standalone_determinism,
)
from softdelta_main_reference import ATOL, RTOL, softdelta_fp64_oracle

SPANS = (1, 16, 64)


def _oracle(span, reset, tensors, metadata):
    """(expected, read, neg_read, FP64 leaves) over the call's already-quantized tensors."""
    leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = softdelta_fp64_oracle(
        **leaves, visibility="chunk", span=span, reset_position_per_segment=reset,
        return_aux=True, **_oracle_metadata(metadata, tensors["q"].shape[1]))
    return expected, read, neg_read, leaves


def _check(span, inputs, dtype, *, reset=False, deterministic=False, high_precision_level=0, **metadata):
    names = [n for n in ("q", "k", "v", "g", "k_read", "prev_k", "prev_v") if inputs[n] is not None]
    actual_inputs = {n: inputs[n].detach().clone().requires_grad_(True) for n in names}

    def call(det, level=high_precision_level):
        return sliding_chunk_softdelta_attention(
            backend="auto", deterministic=det, high_precision_level=level,
            **{'k_read': None, **actual_inputs, 'chunk_size': span, 'reset_chunk_pos_per_seq': reset, **metadata},
        )
    if high_precision_level:
        expect_refusal(lambda: call(deterministic))
        return None
    actual = call(deterministic)
    expected, read, neg_read, expected_inputs = _oracle(span, reset, {n: inputs[n] for n in names}, metadata)
    _assert_output(actual, expected, read, neg_read, inputs["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(actual_inputs.values()), y_grad)
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, actual_grads, expected_grads, dtype)
    autograd_determinism(call, actual_inputs, y_grad, deterministic, actual, actual_grads)
    return actual, actual_grads


HEADS = ((4, 4), (4, 2), (4, 1), (12, 3))
CASES = {
    "length-one": dict(length=1),
    "uneven": dict(length=17),
    "block-boundary": dict(length=129),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
}


def _metadata(spec, batch, total):
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, total, spec["segments"])
    if "bos" in spec:
        metadata["bos_mask"] = _bos_mask(batch, total, spec["bos"])
    return metadata


@pytest.mark.parametrize("span", SPANS, ids=[f"chunk{span}" for span in SPANS])
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("heads", HEADS, ids=[f"h{h}kv{k}" for h, k in HEADS])
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 1)), ids=("level0", "det-level1"))
def test_matches_fp64_oracle(span, case, heads, with_k_read, dtype, deterministic, high_precision_level):
    require_native_b300()
    torch.manual_seed(20261023)
    spec = CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=64,
                     value_dim=64, dtype=dtype, with_k_read=with_k_read)
    _check(span, inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level,
           **_metadata(spec, batch, length))


@pytest.mark.parametrize("q_length", (None, 6), ids=("aligned", "short-q"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_previous_chunk(q_length, with_k_read, dtype, deterministic, high_precision_level):
    """The previous chunk (sliding chunks only) across every backward algorithm."""
    require_native_b300()
    torch.manual_seed(20261024)
    inputs = _inputs(batch=2, length=20, q_heads=4, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=dtype, with_k_read=with_k_read, q_length=q_length, prev_length=8)
    _check(8, inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level)


@pytest.mark.parametrize("q_length", (1, 37, 129), ids=("q1", "q37", "q129"))
@pytest.mark.parametrize("segments", (False, True), ids=("dense", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_short_query_matches_fp64_oracle(q_length, segments, with_k_read, dtype, deterministic,
                                         high_precision_level):
    require_native_b300()
    torch.manual_seed(20261025)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=128, value_dim=128,
                     dtype=dtype, with_k_read=with_k_read, q_length=q_length)
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))} if segments else {}
    _check(48, inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level,
           **metadata)


RESET_METADATA = ("segment_idx", "bos_mask", "q-k-ids")


@pytest.mark.parametrize("metadata_form", RESET_METADATA)
@pytest.mark.parametrize("q_length", (None, 61), ids=("aligned", "short-q"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_chunk_reset_matches_fp64_oracle(metadata_form, q_length, dtype, deterministic, high_precision_level):
    """reset_chunk_pos_per_seq restarts chunk positions in every segment (right-aligned Q / K runs for
    separately given ids; on B300 the native reads serve them directly, SM90's matched-reset bridge is
    not used)."""
    require_native_b300()
    torch.manual_seed(20261026)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=dtype, with_k_read=True, q_length=q_length)
    ids = _segments(batch, length, (5, 70, 2, 33))
    if metadata_form == "segment_idx":
        metadata = {"segment_idx": ids}
    elif metadata_form == "bos_mask":
        metadata = {"bos_mask": _bos_mask(batch, length, (5, 70, 2, 33))}
    else:
        metadata = {"q_segment_idx": ids[:, length - inputs["q"].shape[1]:].clone(), "k_segment_idx": ids}
    _check(24, inputs, dtype, reset=True, deterministic=deterministic,
           high_precision_level=high_precision_level, **metadata)


@pytest.mark.torch_reference
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_torch_backend_matches_oracle(dtype):
    """backend='torch' is still the reference on B300, when asked for (and nothing native runs)."""
    require_native_b300()
    torch.manual_seed(20261027)
    inputs = _inputs(batch=2, length=33, q_heads=4, kv_heads=2, qk_dim=32, value_dim=32,
                     dtype=dtype, device="cuda", with_k_read=True)
    names = [n for n in ("q", "k", "v", "g", "k_read") if inputs[n] is not None]
    leaves = {n: inputs[n].detach().clone().requires_grad_(True) for n in names}
    metadata = dict(segment_idx=_segments(2, 33, (1, 9, 4, 30)))
    actual = sliding_chunk_softdelta_attention(backend="torch", chunk_size=7, **leaves, **metadata)
    expected, read, neg_read, expected_inputs = _oracle(7, False, {n: inputs[n] for n in names}, metadata)
    _assert_output(actual, expected, read, neg_read, inputs["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    _assert_gradients(names, torch.autograd.grad(actual, list(leaves.values()), y_grad),
                      torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double()), dtype)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_covering_span_matches_causal(dtype):
    """One chunk holding the sequence reads like the causal API (both native on B300)."""
    require_native_b300()
    torch.manual_seed(20261028)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=dtype, with_k_read=True)
    tensors = {n: inputs[n] for n in ("q", "k", "v", "g", "k_read")}
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))}
    local = sliding_chunk_softdelta_attention(**{'k_read': None, **tensors, 'chunk_size': length, **metadata})
    causal = softdelta_attention(**{'k_read': None, **tensors, **metadata})
    torch.testing.assert_close(local, causal, atol=ATOL[dtype], rtol=RTOL[dtype])


# Standalone fwd / bwd cases (dense, segments, BOS, short Q, the previous chunk).
STANDALONE_CASES = {
    "dense": dict(length=150),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
    "short-q": dict(length=150, q_length=61),
    "previous": dict(length=40, prev_length=32),
}


@pytest.mark.parametrize("case", list(STANDALONE_CASES), ids=[f"chunk-{case}" for case in STANDALONE_CASES])
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_standalone_fwd_bwd_match_fp64_oracle(case, with_k_read, dtype, deterministic, high_precision_level):
    """The standalone forward returns the autograd output, the negative reads and both reads' LSE; the
    standalone backward turns them into the oracle's gradients, bitwise the autograd ones (B300 is
    deterministic whatever the flag says)."""
    require_native_b300()
    torch.manual_seed(20261029)
    spec = STANDALONE_CASES[case]
    span = 32
    batch, length = 2, spec["length"]
    q_heads, kv_heads = 8, 2
    value_dim = 64
    inputs = _inputs(batch=batch, length=length, q_heads=q_heads, kv_heads=kv_heads, qk_dim=64, value_dim=value_dim,
                     dtype=dtype, with_k_read=with_k_read, q_length=spec.get("q_length"),
                     prev_length=spec.get("prev_length", 0))
    metadata = _metadata(spec, batch, length + spec.get("prev_length", 0))
    names = [n for n in ("q", "k", "v", "g", "k_read", "prev_k", "prev_v") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    options = dict(metadata, high_precision_level=high_precision_level, chunk_size=span)
    options.update({n: tensors[n] for n in ("k_read", "prev_k", "prev_v") if n in tensors})
    base = dict(q=tensors["q"], k=tensors["k"], v=tensors["v"], g=tensors["g"])

    def forward(det):
        with torch.inference_mode():
            return sliding_chunk_softdelta_attention_fwd(**base, **{'k_read': None, **options}, deterministic=det)

    def backward(state, det):
        y, neg_read, lse = state
        return sliding_chunk_softdelta_attention_bwd(y_grad=y_grad, **base, y=y, neg_read=neg_read, lse=lse,
                                                     **{'k_read': None, **options}, deterministic=det)
    if high_precision_level:
        expect_refusal(lambda: forward(deterministic))
        return
    y, actual_neg_read, lse = forward(deterministic)
    q_length = tensors["q"].shape[1]
    assert y.shape == (batch, q_length, q_heads, value_dim) and y.dtype == dtype
    assert actual_neg_read.shape == (batch, q_length, kv_heads, value_dim)
    assert actual_neg_read.dtype == dtype
    assert lse.shape == (batch, kv_heads + q_heads, q_length) and lse.dtype == torch.float32
    expected, read, neg_read, expected_inputs = _oracle(span, False, tensors, metadata)
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    torch.testing.assert_close(actual_neg_read.double(), neg_read, atol=ATOL[dtype], rtol=RTOL[dtype])

    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}
    autograd_y = sliding_chunk_softdelta_attention(
        **{n: leaves[n] for n in ("q", "k", "v", "g")},
        **{'k_read': None, **dict(options, **{n: leaves[n] for n in ('k_read', 'prev_k', 'prev_v') if n in leaves})},
        deterministic=deterministic,
    )
    assert torch.equal(autograd_y, y)
    y_grad = torch.randn_like(y) / y.numel() ** 0.5
    dq, dk, dv, dg, dk_read, *previous = backward((y, actual_neg_read, lse), deterministic)
    assert (dk_read is None) == (not with_k_read)
    assert len(previous) == 2
    dprev_k, dprev_v = previous
    assert (dprev_k is None) == (dprev_v is None) == ("prev_k" not in tensors)
    gradients = {"q": dq, "k": dk, "v": dv, "g": dg, "k_read": dk_read, "prev_k": dprev_k, "prev_v": dprev_v}
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)
    autograd_grads = torch.autograd.grad(autograd_y, list(leaves.values()), y_grad)
    for name, a, b in zip(names, autograd_grads, (gradients[n] for n in names)):
        assert torch.equal(a, b), name
    standalone_determinism(forward, backward, deterministic, (y, actual_neg_read, lse),
                           (dq, dk, dv, dg, dk_read, *previous))
    with pytest.raises(NotImplementedError, match="high_precision_level=1 has no native Blackwell route"):
        sliding_chunk_softdelta_attention_bwd(y_grad=y_grad, **base, y=y, neg_read=actual_neg_read, lse=lse,
                                              **{'k_read': None, **dict(options, high_precision_level=1)})


@pytest.mark.parametrize("scale", (-4.0, 0.0), ids=("negative", "zero"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_nonpositive_scale_matches_fp64_oracle(scale, with_k_read, high_precision_level):
    """Nonpositive scales run over negated or zeroed queries (autograd and standalone fwd / bwd)."""
    require_native_b300()
    torch.manual_seed(20261030)
    dtype, span = torch.bfloat16, 32
    inputs = _inputs(batch=2, length=150, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=dtype, with_k_read=with_k_read)
    names = [n for n in ("q", "k", "v", "g", "k_read") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = softdelta_fp64_oracle(
        **leaves, scale=scale, visibility="chunk", span=span, return_aux=True)
    options = dict(scale=scale, high_precision_level=high_precision_level, chunk_size=span)
    if with_k_read:
        options["k_read"] = tensors["k_read"]
    actual_leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}

    def call(det):
        return sliding_chunk_softdelta_attention(
            **{n: actual_leaves[n] for n in ("q", "k", "v", "g")},
            **{'k_read': None, **dict(options, **{'k_read': actual_leaves['k_read']} if with_k_read else {})},
            deterministic=det)
    if high_precision_level:
        expect_refusal(lambda: call(False))
        return
    actual = call(False)
    _assert_output(actual, expected, read, neg_read, tensors["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    expected_grads = torch.autograd.grad(expected, list(leaves.values()), y_grad.double())
    actual_grads = torch.autograd.grad(actual, list(actual_leaves.values()), y_grad)
    _assert_gradients(names, actual_grads, expected_grads, dtype)
    autograd_determinism(call, actual_leaves, y_grad, False, actual, actual_grads)
    base = dict(q=tensors["q"], k=tensors["k"], v=tensors["v"], g=tensors["g"])
    y, actual_neg_read, lse = sliding_chunk_softdelta_attention_fwd(**base, **{'k_read': None, **options})
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    gradients = dict(zip(("q", "k", "v", "g", "k_read"), sliding_chunk_softdelta_attention_bwd(
        y_grad=y_grad, **base, y=y, neg_read=actual_neg_read, lse=lse, **{'k_read': None, **options})))
    assert (gradients["k_read"] is None) == (not with_k_read)
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)


def test_native_composition_is_used():
    """B300 counterpart of upstream's test_fused_path_is_used: no SM90 fused launch exists on B300, so an
    equal-length local call runs the composition: the native reads (route records), the gate, restore and
    merge kernels (counted here) and never the PyTorch reference (the autouse proof)."""
    require_native_b300()
    module = importlib.import_module("xattn.ops.softdelta_attention")
    inputs = _inputs(batch=1, length=64, q_heads=4, kv_heads=2, qk_dim=64, value_dim=64, dtype=torch.bfloat16)
    local = module._Local(module.AttentionVisibility.SLIDING_CHUNK, 8)
    assert module._fused_ops(inputs["q"], inputs["v"].unsqueeze(3), None, local, inputs["k"].shape[1]) is None
    ops = module._extension_ops()
    seen = []

    class Counting:
        def __getattr__(self, name):
            target = getattr(ops, name)
            if name.startswith("_softdelta_"):
                def counted(*args, **kwargs):
                    seen.append(name)
                    return target(*args, **kwargs)
                return counted
            return target
    original = module._extension_ops
    module._extension_ops = lambda: Counting()
    try:
        leaves = {n: inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g")}
        y = sliding_chunk_softdelta_attention(**leaves, k_read=None, chunk_size=8)
        torch.autograd.grad(y, list(leaves.values()), torch.ones_like(y))
    finally:
        module._extension_ops = original
    assert seen == ["_softdelta_gate_fwd", "_softdelta_gate_bwd", "_softdelta_gate_restore",
                    "_softdelta_merge_gradients"], seen


FUSED_CASES = {
    "uneven": dict(length=77),
    "block-boundary": dict(length=257),
    "segments": dict(length=300, segments=(1, 2, 130, 3, 64, 7)),
    # Every token its own segment: every negative-query tile is empty (zero-tile epilogue).
    "singletons": dict(length=300, segments=(1,)),
}
# Spans below every fused case's length (local kernels, not the causal route), across tile edges.
FUSED_SPANS = (64, 100, 128)


@pytest.mark.parametrize("span", FUSED_SPANS, ids=[f"chunk{span}" for span in FUSED_SPANS])
@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 2), (16, 4), (8, 1), (16, 1)),
                         ids=("h4kv4", "h8kv2", "h16kv4", "h8kv1", "h16kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_bucket_matches_fp64_oracle(span, case, heads, with_k_read, dtype, high_precision_level):
    """D = V = 128 (the native tcgen05 bucket) across head mappings and span / tile edges."""
    require_native_b300()
    torch.manual_seed(20261032)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=with_k_read)
    _check(span, inputs, dtype, high_precision_level=high_precision_level, **_metadata(spec, batch, length))


@pytest.mark.parametrize("dims", DIM_CASES)
@pytest.mark.parametrize("case", ("block-boundary", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_dims_match_fp64_oracle(dims, case, with_k_read, dtype, high_precision_level):
    """Every D, V <= 256 bucket (chunk 48)."""
    require_native_b300()
    torch.manual_seed(20261033)
    qk_dim, value_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    _check(48, inputs, dtype, high_precision_level=high_precision_level, **_metadata(spec, batch, length))


@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 8), (8, 4), (8, 2), (16, 1)),
                         ids=("h4kv4", "h8kv8", "h8kv4", "h8kv2", "h16kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_reader_groups_match_fp64_oracle(monkeypatch, case, heads, with_k_read, dtype,
                                               high_precision_level):
    """Upstream forces the SM90 reader-group backward here; the knob has no effect on B300's composition,
    whose cases still run (chunk 48)."""
    require_native_b300()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261034)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=with_k_read)
    _check(48, inputs, dtype, high_precision_level=high_precision_level, **_metadata(spec, batch, length))


@pytest.mark.parametrize("dims", DIM_CASES)
@pytest.mark.parametrize("case", ("block-boundary", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_fused_reader_groups_dims_match_fp64_oracle(monkeypatch, dims, case, with_k_read, dtype):
    """Reader-group cases across the (D, V) buckets (MHA heads, chunk 48)."""
    require_native_b300()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261035)
    qk_dim, value_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=4, kv_heads=4, qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    _check(48, inputs, dtype, **_metadata(spec, batch, length))


def test_deterministic_backward_is_repeatable():
    require_native_b300()
    torch.manual_seed(20261031)
    inputs = _inputs(batch=2, length=257, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16)
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g")]
    grad = torch.randn(2, 257, 8, 64, dtype=torch.bfloat16, device="cuda")
    runs = []
    for _ in range(2):
        out = sliding_chunk_softdelta_attention(q=leaves[0], k=leaves[1], v=leaves[2], g=leaves[3],
                                                deterministic=True, k_read=None, chunk_size=100)
        runs.append(torch.autograd.grad(out, leaves, grad))
    for a, b in zip(*runs):
        assert torch.equal(a, b)


def test_rejects_invalid_arguments():
    """Upstream's validation on CPU tensors (nothing runs; recorded as refusals for the proof)."""
    inputs = _inputs(batch=1, length=8, q_heads=4, kv_heads=2, qk_dim=32, value_dim=32,
                     dtype=torch.bfloat16, device="cpu", prev_length=3)
    q, k, v, g = inputs["q"], inputs["k"], inputs["v"], inputs["g"]
    window, chunk = sliding_window_softdelta_attention, sliding_chunk_softdelta_attention
    expect_refusal(lambda: window(q=q, k=k, k_read=None, v=v, g=g, window_size=-1), "nonnegative", ValueError)
    expect_refusal(lambda: window(q=q, k=k, k_read=None, v=v, g=g, window_size=True), "integer", TypeError)
    expect_refusal(lambda: window(q=q, k=k, k_read=None, v=v, g=g, window_size=2.0), "integer", TypeError)
    expect_refusal(lambda: chunk(q=q, k=k, k_read=None, v=v, g=g, chunk_size=0), "positive", ValueError)
    expect_refusal(lambda: chunk(q=q, k=k, k_read=None, v=v, g=g, chunk_size=4, reset_chunk_pos_per_seq=1),
                   "reset_chunk_pos_per_seq must be a bool", TypeError)
    expect_refusal(lambda: chunk(q=q, k=k, k_read=None, v=v, g=g, chunk_size=4, prev_k=inputs['prev_k'],
                                 prev_v=inputs['prev_v']), "must equal chunk_size", ValueError)
    expect_refusal(lambda: window(q=q, k=k, v=v, g=g, window_size=4, k_read=q), "k_read must have shape", ValueError)
    expect_refusal(lambda: sliding_window_softdelta_attention_fwd(q=q, k=k, k_read=None, v=v, g=g, window_size=4,
                                                                  backend='torch'), "SM90", ValueError)
    expect_refusal(lambda: chunk(q=q, k=k, k_read=None, v=v, g=g, chunk_size=4, backend='torch',
                                 high_precision_level=1), "SM90", ValueError)
    assert len(REFUSALS) == 9
