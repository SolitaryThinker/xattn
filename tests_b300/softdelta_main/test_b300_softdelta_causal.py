"""Causal SoftDelta (softdelta_attention, _fwd, _bwd) against the FP64 oracle on B300.

The cases of upstream xattn main b112365's tests/softdelta/test_softdelta_attention.py and
test_value_groups.py, ported so that nothing skips on B300 (the B300 gate, backend 'auto', upstream's oracle
and tolerances). On B300 a causal call runs the native composition with both reads as one sliding chunk
holding every key. Every level-0 parametrization runs natively with the determinism trio; every level-1
parametrization asserts the refusal, nothing having run. SM90-only knobs that upstream monkeypatches
(the fused reader-group layout, the BOS varlen router) have no effect on B300; their cases still run.
test_reader_backward_fp32_kv_grads keeps its full (as one covering chunk) and chunk visibilities; its
window visibility has no native reader on B300.
"""
import importlib
from itertools import product

import pytest
import torch
import torch.nn.functional as F

from xattn import softdelta_attention, softdelta_attention_bwd, softdelta_attention_fwd

from softdelta_main_cases import (
    ALL_MODES,
    DIM_CASES,
    FUSED_DIMS,
    GRAD_REL_NORM,
    LEVEL1_REFUSAL,
    MODE_IDS,
    NEG_IDS,
    REFUSALS,
    WINDOW_REFUSAL,
    _assert_gradients,
    _assert_output,
    _bos_mask,
    _inputs,
    _negative_query_transforms,
    _oracle,
    _segments,
    autograd_determinism,
    expect_refusal,
    require_native_b300,
    standalone_determinism,
)
from softdelta_main_reference import ATOL, GRAD_ATOL, RTOL, softdelta_fp64_oracle


def _check(inputs, dtype, *, deterministic=False, high_precision_level=0, **metadata):
    names = [n for n in ("q", "k", "v", "g", "k_read", "prev_k", "prev_v") if inputs[n] is not None]
    actual_inputs = {n: inputs[n].detach().clone().requires_grad_(True) for n in names}

    def call(det, level=high_precision_level):
        return softdelta_attention(backend="auto", deterministic=det, high_precision_level=level,
                                   **{'k_read': None, **actual_inputs, **metadata})
    if high_precision_level:
        expect_refusal(lambda: call(deterministic))
        return None
    actual = call(deterministic)
    expected, read, neg_read, expected_inputs = _oracle({n: inputs[n] for n in names}, metadata)
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
    "length-two": dict(length=2),
    "uneven": dict(length=17),
    "block-boundary": dict(length=129),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("heads", HEADS, ids=[f"h{h}kv{k}" for h, k in HEADS])
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_matches_fp64_oracle(case, heads, with_k_read, dtype, deterministic, high_precision_level):
    require_native_b300()
    torch.manual_seed(20261020)
    spec = CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=64,
                     value_dim=64, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    if "bos" in spec:
        metadata["bos_mask"] = _segments(batch, length, spec["bos"]).diff(dim=1, prepend=torch.full(
            (batch, 1), -1, dtype=torch.int64, device="cuda")) != 0
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level, **metadata)


@pytest.mark.torch_reference
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_torch_backend_matches_oracle(dtype):
    """backend='torch' is still the reference on B300, when asked for (and nothing native runs)."""
    require_native_b300()
    torch.manual_seed(3)
    inputs = _inputs(batch=2, length=33, q_heads=4, kv_heads=2, qk_dim=32, value_dim=32,
                     dtype=dtype, device="cuda", with_k_read=True)
    names = [n for n in ("q", "k", "v", "g", "k_read") if inputs[n] is not None]
    leaves = {n: inputs[n].detach().clone().requires_grad_(True) for n in names}
    metadata = dict(segment_idx=_segments(2, 33, (1, 9, 4, 30)))
    actual = softdelta_attention(backend="torch", **leaves, **metadata)
    expected, read, neg_read, expected_inputs = _oracle({n: inputs[n] for n in names}, metadata)
    _assert_output(actual, expected, read, neg_read, inputs["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    _assert_gradients(names, torch.autograd.grad(actual, list(leaves.values()), y_grad),
                      torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double()), dtype)


def test_gqa16x4_d128_packed_bf16():
    """GQA 16/4, D = V = 128, packed documents."""
    require_native_b300()
    torch.manual_seed(11)
    inputs = _inputs(batch=1, length=320, q_heads=16, kv_heads=4, qk_dim=128, value_dim=128,
                     dtype=torch.bfloat16)
    _check(inputs, torch.bfloat16, segment_idx=_segments(1, 320, (100, 1, 150, 69)))


@pytest.mark.parametrize("gate_value", (-30.0, 30.0))
def test_gate_extremes(gate_value):
    require_native_b300()
    torch.manual_seed(5)
    inputs = _inputs(batch=1, length=65, q_heads=4, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16)
    inputs["g"] = torch.full_like(inputs["g"], gate_value)
    _check(inputs, torch.bfloat16)


def test_deterministic_backward_is_repeatable():
    require_native_b300()
    torch.manual_seed(9)
    inputs = _inputs(batch=2, length=257, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16)
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g")]
    grad = torch.randn(2, 257, 8, 64, dtype=torch.bfloat16, device="cuda")
    runs = []
    for _ in range(2):
        out = softdelta_attention(q=leaves[0], k=leaves[1], k_read=None, v=leaves[2], g=leaves[3], deterministic=True)
        runs.append(torch.autograd.grad(out, leaves, grad))
    for a, b in zip(*runs):
        assert torch.equal(a, b)


def test_rejects_invalid_arguments():
    """Upstream's validation on CPU tensors (nothing runs; recorded as refusals for the proof)."""
    inputs = _inputs(batch=1, length=8, q_heads=4, kv_heads=2, qk_dim=32, value_dim=32,
                     dtype=torch.bfloat16, device="cpu")
    q, k, v, g = inputs["q"], inputs["k"], inputs["v"], inputs["g"]
    expect_refusal(lambda: softdelta_attention(q=q, k=k, v=v, g=g, k_read=q), "k_read must have shape", ValueError)
    expect_refusal(lambda: softdelta_attention(q=q[:, :, :3], k=k, k_read=None, v=v, g=g[:, :, :3]), "divisible",
                   ValueError)
    expect_refusal(lambda: softdelta_attention(q=q, k=k, k_read=None, v=v, g=g[:, :, :2]), "g must have shape",
                   ValueError)
    # Scalar gates only: the value-group layouts are not accepted.
    expect_refusal(lambda: softdelta_attention(q=q, k=k, k_read=None, v=v.unsqueeze(3), g=g), "4D q/k/v", ValueError)
    expect_refusal(lambda: softdelta_attention(q=q, k=k, k_read=None, v=v, g=g[..., None, None]),
                   "g must have shape", ValueError)
    expect_refusal(lambda: softdelta_attention(q=q, k=k, k_read=None, v=v, g=g, scale=float('nan')), "finite",
                   ValueError)
    expect_refusal(lambda: softdelta_attention(q=q, k=k, k_read=None, v=v, g=g, backend='torch',
                                               high_precision_level=1), "SM90", ValueError)
    expect_refusal(lambda: softdelta_attention(q=q, k=k, k_read=None, v=v, g=g, backend='torch',
                                               high_precision_level=2), "0 or 1", ValueError)
    expect_refusal(lambda: softdelta_attention_fwd(q=q, k=k, k_read=None, v=v, g=g, backend='torch'), "SM90",
                   ValueError)
    assert len(REFUSALS) == 9


FUSED_CASES = {
    "length-one": dict(length=1),
    "uneven": dict(length=77),
    "block-boundary": dict(length=257),
    "segments": dict(length=300, segments=(1, 2, 130, 3, 64, 7)),
    # Every token its own segment: every negative-query tile is empty (zero-tile epilogue).
    "singletons": dict(length=300, segments=(1,)),
}


@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 2), (16, 4), (8, 1), (16, 1)),
                         ids=("h4kv4", "h8kv2", "h16kv4", "h8kv1", "h16kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 1)), ids=("level0", "det-level1"))
def test_fused_bucket_matches_fp64_oracle(case, heads, with_k_read, dtype, deterministic, high_precision_level):
    """D = V = 128 (the native tcgen05 bucket) across head mappings."""
    require_native_b300()
    torch.manual_seed(20261021)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level, **metadata)


@pytest.mark.parametrize("dims", DIM_CASES)
@pytest.mark.parametrize("case", ("uneven", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_dims_match_fp64_oracle(dims, case, with_k_read, dtype, high_precision_level):
    """Every D, V <= 256."""
    require_native_b300()
    torch.manual_seed(20261023)
    qk_dim, value_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, high_precision_level=high_precision_level, **metadata)


@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 8), (8, 4), (8, 2), (16, 1)),
                         ids=("h4kv4", "h8kv8", "h8kv4", "h8kv2", "h16kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_reader_groups_match_fp64_oracle(monkeypatch, case, heads, with_k_read, dtype, high_precision_level):
    """Upstream forces the SM90 reader-group backward here; no effect on B300, whose cases still run."""
    require_native_b300()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261026)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, high_precision_level=high_precision_level, **metadata)


@pytest.mark.parametrize("dims", DIM_CASES)
@pytest.mark.parametrize("case", ("uneven", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_fused_reader_groups_dims_match_fp64_oracle(monkeypatch, dims, case, with_k_read, dtype):
    """Reader-group cases across the (D, V) buckets (MHA heads)."""
    require_native_b300()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261027)
    qk_dim, value_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=4, kv_heads=4, qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, **metadata)


@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("segments", (False, True), ids=("dense", "segments"))
def test_fused_mha_automatic_layout_matches_fp64_oracle(with_k_read, segments):
    """MHA with many work tiles (upstream's automatic SM90 layout threshold): causal reads of 2048 keys."""
    require_native_b300()
    torch.manual_seed(20261028)
    batch, length, heads = (8, 2048, 8) if segments else (2, 2048, 8)
    inputs = _inputs(batch=batch, length=length, q_heads=heads, kv_heads=heads, qk_dim=128,
                     value_dim=128, dtype=torch.bfloat16, with_k_read=with_k_read)
    metadata = {"segment_idx": _segments(batch, length, (700, 1, 1300, 64))} if segments else {}
    _check(inputs, torch.bfloat16, **metadata)


# (D, V) for the packed (BOS) route.
BOS_DIMS = {"d128v128": (128, 128), "d64v96": (64, 96), "d40v56": (40, 56),
            "d256v256": (256, 256), "d192v192": (192, 192), "d128v64": (128, 64)}


@pytest.mark.parametrize("dims", list(BOS_DIMS), ids=list(BOS_DIMS))
@pytest.mark.parametrize("heads", ((8, 2), (16, 4), (8, 8), (8, 1)), ids=("h8kv2", "h16kv4", "h8kv8", "h8kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 1)), ids=("level0", "det-level1"))
def test_fused_bos_varlen_forward_matches_fp64_oracle(monkeypatch, dims, heads, with_k_read, dtype, deterministic,
                                                      high_precision_level):
    """BOS masks (upstream forces the SM90 packed varlen forward here; B300 reads segment ids)."""
    require_native_b300()
    module = importlib.import_module("xattn.ops.softdelta_attention")
    monkeypatch.setattr(module, "select_flash_swa_bos_routes", lambda *args, **kwargs: (True, True))
    monkeypatch.setattr(module, "_VARLEN_MIN_TOKENS", 0)
    torch.manual_seed(20261024)
    qk_dim, value_dim = BOS_DIMS[dims]
    batch, length = 2, 300
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level,
           bos_mask=_bos_mask(batch, length, (1, 2, 130, 3, 64, 7)))


BOS_CASES = {"runs": (1, 2, 130, 3, 64, 7), "one-run": (300,), "singletons": (1,), "long-runs": (129, 171)}


@pytest.mark.parametrize("case", list(BOS_CASES), ids=list(BOS_CASES))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_fused_bos_routes_match_fp64_oracle(monkeypatch, case, dtype):
    """BOS masks of every shape (one run per row, singletons, long runs)."""
    require_native_b300()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_VARLEN_MIN_TOKENS", 0)
    torch.manual_seed(20261025)
    batch, length = 2, 300
    inputs = _inputs(batch=batch, length=length, q_heads=16, kv_heads=4, qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=False)
    _check(inputs, dtype, bos_mask=_bos_mask(batch, length, BOS_CASES[case]))


def test_output_flattens_without_copy_semantics_issues():
    """The output may be a strided view; downstream reshapes and in-graph use must behave."""
    require_native_b300()
    torch.manual_seed(13)
    inputs = _inputs(batch=2, length=130, q_heads=8, kv_heads=2, qk_dim=128, value_dim=128,
                     dtype=torch.bfloat16)
    leaves = {n: inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g")}
    out = softdelta_attention(**{'k_read': None, **leaves})
    flat = out.reshape(2, 130, -1)
    weight = torch.randn(flat.shape[-1], 64, device="cuda", dtype=torch.bfloat16)
    loss = (flat @ weight).float().square().mean()
    grads = torch.autograd.grad(loss, list(leaves.values()))
    reference = softdelta_attention(**{'k_read': None, **leaves}).contiguous().reshape(2, 130, -1)
    torch.testing.assert_close(flat, reference)
    assert all(bool(torch.isfinite(g).all()) for g in grads)


@pytest.mark.parametrize("q_length", (1, 37, 129), ids=("q1", "q37", "q129"))
@pytest.mark.parametrize("segments", (False, True), ids=("dense", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_short_query_matches_fp64_oracle(q_length, segments, with_k_read, dtype, deterministic,
                                         high_precision_level):
    """Lq < Lk without previous K/V (xllm's tail call: queries right-aligned to the keys)."""
    require_native_b300()
    torch.manual_seed(20261103)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=128, value_dim=128,
                     dtype=dtype, with_k_read=with_k_read, q_length=q_length)
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))} if segments else {}
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level, **metadata)


STANDALONE_CASES = {
    "dense": dict(length=150),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
    "short-q": dict(length=150, q_length=61),
}


@pytest.mark.parametrize("case", list(STANDALONE_CASES), ids=list(STANDALONE_CASES))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_standalone_fwd_bwd_match_fp64_oracle(case, with_k_read, dtype, deterministic, high_precision_level):
    """softdelta_attention_fwd returns the autograd output, the negative reads and both reads' LSE;
    softdelta_attention_bwd turns them into the oracle's gradients, bitwise the autograd ones."""
    require_native_b300()
    torch.manual_seed(20261104)
    spec = STANDALONE_CASES[case]
    batch, length = 2, spec["length"]
    q_heads, kv_heads = 8, 2
    value_dim = 64
    inputs = _inputs(batch=batch, length=length, q_heads=q_heads, kv_heads=kv_heads, qk_dim=64, value_dim=value_dim,
                     dtype=dtype, with_k_read=with_k_read, q_length=spec.get("q_length"))
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    if "bos" in spec:
        metadata["bos_mask"] = _bos_mask(batch, length, spec["bos"])
    names = [n for n in ("q", "k", "v", "g", "k_read") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    options = dict(metadata, high_precision_level=high_precision_level)
    if "k_read" in tensors:
        options["k_read"] = tensors["k_read"]
    base = dict(q=tensors["q"], k=tensors["k"], v=tensors["v"], g=tensors["g"])

    def forward(det):
        with torch.inference_mode():
            return softdelta_attention_fwd(**base, **{'k_read': None, **options}, deterministic=det)

    def backward(state, det):
        y, neg_read, lse = state
        return softdelta_attention_bwd(y_grad=y_grad, **base, y=y, neg_read=neg_read, lse=lse,
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
    expected, read, neg_read, expected_inputs = _oracle(tensors, metadata)
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    torch.testing.assert_close(actual_neg_read.double(), neg_read, atol=ATOL[dtype], rtol=RTOL[dtype])

    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}
    autograd_y = softdelta_attention(
        **{n: leaves[n] for n in ("q", "k", "v", "g")},
        **{'k_read': None, **dict(options, **{n: leaves[n] for n in ('k_read',) if n in leaves})},
        deterministic=deterministic,
    )
    assert torch.equal(autograd_y, y)
    y_grad = torch.randn_like(y) / y.numel() ** 0.5
    dq, dk, dv, dg, dk_read = backward((y, actual_neg_read, lse), deterministic)
    assert (dk_read is None) == (not with_k_read)
    gradients = {"q": dq, "k": dk, "v": dv, "g": dg, "k_read": dk_read}
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)
    autograd_grads = torch.autograd.grad(autograd_y, list(leaves.values()), y_grad)
    for name, a, b in zip(names, autograd_grads, (gradients[n] for n in names)):
        assert torch.equal(a, b), name
    standalone_determinism(forward, backward, deterministic, (y, actual_neg_read, lse), (dq, dk, dv, dg, dk_read))


@pytest.mark.parametrize("scale", (-4.0, 0.0), ids=("negative", "zero"))
@pytest.mark.parametrize("q_length", (None, 61), ids=("dense", "short-q"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_nonpositive_scale_matches_fp64_oracle(scale, q_length, with_k_read, high_precision_level):
    """Nonpositive scales run as positive ones over negated or zeroed queries; the default negative query
    keeps the keys' sign (autograd and standalone fwd / bwd)."""
    require_native_b300()
    torch.manual_seed(20261106)
    dtype = torch.bfloat16
    inputs = _inputs(batch=2, length=150, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=dtype, with_k_read=with_k_read, q_length=q_length)
    _check(inputs, dtype, high_precision_level=high_precision_level, scale=scale)
    if high_precision_level:
        return
    names = [n for n in ("q", "k", "v", "g", "k_read") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    options = dict(scale=scale, high_precision_level=high_precision_level)
    if with_k_read:
        options["k_read"] = tensors["k_read"]
    base = dict(q=tensors["q"], k=tensors["k"], v=tensors["v"], g=tensors["g"])
    y, actual_neg_read, lse = softdelta_attention_fwd(**base, **{'k_read': None, **options})
    expected, read, neg_read, expected_inputs = _oracle(tensors, {"scale": scale})
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    y_grad = torch.randn_like(y) / y.numel() ** 0.5
    gradients = dict(zip(("q", "k", "v", "g", "k_read"), softdelta_attention_bwd(
        y_grad=y_grad, **base, y=y, neg_read=actual_neg_read, lse=lse, **{'k_read': None, **options})))
    assert (gradients["k_read"] is None) == (not with_k_read)
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)


def test_standalone_rejects_mismatched_state():
    """Upstream checks the level-1 state; on B300 (level 0) the same refusals of mismatched state."""
    require_native_b300()
    torch.manual_seed(20261105)
    inputs = _inputs(batch=1, length=64, q_heads=4, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16)
    base = dict(q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"])
    y, neg_read, lse = softdelta_attention_fwd(**base, k_read=None)
    y_grad = torch.randn_like(y)
    state = dict(y=y, neg_read=neg_read, lse=lse)
    for name, change, match in (("neg_read", neg_read.float(), "neg_read has incompatible"),
                                ("lse", lse[:, 1:], "lse has incompatible"),
                                ("neg_read", neg_read[:, :, 1:], "neg_read has incompatible"),
                                ("y", y.float(), "y has incompatible")):
        with pytest.raises(ValueError, match=match):
            softdelta_attention_bwd(y_grad=y_grad, **base, k_read=None, **dict(state, **{name: change}))
    with pytest.raises(ValueError, match="y_grad has incompatible"):
        softdelta_attention_bwd(y_grad=y_grad.float(), **base, k_read=None, **state)
    with pytest.raises(NotImplementedError, match=LEVEL1_REFUSAL):
        softdelta_attention_fwd(**base, k_read=None, high_precision_level=1)


@pytest.mark.parametrize("source", ("key", "explicit-key", "silu-key", "second-query"))
@pytest.mark.parametrize("heads", ((8, 2), (4, 4)), ids=("h8kv2", "h4kv4"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 0), (False, 1), (True, 1)),
                         ids=("level0", "det-level0", "level1", "det-level1"))
def test_k_read_sources_match_fp64_oracle(source, heads, dtype, deterministic, high_precision_level):
    """The negative branch reads with whatever post-RoPE query the caller builds; gradients flow back
    through the caller's transforms (FP64 oracle chained through the same transforms)."""
    require_native_b300()
    torch.manual_seed(20261106)
    batch, length, (q_heads, kv_heads), dim = 2, 150, heads, 64
    leaves = {
        "q_pre": F.normalize(torch.randn(batch, length, q_heads, dim, device="cuda"), dim=-1),
        "k_pre": F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda"), dim=-1),
        "v": F.silu(torch.randn(batch, length, kv_heads, 64, device="cuda")) + 0.1,
        "g": torch.randn(batch, length, q_heads, device="cuda") * 0.75,
    }
    if source == "second-query":
        leaves["q2_pre"] = F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda"), dim=-1)
    leaves = {n: t.to(dtype).requires_grad_(True) for n, t in leaves.items()}
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))}

    def call(det):
        q, k, k_read = _negative_query_transforms(source, leaves)
        return softdelta_attention(q=q, k=k, v=leaves['v'], g=leaves['g'], k_read=k_read, deterministic=det,
                                   high_precision_level=high_precision_level, **metadata)
    if high_precision_level:
        expect_refusal(lambda: call(deterministic))
        return
    actual = call(deterministic)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(leaves.values()), y_grad)

    q, k, k_read = _negative_query_transforms(source, leaves)
    tensors = {"q": q, "k": k, "v": leaves["v"], "g": leaves["g"]}
    if k_read is not None:
        tensors["k_read"] = k_read
    expected, read, neg_read, post = _oracle(tensors, metadata)
    _assert_output(actual, expected, read, neg_read, leaves["g"], dtype)
    post_grads = dict(zip(post, torch.autograd.grad(expected, list(post.values()), y_grad.double())))
    double_leaves = {n: t.detach().double().requires_grad_(True) for n, t in leaves.items()}
    q64, k64, neg64 = _negative_query_transforms(source, double_leaves)
    outputs, grads = [q64, k64], [post_grads["q"], post_grads["k"]]
    if neg64 is not None:
        outputs.append(neg64)
        grads.append(post_grads["k_read"])
    transform_leaves = [n for n in double_leaves if n not in ("v", "g")]
    chained = dict(zip(transform_leaves, torch.autograd.grad(
        outputs, [double_leaves[n] for n in transform_leaves], grads)))
    chained.update(v=post_grads["v"], g=post_grads["g"])
    _assert_gradients(list(leaves), actual_grads, [chained[n] for n in leaves], dtype)
    autograd_determinism(call, leaves, y_grad, deterministic, actual, actual_grads)


@pytest.mark.parametrize("layout", ("packed-projection", "head-slice", "offset", "last-stride"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_input_views_match_fp64_oracle(layout, dtype, high_precision_level):
    """q / k / k_read as views: slices of one packed projection, head slices, odd base offsets and strided
    last dimensions (the native host copies what TMA cannot address)."""
    require_native_b300()
    torch.manual_seed(20261107)
    batch, length, q_heads, kv_heads, dim = 2, 150, 8, 2, 64

    def make(heads, offset=0):
        if layout == "packed-projection":
            return None
        shape = {"head-slice": (batch, length, 2 * heads, dim), "offset": (batch, length, heads, dim + 1),
                 "last-stride": (batch, length, heads, 2 * dim)}[layout]
        base = F.normalize(torch.randn(shape, device="cuda", dtype=dtype), dim=-1).requires_grad_(True)
        view = {"head-slice": lambda t: t[:, :, 1::2], "offset": lambda t: t[..., 1:],
                "last-stride": lambda t: t[..., ::2]}[layout](base)
        return base, view

    if layout == "packed-projection":
        packed = F.normalize(torch.randn(batch, length, q_heads + 2 * kv_heads, dim, device="cuda", dtype=dtype),
                             dim=-1).requires_grad_(True)
        q, k, k_read = packed[:, :, :q_heads], packed[:, :, q_heads:q_heads + kv_heads], packed[:, :, -kv_heads:]
    else:
        (_, q), (_, k), (_, k_read) = make(q_heads), make(kv_heads), make(kv_heads)
    v = (F.silu(torch.randn(batch, length, kv_heads, 64, device="cuda", dtype=dtype)) + 0.1).requires_grad_(True)
    g = (torch.randn(batch, length, q_heads, device="cuda", dtype=dtype) * 0.75).requires_grad_(True)
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))}
    if high_precision_level:
        expect_refusal(lambda: softdelta_attention(q=q, k=k, v=v, g=g, k_read=k_read,
                                                   high_precision_level=high_precision_level, **metadata))
        return
    actual = softdelta_attention(q=q, k=k, v=v, g=g, k_read=k_read, high_precision_level=high_precision_level,
                                 **metadata)
    tensors = {"q": q, "k": k, "v": v, "g": g, "k_read": k_read}
    expected, read, neg_read, expected_inputs = _oracle(tensors, metadata)
    _assert_output(actual, expected, read, neg_read, g, dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(tensors.values()), y_grad)
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(list(tensors), actual_grads, expected_grads, dtype)

    def call(det):
        return softdelta_attention(q=q, k=k, v=v, g=g, k_read=k_read, deterministic=det, **metadata)
    autograd_determinism(call, tensors, y_grad, False, actual, actual_grads)


# Upstream's (visibility, previous K/V) pairs that B300 reads natively: the full read is one chunk holding
# every key (the causal API takes no previous K/V); the window read has no native reader.
READER_CASES = (("full", False), ("chunk", False), ("chunk", True))


@pytest.mark.parametrize("visibility,prev", READER_CASES,
                         ids=[f"{v}-{'prev' if p else 'dense'}" for v, p in READER_CASES])
@pytest.mark.parametrize("deterministic", (False, True), ids=("nondet", "det"))
def test_reader_backward_fp32_kv_grads(visibility, prev, deterministic):
    """The readers' FP32 dK / dV (the composed level-0 merge input) are the accumulators the 16-bit
    gradients are cast from, bit for bit (B300 is deterministic whatever the flag says)."""
    require_native_b300()
    from xattn.ops.sliding_chunk_attention import _flash_sca_bwd, _flash_sca_fwd

    dtype, span = torch.bfloat16, 64
    torch.manual_seed(0)

    def setup(q_heads, kv_heads):
        t = _inputs(batch=2, length=200, q_heads=q_heads, kv_heads=kv_heads, qk_dim=64, value_dim=64,
                    dtype=dtype, prev_length=span if prev else 0)
        q, k, v, prev_k, prev_v = t["q"], t["k"], t["v"], t["prev_k"], t["prev_v"]
        scale = q.shape[-1] ** -0.5
        chunk = k.shape[1] if visibility == "full" else span
        y, lse = _flash_sca_fwd(q, k, v, chunk, scale, prev_k, prev_v, backend="auto")
        return lambda **options: _flash_sca_bwd(
            torch.ones_like(y), q, k, v, y, lse, chunk, scale, prev_k, prev_v,
            deterministic=deterministic, backend="auto", **options)

    backward = setup(8, 2)
    plain = backward()
    fp32 = backward(fp32_kv_grads=True)
    assert torch.equal(fp32[0], plain[0])
    for index in (1, 2, 3, 4):
        if plain[index] is None:
            assert not prev and fp32[index] is None
            continue
        assert fp32[index].dtype == torch.float32 and fp32[index].shape == plain[index].shape
        assert torch.equal(fp32[index].to(dtype), plain[index])
    with pytest.raises(RuntimeError, match="grouped query heads"):
        setup(4, 4)(fp32_kv_grads=True)


# ---- upstream tests/softdelta/test_value_groups.py: the internal value-group layout ----------------------

# (D, groups, group_dim): fused reads with segmented gate-gradient sums, and the composition (group widths
# that are not a multiple of 8).
GROUP_DIMS = {"d128c2": (128, 2, 64), "d192c3": (192, 3, 64), "d64c4-composed": (64, 4, 12)}
SUPPORTS = {"causal": (None, "full", None), "window40": ("SLIDING_WINDOW", "window", 40),
            "chunk32": ("SLIDING_CHUNK", "chunk", 32)}
MODES = ((False, 0), (True, 1), (False, 1))


def _group_inputs(*, batch, length, q_heads, kv_heads, qk_dim, groups, group_dim, dtype, with_k_read):
    def normal(*shape):
        return torch.randn(*shape, dtype=dtype, device="cuda")

    with torch.no_grad():
        tensors = dict(
            q=F.normalize(normal(batch, length, q_heads, qk_dim), dim=-1),
            k=F.normalize(normal(batch, length, kv_heads, qk_dim), dim=-1),
            v=F.silu(normal(batch, length, kv_heads, groups, group_dim)) + 0.1,
            g=normal(batch, length, q_heads, groups, 1) * 0.75,
        )
        if with_k_read:
            tensors["k_read"] = F.normalize(F.silu(normal(batch, length, kv_heads, qk_dim)), dim=-1)
    return tensors


def _group_cases():
    for dim, support, heads, segments, given, dtype, mode in product(
        GROUP_DIMS, SUPPORTS, ((8, 2), (16, 1)), (False, True), (False, True),
        (torch.float16, torch.bfloat16), MODES,
    ):
        default_mode = MODES[list(GROUP_DIMS).index(dim)]
        default = heads == (8, 2) and segments and given and mode == default_mode
        label = (f"{dim}-{support}-h{heads[0]}kv{heads[1]}-"
                 f"{'segments' if segments else 'dense'}-{'given' if given else 'key'}-"
                 f"{str(dtype).removeprefix('torch.')}-det{int(mode[0])}-level{mode[1]}")
        yield pytest.param(dim, support, heads, segments, given, dtype, *mode,
                           id=label, marks=() if default else pytest.mark.full_matrix)


@pytest.mark.parametrize(
    "dims,support,heads,segments,with_k_read,dtype,deterministic,high_precision_level", list(_group_cases()),
)
def test_value_groups_match_fp64_oracle(dims, support, heads, segments, with_k_read, dtype, deterministic,
                                        high_precision_level):
    """The internal autograd function with several value groups: native at level 0 for causal and chunk
    support; level 1 and the window support refuse on B300."""
    require_native_b300()
    module = importlib.import_module("xattn.ops.softdelta_attention")
    torch.manual_seed(20261025)
    qk_dim, groups, group_dim = GROUP_DIMS[dims]
    kind, visibility, span = SUPPORTS[support]
    local = None if kind is None else module._Local(getattr(module.AttentionVisibility, kind), span)
    batch, length = 2, 150
    tensors = _group_inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=qk_dim,
                            groups=groups, group_dim=group_dim, dtype=dtype, with_k_read=with_k_read)
    segment_idx = _segments(batch, length, (1, 2, 40, 3, 64, 7)) if segments else None
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}

    def call(det):
        return module._SoftDeltaFunc.apply(
            leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves.get("k_read"), None, None, segment_idx,
            segment_idx, qk_dim ** -0.5, det, high_precision_level, None, local,
        )
    if high_precision_level or support == "window40":
        expect_refusal(lambda: call(deterministic), "no native Blackwell route")
        return
    actual = call(deterministic)
    expected_leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = softdelta_fp64_oracle(
        **expected_leaves, q_segment_idx=segment_idx, k_segment_idx=segment_idx, visibility=visibility,
        span=span, return_aux=True,
    )
    _assert_output(actual, expected, read, neg_read, tensors["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(leaves.values()), y_grad)
    expected_grads = torch.autograd.grad(expected, list(expected_leaves.values()), y_grad.double())
    _assert_gradients(list(leaves), actual_grads, expected_grads, dtype)
    autograd_determinism(call, leaves, y_grad, deterministic, actual, actual_grads)
