"""CPU-only contract tests for the public FlashSCA envelope.

Subject: ``xattn/ops/sliding_chunk_attention.py`` (SCA): the public
``flash_sca`` / ``flash_sca_fwd`` / ``flash_sca_bwd`` entry points, the
validators they run before dispatch, and the autograd wiring of
``_FlashSCAFunc``. Every claim below is what the code *does*; where that
differs from the documented contract, the test requires the contract and the
docstring says so.

Nothing here needs CUDA or the ``xattn_cuda`` extension: each test either calls
a pure-Python validator or monkeypatches ``sca._extension_ops`` with a stub.
fp16/bf16 CPU tensors are only ever shaped, never computed on in a kernel.
Every assertion is about shapes, dtypes, errors and call order, so the suite
runs anywhere torch is importable -- with one exception:
``test_validate_qkv_heads_device_mismatch_raises`` needs a second CUDA device
to build a device mismatch and skips without one, so a CPU-only runner sees
46 passed / 1 skipped rather than 47 passed.

Run from the repository root, no built extension required::

    PYTHONPATH=<repo> python -m pytest kernel/tests_b300/sca/test_public_api_envelope.py -q
    PYTHONPATH=<repo> python -O -m pytest kernel/tests_b300/sca/test_public_api_envelope.py -q

Under ``python -O`` pytest warns that test-module ``assert`` statements are not
executed, so that second run is **not** evidence for the cells written with
``assert``; the ``-O`` evidence is carried by
``test_prev_pairing_is_refused_under_python_O``, which starts a child
interpreter with ``-O`` and reads the child's exit code instead of asserting in
a stripped interpreter.

``test_prev_k_without_prev_v_raises_value_error`` pins the prev K/V pairing at
the private validator: ``_validate_prev_kv_heads`` must refuse a lone operand
with ``ValueError`` rather than lean on an ``assert``. This is defense in depth
rather than a public repair -- every public path already pre-checks the pair
with a plain ``if/raise`` before it reaches this helper -- but the helper is
reachable directly (and from anything added later), and an ``assert`` also
raises ``AssertionError`` and is stripped entirely by ``python -O``. The same
stripped-``assert`` pattern survives at SCA:645, SCA:1153 and
``_bos_metadata.py:110/156``; this cell closes only the prev K/V one. The test
asks only for the house-style error and its message, so any fix that raises it
passes unchanged. Like the
rest of the repo's tests it uses no ``pytest.mark.xfail`` (only ``pytest.skip``
/ ``pytest.importorskip``: ``tests/sca/test_flash_sliding_chunk_attention.py``,
``tests/_numerics.py``), so the expectation is plain and unconditional.

Route note for test 8: public ``flash_sca`` has no BOS/segment exclusivity
check of its own; it raises from one of three places -- the internal
``_flash_sca`` (``"BOS masks, segment indices, and cu-seqlens are mutually
exclusive"``), ``flash_sca_fwd``/``flash_sca_bwd``
(``"BOS masks and segment indices are mutually exclusive"``), or, on a native
Blackwell build, ``_sca_native.visibility`` before any kernel is reached. All
three say "mutually exclusive"; the test matches that word and pins the one
property the contract needs: no extension call happens before the raise.
"""

import math
import re

import pytest
import torch

import xattn.ops.sliding_chunk_attention as sca


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _qkv(dtype=torch.float16, batch=1, q_len=4, k_len=4, hq=4, hkv=2,
         d=64, dv=64):
    q = torch.zeros(batch, q_len, hq, d, dtype=dtype)
    k = torch.zeros(batch, k_len, hkv, d, dtype=dtype)
    v = torch.zeros(batch, k_len, hkv, dv, dtype=dtype)
    return q, k, v


class _StubOps:
    """Records what the envelope does with the extension's two entry points."""

    def __init__(self):
        self.fwd_calls = []
        self.fwd_kwargs = []
        self.bwd_calls = []
        self.y = None
        self.lse = None

    def flash_sca_fwd(self, *args, **kwargs):
        self.fwd_calls.append(args)
        self.fwd_kwargs.append(kwargs)
        q, v = args[0], args[2]
        b, n, hq = q.shape[0], q.shape[1], q.shape[2]
        self.y = torch.zeros(b, n, hq, v.shape[3], dtype=q.dtype)
        self.lse = torch.zeros(b, hq, n, dtype=torch.float32)
        return self.y, self.lse

    def flash_sca_bwd(self, *args):
        self.bwd_calls.append(args)
        q, k, v = args[1], args[2], args[3]
        prev_k, prev_v = args[8], args[9]
        return (
            torch.ones_like(q),
            torch.ones_like(k),
            torch.ones_like(v),
            None if prev_k is None else torch.ones_like(prev_k),
            None if prev_v is None else torch.ones_like(prev_v),
        )


class _FakeCtx:
    """The slice of ``torch.autograd.function.FunctionCtx`` these paths use."""

    def __init__(self):
        self.saved = None
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved = tensors
        self.saved_tensors = tensors


# --------------------------------------------------------------------------
# 1. _resolve_scale (SCA:492-493)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d", [32, 64, 128, 256])
def test_resolve_scale_default_is_rsqrt_d(d):
    q = torch.zeros(1, 1, 1, d, dtype=torch.float16)
    value = sca._resolve_scale(q, None)
    assert isinstance(value, float)
    assert value == pytest.approx(1.0 / math.sqrt(d))


def test_resolve_scale_passes_finite_nonpositive_scales_through():
    q = torch.zeros(1, 1, 1, 64, dtype=torch.bfloat16)
    assert sca._resolve_scale(q, 0.0) == 0.0
    assert isinstance(sca._resolve_scale(q, 0.0), float)
    assert sca._resolve_scale(q, -0.5) == -0.5
    assert sca._resolve_scale(q, 2.0) == 2.0
    # No sign flip and no clamp anywhere in the resolver: the non-positive
    # transform lives in _dispatch_nonpositive_scale, downstream of this.
    assert sca._resolve_scale(q, -3) == -3.0


# --------------------------------------------------------------------------
# 2. _validate_qkv_heads (SCA:522-556)
# --------------------------------------------------------------------------
def _case_non_4d():
    q, k, v = _qkv()
    return q.squeeze(0), k, v


def _case_dtype_mismatch():
    q, k, v = _qkv()
    k = k.to(torch.bfloat16)
    return q, k, v


def _case_fp32():
    return _qkv(dtype=torch.float32)


def _case_batch_mismatch():
    q, k, v = _qkv()
    k = k.expand(2, -1, -1, -1).clone()
    v = v.expand(2, -1, -1, -1).clone()
    return q, k, v


def _case_empty_q():
    q, k, v = _qkv()
    return q[:, :0], k, v


def _case_q_longer_than_k():
    q, k, v = _qkv()
    q = torch.zeros(1, 8, 4, 64, dtype=torch.float16)
    return q, k, v


def _case_kv_head_mismatch():
    q, k, v = _qkv()
    v = torch.zeros(1, 4, 3, 64, dtype=torch.float16)
    return q, k, v


def _case_zero_heads():
    q, k, v = _qkv()
    return q[:, :, :0], k[:, :, :0], v[:, :, :0]


def _case_gqa_indivisible():
    q, k, v = _qkv()
    q = torch.zeros(1, 4, 3, 64, dtype=torch.float16)
    return q, k, v


def _case_qk_head_dim():
    q, k, v = _qkv()
    k = k[:, :, :, :32].clone()
    return q, k, v


_QKV_REJECTIONS = [
    ("non_4d", _case_non_4d, "expects batched 4D q/k/v tensors"),
    ("dtype_mismatch", _case_dtype_mismatch, "q/k/v must have the same dtype"),
    ("fp32", _case_fp32,
     "supports only torch.float16 and torch.bfloat16 input; got torch.float32"),
    ("batch_mismatch", _case_batch_mismatch,
     "batch dimensions and k/v lengths must match"),
    ("empty_q", _case_empty_q, "requires 0 < q length <= k/v length"),
    ("q_longer_than_k", _case_q_longer_than_k,
     "requires 0 < q length <= k/v length"),
    ("kv_head_mismatch", _case_kv_head_mismatch, "k/v head counts must match"),
    ("zero_heads", _case_zero_heads,
     "q and kv head counts must be positive"),
    ("gqa_indivisible", _case_gqa_indivisible,
     "q head count must be divisible by kv head count; got Hq=3, Hkv=2"),
    ("qk_head_dim", _case_qk_head_dim, "q/k head dims must match"),
]


@pytest.mark.parametrize("name,build,match", _QKV_REJECTIONS,
                         ids=[c[0] for c in _QKV_REJECTIONS])
def test_validate_qkv_heads_rejections(name, build, match):
    q, k, v = build()
    with pytest.raises(ValueError, match=re.escape(match)):
        sca._validate_qkv_heads(q, k, v, packed=False, api_name="flash_sca")


def test_validate_qkv_heads_accepts_gqa_and_mqa():
    # Hq=4, Hkv=2 (GQA) and Hkv=1 (MQA) are the contract's shapes.
    for hkv in (4, 2, 1):
        q, k, v = _qkv(hkv=hkv)
        sca._validate_qkv_heads(q, k, v, packed=False, api_name="flash_sca")


def test_validate_qkv_heads_device_mismatch_raises():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second device to build a device mismatch")
    q, k, v = _qkv()
    with pytest.raises(ValueError, match=re.escape("q/k/v must be on the same device")):
        sca._validate_qkv_heads(q, k.to("cuda:1"), v,
                                packed=False, api_name="flash_sca")


# --------------------------------------------------------------------------
# 3. _validate_prev_kv_heads: the prev_k/prev_v pairing
# --------------------------------------------------------------------------
def test_prev_k_without_prev_v_raises_value_error():
    """The pairing is a real raise, not an ``assert``.

    ``_validate_prev_kv_heads`` must raise ``ValueError`` in the house style
    when exactly one of ``prev_k``/``prev_v`` is given, and it must do so with
    the same wording ``_validate_metadata_representation`` uses. The seed
    enforced the pairing with a bare ``assert prev_v is not None``, which
    raises ``AssertionError`` and is removed entirely by ``python -O``; the
    message requirement is what keeps the fix aligned with the public path,
    and the ``-O`` case is pinned by
    ``test_prev_pairing_is_refused_under_python_O`` below.
    """
    q, k, v = _qkv()
    prev_k = torch.zeros(1, 4, 2, 64, dtype=torch.float16)
    with pytest.raises(ValueError,
                       match="prev_k and prev_v must be provided together"):
        sca._validate_prev_kv_heads(q, k, v, prev_k, None, 4,
                                    api_name="flash_sca")


# --------------------------------------------------------------------------
# 4. _validate_prev_kv_heads: the checks that do exist (SCA:569-589)
# --------------------------------------------------------------------------
def _prev():
    return (torch.zeros(2, 8, 2, 64, dtype=torch.float16),
            torch.zeros(2, 8, 2, 64, dtype=torch.float16))


_PREV_REJECTIONS = [
    ("length", {"prev_k": torch.zeros(2, 4, 2, 64, dtype=torch.float16)},
     "prev_k/prev_v sequence length must equal chunk_size"),
    ("prev_k_heads", {"prev_k": torch.zeros(2, 8, 3, 64, dtype=torch.float16)},
     "prev_k head count/dim must match current k"),
    ("prev_k_dim", {"prev_k": torch.zeros(2, 8, 2, 32, dtype=torch.float16)},
     "prev_k head count/dim must match current k"),
    ("prev_v_heads", {"prev_v": torch.zeros(2, 8, 3, 64, dtype=torch.float16)},
     "prev_v head count/dim must match current v"),
    ("prev_v_dim", {"prev_v": torch.zeros(2, 8, 2, 32, dtype=torch.float16)},
     "prev_v head count/dim must match current v"),
    ("batch", {"prev_k": torch.zeros(1, 8, 2, 64, dtype=torch.float16)},
     "prev_k/prev_v batch size must match q"),
    ("dtype", {"prev_k": torch.zeros(2, 8, 2, 64, dtype=torch.bfloat16)},
     "previous and current K/V must share a dtype"),
]


@pytest.mark.parametrize("name,over,match", _PREV_REJECTIONS,
                         ids=[c[0] for c in _PREV_REJECTIONS])
def test_validate_prev_kv_heads_rejections(name, over, match):
    q, k, v = _qkv(batch=2, q_len=8, k_len=8)
    prev_k, prev_v = _prev()
    prev_k = over.get("prev_k", prev_k)
    prev_v = over.get("prev_v", prev_v)
    with pytest.raises(ValueError, match=re.escape(match)):
        sca._validate_prev_kv_heads(q, k, v, prev_k, prev_v, 8,
                                    api_name="flash_sca")


def test_validate_prev_kv_heads_accepts_matching_previous():
    q, k, v = _qkv(batch=2, q_len=8, k_len=8)
    prev_k, prev_v = _prev()
    sca._validate_prev_kv_heads(q, k, v, prev_k, prev_v, 8, api_name="flash_sca")
    # prev_v alone must be refused here too: the seed body returned
    # early when prev_k was None and left the pair to the callers, and under
    # `python -O` its bare assert enforced nothing. The public paths do
    # pre-check the pair (several `if/raise` sites above this helper); this
    # pins the helper itself, which is reachable directly.
    with pytest.raises(ValueError,
                       match="prev_k and prev_v must be provided together"):
        sca._validate_prev_kv_heads(q, k, v, None, prev_v, 8,
                                    api_name="flash_sca")


# --------------------------------------------------------------------------
# 5. _validate_metadata_representation (SCA:593-632 at this head: the pairing
#    check above adds three lines; it was 590-629 at the base)
# --------------------------------------------------------------------------
def _meta(**over):
    args = dict(
        q_segment_idx=None, k_segment_idx=None,
        cu_seqlens_q=None, cu_seqlens_k=None,
        max_seqlen_q=None, max_seqlen_k=None,
        position_offsets=None, prev_k=None, prev_v=None,
        reset_chunk_pos_per_seq=False,
    )
    args.update(over)
    return sca._validate_metadata_representation(
        args["q_segment_idx"], args["k_segment_idx"],
        args["cu_seqlens_q"], args["cu_seqlens_k"],
        args["max_seqlen_q"], args["max_seqlen_k"],
        args["position_offsets"],
        args["prev_k"], args["prev_v"],
        args["reset_chunk_pos_per_seq"],
    )


_SEG_Q = torch.zeros(1, 4, dtype=torch.int64)
_SEG_K = torch.zeros(1, 4, dtype=torch.int64)
_CU_Q = torch.tensor([0, 4], dtype=torch.int32)
_CU_K = torch.tensor([0, 4], dtype=torch.int32)
_POS = torch.zeros(1, dtype=torch.int64)

_META_REJECTIONS = [
    ("prev_k_without_prev_v",
     dict(prev_k=torch.zeros(1, 4, 2, 64, dtype=torch.float16)),
     "prev_k and prev_v must be provided together"),
    ("prev_v_without_prev_k",
     dict(prev_v=torch.zeros(1, 4, 2, 64, dtype=torch.float16)),
     "prev_k and prev_v must be provided together"),
    ("q_segment_without_k_segment",
     dict(q_segment_idx=_SEG_Q),
     "q_segment_idx and k_segment_idx must be provided together"),
    ("k_segment_without_q_segment",
     dict(k_segment_idx=_SEG_K),
     "q_segment_idx and k_segment_idx must be provided together"),
    ("segment_and_varlen",
     dict(q_segment_idx=_SEG_Q, k_segment_idx=_SEG_K,
          cu_seqlens_q=_CU_Q, cu_seqlens_k=_CU_K,
          max_seqlen_q=4, max_seqlen_k=4),
     "segment metadata and varlen metadata are mutually exclusive"),
    ("cu_seqlens_q_without_k",
     dict(cu_seqlens_q=_CU_Q, max_seqlen_q=4, max_seqlen_k=4),
     "cu_seqlens_q and cu_seqlens_k must be provided together"),
    ("reset_with_position_offsets",
     dict(cu_seqlens_q=_CU_Q, cu_seqlens_k=_CU_K,
          max_seqlen_q=4, max_seqlen_k=4,
          position_offsets=_POS, reset_chunk_pos_per_seq=True),
     "position_offsets are only used by global-chunk varlen input"),
    ("max_seqlen_without_cu_seqlens",
     dict(max_seqlen_q=4, max_seqlen_k=4),
     "max_seqlen_q/max_seqlen_k require cu_seqlens"),
    ("position_offsets_without_cu_seqlens",
     dict(position_offsets=_POS),
     "position_offsets require cu_seqlens"),
]


@pytest.mark.parametrize("name,over,match", _META_REJECTIONS,
                         ids=[c[0] for c in _META_REJECTIONS])
def test_validate_metadata_representation_rejections(name, over, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        _meta(**over)


def test_validate_metadata_representation_returns_has_varlen():
    assert _meta() is False
    assert _meta(q_segment_idx=_SEG_Q, k_segment_idx=_SEG_K) is False
    assert _meta(cu_seqlens_q=_CU_Q, cu_seqlens_k=_CU_K,
                 max_seqlen_q=4, max_seqlen_k=4) is True


# --------------------------------------------------------------------------
# 6. _validate_bos_mask (SCA:748-764 at this head)
# --------------------------------------------------------------------------
def test_validate_bos_mask_rejections():
    q, _, _ = _qkv()
    match = re.escape(
        "bos_mask must be a bool tensor on the Q/K/V device with shape [1, 8]"
    )
    # shape only: dtype and device are right, the shape is not
    for bos in (
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(2, 8, dtype=torch.bool),
    ):
        with pytest.raises(ValueError, match=match):
            sca._validate_bos_mask(bos, q, 8, "bos_mask")
    # not bool
    with pytest.raises(ValueError, match=match):
        sca._validate_bos_mask(torch.zeros(1, 8, dtype=torch.int8), q, 8,
                               "bos_mask")
    with pytest.raises(ValueError, match=match):
        sca._validate_bos_mask(torch.zeros(1, 8, dtype=torch.int64), q, 8,
                               "bos_mask")


def test_validate_bos_mask_accepts_the_documented_shape():
    q, _, _ = _qkv()
    sca._validate_bos_mask(torch.zeros(1, 8, dtype=torch.bool), q, 8,
                           "bos_mask")
    sca._validate_bos_mask(torch.ones(1, 8, dtype=torch.bool), q, 8,
                           "bos_mask")


# --------------------------------------------------------------------------
# 7. autograd wiring of _FlashSCAFunc (SCA:70-182)
# --------------------------------------------------------------------------
def test_forward_saves_y_and_the_options(monkeypatch):
    ops = _StubOps()
    monkeypatch.setattr(sca, "_extension_ops", lambda: ops)
    q, k, v = _qkv()
    ctx = _FakeCtx()
    y = sca._FlashSCAFunc.forward(ctx, q, k, v, 4, 0.125)

    call = ops.fwd_calls[0]
    assert call[0] is q and call[1] is k and call[2] is v
    assert call[3] == 4 and call[4] == 0.125
    assert len(call) == 11          # no output_state argument

    assert y is ops.y
    saved = ctx.saved
    assert len(saved) == 9
    assert saved[0] is q and saved[1] is k and saved[2] is v
    assert saved[3] is ops.y        # element 3 is y
    assert saved[4] is ops.lse
    assert saved[5:9] == (None, None, None, None)

    assert ctx.chunk_size == 4
    assert ctx.scale == 0.125
    assert ctx.deterministic is False
    assert ctx.backend == "auto"
    assert ctx.reset_chunk_pos_per_seq is False
    assert ctx.fwd_route == "dense" and ctx.bwd_route == "dense"


def test_backward_argument_order_and_arity(monkeypatch):
    ops = _StubOps()
    monkeypatch.setattr(sca, "_extension_ops", lambda: ops)
    q, k, v = _qkv()
    prev_k = torch.zeros(1, 4, 2, 64, dtype=torch.float16)
    prev_v = torch.zeros(1, 4, 2, 64, dtype=torch.float16)
    q_seg = torch.zeros(1, 4, dtype=torch.int64)
    k_seg = torch.zeros(1, 4 + 4, dtype=torch.int64)
    ctx = _FakeCtx()
    sca._FlashSCAFunc.forward(
        ctx, q, k, v, 4, 0.125, prev_k, prev_v, q_seg, k_seg,
        "auto", True, True,
    )
    dy = torch.zeros(1, 4, 4, 64, dtype=torch.float16)
    out = sca._FlashSCAFunc.backward(ctx, dy)

    args = ops.bwd_calls[0]
    # (dy, q, k, v, y, lse, chunk, scale, prev_k, prev_v,
    #  q_seg, k_seg, deterministic, backend, reset)
    assert len(args) == 15
    assert args[0] is dy
    assert args[1] is q and args[2] is k and args[3] is v
    assert args[4] is ops.y
    assert args[5] is ops.lse
    assert args[6] == 4 and args[7] == 0.125
    assert args[8] is prev_k and args[9] is prev_v
    assert args[10] is q_seg and args[11] is k_seg
    assert args[12] is True          # deterministic
    assert args[13] == "auto"        # backend
    assert args[14] is True          # reset_chunk_pos_per_seq

    assert len(out) == 12
    assert out[0].shape == q.shape and out[1].shape == k.shape
    assert out[2].shape == v.shape
    assert out[3] is None and out[4] is None            # chunk, scale
    assert out[5].shape == prev_k.shape                 # prev_k_grad
    assert out[6].shape == prev_v.shape                 # prev_v_grad
    assert all(g is None for g in out[7:])


def test_backward_returns_none_for_absent_inputs(monkeypatch):
    ops = _StubOps()
    monkeypatch.setattr(sca, "_extension_ops", lambda: ops)
    q, k, v = _qkv()
    ctx = _FakeCtx()
    sca._FlashSCAFunc.forward(ctx, q, k, v, 4, 0.125)
    out = sca._FlashSCAFunc.backward(
        ctx, torch.zeros(1, 4, 4, 64, dtype=torch.float16))
    assert len(out) == 12
    assert out[0] is not None and out[1] is not None and out[2] is not None
    assert all(g is None for g in out[3:])


def test_apply_runs_autograd_end_to_end(monkeypatch):
    ops = _StubOps()
    monkeypatch.setattr(sca, "_extension_ops", lambda: ops)
    q, k, v = (_qkv()[0].requires_grad_(), _qkv()[1].requires_grad_(),
               _qkv()[2].requires_grad_())
    out = sca._FlashSCAFunc.apply(q, k, v, 4, 0.125, None, None, None, None,
                                  "auto", False, False)
    assert out.shape == (1, 4, 4, 64)
    assert out.requires_grad
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and t.grad.shape == t.shape
        assert torch.equal(t.grad, torch.ones_like(t.grad))


# --------------------------------------------------------------------------
# 8. flash_sca metadata exclusivity, before any extension call
# --------------------------------------------------------------------------
def test_flash_sca_bos_and_segment_metadata_are_mutually_exclusive(monkeypatch):
    touched = []

    def _boom():
        touched.append(True)
        raise AssertionError("flash_sca touched xattn_cuda before raising")

    monkeypatch.setattr(sca, "_extension_ops", _boom)
    q, k, v = _qkv()
    bos = torch.zeros(1, 4, dtype=torch.bool)
    q_seg = torch.zeros(1, 4, dtype=torch.int64)
    k_seg = torch.zeros(1, 4, dtype=torch.int64)

    with pytest.raises(ValueError, match="mutually exclusive"):
        sca.flash_sca(q, k, v, 4, bos_mask=bos,
                      q_segment_idx=q_seg, k_segment_idx=k_seg)
    assert touched == []

    with pytest.raises(ValueError, match="mutually exclusive"):
        sca.flash_sca_fwd(q, k, v, 4, bos_mask=bos,
                          q_segment_idx=q_seg, k_segment_idx=k_seg)
    assert touched == []


def test_flash_sca_bos_with_segment_idx_is_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sca, "_extension_ops", lambda: _StubOps())
    q, k, v = _qkv()
    bos = torch.zeros(1, 4, dtype=torch.bool)
    segment_idx = torch.zeros(1, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="mutually exclusive"):
        sca.flash_sca(q, k, v, 4, bos_mask=bos, segment_idx=segment_idx)


# --------------------------------------------------------------------------
# 13. The prev-K/V pairing is enforced under `python -O` (regression)
# --------------------------------------------------------------------------
def test_prev_pairing_is_refused_under_python_O():
    """`python -O` strips `assert`, so the pairing must be a real raise.

    Runs the validator in an interpreter started with -O and reads the exit
    code: 0 means both a lone prev_k and a lone prev_v raised ValueError (what
    the contract requires); 2 means some other exception came out
    (the old code path went to prev_v.dim() with prev_v=None and raised
    AttributeError under -O); 3 means no error at all. A bare `assert` passes
    the normal-interpreter test above and fails this one, which is the point:
    the seed enforced the pairing with `assert prev_v is not None`.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    program = "\n".join([
        "import torch",
        "import xattn.ops.sliding_chunk_attention as sca",
        "q = k = v = torch.zeros(1, 4, 2, 64, dtype=torch.float16)",
        "pk = torch.zeros(1, 4, 2, 64, dtype=torch.float16)",
        "for a, b in ((pk, None), (None, pk)):",
        "    try:",
        "        sca._validate_prev_kv_heads(q, k, v, a, b, 4,"
        " api_name='flash_sca')",
        "    except ValueError:",
        "        continue",
        "    except BaseException as exc:",
        "        print('WRONG-TYPE', type(exc).__name__, exc)",
        "        raise SystemExit(2)",
        "    print('NO-ERROR', 'prev_k' if b is None else 'prev_v')",
        "    raise SystemExit(3)",
        "print('OK')",
    ])
    env = dict(os.environ)
    # The documented run sets PYTHONPATH; the fallback is the repo root, which
    # is two levels above tests_b300/sca/ in both layouts (<repo>/... and
    # <workspace>/kernel/...). If the child cannot import xattn it exits 1, and
    # the assertion below prints its stderr, so a harness failure is loud and
    # is not mistaken for the fix misbehaving.
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    done = subprocess.run([sys.executable, "-O", "-c", program],
                          capture_output=True, text=True, env=env,
                          timeout=120)
    assert done.returncode == 0, (
        f"-O run failed: exit={done.returncode} "
        f"stdout={done.stdout!r} stderr={done.stderr!r}"
    )
    assert "OK" in done.stdout


# --------------------------------------------------------------------------
# 14. Supported strides and the no_grad/inference output rule
# --------------------------------------------------------------------------
def test_head_slice_offset_and_strided_views_reach_the_ops_unchanged(
        monkeypatch):
    """Supported views are passed through, not copied or re-laid-out.

    The contract accepts head-slice, offset and last-dimension-stride views.
    At the autograd bridge the CPU-checkable part of that is: the tensors the
    caller passes are the objects the extension call and ``ctx.saved``
    receive, so no Python layer can drop a stride, reorder an axis or
    materialise a copy without this test noticing.
    """
    ops = _StubOps()
    monkeypatch.setattr(sca, "_extension_ops", lambda: ops)
    base_q = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
    base_k = torch.zeros(1, 8, 4, 128, dtype=torch.float16)
    base_v = torch.zeros(1, 8, 4, 64, dtype=torch.float16)
    q = base_q[:, :, 4:12]             # head-slice view, 8 of 16 heads
    k = base_k[..., ::2]               # last-dimension stride 2, size 64
    v = base_v[:, 2:6]                 # storage offset 2, contiguous rows
    assert not q.is_contiguous() and not k.is_contiguous()
    assert q.stride(2) == 64 and k.stride(-1) == 2
    assert v.is_contiguous() and v.storage_offset() == 2 * 4 * 64
    ctx = _FakeCtx()
    sca._FlashSCAFunc.forward(ctx, q, k, v, 4, 0.125)
    call = ops.fwd_calls[0]
    assert call[0] is q and call[1] is k and call[2] is v
    assert ctx.saved[0] is q and ctx.saved[1] is k and ctx.saved[2] is v


def test_no_grad_and_inference_mode_keep_y_in_the_input_dtype(monkeypatch):
    """The output rule must not change under no_grad or inference mode.

    Two layers, both pinned here. The **public** ``flash_sca_fwd`` returns
    ``(y, lse)`` with ``y`` in the input dtype at level 0 and at level 1 --
    under ``no_grad`` and under ``inference_mode`` alike -- and no level
    passes an FP32 output state to the extension: level 1 asks for the
    high-precision softmax by the ``high_precision`` option only. The
    **bridge** ``_FlashSCAFunc`` then saves ``y`` (not an FP32 state), with
    no ``output_state`` argument in the extension call. ``_FlashSCAFunc`` is
    the level-0 autograd bridge: at level 1 the public ``flash_sca``
    dispatches to ``_precision_backend`` first, so that routing is not
    exercised here.
    """
    ops = _StubOps()
    monkeypatch.setattr(sca, "_extension_ops", lambda: ops)
    q, k, v = _qkv()
    for guard in (torch.no_grad(), torch.inference_mode()):
        # public entry point: the output rule itself
        with guard:
            out0 = sca.flash_sca_fwd(q, k, v, 4, 0.125, high_precision_level=0)
            out1 = sca.flash_sca_fwd(q, k, v, 4, 0.125, high_precision_level=1)
        assert len(out0) == 2 and len(out1) == 2     # (y, lse) at every level
        (y0, lse0), (y1, lse1) = out0, out1
        assert y0.shape == (1, 4, 4, 64) and y0.dtype == q.dtype
        assert lse0.shape == (1, 4, 4) and lse0.dtype == torch.float32
        assert y1.shape == y0.shape and y1.dtype == q.dtype
        assert lse1.shape == (1, 4, 4) and lse1.dtype == torch.float32
        # ... and that is what reaches the extension: no FP32 argument at
        # either level; level 1 is the high_precision option alone.
        def _fp32_args(call):
            return [a for a in call
                    if torch.is_tensor(a) and a.dtype == torch.float32]
        assert _fp32_args(ops.fwd_calls[-2]) == []
        assert _fp32_args(ops.fwd_calls[-1]) == []
        assert len(ops.fwd_calls[-2]) == 11 and len(ops.fwd_calls[-1]) == 11
        assert ops.fwd_kwargs[-2] == {}
        assert ops.fwd_kwargs[-1] == {"high_precision": True}
        # autograd bridge: what is saved, and with which arity
        ctx = _FakeCtx()
        with guard:
            y = sca._FlashSCAFunc.forward(ctx, q, k, v, 4, 0.125)
        assert y is ops.y
        assert ctx.saved[3] is ops.y          # y is saved, not an FP32 state
        assert ctx.saved[4] is ops.lse
        assert len(ops.fwd_calls[-1]) == 11   # no output_state argument
