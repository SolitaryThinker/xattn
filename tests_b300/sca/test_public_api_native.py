# Author: Shicheng Wen

"""Public API and autograd contract of native FlashSCA on cc 10.3.

These pin what the FP64 oracle cannot observe: which implementation each
public entry point runs at each precision level, the output dtype contract in
``no_grad`` and inference mode, autograd versus standalone equivalence,
deterministic bit-exactness at every level, the exact nonpositive-scale
normalisation, layout acceptance and the error contract.

Dispatch mapping pinned here (public entry points, backend ``''``/``auto``/
``native``, compute capability 10.3, native FlashSCA linked):

* ``flash_sca_fwd`` level 0 -> native forward, ``(y, lse)``, y in the Q dtype.
* ``flash_sca_fwd`` level 1 -> native forward with the subtract-before-scale
                              softmax, ``(y, lse)``, y in the Q dtype (no
                              FP32 state).
* ``flash_sca_bwd`` level 0 -> native SCA backward.
* ``flash_sca_bwd`` level 1 -> the precision-repair backward
                              (``_precision_backend.backward``) on the native
                              advanced-precision leaf; it recomputes its row
                              statistics and does not read y or lse.
* ``flash_sca`` autograd    -> the same forward; the backward of level 1 is
                              the precision-repair one.

The numerical contract itself is the oracle suite (``tests/sca`` and the B300
contract check); no tolerance is relaxed here.

**Route parity**, stated on purpose to be kept rather than deduced: autograd
calls the public ``flash_sca_fwd``, so the autograd forward tensor must be
bitwise identical to the standalone one at the same level; a mismatch is a
wrapper bug, not numerics. B300 SCA is deterministic-only: the native level-0
backward always writes dQ in a fixed order and the level-1 precision backward
always takes its ordered branch, whatever ``deterministic`` says, so every
gradient is asserted bit-exactly with ``deterministic=False`` as well.

The level 1 backward is the native advanced-precision path
(``csrc/attention/precision/native/``), not the native SCA backward: this file
asserts that the SCA native backward launch counter does *not* move at level
1 and that the precision backend's route record resolved ``native``, and the
skip condition requires the run-time capability to advertise the native
precision leaf and the chunk visibility the advanced backward needs.
"""

import pytest
import torch
import torch.nn.functional as F

import xattn
from xattn.ops import _precision_backend, _sca_native


def _native_available():
    """cc 10.3 with native SCA *and* the native advanced-precision capability.

    A build that links native SCA but not native precision must not run this
    file as if the level 1 backward were native, so the level mask and the
    chunk visibility bit are required here, not only the SCA entry points.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        import xattn_cuda

        probe = torch.zeros(1, device="cuda")
        if not bool(xattn_cuda.ops._flash_sca_native_available(probe)):
            return False
        levels = int(xattn_cuda.ops._native_attention_level_mask(probe))
        visibility = int(xattn_cuda.ops._native_precision_visibility_mask(probe))
        return (levels & 0x1F) == 0x1F and (visibility & 0x4) == 0x4
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="native FlashSCA runs on a cc 10.3 device with a build that links it",
)

DTYPES = (torch.bfloat16, torch.float16)
LEVELS = (0, 1)


def _inputs(batch=2, length=131, heads=4, kv_heads=2, dim=64, value_dim=64,
            previous=0, dtype=torch.bfloat16):
    torch.manual_seed(0)
    total = length + previous
    q = F.normalize(torch.randn(batch, length, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, total, kv_heads, dim, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(batch, total, kv_heads, value_dim, device="cuda").to(dtype)
    dy = torch.randn(batch, length, heads, value_dim, device="cuda").to(dtype)
    dy = dy / dy.float().norm(dim=-1, keepdim=True).to(dtype)
    if previous:
        return q, k[:, previous:], v[:, previous:], k[:, :previous], v[:, :previous], dy
    return q, k, v, None, None, dy


def _forward(q, k, v, pk, pv, chunk, level, scale=None):
    return xattn.flash_sca_fwd(q, k, v, chunk, scale, pk, pv,
                               high_precision_level=level)


def _backward(dy, q, k, v, pk, pv, y, lse, chunk, level, deterministic,
              scale=None):
    return xattn.flash_sca_bwd(dy, q, k, v, y, lse, chunk, scale, pk, pv,
                               deterministic=deterministic,
                               high_precision_level=level)


def _autograd(q, k, v, pk, pv, dy, chunk, level, deterministic, scale=None):
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    extra = []
    previous = {}
    if pk is not None:
        pk = pk.clone().requires_grad_(True)
        pv = pv.clone().requires_grad_(True)
        extra = [pk, pv]
        previous = dict(prev_k=pk, prev_v=pv)
    y = xattn.flash_sca(*leaves, chunk, scale=scale,
                        deterministic=deterministic,
                        high_precision_level=level, **previous)
    grads = torch.autograd.grad(y, leaves + extra, dy)
    return y.detach(), grads


def _resolved_native():
    """The precision backend's last route record resolved the native leaf."""
    record = _precision_backend.route_trace()[-1]
    return record["resolved"] == "native" and record["refused"] is None


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_levels_share_one_native_call(dtype):
    """Both levels run one native forward and return ``(y, lse)``.

    Contractual, and asserted: one native launch per public forward, y in the
    input dtype with the output shape, an FP32 ``[B, Hq, N]`` lse, at level 0
    and at level 1 (the subtract-before-scale softmax keeps no FP32 state).

    Recorded, not asserted: whether level 1 returns bit-identical y and lse to
    level 0. The two softmax forms round differently, so equality is not a
    promise either way; the comparison is printed as evidence.
    """
    q, k, v, pk, pv, _ = _inputs(previous=64, dtype=dtype)
    outputs = {}
    for level in LEVELS:
        before = _sca_native.launch_counts()
        y, lse = _forward(q, k, v, pk, pv, 64, level)
        after = _sca_native.launch_counts()
        assert after[0] == before[0] + 1, "the public forward must launch the native kernel"
        assert tuple(y.shape) == (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
        assert y.dtype == dtype
        assert tuple(lse.shape) == (q.shape[0], q.shape[2], q.shape[1])
        assert lse.dtype == torch.float32
        outputs[level] = (y, lse)
    equal = (torch.equal(outputs[0][0], outputs[1][0])
             and torch.equal(outputs[0][1], outputs[1][1]))
    print(f"forward {dtype}: level 1 bit-identical to level 0 -> {equal} "
          f"(evidence, not a contract)")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("deterministic", (False, True))
def test_autograd_and_standalone_agree(dtype, level, deterministic):
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    y_s, lse_s = _forward(q, k, v, pk, pv, 64, level)
    before = _sca_native.launch_counts()
    grads_s = _backward(dy, q, k, v, pk, pv, y_s, lse_s, 64, level, deterministic)
    after = _sca_native.launch_counts()
    if level == 0:
        assert after[1] == before[1] + 1, (
            "the level 0 public backward must be the native SCA backward")
    else:
        assert after[1] == before[1], (
            "level 1 must run the native advanced-precision backward, not "
            "the native SCA backward")
        assert _resolved_native(), _precision_backend.route_trace()[-1]
    y_a, grads_a = _autograd(q, k, v, pk, pv, dy, 64, level, deterministic)
    assert torch.equal(y_s, y_a), "autograd and standalone forward differ"
    for index, (standalone, autograd) in enumerate(zip(grads_s, grads_a)):
        if standalone is None:
            assert autograd is None
            continue
        assert autograd is not None and autograd.dtype == standalone.dtype
        # Deterministic-only on B300: no order-dependent reduction exists at
        # either level, so the two routes agree bit for bit at both settings.
        assert torch.equal(standalone, autograd), (
            f"gradient {index}: the public autograd and standalone routes differ")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", LEVELS)
def test_deterministic_backward_repeats_bit_exactly(dtype, level):
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    y, lse = _forward(q, k, v, pk, pv, 64, level)
    first = _backward(dy, q, k, v, pk, pv, y, lse, 64, level, True)
    second = _backward(dy, q, k, v, pk, pv, y, lse, 64, level, True)
    for a, b in zip(first, second):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b), "deterministic backward is not bit-exact"


@pytest.mark.parametrize("level", LEVELS)
def test_saved_state_contract_in_no_grad_and_inference(level):
    """The forward returns exactly ``(y, lse)`` at every level, in any grad mode:
    y in the input dtype (no level pays for an FP32 state) and an FP32 lse."""
    q, k, v, pk, pv, _ = _inputs(previous=64)
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            out = _forward(q, k, v, pk, pv, 64, level)
        assert len(out) == 2, "flash_sca_fwd returns (y, lse)"
        y, lse = out
        assert y.dtype == q.dtype
        assert lse.dtype == torch.float32


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nonpositive_scale_is_the_normalised_positive_scale(level, dtype):
    """`scale <= 0` is masked softmax with Q negated / zeroed (exact).

    ``_dispatch_nonpositive_scale`` maps ``scale < 0`` to ``q -> -q, scale ->
    -scale`` and ``scale == 0`` to ``q -> 0, scale -> 1``. The map is exact, so
    the outputs must be bit-identical, ``dQ`` must flip sign or be exactly zero,
    and the standalone backward must agree with the autograd one at every
    level (level 1 takes the precision-repair backend before the
    normalisation dispatch, which is why this test exists).
    """
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    for scale in (0.0, -0.1):
        negative = scale < 0
        signed = -q if negative else q * 0
        normalised = -scale if negative else 1.0
        y_ref, grads_ref = _autograd(signed, k, v, pk, pv, dy, 64, level, True, normalised)
        y, grads = _autograd(q, k, v, pk, pv, dy, 64, level, True, scale)
        assert torch.equal(y, y_ref), (
            f"level {level} scale {scale}: forward is not the exact normalisation")
        if negative:
            assert torch.equal(grads[0], -grads_ref[0]), (
                f"level {level} scale {scale}: dQ is not the sign-flipped normalisation")
        else:
            assert torch.equal(grads[0], torch.zeros_like(grads[0])), (
                f"level {level} scale {scale}: dQ must be exactly zero")
        for got, want in zip(grads[1:], grads_ref[1:]):
            if got is None:
                assert want is None
                continue
            assert torch.equal(got, want), (
                f"level {level} scale {scale}: dK/dV is not the exact normalisation")
        # The standalone route must agree with the autograd one.
        y_s, lse_s = _forward(q, k, v, pk, pv, 64, level, scale)
        assert torch.equal(y_s, y)
        grads_s = _backward(dy, q, k, v, pk, pv, y_s, lse_s, 64, level, True, scale)
        for standalone, autograd in zip(grads_s, grads):
            if standalone is None:
                assert autograd is None
                continue
            assert torch.equal(standalone, autograd), (
                f"level {level} scale {scale}: standalone backward differs from autograd")


@pytest.mark.parametrize("level", LEVELS)
def test_supported_strides_match_contiguous(level):
    q, k, v, pk, pv, dy = _inputs(previous=64)
    wide_q = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 2 * q.shape[-1],
                         device="cuda", dtype=q.dtype)
    wide_q[..., ::2] = q
    wide_k = torch.zeros(k.shape[0], k.shape[1], k.shape[2], 2 * k.shape[-1],
                         device="cuda", dtype=k.dtype)
    wide_k[..., ::2] = k
    wide_v = torch.zeros(v.shape[0], v.shape[1], v.shape[2], 2 * v.shape[-1],
                         device="cuda", dtype=v.dtype)
    wide_v[..., ::2] = v
    strided = (wide_q[..., ::2], wide_k[..., ::2], wide_v[..., ::2])
    for name, tensor in zip(("q", "k", "v"), strided):
        assert tensor.stride(-1) == 2, name
    y_ref, lse_ref = _forward(q, k, v, pk, pv, 64, level)
    y, lse = _forward(*strided, pk, pv, 64, level)
    assert torch.equal(y, y_ref) and torch.equal(lse, lse_ref)
    # Head-slice views: Hq 4 -> 2 and Hkv 2 -> 1, so query head h reads KV head
    # h // 2. Compared against the contiguous copies of the same slice, because
    # shapes alone would let a wrong head mapping pass.
    head = (q[:, :, :2], k[:, :, :1], v[:, :, :1])
    y_head, lse_head = _forward(*head, None, None, 64, level)
    assert tuple(y_head.shape) == (q.shape[0], q.shape[1], 2, v.shape[-1])
    assert tuple(lse_head.shape) == (q.shape[0], 2, q.shape[1])
    y_head_ref, lse_head_ref = _forward(q[:, :, :2].contiguous(),
                                           k[:, :, :1].contiguous(),
                                           v[:, :, :1].contiguous(), None, None,
                                           64, level)
    assert torch.equal(y_head, y_head_ref), "a head-slice view changed y"
    assert torch.equal(lse_head, lse_head_ref), "a head-slice view changed lse"
    grads_head = _backward(dy[:, :, :2], *head, None, None,
                           y_head, lse_head, 64, level, True)
    grads_head_ref = _backward(dy[:, :, :2], q[:, :, :2].contiguous(),
                               k[:, :, :1].contiguous(), v[:, :, :1].contiguous(),
                               None, None, y_head, lse_head, 64, level, True)
    for got, want in zip(grads_head, grads_head_ref):
        assert (got is None) == (want is None)
        if got is not None:
            assert torch.equal(got, want), "a head-slice view changed a gradient"
    grads_ref = _backward(dy, q, k, v, pk, pv, y_ref, lse_ref, 64, level, True)
    grads_str = _backward(dy, *strided, pk, pv, y_ref, lse_ref, 64, level, True)
    for a, b in zip(grads_ref, grads_str):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b), "strided inputs changed the gradient"


def test_availability_flags_and_refusals():
    q, k, v, _, _, _ = _inputs()
    assert isinstance(xattn.flash_sca_sm90_available(q), bool)
    assert _sca_native.selected(q, "auto", "flash_sca") is True
    assert _sca_native.selected(q, "", "flash_sca") is True
    assert _sca_native.selected(q, "native", "flash_sca") is True
    with pytest.raises((RuntimeError, ValueError)):
        xattn.flash_sca(q, k, v, 64, backend="sm90")
    with pytest.raises((RuntimeError, ValueError)):
        xattn.flash_sca(q, k, v, 64, backend="not_a_backend")


@pytest.mark.parametrize("level", LEVELS)
def test_validation_errors_stay_errors(level):
    q, k, v, pk, pv, dy = _inputs(previous=64)
    kwargs = dict(high_precision_level=level)
    with pytest.raises(ValueError):
        xattn.flash_sca(q, k, v, 64, prev_k=pk, **kwargs)
    with pytest.raises(ValueError):
        xattn.flash_sca(q, k, v, 32, prev_k=pk, prev_v=pv, **kwargs)
    with pytest.raises(ValueError):
        xattn.flash_sca(q, k, v, 64, segment_idx=torch.zeros(q.shape[0], k.shape[1] + 64,
                                                             dtype=torch.long, device="cuda"),
                        bos_mask=torch.ones(q.shape[0], k.shape[1] + 64,
                                            dtype=torch.bool, device="cuda"), **kwargs)
    with pytest.raises((TypeError, ValueError)):
        xattn.flash_sca(q, k, v, 64, high_precision_level=True)
    with pytest.raises(ValueError):
        xattn.flash_sca(q, k, v, 64, high_precision_level=5)
    with pytest.raises(ValueError):
        xattn.flash_sca(q, k, v, 64, scale=float("inf"), **kwargs)
    # The 256 cap itself: q, k and v are all built at 264 so the refusal cannot
    # come from a q/k dimension mismatch.
    wide_q = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 264, device="cuda",
                         dtype=q.dtype)
    wide_k = torch.zeros(k.shape[0], k.shape[1], k.shape[2], 264, device="cuda",
                         dtype=k.dtype)
    wide_v = torch.zeros(v.shape[0], v.shape[1], v.shape[2], 264, device="cuda",
                         dtype=v.dtype)
    with pytest.raises(ValueError):
        xattn.flash_sca(wide_q, wide_k, wide_v, 64, **kwargs)


# --- named structure gaps ------------------------------------------------
# The tests above pin the API contract at one shape with previous K/V. The
# four functions below pin structure that an oracle comparison cannot see:
# the no-previous-K/V backward tuple, the public gradient dtype at the
# high-precision level, the refused saved-output dtype at the public entry point,
# and - printed, never asserted - what the ABI edge under it does with the
# same misuse. A second batch (previous K/V shapes and dtypes, GQA with
# unequal lengths, no-visible-key rows, metadata exclusivity, the D/Dv
# widths) sits beside them unchanged.


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", LEVELS)
def test_backward_without_previous_kv_returns_two_nones(dtype, level):
    """No previous K/V: the backward returns ``(dq, dk, dv, None, None)``.

    Asserted on the standalone route and on autograd at every level, with the
    gradients' shapes and dtypes: a previous-chunk gradient that is silently
    allocated (or returned as a zero tensor) would change a training step
    without changing any number an oracle comparison sees at level 0.
    """
    q, k, v, _, _, dy = _inputs(previous=0, dtype=dtype)
    y, lse = _forward(q, k, v, None, None, 64, level)
    grads = _backward(dy, q, k, v, None, None, y, lse, 64, level, True)
    assert len(grads) == 5, (
        "flash_sca_bwd must return (dq, dk, dv, dprev_k, dprev_v)")
    dq, dk, dv, dprev_k, dprev_v = grads
    assert dprev_k is None, "without previous K/V dprev_k must be None"
    assert dprev_v is None, "without previous K/V dprev_v must be None"
    for name, grad, ref in (("dq", dq, q), ("dk", dk, k), ("dv", dv, v)):
        assert grad is not None, f"{name} must not be None"
        assert tuple(grad.shape) == tuple(ref.shape), name
        assert grad.dtype == dtype, name
        assert bool(torch.isfinite(grad.float()).all()), name
    _, grads_a = _autograd(q, k, v, None, None, dy, 64, level, True)
    assert len(grads_a) == 3, "autograd returns one gradient per Q/K/V leaf"
    for name, grad, ref in zip(("dq", "dk", "dv"), grads_a, (q, k, v)):
        assert grad is not None, name
        assert tuple(grad.shape) == tuple(ref.shape), name
        assert grad.dtype == dtype, name


@pytest.mark.parametrize("dtype", DTYPES)
def test_high_precision_level_returns_public_gradients_in_input_dtype(dtype):
    """Level 1 repairs in FP32 internally and casts at the public boundary.

    Every gradient the public API hands back keeps the input dtype and shape;
    the centred operands and the repair buffers stay inside.
    """
    level = 1
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    y, lse = _forward(q, k, v, pk, pv, 64, level)
    assert y.dtype == dtype, "level 1 returns y in the input dtype"
    grads = _backward(dy, q, k, v, pk, pv, y, lse, 64, level, True)
    assert _resolved_native(), _precision_backend.route_trace()[-1]
    refs = (("dq", q), ("dk", k), ("dv", v), ("dprev_k", pk), ("dprev_v", pv))
    assert len(grads) == 5
    for (name, ref), grad in zip(refs, grads):
        assert grad is not None, name
        assert tuple(grad.shape) == tuple(ref.shape), name
        assert grad.dtype == dtype, (
            f"level {level} {name}: public gradient dtype {grad.dtype} is not "
            f"the input dtype {dtype}")
        assert bool(torch.isfinite(grad.float()).all()), name
    _, grads_a = _autograd(q, k, v, pk, pv, dy, 64, level, True)
    for (name, ref), grad in zip(refs, grads_a):
        assert grad is not None, name
        assert tuple(grad.shape) == tuple(ref.shape), name
        assert grad.dtype == dtype, name


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("deterministic", (False, True))
def test_backward_refuses_mismatched_saved_state(dtype, level, deterministic):
    """A backward handed a saved output in the wrong dtype is a Python-level refusal.

    Every level saves ``y`` in the Q/K/V dtype; an FP32 ``y`` (the retired
    FP32 state) is not a saved output of any level. The public wrapper
    validates it before any dispatch, so the call must raise ``ValueError``
    (not a native or host error), say which dtype it needs, and launch
    nothing. Pinned as today's refusal - the message is source-owned, so the
    substrings are asserted, not the whole string.
    """
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    y, lse = _forward(q, k, v, pk, pv, 64, level)
    wrong = y.float()
    before = _sca_native.launch_counts()
    trace_before = len(_precision_backend.route_trace())
    with pytest.raises(ValueError) as excinfo:
        _backward(dy, q, k, v, pk, pv, wrong, lse, 64, level, deterministic)
    after = _sca_native.launch_counts()
    message = str(excinfo.value)
    assert "must have the q/k/v dtype" in message.lower()
    assert str(dtype) in message
    assert after == before, "a refused backward must not launch a native kernel"
    assert len(_precision_backend.route_trace()) == trace_before, (
        "a refused backward must not reach the precision backend")


def _abi_backward(dy, q, k, v, pk, pv, saved, lse, chunk=64):
    """The ABI edge the public wrapper sits on (level 0 only: it takes no level)."""
    return _sca_native.flash_sca_bwd(
        dy, q, k, v, saved, lse, chunk, None, pk, pv, None, None,
        segment_idx=None, bos_mask=None, reset_chunk_pos_per_seq=False,
        attn_method="default", deterministic=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_abi_edge_saved_state_dtype_recorded_not_asserted(dtype):
    """Gap 3b: what the ABI edge does with the wrong saved-output dtype - printed, not pinned.

    The public entry point refuses it (test_backward_refuses_mismatched_saved_state).
    Below that, ``xattn/ops/_sca_native.flash_sca_bwd`` is the level-0 native
    backward and takes no level. This records what it does today with an FP32 y
    (an output no level produces) and with the level-1 forward's y and lse; it
    asserts nothing, and a future change may legitimately make either raise.
    """
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    y0, lse0 = _forward(q, k, v, pk, pv, 64, 0)
    y1, lse1 = _forward(q, k, v, pk, pv, 64, 1)
    documented = _backward(dy, q, k, v, pk, pv, y0, lse0, 64, 0, True)
    for saved, lse, tag in ((y1, lse1, "level-1 forward's y and lse"),
                            (y0.float(), lse0, "FP32 y")):
        before = _sca_native.launch_counts()
        try:
            out = _abi_backward(dy, q, k, v, pk, pv, saved, lse)
        except Exception as exc:  # recorded, never a failure of this file
            print(f"ABI edge {dtype} {tag}: refused {type(exc).__name__}: {exc}")
            continue
        after = _sca_native.launch_counts()
        diff = [float((a.float() - b.float()).abs().max().item())
                for a, b in zip(out[:3], documented[:3])]
        print(f"ABI edge {dtype} {tag}: accepted, launches "
              f"{after[1] - before[1]} native backward(s); max|dq|,|dk|,|dv| vs the "
              f"documented level-0 route = {diff}")


# --- public-API refusals, level bands, one exactness cell ----------------
# The rows the batch above leaves open, on the same C=64 previous-K/V shape:
# E7 (saved output, saved state, LSE), E8 (chunk span type and range), E9
# (attn_method), E10 (Q longer than K/V), E11(a) (empty Q or K/V), plus
# ``scale=None`` being exactly ``1/sqrt(D)``.
#
# Each cell mutates exactly one argument of one documented call. The class is
# asserted strictly and the message is matched, case-insensitively, against
# the documented wording of the route that owns it - level 0 and the level 1
# forward validate in ``xattn/ops/_sca_native.py``, the level 1 backward in
# ``_backward_precision.py`` and ``_precision_backend.py`` - and asserts
# neither route's whole message.
# ``launch_counts()`` must be unchanged across the refused call: that is what
# makes these dispatch proofs and not exception-shape checks.
#
# The class and the wording are a pin on today's routing, not a public
# promise: the contract for this family is "validation errors stay errors", so
# folding two classes that are both conformant today into one would read here
# as a regression (the E14 record: native leaf ValueError vs generic
# RuntimeError, both conformant). A row that two routes can refuse carries an
# accepted string per route where the spellings differ, and says so where they
# share one word sequence instead (`E7-lse-dtype`).
#
# The level band is part of the cell, not a convenience: a level where no
# guard exists is simply not collected, so there is no ``pytest.skip`` here
# and an absent cell can never pass as a guarded one. Every row below is
# collected at both levels, 0 and 1 (``high_precision_level`` admits no
# other value).
#
# Two bands are easy to get wrong, both by the same mistake, an
# over-reading of a probe's "not-executed" as "no guard"; the two
# bands are:
#   * `E7-dy-shape` at level 1 raises from the precision backend, spelled
#     ``dO must have the reader output shape and Q dtype/device``
#     (``_precision_backend.py``), where level 0 says ``y_grad must have
#     the output shape`` (``_sca_native.py``).
#   * the length rows (E10, E11a) are public *forward* cells, and
#     ``_validate_qkv_heads`` is unconditional on that path at every level
#     (``sliding_chunk_attention.py``; the message is prefixed
#     ``flash_sca`` at level 0 and ``flash_sca_fwd`` at level 1, which is why
#     the substring below starts after the API name).
# Outside this file, and a different thing: the level 1 *backward* has no
# ``0 < Nq <= Nk`` guard at all (``csrc/attention/precision/native/adv_bwd.cu``
# admits ``p.Nq >= 0`` with no upper bound), so an over-long Q on that route is
# a finding and never a cell, and the ``host.cc`` launch limits are documented
# rather than tested.

_REFUSALS = (
    # key, levels, expected class, accepted lowercase message substrings
    # Levels 0 and 1 spell the same rule differently: `y_grad must have the
    # output shape` in the native leaf, `dO must have the reader output shape`
    # in the precision backend. Two accepted strings, one per route. Both are
    # written lowercase because the assertion lowercases the message.
    ("E7-dy-shape", (0, 1), ValueError,
     ("y_grad must have the output shape",
      "do must have the reader output shape")),
    ("E7-y-shape", (0, 1), ValueError,
     ("y must have the output shape",
      "saved attention output has incompatible shape")),
    # One string, not two: `lse must be fp32` is the word sequence the native
    # leaf (`flash_sca_bwd lse must be FP32 ...`) and the advanced validator
    # (`attention LSE must be FP32 ...`) share once the message is lowercased.
    # It is a common subsequence, so a rewrite on either side that drops "lse"
    # or "must be fp32" changes which rows speak before anything else fails.
    ("E7-lse-dtype", (0, 1), ValueError,
     ("lse must be fp32",)),
    # `chunk_size` must be an int, positive, and not a bool: the bool clause is
    # the one that matters, because `True` is an int in Python and would
    # otherwise mean `chunk_size = 1` and be accepted silently.
    ("E8-chunk-false", (0, 1), ValueError,
     ("chunk_size must be a positive integer", "invalid window/chunk span")),
    ("E8-chunk-zero", (0, 1), ValueError,
     ("chunk_size must be a positive integer", "invalid window/chunk span")),
    ("E8-chunk-float", (0, 1), ValueError,
     ("chunk_size must be a positive integer", "invalid window/chunk span")),
    ("E8-chunk-true", (0, 1), ValueError,
     ("chunk_size must be a positive integer", "invalid window/chunk span")),
    ("E9-method-unknown", (0, 1), ValueError,
     ("attn_method must be one of",)),
    ("E9-method-reserved", (0, 1), NotImplementedError,
     ("reserved for softdelta",)),
    # Forward cells: the length rule is unconditional there, so both
    # levels speak (`flash_sca requires ...` at 0, `flash_sca_fwd requires
    # ...` at 1). The advanced backward's missing length guard is a separate
    # finding and does not band these.
    ("E10-nq-gt-nt", (0, 1), ValueError,
     ("0 < q length <= k/v length",)),
    ("E11a-empty-q", (0, 1), ValueError,
     ("0 < q length <= k/v length",)),
    ("E11a-empty-k", (0, 1), ValueError,
     ("0 < q length <= k/v length",)),
)

# One flat entry per collected cell: (key, level, class, substrings).
_CELLS = tuple(
    (key, level, expected, substrings)
    for (key, levels, expected, substrings) in _REFUSALS
    for level in levels
)


def _refused_call(key, dtype, level):
    """Build the call the contract must refuse, without calling it.

    Any setup that legitimately launches - a good forward, so the saved-output
    rows have a real saved state and a real LSE to mutate - happens here, and
    the caller snapshots ``launch_counts()`` *after* this returns. A snapshot
    taken before the setup would see the fixture's own launch and fail every
    cell, which reads exactly like the product launching behind a refusal.
    """
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    if key in ("E10-nq-gt-nt", "E11a-empty-q", "E11a-empty-k"):
        # Length rows: through the public forward, so no saved state, LSE or dy
        # shape is in play and the only rule that can speak is the length rule.
        q_bad, k_bad, v_bad = q, k, v
        if key == "E10-nq-gt-nt":
            q_bad = torch.cat([q, q[:, :1]], dim=1).contiguous()  # Nq > Nt
        elif key == "E11a-empty-q":
            q_bad = q[:, :0].contiguous()
        else:
            k_bad, v_bad = k[:, :0].contiguous(), v[:, :0].contiguous()
        return lambda: xattn.flash_sca(q_bad, k_bad, v_bad, 64,
                                       high_precision_level=level)
    if key.startswith("E9"):
        method = "sliding" if key == "E9-method-unknown" else "sda"
        return lambda: xattn.flash_sca(q, k, v, 64, attn_method=method,
                                       high_precision_level=level,
                                       prev_k=pk, prev_v=pv)
    y, lse = _forward(q, k, v, pk, pv, 64, level)
    saved = y
    if key == "E7-dy-shape":
        return lambda: _backward(dy[:, :-1].contiguous(), q, k, v, pk, pv,
                                 saved, lse, 64, level, False)
    if key == "E7-y-shape":
        return lambda: _backward(dy, q, k, v, pk, pv,
                                 saved[:, :-1].contiguous(), lse, 64, level, False)
    if key == "E7-lse-dtype":
        return lambda: _backward(dy, q, k, v, pk, pv, saved, lse.to(dtype),
                                 64, level, False)
    chunk = {"E8-chunk-false": False, "E8-chunk-zero": 0,
             "E8-chunk-float": 2.0, "E8-chunk-true": True}[key]
    return lambda: _backward(dy, q, k, v, pk, pv, saved, lse, chunk, level, False)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cell", _CELLS)
def test_validation_errors_are_refused_before_dispatch(dtype, cell):
    """A refused call raises in Python, with the pinned class, before dispatch.

    The class varies by row: every row here is a ``ValueError`` except
    ``E9-method-reserved``, which is a ``NotImplementedError`` because the
    method it names exists but is not implemented.
    """
    key, level, expected, substrings = cell
    call = _refused_call(key, dtype, level)      # setup launches live here
    before = _sca_native.launch_counts()
    with pytest.raises(expected) as excinfo:
        call()
    after = _sca_native.launch_counts()
    message = str(excinfo.value)
    assert any(s in message.lower() for s in substrings), (key, level, message)
    assert after == before, (
        f"{key} level {level}: a refused call must not launch a native kernel")
    # Print what was refused and by whom: the class, the first message line,
    # and the API name the message carries when it has one (`flash_sca` at
    # level 0, `flash_sca_fwd` at level 1). This is the record the ledger reads;
    # the assertion above is the pin.
    print(f"{key} level {level} {dtype}: "
          f"{type(excinfo.value).__name__}: {message.splitlines()[0]}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_scale_none_is_the_explicit_inverse_sqrt(dtype):
    """``scale=None`` is exactly ``1/sqrt(D)``, LSE included.

    On the native level-0 route the omitted scale resolves in
    ``xattn/ops/_sca_native._scale`` as ``1.0 / q.shape[-1] ** 0.5`` (not
    ``resolve_scale``'s ``1.0 / math.sqrt(...)``, which serves the generic
    path). The explicit value below is that same expression, so the two calls
    differ only in which branch built the same double and the outputs must be
    bit-identical. LSE is compared as well as y: a common positive factor
    would leave y unchanged and still move the saved state, which is the only
    way this equivalence can break silently.

    Scope: D=64 and level 0. The equality is exact there because 64 ** 0.5 is
    exactly 8.0, so this cell pins the branch, not an ulp-level model of every
    D; level 1 resolves the scale on the precision route and is not covered
    here.
    """
    q, k, v, pk, pv, _ = _inputs(previous=64, dtype=dtype)
    explicit = 1.0 / q.shape[-1] ** 0.5
    y_none, lse_none = _forward(q, k, v, pk, pv, 64, 0, None)
    y_expl, lse_expl = _forward(q, k, v, pk, pv, 64, 0, explicit)
    assert y_none.dtype == dtype and y_expl.dtype == dtype, "level 0 returns y in the input dtype"
    assert torch.equal(y_none, y_expl), "scale=None is not the explicit 1/sqrt(D)"
    assert torch.equal(lse_none, lse_expl), "scale=None moved the LSE"
    print(f"scale=None vs 1/sqrt(D) ({dtype}): "
          f"y max |diff| = {(y_none.float() - y_expl.float()).abs().max().item():.3e}, "
          f"lse max |diff| = {(lse_none - lse_expl).abs().max().item():.3e}")
