"""Selection of the two pipelined native backward leaves, pinned by their launch counts.

(128,128): the pipelined fused M128 key leaf runs for every level-0 call its preflight accepts
while XATTN_SCA_BWD_KEY_ARM is unset or empty; any other value ("m64", "pair128p", ...) runs the
fused key leaf, and level 1 (the precision-repair backward) never reaches it. Its preflight is the fused M128 leaf's plus its
own, so a call the fused leaf serves with its M64 fallback is not taken either.
(256,256) and (128,128): the pipelined dQ leaf runs for every level-0 call it accepts while
XATTN_SCA_BWD_DQ_PIPE is unset, empty or "1"; "0" runs DqKernel, and any other value is refused
before the backward allocates anything.

Both pipelined leaves compute their shipped leaf's values bit for bit, so every arm's gradients
must equal the control arm's exactly. They run only on a compute capability 10.3 device whose
extension links the native FlashSCA group.
"""

import os

import pytest
import torch
import torch.nn.functional as F

import xattn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="native FlashSCA runs on compute capability 10.3",
)

KEY_ARM = "XATTN_SCA_BWD_KEY_ARM"
DQ_PIPE = "XATTN_SCA_BWD_DQ_PIPE"
_KNOBS = (KEY_ARM, DQ_PIPE, "XATTN_SCA_BWD", "XATTN_SCA_BWD_ARM", "XATTN_SCA_UMMA_FWD",
          "XATTN_SCA_ORDER", "XATTN_SCA_LAYOUT", "XATTN_SCA_BWD_LEGTIME")


def _ops():
    xattn_cuda = pytest.importorskip("xattn_cuda")
    ops = getattr(xattn_cuda, "ops", None)
    if ops is None or not hasattr(ops, "_flash_sca_native_kv_pipe_launches"):
        pytest.skip("build does not link the pipelined native backward leaves")
    return ops


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in _KNOBS:
        monkeypatch.delenv(name, raising=False)


def _case(dim, dtype=torch.bfloat16, length=700, heads=4, kv_heads=2, chunk=128, previous=False,
          seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def unit(t):
        return F.normalize(t, dim=-1).to(dtype)

    q = unit(torch.randn(1, length, heads, dim, device="cuda", generator=g))
    k = unit(torch.randn(1, length, kv_heads, dim, device="cuda", generator=g))
    v = (F.silu(torch.randn(1, length, kv_heads, dim, device="cuda", generator=g)) + 0.1).to(dtype)
    dy = torch.randn(1, length, heads, dim, device="cuda", generator=g).to(dtype)
    prev_k = prev_v = None
    total = length
    if previous:
        prev_k = unit(torch.randn(1, chunk, kv_heads, dim, device="cuda", generator=g))
        prev_v = (F.silu(torch.randn(1, chunk, kv_heads, dim, device="cuda", generator=g))
                  + 0.1).to(dtype)
        total += chunk
    # Three documents; with previous K/V the first one continues across the call.
    bos = torch.zeros(1, total, dtype=torch.bool, device="cuda")
    bos[0, 0] = True
    bos[0, total // 3 + 7] = True
    bos[0, (2 * total) // 3 + 5] = True
    return dict(q=q, k=k, v=v, dy=dy, prev_k=prev_k, prev_v=prev_v, chunk=chunk, bos=bos)


def _forward(x, level):
    with torch.no_grad():
        return xattn.flash_sca_fwd(
            x["q"], x["k"], x["v"], x["chunk"], None, x["prev_k"], x["prev_v"],
            bos_mask=x["bos"], backend="auto", high_precision_level=level)


def _backward(ops, x, state, lse, level, deterministic, env):
    """One backward with `env` applied (None = unset); returns (grads, kv launches, dQ launches,
    record)."""
    saved = {name: os.environ.get(name) for name in env}
    try:
        for name, value in env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        kv0 = ops._flash_sca_native_kv_pipe_launches()
        dq0 = ops._flash_sca_native_dq_pipe_launches()
        with torch.no_grad():
            grads = xattn.flash_sca_bwd(
                x["dy"], x["q"], x["k"], x["v"], state, lse, x["chunk"], None, x["prev_k"],
                x["prev_v"], bos_mask=x["bos"], backend="auto", deterministic=deterministic,
                high_precision_level=level)
        torch.cuda.synchronize()
        record = ops._flash_sca_last_backward_preparation()
        return (grads, ops._flash_sca_native_kv_pipe_launches() - kv0,
                ops._flash_sca_native_dq_pipe_launches() - dq0, record)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _same(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        if left is None or right is None:
            if not (left is None and right is None):
                return False
        elif left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            return False
    return True


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("previous", [False, True])
def test_kv_pipe_is_the_level0_key_body_at_128(dtype, deterministic, previous):
    # The key arm is the only arm this test varies, so the dQ arm is pinned off in every call:
    # at (128,128) the level-0 dQ body is the pipelined dQ leaf by default too, and an unpinned
    # control would count that launch here. The counts below are the key arm's alone.
    ops = _ops()
    x = _case(128, dtype, previous=previous)
    state, lse = _forward(x, 0)
    control, kv, dq, record = _backward(ops, x, state, lse, 0, deterministic,
                                        {KEY_ARM: "m64", DQ_PIPE: "0"})
    assert (kv, dq) == (0, 0), "the control arm must run the fused key leaf and DqKernel"
    assert record["native_key"] and record["key_rows"] == 128, record
    first = None
    for value in (None, ""):
        grads, kv, dq, record = _backward(ops, x, state, lse, 0, deterministic,
                                          {KEY_ARM: value, DQ_PIPE: "0"})
        assert (kv, dq) == (1, 0), (value, kv, dq)
        assert record["native_key"] and record["key_rows"] == 128, record
        assert _same(grads, control), f"{KEY_ARM}={value!r} differs from the fused key leaf"
        first = first or grads
    for value in ("pair128p", "fused"):
        grads, kv, dq, _ = _backward(ops, x, state, lse, 0, deterministic,
                                     {KEY_ARM: value, DQ_PIPE: "0"})
        assert (kv, dq) == (0, 0), (value, kv, dq)
        assert _same(grads, control)
    if deterministic:
        again, kv, dq, _ = _backward(ops, x, state, lse, 0, True,
                                     {KEY_ARM: None, DQ_PIPE: "0"})
        assert (kv, dq) == (1, 0) and _same(again, first), (
            "deterministic backward must repeat bit-exactly")


@pytest.mark.parametrize("chunk", [64, 192])
def test_kv_pipe_takes_exactly_the_calls_the_fused_m128_leaf_takes(chunk):
    # A previous plane whose length is not a multiple of the 128-row tile may be refused by the
    # fused M128 leaf (which then falls back to its M64 body); the pipelined leaf shares that
    # preflight, so it runs exactly when the control's key tile is 128.
    ops = _ops()
    x = _case(128, chunk=chunk, previous=True, length=640, seed=chunk)
    state, lse = _forward(x, 0)
    control, kv, _, record = _backward(ops, x, state, lse, 0, False, {KEY_ARM: "m64"})
    assert kv == 0
    control_rows = record["key_rows"]
    grads, kv, _, record = _backward(ops, x, state, lse, 0, False, {KEY_ARM: None})
    assert kv == (1 if control_rows == 128 else 0), (control_rows, kv)
    assert record["key_rows"] == control_rows
    assert _same(grads, control)


@pytest.mark.parametrize("dim", [128, 256])
def test_level1_never_takes_a_pipelined_leaf(dim):
    from xattn.ops import _precision_backend

    ops = _ops()
    x = _case(dim)
    state, lse = _forward(x, 1)
    control, kv, dq, _ = _backward(ops, x, state, lse, 1, True, {KEY_ARM: "m64", DQ_PIPE: "0"})
    assert (kv, dq) == (0, 0)
    grads, kv, dq, _ = _backward(ops, x, state, lse, 1, True, {KEY_ARM: None, DQ_PIPE: None})
    assert (kv, dq) == (0, 0), "level 1 must run the shipped leaves"
    assert _precision_backend.route_trace()[-1]["resolved"] == "native"
    assert _same(grads, control)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("previous", [False, True])
def test_dq_pipe_is_the_level0_dq_body_at_256(dtype, deterministic, previous):
    ops = _ops()
    x = _case(256, dtype, previous=previous)
    state, lse = _forward(x, 0)
    control, kv, dq, record = _backward(ops, x, state, lse, 0, deterministic, {DQ_PIPE: "0"})
    assert (kv, dq) == (0, 0), "the control arm must run DqKernel"
    assert record["native_dq"], record
    first = None
    for value in (None, "", "1"):
        grads, kv, dq, record = _backward(ops, x, state, lse, 0, deterministic, {DQ_PIPE: value})
        assert (kv, dq) == (0, 1), (value, kv, dq)
        assert record["native_dq"], record
        assert _same(grads, control), f"{DQ_PIPE}={value!r} differs from DqKernel"
        first = first or grads
    if deterministic:
        again, _, dq, _ = _backward(ops, x, state, lse, 0, True, {DQ_PIPE: None})
        assert dq == 1 and _same(again, first), "deterministic backward must repeat bit-exactly"


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("previous", [False, True])
def test_dq_pipe_is_the_level0_dq_body_at_128(dtype, deterministic, previous):
    # The reciprocal of the key-pipe coverage above, at the bucket where both pipelined leaves
    # live: the key arm is left at its default (so it runs the pipelined key leaf and its count is
    # 1 in every arm of this test), and the dQ arm is what varies - "0" runs DqKernel, unset, empty
    # and "1" run the pipelined dQ leaf.
    ops = _ops()
    x = _case(128, dtype, previous=previous)
    state, lse = _forward(x, 0)
    control, kv, dq, record = _backward(ops, x, state, lse, 0, deterministic, {DQ_PIPE: "0"})
    assert (kv, dq) == (1, 0), "the control arm must run DqKernel behind the pipelined key leaf"
    assert record["native_dq"] and record["qk_bucket"] == 128, record
    first = None
    for value in (None, "", "1"):
        grads, kv, dq, record = _backward(ops, x, state, lse, 0, deterministic, {DQ_PIPE: value})
        assert (kv, dq) == (1, 1), (value, kv, dq)
        assert record["native_dq"] and record["qk_bucket"] == 128, record
        assert _same(grads, control), f"{DQ_PIPE}={value!r} differs from DqKernel"
        first = first or grads
    if deterministic:
        again, kv, dq, _ = _backward(ops, x, state, lse, 0, True, {DQ_PIPE: None})
        assert (kv, dq) == (1, 1) and _same(again, first), (
            "deterministic backward must repeat bit-exactly")


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("value", ["2", "off", " 0", "01"])
def test_dq_pipe_value_naming_no_arm_is_refused(dim, value):
    ops = _ops()
    x = _case(dim)
    state, lse = _forward(x, 0)
    seq = ops._flash_sca_last_backward_preparation()["record_seq"]
    kv0 = ops._flash_sca_native_kv_pipe_launches()
    dq0 = ops._flash_sca_native_dq_pipe_launches()
    with pytest.raises(RuntimeError, match="XATTN_SCA_BWD_DQ_PIPE must be unset, empty, '0' or '1'"):
        _backward(ops, x, state, lse, 0, False, {DQ_PIPE: value})
    assert ops._flash_sca_native_kv_pipe_launches() == kv0
    assert ops._flash_sca_native_dq_pipe_launches() == dq0
    record = ops._flash_sca_last_backward_preparation()
    assert record["record_seq"] != seq and not record["launched"], record
