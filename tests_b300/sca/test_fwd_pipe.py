"""Smoke test of the pipelined native forward at level 0, and of the two forward arms.

Which calls are covered: the square buckets D = Dv in {256, 128, 64} with every scale and previous
kind; every (D, Dv) bucket pair and dims below their bucket (8..256, multiples of 8: PAIRS); the
unequal (64, 128) pair (the `speed` check's mixed-64-128 guard row); and the directed tests (dead rows
inside live CTAs, Nq < Nk, the late-max rescale/clear branches, the FULL-tile rule at chunks that
are not multiples of the key tile) at the buckets of the timed and guarded rows (BUCKETS).

What this file is NOT: the evidence that certifies the pipelined body. That evidence is the pinned,
independent FP64 oracle at this body's shapes: gekko-shapes 12/12 on this head (the training cell and
both prefill cells of each model take the pipelined body, 0.9B at (256,256) and 4B at (128,128);
prefill-continued is its witness with a live previous plane), the full sca-contract (its D = Dv = 128
level-0 configurations take the body by default) and the 26 sca-contract cases re-parameterized to
D = Dv = 256 and to D = Dv = 128 at level 0 (312 configurations each, through
checks/oracle/tests/_numerics.py:check_case, unchanged), each row carrying its launch deltas.
The FP64 comparisons below use that oracle's per-element rule, |y - ref| <= ATOL + RTOL * |ref| with a
mismatch count of zero, and its own ATOL/RTOL (loaded from XATTN_SCA_ORACLE_DIR when set, else cited),
but the FP64 reference here is this file's.

The arms (umma_fwd_pipe_host.h; precedence in host.cc ResolveForwardBody), resolved once per call:
  XATTN_SCA_FWD_PIPE unset/'' and XATTN_SCA_UMMA_FWD unset/''  -> the pipelined body where it accepts
                                                                  the call, else main's generic body
  XATTN_SCA_FWD_PIPE unset/'' and XATTN_SCA_UMMA_FWD '0' / '1'  -> the generic body / the older leaf
  XATTN_SCA_FWD_PIPE '1' and XATTN_SCA_UMMA_FWD unset/''       -> as the first row
  XATTN_SCA_FWD_PIPE '1' and XATTN_SCA_UMMA_FWD '0' or '1'     -> raises (two explicit bodies)
  XATTN_SCA_FWD_PIPE '0'                                        -> main's forward (UMMA_FWD decides)
  XATTN_SCA_FWD_PIPE any other value, '2' included             -> raises before anything is allocated
The forward tag records the pipelined body and the older leaf alike as umma -> umma/ran, so every call
here also reads _flash_sca_native_fwd_pipe_launches and _flash_sca_native_fwd_leaf_launches by
difference around that ONE call: a body is (tag rows, pipe-launch delta, leaf-launch delta).

Not reachable here: a pre-launch refusal of the pipelined body (the two-row fallback) and a launch call
that reports an error (which raises, with no fallback). A healthy sm_103 device with this build cannot
produce either. A probe build (XATTN_FWD_PROBE_HOOKS prepended to copies of the three forward
instantiation TUs; umma_fwd_pipe_host.h) exercises them under XATTN_FWD_PROBE=pipe-refuse,
pipe-posterr and leaf-posterr, and counts the lazy-rescale and clear branches on the directed late-max
cases below. No shipped build compiles those hooks.

Runs only on a compute capability 10.3 device whose extension links the pipelined forward.
"""

import importlib.util
import math
import os

import pytest
import torch
import torch.nn.functional as F

import xattn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="native FlashSCA runs on compute capability 10.3",
)

KNOBS = ("XATTN_SCA_FWD_PIPE", "XATTN_SCA_UMMA_FWD", "XATTN_SCA_ORDER", "XATTN_SCA_LAYOUT")
UMMA_ROW = {("umma", "umma", "ran"): 1}
GENERIC_ROW = {("mma_sync", "mma_sync", "ran"): 1}
# (forward tag rows, pipe-launch delta, leaf-launch delta) of one call
PIPE = (UMMA_ROW, 1, 0)
LEAF = (UMMA_ROW, 0, 1)
GENERIC = (GENERIC_ROW, 0, 0)
NOTHING = ({}, 0, 0)


def _oracle_tolerances():
    """The pinned oracle's tolerances: checks/oracle/tests/softdelta/_inputs.py, which
    checks/oracle/tests/_numerics.py imports. Loaded from XATTN_SCA_ORACLE_DIR (the oracle directory)
    when set; otherwise the same values, cited from that file (sha256 781136d6...86d1da)."""
    root = os.environ.get("XATTN_SCA_ORACLE_DIR")
    if root:
        path = os.path.join(root, "tests", "softdelta", "_inputs.py")
        spec = importlib.util.spec_from_file_location("_xattn_sca_oracle_inputs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ATOL, module.RTOL, path
    return ({torch.float16: 2e-4, torch.bfloat16: 1e-3},
            {torch.float16: 1e-3, torch.bfloat16: 1e-2},
            "cited from checks/oracle/tests/softdelta/_inputs.py")


ATOL, RTOL, TOLERANCE_SOURCE = _oracle_tolerances()


def _ops():
    xattn_cuda = pytest.importorskip("xattn_cuda")
    ops = getattr(xattn_cuda, "ops", None)
    if ops is None or not hasattr(ops, "_flash_sca_native_fwd_pipe_launches") or not hasattr(
            ops, "_flash_sca_native_fwd_leaf_launches"):
        pytest.skip("build does not link the pipelined native forward and its launch counts")
    return ops


def _chunk_mask(length, total, chunk, q_segment=None, k_segment=None):
    """[B or 1, Nq, T] visibility of the chunk rule, Q right-aligned to [previous; current] keys."""
    pos_q = torch.arange(length, device="cuda") + total - length
    pos_k = torch.arange(total, device="cuda")
    low = torch.div(pos_q, chunk, rounding_mode="floor") * chunk - chunk
    mask = ((pos_k[None, :] >= low[:, None]) & (pos_k[None, :] <= pos_q[:, None]))[None]
    if q_segment is not None:
        mask = mask & (q_segment[:, :, None] == k_segment[:, None, :])
    return mask


def _case(q, k, v, previous, chunk, kwargs, mask):
    return dict(q=q, k=k[:, previous:].contiguous(), v=v[:, previous:].contiguous(),
                prev_k=k[:, :previous].contiguous() if previous else None,
                prev_v=v[:, :previous].contiguous() if previous else None,
                chunk=chunk, kwargs=kwargs, keys=k, values=v, mask=mask)


def _bos_inputs(dtype, previous=0, new_document=False, dim=256, value_dim=None, length=333,
                q_length=None):
    """BOS segments over [previous; current] keys, Q right-aligned (q_length <= length rows). With
    previous rows and new_document=False the call continues the document of the previous plane (no
    BOS at the call's first key), so that plane is visible to the first chunk's rows;
    new_document=True starts a segment at the call's first key."""
    torch.manual_seed(0)
    batch, heads, kv_heads, chunk = 2, 4, 2, 128
    value_dim = dim if value_dim is None else value_dim
    q_length = length if q_length is None else q_length
    total = previous + length
    q = F.normalize(torch.randn(batch, q_length, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, total, kv_heads, dim, device="cuda"), dim=-1).to(dtype)
    v = (F.silu(torch.randn(batch, total, kv_heads, value_dim, device="cuda")) + 0.1).to(dtype)
    bos = torch.zeros(batch, total, dtype=torch.bool, device="cuda")
    bos[:, 0] = True
    bos[:, previous + 150] = True
    bos[1, previous + 40] = True
    if previous and new_document:
        bos[:, previous] = True
    segment = torch.cumsum(bos.long(), dim=1)
    mask = _chunk_mask(q_length, total, chunk, segment[:, total - q_length:], segment)
    return _case(q, k, v, previous, chunk, {"bos_mask": bos}, mask)


def _pair_dead_rows_inputs(dtype, length=300, dim=256, value_dim=None):
    """The q/k segment-pair route with every 5th query row in a segment no key carries: rows with no
    visible key inside CTAs that have live rows (the contract's 'zero output, never NaN' clause on a
    mixed CTA, which a bos_mask cannot reach: every BOS segment starts with its own key)."""
    torch.manual_seed(1)
    batch, heads, kv_heads, chunk = 2, 2, 2, 64
    value_dim = dim if value_dim is None else value_dim
    q = F.normalize(torch.randn(batch, length, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda"), dim=-1).to(dtype)
    v = (F.silu(torch.randn(batch, length, kv_heads, value_dim, device="cuda")) + 0.1).to(dtype)
    kseg = (100 - torch.arange(length, device="cuda") // 41).expand(batch, -1).contiguous()
    qseg = kseg.clone()
    qseg[:, ::5] = -7
    mask = _chunk_mask(length, length, chunk, qseg, kseg)
    return _case(q, k, v, 0, chunk, {"q_segment_idx": qseg, "k_segment_idx": kseg}, mask)


def _late_max_inputs(dtype, route, spike, dim=256, value_dim=None):
    """Directed lazy-rescale inputs (D = Dv = 256, C = 128, scale 1, one CTA row-tile per 128 rows):
    every query leans on one direction u, and key j's projection on u rises 6 natural units per
    64-key tile (about 8.7 log2 units, around the kernel's 8-unit threshold given the queries' own
    spread), so a row's running maximum crosses the threshold from tile to tile late in its window
    (the rescale branch). spike=True adds 100 natural units (about 144 log2 units) at keys 200, 290
    and 319, so the rescale factor underflows to exactly 0 (the clear branch). route 'dense' keeps
    whole tiles visible to whole CTAs (the FULL path); route 'pair' gives every 5th query row a segment
    no key carries (dead rows inside live CTAs; the masked path)."""
    g = torch.Generator(device="cuda").manual_seed(7)
    batch, length, heads, kv_heads, chunk = 1, 320, 2, 1, 128
    value_dim = dim if value_dim is None else value_dim
    u = F.normalize(torch.randn(dim, device="cuda", generator=g), dim=0)
    q = F.normalize(torch.randn(batch, length, heads, dim, device="cuda", generator=g), dim=-1) + u
    proj = 6.0 * torch.arange(length, device="cuda", dtype=torch.float32) / 64.0
    if spike:
        proj[torch.tensor([200, 290, 319], device="cuda")] += 100.0
    k = F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda", generator=g), dim=-1)
    k = k + proj[None, :, None, None] * u
    v = F.silu(torch.randn(batch, length, kv_heads, value_dim, device="cuda", generator=g)) + 0.1
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    kwargs, qseg, kseg = {}, None, None
    if route == "pair":
        kseg = torch.zeros(batch, length, dtype=torch.long, device="cuda")
        qseg = kseg.clone()
        qseg[:, ::5] = -7
        kwargs = {"q_segment_idx": qseg, "k_segment_idx": kseg}
    mask = _chunk_mask(length, length, chunk, qseg, kseg)
    return _case(q, k, v, 0, chunk, kwargs, mask)


def _rescale_profile(x, scale):
    """The lazy-rescale decisions the inputs call for, from FP64 scores and independent of the kernel:
    per row, the visible maxima of each 64-key tile (log2 units) walked in key order; a crossing is a
    tile whose maximum exceeds the row's running maximum by more than 8, a clear one whose excess is
    above 126 (where exp2 of the factor's exponent underflows to 0). Returns (rows that cross at
    least once, rows with a clearing crossing)."""
    q, keys, mask = x["q"].double(), x["keys"].double(), x["mask"]
    group = q.shape[2] // keys.shape[2]
    s = torch.einsum("bqhd,bkhd->bhqk", q, keys.repeat_interleave(group, dim=2)) * scale / math.log(2.0)
    s = s.masked_fill(~mask[:, None], float("-inf"))
    tiles = (s.shape[-1] + 63) // 64
    s = F.pad(s, (0, tiles * 64 - s.shape[-1]), value=float("-inf"))
    s = s.view(*s.shape[:-1], tiles, 64).amax(-1)
    m = torch.full(s.shape[:-1], float("-inf"), dtype=torch.float64, device=s.device)
    crossed = torch.zeros(m.shape, dtype=torch.bool, device=s.device)
    cleared = torch.zeros_like(crossed)
    for t in range(tiles):
        tm = s[..., t]
        first = torch.isinf(m) & torch.isfinite(tm)
        jump = torch.isfinite(m) & (tm > m + 8.0)
        crossed |= jump
        cleared |= jump & (tm - m > 126.0)
        m = torch.where(first | jump, tm, m)
    return int(crossed.sum()), int(cleared.sum())


def _reference(x, scale):
    """FP64 y and LSE of the explicit mask; a row with no visible key gives y = 0 and LSE = -inf."""
    q, keys, values, mask = x["q"].double(), x["keys"].double(), x["values"].double(), x["mask"]
    group = q.shape[2] // keys.shape[2]
    s = torch.einsum("bqhd,bkhd->bhqk", q, keys.repeat_interleave(group, dim=2)) * scale
    s = s.masked_fill(~mask[:, None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    has = torch.isfinite(lse)
    p = torch.where(has[..., None], torch.exp(s - torch.where(has, lse, torch.zeros_like(lse))[..., None]),
                    torch.zeros_like(s))
    return torch.einsum("bhqk,bkhd->bqhd", p, values.repeat_interleave(group, dim=2)), lse


def _check_fp64(y, lse, x, scale, dtype):
    """The oracle's rule on every element of y (FP64 rounded to the output dtype) and on every finite
    LSE (FP64 rounded to FP32, the forward ATOL/RTOL); rows with no visible key must be y = 0 and
    LSE = -inf exactly. Returns a description of every violation (empty when it holds)."""
    y_ref, lse_ref = _reference(x, scale)
    bad = []
    e = y_ref.to(dtype).double()
    a = y.double()
    n = int((~((a - e).abs() <= ATOL[dtype] + RTOL[dtype] * e.abs())).sum())
    if n:
        bad.append(f"y: {n} of {a.numel()} elements outside ATOL+RTOL*|ref| ({TOLERANCE_SOURCE}), "
                   f"max abs {float((a - e).abs().max()):.3e}")
    finite = torch.isfinite(lse_ref)
    if not torch.equal(torch.isfinite(lse), finite):
        bad.append("lse: the -inf (no visible key) pattern differs from the reference")
    else:
        le, la = lse_ref[finite].float().double(), lse.double()[finite]
        n = int((~((la - le).abs() <= ATOL[dtype] + RTOL[dtype] * le.abs())).sum())
        if n:
            bad.append(f"lse: {n} of {la.numel()} finite rows outside ATOL+RTOL*|ref|")
        empty = ~finite  # [B, H, Nq]
        if bool(empty.any()):
            if not bool(torch.isneginf(lse[empty]).all()):
                bad.append("lse: a row with no visible key is not -inf")
            if not bool((y.permute(0, 2, 1, 3)[empty] == 0).all()):
                bad.append("y: a row with no visible key is not exactly 0")
    if not bool(torch.isfinite(y.float()).all()):
        bad.append("y: non-finite values")
    return bad


def _set_env(monkeypatch, env):
    for key in KNOBS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _counts(ops):
    rows = {tuple(r[:4]): int(r[4]) for r in ops._flash_sca_native_phase_tags()}
    return rows, int(ops._flash_sca_native_fwd_pipe_launches()), int(ops._flash_sca_native_fwd_leaf_launches())


def _body_delta(before, after):
    (r0, p0, l0), (r1, p1, l1) = before, after
    forward = {key[1:]: r1[key] - r0.get(key, 0) for key in r1 if key[0] == "forward" and r1[key] != r0.get(key, 0)}
    return forward, p1 - p0, l1 - l0


def _forward(ops, monkeypatch, env, x, scale=None, level=0, inference=False):
    """One flash_sca_fwd call under `env`; returns (y, lse, body) with body = (forward tag rows,
    pipe-launch delta, leaf-launch delta) read around that one call."""
    _set_env(monkeypatch, env)
    before = _counts(ops)
    with (torch.inference_mode() if inference else torch.no_grad()):
        y, lse = xattn.flash_sca_fwd(x["q"], x["k"], x["v"], x["chunk"], scale, x["prev_k"], x["prev_v"],
                                     high_precision_level=level, **x["kwargs"])
    torch.cuda.synchronize()
    return y, lse, _body_delta(before, _counts(ops))


SCALES = (None, 0.0, 1.0 / 16, -1.0 / 16)


@pytest.mark.parametrize("scale", SCALES, ids=("default", "zero", "pos", "neg"))
@pytest.mark.parametrize("previous", ["none", "continued", "new-document"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize("dim", [256, 128, 64], ids=("d256", "d128", "d64"))
def test_default_runs_the_pipelined_body_against_fp64(monkeypatch, dim, dtype, previous, scale):
    ops = _ops()
    x = _bos_inputs(dtype, previous=0 if previous == "none" else 128, new_document=previous == "new-document",
                    dim=dim)
    if previous != "none":
        # the case is what it claims: the previous plane is visible exactly when the document continues
        assert bool(x["mask"][:, :, :128].any()) == (previous == "continued")
    ref_scale = dim ** -0.5 if scale is None else scale
    y, lse, body = _forward(ops, monkeypatch, {}, x, scale)
    assert body == PIPE, body
    assert y.dtype == dtype and lse.dtype == torch.float32
    assert not _check_fp64(y, lse, x, ref_scale, dtype), _check_fp64(y, lse, x, ref_scale, dtype)
    if scale == 0.0:
        # every visible score is 0, so LSE = log(number of visible keys) (here through the oracle's rule)
        count = x["mask"].sum(-1).double()[:, None].expand_as(lse)
        assert torch.allclose(lse.double()[count > 0], count[count > 0].log(), atol=ATOL[dtype], rtol=RTOL[dtype])
    # The control arm is main's forward on the same inputs, held to the same rule.
    y0, lse0, body0 = _forward(ops, monkeypatch, {"XATTN_SCA_FWD_PIPE": "0"}, x, scale)
    assert body0 == GENERIC, body0
    assert not _check_fp64(y0, lse0, x, ref_scale, dtype)


PAIRS = ((64, 256), (128, 64), (128, 256), (256, 64), (256, 128), (96, 96), (32, 32), (192, 80),
         (40, 136), (8, 8), (256, 24))


@pytest.mark.parametrize("previous", ["none", "continued"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize("dims", PAIRS, ids=[f"d{d}v{v}" for d, v in PAIRS])
def test_every_dim_pair_runs_the_pipelined_body_against_fp64(monkeypatch, dims, dtype, previous):
    """Every (D, Dv) the native maps describe (8..256, multiples of 8) takes the pipelined body: the
    unequal bucket pairs, and dims below their bucket (zero-filled by the TMA, stored at the logical
    extent). Held to the FP64 oracle, and a repeat is bit-identical."""
    ops = _ops()
    dim, value_dim = dims
    x = _bos_inputs(dtype, previous=0 if previous == "none" else 128, dim=dim, value_dim=value_dim)
    y, lse, body = _forward(ops, monkeypatch, {}, x)
    assert body == PIPE, (dims, body)
    assert y.shape[-1] == value_dim
    assert not _check_fp64(y, lse, x, dim ** -0.5, dtype), _check_fp64(y, lse, x, dim ** -0.5, dtype)
    y2, lse2, _ = _forward(ops, monkeypatch, {}, x)
    assert torch.equal(y, y2) and torch.equal(lse, lse2)


@pytest.mark.parametrize("dim", [256, 128, 64], ids=("d256", "d128", "d64"))
def test_default_repeats_bit_exactly_and_matches_inference_mode(monkeypatch, dim):
    ops = _ops()
    x = _bos_inputs(torch.bfloat16, previous=128, dim=dim)
    first = _forward(ops, monkeypatch, {}, x)
    second = _forward(ops, monkeypatch, {}, x)
    inference = _forward(ops, monkeypatch, {}, x, inference=True)
    assert first[2] == PIPE and second[2] == PIPE and inference[2] == PIPE
    for other in (second, inference):
        assert torch.equal(first[0], other[0]) and torch.equal(first[1], other[1])


@pytest.mark.parametrize("env, want", [
    ({}, PIPE),
    ({"XATTN_SCA_FWD_PIPE": ""}, PIPE),
    ({"XATTN_SCA_UMMA_FWD": ""}, PIPE),
    ({"XATTN_SCA_FWD_PIPE": "", "XATTN_SCA_UMMA_FWD": ""}, PIPE),
    ({"XATTN_SCA_FWD_PIPE": "1"}, PIPE),
    ({"XATTN_SCA_FWD_PIPE": "1", "XATTN_SCA_UMMA_FWD": ""}, PIPE),
    ({"XATTN_SCA_UMMA_FWD": "0"}, GENERIC),
    ({"XATTN_SCA_UMMA_FWD": "1"}, LEAF),
    ({"XATTN_SCA_FWD_PIPE": "", "XATTN_SCA_UMMA_FWD": "0"}, GENERIC),
    ({"XATTN_SCA_FWD_PIPE": "", "XATTN_SCA_UMMA_FWD": "1"}, LEAF),
    ({"XATTN_SCA_FWD_PIPE": "0"}, GENERIC),
    ({"XATTN_SCA_FWD_PIPE": "0", "XATTN_SCA_UMMA_FWD": "0"}, GENERIC),
    ({"XATTN_SCA_FWD_PIPE": "0", "XATTN_SCA_UMMA_FWD": "1"}, LEAF),
])
@pytest.mark.parametrize("dim", [256, 128, 64], ids=("d256", "d128", "d64"))
def test_arm_precedence_selects_its_body(monkeypatch, dim, env, want):
    """The same precedence at both buckets: each has the pipelined body, the older leaf and the generic
    body, so the arms select the same body kind at (256,256) and at (128,128)."""
    ops = _ops()
    x = _bos_inputs(torch.bfloat16, dim=dim)
    y, lse, body = _forward(ops, monkeypatch, env, x)
    assert body == want, (dim, env, body)
    assert not _check_fp64(y, lse, x, dim ** -0.5, torch.bfloat16)


@pytest.mark.parametrize("umma", ["0", "1"])
def test_conflicting_explicit_arms_raise_before_anything_runs(monkeypatch, umma):
    ops = _ops()
    x = _bos_inputs(torch.bfloat16)
    _set_env(monkeypatch, {"XATTN_SCA_FWD_PIPE": "1", "XATTN_SCA_UMMA_FWD": umma})
    before = _counts(ops)
    with pytest.raises(RuntimeError, match="XATTN_SCA_FWD_PIPE=1 conflicts with XATTN_SCA_UMMA_FWD"):
        with torch.no_grad():
            xattn.flash_sca_fwd(x["q"], x["k"], x["v"], x["chunk"], None, None, None, **x["kwargs"])
    torch.cuda.synchronize()
    assert _body_delta(before, _counts(ops)) == NOTHING


@pytest.mark.parametrize("case", ["level1", "d128-level1", "order", "layout"])
@pytest.mark.parametrize("pipe", ["", "1"], ids=("pipe-unset", "pipe-1"))
def test_calls_it_does_not_serve_run_mains_default(monkeypatch, case, pipe):
    """Calls PipeAccepts does not admit (level 1 at either bucket, D != Dv) and calls under a live
    scheduling instrument run main's generic body. D = Dv = 128 at level 0 is not among them: it is the
    pipelined body's second bucket (test_default_runs_the_pipelined_body_against_fp64)."""
    ops = _ops()
    env = {"order": {"XATTN_SCA_ORDER": "sdesc"}, "layout": {"XATTN_SCA_LAYOUT": "hf"}}.get(case, {})
    if pipe:
        env = {**env, "XATTN_SCA_FWD_PIPE": pipe}
    level = {"level1": 1, "d128-level1": 1}.get(case, 0)
    x = _bos_inputs(torch.bfloat16, dim=128 if case.startswith("d128") else 256,
                    value_dim=128 if case.startswith("d128") or case == "dv128" else 256)
    y, _, body = _forward(ops, monkeypatch, env, x, level=level)
    assert body == GENERIC, (case, pipe, body)
    # No level keeps an FP32 output state: y is in the input dtype at level 0 and level 1.
    assert y.dtype == torch.bfloat16
    assert bool(torch.isfinite(y.float()).all())


@pytest.mark.parametrize("value", ["2", "3", "x", " 1", "1 ", "01", "on", "true"])
@pytest.mark.parametrize("dim", [256, 128])
def test_values_that_name_no_arm_raise(monkeypatch, value, dim):
    ops = _ops()
    x = _bos_inputs(torch.bfloat16, dim=dim)
    _set_env(monkeypatch, {"XATTN_SCA_FWD_PIPE": value})
    before = _counts(ops)
    with pytest.raises(RuntimeError, match="XATTN_SCA_FWD_PIPE must be unset, empty, '0' or '1'"):
        with torch.no_grad():
            xattn.flash_sca_fwd(x["q"], x["k"], x["v"], x["chunk"], None, None, None, **x["kwargs"])
    torch.cuda.synchronize()
    assert _body_delta(before, _counts(ops)) == NOTHING


# The bucket pairs of the timed and guarded rows, for the directed tests below: the square buckets and
# the unequal (64, 128) pair (the `speed` check's mixed-64-128 guard row).
BUCKETS = ((256, 256), (128, 128), (64, 64), (64, 128))
BUCKET_IDS = ("d256", "d128", "d64", "d64v128")


@pytest.mark.parametrize("scale", SCALES, ids=("default", "zero", "pos", "neg"))
@pytest.mark.parametrize("previous", ["none", "continued", "new-document"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
def test_unequal_dims_against_fp64(monkeypatch, dtype, previous, scale):
    """(D, Dv) = (64, 128) with every scale and previous kind: the score is over D, the output over Dv,
    and the O store and dead-row writes use the logical value extent."""
    ops = _ops()
    dim, value_dim = 64, 128
    x = _bos_inputs(dtype, previous=0 if previous == "none" else 128,
                    new_document=previous == "new-document", dim=dim, value_dim=value_dim)
    ref_scale = dim ** -0.5 if scale is None else scale
    y, lse, body = _forward(ops, monkeypatch, {}, x, scale)
    assert body == PIPE, body
    assert y.dtype == dtype and y.shape[-1] == value_dim
    assert not _check_fp64(y, lse, x, ref_scale, dtype), _check_fp64(y, lse, x, ref_scale, dtype)
    y2, lse2, _ = _forward(ops, monkeypatch, {}, x, scale)
    assert torch.equal(y, y2) and torch.equal(lse, lse2)


@pytest.mark.parametrize("chunk", [256, 200, 192, 130, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize("dims", BUCKETS, ids=BUCKET_IDS)
def test_full_tile_rule_at_chunks_off_the_key_tile(monkeypatch, dims, dtype, chunk):
    """The FULL-tile fast path at chunks that are not multiples of the key tile, with a whole previous
    chunk (chunk = prev, N = 256): a FULL test narrower than the body's key tile admits a tile whose
    second half lies past a row's last visible key, a silent wrong y/LSE (a randomized check found 106 and
    120 bad rows of 256 at chunk 200 and 192 on the 128-key body). 256 is the aligned control."""
    ops = _ops()
    dim, value_dim = dims
    torch.manual_seed(0)
    n, heads, kv_heads = 256, 2, 2
    total = chunk + n
    q = F.normalize(torch.randn(1, n, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(1, total, kv_heads, dim, device="cuda"), dim=-1).to(dtype)
    v = (F.silu(torch.randn(1, total, kv_heads, value_dim, device="cuda")) + 0.1).to(dtype)
    x = _case(q, k, v, chunk, chunk, {}, _chunk_mask(n, total, chunk))
    y, lse, body = _forward(ops, monkeypatch, {}, x)
    assert body == PIPE, body
    assert not _check_fp64(y, lse, x, dim ** -0.5, dtype), _check_fp64(y, lse, x, dim ** -0.5, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize("dims", BUCKETS, ids=BUCKET_IDS)
def test_dead_rows_inside_live_ctas_on_the_segment_pair_route(monkeypatch, dtype, dims):
    ops = _ops()
    dim, value_dim = dims
    x = _pair_dead_rows_inputs(dtype, dim=dim, value_dim=value_dim)
    dead = ~x["mask"].any(-1)  # [B, Nq]
    assert bool(dead.any()) and bool((~dead).any())
    y, lse, body = _forward(ops, monkeypatch, {}, x)
    assert body == PIPE, body
    assert not _check_fp64(y, lse, x, dim ** -0.5, dtype), _check_fp64(y, lse, x, dim ** -0.5, dtype)
    y0, lse0, body0 = _forward(ops, monkeypatch, {"XATTN_SCA_FWD_PIPE": "0"}, x)
    assert body0 == GENERIC, body0
    assert not _check_fp64(y0, lse0, x, dim ** -0.5, dtype)


@pytest.mark.parametrize("previous", ["none", "continued"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize("dims", BUCKETS, ids=BUCKET_IDS)
def test_fewer_queries_than_keys_right_aligned(monkeypatch, dtype, previous, dims):
    """Unequal current lengths, Nq < Nk: Q is right-aligned to [previous; current] keys, so its first
    row sits at key position T - Nq, mid-chunk (333 - 200 = 133 with C = 128)."""
    ops = _ops()
    dim, value_dim = dims
    x = _bos_inputs(dtype, previous=0 if previous == "none" else 128, q_length=200, dim=dim,
                    value_dim=value_dim)
    y, lse, body = _forward(ops, monkeypatch, {}, x)
    assert body == PIPE, body
    assert not _check_fp64(y, lse, x, dim ** -0.5, dtype), _check_fp64(y, lse, x, dim ** -0.5, dtype)
    y0, lse0, body0 = _forward(ops, monkeypatch, {"XATTN_SCA_FWD_PIPE": "0"}, x)
    assert body0 == GENERIC, body0
    assert not _check_fp64(y0, lse0, x, dim ** -0.5, dtype)


def test_more_queries_than_keys_is_refused_by_the_public_api(monkeypatch):
    """Nq > Nk is outside the API (xattn/ops/sliding_chunk_attention.py: '0 < q length <= k/v
    length'): the default arm refuses it before the extension is called, so no body runs."""
    ops = _ops()
    torch.manual_seed(3)
    dtype, nq, nk, chunk = torch.bfloat16, 200, 130, 64
    q = F.normalize(torch.randn(2, nq, 4, 256, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(2, nk, 2, 256, device="cuda"), dim=-1).to(dtype)
    v = (F.silu(torch.randn(2, nk, 2, 256, device="cuda")) + 0.1).to(dtype)
    for env in ({}, {"XATTN_SCA_FWD_PIPE": "0"}):
        _set_env(monkeypatch, env)
        before = _counts(ops)
        with pytest.raises(ValueError, match="q length <= k/v length"):
            with torch.no_grad():
                xattn.flash_sca_fwd(q, k, v, chunk, None, None, None)
        torch.cuda.synchronize()
        assert _body_delta(before, _counts(ops)) == NOTHING


@pytest.mark.parametrize("spike", [False, True], ids=("rescale", "clear"))
@pytest.mark.parametrize("route", ["dense", "pair"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize("dims", BUCKETS, ids=BUCKET_IDS)
def test_late_max_rescale_and_clear_against_fp64(monkeypatch, dtype, route, spike, dims):
    ops = _ops()
    x = _late_max_inputs(dtype, route, spike, dim=dims[0], value_dim=dims[1])
    crossed, cleared = _rescale_profile(x, 1.0)
    # the inputs are directed as claimed, independently of the kernel
    assert crossed > 0 and (cleared > 0) == spike, (crossed, cleared)
    if route == "pair":
        dead = ~x["mask"].any(-1)
        assert bool(dead.any()) and bool((~dead).any())
    y, lse, body = _forward(ops, monkeypatch, {}, x, scale=1.0)
    assert body == PIPE, body
    assert not _check_fp64(y, lse, x, 1.0, dtype), _check_fp64(y, lse, x, 1.0, dtype)
    y0, lse0, body0 = _forward(ops, monkeypatch, {"XATTN_SCA_FWD_PIPE": "0"}, x, scale=1.0)
    assert body0 == GENERIC, body0
    assert not _check_fp64(y0, lse0, x, 1.0, dtype), _check_fp64(y0, lse0, x, 1.0, dtype)
