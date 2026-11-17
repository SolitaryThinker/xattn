# Author: Shicheng Wen

"""Follow-up cells: previous-K/V strides, the ``'cuda'`` route, scale
finiteness and the first-chunk visibility identity.

These are the cells an earlier suite deliberately left out, written to the same bar:
each one pins a property the FP64 oracle cannot observe, and no numeric
tolerance from the oracle or from the contract is relaxed or re-derived here.

* **Previous K/V strides.** ``prev_k``/``prev_v`` are contract-supported in
  head-slice, offset and last-dimension-strided views. The stride evidence in
  ``test_public_api_native.py`` covers ``q``/``k``/``v``/``dy`` only, and the
  previous tensors are exactly where a host-side canonicalisation bug hides:
  the result stays finite and plausible while every ``dPrevK``/``dPrevV`` is
  transposed. So a strided view must equal its contiguous copy **bit for bit**
  on the gradients, and in particular on ``dPrevK``/``dPrevV``.
* **The ``'cuda'`` route.** ``backend='cuda'`` names the generic CUDA
  (SM80 ``mma.sync``) kernels, not the native Blackwell ones:
  ``_sca_native.selected`` returns False for that name on any device, so the
  route must (a) leave the native launch counters untouched, (b) return the
  documented shapes and dtypes, (c) agree with the native route to within a
  loose absolute tolerance, and (d) still refuse a BOS mask with the documented
  error. Without this cell, ``'auto'``-only coverage would let the native port
  quietly become the only path, which is the one thing the public dispatch
  forbids. The generic route's chunk bounds differ from the native route's, so
  the cell states the ``chunk_size`` it uses and the size is chosen from the
  generic kernel's own guard, not from a guess.
* **Scale finiteness.** ``validate_attention_scale`` refuses a non-finite
  scale in all three public entries (``_attention_core.py:229``; call sites
  ``sliding_chunk_attention.py:2389``, ``:2460``, ``:2631``) with
  ``"attention scale must be finite"``. A NaN scale reaching the softmax would
  make a whole backward NaN with no diagnostic, so the refusal - and the fact
  that it happens **before** any kernel launches - is the property to pin.
  This cell never dispatches - the ``y``/``lse`` its backward entry needs are
  allocated instead of computed - so it carries its own guard and reports on a
  build without native SCA or native precision instead of skipping with the
  rest of the file.

Two message expectations here use ``pytest.raises(match=...)``, which the rest
of this directory does not: ``attention scale must be finite``
(``_attention_core.py:229``) and ``backend='auto'/'sm90'`` (the BOS refusal in
``_prepare_bos_metadata``). Pinning the message is deliberate - it is the
observed public behaviour - but it also means a message rewrite breaks exactly
these two cells and nothing else, which is the intended signal rather than an
accident.
* **First-chunk unit cell.** The contract's first sentence, held once
  independently of any tile geometry: a query in the first chunk sees only its
  own chunk, causally. With one visible key the softmax weight is exactly 1, so
  ``y`` must equal that key's ``v`` bitwise and ``dQ`` on that row must cancel
  to roundoff (``p * (dy.v - y.dy) = dy.v - v.dy = 0`` in exact arithmetic),
  never to an O(1) value.

The numerical contract itself stays the oracle suite; nothing here widens it.
"""

import pytest
import torch
import torch.nn.functional as F

import xattn
from xattn.ops import _sca_native


def _native_available():
    """cc 10.3 with native SCA *and* the native advanced-precision capability.

    Identical to the guard in ``test_public_api_native.py`` on purpose: the op
    names are the extension's, and a guard that guesses them (or a build that
    links native SCA but not native precision) must skip rather than claim a
    native run it did not make.
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


requires_native = pytest.mark.skipif(
    not _native_available(),
    reason="native FlashSCA with levels 0-4 and chunk visibility needs cc 10.3",
)


def _extension_present():
    """The refusal cells need the extension importable, not a native capability."""
    if not torch.cuda.is_available():
        return False
    try:
        import xattn_cuda  # noqa: F401
        return True
    except Exception:
        return False


# The scale-finiteness cell never dispatches, so it only needs CUDA and an
# importable extension: it must report on a build without the native SCA group
# instead of skipping (see its docstring for why a forward is not run).
needs_no_native = pytest.mark.skipif(
    not _extension_present(),
    reason="needs CUDA and the compiled extension; it never dispatches, so no native capability",
)

DTYPES = (torch.bfloat16, torch.float16)


def _inputs(dtype, batch=2, length=64, heads=4, kv_heads=2, dim=64, value_dim=64,
            previous=32):
    torch.manual_seed(0)
    total = length + previous
    q = F.normalize(torch.randn(batch, length, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, total, kv_heads, dim, device="cuda"), dim=-1).to(dtype)
    v = (torch.randn(batch, total, kv_heads, value_dim, device="cuda") * 0.1).to(dtype)
    dy = F.normalize(torch.randn(batch, length, heads, value_dim, device="cuda"), dim=-1).to(dtype)
    if previous:
        return q, k[:, previous:], v[:, previous:], k[:, :previous], v[:, :previous], dy
    return q, k, v, None, None, dy


def _run(q, k, v, pk, pv, dy, chunk=32, deterministic=True, **kwargs):
    """Forward at level 0, then a backward that repeats bit-exactly.

    ``deterministic`` belongs to the backward only: the forward has no reduction
    order to fix, so it is not passed to ``flash_sca_fwd``.
    """
    y, lse = xattn.flash_sca_fwd(q, k, v, chunk, None, pk, pv, **kwargs)
    grads = xattn.flash_sca_bwd(
        dy, q, k, v, y, lse, chunk, None, pk, pv, deterministic=deterministic,
        **kwargs,
    )
    return y, lse, grads


@requires_native
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_previous_kv_strided_views_are_bit_identical(dtype):
    """Gradients through a strided ``prev_k``/``prev_v`` equal the contiguous ones.

    ``prev_k`` is given as a last-dimension-strided view (stride 2, taken from a
    tensor of twice the head width) and ``prev_v`` as a head-slice of a tensor
    of twice the KV head count. Both are read-only views the contract lists as
    supported, and the previous tensors are where a host-side canonicalisation
    bug hides: the result stays finite and plausible while every
    ``dPrevK``/``dPrevV`` is transposed. **The claim of this cell is therefore
    the dPrevK/dPrevV half** - the gradients that flow back through the sliced
    previous tensors - asserted bit-exactly against the contiguous copies.

    ``y`` and ``lse`` bit-equality is carried as the **control line**: it
    corroborates the unmerged forward-side witness that already swept ``k``/
    ``v``/``prev_k``/``prev_v`` views against contiguous copies at levels 0-4
    with a previous chunk present (bit-equal, 0 red). It is not this cell's
    claim, and a red control line would still be reported as a failure.

    What bit-equality means here, stated so nobody reads more into it than it
    holds - and the two arms are *not* the same case. The host reads an
    operand directly when it satisfies ``Readable`` (``csrc/flash_sca/native/
    host.cc``: last-dimension stride 1, every other stride of a size>1 dim a
    multiple of 8, base 16-byte aligned) and otherwise canonicalises it with a
    copy. ``prev_k``'s stride-2 last dimension fails that rule, so ``Prepare``
    copies it and the kernels see dense bytes: the ``dPrevK`` half is evidence
    for the canonicalising copy path. ``prev_v``'s head slice - strides
    (8192, 256, 64, 1) on a 16-byte-aligned base - *passes* the rule and is
    read **in place**, with head stride 64, row stride 256 and batch stride
    8192 taken straight from its own stride fields: the ``dPrevV`` half is a
    genuine strided-read property of the kernel, not a copy artifact. The
    reference arms (``k[:, C:]``/``v[:, C:]``) are read in place too, so the
    comparison is between two live layouts and not between a layout and its
    packed image. The failure mode both halves name is a stride-blind read:
    same kernels, same shapes, finite and plausible output, wrong
    ``dPrevK``/``dPrevV``. ``deterministic`` is on for both arms because
    without it the dQ atomics would make the comparison meaningless. If a
    stride-aware loader for ``prev_k`` ever replaces its copy, a last-ulp
    difference becomes a bar question rather than automatically a red.
    """
    q, k, v, pk, pv, dy = _inputs(dtype)

    # A stride-2 last-dimension view that carries exactly pk's values: write
    # pk into the even lanes of a double-width buffer.
    pk_wide = torch.zeros(*pk.shape[:-1], 2 * pk.shape[-1], device="cuda", dtype=dtype)
    pk_wide[..., ::2] = pk
    pk_strided = pk_wide[..., ::2]
    assert pk_strided.shape == pk.shape and pk_strided.stride(-1) == 2
    assert torch.equal(pk_strided, pk)

    pv_wide = torch.cat([pv, pv + 1], dim=2)
    pv_head = pv_wide[:, :, : pv.shape[2], :]
    assert pv_head.shape == pv.shape and not pv_head.is_contiguous()
    assert torch.equal(pv_head, pv)

    y_ref, lse_ref, grads_ref = _run(q, k, v, pk, pv, dy, deterministic=True)
    y_str, lse_str, grads_str = _run(
        q, k, v, pk_strided, pv_head, dy, deterministic=True
    )

    # The claim: every gradient, and in particular the two that flow back
    # through the sliced previous tensors, is bit-identical.
    for name, got, want in zip(
        ("dq", "dk", "dv", "dprev_k", "dprev_v"), grads_str, grads_ref
    ):
        assert want is not None and want.dtype == dtype, name
        assert got.shape == want.shape, name
        assert torch.equal(got, want), (
            f"{name} differs between the contiguous and the strided previous K/V; "
            f"max abs diff {float((got.float() - want.float()).abs().max())}"
        )
    # dprev_k/dprev_v are the operands that were sliced, so their bit-equality
    # is the point rather than a by-product: state it again on its own.
    for name, got, want in zip(("dprev_k", "dprev_v"), grads_str[3:], grads_ref[3:]):
        assert torch.equal(got, want), (
            f"{name} is not bit-identical under a sliced previous tensor"
        )

    # The control line: the forward and the log-sum-exp agree too (corroborates
    # the unmerged forward-side witness; a red line here is still a failure).
    assert torch.equal(y_str, y_ref), (
        "control line: y differs between the contiguous and the strided previous K/V; "
        f"max abs diff {float((y_str.float() - y_ref.float()).abs().max())}"
    )
    assert torch.equal(lse_str, lse_ref), (
        "control line: lse differs under a sliced previous tensor"
    )


@requires_native
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("previous", [0, 64], ids=["no-prev", "prev"])
def test_cuda_backend_is_the_generic_route_and_agrees_with_native(dtype, previous):
    """``backend='cuda'`` runs the generic kernels and must not be the native path.

    Dispatch is asserted by the native launch counter, not by a symbol list: a
    ``'cuda'`` call must leave ``(forward, backward)`` exactly where it was.
    The agreement bar is **``4 * eps`` absolute** in the output dtype (BF16
    3.1e-2, FP16 3.9e-3) for ``y``, and 3e-3 absolute for the FP32 ``lse`` -
    the log of a ``1 + 3e-3`` probability ratio. The ``y`` bar is deliberately
    **loose**: for outputs of order 0.1 it is many ulps of a typical element,
    because two different kernels accumulating the same product in FP32 are not
    promised to agree to a few ulps and this cell is a route smoke comparison,
    not a precision claim. It errs loose so that it cannot red a correct route
    on rounding; it would still catch a route running on the wrong probability
    normalisation or the wrong mask, which is what the cell is for. The ``lse``
    bar is tight on purpose: both routes compute one FP32 log-sum-exp. Both are
    basis-stated, not tuned: a landing outside them is a finding to report, not
    a bar to widen.

    ``chunk_size=64`` rather than the 32 the other cells use: the generic
    backward guards ``chunk_size`` against its own ``DKV_BN`` and refuses 32
    with ``FlashSCA bwd unsupported chunk_size=32 for DKV_BN=64``. That is the
    generic route's bound, not the contract's; this cell uses a shape both
    routes accept so the comparison is of the routes and not of their bounds.
    """
    q, k, v, pk, pv, dy = _inputs(dtype, length=128, previous=previous)
    chunk = 64
    eps = torch.finfo(dtype).eps

    before = _sca_native.launch_counts()
    y_cuda, lse_cuda = xattn.flash_sca_fwd(
        q, k, v, chunk, None, pk, pv, backend="cuda"
    )
    grads_cuda = xattn.flash_sca_bwd(
        dy, q, k, v, y_cuda, lse_cuda, chunk, None, pk, pv, backend="cuda"
    )
    after = _sca_native.launch_counts()
    assert after == before, f"backend='cuda' launched a native kernel: {before} -> {after}"
    assert y_cuda.dtype == dtype and lse_cuda.dtype == torch.float32

    y_native, lse_native = xattn.flash_sca_fwd(
        q, k, v, chunk, None, pk, pv, backend="auto"
    )
    assert _sca_native.launch_counts()[0] == before[0] + 1, (
        "backend='auto' did not launch a native forward, so the comparison is void"
    )

    assert y_cuda.shape == q.shape[:3] + (v.shape[-1],) and y_cuda.dtype == dtype
    assert lse_cuda.shape == (q.shape[0], q.shape[2], q.shape[1]), lse_cuda.shape
    assert torch.isfinite(y_cuda).all() and torch.isfinite(lse_cuda).all()
    assert torch.allclose(y_cuda.float(), y_native.float(), atol=4 * eps, rtol=0), (
        "generic route disagrees with the native route beyond the 4*eps absolute "
        f"bar: max abs diff {float((y_cuda.float() - y_native.float()).abs().max())}, "
        f"4*eps = {4 * eps}"
    )
    assert torch.allclose(lse_cuda, lse_native, atol=3e-3, rtol=0), (
        f"lse disagrees beyond 3e-3: max abs diff {float((lse_cuda - lse_native).abs().max())}"
    )

    names = ("dq", "dk", "dv", "dprev_k", "dprev_v")
    for name, grad in zip(names, grads_cuda):
        if previous == 0 and name in ("dprev_k", "dprev_v"):
            # Without previous K/V the contract returns None, not a zero tensor.
            assert grad is None, f"{name} must be None without previous K/V"
            continue
        assert grad is not None, name
        assert grad.dtype == dtype, name
        assert torch.isfinite(grad).all(), name
    # Documented shapes of the grads that exist. ``dq`` keeps **Q's** shape -
    # ``host.cc`` allocates it at ``qk_dim``, i.e. [B, N, Hq, D], not Dv - so
    # asserting against ``q.shape`` is exact even where D != Dv; asserting Dv
    # (or ``v.shape[-1]``) there would false-red a correct D != Dv call. dk/dv
    # keep the **current** K/V shapes (allocated at ``k_length``), and only
    # dprev_* carry the previous length.
    assert grads_cuda[0].shape == q.shape, grads_cuda[0].shape
    assert grads_cuda[1].shape == k.shape, grads_cuda[1].shape
    assert grads_cuda[2].shape == v.shape, grads_cuda[2].shape
    if previous:
        assert grads_cuda[3].shape == pk.shape, grads_cuda[3].shape
        assert grads_cuda[4].shape == pv.shape, grads_cuda[4].shape

    # A BOS mask has no generic route: the public dispatch must say so. The
    # mask carries the **full key axis** (previous rows + current, per
    # ``_prepare_bos_metadata``: ``k_length = k.shape[1] + prev_k.shape[1]``),
    # so the refusal that fires is the backend refusal and not the shape check
    # that follows it - with an N-length mask the cell would pass for the wrong
    # reason and would show nothing about a contract-valid mask.
    prev_rows = 0 if pk is None else pk.shape[1]
    bos = torch.zeros(
        q.shape[0], prev_rows + k.shape[1], dtype=torch.bool, device="cuda"
    )
    bos[:, 0] = True
    with pytest.raises(ValueError, match="backend='auto'/'sm90'"):
        xattn.flash_sca_fwd(q, k, v, chunk, None, pk, pv, backend="cuda", bos_mask=bos)


@needs_no_native
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_non_finite_scale_is_refused_before_any_launch(dtype, bad_scale):
    """A non-finite scale must be refused by all three public entries, unlaunched.

    ``validate_attention_scale`` is called at the top of ``flash_sca``,
    ``flash_sca_fwd`` and ``flash_sca_bwd``; a NaN scale reaching the softmax
    would poison an entire backward with no diagnostic, so the refusal and the
    unchanged native launch counter are both pinned. What is unique here is
    NaN, the two standalone entries, and the unchanged counter: the finite
    nonpositive half lives in ``test_nonpositive_scale_is_the_normalised_positive_scale``
    in ``test_public_api_native.py``, which an earlier change merged, and the +inf refusal at
    the autograd entry is already asserted there (``test_public_api_native.py:370-371``).

    The cell **never dispatches** - not even to build the ``y``/``lse`` the
    backward entry needs, which are allocated here instead - so it holds on any
    build with an importable extension and does not depend on the native SCA
    group being linked (``_sca_native.selected`` raises when it is not). Its
    guard says exactly that and nothing more.
    """
    q, k, v, pk, pv, dy = _inputs(dtype)
    # Placeholders with the documented shapes and dtypes: the scale is validated
    # before the saved state is looked at, so a real forward is not needed - and
    # not running one is the point (see the docstring).
    y = torch.zeros(q.shape[:3] + (v.shape[-1],), device="cuda", dtype=dtype)
    lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], device="cuda", dtype=torch.float32)
    before = _sca_native.launch_counts()

    with pytest.raises(ValueError, match="attention scale must be finite"):
        xattn.flash_sca_fwd(q, k, v, 32, bad_scale, pk, pv)
    with pytest.raises(ValueError, match="attention scale must be finite"):
        xattn.flash_sca_bwd(dy, q, k, v, y, lse, 32, bad_scale, pk, pv)
    with pytest.raises(ValueError, match="attention scale must be finite"):
        xattn.flash_sca(q, k, v, 32, bad_scale, pk, pv)

    assert _sca_native.launch_counts() == before, "a refused scale still launched a kernel"


@requires_native
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_first_chunk_query_sees_only_its_own_chunk(dtype):
    """One visible key: ``y`` is that key's ``v`` bitwise and that row's ``dQ`` cancels.

    ``q`` sees ``j == p`` only when it is the first key of the first chunk, so
    with ``chunk=2``, ``N=K=3`` the first query has exactly one visible key and
    the softmax weight is exactly 1.0 in any rounding mode. ``dQ`` on that row
    is ``p * (dy.v - y.dy) * k``; with ``y == v`` the bracket is identically
    zero in exact arithmetic, so the assertion is that it lands at **output
    roundoff** - at most 4 ulps of the largest gradient in the tensor - and not
    at any O(1) value. (FP16 does not round that residual to zero: it lands at
    2**-24, i.e. below the dtype's smallest normal. BF16 does round to zero,
    which the same bar accepts.)
    """
    batch, heads, dim = 1, 2, 64
    torch.manual_seed(0)
    q = F.normalize(torch.randn(batch, 3, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, 3, heads, dim, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(batch, 3, heads, dim, device="cuda").to(dtype)
    dy = torch.randn(batch, 3, heads, dim, device="cuda").to(dtype)
    scale = 1.0 / dim ** 0.5

    y, lse = xattn.flash_sca_fwd(q, k, v, 2, None, None, None)
    grads = xattn.flash_sca_bwd(dy, q, k, v, y, lse, 2, None, None, None,
                                deterministic=True)

    assert torch.equal(y[:, 0], v[:, 0]), (
        "the first query of the first chunk must reproduce v[0] bitwise; max abs "
        f"diff {float((y[:, 0].float() - v[:, 0].float()).abs().max())}"
    )
    assert grads[3] is None and grads[4] is None

    lse_ref = (
        q[:, 0].double() * k[:, 0].double()
    ).sum(dim=-1) * scale
    assert torch.allclose(lse[:, :, 0].double(), lse_ref, atol=1e-5, rtol=0), (
        f"lse of the single-visible-key row is not the single score: "
        f"{lse[:, :, 0]} vs {lse_ref}"
    )

    dq = grads[0]
    row = dq[:, 0].float().abs().max()
    bar = 4 * torch.finfo(dtype).eps * dq.float().abs().max()
    assert row <= bar, (
        f"dQ on a row with exactly one visible key must cancel to roundoff; got "
        f"{float(row)} against a bar of 4 ulps of the largest gradient ({float(bar)})"
    )
