# Author: Shicheng Wen

"""Structural contract of the public FlashSCA API on cc 10.3.

The FP64 oracle pins the numbers. This pins what those numbers are *returned
in*, which an oracle comparison cannot see: which tensors exist at all, their
shapes and dtypes, the ``None`` cases, the empty-row convention, the GQA head
mapping at API level, and the refusals that keep the later 7.1B stage out of
this API.
"""

import pytest
import torch
import torch.nn.functional as F

import xattn


def _native_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        import xattn_cuda

        return bool(xattn_cuda.ops._flash_sca_native_available(torch.zeros(1, device="cuda")))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="native FlashSCA runs on a cc 10.3 device with a build that links it",
)

DTYPES = (torch.bfloat16, torch.float16)
LEVELS = (0, 1)


def _inputs(length=131, chunk=64, heads=8, kv_heads=2, dim=128, value_dim=64,
            previous=0, batch=2, dtype=torch.bfloat16):
    torch.manual_seed(0)
    total = previous + length
    q = F.normalize(torch.randn(batch, length, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, total, kv_heads, dim, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(batch, total, kv_heads, value_dim, device="cuda").to(dtype)
    dy = torch.ones(batch, length, heads, value_dim, device="cuda").to(dtype)
    prev_k = k[:, :previous].clone() if previous else None
    prev_v = v[:, :previous].clone() if previous else None
    return q, k[:, previous:], v[:, previous:], prev_k, prev_v, dy


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_prev_kv_round_trip(level, dtype):
    """dPrevK/dPrevV exist exactly when previous K/V were given."""
    q, k, v, pk, pv, dy = _inputs(previous=64, dtype=dtype)
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv,
                                 high_precision_level=level)
    grads = xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, pk, pv,
                                deterministic=True, high_precision_level=level)
    dq, dk, dv, dprev_k, dprev_v = grads
    for name, got, want in (("dq", dq, q), ("dk", dk, k), ("dv", dv, v),
                            ("dprev_k", dprev_k, pk), ("dprev_v", dprev_v, pv)):
        assert got is not None, name
        assert tuple(got.shape) == tuple(want.shape), name
        assert got.dtype == want.dtype, name
        assert torch.isfinite(got).all(), name
    without = xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, None, None,
                                  deterministic=True, high_precision_level=level)
    assert without[3] is None and without[4] is None, (
        "without previous K/V the backward must return None for both")


@pytest.mark.parametrize("level", LEVELS)
def test_gqa_unequal_lengths_shape_and_dtype(level):
    q, k, v, pk, pv, dy = _inputs(length=100, chunk=64, heads=8, kv_heads=2,
                                  dim=128, value_dim=64, previous=64)
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv,
                                 high_precision_level=level)
    assert tuple(y.shape) == (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
    assert tuple(lse.shape) == (q.shape[0], q.shape[2], q.shape[1])
    assert lse.dtype == torch.float32 and y.dtype == q.dtype
    assert torch.isfinite(y).all() and torch.isfinite(lse).all()
    dq, dk, dv, dprev_k, dprev_v = xattn.flash_sca_bwd(
        dy, q, k, v, y, lse, 64, None, pk, pv,
        deterministic=True, high_precision_level=level)
    assert tuple(dq.shape) == tuple(q.shape) and dq.dtype == q.dtype
    assert tuple(dk.shape) == tuple(k.shape) and dk.dtype == k.dtype
    assert tuple(dv.shape) == tuple(v.shape) and dv.dtype == v.dtype
    assert tuple(dprev_k.shape) == tuple(pk.shape)
    assert tuple(dprev_v.shape) == tuple(pv.shape)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rows_with_no_visible_key_are_zero_and_negative_infinite(level, dtype):
    """Disjoint query and key segments: zero output, -inf lse, zero gradient."""
    q, k, v, _, _, dy = _inputs(previous=0, dtype=dtype)
    q_segments = torch.ones(q.shape[0], q.shape[1], dtype=torch.long, device="cuda")
    k_segments = torch.zeros(k.shape[0], k.shape[1], dtype=torch.long, device="cuda")
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, None, None,
                                 q_segment_idx=q_segments,
                                 k_segment_idx=k_segments,
                                 high_precision_level=level)
    assert torch.equal(y, torch.zeros_like(y)), "an empty row must give zero output"
    assert torch.isneginf(lse).all(), "an empty row must give lse = -inf, never NaN"
    assert not torch.isnan(y).any() and not torch.isnan(lse).any()
    dq, dk, dv, dprev_k, dprev_v = xattn.flash_sca_bwd(
        dy, q, k, v, y, lse, 64, None, None, None,
        deterministic=True, high_precision_level=level,
        q_segment_idx=q_segments, k_segment_idx=k_segments)
    for name, got, want in (("dq", dq, q), ("dk", dk, k), ("dv", dv, v)):
        assert torch.equal(got, torch.zeros_like(want)), (
            f"{name} must be exactly zero when nothing is visible")
        assert not torch.isnan(got).any(), name
    assert dprev_k is None and dprev_v is None


@pytest.mark.parametrize("level", LEVELS)
def test_metadata_shapes_and_exclusivity(level):
    q, k, v, pk, pv, _ = _inputs(previous=64)
    total = pk.shape[1] + k.shape[1]
    one = torch.ones(q.shape[0], q.shape[1], dtype=torch.long, device="cuda")
    with pytest.raises(ValueError):
        xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv, q_segment_idx=one,
                            high_precision_level=level)
    with pytest.raises(ValueError):
        xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv, k_segment_idx=one,
                            high_precision_level=level)
    with pytest.raises(ValueError):
        # Metadata covering keys must include the previous chunk.
        xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv,
                            segment_idx=torch.zeros(q.shape[0], k.shape[1],
                                                    dtype=torch.long, device="cuda"),
                            high_precision_level=level)
    with pytest.raises(ValueError):
        # A bos_mask must also cover the previous chunk's keys.
        xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv,
                            bos_mask=torch.zeros(q.shape[0], k.shape[1], dtype=torch.bool,
                                                 device="cuda"),
                            high_precision_level=level)


@pytest.mark.parametrize("level", LEVELS)
def test_broadcast_views_are_accepted(level):
    """A batch-broadcast (stride 0) input gives the single-row result."""
    q, k, v, pk, pv, _ = _inputs(length=131, previous=64)
    solo_q, solo_k, solo_v, solo_pk, solo_pv = q[:1], k[:1], v[:1], pk[:1], pv[:1]
    y_ref, lse_ref = xattn.flash_sca_fwd(solo_q, solo_k, solo_v, 64, None,
                                         solo_pk, solo_pv, high_precision_level=level)
    broadcast = tuple(t.expand(q.shape[0], *t.shape[1:])
                      for t in (solo_q, solo_k, solo_v, solo_pk, solo_pv))
    for name, tensor in zip(("q", "k", "v", "prev_k", "prev_v"), broadcast):
        assert tensor.stride(0) == 0, name
    bq, bk, bv, bpk, bpv = broadcast
    y, lse = xattn.flash_sca_fwd(bq, bk, bv, 64, None, bpk, bpv,
                                   high_precision_level=level)
    assert torch.equal(y, y_ref.expand_as(y)), "batch broadcast changed the forward"
    assert torch.equal(lse, lse_ref.expand_as(lse))


@pytest.mark.parametrize("level", LEVELS)
def test_row_offset_views_are_accepted(level):
    """Views that start inside a larger buffer run in place, bit for bit."""
    q, k, v, pk, pv, _ = _inputs(previous=64)

    def offset(tensor, pad=7):
        wide = torch.zeros(tensor.shape[0], tensor.shape[1] + pad, *tensor.shape[2:],
                           device="cuda", dtype=tensor.dtype)
        wide[:, pad:] = tensor
        return wide[:, pad:]

    views = tuple(offset(t) for t in (q, k, v, pk, pv))
    for name, view in zip(("q", "k", "v", "prev_k", "prev_v"), views):
        assert view.storage_offset() != 0, name
    vq, vk, vv, vpk, vpv = views
    y, lse = xattn.flash_sca_fwd(vq, vk, vv, 64, None, vpk, vpv,
                                 high_precision_level=level)
    y_ref, lse_ref = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv,
                                        high_precision_level=level)
    assert torch.equal(y, y_ref) and torch.equal(lse, lse_ref), (
        "an offset view changed the forward")
    dy = torch.ones_like(y)
    grads = xattn.flash_sca_bwd(dy, vq, vk, vv, y, lse, 64, None, vpk, vpv,
                                deterministic=True, high_precision_level=level)
    grads_ref = xattn.flash_sca_bwd(dy, q, k, v, y_ref, lse_ref, 64, None, pk, pv,
                                    deterministic=True, high_precision_level=level)
    for index, (got, want) in enumerate(zip(grads, grads_ref)):
        assert (got is None) == (want is None), index
        if got is not None:
            assert torch.equal(got, want), f"gradient {index} changed under an offset view"


@pytest.mark.parametrize("level", LEVELS)
def test_backend_strings_resolve_to_the_same_native_kernels(level):
    """'' , 'auto' and 'native' are one implementation on cc 10.3."""
    q, k, v, pk, pv, _ = _inputs(previous=64)
    reference = None
    for backend in ("", "auto", "native"):
        y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv, backend=backend,
                                     high_precision_level=level)
        if reference is None:
            reference = (y, lse)
        else:
            assert torch.equal(y, reference[0]) and torch.equal(lse, reference[1]), (
                f"backend {backend!r} does not run the same kernels")


def test_traced_widths_work_and_7_1b_width_is_refused():
    """D=Dv=256 (0.9B width) runs; Dv=2048 (7.1B) stays outside this API."""
    q, k, v, _, _, _ = _inputs(length=70, chunk=64, heads=4, kv_heads=1,
                               dim=256, value_dim=256, dtype=torch.bfloat16)
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, None, None)
    assert tuple(y.shape) == (q.shape[0], q.shape[1], q.shape[2], 256)
    assert torch.isfinite(y).all()
    wide = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 2048, device="cuda",
                       dtype=q.dtype)
    with pytest.raises(ValueError) as error:
        xattn.flash_sca(q, k, wide, 64)
    assert "256" in str(error.value)
