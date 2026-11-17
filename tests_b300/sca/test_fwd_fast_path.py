"""The forward's repeat-call fast path: a metadata key built and checked by the
extension inside the one pybind crossing, cached tensor maps, one metadata allocation.

A repeat of a BOS-mask level-0 call skips the Python checks whose answer its metadata key
already fixed, and the extension reuses a tensor map whose complete key (device, pointer,
strides, dtype, extents) it has encoded before. These tests pin that:
  * a repeat returns the same bits as the full path (keys cleared);
  * every changed argument the checks read - shape, dtype, chunk, scale (value and type), mask
    route and contiguity, previous K/V, the reset flag, the backend and attn_method strings, a
    tensor subclass - misses the extension's key; it is validated afresh (an invalid variant
    raises the full path's error even right after a valid call is cached) and returns the full
    path's bits; a new pointer or Q stride is the same key and still gives the full path's bits;
  * the route records and launch counters move on a fast-path call as on the full path.
"""
import pytest
import torch

from xattn.ops import _sca_native
from xattn.ops import _native_dispatch
from xattn.ops.sliding_chunk_attention import flash_sca_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 (sm_103) only")


def _inputs(B=2, N=1024, Hq=8, Hkv=2, D=64, Dv=64, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, N, Hq, D, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(B, N, Hkv, D, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(B, N, Hkv, Dv, device="cuda", dtype=dtype, generator=g)
    bos = torch.zeros(B, N, dtype=torch.bool, device="cuda")
    bos[:, 0] = True
    bos[:, N // 3] = True
    bos[0, N // 2 + 5] = True
    return q, k, v, bos


def _full(*args, **kw):
    _sca_native.fast_keys_clear()
    y, lse = flash_sca_fwd(*args, **kw)
    _sca_native.fast_keys_clear()  # the full call above re-learns its key: forget it
    return y, lse


def _fast(*args, **kw):
    flash_sca_fwd(*args, **kw)  # learns the key (or hits)
    return flash_sca_fwd(*args, **kw)


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _hit(args, kw):
    """Whether the extension's key test admits the public call (args, kw) as a repeat."""
    a = list(args) + [None] * (7 - len(args))
    ops = _sca_native._fast_ops()
    return ops._flash_sca_native_fwd_fast_hit(
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], None, None, kw.get("segment_idx"),
        kw.get("bos_mask"), kw.get("backend", "auto"), kw.get("reset_chunk_pos_per_seq", False),
        kw.get("attn_method", "default"), kw.get("high_precision_level", 0))


def test_repeat_is_bit_identical_and_hits():
    q, k, v, bos = _inputs()
    ref = _full(q, k, v, 256, None, bos_mask=bos)
    flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)
    assert _sca_native.fast_keys_size() == 1
    assert _hit([q, k, v, 256, None], dict(bos_mask=bos))
    fwd0 = _sca_native.launch_counts()[0]
    out = flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)
    assert _sca_native.launch_counts()[0] == fwd0 + 1
    assert _same(out, ref)
    assert _sca_native.fast_keys_size() == 1


class _Sub(torch.Tensor):
    pass


@pytest.mark.parametrize("variant", ["shape", "stride", "dtype", "pointer", "chunk", "scale",
                                     "scale_type", "route", "prev", "mask_stride", "reset",
                                     "backend", "attn_method", "subclass"])
def test_changed_argument_misses(variant):
    q, k, v, bos = _inputs()
    _sca_native.fast_keys_clear()
    flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)  # learn the base call's key
    assert _hit([q, k, v, 256, None], dict(bos_mask=bos))
    kw = dict(bos_mask=bos)
    args = [q, k, v, 256, None]
    if variant == "shape":
        q2, k2, v2, bos2 = _inputs(N=768)
        args = [q2, k2, v2, 256, None]
        kw = dict(bos_mask=bos2)
    elif variant == "stride":
        q2 = torch.empty(q.shape[0], q.shape[1], q.shape[2], q.shape[3] * 2, device="cuda",
                         dtype=q.dtype)[..., : q.shape[3]]
        q2.copy_(q)
        args[0] = q2
    elif variant == "dtype":
        args = [t.half() for t in (q, k, v)] + [256, None]
    elif variant == "pointer":
        args[:3] = [t.clone() for t in (q, k, v)]
    elif variant == "chunk":
        args[3] = 128
    elif variant == "scale":
        args[4] = 0.05
    elif variant == "scale_type":
        args[4] = 1  # an int scale is its own key, even where a float of the same value is known
    elif variant == "mask_stride":
        wide = torch.zeros(bos.shape[0], bos.shape[1] * 2, dtype=torch.bool, device="cuda")
        wide[:, ::2] = bos
        kw = dict(bos_mask=wide[:, ::2])  # same values, not contiguous: a different route
    elif variant == "route":
        kw = dict(segment_idx=bos.long().cumsum(1))
    elif variant == "prev":
        pk = torch.randn(q.shape[0], 256, k.shape[2], k.shape[3], device="cuda", dtype=q.dtype)
        pv = torch.randn(q.shape[0], 256, v.shape[2], v.shape[3], device="cuda", dtype=q.dtype)
        args += [pk, pv]
        kw = dict(bos_mask=torch.cat([bos[:, :256], bos], dim=1))
    elif variant == "reset":
        kw = dict(bos_mask=bos, reset_chunk_pos_per_seq=True)
    elif variant == "backend":
        kw = dict(bos_mask=bos, backend="native")
    elif variant == "attn_method":
        kw = dict(bos_mask=bos, attn_method="default ")
    elif variant == "subclass":
        args[0] = q.as_subclass(_Sub)
    same_key = variant in ("pointer", "stride")
    # Pointers and Q/K/V strides are read by no Python check, so they are the same metadata key;
    # the extension checks layouts every call and its map cache keys pointer + strides.
    assert _hit(args, kw) == same_key
    try:
        ref = _full(*args, **kw)
    except Exception as e:  # noqa: BLE001 - a refused variant must refuse on both paths
        flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)  # learn the base call again
        with pytest.raises(type(e)) as info:
            flash_sca_fwd(*args, **kw)
        assert str(info.value) == str(e)
    else:
        out = _fast(*args, **kw)
        assert _same(out, ref)
    # and the base call, cached before, still gives its own bits
    assert _same(flash_sca_fwd(q, k, v, 256, None, bos_mask=bos),
                 _full(q, k, v, 256, None, bos_mask=bos))


@pytest.mark.parametrize("bad", ["chunk0", "dims", "nan_scale", "neg_scale", "mask_len"])
def test_invalid_variant_after_cached_call_raises_as_full_path(bad):
    q, k, v, bos = _inputs()
    flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)
    flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)  # base key cached
    args = [q, k, v, 256, None]
    kw = dict(bos_mask=bos)
    if bad == "chunk0":
        args[3] = 0
    elif bad == "dims":
        args[0] = torch.randn(q.shape[0], q.shape[1], q.shape[2], 12, device="cuda", dtype=q.dtype)
        args[1] = torch.randn(k.shape[0], k.shape[1], k.shape[2], 12, device="cuda", dtype=q.dtype)
    elif bad == "nan_scale":
        args[4] = float("nan")
    elif bad == "neg_scale":
        args[4] = -0.1  # not an error: the non-positive-scale route, never the cached one
    elif bad == "mask_len":
        kw = dict(bos_mask=bos[:, :-1].contiguous())

    def run():
        try:
            return ("ok", flash_sca_fwd(*args, **kw))
        except Exception as e:  # noqa: BLE001 - the error itself is what is compared
            return ("err", (type(e), str(e)))

    cached = run()
    cached_again = run()  # the variant's own key, if it learned one
    _sca_native.fast_keys_clear()
    fresh = run()
    for got in (cached, cached_again):
        assert got[0] == fresh[0]
        if got[0] == "err":
            assert got[1] == fresh[1]
        else:
            assert _same(got[1], fresh[1])


def test_keys_are_bounded():
    q, k, v, bos = _inputs()
    _sca_native.fast_keys_clear()
    for i in range(70):
        flash_sca_fwd(q, k, v, 256, 0.01 * (i + 1), bos_mask=bos)
        assert 1 <= _sca_native.fast_keys_size() <= 64


def test_route_records_match_full_path():
    q, k, v, bos = _inputs()
    old = _native_dispatch._TRACE_ENABLED
    _native_dispatch.set_route_trace(True)
    try:
        _sca_native.fast_keys_clear()
        _native_dispatch.clear_route_trace()
        flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)
        full = [dict(r) for r in _native_dispatch.route_trace()]
        assert _sca_native.fast_keys_size() == 1  # the full call learned its key
        _native_dispatch.clear_route_trace()
        fwd0 = _sca_native.launch_counts()[0]
        flash_sca_fwd(q, k, v, 256, None, bos_mask=bos)
        fast = [dict(r) for r in _native_dispatch.route_trace()]
        assert _sca_native.launch_counts()[0] == fwd0 + 1
        assert full and full == fast
    finally:
        _native_dispatch.set_route_trace(old)


def test_public_head_hit_skips_routed_entry_and_matches_full_path(monkeypatch):
    # With tracing off, a hit returns from the public frame: the routed entry point is not run.
    from xattn.ops import sliding_chunk_attention as sca
    q, k, v, bos = _inputs()
    old = _native_dispatch._TRACE_ENABLED
    _native_dispatch.set_route_trace(False)
    try:
        ref = _full(q, k, v, 256, 0.125, bos_mask=bos)
        flash_sca_fwd(q, k, v, 256, 0.125, bos_mask=bos)  # learns the key
        routed = sca._flash_sca_fwd_routed
        calls = []

        def spy(*a, **kw):
            calls.append(1)
            return routed(*a, **kw)

        monkeypatch.setattr(sca, "_flash_sca_fwd_routed", spy)
        fwd0 = _sca_native.launch_counts()[0]
        y, lse = flash_sca_fwd(q, k, v, 256, 0.125, bos_mask=bos)
        assert calls == [] and _sca_native.launch_counts()[0] == fwd0 + 1
        assert torch.equal(y, ref[0]) and torch.equal(lse, ref[1])
        y2, _ = flash_sca_fwd(q, k, v, 512, 0.125, bos_mask=bos)  # a miss runs the routed path
        assert calls == [1]
        assert torch.equal(y2, _full(q, k, v, 512, 0.125, bos_mask=bos)[0])
    finally:
        _native_dispatch.set_route_trace(old)
