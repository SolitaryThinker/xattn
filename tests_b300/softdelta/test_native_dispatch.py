# Author: Shicheng Wen

"""Truthful backend-selection rules for the native SoftDelta route.

These tests are pure host logic: the device probe and the extension are
patched, so nothing here claims that a native kernel exists, runs, or is
numerically correct. They pin the rules an operator depends on:

* a backend name never implies support;
* every route that cannot serve a call raises instead of degrading;
* ``auto`` on a Blackwell device raises rather than silently running the
  torch reference path, so a passing result can never be mistaken for native;
* ``sm90`` is never an alias for native execution on a non-SM90 device.
"""

import contextlib

import pytest
import torch

from xattn.ops import _native_dispatch as nd

SM90 = (9, 0)
BLACKWELL = (10, 3)
AMPERE = (8, 0)
CPU_TENSOR = torch.zeros(1)


class _Ops:
    def __init__(self, mask, *, bindings=True):
        self._mask = mask
        if bindings:
            self._native_attention_fwd = lambda *a, **k: None
            self._native_attention_bwd = lambda *a, **k: None

    def _native_attention_level_mask(self, q):
        return self._mask


@contextlib.contextmanager
def env(capability=BLACKWELL, mask=0b00011, *, bindings=True):
    saved = (
        nd.device_capability,
        nd.extension_ops,
        nd.native_reader_level_mask,
    )
    nd.device_capability = lambda q: capability
    nd.extension_ops = lambda: _Ops(mask, bindings=bindings)
    nd.native_reader_level_mask = lambda q: mask
    try:
        yield
    finally:
        (
            nd.device_capability,
            nd.extension_ops,
            nd.native_reader_level_mask,
        ) = saved


def select(backend, q=CPU_TENSOR, **kwargs):
    return nd.select_backend(backend, q, **kwargs)


# --- names ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("", "auto"),
        ("auto", "auto"),
        ("torch", "torch"),
        ("sm90", "sm90"),
        ("native", "native"),
        ("  Native ", "native"),
    ],
)
def test_normalize_backend_accepts_known_names(name, expected):
    assert nd.normalize_backend(name) == expected


@pytest.mark.parametrize("name", ["sm100", "cuda", "blackwell", "sm_90", None, 3])
def test_normalize_backend_rejects_unknown_names(name):
    with pytest.raises((ValueError, TypeError)):
        nd.normalize_backend(name)


# --- the probe -----------------------------------------------------------


def test_level_mask_is_zero_without_a_cuda_tensor():
    assert nd.native_reader_level_mask(CPU_TENSOR) == 0


def test_level_mask_is_zero_without_the_extension():
    saved = nd.extension_ops
    nd.extension_ops = lambda: None
    try:
        assert nd.native_reader_level_mask(CPU_TENSOR) == 0
    finally:
        nd.extension_ops = saved


def test_level_mask_is_zero_when_the_probe_is_absent():
    saved = nd.extension_ops
    nd.extension_ops = lambda: object()
    try:
        assert nd.native_reader_level_mask(CPU_TENSOR) == 0
    finally:
        nd.extension_ops = saved


def test_level_mask_treats_a_raising_probe_as_no_native_support():
    saved = nd.extension_ops

    class _Broken:
        def _native_attention_level_mask(self, q):
            raise RuntimeError("device query failed")

    nd.extension_ops = lambda: _Broken()
    try:
        assert nd.native_reader_level_mask(torch.zeros(1, device="cpu")) == 0
    finally:
        nd.extension_ops = saved


def test_level_mask_masks_bits_outside_the_declared_levels():
    saved = nd.extension_ops
    nd.extension_ops = lambda: _Ops(0b11111111)
    try:
        # No CUDA tensor here, so the device guard still answers 0; the bit
        # masking itself is checked on the arithmetic below.
        assert nd.native_reader_level_mask(CPU_TENSOR) == 0
        assert 0b11111111 & nd._NATIVE_LEVEL_MASK == 0b11111
    finally:
        nd.extension_ops = saved


# --- selection -----------------------------------------------------------


def test_torch_is_always_the_reference_path():
    for capability in (AMPERE, SM90, BLACKWELL):
        with env(capability):
            assert select("torch") == "torch"


def test_auto_routes_sm90_to_the_sm90_backend():
    with env(SM90):
        assert select("auto") == "sm90"


def test_auto_routes_pre_blackwell_cuda_to_torch():
    with env(AMPERE, mask=0):
        assert select("auto") == "torch"


def test_auto_on_blackwell_never_degrades_to_torch():
    with env(BLACKWELL, mask=0):
        with pytest.raises(RuntimeError, match="native"):
            select("auto")


def test_auto_on_blackwell_selects_native_when_the_level_is_linked():
    with env(BLACKWELL, mask=0b00011):
        assert select("auto", high_precision_level=0) == "native"
        assert select("auto", high_precision_level=1) == "native"


def test_auto_on_blackwell_rejects_an_unlinked_precision_level():
    with env(BLACKWELL, mask=0b00011):
        with pytest.raises(RuntimeError, match="high_precision_level=2"):
            select("auto", high_precision_level=2)


def test_auto_on_blackwell_requires_the_bindings_not_only_the_mask():
    with env(BLACKWELL, mask=0b11111, bindings=False):
        with pytest.raises(RuntimeError, match="native"):
            select("auto")


def test_explicit_sm90_is_sm90_only_and_never_an_alias():
    with env(BLACKWELL, mask=0b11111):
        with pytest.raises(RuntimeError, match="SM90"):
            select("sm90")


def test_explicit_native_never_becomes_sm90_on_an_sm90_device():
    # The device probe, not the capability name, decides whether a native
    # reader exists; on an SM90 device it reports no native level, so the
    # explicit request raises instead of being served by the SM90 kernels.
    with env(SM90, mask=0):
        with pytest.raises(RuntimeError, match="native"):
            select("native")


def test_explicit_native_on_a_non_blackwell_device_raises():
    with env(AMPERE, mask=0):
        with pytest.raises(RuntimeError, match="native"):
            select("native")


def test_explicit_native_requires_a_cuda_tensor():
    saved = nd.device_capability
    nd.device_capability = lambda q: None
    try:
        with pytest.raises(RuntimeError, match="CUDA"):
            select("native")
        assert select("torch") == "torch"
    finally:
        nd.device_capability = saved


def test_non_full_causal_visibility_keeps_historical_routing():
    with env(BLACKWELL, mask=0b11111):
        # Sliding window/chunk stay explicit follow-ups: auto keeps routing
        # them the way it always has instead of claiming a native kernel.
        assert select("auto", full_causal=False) == "torch"
        with pytest.raises(NotImplementedError, match="full-causal"):
            select("native", full_causal=False)


def test_messages_name_the_device_and_the_available_levels():
    with env(BLACKWELL, mask=0b00101):
        with pytest.raises(RuntimeError) as info:
            select("native", high_precision_level=1)
    message = str(info.value)
    assert "capability" in message
    assert "0, 2" in message


# --- route trace ----------------------------------------------------------


def test_route_trace_records_the_resolved_route_and_its_refusals():
    nd.set_route_trace(True)
    try:
        with env(BLACKWELL, mask=0b00011):
            assert select("auto", high_precision_level=1) == "native"
            record = nd.last_route()
            assert record["resolved"] == "native"
            assert record["level"] == 1
            assert record["mask"] == 0b00011
            assert record["capability"] == BLACKWELL
            with pytest.raises(RuntimeError):
                select("auto", high_precision_level=3)
            refused = nd.last_route()
            assert refused["resolved"] is None
            assert "high_precision_level=3" in refused["error"]
        assert [r["resolved"] for r in nd.route_trace()] == ["native", None]
        assert all(r["kind"] == "select" for r in nd.route_trace())
    finally:
        nd.set_route_trace(False)
    assert nd.route_trace() == [] and nd.last_route() is None


def test_route_trace_is_off_by_default_and_records_native_leaf_entries():
    assert nd.route_trace() == []
    with env(BLACKWELL, mask=0b00011):
        assert select("auto") == "native"
        assert nd.route_trace() == []
        nd.set_route_trace(True)
        try:
            nd.record_native_leaf("fwd", CPU_TENSOR, 0)
            leaf = nd.last_route()
            assert leaf["kind"] == "native_fwd"
            assert leaf["mask"] == 0b00011
        finally:
            nd.set_route_trace(False)
