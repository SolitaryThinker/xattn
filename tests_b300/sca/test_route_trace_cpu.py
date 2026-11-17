# Author: Shicheng Wen

"""CPU contract tests for the SCA route trace.

These tests are deliberately device-free: they import ``_native_dispatch`` only,
use CPU tensors for the device/capability fields, and never touch the compiled
extension. The route trace is a host-side record written *after* the dispatch
decision, so its contract can be checked without a GPU at all.
"""

import pytest
import torch

from xattn.ops import _native_dispatch as nd


@pytest.fixture(autouse=True)
def _clean_trace():
    nd.set_route_trace(False)
    yield
    nd.set_route_trace(False)


def _cpu():
    return torch.zeros(2, 2)


def test_tracing_off_allocates_nothing_and_records_nothing():
    scope = nd.route_scope(
        "flash_sca_fwd", "forward", q=_cpu(), requested="auto", level=0
    )
    assert scope is nd._NULL_ROUTE_SCOPE
    with scope as entered:
        entered.resolved("native")
        entered.set(deterministic=False)
    assert nd.record_route("native") is None
    assert nd.route_trace() == []
    assert nd.last_route() is None


def test_one_entry_per_public_call_with_selector_field_names():
    nd.set_route_trace(True)
    with nd.route_scope(
        "flash_sca_fwd", "forward", q=_cpu(), requested="auto", level=1
    ) as scope:
        scope.resolved("native")
    trace = nd.route_trace()
    assert len(trace) == 1
    record = trace[0]
    # Same names as the selector record, so a reader joins them directly.
    assert record["kind"] == "sca_route"
    assert record["api_name"] == "flash_sca_fwd"
    assert record["phase"] == "forward"
    assert record["requested"] == "auto"
    assert record["level"] == 1
    assert record["device"] == "cpu"
    assert record["capability"] is None
    assert record["resolved"] == "native"
    assert "error" not in record
    assert "nested" not in record


def test_nested_call_folds_into_the_outer_entry_and_is_never_appended_alone():
    nd.set_route_trace(True)
    with nd.route_scope(
        "flash_sca", "forward", q=_cpu(), requested="auto", level=0
    ) as outer:
        outer.resolved("nonpositive_scale")
        # The nonpositive-scale re-entry is a second call to the same entry
        # point; it must not become a second trace entry for one public call.
        with nd.route_scope(
            "flash_sca", "forward", q=_cpu(), requested="auto", level=0
        ) as inner:
            inner.resolved("native")
    trace = nd.route_trace()
    assert len(trace) == 1
    record = trace[0]
    assert record["resolved"] == "nonpositive_scale"
    assert isinstance(record["nested"], list) and len(record["nested"]) == 1
    assert record["nested"][0]["api_name"] == "flash_sca"
    assert record["nested"][0]["resolved"] == "native"


def test_record_route_inside_a_scope_fills_the_open_record():
    nd.set_route_trace(True)
    with nd.route_scope("flash_sca_bwd", "backward", q=_cpu(), requested="auto", level=1):
        assert nd.record_route("high_precision", deterministic=False) is None
    record = nd.last_route()
    assert record["resolved"] == "high_precision"
    assert record["deterministic"] is False
    assert len(nd.route_trace()) == 1


def test_a_raising_call_is_recorded_once_and_the_error_propagates():
    nd.set_route_trace(True)
    with pytest.raises(RuntimeError):
        with nd.route_scope("flash_sca", "forward", q=_cpu(), requested="native", level=0):
            raise RuntimeError("no native route")
    trace = nd.route_trace()
    assert len(trace) == 1
    assert trace[0]["error"] == "RuntimeError: no native route"


def test_record_route_outside_a_scope_appends_its_own_entry():
    nd.set_route_trace(True)
    nd.record_route("generic", api_name="flash_sca_fwd", phase="forward", level=0)
    record = nd.last_route()
    assert record["kind"] == "sca_route"
    assert record["resolved"] == "generic"
    assert record["api_name"] == "flash_sca_fwd"
    assert record["phase"] == "forward"


def test_a_companion_dict_record_is_stored_once_verbatim_with_kind_filled():
    # The companion emitter (`_precision_backend`'s sink) builds its own record;
    # its fields must reach the ring unchanged, so a reader of the one ring sees
    # the same values as a reader of that emitter's own history.
    offered = {
        "call": "advanced-precision-backward",
        "level": 4,
        "deterministic": True,
        "ops": ["center_k", "row_statistics", "main_backward"],
        "mask": 5,
    }
    nd.set_route_trace(True)
    entry = nd.record_route(dict(offered))
    assert entry["kind"] == "advanced-precision-backward"
    assert {k: v for k, v in entry.items() if k != "kind"} == offered
    assert nd.route_trace() == [entry]


def test_a_companion_dict_record_keeps_an_explicit_kind():
    nd.set_route_trace(True)
    entry = nd.record_route({"kind": "advanced_precision_sink", "call": "bwd", "level": 2})
    assert entry["kind"] == "advanced_precision_sink"
    assert entry["call"] == "bwd"


def test_a_companion_dict_record_is_one_entry_even_inside_an_open_scope():
    # A public call plus its companion record are two chain elements -- the
    # scope's own entry and the companion's -- never one folded into the other,
    # and in that order: the call's entry comes first because it was appended
    # when the scope opened, not when it closed.
    nd.set_route_trace(True)
    with nd.route_scope("flash_sca_bwd", "backward", q=_cpu(), requested="auto", level=1) as scope:
        scope.resolved("high_precision")
        nd.record_route({"call": "advanced-precision-backward", "level": 4})
    trace = nd.route_trace()
    assert len(trace) == 2
    assert trace[0]["kind"] == "sca_route"
    assert trace[0]["resolved"] == "high_precision"
    assert "nested" not in trace[0]
    assert trace[1]["kind"] == "advanced-precision-backward"


def test_a_partial_level_that_is_not_an_int_does_not_make_the_traced_call_raise():
    # Tracing on must not change the exception class of a call: an unusable
    # level is recorded as given instead of being coerced.
    nd.set_route_trace(True)
    with nd.route_scope("flash_sca_fwd", "forward", q=_cpu(), level="high") as scope:
        scope.resolved("generic")
    assert nd.last_route()["level"] == "high"


def test_a_raising_nested_call_records_the_failure_once():
    nd.set_route_trace(True)
    with pytest.raises(ValueError):
        with nd.route_scope("flash_sca", "forward", q=_cpu(), level=0) as outer:
            outer.resolved("nonpositive_scale")
            with nd.route_scope("flash_sca", "forward", q=_cpu(), level=0) as inner:
                inner.resolved("native")
                raise ValueError("bad scale")
    trace = nd.route_trace()
    assert len(trace) == 1
    record = trace[0]
    assert record["error"] == "ValueError: bad scale"
    assert record["nested"][0]["raised"] == "ValueError"
    assert "error" not in record["nested"][0]


def test_a_third_form_is_ignored_rather_than_recorded_as_a_route():
    nd.set_route_trace(True)
    assert nd.record_route(None) is None
    assert nd.record_route(17) is None
    assert nd.record_route(["native"]) is None
    assert nd.route_trace() == []


def test_the_native_leaf_and_selector_records_sit_beside_one_sca_route_entry():
    # The ring is shared: a real call that reaches the native leaf writes a
    # `select` record and a `native_*` record too. The invariant is one record
    # of kind `sca_route` per public call, not one record in total.
    nd.set_route_trace(True)
    with nd.route_scope("flash_sca_fwd", "forward", q=_cpu(), requested="auto", level=0) as scope:
        nd.record_route("native")
        nd._trace({"kind": "select", "api_name": "flash_sca_fwd", "resolved": "native"})
        nd._trace({"kind": "native_fwd", "level": 0})
    trace = nd.route_trace()
    assert [r["kind"] for r in trace] == ["sca_route", "select", "native_fwd"]
    assert sum(1 for r in trace if r["kind"] == "sca_route") == 1


def test_the_record_carries_the_annex_route_axes_read_from_the_call():
    # The dispatch annex keys R1 (dense) / R2 (segmented) / R3 (prev K/V) and
    # R4 / R5 / R6 on three shape axes. A reader must not have to re-infer them
    # from the arguments after the fact, so the record carries them.
    nd.set_route_trace(True)

    @nd.route_traced("flash_sca_fwd", "forward")
    def fwd(q, k, v, prev_k=None, prev_v=None, segment_idx=None, bos_mask=None,
            reset_chunk_pos_per_seq=False, deterministic=False, backend="auto",
            high_precision_level=0):
        return "ok"

    @nd.route_traced("flash_sca_bwd", "backward")
    def bwd(q, prev_k=None, prev_v=None, q_segment_idx=None, k_segment_idx=None,
            reset_chunk_pos_per_seq=False, deterministic=False):
        return "ok"

    assert fwd(_cpu(), None, None) == "ok"
    assert fwd(_cpu(), None, None, prev_k=_cpu(), prev_v=_cpu()) == "ok"
    assert fwd(_cpu(), None, None, bos_mask=_cpu(), reset_chunk_pos_per_seq=True) == "ok"
    assert bwd(_cpu(), q_segment_idx=_cpu(), k_segment_idx=_cpu(), deterministic=True) == "ok"
    # Positional backend and level, which kwargs reading would have missed.
    assert fwd(_cpu(), None, None, None, None, None, None, False, False, "cuda", 1) == "ok"

    trace = nd.route_trace()
    assert [(r["metadata"], r["has_prev"], r["reset"], r["deterministic"]) for r in trace] == [
        ("none", False, False, False),
        ("none", True, False, False),
        ("bos", False, True, False),
        ("segment", False, False, True),
        ("none", False, False, False),
    ]
    assert trace[4]["requested"] == "cuda"
    assert trace[4]["level"] == 1


def test_the_live_sink_emits_the_real_record_onto_the_shared_ring():
    # Drive the real emitter, not a hand-built dict: the ring entry is pinned
    # to the real field set, the verbatim-copy contract and the one-entry rule.
    # `_resolve_call_backend` records the refusal before re-raising, so a CPU
    # tensor here exercises the recorded-refusal path without a GPU.
    from xattn.ops import _precision_backend as pb

    pb._ROUTE_TRACE.clear()
    nd.set_route_trace(True)
    try:
        # The level the public level-1 backward resolves is the native leaf's repair bit.
        pb._resolve_call_backend("auto", _cpu(), level=pb._NATIVE_REPAIR_BIT, deterministic=False,
                                 visibility=2)
    except Exception:
        # Refused and resolved both leave exactly one record; the class of the
        # refusal is the resolver's business, not the trace's.
        pass
    ours = pb.route_trace()
    ring = nd.route_trace()
    assert len(ours) == 1
    assert len(ring) == 1
    assert ring[0]["kind"] == "advanced-precision-backward"
    assert ring[0]["call"] == "advanced-precision-backward"
    assert ring[0]["level"] == pb._NATIVE_REPAIR_BIT
    assert ring[0]["deterministic"] is False
    # Verbatim: `kind` filled from `call`, every other field identical.
    assert {k: v for k, v in ring[0].items() if k != "kind"} == ours[0]


def test_disabling_the_trace_clears_the_history():
    nd.set_route_trace(True)
    with nd.route_scope("flash_sca_fwd", "forward", q=_cpu()):
        pass
    assert len(nd.route_trace()) == 1
    nd.set_route_trace(False)
    assert nd.route_trace() == []


def test_route_traced_wrapper_is_transparent_when_off_and_records_once_when_on():
    seen = []

    @nd.route_traced("flash_sca_fwd", "forward")
    def entry(q, *, backend="auto", high_precision_level=0):
        seen.append((backend, high_precision_level))
        nd.record_route("native")
        return "y"

    assert entry(_cpu(), backend="auto", high_precision_level=0) == "y"
    assert seen == [("auto", 0)]
    assert nd.route_trace() == []

    nd.set_route_trace(True)
    assert entry(_cpu(), backend="native", high_precision_level=1) == "y"
    trace = nd.route_trace()
    assert len(trace) == 1
    assert trace[0]["api_name"] == "flash_sca_fwd"
    assert trace[0]["requested"] == "native"
    assert trace[0]["level"] == 1
    assert trace[0]["resolved"] == "native"
