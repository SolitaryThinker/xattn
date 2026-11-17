"""Device-free host-logic rows for xattn/ops/_precision_backend.py backend selection.

Port of a precision-backend selection probe (48 declared
rows plus the probe's P10/P11 rows), so that a gate that runs `pytest` decides
the same questions the probe decides, on a machine with no GPU at all.

What this decides:
  * on a B300-shaped device (capability patched to (10,3)) with NO native
    precision linked, every backend choice fails loudly and names what is
    missing (no SM90 alias, no silent lower level);
  * with a simulated linked native implementation (fake ops module + fake level
    mask), 'auto'/'native' resolve to 'native' only when the bit of the native
    precision-repair leaf (``pb._NATIVE_REPAIR_BIT``, the bit the public
    ``high_precision_level=1`` backward resolves) is set, and only for
    full-causal visibility;
  * the required-symbol set is per branch: a deterministic call does not
    require the probability repair, a non-deterministic one does;
  * the SM90 fallback still resolves from logic alone (get_device_capability
    is patched to (9,0)); that arm is logic-only on this hardware - no SM90
    device exists here, so it is NOT evidence about SM90 execution;
  * the entry points agree with the per-branch table: apply()'s precheck asks
    for the symbols the requested (deterministic or not) branch executes;
  * every failure names its cause: a broken extension import, a crashing
    availability probe, a missing symbol with the device, and a build whose
    mask contradicts a forced 'sm90' are four different messages;
  * one refused call and one resolved call each leave their own route record.

Why it is device-free
---------------------
Every tensor is a `_StubTensor`: a descriptor exposing ``is_cuda = True``,
``device``, ``dtype``, ``shape`` (a ``torch.Size``), ``dim()`` and ``numel()``,
with no storage. ``torch.cuda.get_device_capability`` is patched per case and
restored by monkeypatch. Every case is expected to stop during argument
validation or at the capability decision, before any tensor data would be read,
so nothing here needs a device: no CUDA tensor is created, nothing is moved to
a device, and ``torch.cuda.is_available()`` may be False. `backward()` rows
(A5, A8, P6) are asserted on the exception they raise during validation or at
resolution; if such a call ever got past the decision the stub would raise
`AttributeError`/`TypeError` on the first data touch, and the row would fail
with that type instead of the expected one - the row cannot silently pass by
running a kernel.

Not ported here
---------------
None: all 50 probe rows reach only host logic and are ported as they are. Two
differences from a GPU run are the probe's own: A1-A2/A5 exercise the
missing-extension path because `xattn_cuda` is not importable in a CPU-only
checkout (the probe's not_covered records the same for its job workspace), and
A5/A8 stop at the capability decision rather than inside a kernel, exactly as
they do on a device with nothing linked.

Not covered (the probe's `not_covered` list is kept verbatim in `NOT_COVERED`
below and summarised here)
--------------------------------------------------------------------------
  * the SM90 hardware arm (no SM90 device here): E1/E2 check logic only;
  * a real linked native precision build (mask is simulated in B-D);
  * the GQA/state/shape validations of backward() (untouched code);
  * a real xattn_cuda extension in the job workspace;
  * any numerical agreement, any timing;
  * G1-G13 use simulated builds and a stubbed autograd Function;
  * P1-P11 are host-logic rows on simulated builds, so SM90 selection is
    logic only, never SM90 hardware;
  * P6 asserts only that an argument error is raised before the capability
    decision, not the wording of each check.

Nothing here is a claim about GPU behaviour: this module decides host logic
only.
"""
import builtins
import inspect
import os
import sys
import types

import pytest
import torch

KERNEL_ROOT = os.environ.get("KERNEL_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if KERNEL_ROOT not in sys.path:
    sys.path.insert(0, KERNEL_ROOT)

from xattn.ops import _precision_backend as pb  # noqa: E402
from xattn.ops.softdelta_attention import _normalize_backend as _public_normalize  # noqa: E402


ALL_OPS = (
    "_attention_precision_center", "_attention_precision_scale_do",
    "_attention_precision_row_statistics", "_attention_precision_main_backward",
    "_attention_precision_ordered_backward", "_attention_precision_zero_sum_repair",
    "_attention_precision_deterministic_repair", "_attention_precision_probability_repair",
    "_attention_precision_fallback_dv", "_attention_precision_key_runs",
    "_attention_precision_enforce_constant_keys",
)

_OPS_MINUS_ZERO_SUM = tuple(n for n in ALL_OPS if n != "_attention_precision_zero_sum_repair")
_OPS_NO_MAIN = tuple(n for n in ALL_OPS if n != "_attention_precision_main_backward")
_OPS_MINUS_PROBABILITY = tuple(n for n in ALL_OPS
                               if n != "_attention_precision_probability_repair")

B300 = (10, 3)
SM90 = (9, 0)
NATIVE_LEVELS = 0b11100
# The mask bit the public level-1 backward resolves: the native leaf's repair algorithm.
L = pb._NATIVE_REPAIR_BIT

NOT_COVERED = (
    "SM90 hardware arm (no SM90 device here): E1/E2 check logic only",
    "a real linked native precision build (mask is simulated in B-D)",
    "the GQA/state/shape validations of backward() (untouched code)",
    "a real xattn_cuda extension in the job workspace (import fails there, so A1-A2 "
    "exercise the missing-extension path and A6-A9 the SM90-shaped build path)",
    "any numerical agreement, any timing",
    "G1-G13 use simulated builds and a stubbed autograd Function: they decide the "
    "selection and diagnostic logic, not any kernel behaviour",
    "P1-P11 are host-logic rows on simulated builds; P3/P4/P5/P10 patch the capability to "
    "(9,0), so SM90 selection is logic only, never SM90 hardware",
    "P6 asserts only that an argument error is raised before the capability decision "
    "(class and absence of the capability text), not the wording of each check",
)

_SPELLING_EXPECTED = (
    ("", "auto", "auto"), ("auto", "auto", "auto"), (" AUTO ", "auto", "auto"),
    ("Auto", "auto", "auto"), ("sm90", "sm90", "sm90"), (" SM-90 ", "ValueError", "ValueError"),
    ("torch", "ValueError", "torch"),
)

_NO_WANT = object()
_REAL_EXTENSION = pb._extension
_REAL_ATTENTION = pb._Attention

# A device descriptor, not an allocation: it is never queried, because every
# capability read in this module is patched.
_STUB_DEVICE = torch.device("cuda", 0)


class _StubTensor:
    """A CUDA tensor's descriptor: shape/dtype/device yes, storage never."""

    is_cuda = True

    def __init__(self, shape, dtype=torch.float16, device=_STUB_DEVICE):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = device

    def dim(self):
        return len(self.shape)

    def numel(self):
        count = 1
        for size in self.shape:
            count *= size
        return count

    def __getitem__(self, index):
        """Shape-only slicing, so a row can build `k[:, :4]` without storage."""
        items = index if isinstance(index, tuple) else (index,)
        sizes, kept, taken = list(self.shape), [], 0
        for position, item in enumerate(items):
            if item is Ellipsis:
                fill = len(sizes) - taken - (len(items) - position - 1)
                kept.extend(sizes[taken:taken + fill])
                taken += fill
                continue
            if taken >= len(sizes):
                raise IndexError("too many indices for a stub tensor")
            size, taken = sizes[taken], taken + 1
            if item is None:
                kept.append(1)
            elif isinstance(item, int):
                if not -size <= item < size:
                    raise IndexError("stub tensor index out of range")
            elif isinstance(item, slice):
                kept.append(len(range(size)[item]))
            else:
                raise TypeError("stub tensors index with int, slice, Ellipsis or None only")
        kept.extend(sizes[taken:])
        return _StubTensor(tuple(kept), self.dtype, self.device)

    def __repr__(self):
        return "<stub CUDA %s %s>" % (self.dtype, tuple(self.shape))


class _Inputs:
    """The probe's q/k/v/dO, as descriptors of the same shapes and dtype."""

    def __init__(self):
        self.q = _StubTensor((1, 8, 2, 64))
        self.k = _StubTensor((1, 8, 1, 64))
        self.v = _StubTensor((1, 8, 1, 64))
        self.dy = _StubTensor((1, 8, 2, 64))


class _EnteredAttention(torch.autograd.Function):
    """Stands in for the autograd Function: `apply` records that it was entered."""

    @staticmethod
    def forward(*args, **kwargs):
        return "entered"

    @staticmethod
    def apply(*args, **kwargs):
        return "entered"


def _ops_namespace(names, mask=_NO_WANT):
    ops = types.SimpleNamespace(**{name: (lambda *a, **k: None) for name in names})
    if mask is not _NO_WANT:
        ops._native_attention_level_mask = lambda q: mask
    return ops


def _install(names, mask=_NO_WANT):
    """Simulate a linked build: the named ops, and (optionally) a level mask."""
    ops = _ops_namespace(names, mask)
    pb._extension = lambda: ops
    return ops


def _install_raising_probe(names):
    """Simulate a linked build whose capability probe raises."""

    def _explode(_q):
        raise RuntimeError("probe exploded")

    ops = types.SimpleNamespace(**{name: (lambda *a, **k: None) for name in names})
    ops._native_attention_level_mask = _explode
    pb._extension = lambda: ops


def _capture(fn):
    try:
        fn()
        return ""
    except BaseException as exc:  # noqa: BLE001 - the message is the result
        return str(exc)


def _value_or_kind(fn):
    """The value when the call returns, the exception class name when it raises."""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001
        return type(exc).__name__


# --- the rows, one function each, in the probe's order and construction ------


def _a1(t, mp):
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _a2(t, mp):
    return pb._resolve_backend("native", t.q, L, False, 0)


def _a3(t, mp):
    return pb._resolve_backend("sm90", t.q, L, False, 0)


def _a4(t, mp):
    return pb._resolve_backend("torch", t.q, L, False, 0)


def _a5(t, mp):
    return pb.backward(t.dy, t.q, t.k, t.v, visibility=0, scale=0.125,
                       backend="auto", deterministic=False)


def _a6(t, mp):
    _install(ALL_OPS)  # an SM90-shaped build: all eleven ops, no native mask
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _a7(t, mp):
    _install(ALL_OPS)
    return pb._resolve_backend("sm90", t.q, L, False, 0)


def _a8(t, mp):
    _install(ALL_OPS)
    return pb.backward(t.dy, t.q, t.k, t.v, visibility=0, scale=0.125,
                       backend="auto", deterministic=False)


def _a9(t, mp):
    _install(ALL_OPS)
    return "does not set bit" in _capture(lambda: pb._resolve_backend("auto", t.q, L, False, 0))


def _b1(t, mp):
    _install(ALL_OPS, NATIVE_LEVELS)
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _b2(t, mp):
    _install(ALL_OPS, NATIVE_LEVELS)
    return pb._resolve_backend("auto", t.q, L, True, 0)


def _b3(t, mp):
    _install(ALL_OPS, NATIVE_LEVELS)
    return pb._resolve_backend("native", t.q, L, False, 1)


def _b4(t, mp):
    _install(ALL_OPS, NATIVE_LEVELS)
    return pb._resolve_backend("auto", t.q, L, False, 2)


def _b5(t, mp):
    _install(ALL_OPS, 0b01100)  # levels 2 and 3 only
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _c1(t, mp):
    return sorted(pb.required_precision_ops(False))


def _c2(t, mp):
    return "_attention_precision_probability_repair" not in pb.required_precision_ops(True)


def _c3(t, mp):
    return "_attention_precision_probability_repair" in pb.required_precision_ops(False)


def _c4(t, mp):
    return ("_attention_precision_ordered_backward" in pb.required_precision_ops(True)
            and "_attention_precision_main_backward" not in pb.required_precision_ops(True))


def _c5(t, mp):
    return not any(name in pb.required_precision_ops(False) for name in (
        "_attention_precision_ordered_backward", "_attention_precision_deterministic_repair"))


def _d1(t, mp):
    _install(_OPS_MINUS_PROBABILITY, NATIVE_LEVELS)
    return pb._resolve_backend("auto", t.q, L, True, 0)


def _d2(t, mp):
    _install(_OPS_MINUS_PROBABILITY, NATIVE_LEVELS)
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _d3(t, mp):
    _install(_OPS_MINUS_PROBABILITY, NATIVE_LEVELS)
    return "probability_repair" in _capture(lambda: pb._resolve_backend("auto", t.q, L, False, 0))


def _e1(t, mp):
    _install(ALL_OPS, 0)
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _e2(t, mp):
    _install(ALL_OPS, 0)
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    return pb._resolve_backend("native", t.q, L, False, 0)


def _f1(t, mp):
    pb._extension = lambda: None
    return pb._ops(L, False)


def _f2(t, mp):
    pb._extension = lambda: None
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: B300)
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _g1(t, mp):
    _install(_OPS_MINUS_PROBABILITY, NATIVE_LEVELS)
    return pb._ops(L, True, device=t.q.device)


def _g2(t, mp):
    _install(_OPS_MINUS_PROBABILITY, NATIVE_LEVELS)
    pb._Attention = _EnteredAttention
    return pb.apply(t.q, t.k, t.v, visibility=0, high_precision_level=1, deterministic=True)


def _g3(t, mp):
    _install(_OPS_MINUS_PROBABILITY, NATIVE_LEVELS)
    pb._Attention = _EnteredAttention
    return "probability_repair" in _capture(
        lambda: pb.apply(t.q, t.k, t.v, visibility=0, high_precision_level=1, deterministic=False))


def _g4(t, mp):
    _install(_OPS_MINUS_ZERO_SUM, NATIVE_LEVELS)
    pb._Attention = _EnteredAttention
    return pb.apply(t.q, t.k, t.v, visibility=0, high_precision_level=1)


def _g5(t, mp):
    _install(_OPS_NO_MAIN, NATIVE_LEVELS)
    return str(t.q.device) in _capture(lambda: pb._ops(L, False, device=t.q.device))


def _g6(t, mp):
    _boom_import(mp)
    return pb._ops(L, False)


def _g7(t, mp):
    _boom_import(mp)
    return "OSError" in _capture(lambda: pb._ops(L, False))


def _boom_import(mp):
    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "xattn_cuda":
            raise OSError("libcudart.so.13: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    mp.setattr(builtins, "__import__", _boom)


def _g8(t, mp):
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: B300)
    _install_raising_probe(ALL_OPS)
    return "availability probe raised" in _capture(
        lambda: pb._resolve_backend("auto", t.q, L, False, 0))


def _g9(t, mp):
    _install(ALL_OPS, 1 << L)
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    return pb._resolve_backend("sm90", t.q, L, False, 0)


def _g10(t, mp):
    return pb._check_backend_name(""), pb._check_backend_name(" AUTO ")


def _g11(t, mp):
    return ("not an advanced-precision backward backend"
            in _capture(lambda: pb._check_backend_name("torch")))


def _g12(t, mp):
    return ("'native' or 'sm90'" in _capture(lambda: pb._check_backend_name("bogus"))
            and "reference route" not in _capture(lambda: pb._check_backend_name("bogus")))


def _g13(t, mp):
    return (inspect.signature(pb.backward).parameters["backend"].default == "auto"
            and "backend" in inspect.signature(pb._matched_backward).parameters)


def _p1(t, mp):
    return _branch_symbol_guard(t.q)


def _p2(t, mp):
    return _spelling_rows()


def _p3(t, mp):
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    _install(ALL_OPS, NATIVE_LEVELS)
    return pb._resolve_backend("auto", t.q, L, False, 0)


def _p4(t, mp):
    return _trace_rows(t.q)


def _p5(t, mp):
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    _install(ALL_OPS, 0b1000)
    return (pb._resolve_backend("sm90", t.q, L, False, 0),
            pb._resolve_backend("auto", t.q, L, False, 0))


def _p6(t, mp):
    return _order_rows(t)


def _p7(t, mp):
    return _probe_rows(t.q)


def _p8(t, mp):
    return _crash_row(t.q, mp)


def _p9(t, mp):
    return _no_ops_row(mp)


def _p10(t, mp):
    return _sm90_auto_disagreement(t.q, mp)


def _p11(t, mp):
    return _record_is_per_call(t.q, mp)


class _Case:
    """One probe row: its name, the capability it runs at, and its expectation."""

    __slots__ = ("name", "capability", "run", "expect", "want")

    def __init__(self, name, run, capability=B300, expect="returned", want=_NO_WANT):
        self.name, self.run, self.capability = name, run, capability
        self.expect, self.want = expect, want


def _row(name, run, capability=B300, expect="returned", want=_NO_WANT):
    return pytest.param(_Case(name, run, capability, expect, want), id=name)


CASES = (
    # A. real extension state, B300-shaped device: fail loudly, naming what is absent.
    _row("A1_real_auto", _a1, expect="ValueError"),
    _row("A2_real_forced_native", _a2, expect="ValueError"),
    _row("A3_forced_sm90_on_non_sm90", _a3, expect="ValueError"),
    _row("A4_backend_torch_rejected", _a4, expect="ValueError"),
    _row("A5_end_to_end_no_native", _a5, expect="ValueError"),
    # A6-A9: an SM90-built extension (all eleven ops, no native mask) on this device.
    _row("A6_sm90_build_auto", _a6, expect="ValueError"),
    _row("A7_sm90_build_forced_sm90", _a7, expect="ValueError"),
    _row("A8_sm90_build_end_to_end", _a8, expect="ValueError"),
    _row("A9_mask_missing_message_names_bit", _a9, want=True),
    # B. simulated complete native build: mask bits 2..4 (the repair bit L is 4).
    _row("B1_auto_native", _b1, want="native"),
    _row("B2_auto_det_native", _b2, want="native"),
    _row("B3_native_window_visibility_rejected", _b3, expect="NotImplementedError"),
    _row("B4_native_chunk_visibility_rejected", _b4, expect="NotImplementedError"),
    _row("B5_repair_bit_not_in_mask", _b5, expect="ValueError"),
    # C. per-branch symbol table.
    _row("C1_plain_set", _c1, want=sorted((
        "_attention_precision_center", "_attention_precision_row_statistics",
        "_attention_precision_key_runs", "_attention_precision_enforce_constant_keys",
        "_attention_precision_main_backward", "_attention_precision_zero_sum_repair",
        "_attention_precision_probability_repair", "_attention_precision_scale_do",
        "_attention_precision_fallback_dv"))),
    _row("C2_det_has_no_probability_repair", _c2, want=True),
    _row("C3_plain_has_probability_repair", _c3, want=True),
    _row("C4_det_uses_ordered_only", _c4, want=True),
    _row("C5_plain_does_not_require_the_det_branch", _c5, want=True),
    # D. a build missing one symbol fails only where that symbol is used.
    _row("D1_det_ok_without_probability_repair", _d1, want="native"),
    _row("D2_plain_names_missing_probability_repair", _d2, expect="NotImplementedError"),
    _row("D3_error_text_names_symbol", _d3, want=True),
    # E. SM90 fallback resolves from logic only (not hardware evidence).
    _row("E1_auto_on_sm90_logic_sm90", _e1, want="sm90"),
    _row("E2_forced_native_on_sm90_rejected", _e2, expect="ValueError"),
    # F. missing extension.
    _row("F1_ops_without_extension", _f1, expect="RuntimeError"),
    _row("F2_resolve_without_extension", _f2, expect="ValueError"),
    # G. entry-point granularity and diagnostics (review items).
    _row("G1_ops_det_passes_without_probability_repair", _g1),
    _row("G2_apply_det_precheck_not_plain", _g2, want="entered"),
    _row("G3_apply_plain_precheck_names_probability_repair", _g3, want=True),
    _row("G4_apply_L1_requires_the_branch_set", _g4, expect="RuntimeError"),
    _row("G5_missing_symbol_names_device", _g5, want=True),
    _row("G6_broken_import_fails_closed", _g6, expect="RuntimeError"),
    _row("G7_broken_import_named", _g7, want=True),
    _row("G8_probe_crash_reported_as_crash", _g8, want=True),
    _row("G9_sm90_named_but_native_mask_rejected", _g9, expect="ValueError"),
    _row("G10_backend_empty_and_auto", _g10, want=("auto", "auto")),
    _row("G11_torch_message_names_torch", _g11, want=True),
    _row("G12_other_message_lists_names_not_torch", _g12, want=True),
    _row("G13_defaults_auto_and_propagated", _g13, want=True),
    # P. later review items, each decided by its own row.
    _row("P1_branch_symbol_guard", _p1, want=(18, [])),
    _row("P2_spelling_rule_matches_public", _p2, want=_SPELLING_EXPECTED),
    _row("P3_auto_on_sm90_keeps_sm90", _p3, want="sm90"),
    _row("P4_route_trace_records_both_ways", _p4,
         want=(("native", NATIVE_LEVELS, None), (None, 0, "ValueError"))),
    _row("P5_partial_mask_sm90_agrees_with_auto", _p5, want=("sm90", "sm90")),
    _row("P6_argument_error_wins_over_capability", _p6,
         want=(("ValueError", False), ("ValueError", False))),
    _row("P7_probe_error_distinct_from_zero", _p7, want=("raised", "does not set bit")),
    _row("P8_route_record_carries_probe_error", _p8, want=("sm90", "RuntimeError")),
    _row("P9_no_ops_submodule_message", _p9, want=True),
    _row("P10_sm90_auto_checks_mask_symbol_agreement", _p10,
         want=(("NotImplementedError", True), "sm90")),
    _row("P11_route_record_is_per_call", _p11, want=(0b00100, (9, 0), None)),
)


@pytest.fixture(autouse=True)
def host_state(monkeypatch):
    """Start every row from the module's own state, and leave it as it was."""
    pb._extension = _REAL_EXTENSION
    pb._Attention = _REAL_ATTENTION
    pb._ROUTE_TRACE.clear()
    pb._EXTENSION_ERROR = None
    pb._PROBE_ERROR = None
    pb._LAST_MASK, pb._LAST_CAPABILITY, pb._LAST_PROBE_ERROR = 0, (0, 0), None
    yield
    pb._extension = _REAL_EXTENSION
    pb._Attention = _REAL_ATTENTION
    pb._ROUTE_TRACE.clear()
    pb._EXTENSION_ERROR = None
    pb._PROBE_ERROR = None
    pb._LAST_MASK, pb._LAST_CAPABILITY, pb._LAST_PROBE_ERROR = 0, (0, 0), None


def test_stub_tensor_carries_the_metadata_the_rows_need():
    """The precondition of every row below: a descriptor, never an allocation."""
    tensor = _StubTensor((1, 8, 2, 64))
    assert tensor.is_cuda is True
    assert isinstance(tensor.shape, torch.Size) and tuple(tensor.shape) == (1, 8, 2, 64)
    assert tensor.dim() == 4 and tensor.numel() == 1024
    assert tensor.dtype is torch.float16
    sliced = tensor[:, :4]
    assert isinstance(sliced, _StubTensor) and tuple(sliced.shape) == (1, 4, 2, 64)
    assert sliced.is_cuda and sliced.dtype is tensor.dtype and sliced.device == tensor.device


@pytest.mark.parametrize("case", CASES)
def test_precision_backend_selection(case, monkeypatch, host_state):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: case.capability)
    try:
        value = case.run(_Inputs(), monkeypatch)
        outcome, detail = "returned", value
    except BaseException as exc:  # noqa: BLE001 - the exact type and text are the result
        outcome, detail = type(exc).__name__, str(exc)
        value = None
    if case.expect == "returned":
        assert outcome == "returned", "%s raised %s: %s" % (case.name, outcome, detail)
        if case.want is not _NO_WANT:
            assert value == case.want, "%s returned %r, expected %r" % (case.name, value, case.want)
    else:
        assert outcome == case.expect, "%s raised %s (%s), expected %s" % (
            case.name, outcome, detail, case.expect)


# --- row helpers, one per probe helper ---------------------------------------


def _branch_symbol_guard(q):
    """P1: for each of the two branches, each required symbol missing alone must raise naming it."""
    subcases, failures = 0, []
    for level in (L,):
        for deterministic in (False, True):
            for missing in pb.required_precision_ops(deterministic):
                _install(tuple(n for n in ALL_OPS if n != missing), NATIVE_LEVELS)
                subcases += 1
                try:
                    pb._resolve_backend("native", q, level, deterministic, 0)
                    failures.append((level, deterministic, missing, "returned"))
                except NotImplementedError as exc:
                    if missing not in str(exc):
                        failures.append((level, deterministic, missing, "text"))
                except BaseException as exc:  # noqa: BLE001
                    failures.append((level, deterministic, missing, type(exc).__name__))
    _install(ALL_OPS, NATIVE_LEVELS)
    return subcases, failures


def _spelling_rows():
    """P2: the same spelling rule as the public normalizer, plus native's policy sentence."""
    rows = []
    for name in ("", "auto", " AUTO ", "Auto", "sm90", " SM-90 ", "torch"):
        rows.append((name, _value_or_kind(lambda: pb._check_backend_name(name)),
                     _value_or_kind(lambda: _public_normalize(name))))
    return tuple(rows)


def _trace_rows(q):
    """P4: a resolved call and a refused one both leave a record."""
    pb._ROUTE_TRACE.clear()
    _install(ALL_OPS, NATIVE_LEVELS)
    pb._resolve_call_backend("auto", q, L, False, 0)
    _install(ALL_OPS, 0)
    try:
        pb._resolve_call_backend("auto", q, L, False, 0)
    except ValueError:
        pass
    return tuple((row["resolved"], row["native_mask"],
                  None if row["refused"] is None else row["refused"].split(":")[0])
                 for row in pb.route_trace())


def _order_rows(t):
    """P6: an argument error must be reported before the capability decision."""

    def _row(fn):
        text = _capture(fn)
        return _value_or_kind(fn), "no native advanced-precision" in text

    return (_row(lambda: pb.backward(t.dy, t.q, t.k, t.v, visibility=1, span=-1,
                                     scale=0.125)),
            _row(lambda: pb.backward(t.dy, t.q, t.k, t.v, visibility=0, scale=0.125,
                                     prev_k=t.k[:, :4], prev_v=None)))


def _probe_rows(q):
    """P7: a probe that raises must not read as a probe that says zero."""
    _install_raising_probe(ALL_OPS)
    raised = _capture(lambda: pb._resolve_backend("auto", q, L, False, 0))
    _install(ALL_OPS, 0)
    zero = _capture(lambda: pb._resolve_backend("auto", q, L, False, 0))
    return ("raised" if "availability probe raised" in raised else raised[:200],
            "does not set bit" if "does not set bit" in zero else zero[:200])


def _crash_row(q, mp):
    """P8: a probe crash on the auto->sm90 path is recorded, not lost, and does not move the route."""
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    _install_raising_probe(ALL_OPS)
    pb._ROUTE_TRACE.clear()
    resolved = pb._resolve_call_backend("auto", q, L, False, 0)
    rows = pb.route_trace()
    if not rows:
        return resolved, "no record"
    error = rows[0].get("probe_error")
    return resolved, ("RuntimeError" if error is not None and "RuntimeError" in str(error)
                      else str(error))


def _no_ops_row(mp):
    """P9: a module that imports but has no .ops is named as that, not as '(None)'."""
    mp.setitem(sys.modules, "xattn_cuda", types.SimpleNamespace())
    got = _REAL_EXTENSION()
    error = str(pb._EXTENSION_ERROR)
    if got is not None:
        return "returned an ops module"
    return True if (".ops" in error and "None" != error) else error


def _sm90_auto_disagreement(q, mp):
    """P10: on major 9 the auto route still names a mask/symbol disagreement.

    Device class decides the route, but a build that advertises the level while
    its symbols disagree must be reported as that, not later as a RuntimeError
    from the legacy symbol check.
    """
    mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    _install(tuple(n for n in ALL_OPS if n != "_attention_precision_key_runs"), 1 << L)
    try:
        pb._resolve_backend("auto", q, L, False, 0)
        first = ("returned", False)
    except BaseException as exc:  # noqa: BLE001
        first = (type(exc).__name__, "_attention_precision_key_runs" in str(exc))
    _install(ALL_OPS, 1 << L)
    second = _value_or_kind(lambda: pb._resolve_backend("auto", q, L, False, 0))
    return first, second


def _record_is_per_call(q, mp):
    """P11: the route record carries THIS call's mask/capability/probe error.

    A second backward call arriving between this call's probe and its record
    must not be able to relabel it, so the value has to come from the call, not
    from module globals.
    """
    real = pb._resolve_backend

    def nested(backend, q_, level, deterministic, visibility, info=None):
        pb._LAST_MASK, pb._LAST_CAPABILITY, pb._LAST_PROBE_ERROR = 0xDEAD, (1, 1), "other-call"
        if info is not None:
            info.update(capability=SM90, native_mask=0b00100, probe_error=None)
        return "native"

    pb._resolve_backend = nested
    try:
        pb._resolve_call_backend("native", q, L, False, 0)
    finally:
        pb._resolve_backend = real
    record = pb.route_trace()[-1]
    return record["native_mask"], tuple(record["capability"]), record["probe_error"]
