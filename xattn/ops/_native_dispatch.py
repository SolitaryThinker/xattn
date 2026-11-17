# Author: Shicheng Wen

"""Truthful backend selection for the SoftDelta attention entry points.

The backend name a caller passes never implies that a kernel exists. The only
source of truth for native support is the extension's own capability probe
``xattn_cuda.ops._native_attention_level_mask(q)``: bit ``k`` is set when the
build links the native full-causal reader forward *and* backward for level
``k`` on the actual device of ``q``. The mask is support metadata only; a
numerical or performance claim needs separately recorded evidence.

Two rules keep selection honest:

* An explicit backend that this device or build cannot serve raises. It never
  degrades to another implementation.
* ``auto`` on a Blackwell CUDA device resolves to ``native``. If the build
  links no native reader for the requested level it raises instead of silently
  running the pure-PyTorch reference path, so a passing result can never be
  mistaken for native execution.

Sliding-window and sliding-chunk visibility keep their historical routing
(SM90 when the device is SM90, otherwise the torch reference path). They are
explicit follow-ups, not part of the native full-causal selector.
"""

from typing import Optional, Tuple
from collections import deque
import functools
import inspect
import threading

import torch
from torch import Tensor

_BACKENDS = ("auto", "torch", "sm90", "native")
_NATIVE_LEVEL_COUNT = 5
_NATIVE_LEVEL_MASK = (1 << _NATIVE_LEVEL_COUNT) - 1
_SM90_CAPABILITY = (9, 0)
_BLACKWELL_MAJOR = 10


def normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("SoftDelta backend must be a string")
    normalized = backend.strip().lower().replace("-", "_")
    if normalized not in ("", *_BACKENDS):
        raise ValueError(
            "SoftDelta backend must be one of: auto, torch, sm90, native"
        )
    return "auto" if normalized == "" else normalized


def extension_ops():
    """Return ``xattn_cuda.ops`` when the extension is importable, else None."""
    try:
        import xattn_cuda
    except ImportError:
        return None
    return getattr(xattn_cuda, "ops", None)


def device_capability(q: Tensor) -> Optional[Tuple[int, int]]:
    if not isinstance(q, Tensor) or not q.is_cuda:
        return None
    return tuple(torch.cuda.get_device_capability(q.device))


def device_is_sm90(q: Tensor) -> bool:
    return device_capability(q) == _SM90_CAPABILITY


def device_is_blackwell(q: Tensor) -> bool:
    capability = device_capability(q)
    return capability is not None and capability[0] >= _BLACKWELL_MAJOR


def native_reader_level_mask(q: Tensor) -> int:
    """Implemented and linked native full-causal reader levels for q's device.

    Bit ``k`` covers level ``k`` of both the native reader forward and the
    native basic reader backward. Zero means no native reader. A probe that
    raises or is absent means zero: absence of native support must never be
    read as permission to run another implementation.
    """
    if not isinstance(q, Tensor) or not q.is_cuda:
        return 0
    ops = extension_ops()
    probe = getattr(ops, "_native_attention_level_mask", None) if ops else None
    if probe is None:
        return 0
    try:
        mask = int(probe(q))
    except Exception:  # A failing probe reports "no native", never a fallback.
        return 0
    return mask & _NATIVE_LEVEL_MASK if mask > 0 else 0


def native_reader_bindings_present() -> bool:
    ops = extension_ops()
    if ops is None:
        return False
    return (
        getattr(ops, "_native_attention_fwd", None) is not None
        and getattr(ops, "_native_attention_bwd", None) is not None
        and getattr(ops, "_native_attention_level_mask", None) is not None
    )


def _level_names(mask: int) -> str:
    levels = [str(level) for level in range(_NATIVE_LEVEL_COUNT) if mask & (1 << level)]
    return ", ".join(levels) if levels else "none"


# --- route trace ----------------------------------------------------------
#
# Selection is a host decision that can pick a route the caller did not intend
# (for example the torch reference path). A qualification test that compares two
# streams, two replays or two devices must be able to prove which route each call
# took, otherwise a bitwise equality between two reference-path calls looks like
# a native result. The trace is off by default and costs one boolean test per
# call; when enabled it records the resolved route and every native leaf entry.

_TRACE_LIMIT = 512
_TRACE_ENABLED = False
_TRACE = deque(maxlen=_TRACE_LIMIT)


def set_route_trace(enabled: bool = True) -> None:
    """Turn route tracing on or off; disabling it also clears the history."""
    global _TRACE_ENABLED
    _TRACE_ENABLED = bool(enabled)
    if not _TRACE_ENABLED:
        _TRACE.clear()


def route_trace():
    """The recorded route decisions, oldest first."""
    return list(_TRACE)


def last_route() -> Optional[dict]:
    """The most recent route decision, or None when nothing was traced."""
    return _TRACE[-1] if _TRACE else None


def clear_route_trace() -> None:
    _TRACE.clear()


def _trace(record: dict) -> None:
    if _TRACE_ENABLED:
        _TRACE.append(record)


def record_native_leaf(kind: str, q: Tensor, level: int) -> None:
    """Record that the native leaf was entered, before it can fail."""
    if not _TRACE_ENABLED:
        return
    _trace({
        "kind": f"native_{kind}",
        "level": int(level),
        "device": str(q.device),
        "capability": device_capability(q),
        "mask": native_reader_level_mask(q),
    })


def _device_label(q: Tensor) -> str:
    try:
        name = torch.cuda.get_device_name(q.device)
    except Exception:
        name = "unknown device"
    capability = device_capability(q)
    return f"{name} (capability {capability})"


def _require_native(
    q: Tensor,
    level: int,
    full_causal: bool,
    api_name: str,
) -> str:
    if not full_causal:
        raise NotImplementedError(
            f"{api_name} has no native backend for sliding-window or "
            "sliding-chunk visibility yet (only full-causal is implemented); "
            "pass backend='torch' or backend='sm90' explicitly"
        )
    if not native_reader_bindings_present():
        raise RuntimeError(
            f"{api_name} native backend is unavailable on {_device_label(q)}: "
            "this xattn build does not link the native full-causal reader. "
            "Pass backend='torch' for the reference path or build with the "
            "native reader enabled."
        )
    mask = native_reader_level_mask(q)
    if not mask:
        raise RuntimeError(
            f"{api_name} native backend reports no native full-causal reader "
            f"level on {_device_label(q)}; pass backend='torch' for the "
            "reference path or build with the native reader enabled."
        )
    if level < 0 or level >= _NATIVE_LEVEL_COUNT or not (mask >> level) & 1:
        raise RuntimeError(
            f"{api_name} high_precision_level={level} is not implemented by "
            f"the native full-causal reader on {_device_label(q)} "
            f"(implemented levels: {_level_names(mask)}); pass a supported "
            "level or backend='torch' for the reference path"
        )
    return "native"


def select_backend(
    backend: str,
    q: Tensor,
    *,
    high_precision_level: int = 0,
    full_causal: bool = True,
    api_name: str = "attention",
) -> str:
    """Resolve a backend name to ``torch``, ``sm90`` or ``native``.

    Raises instead of degrading when the named backend cannot serve this
    device, build or precision level. Every decision, refusals included, is
    recorded for the route trace when tracing is on.
    """
    level = int(high_precision_level)
    record = {
        "kind": "select",
        "api_name": api_name,
        "requested": backend if isinstance(backend, str) else repr(backend),
        "level": level,
        "full_causal": bool(full_causal),
        "device": str(getattr(q, "device", None)),
        "capability": device_capability(q) if isinstance(q, Tensor) else None,
    }
    try:
        resolved = _resolve_backend(
            backend,
            q,
            high_precision_level=level,
            full_causal=full_causal,
            api_name=api_name,
        )
    except Exception as error:  # Recorded, then re-raised unchanged.
        if _TRACE_ENABLED:
            record["resolved"] = None
            record["error"] = f"{type(error).__name__}: {error}"
            _trace(record)
        raise
    if _TRACE_ENABLED:
        record["resolved"] = resolved
        record["mask"] = native_reader_level_mask(q)
        _trace(record)
    return resolved


def _resolve_backend(
    backend: str,
    q: Tensor,
    *,
    high_precision_level: int = 0,
    full_causal: bool = True,
    api_name: str = "attention",
) -> str:
    normalized = normalize_backend(backend)
    level = int(high_precision_level)
    capability = device_capability(q)
    if capability is None:
        if normalized in ("sm90", "native"):
            raise RuntimeError(
                f"{api_name} backend='{normalized}' requires a CUDA tensor"
            )
        return "torch"
    if normalized == "torch":
        return "torch"
    if normalized == "sm90":
        if capability != _SM90_CAPABILITY:
            raise RuntimeError(
                f"{api_name} backend='sm90' requires an SM90 GPU with "
                f"capability (9, 0); {_device_label(q)} is not SM90. "
                "Pass backend='native' for the native B300 path or "
                "backend='torch' for the reference path."
            )
        return "sm90"
    if normalized == "native":
        return _require_native(q, level, full_causal, api_name)
    if capability == _SM90_CAPABILITY:
        return "sm90"
    if full_causal and capability[0] >= _BLACKWELL_MAJOR:
        return _require_native(q, level, full_causal, api_name)
    return "torch"


# --- SCA route records ----------------------------------------------------
#
# The selector above records *its* decision. A sliding-chunk entry point
# decides more than a backend: one call resolves to one of five routes (the
# advanced-precision composite, the native leaf, the matched-reset bridge, the
# generic kernel or its nonpositive-scale re-entry), and no record above names
# that route. A qualification test comparing two runs therefore cannot say from
# the trace whether an SCA call ran the route it meant to compare.
#
# ``route_scope`` records one entry per public call. It is written after the
# decision and never read by dispatch, so enabling it cannot change which route
# runs. With tracing off it returns a shared null scope: one boolean test, no
# allocation. A call made while another scope is open -- the nonpositive-scale
# re-entry, or a helper calling back into a public entry point -- folds into the
# record of the call that made it, so one public call is one ``sca_route``
# entry whose nested arms are named inside it, never two for one call.
#
# The ring itself is shared with the emitters that were already writing to it
# (``select_backend``'s ``select`` records, ``record_native_leaf``'s
# ``native_*`` records, and a companion's own dicts), so the invariant is
# exactly one record of kind ``sca_route`` per public call, not one record in
# total: a real call that reaches the native leaf legitimately adds entries
# beside its own.

_ROUTE_KIND = "sca_route"
_ROUTE_DEPTH = threading.local()


class _NullRouteScope:
    """Stand-in scope used when tracing is off: no allocation, no record."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set(self, **fields):
        return self

    def resolved(self, route):
        return self


_NULL_ROUTE_SCOPE = _NullRouteScope()


class _RouteScope:
    """One open route record; nested scopes fold into their parent."""

    __slots__ = ("record", "outermost")

    def __init__(self, record, outermost):
        self.record = record
        self.outermost = outermost

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # The entry was appended when the scope opened, so a companion record
        # emitted during the call follows it in the ring and the chain keeps the
        # order the call actually had. Nothing is appended here.
        stack = getattr(_ROUTE_DEPTH, "stack", None) or []
        if stack and stack[-1] is self:
            stack.pop()
        if exc_type is not None:
            if self.outermost:
                self.record["error"] = f"{exc_type.__name__}: {exc}"
            else:
                # The outer entry carries the error text; the nested entry says
                # only that it raised, so one failure is reported once.
                self.record["raised"] = exc_type.__name__
        return False

    def set(self, **fields):
        self.record.update(fields)
        return self

    def resolved(self, route):
        self.record["resolved"] = route
        return self


def _route_stack():
    stack = getattr(_ROUTE_DEPTH, "stack", None)
    if stack is None:
        stack = []
        _ROUTE_DEPTH.stack = stack
    return stack


def _as_level(level):
    """The precision level as an int when it is one, else as given.

    Tracing must never turn a call that succeeds with tracing off into a call
    that raises: ``int(level)`` on a non-numeric level would do exactly that, so
    the raw value is recorded instead of coerced.
    """
    try:
        return int(level)
    except (TypeError, ValueError):
        return level


def _safe_capability(q):
    try:
        return device_capability(q) if isinstance(q, Tensor) else None
    except Exception:
        return None


def route_scope(
    api_name: str,
    phase: str,
    q=None,
    requested=None,
    level: int = 0,
    **fields,
):
    """Open one route record for a public entry point.

    ``phase`` is ``"forward"`` or ``"backward"``; ``requested`` is the backend
    name exactly as the caller passed it. The field names are the selector
    record's (``requested``, ``level``, ``device``, ``capability``) so a reader
    joins the two records without a second vocabulary.
    """
    if not _TRACE_ENABLED:
        return _NULL_ROUTE_SCOPE
    stack = _route_stack()
    record = {
        "kind": _ROUTE_KIND,
        "api_name": api_name,
        "phase": phase,
        "requested": requested if isinstance(requested, str) else repr(requested),
        "level": _as_level(level),
        "device": str(getattr(q, "device", None)),
        "capability": _safe_capability(q),
    }
    record.update(fields)
    scope = _RouteScope(record, outermost=not stack)
    if stack:
        stack[-1].record.setdefault("nested", []).append(record)
    else:
        # Appended now and filled at close, so a companion record emitted during
        # the call lands after its call's entry and the chain order matches the
        # order the call made them in.
        _trace(record)
    stack.append(scope)
    return scope


def record_route(route, **fields) -> Optional[dict]:
    """Name the route the enclosing SCA call resolved to.

    Two forms, one name. With a route string this fills the open scope's record,
    or appends an entry of its own when no scope is open. With a record dict --
    a companion emitter handing over a record it built itself -- the dict is
    stored once, verbatim, as its own entry (`kind` filled in when the emitter
    left it out) and never folded into an open scope, because a companion record
    is a second element of the chain rather than a nested call. With tracing off
    it returns None after one boolean test and allocates nothing; anything that
    is neither a string nor a dict is ignored rather than recorded as a route.
    """
    if not _TRACE_ENABLED:
        return None
    if isinstance(route, dict):
        entry = dict(route)
        if "kind" not in entry:
            call = entry.get("call")
            entry["kind"] = str(call) if call is not None else "route"
        entry.update(fields)
        _trace(entry)
        return entry
    if not isinstance(route, str):
        return None
    stack = getattr(_ROUTE_DEPTH, "stack", None) or []
    if stack:
        stack[-1].record["resolved"] = route
        if fields:
            stack[-1].record.update(fields)
        return None
    record = {
        "kind": _ROUTE_KIND,
        "api_name": fields.pop("api_name", None),
        "phase": fields.pop("phase", None),
        "resolved": route,
    }
    record.update(fields)
    _trace(record)
    return record


def _call_axes(bound):
    """The call's shape axes, as the dispatch annex's route table keys them.

    The annex's forward routes are R1 (dense), R2 (segmented: `segment_idx`, the
    q/k pair or `bos_mask`) and R3 (previous K/V); the level 0/1 backward routes
    are R4/R5/R6 on the same three axes. A bare `resolved` token cannot tell R1
    from R2, so the record carries the axes that can, read from the call's own
    arguments rather than re-inferred later.
    """
    has_prev = bound.get("prev_k") is not None or bound.get("prev_v") is not None
    if bound.get("bos_mask") is not None:
        metadata = "bos"
    elif (
        bound.get("segment_idx") is not None
        or bound.get("q_segment_idx") is not None
        or bound.get("k_segment_idx") is not None
    ):
        metadata = "segment"
    else:
        metadata = "none"
    return {
        "has_prev": has_prev,
        "metadata": metadata,
        "reset": bool(bound.get("reset_chunk_pos_per_seq", False)),
        "deterministic": bool(bound.get("deterministic", False)),
    }


def route_traced(api_name: str, phase: str):
    """Record one route entry per call of the decorated entry point.

    When tracing is off the wrapper calls straight through and adds nothing, so
    a decorated entry point behaves exactly as it did before.
    """

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _TRACE_ENABLED:
                return fn(*args, **kwargs)
            try:
                # Bind properly rather than reading kwargs: a caller may pass the
                # backend or the level positionally, and the record must not
                # misreport what was asked for.
                bound = signature.bind_partial(*args, **kwargs).arguments
            except TypeError:
                bound = {}
            with route_scope(
                api_name,
                phase,
                q=bound.get("q"),
                requested=bound.get("backend", "auto"),
                level=bound.get("high_precision_level", 0),
                **_call_axes(bound),
            ):
                return fn(*args, **kwargs)

        return wrapper

    return decorate
