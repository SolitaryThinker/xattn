"""Precision-repair backward (high_precision_level=1) shared by the attention
families, on SM90 and on native SM103. SoftDelta calls it for each read and
combines the FP32 accumulators before the public cast.

Backend selection is a capability decision, never a name alias: ``sm90`` is
SM90-only, ``native`` demands a linked native implementation of the exact
branch, and ``auto`` follows the device class -- the SM90 route on a major-9
device, the native route on a device that has no SM90 route -- unless the
caller names one.  It permits a route; it never picks a symbol: both
implementations register the same ``_attention_precision_*`` op names and the
linked build decides which body runs.  A missing component is reported as a
missing component; nothing is silently replaced by another backend or by the
level-0 backward.

The native extension reports what it links in ``_native_attention_level_mask``;
the precision-repair algorithm of level 1 is the native leaf's internal level 4
(dual-operand P/dS, range protection, probability-mass repair), so its bit is
``_NATIVE_REPAIR_BIT``.

Every resolved call, and every refused one, leaves a record in `route_trace()`,
so that a passing result cannot be mistaken for native execution.
"""
import torch
from ._segment_metadata import resolve_segment_indices, validate_segment_indices
from ._attention_core import resolve_scale


# The zero-sum repair spreads the non-anchor share of each row's dS operand
# residual over its last ZERO_SUM_WINDOW admitted keys.
ZERO_SUM_WINDOW = 64
_PRESCALE_TARGET = 2.0 ** 13

# Bit of `_native_attention_level_mask` that advertises the native leaf of the
# precision-repair backward (see the module docstring).
_NATIVE_REPAIR_BIT = 4

# Ops every precision-repair branch needs.
_PRECISION_COMMON_OPS = (
    '_attention_precision_center',
    '_attention_precision_row_statistics',
    '_attention_precision_key_runs',
    '_attention_precision_enforce_constant_keys',
)

# Ops the forward-side precheck of `apply` requires.
_LEGACY_REQUIRED_OPS = (
    '_attention_precision_main_backward',
    '_attention_precision_zero_sum_repair',
    '_attention_precision_ordered_backward',
    '_attention_precision_key_runs',
    '_attention_precision_enforce_constant_keys',
)

_BACKEND_NAMES = ('', 'auto', 'native', 'sm90')

# What the last capability read saw, for the route record below.
_LAST_MASK = 0
_LAST_CAPABILITY = (0, 0)
_LAST_PROBE_ERROR = None

# One route record per call, resolved or refused: what was asked for, what was
# resolved, and on what mask.  The same record is offered to the Python
# dispatcher's trace when that module provides a sink; a sink that fails never
# makes the op fail.
_ROUTE_TRACE = []
_ROUTE_TRACE_MAX = 64


def route_trace():
    """The route records this module made, oldest first."""
    return [dict(record) for record in _ROUTE_TRACE]


def _route_sink():
    try:
        from . import _native_dispatch
    except Exception:  # the dispatcher module is a separate proposal
        return None
    return getattr(_native_dispatch, 'record_route', None)


def _record_route(record):
    _ROUTE_TRACE.append(dict(record))
    del _ROUTE_TRACE[:-_ROUTE_TRACE_MAX]
    sink = _route_sink()
    if sink is not None:
        try:
            sink(dict(record))
        except Exception:  # noqa: BLE001 - a trace sink must not break the op
            pass


def _resolve_call_backend(backend, q, level, deterministic, visibility):
    """Resolve the backend for one call and record what was decided.

    The capability, mask and probe error go into the record straight from this
    call's own `info` dict, not from the module globals: two backward calls on
    different threads would otherwise be able to attribute one call's mask or
    probe failure to the other.  The globals stay for debugging only.
    """
    record = {'call': 'advanced-precision-backward', 'requested': backend,
              'level': level, 'deterministic': bool(deterministic),
              'visibility': visibility, 'device': str(q.device)}
    info = {}
    try:
        resolved = _resolve_backend(backend, q, level, deterministic, visibility, info)
    except Exception as exc:
        record.update(resolved=None, capability=list(info.get('capability') or (0, 0)),
                      native_mask=info.get('native_mask', 0),
                      probe_error=info.get('probe_error'),
                      refused='%s: %s' % (type(exc).__name__, exc))
        _record_route(record)
        raise
    record.update(resolved=resolved, capability=list(info.get('capability') or (0, 0)),
                  native_mask=info.get('native_mask', 0),
                  probe_error=info.get('probe_error'), refused=None)
    _record_route(record)
    return resolved


def required_precision_ops(deterministic=False):
    """The exact op set that the (deterministic) branch of `backward` calls.

    A symbol that the branch never executes must not be required, and a symbol
    the branch does execute must be. Keep this in step with `backward`: the
    deterministic branch replaces the main backward and the probability repair
    by the ordered backward and the deterministic repair, while the row scaling,
    the zero-sum repair and the fallback dV apply to both.
    """
    names = list(_PRECISION_COMMON_OPS)
    names.append('_attention_precision_ordered_backward' if deterministic
                 else '_attention_precision_main_backward')
    names.append('_attention_precision_zero_sum_repair')
    names.append('_attention_precision_deterministic_repair' if deterministic
                 else '_attention_precision_probability_repair')
    names.append('_attention_precision_scale_do')
    names.append('_attention_precision_fallback_dv')
    return tuple(names)


# Why the extension or its capability probe last failed, for the error text.
_EXTENSION_ERROR = None
_PROBE_ERROR = None


def _extension():
    """The linked extension's ops module, or None when it is not usable.

    Any exception counts: a missing module is ImportError, a present module
    whose shared libraries do not load is OSError, and a build against another
    CUDA/ABI raises RuntimeError.  All of them mean "no implementation".
    """
    global _EXTENSION_ERROR
    try:
        import xattn_cuda
    except Exception as exc:  # noqa: BLE001 - every failure means the same thing
        _EXTENSION_ERROR = '%s: %s' % (type(exc).__name__, exc)
        return None
    ops = getattr(xattn_cuda, 'ops', None)
    if ops is None:
        _EXTENSION_ERROR = 'the imported xattn_cuda module has no .ops submodule'
        return None
    _EXTENSION_ERROR = None
    return ops


def _ops(level=None, deterministic=False, device=None):
    ops = _extension()
    if ops is None:
        raise RuntimeError(
            'high_precision_level=1 requires the compiled xattn CUDA extension and importing it '
            'failed (%s)%s; nothing was run and no fallback was substituted'
            % (_EXTENSION_ERROR, '' if device is None else ' on device %s' % (device,)))
    names = _LEGACY_REQUIRED_OPS if level is None else required_precision_ops(deterministic)
    missing = [name for name in names if not hasattr(ops, name)]
    if missing:
        raise RuntimeError(
            'high_precision_level=1 requires an xattn CUDA extension whose ops module provides '
            + ', '.join(missing)
            + '; the linked build lacks ' + ('it' if len(missing) == 1 else 'them')
            + ('' if level is None else ', so level %d%s cannot run' % (
                level, ' (deterministic)' if deterministic else ''))
            + ('' if device is None else ' on device %s' % (device,))
            + '. No SM90-named alias or lower precision level is substituted.')
    return ops


def _check_backend_name(backend):
    """Normalize the backend name the way the public API does, or refuse it.

    `_normalize_backend` in `softdelta_attention` strips, lowercases and folds
    '-' to '_', and maps '' to 'auto'; this applies the same spelling rule and
    adds 'native' to the accepted set, so a name that the public API accepts is
    spelled the same way here.  The only policy sentence added is about
    'torch': it is the public API's reference route, not a precision backward.
    """
    if not isinstance(backend, str):
        raise TypeError('advanced precision backend must be a string')
    name = backend.strip().lower().replace('-', '_')
    if name not in _BACKEND_NAMES:
        raise ValueError(
            "advanced precision backend must be '', 'auto', 'native' or 'sm90' (got %r)%s"
            % (backend, "; 'torch' is the reference route of the public API and is not an "
                        'advanced-precision backward backend' if name == 'torch' else ''))
    return 'auto' if name == '' else name


def _native_level_mask(q):
    """Per-bit availability of the native implementation on q's device.

    Bit k set means, on THIS device and THIS build: for k in (0, 1) the basic
    reader level k is linked forward and backward (the Python dispatcher's
    promise), and for k in (2, 3, 4) the advanced-precision level k is linked
    with the exact op set `required_precision_ops(k, deterministic)`.  The mask
    must be device-derived, never a build-time constant, or a Hopper device
    would be told it has native levels it cannot run.  Three different absences
    all give 0 and only one of them is `_PROBE_ERROR`: no probe symbol on the
    extension gives 0 with no reason, an extension that cannot be imported
    gives 0 with its reason in `_EXTENSION_ERROR`, and a probe that raises
    gives 0 with its reason in `_PROBE_ERROR`.
    """
    global _PROBE_ERROR
    ops = _extension()
    probe = getattr(ops, '_native_attention_level_mask', None) if ops is not None else None
    if probe is None:
        _PROBE_ERROR = None
        return 0
    try:
        value = int(probe(q))
    except Exception as exc:  # a capability probe fails closed, never open
        _PROBE_ERROR = '%s: %s' % (type(exc).__name__, exc)
        return 0
    _PROBE_ERROR = None
    return value


def _native_visibility_mask(q):
    """Visibilities (bit 0 full, 1 window, 2 chunk) the native leaf serves.

    A build without the probe implements full-causal visibility only.
    """
    ops = _extension()
    probe = getattr(ops, '_native_precision_visibility_mask', None) if ops is not None else None
    if probe is None:
        return 0x1
    try:
        return int(probe(q))
    except Exception:  # a capability probe fails closed
        return 0


def _no_native_message(q, level, deterministic, native_levels, major, minor):
    ops = _extension()
    if ops is None:
        return ('no advanced-precision backward is available: importing the extension failed (%s), '
                'so no implementation is linked; nothing was run and no fallback was substituted'
                % (_EXTENSION_ERROR,))
    missing = [name for name in required_precision_ops(deterministic) if not hasattr(ops, name)]
    if missing:
        detail = 'missing ops: ' + ', '.join(missing)
    elif _PROBE_ERROR is not None:
        detail = 'the availability probe raised (%s), so nothing is claimed' % _PROBE_ERROR
    else:
        detail = 'the level mask does not set bit %d' % level
    return ('no native advanced-precision backward for level %d%s on %s (sm_%d%d): '
            '_native_attention_level_mask=0x%x, %s. A native build must link the complete component '
            'set of that branch; nothing was run and no SM90-named alias or lower level is '
            'substituted' % (level, ' (deterministic)' if deterministic else '', q.device,
                             major, minor, native_levels, detail))


def _resolve_backend(backend, q, level, deterministic, visibility, info=None):
    """Return 'sm90' or 'native' for this call, or raise saying what is absent.

    Both implementations are registered under the same op names
    (DefineAttentionPrecisionOps is built once, for the SM90 or for the native
    precision source set), so this function decides whether the call may
    proceed and which implementation the caller is claiming; it does not pick a
    symbol.  The name alone never decides: `sm90` is refused off SM90 hardware
    and on a build that advertises native support for this level there,
    `native` is refused unless the build and device link the exact branch, and
    `auto` follows the device class -- the SM90 route on a major-9 device
    whatever the mask says, so that a mask which is a build-time constant can
    never move an SM90 baseline onto a native kernel, and the native route
    everywhere else.

    Classes: ValueError when no implementation of this level exists on this
    device/build (as the seed raised), NotImplementedError when the build
    advertises the level and its symbols disagree with the mask, or when the
    advertised native implementation does not cover the requested visibility.
    """
    global _LAST_MASK, _LAST_CAPABILITY, _LAST_PROBE_ERROR
    _LAST_MASK, _LAST_CAPABILITY, _LAST_PROBE_ERROR = 0, (0, 0), None
    if info is None:
        info = {}
    info.clear()
    info.update(capability=(0, 0), native_mask=0, probe_error=None)
    name = _check_backend_name(backend)
    major, minor = torch.cuda.get_device_capability(q.device)
    native_levels = _native_level_mask(q)
    _LAST_MASK, _LAST_CAPABILITY = native_levels, (major, minor)
    _LAST_PROBE_ERROR = _PROBE_ERROR
    info.update(capability=(major, minor), native_mask=native_levels,
                probe_error=_PROBE_ERROR)
    if name == 'sm90':
        if major != 9:
            raise ValueError(
                "backend='sm90' is SM90-only and device %s is sm_%d%d; it is never an alias for a "
                "native implementation, so pass backend='native' (or 'auto') where a native one "
                "exists" % (q.device, major, minor))
        if native_levels & (1 << level):
            raise ValueError(
                "backend='sm90' on the SM90 device %s, but the loaded extension advertises a native "
                "implementation for level %d there (mask 0x%x); the request and the build disagree "
                "about which implementation would run" % (q.device, level, native_levels))
        return 'sm90'
    if name == 'auto' and major == 9:
        # The device class decides the route, but a build that advertises
        # native support for this level while its linked symbols disagree is
        # an inconsistent build.  It is named here, as the same
        # NotImplementedError the native route raises, instead of surfacing
        # later from the legacy symbol check as a RuntimeError.
        if native_levels & (1 << level):
            ops = _extension()
            missing = [op for op in required_precision_ops(deterministic)
                       if ops is None or not hasattr(ops, op)]
            if missing:
                raise NotImplementedError(
                    'device %s advertises native advanced-precision level %d%s (mask 0x%x) but the '
                    'linked build lacks %s; the availability mask and the symbol table disagree, so '
                    'nothing was run' % (q.device, level,
                                         ' (deterministic)' if deterministic else '',
                                         native_levels, ', '.join(missing)))
        return 'sm90'
    if native_levels & (1 << level):
        ops = _extension()
        missing = [op for op in required_precision_ops(deterministic)
                   if ops is None or not hasattr(ops, op)]
        if missing:
            raise NotImplementedError(
                'device %s advertises native advanced-precision level %d%s (mask 0x%x) but the '
                'linked build lacks %s; the availability mask and the symbol table disagree, so '
                'nothing was run' % (q.device, level,
                                     ' (deterministic)' if deterministic else '',
                                     native_levels, ', '.join(missing)))
        if not (_native_visibility_mask(q) >> visibility) & 1:
            raise NotImplementedError(
                'the native advanced-precision backward of this build does not implement '
                'visibility=%r (implemented visibility mask 0x%x); level %d is not silently '
                'replaced by another schedule' % (visibility, _native_visibility_mask(q), level))
        return 'native'
    raise ValueError(_no_native_message(q, level, deterministic, native_levels, major, minor))


def backward(dy, q, k, v, *, visibility, span=None, scale=None,
             prev_k=None, prev_v=None, q_segment_idx=None, k_segment_idx=None,
             segment_idx=None, bos_mask=None, reset=False, strict=False,
             deterministic=False, backend='auto', return_fp32=False):
    _check_backend_name(backend)
    if not q.is_cuda:
        raise ValueError('high precision requires a CUDA device')
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError('precision readers require rank-four Q/K/V')
    if (prev_k is None) != (prev_v is None):
        raise ValueError('prev_k and prev_v must be provided together')
    if dy.shape != (*q.shape[:-1], v.shape[-1]) or dy.dtype != q.dtype or dy.device != q.device:
        raise ValueError('dO must have the reader output shape and Q dtype/device')
    if visibility not in (0, 1, 2):
        raise ValueError('invalid precision visibility')
    if visibility and (isinstance(span, bool) or not isinstance(span, int) or span < (1 if visibility == 2 else 0)):
        raise ValueError('invalid window/chunk span')
    if prev_k is not None:
        if visibility == 2 and prev_k.shape[1] != span:
            raise ValueError('previous chunk length must equal chunk_size')
        for a, b in ((k, prev_k), (v, prev_v)):
            if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:] or a.dtype != b.dtype or a.device != b.device:
                raise ValueError('previous K/V shapes, dtypes and devices must match current K/V')
    previous = 0 if prev_k is None else prev_k.shape[1]
    kt = k if prev_k is None else torch.cat((prev_k, k), dim=1)
    vt = v if prev_v is None else torch.cat((prev_v, v), dim=1)
    b, nq, h, _ = q.shape
    nk = kt.shape[1]
    if bos_mask is not None:
        if any(x is not None for x in (segment_idx, q_segment_idx, k_segment_idx)):
            raise ValueError('BOS masks and segment indices are mutually exclusive')
        if bos_mask.shape != (b, nk) or bos_mask.dtype != torch.bool or bos_mask.device != q.device:
            raise ValueError('bos_mask must be boolean [B,total_K] on Q device')
        segment_idx = bos_mask.to(torch.int64).cumsum(-1)
    qi, ki = resolve_segment_indices(segment_idx, q_segment_idx, k_segment_idx,
                                    batch_size=b, q_length=nq, k_length=nk)
    validate_segment_indices(qi, ki, batch_size=b, q_length=nq, k_length=nk,
                             device=q.device, api_name='precision backward')
    scale = resolve_scale(q, scale)
    # The kernels use a positive scale so masked -inf logits stay masked.
    # Exact sign/zero transforms preserve the public finite-scale contract.
    original_scale = scale
    if scale < 0:
        q, scale = -q, -scale
    elif scale == 0:
        q, scale = torch.zeros_like(q), 1.0
    if reset and visibility != 2:
        raise ValueError('reset positions apply only to sliding chunks')
    # Every argument is validated before the capability decision, so a call
    # with a bad argument on an unsupported device reports the argument. The
    # resolution then decides which implementation this call may use, or
    # raises naming exactly what is absent, and it leaves a route record
    # either way.
    if q.is_cuda and torch.cuda.get_device_capability(q.device)[0] == 10:
        # Blackwell supports only deterministic attention: the unordered-atomic branch
        # (main backward + probability repair, anchor-scatter zero-sum repair) is never
        # run there, whatever the caller's flag says.
        deterministic = True
    resolved_backend = _resolve_call_backend(backend, q, _NATIVE_REPAIR_BIT, deterministic,
                                             visibility)
    if reset and qi is not None:
        from ._sca_segment_alignment import needs_matched_reset
        if needs_matched_reset(q, qi, ki, True):
            outputs = _matched_backward(dy, q, kt, vt, qi, ki, span, scale, strict, deterministic,
                                        resolved_backend)
            outputs = _restore_scale_gradients(outputs, original_scale)
            return _finish(outputs, q.dtype, previous, return_fp32)
    q, kt, vt, dy = (x.contiguous() for x in (q, kt, vt, dy))
    # A contiguous view can still start at an odd 16-bit storage offset.
    # Statistics uses 16-byte async transfers; repair alignment without casts.
    q, kt, vt, dy = (x.clone() if x.data_ptr() % 16 else x for x in (q, kt, vt, dy))
    qi = None if qi is None else qi.contiguous()
    ki = None if ki is None else ki.contiguous()
    position = torch.arange(nq, device=q.device, dtype=torch.int32)[None].expand(b, -1) + nk - nq
    right = position - int(strict)
    q_positions = None
    if visibility == 0:
        left = torch.zeros_like(position)
    elif visibility == 1:
        left = (position - span).clamp_min(0)
    elif reset and qi is not None:
        # Same formulas as FlashSCABuildDenseResetChunkPositionsKernel. Segment
        # IDs use the existing sorted-run contract, including short Q suffixes.
        qlo = torch.searchsorted(qi, qi)
        qhi = torch.searchsorted(qi, qi, right=True)
        klo = torch.searchsorted(ki, qi)
        khi = torch.searchsorted(ki, qi, right=True)
        local = (torch.arange(nq, device=q.device)[None] - qlo) + (khi - klo) - (qhi - qlo)
        q_positions = local.to(torch.int32).contiguous()
        left = (klo + ((local // span - 1) * span).clamp_min(0)).to(torch.int32)
        right = torch.minimum(klo + local - int(strict), khi - 1).to(torch.int32)
    else:
        left = ((position // span - 1) * span).clamp_min(0)
    exact = None
    if qi is not None and q_positions is None:
        # With nondecreasing key runs (always true for BOS masks) a query's
        # segment support is the key run [first, last]; intersecting it with
        # the visibility bounds makes them exact, so the row scans skip other
        # segments and per-key segment tests. Other layouts keep the
        # visibility bounds and the segment test. No host sync.
        runs = (ki[:, 1:] >= ki[:, :-1]).all(dim=1, keepdim=True)
        first = torch.searchsorted(ki, qi).to(torch.int32)
        last = torch.searchsorted(ki, qi, right=True).to(torch.int32) - 1
        left = torch.where(runs, torch.maximum(left, first), left)
        right = torch.where(runs, torch.minimum(right, last), right)
        exact = runs.to(torch.int32).reshape(b).contiguous()
    left, right = left.contiguous(), right.contiguous()
    ops = _ops(_NATIVE_REPAIR_BIT, deterministic, device=q.device)
    kc, _ = ops._attention_precision_center(kt)
    vc, _ = ops._attention_precision_center(vt)
    dy, inverse_prescale = _prescale_fp16(dy, vc)
    delta, logz, metadata = ops._attention_precision_row_statistics(q, kc, vc, dy, left, right, qi, ki, scale, exact)
    scaled_dy, factors = ops._attention_precision_scale_do(dy)
    metadata[2].copy_(factors)
    # Row statistics of the dS operand: sum (rho), P mass, an unused plane and
    # the anchor probability mass.
    stats = torch.zeros((4, b, h, nq), dtype=torch.float64, device=q.device)
    if deterministic:
        # dS is in unscaled dO units: the row scale cancels in-kernel.
        exponent = _rho_exponent(dy, vc)
        dq, dk, dv = ops._attention_precision_ordered_backward(
            q, kc, vc, scaled_dy, delta, logz, stats, metadata, qi, ki, q_positions,
            scale, visibility, span or 0, strict, reset, exponent)
        _decode_fixed_statistics(stats, exponent)
        # The uniform shares need no atomics; the anchor share left in
        # stats[0] goes through the ordered repair below.
        ops._attention_precision_zero_sum_repair(
            q, kc, stats[0], metadata, logz, ki, ZERO_SUM_WINDOW, False, stats[0], dq, dk, scale)
        # Stable order within each KV head makes anchor scatter reductions
        # independent of CTA scheduling without an N² workspace.
        order = torch.argsort(metadata[3].reshape(b, kt.shape[2], -1), dim=-1, stable=True)
        flags = ops._attention_precision_deterministic_repair(
            q, kc, dy, stats, metadata, order, dq, dk, dv, scale)
    else:
        dq, dk, dv = ops._attention_precision_main_backward(
            q, kc, vc, scaled_dy, delta, logz, stats, metadata, qi, ki, q_positions,
            scale, visibility, span or 0, strict, reset)
        ops._attention_precision_zero_sum_repair(
            q, kc, stats[0], metadata, logz, ki, ZERO_SUM_WINDOW, True,
            torch.empty_like(stats[0]), dq, dk, scale)
        flags = ops._attention_precision_probability_repair(dy, stats, metadata, dv)
    ops._attention_precision_fallback_dv(q, kc, scaled_dy, left, right, logz,
                                       metadata, flags, dv, qi, ki, scale)
    # Output-space subtraction can leave a huge FP32 cancellation residual
    # when all visible keys coincide and dO/V have a very large range.
    # Equality runs prove the exact dQ=0 invariant in O(ND), without an
    # extra attention contraction or a guessed small-gradient threshold.
    key_runs = ops._attention_precision_key_runs(kt)
    ops._attention_precision_enforce_constant_keys(key_runs, metadata, dq)
    if inverse_prescale is not None:
        for x in (dq, dk, dv):
            x.mul_(inverse_prescale)
    outputs = _restore_scale_gradients((dq, dk, dv), original_scale)
    return _finish(outputs, q.dtype, previous, return_fp32)


def _power_of_two(exponent):
    """Exact FP64 2**exponent for int64 exponents in the normal range."""
    return ((exponent + 1023) << 52).view(torch.float64)


def _rho_exponent(dy, vc):
    """Per-row int32 [B,H,N] exponent e with sum_j |dS_ij| < 2^(e-3).

    |dS_ij| <= P_ij (|dP_ij - dP_ia| + |D'_i|) <= 4 P_ij |dO_i| max_j |V_j|, and
    2^(e-5) exceeds |dO_i| max|V_j|, so the deterministic kernel's int64 sums on
    the 2^(e-62) grid cannot overflow. Finite
    16-bit inputs give |e| < 300; the clamp keeps both grids normal FP64.
    """
    g = torch.linalg.vector_norm(dy, dim=-1, dtype=torch.float32).double()
    w = torch.linalg.vector_norm(vc, dim=-1, dtype=torch.float32).amax().double() if vc.numel() else 0.0
    bound = g * w
    e = torch.frexp(bound)[1].long() + 5
    e = torch.where(torch.isfinite(bound), e, torch.full_like(e, 300)).clamp(-300, 300)
    return e.to(torch.int32).transpose(1, 2).contiguous()


def _decode_fixed_statistics(stats, exponent):
    """Convert the deterministic kernel's int64 planes back to FP64 in place."""
    fixed = stats.view(torch.int64)
    rho = fixed[0].double() * _power_of_two(exponent.long() - 62)
    stats[1].copy_(fixed[1].double().mul_(2.0 ** -60))
    stats[3].copy_(fixed[3].double().mul_(2.0 ** -60))
    stats[0].copy_(rho)


def _prescale_fp16(dy, vc):
    """Exact power-of-two dO scaling for FP16 operands.

    Gradients are linear in dO, so dO * 2^e followed by output * 2^-e is
    exact. The scale keeps dS = P(dP - D), bounded by 2 max|dO_i| max|Vc_j|,
    near 2^13 so small entries stay out of FP16 subnormals; it only scales
    down when that bound risks FP16 overflow. BF16 shares FP32's exponent
    range and is returned unchanged. The scale stays on the device.
    """
    if dy.dtype != torch.float16 or not dy.numel() or not vc.numel():
        return dy, None
    g = torch.linalg.vector_norm(dy, dim=-1, dtype=torch.float32).amax()
    w = torch.linalg.vector_norm(vc, dim=-1, dtype=torch.float32).amax()
    bound = 2 * g * w
    e = torch.floor(torch.log2(_PRESCALE_TARGET / bound))
    # dO itself must stay finite in FP16.
    e = torch.minimum(e, torch.floor(torch.log2(30000.0 / dy.abs().amax().float())))
    e = torch.where(bound > 32768.0, e, e.clamp_min(0))
    e = torch.where(torch.isfinite(e), e, torch.zeros_like(e)).clamp(-60, 60).to(torch.int32)
    # Build 2^e and 2^-e exactly from FP32 exponent bits.
    factor = ((e + 127) << 23).view(torch.float32)
    # Scale in FP32: an FP16 product would round factors above 2^15 to inf.
    return (dy.float() * factor).to(dy.dtype), ((127 - e) << 23).view(torch.float32)


def _restore_scale_gradients(outputs, original_scale):
    dq, dk, dv = outputs
    if original_scale < 0:
        dq = -dq
    elif original_scale == 0:
        dq, dk = torch.zeros_like(dq), torch.zeros_like(dk)
    return dq, dk, dv


def _finish(outputs, dtype, previous, keep_fp32):
    dq, dk, dv = outputs
    if not keep_fp32:
        dq, dk, dv = (x.to(dtype) for x in (dq, dk, dv))
    if not previous:
        return dq, dk, dv, None, None
    return (dq, dk[:, previous:].contiguous(), dv[:, previous:].contiguous(),
            dk[:, :previous].contiguous(), dv[:, :previous].contiguous())


def _matched_backward(dy, q, k, v, qi, ki, chunk, scale, strict, deterministic,
                      backend='auto'):
    # Irregular independently aligned reset runs use the same exact packing
    # plan as the existing API. No N² mask is allocated. The common aligned
    # BOS path above remains entirely GPU-driven.
    from ._sca_segment_alignment import _build_plan
    plan = _build_plan(qi, ki)
    qp = q.flatten(0, 1).index_select(0, plan.q_indices)
    kp = k.flatten(0, 1).index_select(0, plan.k_indices)
    vp = v.flatten(0, 1).index_select(0, plan.k_indices)
    gp = dy.flatten(0, 1).index_select(0, plan.q_indices)
    oq, ok, ov = (torch.zeros_like(x, dtype=torch.float32) for x in (qp, kp, vp))
    cq, ck = plan.cu_q.cpu().tolist(), plan.cu_k.cpu().tolist()
    for index in range(len(cq) - 1):
        qa, qb, ka, kb = cq[index], cq[index + 1], ck[index], ck[index + 1]
        a, b, c, _, _ = backward(gp[qa:qb][None], qp[qa:qb][None], kp[ka:kb][None], vp[ka:kb][None],
                                visibility=2, span=chunk, scale=scale,
                                strict=strict, deterministic=deterministic, backend=backend,
                                return_fp32=True)
        oq[qa:qb], ok[ka:kb], ov[ka:kb] = a[0], b[0], c[0]
    result = []
    for tensor, original, indices in ((oq, q, plan.q_indices), (ok, k, plan.k_indices), (ov, v, plan.k_indices)):
        output = torch.zeros_like(original, dtype=torch.float32)
        output.flatten(0, 1).index_copy_(0, indices, tensor)
        result.append(output)
    return tuple(result)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, prev_k, prev_v, visibility, options):
        # Call the existing standalone FWD so all routing and return semantics
        # remain shared with the public API. Only BWD algorithm selection differs.
        from .causal_flash_attn import causal_flash_attn_fwd
        from .sliding_window_attention import flash_swa_fwd
        from .sliding_chunk_attention import flash_sca_fwd
        functions = (causal_flash_attn_fwd, flash_swa_fwd, flash_sca_fwd)
        forward_options = dict(options)
        forward_options.pop('deterministic', None)
        if visibility:
            forward_options.update(prev_k=prev_k, prev_v=prev_v)
        # The backward recomputes the row statistics: neither y nor the LSE is saved.
        y, _ = functions[visibility](q, k, v, **forward_options)
        ctx.save_for_backward(q, k, v, prev_k, prev_v)
        ctx.visibility, ctx.options = visibility, options
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, prev_k, prev_v = ctx.saved_tensors
        opts = dict(ctx.options)
        span = opts.pop('window_size', opts.pop('chunk_size', None))
        opts.pop('high_precision_level')
        reset = opts.pop('reset_chunk_pos_per_seq', False)
        opts.pop('attn_method', None)
        grads = backward(dy, q, k, v, visibility=ctx.visibility, span=span,
                         prev_k=prev_k, prev_v=prev_v, reset=reset, **opts)
        return (*grads, None, None)


def apply(q, k, v, *, visibility, prev_k=None, prev_v=None, **options):
    # Fail before doing a forward if the installed binary lacks the branch the
    # backward will execute.
    _ops(_NATIVE_REPAIR_BIT, bool(options.get('deterministic', False)), device=q.device)
    return _Attention.apply(q, k, v, prev_k, prev_v, visibility, options)
