# Author: Shicheng Wen

"""Native Blackwell (sm_103a) route of the public FlashSCA entry points and of
the private ``_flash_sca_fwd`` / ``_flash_sca_bwd`` that SoftDelta is composed
from (strict-past reads, FP32 dK/dV).

Selection is a capability decision. On a CUDA device of compute capability
10.x, ``backend`` ``''``/``'auto'``/``'native'`` resolve to the native kernels
``xattn_cuda.ops._flash_sca_native_*``. When the build does not link them, or
the device cannot run the sm_103a cubin, the call raises: nothing silently
falls back to the generic SM80 kernels or to a PyTorch reference. ``'cuda'``
keeps naming the generic CUDA kernels and ``'sm90'`` keeps refusing a device
that is not SM90.

Visibility metadata is reduced here, on the device and without host
synchronization, to what the kernels evaluate (see
``csrc/flash_sca/native/params.h``): per query row an inclusive key interval
``[lo, hi]`` on the total key axis ``[prev_k ; k]`` plus, only where a batch
row's key segment ids are not sorted, a per-key segment equality test.

* Dense metadata: analytic chunk bounds, nothing is materialized.
* Segment ids or a BOS mask: the chunk bounds intersected with the key run of
  the row's segment (exact when the key ids of the batch row are sorted, which
  a BOS mask always is); unsorted rows keep the chunk bounds and the per-key
  test.
* ``reset_chunk_pos_per_seq`` with segment ids: chunk positions restart in
  every segment. Each row's interval lies inside the key run of its own
  segment, so a segment id must form one contiguous run in Q and in K **unless
  no row can use it** — a Q id with no matching key, or a K id that no query
  carries, is dead by construction (zero output, exactly zero gradient) and is
  accepted in any number of runs. A scatter that some row can use raises
  ``ValueError`` instead of producing an interval over foreign keys.
"""

from dataclasses import dataclass
from typing import Optional

import os
import torch
from torch import Tensor

from ._attention_core import _run_coordinates
from . import _native_dispatch
from ._native_dispatch import record_native_leaf
from ._segment_metadata import resolve_segment_indices, validate_segment_indices

_NATIVE_NAMES = ("", "auto", "native")
_BLACKWELL_MAJOR = 10


def _extension_ops():
    try:
        import xattn_cuda
    except ImportError:
        return None
    return getattr(xattn_cuda, "ops", None)


def _normalize(backend) -> str:
    if not isinstance(backend, str):
        raise TypeError("FlashSCA backend must be a string")
    return backend.strip().lower().replace("-", "_")


def selected(q, backend, api_name: str) -> bool:
    """Whether this call runs on the native Blackwell kernels.

    Raises when the native route is the only one the device and backend name
    allow but the build or device cannot serve it.
    """
    if not isinstance(q, Tensor) or not q.is_cuda:
        return False
    name = _normalize(backend)
    major, minor = torch.cuda.get_device_capability(q.device)
    if name == "native" and major < _BLACKWELL_MAJOR:
        raise RuntimeError(
            f"{api_name} backend='native' requires a Blackwell GPU; device "
            f"{q.device} has compute capability {major}.{minor}"
        )
    if major < _BLACKWELL_MAJOR:
        return False
    if name not in _NATIVE_NAMES:
        # Strict mode: on capability 10.x only the native kernels or the explicitly requested PyTorch reference
        # ('torch') run; any other backend would be a non-native route. XLLM_ALLOW_FALLBACK=1 allows it.
        if major == _BLACKWELL_MAJOR and name != "torch" and os.environ.get("XLLM_ALLOW_FALLBACK", "") in ("", "0"):
            raise RuntimeError(
                f"{api_name} backend={backend!r} would take a non-native sliding-chunk route on {q.device} "
                f"(compute capability {major}.{minor}); q: {tuple(q.shape)} {q.dtype}. Strict mode raises instead "
                "of running the fallback; set XLLM_ALLOW_FALLBACK=1 to allow it."
            )
        return False
    ops = _extension_ops()
    if ops is None or not hasattr(ops, "_flash_sca_native_fwd"):
        raise RuntimeError(
            f"{api_name} on {q.device} (compute capability {major}.{minor}) needs "
            "the native Blackwell FlashSCA kernels, and this xattn build does not "
            "link them; nothing was run and no other implementation was "
            "substituted. Build xattn for sm_103a (XATTN_TARGET_SM=103a or auto)."
        )
    if not ops._flash_sca_native_available(q):
        raise RuntimeError(
            f"{api_name}: the native FlashSCA kernels are built for sm_103a and "
            f"cannot run on {q.device} (compute capability {major}.{minor}); "
            "nothing was run"
        )
    return True


def launch_counts():
    """(forward, backward) native kernel launches in this process."""
    ops = _extension_ops()
    if ops is None or not hasattr(ops, "_flash_sca_native_launch_counts"):
        return (0, 0)
    return tuple(ops._flash_sca_native_launch_counts())


# Rows up to this many keys take the mask-side metadata route (the extension's per-chunk segment
# table is sized by the row length); above it the cumsum and the ids route below are used. Both
# routes compute the same integers, so the switch changes the cost only.
_MASK_MAX_KEYS = 131072


@dataclass(frozen=True)
class Visibility:
    q_segment_idx: Optional[Tensor] = None
    k_segment_idx: Optional[Tensor] = None
    segment_exact: Optional[Tensor] = None
    row_lo: Optional[Tensor] = None
    row_hi: Optional[Tensor] = None
    # The BOS mask itself, when the extension can derive the bounds from it in the call that
    # needs them (one metadata launch, no cumsum and no ids plane): the metadata boundary then
    # costs one Python->C++ call instead of a chain of launches. Rows above kMaskMaxKeys keep
    # the ids route below, which computes the same integers.
    bos_mask: Optional[Tensor] = None


def _validate_bos(bos_mask: Tensor, q: Tensor, total: int) -> None:
    if (
        not isinstance(bos_mask, Tensor)
        or bos_mask.dtype != torch.bool
        or bos_mask.device != q.device
        or tuple(bos_mask.shape) != (q.shape[0], total)
    ):
        raise ValueError(
            "bos_mask must be a bool tensor on the Q/K/V device with shape "
            f"[{q.shape[0]}, {total}]"
        )


@torch.no_grad()
def _scattered_ids(ids: Tensor):
    """Per batch row, the ids that occupy two or more runs in this plane."""
    found = []
    for b in range(ids.shape[0]):
        row = ids[b]
        if row.numel() <= 1:
            found.append(row.new_zeros(0))
            continue
        starts = torch.ones_like(row, dtype=torch.bool)
        starts[1:] = row[1:] != row[:-1]
        uniq, inverse = torch.unique(row, return_inverse=True)
        runs = torch.zeros(uniq.numel(), dtype=torch.int64, device=row.device)
        runs.index_add_(0, inverse, starts.to(torch.int64))
        found.append(uniq[runs > 1])
    return found


@torch.no_grad()
def _require_one_run_per_segment(q_ids: Tensor, k_ids: Tensor, api_name: str) -> None:
    """Refuse ids that are not one contiguous run, on the reset route only.

    The reset form measures positions from each segment's start and length
    (``p = q - q_start(s) + len_k(s) - len_q(s)``, ``j = k - k_start(s)``), and
    the native route hands the kernels a single ``[row_lo, row_hi]`` interval
    with no per-key segment test. Both presuppose that every id occupies one
    contiguous run: with two runs the interval covers keys of another segment
    and misses some of its own, and nothing downstream can tell. The reference
    answers such a call with run-local coordinates instead, so it is not a
    refused-by-nobody case but a silently different one; refuse it here rather
    than answer it two ways.

    An id that no row can use is exempt on both planes, symmetrically: a
    K id with no query carrying it never has its interval consulted (those
    rows are dead, so output and gradient are exactly zero), and a Q id with
    no key to match has ``count == 0`` and an empty interval by construction.
    Both must keep the documented zero-output semantics instead of becoming a
    validation error, so neither is refused.
    """
    scattered_k = _scattered_ids(k_ids)
    scattered_q = _scattered_ids(q_ids)
    for b in range(k_ids.shape[0]):
        if scattered_k[b].numel() and bool(
                torch.isin(scattered_k[b], q_ids[b]).any()):
            raise ValueError(
                f"{api_name} reset_chunk_pos_per_seq=True requires every segment "
                f"id to occupy one contiguous run in K; got an id split across "
                f"several runs (pass reset_chunk_pos_per_seq=False)"
            )
        if scattered_q[b].numel() and bool(
                torch.isin(scattered_q[b], k_ids[b]).any()):
            raise ValueError(
                f"{api_name} reset_chunk_pos_per_seq=True requires every segment "
                f"id to occupy one contiguous run in Q; got an id split across "
                f"several runs (pass reset_chunk_pos_per_seq=False)"
            )


@torch.no_grad()
def visibility(q: Tensor, total: int, chunk_size: int, *, q_segment_idx=None,
               k_segment_idx=None, segment_idx=None, bos_mask=None,
               reset: bool = False, strict: bool = False,
               bos_in_call: bool = True) -> Visibility:
    """Reduce public metadata to the kernels' row bounds and segment planes.

    A BOS mask has two device routes, which give the same integers: with ``bos_in_call``
    (the default, the forward's route) the mask itself is the metadata boundary and the
    extension call derives the bounds from it (``Visibility.bos_mask``); without it (the
    public backward's route) the bounds come from one ``_flash_sca_row_bounds`` launch here.
    """
    batch, length = q.shape[0], q.shape[1]
    if bos_mask is not None:
        if any(x is not None for x in (segment_idx, q_segment_idx, k_segment_idx)):
            raise ValueError("BOS masks and segment indices are mutually exclusive")
        _validate_bos(bos_mask, q, total)
        if bos_in_call and not reset and total <= _MASK_MAX_KEYS:
            # The bounds come from the mask inside the forward/backward call itself, so the
            # metadata boundary is one Python->C++ call and one launch, with no cumsum, no
            # ids plane and no bool->int64 cast to build (Visibility.bos_mask).
            return Visibility(bos_mask=bos_mask.contiguous())
        # One launch on the device for the whole reduction: the segment start and
        # end of every query row are the last BOS at or before it and the first BOS
        # after it, so the row bounds follow from the mask alone. The same values
        # in the same dtype as the tensor pipeline below, which stays as the
        # fallback for a device or build that cannot run the native op.
        ops = _extension_ops()
        if (not bos_in_call and q.is_cuda and ops is not None
                and hasattr(ops, "_flash_sca_row_bounds")):
            if ops._flash_sca_native_available(q):
                lo, hi = ops._flash_sca_row_bounds(bos_mask, length, total, chunk_size,
                                                   bool(strict), bool(reset))
                return Visibility(row_lo=lo, row_hi=hi)
        k_ids = bos_mask.cumsum(dim=1, dtype=torch.int64)
        q_ids = k_ids[:, total - length:]
        sorted_ids = True
    else:
        q_ids, k_ids = resolve_segment_indices(
            segment_idx, q_segment_idx, k_segment_idx,
            batch_size=batch, q_length=length, k_length=total,
        )
        validate_segment_indices(
            q_ids, k_ids, batch_size=batch, q_length=length, k_length=total,
            device=q.device, api_name="flash_sca",
        )
        sorted_ids = False
    if q_ids is None:
        return Visibility()
    q_ids = q_ids.contiguous()
    k_ids = k_ids.contiguous()
    device = q.device
    if sorted_ids and not reset:
        # One native launch for exactly the sorted branch below (xattn_cuda's
        # _flash_sca_native_row_bounds): the ids are nondecreasing per row by construction
        # (a BOS mask's cumsum), and every value is an integer, so this route and the Python
        # chain agree bit for bit. A build without the op keeps the chain below. Reached only
        # above _MASK_MAX_KEYS: shorter rows take the mask route further up.
        op = getattr(_extension_ops(), "_flash_sca_native_row_bounds", None)
        if op is not None:
            lo32, hi32 = op(k_ids, int(length), int(total), int(chunk_size), bool(strict))
            return Visibility(row_lo=lo32, row_hi=hi32)
    rows = torch.arange(length, device=device, dtype=torch.int64)[None, :]

    if reset:
        # Positions restart in every segment (one contiguous run in Q and K).
        # BOS ids come from a cumsum, so they are nondecreasing and one run by
        # construction; ids handed in by the caller are checked.
        if not sorted_ids:
            _require_one_run_per_segment(q_ids, k_ids, "flash_sca")
        if sorted_ids:
            order = None
            sorted_k = k_ids
        else:
            sorted_k, order = torch.sort(k_ids, dim=1, stable=True)
        first = torch.searchsorted(sorted_k, q_ids)
        count = torch.searchsorted(sorted_k, q_ids, right=True) - first
        clamped = first.clamp(max=total - 1)
        k_start = clamped if order is None else order.gather(1, clamped)
        q_local, q_run = _run_coordinates(q_ids)
        local = q_local + count - q_run
        lower = (torch.div(local, chunk_size, rounding_mode="floor") - 1) * chunk_size
        lo = k_start + lower.clamp_min(0)
        hi = torch.minimum(k_start + local - int(strict), k_start + count - 1)
        lo = torch.where(count > 0, lo, torch.ones_like(lo))
        hi = torch.where(count > 0, hi, torch.zeros_like(hi))
        return Visibility(row_lo=lo.to(torch.int32).contiguous(),
                          row_hi=hi.to(torch.int32).contiguous())

    position = rows + (total - length)
    lo = ((torch.div(position, chunk_size, rounding_mode="floor") - 1) * chunk_size).clamp_min(0)
    hi = position - int(strict)
    lo = lo.expand(batch, length)
    hi = hi.expand(batch, length)
    if sorted_ids:
        exact = None
    else:
        exact = (k_ids[:, 1:] >= k_ids[:, :-1]).all(dim=1) if total > 1 else torch.ones(
            batch, dtype=torch.bool, device=device)
    first = torch.searchsorted(k_ids, q_ids)
    last = torch.searchsorted(k_ids, q_ids, right=True) - 1
    run_lo = torch.maximum(lo, first)
    run_hi = torch.minimum(hi, last)
    if exact is None:
        # Sorted ids: the run intersection is the whole visibility rule.
        return Visibility(row_lo=run_lo.to(torch.int32).contiguous(),
                          row_hi=run_hi.to(torch.int32).contiguous())
    keep = exact[:, None]
    return Visibility(
        q_segment_idx=q_ids,
        k_segment_idx=k_ids,
        segment_exact=exact.to(torch.int32).contiguous(),
        row_lo=torch.where(keep, run_lo, lo).to(torch.int32).contiguous(),
        row_hi=torch.where(keep, run_hi, hi).to(torch.int32).contiguous(),
    )


def _check_dimensions(q: Tensor, v: Tensor, api_name: str) -> None:
    dim, width = q.shape[-1], v.shape[-1]
    if not (0 < dim <= 256 and 0 < width <= 256 and dim % 8 == 0 and width % 8 == 0):
        raise ValueError(
            f"{api_name} requires Q/K and V dimensions to be multiples of 8 in "
            f"[8, 256]; got D={dim}, V={width}"
        )


def _validate(q, k, v, prev_k, prev_v, chunk_size, attn_method, api_name):
    from . import sliding_chunk_attention as sca
    sca._check_attn_method_supported(attn_method)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"{api_name} chunk_size must be a positive integer")
    if (prev_k is None) != (prev_v is None):
        raise ValueError("prev_k and prev_v must be provided together")
    sca._validate_qkv_heads(q, k, v, packed=False, api_name=api_name)
    sca._validate_prev_kv_heads(q, k, v, prev_k, prev_v, chunk_size, api_name=api_name)
    _check_dimensions(q, v, api_name)
    return k.shape[1] + (0 if prev_k is None else prev_k.shape[1])


# These trace events identify the extension route before it is called, not the
# device body selected inside it. Native-product evidence comes from the C++
# per-phase dispatch tags after a successful launch.
def _forward(q, k, v, prev_k, prev_v, chunk_size, scale, meta, strict, level):
    record_native_leaf("sca_fwd", q, level)
    # Level 1 is the subtract-before-scale softmax; no FP32 output state is
    # written (the level-1 backward recomputes its row statistics).
    return _extension_ops()._flash_sca_native_fwd(
        q, k, v, prev_k, prev_v, chunk_size, scale,
        meta.q_segment_idx, meta.k_segment_idx, meta.segment_exact,
        meta.row_lo, meta.row_hi, bool(strict), level >= 1, None,
        bos_mask=meta.bos_mask,
    )


def _backward(dy, q, k, v, prev_k, prev_v, y, lse, chunk_size, scale, meta, strict,
              deterministic, fp32_kv_grads=False):
    record_native_leaf("sca_bwd", q, 0)
    return _extension_ops()._flash_sca_native_bwd(
        dy, q, k, v, prev_k, prev_v, y, lse, chunk_size, scale,
        meta.q_segment_idx, meta.k_segment_idx, meta.segment_exact,
        meta.row_lo, meta.row_hi, bool(strict), bool(deterministic), bool(fp32_kv_grads),
        bos_mask=meta.bos_mask,
    )


class _NativeFlashSCA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, prev_k, prev_v, chunk_size, scale, meta, deterministic):
        y, lse = _forward(q, k, v, prev_k, prev_v, chunk_size, scale, meta, False, 0)
        ctx.save_for_backward(q, k, v, prev_k, prev_v, y, lse)
        ctx.meta, ctx.chunk_size, ctx.scale = meta, chunk_size, scale
        ctx.deterministic = deterministic
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, prev_k, prev_v, y, lse = ctx.saved_tensors
        dq, dk, dv, dpk, dpv = _backward(
            dy, q, k, v, prev_k, prev_v, y, lse, ctx.chunk_size, ctx.scale,
            ctx.meta, False, ctx.deterministic,
        )
        return dq, dk, dv, dpk, dpv, None, None, None, None


def _scale(q, scale):
    return 1.0 / q.shape[-1] ** 0.5 if scale is None else float(scale)


def flash_sca(q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx, *,
              segment_idx, bos_mask, reset_chunk_pos_per_seq, attn_method, deterministic):
    """Autograd FlashSCA on the native kernels at level 0.

    Level 1 never reaches this function: the public entry sends it to the
    precision-repair autograd function, whose forward is :func:`flash_sca_fwd`
    below and whose backward is the precision backend.
    """
    total = _validate(q, k, v, prev_k, prev_v, chunk_size, attn_method, "flash_sca")
    meta = visibility(q, total, chunk_size, q_segment_idx=q_segment_idx,
                      k_segment_idx=k_segment_idx, segment_idx=segment_idx,
                      bos_mask=bos_mask, reset=reset_chunk_pos_per_seq)
    return _NativeFlashSCA.apply(q, k, v, prev_k, prev_v, chunk_size, _scale(q, scale), meta,
                                 deterministic)


def _mask_forward_route(q, k, v, prev_k, prev_v, chunk_size, bos_mask, q_segment_idx,
                        k_segment_idx, segment_idx, reset_chunk_pos_per_seq, attn_method,
                        high_precision_level) -> bool:
    """Whether a forward call is unambiguously the BOS-mask level-0 training route.

    Pure Python reads of the arguments: no tensor operation, no allocation, no
    synchronisation. Every operand test ``_validate`` and the routes below make is
    re-checked here, so a call this admits is one the full path would have run to
    completion, and a call it refuses (an error, another route, another level) takes
    the full path and main's own answer or error.
    """
    if (bos_mask is None or high_precision_level != 0 or reset_chunk_pos_per_seq
            or q_segment_idx is not None or k_segment_idx is not None
            or segment_idx is not None or attn_method != "default"):
        return False
    if type(chunk_size) is not int or chunk_size <= 0:
        return False
    if (bos_mask.dtype is not torch.bool or bos_mask.dim() != 2
            or bos_mask.device != q.device or not bos_mask.is_contiguous()):
        return False
    if (q.dim() != 4 or k.dim() != 4 or v.dim() != 4
            or q.device != k.device or q.device != v.device
            or q.dtype != k.dtype or q.dtype != v.dtype
            or (q.dtype is not torch.float16 and q.dtype is not torch.bfloat16)):
        return False
    if (q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]
            or not 0 < q.shape[1] <= k.shape[1]):
        return False
    heads_q, heads_kv = q.shape[2], k.shape[2]
    if (heads_kv != v.shape[2] or heads_q <= 0 or heads_kv <= 0
            or heads_q % heads_kv != 0 or q.shape[3] != k.shape[3]):
        return False
    dim, width = q.shape[3], v.shape[3]
    if not (0 < dim <= 256 and 0 < width <= 256 and dim % 8 == 0 and width % 8 == 0):
        return False
    total = k.shape[1]
    if prev_k is not None or prev_v is not None:
        if prev_k is None or prev_v is None:
            return False
        total += prev_k.shape[1]
        if (prev_k.dim() != 4 or prev_v.dim() != 4
                or prev_k.device != q.device or prev_v.device != q.device
                or prev_k.dtype != q.dtype or prev_v.dtype != q.dtype
                or prev_k.shape[0] != q.shape[0] or prev_v.shape[0] != q.shape[0]
                or prev_k.shape[1] != chunk_size or prev_v.shape[1] != chunk_size
                or prev_k.shape[2:] != k.shape[2:] or prev_v.shape[2:] != v.shape[2:]):
            return False
    return (bos_mask.shape[0] == q.shape[0] and bos_mask.shape[1] == total
            and total <= _MASK_MAX_KEYS)


# The BOS-mask level-0 training route, memoised on the call's metadata. The key -
# every value the public entry (flash_sca_fwd in sliding_chunk_attention.py: the precision-level,
# scale and dimension checks, the non-positive-scale branch, `selected`) and _mask_forward_route
# read: the operands' exact type, sizes, dtypes and devices, the mask's contiguity, the absence
# of segment arguments, and the scalar arguments with their exact types - is built and checked by
# the extension inside the one pybind crossing (host.cc, ForwardFast). A key is learned only by
# a call the full path ran to the mask route's extension call (flash_sca_fwd below), so a hit is
# a call the full path would run there again; a miss returns None and takes the full path.
# Nothing that depends on tensor contents or data pointers is decided by the key: the extension
# validates, allocates and prepares every call. Bounded (64 keys, cleared whole when full).
_FAST_OPS = None


def _fast_ops():
    """The extension's ops when it has the repeat-call route, else None."""
    global _FAST_OPS
    if _FAST_OPS is None:
        ops = _extension_ops()
        if ops is not None and hasattr(ops, "_flash_sca_native_fwd_fast"):
            _FAST_OPS = ops
    return _FAST_OPS


def mask_forward_fast(q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
                      segment_idx, bos_mask, backend, reset_chunk_pos_per_seq, attn_method,
                      high_precision_level):
    """``(y, lse)`` of a memoised call, or None (the caller takes the full path).

    The route records are the full path's: ``record_route("native")`` and the native-leaf
    record, before the extension call can fail. With route tracing off both record nothing, so
    the key test and the call are one crossing; with it on, the hit test comes first."""
    ops = _FAST_OPS or _fast_ops()
    if ops is None:
        return None
    if not _native_dispatch._TRACE_ENABLED:
        return ops._flash_sca_native_fwd_fast(
            q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
            segment_idx, bos_mask, backend, reset_chunk_pos_per_seq, attn_method,
            high_precision_level)
    if not ops._flash_sca_native_fwd_fast_hit(
            q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
            segment_idx, bos_mask, backend, reset_chunk_pos_per_seq, attn_method,
            high_precision_level):
        return None
    _native_dispatch.record_route("native")
    record_native_leaf("sca_fwd", q, 0)
    with torch.no_grad():
        return ops._flash_sca_native_fwd_mask(
            q, k, v, prev_k, prev_v, chunk_size, _scale(q, scale), bos_mask)


def fast_keys_clear():
    """Forget every learned repeat-call key (tests)."""
    ops = _fast_ops()
    if ops is not None:
        ops._flash_sca_native_fwd_fast_clear()


def fast_keys_size():
    """How many repeat-call keys are learned (tests)."""
    ops = _fast_ops()
    return 0 if ops is None else ops._flash_sca_native_fwd_fast_size()


@torch.no_grad()
def flash_sca_fwd(q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx, *,
                  segment_idx, bos_mask, reset_chunk_pos_per_seq, attn_method,
                  high_precision_level, fast_backend=None):
    """Standalone forward: ``(y, lse)``; level 1 runs the high-precision softmax.

    ``fast_backend`` is the public entry's ``backend`` argument when it passed its checks for
    this call (mask_forward_fast's key needs it); the call's key is learned once it takes the
    mask route."""
    if _mask_forward_route(q, k, v, prev_k, prev_v, chunk_size, bos_mask, q_segment_idx,
                           k_segment_idx, segment_idx, reset_chunk_pos_per_seq, attn_method,
                           high_precision_level):
        if fast_backend is not None:
            ops = _fast_ops()
            if ops is not None:
                ops._flash_sca_native_fwd_fast_learn(
                    q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
                    segment_idx, bos_mask, fast_backend, reset_chunk_pos_per_seq, attn_method,
                    high_precision_level)
        # The mask route's metadata boundary is the mask itself: the extension derives the row
        # bounds from it inside the call, so this is the one Python->C++ crossing of the call.
        # The native-leaf record is the one _forward makes, before the entry can fail.
        record_native_leaf("sca_fwd", q, 0)
        return _extension_ops()._flash_sca_native_fwd_mask(
            q, k, v, prev_k, prev_v, chunk_size, _scale(q, scale), bos_mask)
    total = _validate(q, k, v, prev_k, prev_v, chunk_size, attn_method, "flash_sca_fwd")
    meta = visibility(q, total, chunk_size, q_segment_idx=q_segment_idx,
                      k_segment_idx=k_segment_idx, segment_idx=segment_idx,
                      bos_mask=bos_mask, reset=reset_chunk_pos_per_seq)
    return _forward(q, k, v, prev_k, prev_v, chunk_size, _scale(q, scale), meta, False,
                    high_precision_level)


@torch.no_grad()
def flash_sca_bwd(y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v, q_segment_idx,
                  k_segment_idx, *, segment_idx, bos_mask, reset_chunk_pos_per_seq,
                  attn_method, deterministic):
    """Standalone level-0 backward: ``(dq, dk, dv, dprev_k, dprev_v)``.

    B300 supports only deterministic SCA: the native backward always takes the
    fixed-order schedule (no FP32 atomic dQ), so ``deterministic`` changes
    nothing and every call repeats bit-exactly. The argument is kept for API
    compatibility.
    """
    total = _validate(q, k, v, prev_k, prev_v, chunk_size, attn_method, "flash_sca_bwd")
    expected = (*q.shape[:-1], v.shape[-1])
    if tuple(y_grad.shape) != expected or y_grad.dtype != q.dtype or y_grad.device != q.device:
        raise ValueError("flash_sca_bwd y_grad must have the output shape and Q dtype/device")
    if tuple(y.shape) != expected or y.dtype != q.dtype or y.device != q.device:
        raise ValueError("flash_sca_bwd y must have the output shape and Q dtype/device")
    if (tuple(lse.shape) != (q.shape[0], q.shape[2], q.shape[1]) or lse.dtype != torch.float32
            or lse.device != q.device):
        raise ValueError("flash_sca_bwd lse must be FP32 [B, Hq, Nq] on Q's device")
    meta = visibility(q, total, chunk_size, q_segment_idx=q_segment_idx,
                      k_segment_idx=k_segment_idx, segment_idx=segment_idx,
                      bos_mask=bos_mask, reset=reset_chunk_pos_per_seq, bos_in_call=False)
    return _backward(y_grad, q, k, v, prev_k, prev_v, y, lse, chunk_size, _scale(q, scale), meta,
                     False, deterministic)


@torch.no_grad()
def varlen_visibility(cu_seqlens_q: Tensor, cu_seqlens_k: Tensor, total_q: int, chunk_size: int,
                      position_offsets: Optional[Tensor], strict: bool = False) -> Visibility:
    """Row bounds of packed varlen sequences on the packed key axis.

    Sequence s owns queries ``[cu_q[s], cu_q[s+1])`` and keys
    ``[cu_k[s], cu_k[s+1])``; its queries are right-aligned to its keys
    (local query i sits at key position ``i + Lk - Lq``). Chunk positions are
    measured from the sequence's first key, shifted by ``position_offsets[s] -
    (Lk - Lq)`` when offsets are given (the SM90 kernels'
    ``SlidingChunkPositionAdapter``). The bounds are exact, so no segment plane
    is needed: the packed call is one batch row whose rows never reach another
    sequence's keys. ``strict`` drops each query's own key (the strict-past
    read); a sequence's first query then sees nothing.
    """
    device = cu_seqlens_q.device
    cu_q = cu_seqlens_q.to(torch.int64)
    cu_k = cu_seqlens_k.to(torch.int64)
    token = torch.arange(total_q, device=device, dtype=torch.int64)
    seq = torch.searchsorted(cu_q[1:].contiguous(), token, right=True)
    q_start = cu_q[seq]
    k_start = cu_k[seq]
    lq = cu_q[seq + 1] - q_start
    lk = cu_k[seq + 1] - k_start
    shift = lk - lq
    local = token - q_start + shift
    if position_offsets is None:
        offset = torch.zeros_like(local)
    else:
        offset = position_offsets.to(device=device, dtype=torch.int64)[seq] - shift
    first = (torch.div(local + offset, chunk_size, rounding_mode="floor") - 1) * chunk_size - offset
    lo = k_start + first.clamp_min(0)
    hi = k_start + local - int(strict)
    live = local >= 0
    lo = torch.where(live, lo, torch.ones_like(lo))
    hi = torch.where(live, hi, torch.zeros_like(hi))
    return Visibility(row_lo=lo[None].to(torch.int32).contiguous(),
                      row_hi=hi[None].to(torch.int32).contiguous())


def flash_sca_varlen(q, k, v, chunk_size, scale, *, cu_seqlens_q, cu_seqlens_k, position_offsets,
                     attn_method, deterministic, api_name="_flash_sca_varlen"):
    """Autograd FlashSCA on packed ``[T, H, D]`` varlen Q/K/V, native kernels, level 0.

    The packed tensors run as a single batch row with exact per-row key bounds
    (``varlen_visibility``); the output is ``[Tq, Hq, Dv]``.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"{api_name} requires packed [T, H, D] Q/K/V")
    q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    _validate(q4, k4, v4, None, None, chunk_size, attn_method, api_name)
    meta = varlen_visibility(cu_seqlens_q, cu_seqlens_k, q.shape[0], chunk_size, position_offsets)
    y = _NativeFlashSCA.apply(q4, k4, v4, None, None, chunk_size, _scale(q, scale), meta,
                              deterministic)
    return y.squeeze(0)


# ---- the private entry points SoftDelta is composed from -------------------
#
# Upstream composes SoftDelta from the private sliding_chunk_attention
# `_flash_sca_fwd` / `_flash_sca_bwd`: the inclusive read by Q and the
# strict-past read by the per-KV-head read keys, whose grouped read can return
# its dK/dV in FP32 (`fp32_kv_grads`) for the merge with the other read's. On
# capability 10.x those two functions call these, on the same native kernels and
# with the same determinism as the public entry points: `deterministic` changes
# nothing.


def _packed_varlen(q, k, v, prev_k, prev_v, chunk_size, cu_seqlens_q, cu_seqlens_k,
                   position_offsets, reset, strict, api_name):
    """Batched varlen metadata (cu_seqlens over the flattened batch rows) as one packed
    batch row: ``[1, T, H, *]`` Q/K/V and their exact row bounds, as the native route of
    ``sliding_chunk_attention._flash_sca`` builds them."""
    from . import sliding_chunk_attention as sca
    if cu_seqlens_k.numel() != cu_seqlens_q.numel():
        raise NotImplementedError(
            f"{api_name} native varlen needs one key run per query run; row-aligned runs "
            "(cu_seqlens_k with a different run count) are not implemented on the native "
            "Blackwell kernels, and nothing else was run"
        )
    q_packed, k_combined, v_combined = sca._pack_batched_varlen_inputs(q, k, v, prev_k, prev_v)
    offsets = position_offsets
    if q.shape[0] > 1 and position_offsets is None and not reset:
        # Chunk positions of each run inside its batch row, as upstream's row alignment.
        offsets = torch.remainder(cu_seqlens_q[:-1], q.shape[1]).contiguous()
    meta = varlen_visibility(cu_seqlens_q, cu_seqlens_k, q_packed.shape[0], chunk_size, offsets,
                             strict=strict)
    return q_packed.unsqueeze(0), k_combined.unsqueeze(0), v_combined.unsqueeze(0), meta


@torch.no_grad()
def private_fwd(q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
                reset_chunk_pos_per_seq, *, attn_method, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, position_offsets, strict_past, output_state,
                high_precision):
    """``_flash_sca_fwd`` on the native kernels: ``(y, lse)``.

    ``strict_past`` drops each query's own key (a row with no earlier key in its window
    and segment gives y = 0 and LSE = -inf); ``high_precision`` is the level-1
    subtract-before-scale softmax. ``output_state`` (the FP32 output that only the
    level-1 SoftDelta composition asks for) is not routed and raises.
    """
    from . import sliding_chunk_attention as sca
    api_name = "_flash_sca_fwd"
    total = _validate(q, k, v, prev_k, prev_v, chunk_size, attn_method, api_name)
    is_varlen = sca._validate_metadata_representation(
        q_segment_idx, k_segment_idx, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        position_offsets, prev_k, prev_v, reset_chunk_pos_per_seq,
    )
    if output_state is not None:
        raise NotImplementedError(
            f"{api_name} output_state (the FP32 forward output of the level-1 SoftDelta "
            "composition) has no native Blackwell route; nothing was run. Level 0 does not "
            "use it: call without output_state"
        )
    level = 1 if high_precision else 0
    scale = _scale(q, scale)
    if is_varlen:
        q1, k1, v1, meta = _packed_varlen(q, k, v, prev_k, prev_v, chunk_size, cu_seqlens_q,
                                          cu_seqlens_k, position_offsets, reset_chunk_pos_per_seq,
                                          strict_past, api_name)
        y, lse = _forward(q1, k1, v1, None, None, chunk_size, scale, meta, strict_past, level)
        return sca._restore_varlen_output(y.squeeze(0), q, v), sca._restore_varlen_lse(lse.squeeze(0), q)
    meta = visibility(q, total, chunk_size, q_segment_idx=q_segment_idx,
                      k_segment_idx=k_segment_idx, reset=reset_chunk_pos_per_seq,
                      strict=strict_past)
    return _forward(q, k, v, prev_k, prev_v, chunk_size, scale, meta, strict_past, level)


@torch.no_grad()
def softdelta_pair_bwd(dy, q, k, v, y, lse, s_dy, s_q, s_y, s_lse, chunk_size, scale,
                       q_segment_idx, k_segment_idx, reset, bos_mask=None, delta=None,
                       s_delta=None):
    """The native SoftDelta backward of both reads over the same K/V in one call: the grouped read
    (``q``, ``dy``, ``y``, ``lse``) and the strict-past read (``s_q`` the read keys ``[B, Nq, Hkv,
    D]``, ``s_dy``, ``s_y``, ``s_lse``), the strict read taken by the grouped read's key leaf as one
    more reader of each KV head. ``(dq, dk, dv, dk_read)`` with dk / dv the sum of both reads,
    rounded once; None when the call is not eligible (no such op, or the native preflight refused),
    nothing having run.

    ``bos_mask``, when the caller's own call used a BOS mask rather than a resolved segment id,
    is read by both reads' ``visibility()`` instead of ``q_segment_idx`` / ``k_segment_idx``: the
    same one-launch-or-zero-launch native route the forward already takes (``Visibility.bos_mask``),
    not the segment-id chain over ids that were only resolved because some other reader of the
    call's arguments (the fused path, or a caller giving explicit segment ids) needs them.

    ``delta`` / ``s_delta`` (FP32 ``[B, Hq, Nq]`` / ``[B, Hkv, Nq]``), when given, are the two
    reads' rowsum(dY * O) in the preprocessing's summation order (the SoftDelta gate backward's
    ``_softdelta_gate_bwd_deltas``); ``y`` / ``s_y`` are then None: the reads are not needed, and
    the preprocessing writes only its tile tables."""
    ops = _extension_ops()
    op = getattr(ops, "_flash_sca_native_softdelta_pair_bwd", None)
    if op is None:
        return None
    total = _validate(q, k, v, None, None, chunk_size, "default", "_flash_sca_bwd")
    if bos_mask is not None:
        meta = visibility(q, total, chunk_size, bos_mask=bos_mask, reset=reset, strict=False)
        s_meta = visibility(s_q, total, chunk_size, bos_mask=bos_mask, reset=reset, strict=True)
    else:
        meta = visibility(q, total, chunk_size, q_segment_idx=q_segment_idx,
                          k_segment_idx=k_segment_idx, reset=reset, strict=False)
        s_meta = visibility(s_q, total, chunk_size, q_segment_idx=q_segment_idx,
                            k_segment_idx=k_segment_idx, reset=reset, strict=True)
    if (meta.bos_mask is None) != (s_meta.bos_mask is None):
        return None
    out = op(dy, q, k, v, y, lse, s_dy, s_q, s_y, s_lse, chunk_size, float(scale),
             meta.q_segment_idx, meta.k_segment_idx, meta.segment_exact, meta.row_lo, meta.row_hi,
             s_meta.row_lo, s_meta.row_hi, bos_mask=meta.bos_mask, delta=delta,
             strict_delta=s_delta)
    if not out:
        return None
    record_native_leaf("sca_bwd", q, 0)
    return tuple(out)


@torch.no_grad()
def private_bwd(y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v, q_segment_idx,
                k_segment_idx, deterministic, reset_chunk_pos_per_seq, *, attn_method,
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, position_offsets,
                strict_past, fp32_kv_grads):
    """``_flash_sca_bwd`` on the native kernels: ``(dq, dk, dv, dprev_k, dprev_v)``.

    ``fp32_kv_grads`` (grouped query heads, dense or segment metadata, not
    ``strict_past``) returns dk/dv/dprev_k/dprev_v as FP32: the values the 16-bit
    gradients of the same call are rounded from, bit for bit.
    """
    from . import sliding_chunk_attention as sca
    api_name = "_flash_sca_bwd"
    total = _validate(q, k, v, prev_k, prev_v, chunk_size, attn_method, api_name)
    is_varlen = sca._validate_metadata_representation(
        q_segment_idx, k_segment_idx, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        position_offsets, prev_k, prev_v, reset_chunk_pos_per_seq,
    )
    if fp32_kv_grads and (is_varlen or strict_past):
        raise ValueError("FP32 dK / dV output needs the dense, non-strict FlashSCA backward")
    expected = (*q.shape[:-1], v.shape[-1])
    if tuple(y_grad.shape) != expected or y_grad.dtype != q.dtype or y_grad.device != q.device:
        raise ValueError(f"{api_name} y_grad must have the output shape and Q dtype/device")
    if tuple(y.shape) != expected or y.dtype != q.dtype or y.device != q.device:
        raise ValueError(f"{api_name} y must have the output shape and Q dtype/device")
    if (tuple(lse.shape) != (q.shape[0], q.shape[2], q.shape[1]) or lse.dtype != torch.float32
            or lse.device != q.device):
        raise ValueError(f"{api_name} lse must be FP32 [B, Hq, Nq] on Q's device")
    scale = _scale(q, scale)
    if is_varlen:
        q1, k1, v1, meta = _packed_varlen(q, k, v, prev_k, prev_v, chunk_size, cu_seqlens_q,
                                          cu_seqlens_k, position_offsets, reset_chunk_pos_per_seq,
                                          strict_past, api_name)
        dy1, y1, lse1 = sca._pack_varlen_backward_tensors(y_grad, y, lse, q)
        dq, dk, dv, _, _ = _backward(dy1.unsqueeze(0), q1, k1, v1, None, None, y1.unsqueeze(0),
                                     lse1.unsqueeze(0), chunk_size, scale, meta, strict_past,
                                     deterministic)
        return sca._unpack_batched_varlen_grads(dq.squeeze(0), dk.squeeze(0), dv.squeeze(0), q, k, v,
                                                prev_k, prev_v)
    meta = visibility(q, total, chunk_size, q_segment_idx=q_segment_idx,
                      k_segment_idx=k_segment_idx, reset=reset_chunk_pos_per_seq,
                      strict=strict_past)
    return _backward(y_grad, q, k, v, prev_k, prev_v, y, lse, chunk_size, scale, meta,
                     strict_past, deterministic, fp32_kv_grads)
