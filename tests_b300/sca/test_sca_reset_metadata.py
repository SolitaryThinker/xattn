"""Reset-metadata legality on the native SCA route (owner call).

``reset_chunk_pos_per_seq=True`` measures positions from each segment's start
and length (``p = q - q_start(s) + len_k(s) - len_q(s)``, ``j = k - k_start(s)``),
and the native route hands the kernels one ``[row_lo, row_hi]`` interval with
no per-key segment test (``SegmentTest`` needs segment planes, and the reset
branch of ``_sca_native.visibility`` returns none). Both presuppose that every
segment id occupies **one contiguous run** in Q and in K.

With an id split across two runs the interval covers keys of another segment
and misses some of its own, with nothing downstream able to tell: silent wrong
``y`` and silent wrong gradients. The reference answers that call differently
rather than refusing it (run-local coordinates), so the two routes cannot both
be right; the native route refuses it instead of answering it a second way.

What must stay legal, and is pinned below: one run per id in any order, the
BOS route (ids from a cumsum are nondecreasing by construction), a query
segment with no key of its id (``reset-unmatched``: zero output, empty
interval), a scattered id that no row can use on either plane (a scattered Q
id with no key match, a scattered K id no query carries), and every call with
``reset_chunk_pos_per_seq=False``.

CPU-only: ``visibility`` is pure torch and needs no built extension.
"""

import pytest
import torch

import xattn.ops._sca_native as native


def _q(length=4, batch=1, dim=64):
    return torch.zeros(batch, length, 1, dim, dtype=torch.float16)


def _ids(*rows):
    return torch.tensor(rows, dtype=torch.int64)


def _vis(q, total, chunk, **kw):
    return native.visibility(q, total, chunk, **kw)


def test_reset_single_run_per_id_is_accepted():
    q, k = _ids([0, 0, 1, 1]), _ids([0, 0, 1, 1])
    out = _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)
    assert out.row_lo is not None and out.row_hi is not None
    assert out.row_lo.dtype == torch.int32 and out.row_lo.shape == (1, 4)
    # No per-key plane: on the reset route the bounds are the whole rule.
    assert out.q_segment_idx is None and out.k_segment_idx is None


def test_reset_single_run_in_unsorted_order_is_accepted():
    """One run per id is the rule, not sortedness: [1,1,0,0] is legal."""
    q, k = _ids([1, 1, 0, 0]), _ids([1, 1, 0, 0])
    out = _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)
    assert out.row_lo.shape == (1, 4)


def test_reset_multi_run_id_is_refused():
    """An id in two runs under reset is refused, not silently mis-answered."""
    q, k = _ids([0, 0, 0, 0]), _ids([0, 1, 0, 1])
    with pytest.raises(ValueError, match="one contiguous run in K"):
        _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)


def test_reset_multi_run_q_id_is_refused():
    q, k = _ids([0, 1, 0, 1]), _ids([0, 0, 1, 1])
    with pytest.raises(ValueError, match="one contiguous run in Q"):
        _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)


def test_reset_unmatched_q_segment_stays_legal():
    """A query segment with no key of its id keeps its empty interval."""
    q, k = _ids([0, 0, 7, 7]), _ids([0, 0, 1, 1])
    out = _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)
    lo, hi = out.row_lo[0].tolist(), out.row_hi[0].tolist()
    assert lo[0] <= hi[0] and lo[1] <= hi[1]          # segment 0 is live
    assert lo[2] > hi[2] and lo[3] > hi[3]            # segment 7 sees nothing


def test_reset_scattered_dead_q_id_is_exempt():
    """An id with no key match may be scattered: its rows are dead by count==0.

    Refusing it would turn the documented zero-output case into a validation
    error, so the Q-side check is gated on the id having matches in K.
    """
    q, k = _ids([7, 0, 0, 7]), _ids([0, 0, 1, 1])
    out = _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)
    lo, hi = out.row_lo[0].tolist(), out.row_hi[0].tolist()
    assert lo[1] <= hi[1] and lo[2] <= hi[2]          # id 0 is live
    assert lo[0] > hi[0] and lo[3] > hi[3]            # id 7 sees nothing


def test_reset_scattered_unused_k_id_is_exempt():
    """A scattered K id that no query carries is never consulted: legal.

    No row uses that id's interval, so the wrong interval cannot be read and
    its rows stay dead (zero output, exactly zero gradient). Refusing it would
    be an over-refusal, so the K-side check is gated symmetrically.
    """
    q, k = _ids([0, 0, 0, 0]), _ids([5, 0, 0, 5])
    out = _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)
    lo, hi = out.row_lo[0].tolist(), out.row_hi[0].tolist()
    # Whatever the rows get, no interval may reach key 0 or key 3 (id 5).
    for a, b in zip(lo, hi):
        assert a > b or (a >= 1 and b <= 2)


def test_reset_mixed_batch_refuses_the_scattered_row():
    """The check is per row: one scattered row in a batch of two refuses."""
    q = _ids([0, 0, 0, 0], [0, 0, 0, 0])
    k = _ids([0, 0, 1, 1], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="one contiguous run in K"):
        _vis(_q(batch=2), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)


def test_reset_mixed_batch_legal_when_the_scattered_row_is_dead():
    q = _ids([0, 0, 0, 0], [9, 0, 0, 9])
    k = _ids([0, 0, 1, 1], [0, 0, 1, 1])
    out = _vis(_q(batch=2), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)
    assert out.row_lo.shape == (2, 4)


def test_reset_mixed_batch_legal_when_the_k_scatter_is_dead():
    """Batch 2, the scattered K id in one row only and carried by no query:
    legal, and the row's intervals stay confined to its live key run."""
    q = _ids([0, 0, 0, 0], [0, 0, 0, 0])
    k = _ids([0, 0, 1, 1], [5, 0, 0, 5])
    out = _vis(_q(batch=2), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=True)
    lo, hi = out.row_lo[1].tolist(), out.row_hi[1].tolist()
    for a, b in zip(lo, hi):
        assert a > b or (a >= 1 and b <= 2)


def test_reset_bos_route_is_single_run_by_construction():
    """BOS ids come from a cumsum, so the guard cannot fire on that route."""
    bos = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    out = _vis(_q(), 4, 2, bos_mask=bos, reset=True)
    assert out.row_lo.shape == (1, 4)


def test_multi_run_ids_without_reset_are_unaffected():
    """Without reset the per-key segment planes carry the rule, so the same ids
    are legal there and the guard must not touch them."""
    q, k = _ids([0, 1, 0, 1]), _ids([0, 1, 0, 1])
    out = _vis(_q(), 4, 2, q_segment_idx=q, k_segment_idx=k, reset=False)
    assert out.q_segment_idx is not None and out.k_segment_idx is not None
    assert out.row_lo.shape == (1, 4)
