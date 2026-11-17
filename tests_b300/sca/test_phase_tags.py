# Author: Shicheng Wen

"""Per-phase dispatch tags: what a row may be read as, and what it may not.

The tags are the runtime half of the dispatch proof; the sm_103a cubin/SASS is
the device half. These tests pin the three conclusions a reader is allowed to
draw from them at this head, and nothing wider: the row shape (phase, selected,
launched, state, count) with one row per decision, the scalar body of the delta
phase rather than a tensor-core body it does not have, and -- B300 SCA being
deterministic-only -- the row-owned dQ kernel running for deterministic=False
too, with no dQ skip row (the atomic-dQ route that skipped it is never taken).

They run only on a compute capability 10.3 device whose extension links the
native FlashSCA group and the phase-tag bindings.
"""

import pytest
import torch
import torch.nn.functional as F

import xattn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="native FlashSCA runs on compute capability 10.3",
)


def _ops():
    xattn_cuda = pytest.importorskip("xattn_cuda")
    ops = getattr(xattn_cuda, "ops", None)
    if ops is None or not hasattr(ops, "_flash_sca_native_phase_tags"):
        pytest.skip("build does not link the native FlashSCA phase tags")
    return ops


def _rows(ops):
    return {(phase, selected, launched, state): int(count)
            for phase, selected, launched, state, count in ops._flash_sca_native_phase_tags()}


def _inputs(batch=2, length=300, heads=4, kv_heads=2, dim=64, value_dim=96, previous=64):
    torch.manual_seed(0)
    total = length + previous
    q = F.normalize(torch.randn(batch, length, heads, dim, device="cuda"), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(batch, total, kv_heads, dim, device="cuda"), dim=-1).to(torch.bfloat16)
    v = torch.randn(batch, total, kv_heads, value_dim, device="cuda").to(torch.bfloat16)
    dy = torch.randn(batch, length, heads, value_dim, device="cuda").to(torch.bfloat16)
    return q, k[:, previous:], v[:, previous:], k[:, :previous], v[:, :previous], dy


def test_rows_name_the_body_of_each_phase_and_no_dq_skip():
    ops = _ops()
    q, k, v, pk, pv, dy = _inputs()

    ops._flash_sca_native_phase_tags_reset()
    assert _rows(ops) == {}, "a reset must leave no row behind"

    # Level 0: the level-1 backward is the precision-repair leaf, which has no
    # FlashSCA phase rows.
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv)
    rows = _rows(ops)
    assert rows == {("forward", "mma_sync", "mma_sync", "ran"): 1}, rows

    xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, pk, pv, deterministic=True)
    rows = _rows(ops)
    # delta is a scalar dot-product/reduction with no T::Mma: it is NOT mma_sync.
    assert rows[("delta", "scalar", "scalar", "ran")] == 1, rows
    assert rows[("dq", "mma_sync", "mma_sync", "ran")] == 1, rows
    assert rows[("dk", "mma_sync", "mma_sync", "ran")] == 1, rows
    assert rows[("dv", "mma_sync", "mma_sync", "ran")] == 1, rows
    assert rows[("forward", "mma_sync", "mma_sync", "ran")] == 1, rows
    assert len(rows) == 5, rows
    assert all(state != "refused" for (_, _, _, state) in rows), rows

    xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, pk, pv)
    rows = _rows(ops)
    # B300 SCA is deterministic-only: deterministic=False runs the same fixed-order
    # schedule, so the row-owned dQ kernel runs again and no dQ skip row appears.
    assert ("dq", "none", "none", "skipped") not in rows, rows
    assert rows[("dq", "mma_sync", "mma_sync", "ran")] == 2, rows
    assert rows[("dk", "mma_sync", "mma_sync", "ran")] == 2, rows
    assert rows[("dv", "mma_sync", "mma_sync", "ran")] == 2, rows
    assert rows[("delta", "scalar", "scalar", "ran")] == 2, rows
    assert len(rows) == 5, rows
