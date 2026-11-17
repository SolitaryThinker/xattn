# Author: Shicheng Wen

"""Native Blackwell FlashSCA: dispatch, determinism and metadata equivalences.

These run only on a compute capability 10.3 device with an extension that
links the native FlashSCA group. The numerical contract itself is the FP64
oracle suite (tests/sca and the B300 contract check); here each test pins one
property that the contract cannot observe directly: the public entry points
actually launch the native kernels, a requested deterministic backward repeats
bit-exactly, equivalent metadata spellings give identical results, and a
backend the device cannot serve raises.
"""

import pytest
import torch
import torch.nn.functional as F

import xattn
from xattn.ops import _sca_native

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="native FlashSCA runs on compute capability 10.3",
)


def _inputs(batch=2, length=300, heads=4, kv_heads=2, dim=64, value_dim=96, previous=0, dtype=torch.bfloat16):
    torch.manual_seed(0)
    total = length + previous
    q = F.normalize(torch.randn(batch, length, heads, dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, total, kv_heads, dim, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(batch, total, kv_heads, value_dim, device="cuda").to(dtype)
    dy = torch.randn(batch, length, heads, value_dim, device="cuda").to(dtype)
    if previous:
        return q, k[:, previous:], v[:, previous:], k[:, :previous], v[:, :previous], dy
    return q, k, v, None, None, dy


def _bos(batch, total, run):
    ids = (torch.arange(total, device="cuda") // run).expand(batch, -1)
    bos = torch.ones(batch, total, dtype=torch.bool, device="cuda")
    bos[:, 1:] = ids[:, 1:] != ids[:, :-1]
    return bos, bos.long().cumsum(1)


def test_public_entry_points_launch_native_kernels():
    from xattn.ops import _precision_backend

    q, k, v, pk, pv, dy = _inputs(previous=64)
    before = _sca_native.launch_counts()
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv)
    xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, pk, pv)
    after = _sca_native.launch_counts()
    assert after[0] == before[0] + 1 and after[1] == before[1] + 1
    assert y.dtype == q.dtype and lse.dtype == torch.float32

    # Level 1: the native high-precision forward, and the precision-repair
    # backward on the native leaf (it recomputes its row statistics).
    before = _sca_native.launch_counts()
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv, high_precision_level=1)
    xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, pk, pv, high_precision_level=1)
    after = _sca_native.launch_counts()
    assert after[0] == before[0] + 1
    assert _precision_backend.route_trace()[-1]["resolved"] == "native"
    assert y.dtype == q.dtype and lse.dtype == torch.float32


def test_deterministic_backward_repeats_bit_exactly():
    q, k, v, pk, pv, dy = _inputs(previous=64)
    bos, _ = _bos(2, 364, 50)
    y, lse = xattn.flash_sca_fwd(q, k, v, 64, None, pk, pv, bos_mask=bos)
    first = xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, pk, pv, bos_mask=bos, deterministic=True)
    second = xattn.flash_sca_bwd(dy, q, k, v, y, lse, 64, None, pk, pv, bos_mask=bos, deterministic=True)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_bos_mask_and_segment_ids_agree():
    q, k, v, _, _, dy = _inputs()
    bos, segments = _bos(2, 300, 37)
    y_bos, lse_bos = xattn.flash_sca_fwd(q, k, v, 64, bos_mask=bos)
    y_seg, lse_seg = xattn.flash_sca_fwd(q, k, v, 64, segment_idx=segments)
    assert torch.equal(y_bos, y_seg) and torch.equal(lse_bos, lse_seg)


def test_unsorted_segment_ids_match_their_sorted_relabeling():
    # Relabeling segment ids does not change visibility; unsorted ids take the
    # per-key segment test, sorted ids the run-bound path.
    q, k, v, _, _, _ = _inputs()
    ids = (torch.arange(300, device="cuda") // 37).expand(2, -1).contiguous()
    y_sorted, _ = xattn.flash_sca_fwd(q, k, v, 64, segment_idx=ids)
    y_unsorted, _ = xattn.flash_sca_fwd(q, k, v, 64, segment_idx=(100 - ids).contiguous())
    torch.testing.assert_close(y_sorted, y_unsorted, atol=0, rtol=0)


def test_empty_rows_are_zero_with_negative_infinite_lse():
    q, k, v, _, _, dy = _inputs(length=40)
    qi = torch.full((2, 40), -1, dtype=torch.int64, device="cuda")
    ki = torch.zeros((2, 40), dtype=torch.int64, device="cuda")
    y, lse = xattn.flash_sca_fwd(q, k, v, 16, None, None, None, qi, ki)
    assert torch.equal(y, torch.zeros_like(y)) and bool(torch.isneginf(lse).all())
    grads = xattn.flash_sca_bwd(dy, q, k, v, y, lse, 16, None, None, None, qi, ki)
    assert all(bool((g == 0).all()) for g in grads[:3])


def test_sm90_backend_is_refused_on_blackwell():
    q, k, v, _, _, _ = _inputs()
    with pytest.raises(RuntimeError):
        xattn.flash_sca_fwd(q, k, v, 64, backend="sm90")
