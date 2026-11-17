"""The public FlashSCA entry points at a group ratio of six, Gekko-7B's (48 query heads over 8
KV heads).

The upstream cases run four query heads over 4/2/1 KV heads (ratios 1, 2 and 4). These run
the same curated cases with six query heads per KV head (6/1 and 12/2) through flash_sca,
flash_sca_fwd and flash_sca_bwd against the FP64 oracle with its tolerances, and the
deterministic-only contract of test_b300_deterministic_only (forward repeat, backward
repeat, deterministic=False == True) at levels 0 and 1; the full D/V matrix at 6/1 under
--full-matrix (its long rows under --long too). The Gekko-7B calls themselves are in
test_b300_deterministic_only (under --long).
"""
import pytest
import torch

import xattn
from sca_main_numerics import (DTYPES, HEAD_DIMS, MATRIX_LENGTHS, cases_for, check_case, check_dimensions,
                               make_inputs, options_and_segments, require_native_b300)
from test_b300_deterministic_only import check_deterministic_only

RATIOS = ((6, 1), (12, 2))


@pytest.mark.parametrize('case', cases_for('chunk'), ids=lambda case: case.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('heads,kv_heads', RATIOS, ids=('6-1', '12-2'))
@pytest.mark.parametrize('deterministic,high_precision', ((False, False), (True, True)))
def test_output_and_gradients_ratio_six(case, dtype, heads, kv_heads, deterministic, high_precision):
    check_case('chunk', case, dtype, kv_heads, deterministic, high_precision, heads=heads)


@pytest.mark.parametrize('case', cases_for('chunk'), ids=lambda case: case.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('level', (0, 1))
def test_ratio_six_is_deterministic_only(case, dtype, level):
    require_native_b300()
    inputs = make_inputs('chunk', case, dtype, 2, 'cuda', heads=12)
    options, _, _ = options_and_segments('chunk', case, inputs, 'auto')
    base = [t.detach() for t in inputs[:3]]
    if case.previous:
        options['prev_k'], options['prev_v'] = inputs[-2].detach(), inputs[-1].detach()
    generator = torch.Generator(device='cuda').manual_seed(7)
    dy = torch.randn(*base[0].shape[:-1], base[2].shape[-1], device='cuda', dtype=dtype, generator=generator)
    check_deterministic_only(
        lambda: xattn.flash_sca_fwd(*base, **options, high_precision_level=level),
        lambda f, det: xattn.flash_sca_bwd(dy, *base, *f, **options, deterministic=det,
                                           high_precision_level=level))


@pytest.mark.full_matrix
@pytest.mark.parametrize('length', MATRIX_LENGTHS)
@pytest.mark.parametrize('dim', HEAD_DIMS)
@pytest.mark.parametrize('value_dim', HEAD_DIMS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('high_precision', (False, True))
def test_full_dimensions_ratio_six(length, dim, value_dim, dtype, high_precision):
    check_dimensions('chunk', length, dim, value_dim, dtype, 1, True, high_precision, heads=6)
