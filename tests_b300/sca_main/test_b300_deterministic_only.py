"""B300 supports only deterministic SCA.

On sm_103 every flash_sca_fwd / flash_sca_bwd call repeats bit-exactly, and
flash_sca_bwd(deterministic=False) returns exactly the bits of
deterministic=True (no unordered atomic dQ is ever selected). Checked on the
curated upstream tests/sca cases (levels 0 and 1) and on the six traced Gekko
calls (BF16, level 0, bos_mask; the Gekko shape check's inputs), plus the
same three calls of the Gekko-7B preset (48/8 heads, D = 128) under --long.
"""
import hashlib
import json
import math
import pathlib
import random

import pytest
import torch
import torch.nn.functional as F

import xattn
from sca_main_numerics import DTYPES, cases_for, make_inputs, options_and_segments, require_native_b300

CHUNK = 4096
DOC_LENGTHS = pathlib.Path(__file__).resolve().parent / 'gekko_doc_lengths.json'
GEKKO = {
    'gekko-0.9B/train': dict(batch=4, length=65536, heads=8, kv_heads=4, dim=256, previous=False),
    'gekko-4B/train': dict(batch=1, length=65536, heads=32, kv_heads=8, dim=128, previous=False),
    'gekko-0.9B/prefill': dict(batch=1, length=16384, heads=8, kv_heads=4, dim=256, previous=True),
    'gekko-4B/prefill': dict(batch=1, length=16384, heads=32, kv_heads=8, dim=128, previous=True),
    'gekko-0.9B/prefill-continued': dict(batch=1, length=16384, heads=8, kv_heads=4, dim=256, previous=True,
                                         continued=True),
    'gekko-4B/prefill-continued': dict(batch=1, length=16384, heads=32, kv_heads=8, dim=128, previous=True,
                                       continued=True),
    # xllm d31a245 models/gekko.py gekko-7B (48 query over 8 KV heads, D = Dv = 128), with
    # the 4B rows' lengths and packing: a group of six, which the upstream cases never have.
    'gekko-7B/train': dict(batch=1, length=65536, heads=48, kv_heads=8, dim=128, previous=False, long=True),
    'gekko-7B/prefill': dict(batch=1, length=16384, heads=48, kv_heads=8, dim=128, previous=True, long=True),
    'gekko-7B/prefill-continued': dict(batch=1, length=16384, heads=48, kv_heads=8, dim=128, previous=True,
                                       continued=True, long=True),
}


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def assert_same_bits(first, second, names):
    for name, a, b in zip(names, first, second):
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(_bits(a), _bits(b)), f'{name} is not bit-identical'


def check_deterministic_only(fwd, bwd):
    with torch.no_grad():
        first = fwd()
        assert_same_bits(first, fwd(), ('y', 'lse'))
        reference = bwd(first, True)
        names = ('dq', 'dk', 'dv', 'dprev_k', 'dprev_v')
        nondet = bwd(first, False)
        assert_same_bits(reference, nondet, names)
        assert_same_bits(nondet, bwd(first, False), names)
        assert_same_bits(reference, bwd(first, True), names)


@pytest.mark.parametrize('case', cases_for('chunk'), ids=lambda case: case.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('level', (0, 1))
def test_upstream_cases_are_deterministic_only(case, dtype, kv_heads, level):
    require_native_b300()
    inputs = make_inputs('chunk', case, dtype, kv_heads, 'cuda')
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


def _document_ends(batch, seq_len, seed):
    lengths = json.loads(DOC_LENGTHS.read_text())['lengths']
    rng = random.Random(seed)
    rows = []
    for _ in range(batch):
        ends, position = [], 0
        while position < seq_len:
            position += rng.choice(lengths)
            if position <= seq_len:
                ends.append(position - 1)
        rows.append(ends)
    return rows


def _gekko_bos(shape):
    def from_eos(eos):
        return F.pad(eos[:, :-1], (1, 0), value=True)

    def eos_mask(batch, seq_len, seed):
        eos = torch.zeros(batch, seq_len, dtype=torch.bool)
        for row, ends in enumerate(_document_ends(batch, seq_len, seed)):
            eos[row, ends] = True
        return eos
    if not shape['previous']:
        return from_eos(eos_mask(shape['batch'], shape['length'], 1)).cuda()
    eos = eos_mask(1, 4 * shape['length'], 2)
    first, second = (from_eos(eos[:, i * shape['length']:(i + 1) * shape['length']]) for i in (0, 1))
    if shape.get('continued'):
        second[:, 0] = eos[:, shape['length'] - 1]
    return torch.cat([first[:, -CHUNK:], second], 1).cuda()


@pytest.mark.parametrize('name', [pytest.param(name, marks=pytest.mark.long) if shape.get('long') else name
                                  for name, shape in GEKKO.items()])
def test_gekko_traced_calls_are_deterministic_only(name):
    require_native_b300()
    shape = GEKKO[name]
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
    generator = torch.Generator(device='cuda').manual_seed(seed)
    B, N, H, G, D = (shape[k] for k in ('batch', 'length', 'heads', 'kv_heads', 'dim'))

    def normal(*size):
        return torch.randn(*size, generator=generator, device='cuda', dtype=torch.float32)

    def unit(*size):
        return F.normalize(normal(*size), dim=-1).to(torch.bfloat16)

    def positive(*size):
        return (F.silu(normal(*size)) + 0.1).to(torch.bfloat16)
    q, k, v = unit(B, N, H, D), unit(B, N, G, D), positive(B, N, G, D)
    pk, pv = (unit(B, CHUNK, G, D), positive(B, CHUNK, G, D)) if shape['previous'] else (None, None)
    dy = (normal(B, N, H, D) / math.sqrt(H * D)).to(torch.bfloat16)
    bos = _gekko_bos(shape)
    scale = 1.0 / math.sqrt(D)
    check_deterministic_only(
        lambda: xattn.flash_sca_fwd(q, k, v, CHUNK, scale, pk, pv, bos_mask=bos, segment_idx=None,
                                    backend='auto', high_precision_level=0),
        lambda f, det: xattn.flash_sca_bwd(dy, q, k, v, *f, CHUNK, scale, pk, pv, bos_mask=bos,
                                           segment_idx=None, backend='auto', deterministic=det,
                                           high_precision_level=0))
