"""The private sliding-chunk entry points SoftDelta is composed from, on B300.

Upstream composes SoftDelta (xattn/ops/softdelta_attention.py) from the private
``_flash_sca_fwd`` / ``_flash_sca_bwd``: an inclusive read by Q over grouped heads, whose
dK/dV come back in FP32 with ``fp32_kv_grads`` (the input of the gradient merge), and a
strict-past read whose queries are the per-KV-head read keys (ungrouped heads, by default
the keys themselves). On sm_103 both run on the native kernels (the autouse fixture of
conftest.py proves it for every test) and are deterministic whatever ``deterministic``
says.

Checked against the FP64 reference with the oracle's tolerances (sca_main_reference):
strict-past and inclusive reads, FP32 dK/dV, previous K/V, segment ids (sorted, as a BOS
mask gives them, and not), batched varlen runs, reset positions, short Q, empty rows, FP16
and BF16, grouped and ungrouped heads; and the determinism trio on every case: the forward
repeats bit-exactly, the backward repeats bit-exactly from the same and from a fresh
forward, and deterministic=False returns the bits of deterministic=True; high_precision (the level-1
forward) against FP64 too. The SoftDelta
reads of the Gekko layers (strict at 4/4 x 256 and 8/8 x 128, FP32 inclusive at 8/4 x 256,
32/8 x 128 and 48/8 x 128, chunk 4096, packed documents) run at two chunks by default and at
the training length 65536 under --long.
"""
import hashlib
import json
import math
import pathlib
import random
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from sca_main_numerics import DTYPES, require_native_b300
from sca_main_reference import ATOL, GRAD_ATOL, RTOL, _allowed_masks, _attention_from_mask
from xattn.ops.sliding_chunk_attention import _flash_sca_bwd, _flash_sca_fwd

GRAD_NAMES = ('dq', 'dk', 'dv', 'dprev_k', 'dprev_v')


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


@dataclass(frozen=True)
class Read:
    name: str
    q_length: int = 193
    k_length: int = 193
    dim: int = 64
    value_dim: int = 64
    span: int = 64
    previous: bool = False  # previous K/V of exactly one chunk
    metadata: str = 'dense'  # dense | segment | sorted | independent | unmatched | varlen
    reset: bool = False
    batch: int = 2
    layout: str = 'contiguous'
    segment_length: int = 67


READS = (
    Read('dense'),
    Read('dense-uneven', dim=40, value_dim=56),
    Read('singleton', 1, 1, 32, 64, batch=1),
    Read('previous', dim=128, value_dim=104, previous=True, span=65),
    Read('segment', dim=96, value_dim=128, metadata='segment'),
    Read('sorted', dim=160, value_dim=192, metadata='sorted'),
    Read('short-segment', 65, 193, 192, 160, metadata='segment'),
    Read('short-independent', 129, 257, 128, 192, metadata='independent'),
    Read('empty-rows', 17, 41, 64, 64, metadata='unmatched'),
    Read('previous-sorted', 65, 193, 192, 96, previous=True, span=128, metadata='sorted'),
    Read('reset', metadata='segment', reset=True),
    Read('reset-independent', 17, 41, 64, 192, metadata='independent', reset=True),
    Read('reset-previous', 65, 193, 192, 160, previous=True, span=128, metadata='sorted', reset=True),
    Read('varlen', metadata='varlen'),
    Read('varlen-reset', 129, 129, 96, 128, metadata='varlen', reset=True),
    Read('head-view', layout='head_slice'),
    Read('strided-view', layout='last_stride'),
    # The tcgen05/UMMA buckets (D = Dv = 128 or 256) at a few query tiles.
    Read('umma-128', 1024, 1024, 128, 128, span=256, metadata='sorted', batch=1, segment_length=301),
    Read('umma-256', 1024, 1024, 256, 256, span=256, metadata='sorted', batch=1, segment_length=301),
    Read('umma-128-previous', 512, 512, 128, 128, span=256, previous=True, batch=1),
    Read('umma-256-short', 384, 1024, 256, 256, span=256, metadata='segment', batch=1,
         segment_length=301),
    # The queries are the keys (xattn's SoftDelta default read keys, k_read=None): one tensor
    # for the ungrouped read, the keys repeated over each group otherwise.
    Read('self-read', 512, 512, 128, 128, span=256, metadata='sorted', batch=1, segment_length=301,
         layout='self'),
)
# (query heads, KV heads): the ungrouped strict-past read, a grouped read, a group of six.
HEADS = ((4, 4), (8, 2), (12, 2))


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'little')


def _layout(t, layout):
    """The same values in a non-contiguous view, when the case asks for one."""
    if layout == 'head_slice':
        wide = torch.stack((torch.zeros_like(t), t), dim=3).flatten(2, 3)
        return wide[:, :, 1::2]
    if layout == 'last_stride':
        return torch.stack((t, torch.zeros_like(t)), dim=-1).flatten(-2)[..., ::2]
    return t


def make_read(read, dtype, heads, kv_heads):
    """Inputs of one read: q, k, v, prev_k, prev_v, the metadata keyword arguments and the
    reference's q/k segment ids (None for a dense read)."""
    torch.manual_seed(_seed(read, dtype, heads, kv_heads))
    B, P = read.batch, read.span if read.previous else 0
    total = read.k_length + P

    def rand(*shape):
        return torch.randn(*shape, device='cuda', dtype=torch.float32)
    q = F.normalize(rand(B, read.q_length, heads, read.dim), dim=-1).to(dtype)
    kt = F.normalize(rand(B, total, kv_heads, read.dim), dim=-1).to(dtype)
    vt = (F.silu(rand(B, total, kv_heads, read.value_dim)) + 0.1).to(dtype)
    q, k, v = _layout(q, read.layout), _layout(kt[:, P:], read.layout), _layout(vt[:, P:], read.layout)
    if read.layout == 'self':
        assert read.q_length == read.k_length and not read.previous
        q = k if heads == kv_heads else k.repeat_interleave(heads // kv_heads, dim=2)
    prev_k, prev_v = (kt[:, :P], vt[:, :P]) if read.previous else (None, None)
    metadata, qi, ki = {}, None, None
    if read.metadata != 'dense':
        ki = (torch.arange(total, device='cuda') // read.segment_length).expand(B, -1).clone()
        ki += torch.arange(B, device='cuda')[:, None] * 100
        qi = ki[:, -read.q_length:].clone()
        if read.metadata == 'segment':
            # Unsorted ids: the rows' segments in descending order (the per-key test stays on).
            ki = ki.max() - ki
            qi = ki[:, -read.q_length:].clone()
        elif read.metadata == 'independent':
            qi[:, :min(max(1, read.q_length // 3), 33)] = ki[:, :1]
        elif read.metadata == 'unmatched':
            qi.fill_(-1)
        if read.metadata == 'varlen':
            assert P == 0 and read.q_length == read.k_length
            lengths = torch.unique_consecutive(ki.flatten(), return_counts=True)[1]
            cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
            starts = (cu[:-1].long() % read.q_length).to(torch.int32)
            metadata = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=int(lengths.max()),
                            max_seqlen_k=int(lengths.max()),
                            position_offsets=None if read.reset else starts)
        else:
            # SoftDelta's form of a BOS mask: q/k segment ids, Q the suffix of K.
            metadata = dict(q_segment_idx=qi.contiguous(), k_segment_idx=ki.contiguous())
    return q, k, v, prev_k, prev_v, metadata, qi, ki


def reference_read(read, q, k, v, prev_k, prev_v, qi, ki, scale, strict):
    """FP64 read and the leaves its gradients are taken for (q, all keys, all values)."""
    kt = k if prev_k is None else torch.cat((prev_k, k), 1)
    vt = v if prev_v is None else torch.cat((prev_v, v), 1)
    refs = [t.detach().double().requires_grad_() for t in (q, kt, vt)]
    inclusive, strict_mask = _allowed_masks(q.shape[1], kt.shape[1], q.shape[0], q.device, 'chunk',
                                            read.span, qi, ki, read.reset)
    mask = strict_mask if strict else inclusive
    return refs, mask, _attention_from_mask(*refs, mask, scale)


def split_key_grads(grads, previous):
    """(dq, dk over [prev ; current], dv over [prev ; current]) of a backward's five outputs."""
    dq, dk, dv, dprev_k, dprev_v = grads
    if not previous:
        assert dprev_k is None and dprev_v is None
        return dq, dk, dv
    return dq, torch.cat((dprev_k, dk), 1), torch.cat((dprev_v, dv), 1)


def check_read(read, dtype, heads, kv_heads, strict):
    require_native_b300()
    q, k, v, prev_k, prev_v, metadata, qi, ki = make_read(read, dtype, heads, kv_heads)
    scale = read.dim ** -0.5
    common = (read.span, scale, prev_k, prev_v)

    def fwd():
        return _flash_sca_fwd(q, k, v, *common, backend='auto',
                              reset_chunk_pos_per_seq=read.reset, strict_past=strict, **metadata)

    def bwd(forward, deterministic, **options):
        return _flash_sca_bwd(dy, q, k, v, *forward, *common, deterministic=deterministic,
                              backend='auto', reset_chunk_pos_per_seq=read.reset,
                              strict_past=strict, **metadata, **options)

    # Forward: FP64 reference, empty rows, and a bit-exact repeat.
    y, lse = fwd()
    assert y.dtype == dtype and lse.dtype == torch.float32
    assert tuple(y.shape) == (*q.shape[:3], read.value_dim)
    assert tuple(lse.shape) == (q.shape[0], heads, q.shape[1])
    assert_same_bits((y, lse), fwd(), ('y', 'lse'))
    refs, mask, expected = reference_read(read, q, k, v, prev_k, prev_v, qi, ki, scale, strict)
    torch.testing.assert_close(y, expected.to(dtype), atol=ATOL[dtype], rtol=RTOL[dtype])
    live = mask.any(-1)[:, None, :].expand_as(lse)
    assert not torch.isnan(lse).any()
    assert torch.equal(y[~live.permute(0, 2, 1)], torch.zeros_like(y[~live.permute(0, 2, 1)]))
    if live.any():
        scores = torch.einsum('bqhd,bkhd->bhqk', refs[0].detach(),
                              refs[1].detach().repeat_interleave(heads // kv_heads, dim=2)) * scale
        expected_lse = scores.masked_fill(~mask[:, None], float('-inf')).logsumexp(-1)
        torch.testing.assert_close(lse[live], expected_lse[live].float(), atol=ATOL[dtype], rtol=RTOL[dtype])

    # Backward: FP64 reference and the determinism trio.
    torch.manual_seed(_seed('dy', read, dtype, heads, kv_heads))
    dy = (torch.randn_like(y, dtype=torch.float32) / math.sqrt(y.numel())).to(dtype)
    grads = bwd((y, lse), True)
    expected_grads = torch.autograd.grad(expected, refs, dy.double())
    for name, actual, reference in zip(('dq', 'dk', 'dv'), split_key_grads(grads, read.previous),
                                       expected_grads):
        assert actual.dtype == dtype and torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, reference.to(dtype), atol=GRAD_ATOL[dtype], rtol=RTOL[dtype],
                                   msg=lambda text: f'{name}: {text}')
    assert_same_bits(grads, bwd((y, lse), True), GRAD_NAMES)
    assert_same_bits(grads, bwd((y, lse), False), GRAD_NAMES)
    assert_same_bits(grads, bwd(fwd(), True), GRAD_NAMES)

    # FP32 dK/dV (grouped, inclusive, not varlen): the values the 16-bit gradients round from.
    if heads > kv_heads and not strict and read.metadata != 'varlen':
        fp32 = bwd((y, lse), True, fp32_kv_grads=True)
        assert_same_bits(fp32[:1], grads[:1], ('dq',))
        for name, wide, narrow in zip(GRAD_NAMES[1:], fp32[1:], grads[1:]):
            if narrow is None:
                assert wide is None, name
                continue
            assert wide.dtype == torch.float32 and wide.shape == narrow.shape, name
            assert torch.equal(_bits(wide.to(dtype)), _bits(narrow)), f'FP32 {name} does not round to {name}'
        assert_same_bits(fp32, bwd((y, lse), False, fp32_kv_grads=True), GRAD_NAMES)
        assert_same_bits(fp32, bwd(fwd(), True, fp32_kv_grads=True), GRAD_NAMES)


@pytest.mark.parametrize('read', READS, ids=lambda read: read.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('heads,kv_heads', HEADS, ids=('mha', 'gqa', 'gqa6'))
@pytest.mark.parametrize('strict', (True, False), ids=('strict', 'inclusive'))
def test_private_read_matches_reference(read, dtype, heads, kv_heads, strict):
    check_read(read, dtype, heads, kv_heads, strict)


HIGH_PRECISION_READS = tuple(read for read in READS if read.name in (
    'dense', 'previous', 'segment', 'reset-independent', 'varlen', 'umma-128', 'umma-256'))


@pytest.mark.parametrize('read', HIGH_PRECISION_READS, ids=lambda read: read.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('heads,kv_heads', HEADS[:2], ids=('mha', 'gqa'))
@pytest.mark.parametrize('strict', (True, False), ids=('strict', 'inclusive'))
def test_private_high_precision_forward(read, dtype, heads, kv_heads, strict):
    """high_precision: the level-1 subtract-before-scale forward, against FP64, repeating bit-exactly."""
    require_native_b300()
    q, k, v, prev_k, prev_v, metadata, qi, ki = make_read(read, dtype, heads, kv_heads)
    scale = read.dim ** -0.5

    def fwd():
        return _flash_sca_fwd(q, k, v, read.span, scale, prev_k, prev_v, backend='auto',
                              reset_chunk_pos_per_seq=read.reset, strict_past=strict, high_precision=True,
                              **metadata)
    y, lse = fwd()
    assert y.dtype == dtype and lse.dtype == torch.float32 and not torch.isnan(lse).any()
    assert_same_bits((y, lse), fwd(), ('y', 'lse'))
    _, _, expected = reference_read(read, q, k, v, prev_k, prev_v, qi, ki, scale, strict)
    torch.testing.assert_close(y, expected.to(dtype), atol=ATOL[dtype], rtol=RTOL[dtype])


# ---- the SoftDelta reads of the Gekko layers ---------------------------------------------

CHUNK = 4096
DOC_LENGTHS = pathlib.Path(__file__).resolve().parent / 'gekko_doc_lengths.json'
# The two reads of one SoftDelta layer (xllm d31a245, models/gekko.py presets): neg_read is the
# strict-past read by the read keys (xllm passes its own [B, L, Hkv, D] projection as k_read),
# ungrouped; read is the inclusive read by Q over the grouped heads, with FP32 dK/dV for the
# gradient merge.
SOFTDELTA = {
    'gekko-0.9B/neg_read': dict(batch=4, heads=4, kv_heads=4, dim=256, strict=True),
    'gekko-4B/neg_read': dict(batch=1, heads=8, kv_heads=8, dim=128, strict=True),
    'gekko-0.9B/read': dict(batch=4, heads=8, kv_heads=4, dim=256, strict=False),
    'gekko-4B/read': dict(batch=1, heads=32, kv_heads=8, dim=128, strict=False),
    'gekko-7B/read': dict(batch=1, heads=48, kv_heads=8, dim=128, strict=False),
}


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


def _segment_ids(batch, seq_len):
    """SoftDelta's segment ids of a packed-document BOS mask (bos_mask.cumsum)."""
    eos = torch.zeros(batch, seq_len, dtype=torch.bool)
    for row, ends in enumerate(_document_ends(batch, seq_len, 1)):
        eos[row, ends] = True
    bos = F.pad(eos[:, :-1], (1, 0), value=True)
    return bos.cumsum(1).cuda()


def _block_reference(q, k, v, dy, ids, scale, strict, kv_checked):
    """Exact FP64 results of complete chunk blocks: for key chunk g of every batch row (first,
    middle and last), every query of chunks g and g+1 against keys of chunks g-1..g+1. y, LSE
    and dQ hold for those queries, dK/dV for the keys of chunk g; the query heads of the KV
    heads in ``kv_checked`` only."""
    B, N, H, _ = q.shape
    G = k.shape[2]
    rep = H // G
    blocks = []
    for b in range(B):
        for g in sorted({0, N // CHUNK // 2, N // CHUNK - 1}):
            k_lo, k_hi = max(0, (g - 1) * CHUNK), min((g + 2) * CHUNK, N)
            q_lo, q_hi = g * CHUNK, min((g + 2) * CHUNK, N)
            qp = torch.arange(q_lo, q_hi, device='cuda')
            kp = torch.arange(k_lo, k_hi, device='cuda')
            upper = (kp[None] < qp[:, None]) if strict else (kp[None] <= qp[:, None])
            allowed = (kp[None] >= (qp // CHUNK * CHUNK - CHUNK)[:, None]) & upper & (
                ids[b, qp][:, None] == ids[b, kp][None])
            for kh in kv_checked:
                K, V = k[b, k_lo:k_hi, kh].double(), v[b, k_lo:k_hi, kh].double()
                dk, dv = torch.zeros_like(K), torch.zeros_like(V)
                for h in range(kh * rep, (kh + 1) * rep):
                    Q, dY = q[b, q_lo:q_hi, h].double(), dy[b, q_lo:q_hi, h].double()
                    s = (Q @ K.T * scale).masked_fill(~allowed, float('-inf'))
                    lse = torch.logsumexp(s, -1)
                    live = torch.isfinite(lse)
                    p = torch.where(live[:, None], torch.exp(s - lse[:, None]), torch.zeros_like(s))
                    del s
                    y = p @ V
                    ds = p * (dY @ V.T - (dY * y).sum(-1)[:, None])
                    dk += scale * (ds.T @ Q)
                    dv += p.T @ dY
                    blocks.append(dict(kind='query', b=b, h=h, rows=(q_lo, q_hi), y=y, lse=lse, live=live,
                                       dq=scale * (ds @ K)))
                    del p, ds
                own = slice(g * CHUNK - k_lo, min((g + 1) * CHUNK, N) - k_lo)
                blocks.append(dict(kind='key', b=b, h=kh, rows=(g * CHUNK, min((g + 1) * CHUNK, N)),
                                   dk=dk[own], dv=dv[own]))
    return blocks


def check_softdelta_read(name, length):
    require_native_b300()
    shape = SOFTDELTA[name]
    B, H, G, D = shape['batch'], shape['heads'], shape['kv_heads'], shape['dim']
    strict = shape['strict']
    if length < 65536:
        B = 1
    dtype = torch.bfloat16
    generator = torch.Generator(device='cuda').manual_seed(_seed(name, length))

    def normal(*size):
        return torch.randn(*size, generator=generator, device='cuda', dtype=torch.float32)
    k = F.normalize(normal(B, length, G, D), dim=-1).to(dtype)
    v = (F.silu(normal(B, length, G, D)) + 0.1).to(dtype)
    # Q of the read, or neg_read's read keys (H == G).
    q = F.normalize(normal(B, length, H, D), dim=-1).to(dtype)
    dy = (normal(B, length, H, D) / math.sqrt(H * D)).to(dtype)
    ids = _segment_ids(B, length)
    scale = D ** -0.5
    segments = dict(q_segment_idx=ids, k_segment_idx=ids)

    def fwd():
        return _flash_sca_fwd(q, k, v, CHUNK, scale, None, None, backend='auto', strict_past=strict,
                              **segments)

    def bwd(forward, deterministic, **options):
        return _flash_sca_bwd(dy, q, k, v, *forward, CHUNK, scale, None, None,
                              deterministic=deterministic, backend='auto', strict_past=strict,
                              **segments, **options)

    y, lse = fwd()
    assert_same_bits((y, lse), fwd(), ('y', 'lse'))
    grads = bwd((y, lse), True)
    assert grads[3] is None and grads[4] is None
    assert_same_bits(grads, bwd((y, lse), True), GRAD_NAMES)
    assert_same_bits(grads, bwd((y, lse), False), GRAD_NAMES)
    assert_same_bits(grads, bwd(fwd(), True), GRAD_NAMES)
    if not strict:
        fp32 = bwd((y, lse), True, fp32_kv_grads=True)
        assert_same_bits(fp32[:1], grads[:1], ('dq',))
        for wide, narrow, label in zip(fp32[1:3], grads[1:3], ('dk', 'dv')):
            assert wide.dtype == torch.float32
            assert torch.equal(_bits(wide.to(dtype)), _bits(narrow)), f'FP32 {label} does not round to {label}'
        assert_same_bits(fp32, bwd((y, lse), False, fp32_kv_grads=True), GRAD_NAMES)
        assert_same_bits(fp32, bwd(fwd(), True, fp32_kv_grads=True), GRAD_NAMES)
        del fp32
    dq, dk, dv = grads[:3]
    for block in _block_reference(q, k, v, dy, ids, scale, strict, sorted({0, G - 1})):
        lo, hi = block['rows']
        b, h = block['b'], block['h']
        if block['kind'] == 'query':
            live = block['live']
            torch.testing.assert_close(y[b, lo:hi, h], block['y'].to(dtype), atol=ATOL[dtype], rtol=RTOL[dtype])
            torch.testing.assert_close(lse[b, h, lo:hi][live], block['lse'][live].float(), atol=ATOL[dtype],
                                       rtol=RTOL[dtype])
            assert not torch.isnan(lse[b, h, lo:hi]).any()
            torch.testing.assert_close(dq[b, lo:hi, h], block['dq'].to(dtype), atol=GRAD_ATOL[dtype],
                                       rtol=RTOL[dtype])
        else:
            torch.testing.assert_close(dk[b, lo:hi, h], block['dk'].to(dtype), atol=GRAD_ATOL[dtype],
                                       rtol=RTOL[dtype])
            torch.testing.assert_close(dv[b, lo:hi, h], block['dv'].to(dtype), atol=GRAD_ATOL[dtype],
                                       rtol=RTOL[dtype])


@pytest.mark.parametrize('name', list(SOFTDELTA))
def test_softdelta_reads_two_chunks(name):
    check_softdelta_read(name, 2 * CHUNK)


@pytest.mark.long
@pytest.mark.parametrize('name', list(SOFTDELTA))
def test_softdelta_reads_training_length(name):
    check_softdelta_read(name, 65536)


# ---- what the private entry points refuse on B300 ----------------------------------------

def _small(heads, kv_heads, dtype=torch.bfloat16):
    torch.manual_seed(5)
    q = F.normalize(torch.randn(1, 128, heads, 64, device='cuda'), dim=-1).to(dtype)
    k = F.normalize(torch.randn(1, 128, kv_heads, 64, device='cuda'), dim=-1).to(dtype)
    v = (F.silu(torch.randn(1, 128, kv_heads, 64, device='cuda')) + 0.1).to(dtype)
    return q, k, v


def test_private_refusals():
    """Each refusal raises before anything runs, with upstream's words where upstream has them.
    One native read runs first, so that the autouse fixture's native-launch proof holds."""
    require_native_b300()
    q, k, v = _small(4, 4)
    y, lse = _flash_sca_fwd(q, k, v, 64, 0.125, backend='auto', strict_past=True)
    dy = torch.ones_like(y)
    with pytest.raises(RuntimeError, match='grouped query heads'):
        _flash_sca_bwd(dy, q, k, v, y, lse, 64, 0.125, backend='auto', fp32_kv_grads=True)
    with pytest.raises(ValueError, match='dense, non-strict'):
        _flash_sca_bwd(dy, q, k, v, y, lse, 64, 0.125, backend='auto', strict_past=True, fp32_kv_grads=True)
    cu = torch.tensor([0, 64, 128], device='cuda', dtype=torch.int32)
    varlen = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=64, max_seqlen_k=64)
    qg, kg, vg = _small(8, 2)
    yg, lseg = _flash_sca_fwd(qg, kg, vg, 64, 0.125, backend='auto', **varlen)
    with pytest.raises(ValueError, match='dense, non-strict'):
        _flash_sca_bwd(torch.ones_like(yg), qg, kg, vg, yg, lseg, 64, 0.125, backend='auto',
                       fp32_kv_grads=True, **varlen)
    with pytest.raises(NotImplementedError, match='output_state'):
        _flash_sca_fwd(q, k, v, 64, 0.125, backend='auto',
                       output_state=torch.empty(*q.shape[:3], 64, device='cuda'))
    rows = torch.tensor([0, 64, 96, 128], device='cuda', dtype=torch.int32)
    with pytest.raises(NotImplementedError, match='one key run per query run'):
        _flash_sca_fwd(q, k, v, 64, 0.125, backend='auto', cu_seqlens_q=cu, cu_seqlens_k=rows,
                       max_seqlen_q=64, max_seqlen_k=64)
