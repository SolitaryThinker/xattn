"""The training calls xllm makes with attn_act_func='softdelta', on B300, against FP64 on complete chunk blocks.

xllm main d31a245 (read, not run): modules/gated_delta_attention.py builds q (wq, [B, L, H*D]), the key
projection wk of width 2 * Hkv * D and the gate projection wr of width H * (Dv + 1); g = rg[:, :, :, -1]
with rg = wr(x) viewed [B, L, H, Dv + 1] (a strided [B, L, H] view). modules/sliding_chunk_attention.py
(SlidingChunkSoftdeltaAttention) splits the keys [B, L, Hkv, 2D] into the read keys kq = xk[..., :D] and
the keys xk[..., D:], applies RoPE to q, kq and xk (contiguous [B, L, *, D] results; without RoPE they
stay strided views of the projection) and, for L >= chunk, calls fused_ops/attention/softdelta/sca.py,
which calls xattn.sliding_chunk_softdelta_attention(q, k=xk, k_read=kq, v [B, L, Hkv, Dv], g, chunk_size,
scale, prev_k, prev_v, bos_mask=bos_mask, segment_idx=None, high_precision_level=0, deterministic=True)
with autograd. Shorter tails call the causal softdelta_attention (test_b300_softdelta_causal.py).

Shapes, BF16, BOS masks packed from validation-document lengths as the SCA shape check packs
them: the Gekko presets 0.9B (traced shape), 4B (traced), 7B (preset), the MoE stand-ins, and the ladder
stand-ins (head dim 64, chunks 1024 and 2048, unequal dims), all not traces except where said. By default
each runs at two chunks and one batch row; --long runs the training size. Each case: y, the negative
reads, both LSEs and every gradient (q, k, k_read, v, g; previous K/V where given) against FP64 on complete
chunk blocks with upstream's tolerances; the forward repeats bit-exactly, the backward repeats bit-exactly
from the same and from a fresh forward, and deterministic=False gives the bits of deterministic=True.
"""
import hashlib
import json
import math
import pathlib
import random

import pytest
import torch
import torch.nn.functional as F

from xattn import sliding_chunk_softdelta_attention, sliding_chunk_softdelta_attention_fwd

from softdelta_main_cases import GRAD_REL_NORM, autograd_determinism, require_native_b300
from softdelta_main_reference import ATOL, GRAD_ATOL, RTOL

DOC_LENGTHS = pathlib.Path(__file__).resolve().parent / 'gekko_doc_lengths.json'
# name: (batch, length, query heads, KV heads, D, Dv, chunk, source)
SHAPES = {
    'gekko-0.9B/train': (4, 65536, 8, 4, 256, 256, 4096, 'traced'),
    'gekko-4B/train': (1, 65536, 32, 8, 128, 128, 4096, 'traced'),
    'gekko-7B/train': (1, 65536, 48, 8, 128, 128, 4096, 'preset'),
    'moe-mp2/train': (16, 8192, 16, 4, 128, 128, 4096, 'standin'),
    'moe-mp2-long/train': (4, 32768, 16, 4, 128, 128, 4096, 'standin'),
    'moe-mp8/train': (16, 8192, 4, 1, 128, 128, 4096, 'standin'),
    'ladder-100M/train': (64, 4096, 8, 2, 64, 64, 1024, 'standin'),
    'ladder-300M/train': (32, 8192, 16, 4, 64, 64, 2048, 'standin'),
    'ladder-1B/train': (8, 16384, 32, 8, 64, 64, 4096, 'standin'),
    'ladder-2B/train': (4, 32768, 16, 4, 128, 128, 2048, 'standin'),
    'mixed-64-128/train': (8, 16384, 16, 4, 64, 128, 4096, 'standin'),
}


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'little')


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


def make_bos(batch, length, seed=1):
    """xllm's per-call bos_mask (get_bos_mask_from_eos) over packed validation documents."""
    eos = torch.zeros(batch, length, dtype=torch.bool)
    for row, ends in enumerate(_document_ends(batch, length, seed)):
        eos[row, ends] = True
    return F.pad(eos[:, :-1], (1, 0), value=True).cuda()


def make_call(name, batch, length, previous=False):
    """xllm's tensors: q, the key projection [B, L, Hkv, 2D] (read key ; key per KV head), v, the gate
    projection rg [B, L, H, Dv + 1] whose last channel is g, the BOS mask, dY and, with ``previous``, one
    chunk of previous K/V (the context-parallel rank's history)."""
    _, _, H, G, D, V, C, _ = SHAPES[name]
    generator = torch.Generator(device='cuda').manual_seed(_seed(name, batch, length, previous))

    def normal(*size):
        return torch.randn(*size, generator=generator, device='cuda', dtype=torch.float32)

    def unit(*size):
        return F.normalize(normal(*size), dim=-1).to(torch.bfloat16)
    parts = dict(q=unit(batch, length, H, D), keys=unit(batch, length, G, 2 * D),
                 v=(F.silu(normal(batch, length, G, V)) + 0.1).to(torch.bfloat16),
                 rg=(normal(batch, length, H, V + 1) * 0.75).to(torch.bfloat16))
    dy = (normal(batch, length, H, V) / math.sqrt(H * V)).to(torch.bfloat16)
    total = length
    if previous:
        parts['prev_k'] = unit(batch, C, G, D)
        parts['prev_v'] = (F.silu(normal(batch, C, G, V)) + 0.1).to(torch.bfloat16)
        total += C
    bos = make_bos(batch, total)
    if previous:
        bos[:, C] = False                         # the document continues across the call
    return parts, dict(chunk_size=C, scale=D ** -0.5, bos_mask=bos), dy


def call_tensors(parts, layout):
    """The call's q, k, k_read, v, g from xllm's tensors: g = rg[..., -1] (strides (L H (Dv + 1),
    H (Dv + 1), Dv + 1)); the read keys and keys contiguous after RoPE ('post-rope') or strided views
    of the projection ('views', a model without RoPE)."""
    D = parts['q'].shape[-1]
    if layout == 'views':
        k_read, k = parts['keys'][..., :D], parts['keys'][..., D:]
    else:
        k_read, k = parts['k_read'], parts['k']
    return dict(q=parts['q'], k=k, k_read=k_read, v=parts['v'], g=parts['rg'][..., -1],
                prev_k=parts.get('prev_k'), prev_v=parts.get('prev_v'))


def block_reference(call, options, dy, rows, kv_checked):
    """Exact FP64 SoftDelta on complete chunk blocks: for key chunk c of a batch row (first, middle, last
    of the call's key axis [prev ; current]), every query of chunks c and c+1 against the keys of chunks
    c-1..c+1. y, the negative reads, both LSEs and dq / dk_read / dg hold for those queries, dK / dV (or
    dPrevK / dPrevV) for the keys of chunk c; the KV heads in ``kv_checked`` and their query heads only."""
    q, k, k_read, v, g = (call[n] for n in ('q', 'k', 'k_read', 'v', 'g'))
    C, scale = options['chunk_size'], options['scale']
    keys = k if call['prev_k'] is None else torch.cat((call['prev_k'], k), 1)
    values = v if call['prev_v'] is None else torch.cat((call['prev_v'], v), 1)
    ids = options['bos_mask'].cumsum(1)
    N, T = q.shape[1], keys.shape[1]
    offset = T - N
    H, G = q.shape[2], keys.shape[2]
    rep = H // G
    blocks = []
    for b in rows:
        for c in sorted({0, T // C // 2, T // C - 1}):
            k_lo, k_hi = max(0, (c - 1) * C), min((c + 2) * C, T)
            q_lo, q_hi = max(c * C, offset), min((c + 2) * C, T)
            if q_lo >= q_hi:
                continue
            qp = torch.arange(q_lo, q_hi, device='cuda')
            kp = torch.arange(k_lo, k_hi, device='cuda')
            base = (kp[None] >= (qp // C * C - C)[:, None]) & (ids[b, qp][:, None] == ids[b, kp][None])
            inclusive, strict = base & (kp[None] <= qp[:, None]), base & (kp[None] < qp[:, None])
            rows_q = slice(q_lo - offset, q_hi - offset)
            for kh in kv_checked:
                K, Vv = keys[b, k_lo:k_hi, kh].double(), values[b, k_lo:k_hi, kh].double()
                dK, dV = torch.zeros_like(K), torch.zeros_like(Vv)

                def read(Q, mask):
                    s = (Q @ K.T * scale).masked_fill(~mask, float('-inf'))
                    lse = torch.logsumexp(s, -1)
                    live = torch.isfinite(lse)
                    p = torch.where(live[:, None], torch.exp(s - lse[:, None]), torch.zeros_like(s))
                    return p, p @ Vv, lse

                def backward(Q, p, out, d_out):
                    ds = p * (d_out @ Vv.T - (d_out * out).sum(-1)[:, None])
                    dK.add_(scale * (ds.T @ Q))
                    dV.add_(p.T @ d_out)
                    return scale * (ds @ K)
                Kr = k_read[b, rows_q, kh].double()
                p_neg, neg, lse_neg = read(Kr, strict)
                d_neg = torch.zeros_like(neg)
                heads = {}
                for h in range(kh * rep, (kh + 1) * rep):
                    Q, dY = q[b, rows_q, h].double(), dy[b, rows_q, h].double()
                    gate = torch.sigmoid(g[b, rows_q, h].double())[:, None]
                    p, out, lse = read(Q, inclusive)
                    d_neg -= gate * dY
                    heads[h] = dict(y=out - gate * neg, read=out, lse=lse, dq=backward(Q, p, out, dY),
                                    dg=-(gate * (1 - gate))[:, 0] * (dY * neg).sum(-1), gate=gate)
                    del p
                dk_read = backward(Kr, p_neg, neg, d_neg)
                own = slice(c * C - k_lo, min((c + 1) * C, T) - k_lo)
                blocks.append(dict(b=b, kh=kh, query_rows=rows_q, key_rows=(c * C, min((c + 1) * C, T)), heads=heads,
                                   neg=neg, lse_neg=lse_neg, dk_read=dk_read, dk=dK[own], dv=dV[own]))
    return blocks


def assert_grad_close(name, actual, expected):
    dtype = actual.dtype
    torch.testing.assert_close(actual, expected.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype],
                               msg=lambda m: f'{name}: {m}')
    floor = GRAD_ATOL[dtype] * expected.numel() ** 0.5
    relative = ((actual.double() - expected).norm() / expected.norm().clamp_min(floor)).item()
    assert relative < GRAD_REL_NORM[dtype], f'{name}: relative norm error {relative}'


def key_rows(grad, prev_grad, offset, key_rows_range):
    lo, hi = key_rows_range
    if offset and lo < offset:
        return prev_grad[lo:hi]
    return grad[lo - offset:hi - offset]


def check_training_call(name, batch, length, previous=False, layout='post-rope'):
    require_native_b300()
    parts, options, dy = make_call(name, batch, length, previous)
    dtype = torch.bfloat16
    D = parts['q'].shape[-1]
    if layout != 'views':
        parts['k_read'], parts['k'] = parts['keys'][..., :D].contiguous(), parts['keys'][..., D:].contiguous()
        del parts['keys']
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in parts.items()}

    def run(deterministic):
        # sca.py: xattn_softdelta(q, kk, qk, v, g, chunk_size, scale, prev_k, prev_v, bos_mask=bos_mask,
        # segment_idx=None, high_precision_level=0, deterministic=deterministic)
        return sliding_chunk_softdelta_attention(**call_tensors(leaves, layout), segment_idx=None,
                                                 high_precision_level=0, deterministic=deterministic, **options)
    y = run(True)
    grads = torch.autograd.grad(y, list(leaves.values()), dy)
    autograd_determinism(run, leaves, dy, True, y, grads)
    with torch.no_grad():
        y_state, neg_state, lse = sliding_chunk_softdelta_attention_fwd(**call_tensors(parts, layout), **options)
    assert torch.equal(y_state, y)
    by_leaf = dict(zip(leaves, grads))
    assert not by_leaf['rg'][..., :-1].any(), 'gradient outside the gate channel of rg'
    gradients = dict(q=by_leaf['q'], v=by_leaf['v'], g=by_leaf['rg'][..., -1],
                     prev_k=by_leaf.get('prev_k'), prev_v=by_leaf.get('prev_v'))
    if layout == 'views':
        gradients.update(k_read=by_leaf['keys'][..., :D], k=by_leaf['keys'][..., D:])
    else:
        gradients.update(k_read=by_leaf['k_read'], k=by_leaf['k'])
    call = call_tensors(parts, layout)
    G = call['k'].shape[2]
    offset = call['prev_k'].shape[1] if previous else 0
    rows = sorted({0, batch // 2, batch - 1})
    for block in block_reference(call, options, dy, rows, sorted({0, G - 1})):
        b, kh, rq = block['b'], block['kh'], block['query_rows']
        tag = f"b{b}/kv{kh}/keys{block['key_rows']}"
        torch.testing.assert_close(neg_state[b, rq, kh], block['neg'].to(dtype), atol=ATOL[dtype], rtol=RTOL[dtype],
                                   msg=lambda m: f'{tag} neg_read: {m}')
        live = torch.isfinite(block['lse_neg'])
        torch.testing.assert_close(lse[b, kh, rq][live], block['lse_neg'][live].float(), atol=ATOL[dtype],
                                   rtol=RTOL[dtype])
        assert torch.equal(torch.isneginf(lse[b, kh, rq]), ~live), f'{tag}: empty strict rows'
        for h, ref in block['heads'].items():
            # The composed bound of upstream's _assert_composed_close.
            difference = (y[b, rq, h].double() - ref['y']).abs()
            bound = (ATOL[dtype] + RTOL[dtype] * ref['read'].abs()
                     + ref['gate'] * (ATOL[dtype] + RTOL[dtype] * block['neg'].abs())
                     + ATOL[dtype] + RTOL[dtype] * ref['y'].abs())
            assert bool((difference <= bound).all()), f'{tag} h{h} y: max normalized {(difference / bound).max().item()}'
            torch.testing.assert_close(lse[b, G + h, rq], ref['lse'].float(), atol=ATOL[dtype], rtol=RTOL[dtype])
            assert_grad_close(f'{tag} h{h} dq', gradients['q'][b, rq, h], ref['dq'])
            assert_grad_close(f'{tag} h{h} dg', gradients['g'][b, rq, h], ref['dg'])
        assert_grad_close(f'{tag} dk_read', gradients['k_read'][b, rq, kh], block['dk_read'])
        prev_k_grad = gradients['prev_k'][b, :, kh] if previous else None
        prev_v_grad = gradients['prev_v'][b, :, kh] if previous else None
        assert_grad_close(f'{tag} dk', key_rows(gradients['k'][b, :, kh], prev_k_grad, offset, block['key_rows']),
                          block['dk'])
        assert_grad_close(f'{tag} dv', key_rows(gradients['v'][b, :, kh], prev_v_grad, offset, block['key_rows']),
                          block['dv'])


@pytest.mark.parametrize('name', list(SHAPES))
def test_training_call_two_chunks(name):
    """Every training shape's heads, dims and chunk at two chunks of one batch row."""
    check_training_call(name, 1, 2 * SHAPES[name][6])


@pytest.mark.parametrize('name', ('gekko-0.9B/train', 'gekko-4B/train', 'ladder-1B/train', 'mixed-64-128/train'))
def test_training_call_previous_chunk(name):
    """The context-parallel form: one chunk of previous K/V (dPrevK / dPrevV live), the document
    continuing across the call."""
    check_training_call(name, 1, 2 * SHAPES[name][6], previous=True)


@pytest.mark.parametrize('name', ('gekko-4B/train', 'ladder-300M/train'))
def test_training_call_projection_views(name):
    """Without RoPE xllm passes the read keys and the keys as strided views of one projection."""
    check_training_call(name, 2, 2 * SHAPES[name][6], layout='views')


@pytest.mark.long
@pytest.mark.parametrize('name', list(SHAPES))
def test_training_call_full_size(name):
    batch, length = SHAPES[name][:2]
    check_training_call(name, batch, length)
