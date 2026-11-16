# SoftDelta Attention

[Back to supported attention](../README.md#supported-attention)

```python
from xattn import softdelta_attention

k_read = None

y = softdelta_attention(
    q, k, k_read, v, g,
    scale=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
    high_precision_level=0,
)
```

Tensor layouts:

- `q`: `[batch, seqlen, q_heads, qk_dim]`
- `k`: `[batch, seqlen, kv_heads, qk_dim]`
- `k_read`: Tensor or `None`: `[batch, seqlen, kv_heads, qk_dim]`
- `v`: `[batch, seqlen, kv_heads, v_dim]`
- `g`: `[batch, seqlen, q_heads]`, gate logits
- `q_segment_idx`: optional int64 `[batch, seqlen]`
- `k_segment_idx`, `segment_idx`: optional int64 `[batch, seqlen]`
- `bos_mask`: optional CUDA bool `[batch, seqlen]`
- `high_precision_level`: `0` or `1`

Standalone forward/backward:

```python
from xattn import softdelta_attention_fwd, softdelta_attention_bwd

k_read = None

y, neg_read, lse = softdelta_attention_fwd(
    q, k, k_read, v, g,
    scale=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
    high_precision_level=1,
)
dq, dk, dv, dg, dk_read = softdelta_attention_bwd(
    dy, q, k, k_read, v, g, y, neg_read, lse,
    scale=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
    high_precision_level=1,
)
```

Output layouts:

- `y`, `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `neg_read`: `[batch, seqlen, kv_heads, v_dim]`, input dtype at level 0, FP32 at level 1
- `lse`: `[batch, kv_heads + q_heads, seqlen]`, FP32, the negative reads first
- `dq`, `dk`, `dv`, `dg`: same shapes and dtypes as `q`, `k`, `v`, `g`
- `dk_read`: same shape and dtype as `k_read`, or `None` when `k_read=None`

## Blackwell (sm_103a)

On a compute capability 10.x GPU with an extension built for `sm_103a`,
`backend="auto"` runs causal SoftDelta at `high_precision_level=0` on the
native Blackwell FlashSCA reads, each read as one sliding chunk holding every
key (with short queries right-aligned to the keys, segment ids and BOS masks),
composed with the gate, restore and gradient-merge kernels. Every call is
deterministic whatever `deterministic` is set to. `high_precision_level=1`
raises `NotImplementedError` there before anything runs; a sliding window
(`sliding_window_softdelta_attention`) that does not cover every earlier key
raises `NotImplementedError` too, and a covering one runs as causal.
