# SoftDelta on B300: the composed native route

Upstream xattn main `b112365`'s SoftDelta attention runs natively and deterministically on B300 (compute capability
10.3, `sm_103a`) at precision level 0, composed from the native sliding-chunk reads (the private `_flash_sca_fwd` /
`_flash_sca_bwd`, PORT.md) and the gate, restore and gradient-merge kernels of `csrc/softdelta/gate.cu`. No SCA kernel,
header or SCA Python module is changed by it: its diff is `xattn/ops/softdelta_attention.py`, the two SoftDelta docs,
`tests_b300/softdelta_main/` and this file. The reads it calls are the combined tree's (COMBINED.md): the forward and
backward optimizations of the SCA kernels apply to both reads.

## What runs natively, what raises

On capability 10.x, `backend='auto'` (or `''`) resolves to the native composition (`softdelta_attention._route`).

| entry point / option | B300 |
|---|---|
| `sliding_chunk_softdelta_attention` (autograd), `_fwd`, `_bwd`, level 0 | native |
| `k_read` given or `None` (the aligned keys; their query-side gradient folded into `dk` by the merge) | native |
| `prev_k`/`prev_v` (exactly one chunk), `dprev_k`/`dprev_v` | native |
| `bos_mask`, `segment_idx`, `q_segment_idx`/`k_segment_idx` (sorted or not), short Q (right-aligned) | native |
| `reset_chunk_pos_per_seq`, including Q ids that are not K's suffix (SM90's matched-reset bridge is not used: the native reads serve it, FP32 dK/dV included) | native |
| FP16, BF16; D, Dv multiples of 8 up to 256 (D != Dv allowed); MHA, GQA, MQA | native |
| nonpositive scales (negated or zeroed queries, as upstream) | native |
| `deterministic=False` | identical bits to `True` (the reads are deterministic only; the gate and merge kernels have no atomics) |
| `softdelta_attention` (causal, autograd), `_fwd`, `_bwd`, level 0 | native: each read is one sliding chunk holding every key (chunk = K/V length, so every position is in chunk 0, whose rule is the causal one) |
| a sliding window or chunk that covers every earlier key | native, as causal (upstream's `_effective_local`) |
| `high_precision_level=1` (every API) | `NotImplementedError` before anything runs: it needs the reads' FP32 output state (`output_state`, unrouted) and the strict-past precision-repair backward |
| `sliding_window_softdelta_attention*` with a window that does not cover every earlier key | `NotImplementedError` before anything runs (no native window read) |
| `backend='sm90'` | `RuntimeError` (as upstream on a non-SM90 device) |
| `backend='torch'` | the PyTorch reference, as asked; never chosen otherwise on B300 |

Without this route, on B300 the autograd APIs silently ran the PyTorch reference (`_call` sent every non-SM90 device
to `_torch_softdelta`) and the standalone `_fwd`/`_bwd` raised "requires the SM90 CUDA backend".

## The composition

Forward (`_forward`, level 0, no fused SM90 launch on B300):

1. the grouped inclusive read by `q`: `_flash_sca_fwd(q, k, v, chunk, scale, prev_k, prev_v, q_ids, k_ids, 'auto',
   reset)` (native; segment ids from `bos_mask.cumsum(1)`);
2. the strict-past read by the read keys (Hkv heads, ungrouped): `_flash_sca_fwd(k_read, ..., strict_past=True)`;
3. the gate, `_softdelta_gate_fwd`: `y = read - sigmoid(g) * rep(neg_read)` in FP32, rounded once;
4. `lse = cat(neg_lse, read_lse)`. Saved for the backward: `y`, `neg_read`, `lse`.

Backward (`_backward`): `_softdelta_gate_bwd` (the negative read's gradient, summed over each group's query heads in a
fixed order, and the gate gradient), `_softdelta_gate_restore` (the read from `y`), the grouped read's backward with FP32
dK/dV (`fp32_kv_grads`, grouped heads), the strict read's backward, and `_softdelta_merge_gradients` (FP32 sum of both
reads' dK/dV and the folded read-key gradient, rounded once; the previous chunk's gradients the same way).

## Gate and merge kernels on B300 (`csrc/softdelta/gate.cu`)

Unchanged. They are per-element or per-row kernels with fixed reduction order (the backward accumulates a group's query
heads in a loop and reduces the gate gradient's dot product with a fixed butterfly of `__shfl_xor_sync`); no atomics.
Checked on B300: 760 gate cases (fwd, restore, bwd; FP16/BF16; 9 head mappings; V in 8..256 including unaligned widths;
contiguous, readers-slice and FP32-state layouts; the training sizes) within one unit in the last place of FP64 (plus
the FP32 arithmetic's own rounding where terms cancel), and 1154 merge cases bit-identical to the same FP32 sum in
PyTorch; every op repeated 3 times bit-exactly; the combined SHA-256 of all outputs identical in 3 fresh processes on
three GPUs.

## xLLM's calls (main `d31a245`, read, not run)

- `xllm/modules/gated_delta_attention.py`: `nk = 2` for softdelta, so `wk` projects `2 * Hkv * D` per token; `wr`
  projects `H * Dv + H`; `g = rg[:, :, :, -1]` with `rg = rearrange(r, 'b l (h v) -> b l h v')`, a strided `[B, L, H]`
  view (strides `L*H*(Dv+1)`, `H*(Dv+1)`, `Dv+1`).
- `xllm/modules/sliding_chunk_attention.py` `SlidingChunkSoftdeltaAttention.forward`: `xq` `[B, L, H, D]`; the keys
  `[B, L, Hkv, 2D]` split into the read key `kq = xk[..., :D]` and the key `xk = xk[..., D:]`; `xv` `[B, L, Hkv, Dv]`;
  RoPE on `xq`, `kq`, `xk`; for `L >= chunk`, `sliding_chunk_softdelta_attention(xq, kq, xk, xv, g, chunk, scale,
  prev_k, prev_v, bos_mask, segment_idx, ...)`; for a tail shorter than a chunk, the keys and values become
  `[prev ; current]` and the causal `softdelta_attention`.
- `xllm/modules/fused_ops/attention/softdelta/sca.py`: `sliding_chunk_softdelta_attention(q, k=xk, k_read=kq, ...)`;
  `segment_idx` is dropped when a BOS mask is given. `high_precision_level` is the model's `attn_stability_control`
  (0 by default), `deterministic` defaults to True.
- Presets (`xllm/models/gekko.py`): gekko-0.9B 8/4 heads, D = Dv = 256; gekko-4B 32/8, 128; gekko-7B 48/8, 128; chunk
  4096. `attn_act_func='softdelta'` is a config option; no preset sets it. The Gekko fused block always runs the
  softmax SCA, so SoftDelta training goes through the module path above (autograd `sliding_chunk_softdelta_attention`).

## Tests (`tests_b300/softdelta_main`)

`pytest tests_b300/softdelta_main` (options as `tests_b300/sca_main`: `--full-matrix`, `--long`, `--shard-count/-index`):

- `test_b300_softdelta_sliding_chunk.py`: the sliding-chunk cases of upstream's
  `tests/softdelta/test_local_softdelta_attention.py` (window variants dropped);
- `test_b300_softdelta_causal.py`: upstream's `test_softdelta_attention.py` and `test_value_groups.py` (the internal
  value-group function, causal and chunk support; its window support asserts the refusal);
- `test_b300_softdelta_training.py`: xLLM's training call at the eleven training shapes (below), at two chunks of one
  row by default (plus a previous-chunk and a projection-view variant), at the training size under `--long`, against
  FP64 on complete chunk blocks (first, middle and last key chunk of three rows; y, the negative reads, both LSEs, dq,
  dk, dk_read, dv, dg);
- `test_b300_softdelta_refusals.py`: level 1, non-covering windows, `backend='sm90'`; a covering window equals the
  causal and covering-chunk calls bitwise.

Ported so that nothing skips: the SM90 gate becomes the B300 gate, calls take `backend='auto'`, the FP64 oracle and the
tolerances are upstream's (verbatim copies in `softdelta_main_reference.py`). Every level-0 parametrization runs
natively and adds the determinism trio (the forward repeats bit-exactly; the backward repeats bit-exactly from the same
forward and from a fresh one; `deterministic=False` gives the bits of `True`, forward and backward); every level-1
parametrization asserts the refusal, nothing having run. Upstream tests that only exercise SM90 knobs
(`_BWD_READER_GROUPS`, the BOS varlen router) keep their cases, which run natively; `test_fused_path_is_used` becomes
`test_native_composition_is_used` (the gate, bwd, restore and merge ops are called, in that order).

The autouse fixture (`conftest.py`) fails a test that ran the PyTorch reference (a spy on `_torch_softdelta`), resolved a
SoftDelta call (`softdelta_route` records) or a read (`sca_route` records of `_flash_sca_fwd/_bwd`) to anything but
native, launched no native forward (unless its only calls were refused, which must then have launched nothing), or ran a
read backward without a native backward launch. `SOFTDELTA_MAIN_ROUTE_LOG` keeps one JSON line per test with the
native phase tags.

Not ported: the sliding-window variants of `test_local_softdelta_attention.py` (no native window read; a non-covering
window raises, a covering one is tested as causal) and the window visibility of `test_reader_backward_fp32_kv_grads`.

Results on B300 (one shard per GPU), 0 failed, 0 errors, 0 skipped:

| run | tests | native (each with the determinism trio) | refusals only | explicit torch reference |
|---|---|---|---|---|
| default (`tests_b300/softdelta_main`) | 2885 | 1487 | 1394 | 4 |
| `--full-matrix --long` | 3886 | 1974 | 1908 | 4 |

Every native test runs both reads' forwards on the pipelined tcgen05 kernel, and every backward phase on the tcgen05
kernels except in the two gate-extreme tests, whose backward runs the `mma.sync` body. Most refusal-only tests are
upstream's level-1 parametrizations (precision level 1 is half of upstream's mode matrix). Run by a check harness
from a frozen copy (full matrix and long, a skip fails) the suite gives the same 3886/3886, and xLLM's training call at
the eleven training shapes at full size (FP64 blocks, the determinism trio, no PyTorch attention operator inside a
call) passes 11/11. Before the fix to the backward key leaves (COMBINED.md), 8 value-group cases at D = 192 with
chunk 32 failed (dK of the strict read).

## The SCA reads under it

The SCA contract (upstream's `tests/sca` on the native path at the full matrix and long, the deterministic-only,
private-entry-point and group-ratio-six suites; a skip fails) passes 11017/11017, and the Gekko-shape check (the traced
xLLM calls and the stand-ins against FP64, determinism required) 34/34. Each direction times as its optimization line
does, row by row, on one GPU (COMBINED.md).

## Determinism

- Within a process: every native test case runs the trio (forward repeat; backward repeat from the same and from a
  fresh forward; `deterministic=False` equal to `True`, forward and backward), at the upstream shapes and at the eleven
  training shapes (two chunks by default, full size under `--long` and in the full-size shape check).
- Across processes: the timing below requires every timed and warm-up call to repeat the first call's bits and the
  first call's SHA-256 to agree across 3 fresh processes: 145 rows (the eleven shapes' fwd, bwd, autograd layer and
  the composition's parts), bit-identical.
- Across GPUs and builds: the 134 first-call digests of two separate builds of this tree, timed on two GPUs, are
  identical.
- The gate and merge kernels alone: identical combined SHA-256 over 1914 cases in 3 fresh processes on three GPUs.
- Not checked: across nodes.

## Timing

Protocol of the SCA speed check: 3 fresh processes x 4 rounds x 2 repetitions, L2 flushed and the device synchronized
before every repetition (so the host's issue time is inside each timed call), medians. `fwd` =
`sliding_chunk_softdelta_attention_fwd`, `bwd` = `_bwd` from a saved forward, at xLLM's layout and BOS packing;
`2 x fwd + bwd` is a training layer that recomputes the forward. Absolute times depend on the clock a GPU sustains (a
B300 under its power cap can run its SM clock well below 2032 MHz at these loads), so this tree is compared with the
native port's reads (the same composition on the reads before the forward and backward optimizations) in the speed
check's paired mode: fresh process pairs alternating the two trees on one GPU. The last column scales the port's times,
measured on a GPU held at 2032 MHz, by those ratios. No row of the comparison was slower than its bound.

| shape (B x N, Hq/Hkv, D/Dv, chunk) | port fwd / bwd (ms) | this tree / port, one GPU: fwd / bwd / 2 x fwd + bwd | this tree at the port's conditions (ms): fwd / bwd / 2 x fwd + bwd |
|---|---|---|---|
| gekko-0.9B (4 x 65536, 8/4, 256/256, 4096) | 8.91 / 45.75 | 0.783 / 0.680 / 0.709 | 6.98 / 31.11 / 45.09 |
| gekko-4B (1 x 65536, 32/8, 128/128, 4096) | 5.33 / 20.90 | 0.548 / 0.458 / 0.492 | 2.92 / 9.58 / 15.54 |
| gekko-7B (1 x 65536, 48/8, 128/128, 4096) | 7.25 / 28.83 | 0.557 / 0.461 / 0.493 | 4.04 / 13.29 / 21.37 |
| moe-mp2 (16 x 8192, 16/4, 128/128, 4096) | 4.23 / 16.12 | 0.570 / 0.474 / 0.506 | 2.41 / 7.64 / 12.44 |
| moe-mp2-long (4 x 32768, 16/4, 128/128, 4096) | 4.87 / 18.44 | 0.573 / 0.490 / 0.518 | 2.79 / 9.04 / 14.61 |
| moe-mp8 (16 x 8192, 4/1, 128/128, 4096) | 1.50 / 5.46 | 0.682 / 0.447 / 0.531 | 1.02 / 2.44 / 4.48 |
| ladder-100M (64 x 4096, 8/2, 64/64, 1024) | 3.99 / 15.26 | 0.349 / 0.239 / 0.277 | 1.39 / 3.65 / 6.45 |
| ladder-300M (32 x 8192, 16/4, 64/64, 2048) | 10.42 / 43.11 | 0.273 / 0.208 / 0.229 | 2.84 / 8.95 / 14.64 |
| ladder-1B (8 x 16384, 32/8, 64/64, 4096) | 15.28 / 65.46 | 0.249 / 0.193 / 0.210 | 3.80 / 12.61 / 20.21 |
| ladder-2B (4 x 32768, 16/4, 128/128, 2048) | 4.09 / 14.75 | 0.566 / 0.500 / 0.523 | 2.31 / 7.37 / 11.99 |
| mixed-64-128 (8 x 16384, 16/4, 64/128, 4096) | 10.31 / 40.70 | 0.238 / 0.184 / 0.202 | 2.46 / 7.48 / 12.40 |

The 0.9B and 4B autograd training calls (`layer`: forward and backward as xLLM runs them) take 0.695 and 0.491 of the
port's 54.74 and 26.24 ms (38.0 and 12.9 ms). The gains follow the reads: the forward line's pipelined forward now
serves every head dim (the port ran D = 64 and unequal dims on the `mma.sync` body) and the backward line's
pipelined backward does too, so the shapes with D = 64 (ladder-100M to 1B) and D 64 / Dv 128 gain 2.9 to 5.4x, the
D = 128 and 256 shapes 1.3 to 2.2x.

The composition's parts, timed alone the same way (this tree on one GPU whose SM clock ran 1.7 to 2.0 GHz under its
power cap, the port on a GPU held at 2.03 GHz; compare shares within a row, not times across rows; ms):

| shape | tree | fwd | fwd parts: read / 2nd read / gate | bwd | bwd parts: gate bwd / restore / read (FP32 dK/dV) / 2nd read / merge | SCA fwd / bwd |
|---|---|---|---|---|---|---|
| gekko-0.9B | port | 8.91 | 5.63 / 3.01 / 0.40 | 45.75 | 0.58 / 0.40 / 28.98 / 15.51 / 0.62 | 5.60 / 28.66 |
| gekko-0.9B | this tree | 6.98 | 4.58 / 2.47 / 0.42 | 32.56 | 0.64 / 0.41 / 20.86 / 11.35 / 0.62 | 4.24 / 20.25 |
| gekko-4B | port | 5.33 | 3.93 / 1.30 / 0.20 | 20.90 | 0.23 / 0.19 / 16.11 / 4.60 / 0.17 | 3.90 / 15.90 |
| gekko-4B | this tree | 3.17 | 2.19 / 0.82 / 0.20 | 10.82 | 0.25 / 0.19 / 7.81 / 2.58 / 0.17 | 1.86 / 7.22 |
| gekko-7B | port | 7.25 | 5.77 / 1.30 / 0.29 | 28.83 | 0.31 / 0.29 / 23.85 / 4.60 / 0.17 | 5.74 / 23.61 |
| gekko-7B | this tree | 4.17 | 3.17 / 0.83 / 0.31 | 14.41 | 0.33 / 0.30 / 11.42 / 2.61 / 0.17 | 2.85 / 10.85 |
| moe-mp8 | port | 1.50 | 1.06 / 0.51 / 0.06 | 5.46 | 0.07 / 0.06 / 4.12 / 1.39 / 0.06 | 1.02 / 4.02 |
| moe-mp8 | this tree | 1.07 | 0.68 / 0.45 / 0.07 | 2.53 | 0.07 / 0.06 / 1.79 / 0.79 / 0.06 | 0.40 / 1.50 |
| ladder-1B | port | 15.28 | 11.96 / 3.28 / 0.20 | 65.46 | 0.23 / 0.19 / 51.72 / 13.58 / 0.17 | 11.91 / 51.54 |
| ladder-1B | this tree | 3.80 | 2.80 / 0.95 / 0.20 | 12.62 | 0.23 / 0.19 / 9.32 / 3.11 / 0.17 | 2.50 / 8.90 |
| mixed-64-128 | port | 10.31 | 7.97 / 2.30 / 0.20 | 40.70 | 0.23 / 0.19 / 32.08 / 8.52 / 0.14 | 7.93 / 32.01 |
| mixed-64-128 | this tree | 2.44 | 1.72 / 0.69 / 0.20 | 7.55 | 0.24 / 0.19 / 5.53 / 1.91 / 0.14 | 1.41 / 5.21 |

Where one composed call's time goes on this tree (profiled: kernel times of each phase attributed through their
launches, median of 5 calls, as a share of the call's GPU span; `metadata` = the PyTorch kernels that build each
read's visibility bounds; `glue` = the call's other PyTorch kernels; `idle` = gaps between the call's kernels while the
host issues the next ones, negative where the backward's dQ and key leaves overlap on two streams), and the host's cost
without a profiler (`issue` = CPU time to issue one call; `back-to-back` = per-call device time of calls issued without
synchronizing; the public SCA calls at the same shape for comparison):

| shape | fwd: two reads / metadata / gate / glue / idle | bwd: two reads / metadata / gate, restore, merge / glue / idle | issue ms fwd / bwd (SCA fwd / bwd) | back-to-back ms fwd / bwd |
|---|---|---|---|---|
| gekko-0.9B | 85% / 2% / 5% / 2% / 6% | 95% / 0% / 5% / 1% / -1% | 0.68 / 0.94 (0.02 / 0.10) | 7.63 / 34.14 |
| gekko-4B | 72% / 3% / 5% / 2% / 17% | 96% / 1% / 5% / 1% / -3% | 0.65 / 0.84 (0.02 / 0.10) | 2.90 / 10.84 |
| gekko-7B | 77% / 3% / 7% / 2% / 12% | 99% / 1% / 5% / 1% / -5% | 0.67 / 0.86 (0.02 / 0.09) | 4.19 / 14.77 |
| moe-mp8 | 40% / 7% / 3% / 3% / 47% | 85% / 3% / 5% / 1% / 5% | 0.63 / 0.84 (0.02 / 0.09) | 0.69 / 2.13 |
| ladder-1B | 78% / 2% / 4% / 3% / 12% | 95% / 1% / 4% / 1% / -1% | 0.66 / 0.86 (0.02 / 0.09) | 3.60 / 12.85 |
| ladder-2B | 66% / 4% / 7% / 5% / 18% | 90% / 1% / 7% / 2% / 0% | 0.64 / 0.85 (0.02 / 0.10) | 2.15 / 7.92 |
| mixed-64-128 | 67% / 4% / 7% / 3% / 18% | 92% / 1% / 7% / 1% / -1% | 0.64 / 0.83 (0.02 / 0.09) | 2.19 / 7.90 |

- At 7B the forward is the grouped read 65% of its span, the strict read 12%, the gate 7%, the reads' metadata 3%,
  glue 2% and GPU idle 12%; the backward is the grouped read's backward 82%, the strict read's 17%, gate backward,
  restore and merge 5%, metadata 1%, with the two leaves of each read's backward overlapping.
- The public SCA calls now cost 0.02 ms (forward) and 0.09 to 0.10 ms (backward) of host time; the composition still
  costs 0.63 to 0.68 ms per forward and 0.81 to 0.94 ms per backward: the reads take segment ids, so each builds its
  visibility bounds with PyTorch ops, plus validation and the route logic. At moe-mp8 the composed forward is
  host-bound (0.63 ms to issue a call that runs 0.69 ms back to back; 47% idle in an isolated call).
- Native bodies, at every shape above including D = 64 and D 64 / Dv 128: both reads' forwards run the pipelined
  forward (`PipeForwardKernel`), both backwards the pipelined dQ leaf and the pipelined key leaves (the fused key leaf
  where D + Dv <= 256, the dV-only and dK-only pair at D = Dv = 256), and the preprocessing (delta and the tile tables)
  in one launch.

## The causal API

`softdelta_attention` runs natively with each read as one chunk holding every key (CUDA events, L2 flush, median of
15; the same GPU as the parts above):

- xLLM's tail call (queries shorter than a chunk after the previous 4096-key chunk, BOS packing): 0.63 to 0.69 ms
  forward and 0.78 to 1.46 ms backward at the 0.9B and 4B heads with Lq 1024 and 2048: the forward is about the
  composition's host issue time, so host-bound.
- One 4096/8192/16384-token document at 32/8 heads, D 128: the causal call equals the chunk-4096 call where both see the
  same keys (N 4096: 0.28 / 0.72 ms; N 8192: 0.69 / 2.07 ms) and costs 1.49x / 1.46x the chunked call at N 16384
  (2.15 / 7.77 against 1.44 / 5.31 ms), in line with its 1.6x more visible pairs.

## Not native on B300, and why

- Precision level 1 (every SoftDelta API): the forward needs both reads' FP32 output state (`output_state`, which the
  native forward serves only from its `mma.sync` body and which is left unrouted), and the backward is the
  precision-repair backward of both reads (`_precision_backend.backward` with `strict` and `return_fp32`), whose native
  main loop is the SIMT advanced-precision leaf without tensor cores. Raises `NotImplementedError` before anything runs.
- Sliding-window SoftDelta with a window shorter than the sequence: there is no native sliding-window read. The native
  kernels take per-row key bounds (`row_lo`/`row_hi`), so a window could be served by the same reads with window bounds
  instead of chunk bounds; not attempted. Raises `NotImplementedError`.
- SM90-only machinery is unused on B300: the fused SoftDelta launches (`_softdelta_fwd`/`_bwd`, not built for sm_103a),
  the packed varlen BOS route and the matched-reset bridge.

## What to fuse first

1. The second read inside the grouped read, backward first: the backward is 2.4x to 4.5x the forward (3.3x at 4B and
   7B) and 54 to 69% of a recomputing training layer, and the strict read's backward is 18% (7B) to 35% (0.9B) of
   the backward, at the grouped read's per-head cost because its Hkv heads are ungrouped. One key-side pass that
   treats the read key as one more reader of each KV head (strict mask) would load each K/V tile once for both reads
   and accumulate dK/dV of both in a fixed order, dropping the FP32 dK/dV round trip of the grouped read and the
   merge. Same idea in the forward, where the strict read is 12% (7B) to 28% (0.9B) of the span.
2. Host work, now the largest fixed cost at small per-call shapes: the composition's 0.6 to 0.9 ms per call against
   the SCA calls' 0.02 to 0.1 ms. The reads take segment ids, so each rebuilds its visibility bounds with PyTorch ops;
   handing them the BOS mask instead would let them use the one-launch routes the SCA forward and backward now have
   (the backward's `_flash_sca_row_bounds` takes `strict_past`), and the strict read's bounds are the inclusive ones
   minus one. Per-call validation the public API already did can go too.
3. The gate in the forward epilogue and the restore/gate backward in the backward prologue: 3 to 7% of each
   direction now that the reads are faster.
4. D = 64 and unequal dims no longer fall back to the `mma.sync` body: ladder-1B (8 x 16384, 32/8, D 64) takes 1.3x
   gekko-4B (1 x 65536, 32/8, D 128) at equal attention FLOPs, against 2.9x / 3.1x on the port's reads.
