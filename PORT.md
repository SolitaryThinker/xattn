# B300 seed: native SCA port on upstream xattn main

The native B300 port (an earlier native Blackwell `flash_sca` forward/backward on upstream `9bfc56ac`) merged
with upstream xattn main `b112365`, plus the deterministic-only contract for B300, then the native route of the
private entry points SoftDelta is composed from, FP32 dK/dV, their tests and the group-of-six public tests
(section "The private entry points SoftDelta is composed from"). This file describes that port as it was
merged; COMBINED.md says what the forward and backward optimizations changed since.

## Merge

Three-way merge with base `9bfc56ac` (the port's root commit holds that tree without its history).
Public API = upstream main: `flash_sca_fwd` returns `(y, lse)`, `high_precision_level` is 0 or 1,
`attn_method`, packed/varlen entry points. Conflicts and their resolution:

| file | resolution |
|---|---|
| `csrc/softdelta/gate.cu`, `gradient_merge.cu`, `xattn/ops/softdelta_attention.py`, `_softdelta_attention.py` | upstream (non-SCA family); the port's SoftDelta native dispatch is dropped, SoftDelta is SM90-only as upstream |
| `csrc/causal_flash_attn/causal_flash_attn.cc` | upstream |
| `csrc/xattn_extension.cc` | upstream op set + native reader / native SCA / precision bindings |
| `setup.py` | upstream source groups + the port's sm_103a build (`XATTN_TARGET_SM` unset on a B300 selects `103a`) |
| `xattn/ops/sliding_chunk_attention.py` | upstream + native dispatch (`auto`/`''`/`native` on cc 10.x), route records, native route for the packed (`_flash_sca_varlen`) and batched varlen autograd entry points |
| `xattn/ops/_precision_backend.py` | upstream level-1 algorithm + the port's capability-based backend resolution |
| `docs/sliding_chunk_attention.md` | upstream + a Blackwell section |

Adapted beyond the conflicts: `xattn/ops/_sca_native.py` (new API), `csrc/attention/precision/native/adv_bwd_host.cc`
(upstream `MainBackward`/`ordered::MainBackward` signatures without `level`; the native leaf runs its internal
level 4, which is upstream's level 1).

## Precision levels on B300

- Level 0: native forward and backward.
- Level 1, forward: native module, `subtract_before_scale` softmax, no FP32 state (upstream's `high_precision`).
  It runs the native module's warp-level `mma.sync` body (`flash_sca_native::detail::ForwardKernel<…, true>`),
  not the tcgen05 pipeline, which accepts level 0 only (`PipeAccepts`).
- Level 1, backward: `_precision_backend.backward` resolves to `native` (route record `resolved='native'`). Its
  main loop is the native advanced-precision leaf (`native_adv::DqKernel`, `AdvBwdDkdvKernel`: SIMT, no matrix
  instruction). Row statistics use the portable `native_statistics::RowStatistics` kernel (WMMA). The repairs
  are upstream's CUDA helpers compiled for sm_103a, plus small PyTorch tensor ops in the Python driver, as on SM90.
  There is no PyTorch attention fallback.
- Not native: the standalone private varlen functions `_flash_sca_varlen_fwd`/`_flash_sca_varlen_bwd`. They call
  `_flash_sca_sm90_varlen_*`, which a B300 build does not link, so they raise; so do
  `_sca_segment_alignment.matched_forward`/`matched_backward`, which upstream's SoftDelta uses for reset chunks
  whose Q ids are not K's suffix (on B300, `_flash_sca_fwd`/`_flash_sca_bwd` with `reset_chunk_pos_per_seq=True`
  serve that case directly, below). Batched varlen whose `cu_seqlens_k` has a different run count than
  `cu_seqlens_q` (row-aligned runs) raises `NotImplementedError` on the native route. Nothing else is substituted.

## The private entry points SoftDelta is composed from

Upstream composes SoftDelta (`xattn/ops/softdelta_attention.py`, `_local_reader`, `_forward`, `_backward`) from the
private `sliding_chunk_attention._flash_sca_fwd` / `_flash_sca_bwd`: an inclusive read by Q over grouped heads,
whose dK/dV come back in FP32 (`fp32_kv_grads`) for the gradient merge, and a strict-past read by the per-KV-head
read keys. On cc 10.x, for backend `''`/`'auto'`/`'native'` as the public API, both now run on the native kernels
(`_sca_native.private_fwd`/`private_bwd`), with upstream's signatures; `deterministic` changes nothing.

| option | B300 |
|---|---|
| `strict_past` True/False | native: the kernels' strict visibility (analytic `hi = pos - 1`, folded into the segment, reset and varlen row bounds). A row with no earlier key in its window and segment gives y = 0, LSE = -inf and no gradient (the backward treats any non-finite LSE as zero probability). |
| `fp32_kv_grads` | native, grouped heads, dense or segment metadata, not strict, not varlen (upstream's rule): dk/dv/dprev_k/dprev_v in FP32, the values every key writer would round (`Params::key_grads_fp32`: the umma dK/dV pipes, the KV pipe, the fused umma leaf, the mma.sync bodies), so `.to(dtype)` gives the 16-bit gradients bit for bit. dq is unchanged. |
| `prev_k`/`prev_v`, short Q, `q_segment_idx`/`k_segment_idx` (sorted as SoftDelta derives them from a BOS mask, or not) | native |
| `reset_chunk_pos_per_seq` | native, including Q ids that are not the suffix of K's (upstream's matched-reset case) |
| `cu_seqlens_q`/`cu_seqlens_k` (batched varlen) | native when there is one key run per query run, as `_flash_sca`'s native route; row-aligned runs raise `NotImplementedError` |
| `high_precision` | native: the level-1 subtract-before-scale forward (mma.sync body) |
| `output_state` | raises `NotImplementedError`: only the level-1 SoftDelta composition passes it (its FP32 reads); level 0 never does. The native forward has a `y_state` field, served only by the mma.sync body (`PipeAccepts` refuses it), so it was left unrouted rather than shipped untested. |
| refusals | `fp32_kv_grads` with ungrouped heads: `RuntimeError` "FlashSCA FP32 dK / dV gradients need grouped query heads"; with strict or varlen: `ValueError` "FP32 dK / dV output needs the dense, non-strict FlashSCA backward" (upstream's words). An explicit `backend='sm90'` still raises on B300 (no SM90 backend in the build), `'cuda'` still names the generic kernels. |

Both functions record one route entry (`resolved: native`). At D = Dv = 128 or 256 the strict and FP32 reads run
the tcgen05/UMMA bodies (`forward`, `dq`, `dk`, `dv` phase tags `umma`); other dimensions run the native mma.sync
body. The timed key kernels (`BackwardDkPipeKernel`, `BackwardDvPipeKernel`, `BackwardKvPipeKernel`) take the FP32
store as a template parameter (`kFp32Keys`), so their 16-bit instantiations carry no FP32 path: a run-time branch
there had cost the 4B training backward's key leg +1.1% (7.61 -> 7.69 ms). The fused leaf, a fallback off the
traced shapes, and the generic `DkdvKernel`/`DkdvRegKernel` branch at run time. Register counts of the pipes and the
generic kernels equal the seed's; some fused-leaf instantiations gained registers, none spills.

Tests (`tests_b300/sca_main`):

- `test_b300_private_entry_points.py`: both functions against the FP64 oracle (tolerances unchanged) on 22 curated
  reads (dense, uneven dims, previous K/V, sorted and unsorted segment ids, non-suffix Q ids, empty rows, reset,
  batched varlen, short Q, strided views, queries that are the keys, the umma buckets) x FP16/BF16 x heads 4/4,
  8/2, 12/2 x strict/inclusive; FP32 dK/dV round to the 16-bit gradients bit for bit; every case runs the
  determinism trio (forward repeat, backward repeat from the same and from a fresh forward, `deterministic=False`
  == `True`). The SoftDelta reads of the Gekko layers (strict 4/4 x 256 and 8/8 x 128; FP32 inclusive 8/4 x 256,
  32/8 x 128, 48/8 x 128; chunk 4096; packed-document segment ids) at two chunks by default and at 65536 under
  `--long`, against FP64 on complete chunk blocks. `high_precision` (the level-1 forward) against FP64. The
  refusals.
- `test_b300_gqa_ratio_six.py`: the public API at Gekko-7B's group of six (6/1, 12/2), which the upstream cases
  (four query heads) never reach: curated cases against FP64 at levels 0 and 1, the deterministic-only trio, and
  the D/V matrix at 6/1 under `--full-matrix`.
- `test_b300_deterministic_only.py`: the gekko-7B train, prefill and prefill-continued calls under `--long`.

Counts on B300, 0 skipped: default 1405 (767 + 326 + 312); `--full-matrix --long` 11017 (the seed's 8999, all
present and passing, + 2018). Every test passed the autouse native-route proof; at the Gekko and SoftDelta shapes
(0.9B, 4B and 7B) every phase is `umma`. Gaps the composition has to know: upstream's `_local_reader` passes
`backend="sm90"`, which raises on B300 (pass `'auto'`); upstream's `_call` sends a non-SM90 device to the PyTorch
reference (`_torch_softdelta`); the reset case upstream routes through `matched_forward`/`matched_backward` is
served by `_flash_sca_fwd`/`_flash_sca_bwd` with `reset_chunk_pos_per_seq=True` directly (where upstream also
refuses `fp32_kv_grads`, the native path does not need to); `output_state` (level 1) is not routed.

## Deterministic-only on B300

Every SCA call on sm_103 is deterministic; `deterministic` stays in the API and changes nothing there.

- `csrc/flash_sca/native/host.cc` `Backward()`: the flag is ignored (`constexpr bool deterministic = true`). The
  planner always gets a deterministic request. The generic-body plan is always `MakeBackwardPlan(kFusedKey, direct)`,
  and the generic `mma.sync` body never gets an FP32 `dq_accum` (no `red.global.add` dQ).
- `csrc/flash_sca/native/umma_bwd.cu` `SelectWriters()`: the `kFusedAtomicDq` key writer (`dkdv_atomic_dq`) is
  never selected; the plan is always `kFusedKey` with direct row-owned dQ.
- `xattn/ops/_precision_backend.py`: on cc 10.x the level-1 backward always runs its ordered branch. The unordered
  branch (main backward, probability repair and anchor-scatter zero-sum repair, all with FP32 atomics) never runs there.
- Test: `tests_b300/sca_main/test_b300_deterministic_only.py` checks that `deterministic=False` matches `True`
  bitwise and that it repeats bit-exactly, and that the forward repeats. It covers the curated upstream cases at
  levels 0 and 1 and the six traced Gekko calls. On the tree before this change it fails 132 of 318 cases.

## How native execution is proven

1. `tests_b300/sca_main/conftest.py` (autouse) checks every test:
   - the native forward launch counter (`_flash_sca_native_launch_counts`) moved;
   - every level-1 backward resolved to `native` (`_precision_backend.route_trace()`);
   - no public SCA route record resolved to `generic` or `matched_reset` (`_native_dispatch.route_trace()`).

   It records the native per-phase dispatch tags (`_flash_sca_native_phase_tags`) per test in `SCA_MAIN_ROUTE_LOG`.
2. A determinism script (outside this tree) collects the profiler kernel names of one forward and backward per Gekko
   shape, with the native launch deltas and phase tags. It checks that no `FlashSCAReg*` (generic
   `csrc/flash_sca/cuda`) kernel and no PyTorch attention operator ran.
   - D=Dv=256: `PipeForwardKernel<bf16,…,256>`, `DqPipeKernel<…,256,256>`, `BackwardDkPipeKernel`,
     `BackwardDvPipeKernel`, plus `DeltaKernel`, `TileBoundsKernel` and `KeyTileRangeKernel`.
   - D=Dv=128: `PipeForwardKernel<…,128>`, `DqPipeKernel<…,128,128>`, `BackwardKvPipeKernel<bf16,3>`, plus the same
     metadata kernels.
   - The remaining kernels are PyTorch index ops that build the BOS row bounds.

   Phase tags: `forward umma`, `dq/dk/dv umma`, `delta scalar`.
3. On the upstream test shapes, D/Dv are not exactly 128/128 or 256/256, so the forward uses the native module's
   `mma.sync` body (761 of 767 tests; the 6 Gekko tests use umma). The backward is split as follows:
   - level 0: `umma` in 122 tests and the native `mma_sync` body in 331;
   - level 1: the native SIMT precision leaf (phase rows `scalar`), 302 tests.

   None of them uses the generic `csrc/flash_sca/cuda` kernels or a PyTorch fallback.
