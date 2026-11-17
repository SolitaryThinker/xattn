# The combined tree: forward and backward optimizations with SoftDelta on top

This tree is the native deterministic B300 sliding-chunk attention (SCA) port (PORT.md) with two later lines of
optimization combined on it, the forward's and the backward's, each of which kept the other direction working but
carried only its own direction's work, and the native SoftDelta composition (SOFTDELTA.md) on top of both. It is the
starting point for optimizing SoftDelta attention, which composes two SCA reads (an inclusive read by the queries
and a strict-past read by the read keys) with gate, restore and gradient-merge kernels.

| commit | what |
|---|---|
| `flash_sca: deterministic native SCA on Blackwell (sm_103a)` | the port, on upstream main `b112365` |
| `flash_sca native: faster pipelined forward at every (D, Dv) up to 256` | the forward line, as its head has it |
| `flash_sca native: faster deterministic backward, combined with the forward` | the backward line, merged with the forward line (below) |
| `flash_sca native: the D=256 key leaves' analytic suffix mask needs a live first column` | a fix to the backward line (below) |
| `softdelta: ...` (two), `tests_b300/softdelta_main: ...` (two), `docs: ...`, `SOFTDELTA.md: ...` | the native SoftDelta composition, its tests and docs |
| `COMBINED.md: ...` | this file |

## How the two lines were combined

A three-way merge of the two lines' heads with the port as the base. A file only one line changed is that line's
file byte for byte (the merge takes it unchanged). Three files were changed by both:

| file | forward line | backward line | kept | why |
|---|---|---|---|---|
| `csrc/flash_sca/native/row_bounds.h` (both added it) | `RowBoundsArgs` / `RowBoundsMaskArgs`, `LaunchRowBounds` (the ids route) and `LaunchRowBoundsMask` (row bounds straight from the BOS mask up to `kMaskMaxKeys` = 131072 keys), defined in `dispatch.cu`, called by the forward's host route | `LaunchRowBoundsFromBos` (row bounds from the BOS mask, `reset` included), defined in the new `row_bounds.cu`, called by `_flash_sca_row_bounds` | both sets of declarations, verbatim, in one header with one file comment | each line's host route calls its own kernel, and nothing collides: the names differ, and the two `RowBoundsKernel` definitions live in different translation units (the backward's in an anonymous namespace) |
| `xattn/ops/_sca_native.py` | `visibility()` with a BOS mask, no reset and at most 131072 keys returns the mask itself (`Visibility.bos_mask`) and the extension call derives the bounds (one launch, no Python metadata chain); above that, an ids route (`_flash_sca_native_row_bounds`); `_forward`/`_backward` pass the mask on; the BOS-mask fast path of the public forward | `visibility()` with any BOS mask (reset too) calls `_flash_sca_row_bounds` once and passes row bounds | the forward line's file, with `visibility()` taking a keyword `bos_in_call`: `True` (the default) is the forward line's branch exactly, `False` the backward line's; the public `flash_sca_bwd` passes `False` | each direction's public entry takes its own line's metadata route, as each line measured it. The autograd `flash_sca` keeps the forward line's route (its metadata serves both directions there, as in the forward line); SoftDelta's private reads pass segment ids, not a mask, and take neither branch |
| `csrc/flash_sca/native/host.cc` (merged without a textual conflict) | `Forward` takes a BOS mask and builds its three metadata arrays in one allocation; `ForwardMask`, the repeat-call key and its bindings; `Backward` takes a BOS mask too; `PrepareCommon` no longer allocates the tile-bounds table: `Forward` does for a body that reads it, `Backward` for every schedule; `RowBounds` / `RowBoundsFromMask` | `Backward` no longer launches `LaunchTileBounds` / `LaunchKeyTileRange` (the delta stage builds delta and both tables in one launch); `RowBoundsFromBos` and `_flash_sca_row_bounds`; an environment-variable row of the header table | the union of both | checked by hand: `Backward` allocates the table (forward line) before `PrepareBackwardUmma` copies `Params` and before the delta stage fills it (backward line), on the native and the generic paths. Both lines added `#include "flash_sca/native/row_bounds.h"`; both lines are kept, so that a later merge of either line applies cleanly |

Files one line changed that the other direction also compiles or calls, each reviewed for its effect there:

| file | changed by | effect on the other direction |
|---|---|---|
| `params.h` | forward: `IsUmmaForwardLeafBucket` takes buckets and admits every 64/128/256 pair, `IsUmmaForwardDims`; `LaunchTileBounds(const Params&)` | none: the backward does not read the forward predicates; its `LaunchTileBounds` calls pass a `Params` lvalue |
| `umma_params.h` | forward: `UmmaInputMaps` gains `y` (the epilogue's TMA store map) | the backward leaves it zero; its kernel parameters grow by 128 bytes |
| `umma_prepare.h` | forward: a per-thread TMA descriptor cache in `PrepareUmmaCall` | none: the backward prepares its maps in `PrepareBackwardUmma`, unchanged |
| `tma_load.cuh` | forward: TMA store helpers | none (additive) |
| `dispatch.cu` | forward: the two row-bounds kernels | none: the backward's metadata kernels are unchanged |
| `kernels.cuh`, `launch.cuh` | backward: the one-launch preprocessing kernel and the delta stage | none: the forward's launch code there is unchanged |
| `setup.py` | backward: `row_bounds.cu` | none |

So the forward runs the forward line's code exactly, and the backward the backward line's code with the fix below.

## A fix to the backward line

The backward line's dK-only and dV-only pipelined key leaves (the D = 256 key bucket, every (D, Dv) with D + Dv >
256) take a launch-wide suffix mask when the bounds are analytic (dense metadata: no row bounds, no segment test)
and every 32-column block lies in one chunk: the visible columns of key row j are then a suffix of the block, built
from the block's first column. `RowBounds` encodes an empty row as `{INT_MAX, -1}`, and a strict-past read's first
query (position 0, no earlier key) is one, so the mask came out empty for every key row and the whole first block's
contribution to dK and dV was dropped: relative errors of about 0.7 in dK and dV of `_flash_sca_bwd(strict_past=True)`
at D = Dv = 192 and 256 with dense metadata and chunks that are multiples of 32. The public SCA API never asks for a
strict read; SoftDelta does, and its value-group tests (8 cases at D = 192, chunk 32) failed on the backward line
alone and on the combination, while the forward line (the port's backward) passed them. The analytic guard now also
requires the block's first column to be live; that block takes the per-block test, which it fails, and the general
per-column mask. Two dense reads at the D = 256 key bucket join `tests_b300/sca_main/test_b300_private_entry_points.py`
(24 cases; the 12 strict ones fail without the fix).

## SoftDelta on the combined tree

The six SoftDelta commits apply unchanged: they touch only `xattn/ops/softdelta_attention.py`, the SoftDelta docs,
`tests_b300/softdelta_main/` and SOFTDELTA.md, and the private reads they call keep their signatures and contract on
both lines. What changed under them is the reads' kernels: the forward line's pipelined forward now serves the
strict read and the inclusive read at every (D, Dv) (the port ran D = 64 and unequal dims on the `mma.sync` body),
and the backward line's leaves serve both backwards; the strict read's dense case is what exposed the fix above.

## Checks on the combined tree (B300)

Correctness and determinism, run before and after the fix (the counts are after it):

- the SCA contract (upstream's `tests/sca` on the native path at the full matrix and long, with the
  deterministic-only, private-entry-point and group-ratio-six suites; a skip fails): 11017/11017 passed, 0 failed,
  0 errors, 0 skipped;
- the Gekko-shape check (the traced xLLM calls and the stand-ins against FP64 on complete chunk blocks; the forward
  repeated bit-exactly, the backward repeated bit-exactly from the same and from a fresh forward, `deterministic=False`
  equal to `True`): 34/34;
- this tree's `tests_b300/sca_main` (default options): 1429 passed, the 24 new cases included; the forward line's own
  fast-path and pipe tests (`tests_b300/sca/test_fwd_fast_path.py`, `test_fwd_pipe.py`): 328 passed;
- SoftDelta: `tests_b300/softdelta_main` 2885 (default) and 3886 (`--full-matrix --long`) passed, 0 skipped; xLLM's
  SoftDelta training call at the eleven training shapes at full size: 11/11;
- the outputs of every timed call are bit-identical within a process and across 3 fresh processes (39 SCA rows, 145
  SoftDelta rows), and SoftDelta's first-call digests are identical across two GPUs and two builds (134 rows).

Speed, in the speed check's paired protocol (fresh process pairs alternating the two trees on one GPU, 3 pairs x 4
rounds x 2 repetitions, L2 flushed before each; each row held to max(1%, 30 us, 3 x the base's spread)), measured and
not judged:

| rows | against | the 7B objective row | every row of that direction | rows slower than their bound |
|---|---|---|---|---|
| the 13 forward rows | the forward line's head | 0.9992 | 0.9959 to 1.0057 | none |
| the 13 backward rows | the backward line's head | 1.0025 | 0.9935 to 1.0067 | none |

Each direction times as its own line does. The other direction's rows show what the combination adds to each line:
against the forward line the backward rows take 0.16 to 0.67 of its time, against the backward line the forward rows
0.18 to 0.78. Absolute times depend on the clock a GPU holds under its power cap (1.7 to 2.0 GHz on the GPUs used here,
2.03 GHz where the lines measured theirs), so trees are compared on one GPU.

SoftDelta, against the same composition on the native port's reads, on one GPU: the composed forward takes 0.55 to
0.78 and the backward 0.46 to 0.68 of the port's time at the Gekko shapes, and 0.18 to 0.35 at D = 64 and D 64 /
Dv 128. SOFTDELTA.md has the table, the composition's parts, where the time goes and what to fuse first.

## Recipe for later heads of the two lines

1. Fetch both heads read-only and check that each starts from the port's tree.
2. Three-way merge with the port as the base. A conflict whose three blobs equal a recorded one gets its recorded
   resolution back; any other conflict needs judgment, by the rules above: keep each direction's code as its line
   has it, and where a function serves both directions (`visibility()`), select each line's branch by direction.
3. Review every file both lines changed, also when it merged cleanly (judgment: semantic interplay, as for
   `host.cc` above), and every shared header one line changed.
4. Replay the carried commits: the fix (dropped once the backward line has it: its cherry-pick then comes out
   empty), the six SoftDelta commits, the docs. A conflict needs judgment.
5. Build, then run the SCA contract and the Gekko-shape check, the speed check paired against each line's head on
   one GPU (absolute times differ with the clock a GPU sustains), the SoftDelta suites (default and
   `--full-matrix --long`), the determinism checks and the SoftDelta timing with its parts.

Steps 2 (new conflicts), 3 and 4 (conflicts) need judgment; the rest is mechanical.
