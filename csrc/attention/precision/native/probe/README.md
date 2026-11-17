# Standalone numerical probe for the native advanced (L2-4) reader backward

Not part of the extension build: `setup.py` lists every source explicitly
(`ATTENTION_PRECISION_CSRCS` and the instantiation globs), and nothing under
`csrc/attention/precision/native/` is on any of those lists, so this directory
is built by hand and by nothing else. It exists so that the numbers quoted for
this leaf can be reproduced instead of taken on trust.

## What it measures

`run_leaf.cu` builds every operand on the device, launches one leaf through its
real `Launch*` entry, and compares the result against `ref64.hpp`, an
independent FP64 model that shares no code with the leaves: it recomputes the
softmax, the normalizer, `dP`, `delta`, the anchored `dS`, `dQ` and the four
statistics planes in double precision from the same quantized operands.

The probe covers, for each case, both ownership arms (`LaunchMainDq`, tiled, and
`LaunchMainDqScalar`, one CTA per row) and both modes (Main and ordered), and
prints one JSON line per launch with:

* `dq_pin_rel` / `dq_pin_abs` - dQ against the FP64 contraction of the operand
  the leaf actually multiplies: the element-rounded `dS` below level 4, and the
  rounded head plus its residual at level 4 (where the pin runs two
  components). `hi + lo` is that two-component operand, not the original FP32
  `dS`.
* `dq_max_rel` / `dq_max_abs` - dQ against the same model fed the unrounded
  FP32 `dS`, so the element-rounding contribution is visible on its own.
* `stats_max_rel` / `stats_max_abs` - the statistics planes against FP64 with
  the pin's operand: the sink is element typed exactly as the GEMM operands are
  (the pin builds `rdS` as `Element` and adds `rdSLo` only at level 4, and it
  splits the `P*2^e` operand only when `TwoComponentDV`), so rho/variance use
  the element head below level 4 and `hi + lo` at level 4, and the mass/anchor
  planes use the split `P*2^e`. Only the planes the level produces are
  compared (level 2: none, level 3: rho and variance, level 4: rho, mass and
  anchor probability).
* `stats_exact_max_rel` / `stats_exact_max_abs` - the same rows against exact
  FP32 `dS` and exact `P`, i.e. what the statistics would be with no element
  cast. The gap between the two residuals is the cast floor, on the record as a
  number rather than hidden by a matching reference.
* `untouched_bad` / `untouched_max_abs` - the planes the level does not produce
  are poison-filled before the launch and must come back byte-identical, which
  is the pinned behaviour (the sink block stores its own planes and no zeros).
* `empty_dq_nonzero` - dQ rows whose admitted key set is empty must be exactly
  zero.
* `repeat_identical` - a second launch with the same inputs must be
  byte-identical, in both modes.
* `sink_status` - the optional status word: bit 1 rho grid overflow, 2 `dS^2`
  overflow, 4 mass/anchor overflow, 8 a non-finite partial.

Ordered mode is compared after decoding the int64 grids back to FP64 exactly as
the caller's decode does (rho `2^(e-62)`, variance `2^(2e-68)`, mass and anchor
`2^-60`; the last two do not scale with `e`).

## Cases

24 launches per arm: levels 2, 3 and 4; fp16 and bf16; MHA, GQA 2:1 and MQA 8:1;
strict past; segment planes; previous K/V with a positive offset; `D = Dv = 32`,
`64`, `256`; empty rows; and three long rows (`Nq 24 / Nt 96`, `40 / 200`,
`8 / 160`) that cross the 32-key sink tile boundary several times, one of them on
the level-4 `2^e` grid so the per-tile rounding and the once-per-row de-scale are
both exercised. `L4_one_key` is one admitted key per row with `2^e` in
`{1, 2, 4, 8}` across four query heads, which is what isolated the `2^(e/2)`
scale defect.

## Build

From `kernel/csrc/attention/precision/native/probe`, with the repo's own torch:

```sh
TI=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include'))")
TL=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
for f in run_leaf.cu ../adv_bwd_dq.cu ../adv_bwd_dq_scalar.cu; do
  nvcc -std=c++20 -O2 -Xcompiler -Wall -arch=sm_103a \
    -I ../../../../csrc -I .. -I "$TI" -I "$TI/torch/csrc/api/include" \
    -I /usr/include/python3.12 --cudart shared -c "$f" -o "obj/$(basename $f .cu).o"
done
g++ -o run_leaf.bin obj/run_leaf.o obj/adv_bwd_dq.o obj/adv_bwd_dq_scalar.o \
  -L "$TL" -ltorch -ltorch_cuda -lc10 -lc10_cuda \
  -L "$(dirname $(dirname $(which nvcc)))/lib64" -lcudart \
  -Xlinker -rpath -Xlinker "$TL"
./run_leaf.bin scalar; ./run_leaf.bin tiled; ./run_leaf.bin agg
```

Add `../adv_bwd.cu` to the same compile line and link it to run `agg`, which
drives the aggregate `Launch` instead of one leaf. `Launch` chains the
key-owned dK/dV leaf, so with that leaf absent `agg` measures the refusal
contract instead: 24 launches must each refuse by name and leave dQ
byte-identical to its poison fill (`refused`, `names_missing_leaf`,
`dq_bytes_touched`). With the dK/dV leaf linked in, the same mode compares dK
and dV against the FP64 model.

## What this probe does not do

* It builds the operands itself, so it does not exercise the public Python
  entry, the caller's validation or the saved-state producer; a separate pairing
  audit covers the ABI side and the pinned-oracle check is elsewhere.
* The ordered grid exponent is computed with the caller's formula
  (`_rho_exponent`: `g = ||dY_i||_2` in float, `w =` the maximum row `L2` norm
  of the centered V that `preprocess.cu` builds, `e = frexp(g*w)[1] + 5`,
  clamped to `+-300`), but on the dY the case defines: this probe has no fp16
  prescale, so the production exponent differs by the exact power of two that
  prescale applies. That shifts every grid by one power of two and changes no
  tested arithmetic, since the decode uses the exponent the leaf was given.
* No timing, no memory measurement, no claim about dGate or dPrevK/dPrevV.

`mkdir -p obj` first. sm_103a is the B300 target; the leaves themselves are
compiled nowhere else.
