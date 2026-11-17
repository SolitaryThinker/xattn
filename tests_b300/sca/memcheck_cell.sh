#!/usr/bin/env bash
# Memcheck the PUBLIC backward at a reduced shape, per cell -- the leg stage 2 names
# ("compute-sanitizer memcheck on a reduced shape") and that nothing in this tree runs.
#
# WHAT IT ANSWERS: does a public `flash_sca_bwd` at cell (D,Dv) touch memory out of bounds, leak,
# or trap a device thread, on the route the default environment selects? It drives the cells of
# $CELLS (default: all nine mounted ones) in ONE loop: (64,64), (64,128), (64,256), (128,64),
# (128,128), (128,256), (256,64), (256,128), (256,256).
#
# WHAT IT DOES NOT ANSWER: whether the backward has a DATA RACE. `--tool memcheck` observes
# invalid and out-of-bounds accesses, leaks, uninitialised reads and device traps; it does not
# observe racy-but-in-bounds accesses -- that is `--tool racecheck`, a different run. A clean row
# here is a statement about that class and no more. (The trap this file was built for IS in the
# class: `Trace/breakpoint trap` is reported by memcheck, which is why a zero-error row over
# repeated calls is evidence about it.)
# (128,128) is here because it TRAPPED (34 `Trace/breakpoint trap` reports in one block
# of the in-situ dKdV leaf). (256,256) is here because an earlier revision of this file EXCLUDED
# it on the grounds that it "carries `ERROR SUMMARY: 0 errors` rows from other lanes" -- and that
# reason was wrong on this lane's own rule: a memcheck row is a fact about a BINARY, other lanes'
# rows were measured on OTHER bytes, and nothing transfers. It stays in the grid for COVERAGE of
# the (256,256) bucket -- NOT because it shares the trapped cell's plan. An earlier revision of
# this comment said `kWrite == 3` and `ABL` sit at BOTH cells; that is wrong and the base source
# says so: `csrc/flash_sca/native/umma_bwd_dkdv.cuh:1243-1254` states that every other bucket,
# (256,256) included, keeps the ONE-stage body, and statically asserts
# `InSituKeyArm<256,256>::value == 0` beside `InSituKeyArm<128,128>::value == 4`. Only the M128
# (128,128) bucket ships ABL4/two stages. A bucket is covered because a router can ask for it,
# not because it resembles the cell that failed.
#
# WHY ONE LOOP, AND WHY THAT IS THE WHOLE POINT OF THIS REVISION: the previous revision drove
# (64,64) and (128,128) in a `for D in 64 128` loop and then (64,256) in a SECOND, duplicated
# block. Two copies of the same sanitizer invocation drifted: the banner describing the coverage
# went stale against the code (a review finding), and a repair applied to one copy did
# not reach the other. There is now ONE invocation of the sanitizer in this file, over a cell
# list, so a cell cannot be driven without the banner counting it.
#
# WHAT A FAIL PRINTS: the FIRST TWELVE `=========` HEADER LINES of the log, of ANY class. The
# previous revision grepped a whitelist -- `Invalid|out of bounds|Leak|Race` -- and on a log whose
# 34 errors were all `Trace/breakpoint trap` it printed a BARE COUNT with no kind, so a failing
# cell could not be routed from its own row. A count is not a finding; the header line is. This
# is the same class of defect as a verdict better than its evidence, fixed by printing whatever
# the tool said rather than the subset this file expected it to say.
#
# WHAT A ROW MUST NAME ABOUT ITSELF: the `.so` sha256. A memcheck verdict is a statement about
# the BINARY, not the source, and one source digest has been measured producing three builds with
# three digests. The previous revision read the digest with a one-liner that fell back to
# `UNREADABLE-xattn_cuda` -- and that fallback FIRED while the caller's own block printed the real
# digest one line earlier, so the row named its product as unreadable when it was readable. The
# reading now reports WHY it failed when it fails, and an unreadable digest is a FAILURE of the
# row (exit 5), never a PASS with a shrug: a green that cannot name its binary is not a green.
#
# RUN HISTORY, and exactly what it is worth: an EARLIER COPY of this instrument reported zero
# errors at (64,64) and (128,128), from another runner. Its bytes are not this
# file's bytes and the green does not transfer. On THIS branch's bytes a later row
# reported (64,64) PASS, (64,256) PASS, and (128,128) FAIL with 34 traps -- the finding that
# motivates this file's one-loop form. Read a green here only from a row that names this file's
# own sha256 and its `.so` sha256.
#
# IT PRINTS THE ROUTE of every cell. The per-cell ROUTE/PHASES lines are read inside the sanitized
# process from the host's backward preparation record and the dq/dk/dv phase tags, so the body that
# served a cell is reported by the same row as its memcheck verdict. Only some of the mounted
# buckets have a native backward body, and a clean generic cell must not read like a clean native
# one. The route is a record of what the host PREPARED, not a numerical proof of correctness.
#
# IT PRINTS THE FORWARD BODY TOO, and the environment arm that selected it: what an unset
# `XATTN_SCA_UMMA_FWD` selects is a property of the tree, so a row is comparable with another row
# only when it states the arm. The ROUTE line carries `arm_fwd=`/`arm_bwd=` and the row carries
# the forward rows of the same phase-tag table: `FWD-N <count>`, one `FWDROW <phase> <selected>
# <launched> <state> [<count>]` for EVERY forward row of the cell (the table is cumulative and
# unordered, and a cell can hold a native refusal beside a generic run), `FWD-MIXED yes|no` when
# those rows disagree on (selected, launched), and a one-line `FWD <selected>/<launched> ...` for
# reading. All of it is carried into every PASS, FAIL and UNPARSED line and into summary.json,
# where each cell carries `rc`, `fwd`, `fwd_kinds`, `fwd_rows` and `fwd_mixed`, and the JSON
# verdict is the same fail-closed predicate as the shell's. `FWD unreported` means the row could
# not read a forward row, and it says so instead of implying native.
#
# THE TWO LEVEL-0 DQ BODIES: `selected=umma` does not distinguish DqKernel from DqPipeKernel.
# Each differential backward therefore brackets exactly that call with
# `_flash_sca_native_dq_pipe_launches` and emits `DQPIPE arm=<arm> before=<n> after=<n> delta=<d>`.
# One per completed ARM is required, at every level and on a generic route too (zero is then the
# observed non-pipe count, not an inference from the arm). Missing, non-integer, decreasing or
# multi-pipe-per-call observations fail the cell. `summary.json` preserves every (arm, delta)
# observation, its total, and whether every completed ARM has a witness. A witness is complete
# only when the raw rows are exactly `dense,bos` once per repeat, in that order, with no missing,
# duplicate, malformed or extra row and with `delta == after-before`; counting two valid-looking
# rows is not enough (two `dense` rows must not stand in for a missing `bos` row). Both the shell
# verdict and summary.json apply that predicate. This distinguishes the (128,128) pipe from the
# native non-pipe dQ body on the exact bytes the sanitizer imported.
#
# Conventions, each learned from a fault in this lane:
#   * compute-sanitizer absent => TOOL-ABSENT, exit 3. NOT a pass.
#   * ERROR SUMMARY line missing => UNPARSED, exit 4. NOT a pass: an absent line is not a zero
#     line, and a grep for "0 errors" over empty input is the classic vacuous green.
#     The VERDICT block now EXITS 4 for it: the convention promised 4 and the code exited 1
#     through VERDICT-FAIL, so a reader scripting `case $? in 4)` saw a pass-adjacent code
#     where the file's own contract said UNPARSED (a review finding).
#   * `.so` digest unreadable => NO-BINARY-IDENTITY, exit 5. NOT a pass.
#   * a driver call returning nonzero rc fails the leg (a review finding: the rc was printed
#     and never read, so a raised call still closed VERDICT-PASS).
#   * every cell prints a ROUTE line: a cell whose route cannot be read prints ROUTE-UNAVAILABLE
#     and the verdict table carries that string, so no cell reads as native without a route record.
#
# TWO ARMS PER SANITIZED CELL (a review finding): every cell runs arm=dense (no metadata)
# and then arm=bos (`bos_mask`, segments starting at 0, 200, 256, 400), in ONE process and ONE
# sanitizer log. The previous revision drove only the dense arm -- so the segmented-metadata
# path, which is the shape xllm actually sends, was NEVER sanitized while the banner's `no
# bos_mask` read as if the cells covered the route. Both arms in one process keeps the cell
# count at eighteen launches (nine cells x two arms); the cost is that a FAIL names the cell but
# not which arm trapped, so
# each arm prints an `ARM <name> start` / `ARM <name> ok` marker into the per-cell `.out`.
#
# LEVELS, ROUTE AND SUMMARY:
#   * LEVEL=0..4 (default 0, one value per process, never a list). The backward is handed the
#     saved forward state the public contract names for the level: `flash_sca_fwd` returns
#     y_fp32=None at level 0 and a y_fp32 tensor at levels 1-4, and `flash_sca_bwd` takes y at
#     level 0 and y_fp32 at levels 1-4. The driver asserts `(y_fp32 is None) == (L == 0)`
#     in-process; that checks the presence of y_fp32 against the level, not its dtype. Any LEVEL
#     outside the integers 0-4 is refused with exit 2 / EXIT92, the same class as REPEATS<1 and a
#     blank CELLS, so the runner's guard block covers it with one more invocation (`LEVEL=5` must
#     also return 2). One level per process, because a sanitizer summary shared by two levels could
#     not say which level trapped.
#   * a ROUTE line per cell, read inside the sanitized process after the last arm: the host's
#     `_flash_sca_last_backward_preparation()` record plus the dq/dk/dv phase tags. Only some of
#     the mounted buckets have a native backward body (`umma_bwd_entry.h`), so the log says which
#     body served each cell. A level that does not enter the native leaf (`refusal=not_called`,
#     levels 2-4) reads as such, so a level row cannot be read as a native-backward row.
#   * $OUT/summary.json: one machine-readable object per cell (errors, traps, arms completed and
#     expected, rc, route, forward rows), so a number can be read from this row without parsing
#     the prose log.
set -uo pipefail
OUT=${OUT:?OUT must be set: a job must be given a durable output directory}
mkdir -p "$OUT"
LOG=$OUT/memcheck.log
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }
fail=0

# Both knobs are read HERE, before the first use of either: the banners below print $N_VAL, and
# under `set -u` an expansion before assignment is an abort, not an empty string. A previous
# revision assigned them 40 lines below the banner that uses them, so the script died before the
# first sanitizer cell and a `bash -n` could not see it (a review finding: the defect
# is an execution-order defect and only an execution smoke test finds it).
REPEATS_VAL=${REPEATS:-1}
# N is a knob: every row this instrument has produced is N=512, so nothing is known
# about how the (128,128) trap depends on length. C stays 256, so the CHUNK COUNT (rounds of the
# ring) varies with N: 512 -> 2 chunks, 1024 -> 4, 2048 -> 8. Default 512 keeps every earlier row's
# shape exactly, so a row that does not set N is comparable to the earlier trapping and clean rows.
N_VAL=${N:-512}
# The precision level of the CALL, 0-4 (contract: "Only integers 0-4"). One value, not a list:
# one process per level, so a sanitizer summary belongs to exactly one level.
LEVEL_VAL=${LEVEL:-0}

# The nine cells, in the order they are driven. D:Dv, colon-separated, no spaces.
# CELLS is overridable so a single-cell row is the SAME driver with a different population rather
# than a second copy of the sanitizer invocation (this file's one-invocation rule); the default is
# the full grid, so a row that does not set CELLS is comparable to every earlier row.
CELLS=${CELLS:-"64:64 64:128 64:256 128:64 128:128 128:256 256:64 256:128 256:256"}

# ---- invocation validation: this row must be INCAPABLE of a vacuous pass ---------------------
# Found in review: `REPEATS=0` ran no arm while compute-sanitizer still printed
# `ERROR SUMMARY: 0 errors`, and a whitespace-only `CELLS` ran no cell and reached the final PASS.
# Both are closed in TWO places, because either alone leaves the other open: here a bad population
# is refused before anything runs (exit 2 = bad invocation, distinct from 1/3/4/5), and further
# down the pass condition requires the expected number of completed ARM markers.
case "${REPEATS_VAL}" in
  ''|*[!0-9]*) say "EXIT92: REPEATS='${REPEATS_VAL}' is not a non-negative integer"; exit 2 ;;
esac
[ "${REPEATS_VAL}" -ge 1 ] || { say "EXIT92: REPEATS=${REPEATS_VAL} runs no arm, and a sanitizer that ran no arm can still print a zero-error summary"; exit 2; }
CELL_LIST=$(printf '%s\n' "$CELLS" | tr -s '[:space:]' '\n' | sed '/^$/d')
[ -n "$CELL_LIST" ] || { say "EXIT92: CELLS='${CELLS}' names no cell"; exit 2; }
for CELL in $CELL_LIST; do
  printf '%s' "$CELL" | grep -qE '^[0-9]+:[0-9]+$' || { say "EXIT92: CELLS token '$CELL' is not D:Dv"; exit 2; }
done
# LEVEL is refused here for the same reason REPEATS and CELLS are: a knob that silently falls back
# to level 0 measures a body the caller did not ask for, and the row would look exactly like a
# correct one. The contract allows only the integers 0-4.
case "${LEVEL_VAL}" in
  ''|*[!0-9]*) say "EXIT92: LEVEL='${LEVEL_VAL}' is not an integer; the contract allows only 0-4"; exit 2 ;;
esac
[ "${LEVEL_VAL}" -le 4 ] || { say "EXIT92: LEVEL=${LEVEL_VAL} is outside 0-4; the contract allows only the integers 0-4"; exit 2; }
NCELLS_REQ=$(printf '%s\n' "$CELL_LIST" | wc -l | tr -d ' ')
say "  requested cells (${NCELLS_REQ}): $(printf '%s' "$CELL_LIST" | tr '\n' ' ')"

say "=========== IDENT: what this row is about"
say "  nvcc: $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*, V[0-9.]*' | head -1)"
say "  torch: $(python3 -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null)"
say "  head: NOT READABLE IN A JOB (no .git in the copy) -- cite the source digest and the .so sha256"
[ -d kernel ] || { say "EXIT90: no kernel/ under $PWD"; exit 90; }
# SCOPE, stated because a digest is only as wide as its file list: this covers the SCA sources the
# public call EXECUTES -- csrc/flash_sca/** (host.cc, the dispatch .cu, the native leaves and the
# instantiations) -- and the Python dispatch that selects them. It does NOT cover csrc/cutlass/**,
# the other csrc/ trees, or the build flags: for the binary, the .so digest below is the identity
# that matters. An earlier revision hashed only native/*.h, native/*.cuh and instantiations/*.cu and
# then called the result "sources", which did not support the wider provenance claim (a review finding).
SRC_SCOPE="csrc/flash_sca xattn/ops/sliding_chunk_attention.py"
SRC_LIST=$( cd kernel && find $SRC_SCOPE -type f \( -name '*.h' -o -name '*.cuh' -o -name '*.cc' \
    -o -name '*.cu' -o -name '*.py' \) 2>/dev/null | LC_ALL=C sort )
SRC_LIST=$(printf '%s\n' "$SRC_LIST" | sed '/^$/d')
SRC_N=$(printf '%s\n' "$SRC_LIST" | wc -l | tr -d ' ')
SRC_SHA=$( cd kernel && printf '%s\n' "$SRC_LIST" | xargs cat 2>/dev/null | sha256sum | cut -d' ' -f1 )
say "  sources sha256: ${SRC_SHA:-<unreadable>}  over ${SRC_N} files"
say "    scope (exact): ${SRC_SCOPE}  -- *.h *.cuh *.cc *.cu *.py, LC_ALL=C sorted"
say "    NOT in scope: csrc/cutlass/**, the other csrc/ trees, and the build flags; the .so is the binary identity"
say "  file: $(sha256sum "$0" | cut -d" " -f1)  $(wc -c < "$0")B  $0"
# The digest of the module that will actually be loaded, printed WITH its path and size, and with
# the reason when it cannot be read. `python3 -` reads the program from stdin, so this probe does
# not depend on any file of the tree being importable as a path.
SOSHA=$(python3 - <<'PROBE' 2>&1
import hashlib, sys
# LOAD torch FIRST. `xattn_cuda` links against libc10/libtorch, and a bare `import xattn_cuda` in
# a fresh interpreter dies with `libc10.so: cannot open shared object file` unless something has
# already loaded them. That is exactly what an earlier row did: the probe printed
# `IMPORT-ERROR ImportError: libc10.so: ...`, the fail-closed rule below refused the row, and a
# row whose EIGHT cells were all `ERROR SUMMARY: 0 error` closed VERDICT-FAIL -- for the
# interpreter's import order, not for the binary. The rule is right; the probe was wrong.
try:
    import torch  # noqa: F401  (loads libc10/libtorch into the process)
except Exception:
    pass
try:
    import xattn_cuda
except Exception as exc:
    print("IMPORT-ERROR %s: %s" % (type(exc).__name__, exc)); sys.exit(0)
path = getattr(xattn_cuda, "__file__", None)
if not path:
    print("NO-FILE-ATTR xattn_cuda imported but __file__ is %r" % (path,)); sys.exit(0)
try:
    blob = open(path, "rb").read()
except Exception as exc:
    print("READ-ERROR %s: %s (path %s)" % (type(exc).__name__, exc, path)); sys.exit(0)
print("%s  %dB  %s" % (hashlib.sha256(blob).hexdigest(), len(blob), path))
PROBE
)
say "  .so sha256: ${SOSHA:-<probe printed nothing>}"
if printf '%s' "$SOSHA" | grep -qE "^[0-9a-f]{64}  "; then
  say "  .so identity: READ (this row names its binary)"
else
  say "  .so identity: UNREAD -- STOPPING HERE. A memcheck green that cannot name its binary is not a"
  say "  green, and running the grid would spend minutes producing a row nobody can attribute."
  say "  exit 5, as this file's own conventions promise. (Round 1 found that this branch set a flag"
  say "  and then ran the whole grid anyway, ending at exit 1 -- the code did not do what the comment"
  say "  above said it did.)"
  exit 5
fi
say "  gpu: $(nvidia-smi --query-gpu=name,compute_cap,clocks.sm --format=csv,noheader 2>/dev/null | head -1)"
say "  host: $(hostname 2>/dev/null)"
say "  route-selecting env in this job (names only; values are printed, they are not secrets):"
( env | grep "^XATTN_" | sort | sed "s/^/    /" ) || say "    (none set: XATTN_SCA_BWD_ARM, XATTN_SCA_TEST_NARROW_KEY_BUCKETS and the dump switches are all unset)"
say "  the calls below therefore run the DEFAULT route; a row that omits this block cannot say which"

if ! command -v compute-sanitizer >/dev/null 2>&1; then
  say "  TOOL-ABSENT compute-sanitizer is not on PATH -- this leg is UNTESTED, not passed"; exit 3
fi
say "  tool: $(compute-sanitizer --version 2>&1 | head -2 | tr '\n' ' ')"

# The unsanitized control is driven on exactly those cells of THIS ROW that are the two reduced
# ones. A single-cell row (CELLS=128:128) must not silently drive a cell it does not report, and
# the banner must not describe a population the row did not request (a review finding: the banner said
# "TWO of the nine cells" on a row that asked for one).
DRIVER_CELLS=$(for CELL in $CELL_LIST; do case "$CELL" in 64:64|128:128) printf '%s ' "$CELL";; esac; done)
if [ -n "$DRIVER_CELLS" ]; then
say "=========== the DRIVER leg: an UNSANITIZED control at ${DRIVER_CELLS}(of this row's ${NCELLS_REQ} cell(s))"
say "  shape: N=1024, bos_mask -- the call shaped as xllm makes it."
say "  It is the unsanitized control: what a reader who never runs the sanitizer would see."
say "  It is NOT a verdict; the sanitizer loop below, at N=$N_VAL, is what this row reports."
say "  driver: one public fwd+bwd per cell, bf16, level ${LEVEL_VAL}, deterministic off"
python3 - "$OUT" "$DRIVER_CELLS" "$LEVEL_VAL" <<'DRIVER' 2>&1 | tee -a "$LOG"
import sys, torch
import xattn
L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(0)
B, N, C, Hq, Hkv = 1, 1024, 256, 4, 2
for pair in (sys.argv[2].split() if len(sys.argv) > 2 else []):
    D, DV = (int(x) for x in pair.split(":"))
    q = torch.randn(B, N, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, N, Hkv, DV, device="cuda", dtype=torch.bfloat16)
    bos = torch.zeros(B, N, device="cuda", dtype=torch.bool)
    bos[:, 0] = True
    bos[:, N // 2] = True
    dy = torch.randn(B, N, Hq, DV, device="cuda", dtype=torch.bfloat16)
    y, y32, lse = xattn.ops.flash_sca_fwd(q, k, v, chunk_size=C, bos_mask=bos,
                                          scale=None, high_precision_level=L)
    # y_fp32 is None at level 0 and present at levels 1-4 (the assert checks presence against the
    # level, not the dtype), and the backward is handed the saved state the contract names for the
    # level: y at level 0, y_fp32 at levels 1-4.
    assert (y32 is None) == (L == 0), "level %d returned y_fp32=%r" % (L, y32)
    saved = y if L == 0 else y32
    dq, dk, dv, dpk, dpv = xattn.ops.flash_sca_bwd(
        dy, q, k, v, saved, lse, C, scale=None, bos_mask=bos, high_precision_level=L)
    torch.cuda.synchronize()
    finite = all(t is None or torch.isfinite(t.float()).all().item() for t in (y, dq, dk, dv))
    print("  CASE D=%d Dv=%d  y=%s dq=%s dk=%s dv=%s  all_finite=%s  lse=%s"
          % (D, DV, tuple(y.shape), tuple(dq.shape), tuple(dk.shape), tuple(dv.shape),
             finite, tuple(lse.shape)))
DRIVER
DRC=$?
[ "$DRC" -eq 0 ] || fail=1
say "  driver rc=$DRC (0 expected; a nonzero rc is a finding about the call, not about this script, and it fails this leg)"
say "  NOTE: a clean driver leg is NOT evidence of memory safety -- one run measured a clean driver"
say "  leg (all_finite=True, rc=0) at (128,128) on the same build whose sanitizer row traps there."
else
say "=========== the DRIVER leg: SKIPPED -- no 64:64 or 128:128 cell among this row's CELLS"
fi

say "=========== memcheck: ONE process per cell, ONE loop over the ${NCELLS_REQ} cells this row requested"
say "  shape per cell: B=1 N=$N_VAL C=256 Hq=4 Hkv=2, bf16, level ${LEVEL_VAL}, deterministic off"
say "  saved state handed to the backward: y at level 0, y_fp32 at levels 1-4"
say "  (asserted in-process: y_fp32 is present exactly when the level is nonzero; dtype not checked)"
say "  TWO ARMS in each cell's process, one sanitizer log per cell: arm=dense (no metadata),"
say "  then arm=bos (bos_mask with segment starts at 0, 200, 256, 400). The segmented metadata"
say "  path is the shape xllm sends; driving only the dense arm would leave it unsanitized."
say "  REPEATS=${REPEATS_VAL} pair(s) per cell: this defect class is INTERMITTENT -- one run trapped"
say "  34x at (128,128) on one build and another measured 0 errors at all eight cells on another"
say "  build of the SAME sources digest -- so a single pass per cell cannot measure a rate. One"
say "  process per cell still, so the repeats share one sanitizer summary."
say "  cells (this row, ${NCELLS_REQ} of them): $(printf '%s' "$CELL_LIST" | tr '\n' ' ')"
RESULTS=""
unparsed=0
npass=0
for CELL in $CELL_LIST; do
  D=${CELL%%:*}; DV=${CELL##*:}
  TAG="D${D}_Dv${DV}_L${LEVEL_VAL}"
  say "----------- cell D=$D Dv=$DV level=${LEVEL_VAL}  (log mc_${TAG}.log, REPEATS=$REPEATS_VAL, N=$N_VAL)"
  CUDA_LAUNCH_BLOCKING=1 compute-sanitizer --tool memcheck --error-exitcode 71 \
    --launch-timeout 120 --log-file "$OUT/mc_${TAG}.log" \
    python3 - "$D" "$DV" "$REPEATS_VAL" "$N_VAL" "$LEVEL_VAL" <<'CASE' > "$OUT/mc_${TAG}.out" 2>&1
import os, sys, torch, xattn
D, DV = int(sys.argv[1]), int(sys.argv[2])
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 1
N = int(sys.argv[4]) if len(sys.argv) > 4 else 512
L = int(sys.argv[5]) if len(sys.argv) > 5 else 0
C = 256
torch.manual_seed(0)
B, Hq, Hkv = 1, 4, 2
q = torch.randn(B, N, Hq, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, N, Hkv, DV, device="cuda", dtype=torch.bfloat16)
dy = torch.randn(B, N, Hq, DV, device="cuda", dtype=torch.bfloat16)
bos = torch.zeros(B, N, device="cuda", dtype=torch.bool)
for p in (0, 200, 256, 400):
    if p < N:
        bos[:, p] = True

def route_line():
    # Read INSIDE the sanitized process, after the last synchronize, so the record belongs to the
    # last call of the arm whose summary the cell reports -- and so the route is covered by this
    # row instead of by a second, unsanitized one. Nothing here launches: it is a host-side read.
    # A row that cannot read it says so; it never guesses.
    try:
        import xattn_cuda
    except Exception as exc:
        print("ROUTE-UNAVAILABLE import %s: %s" % (type(exc).__name__, exc), flush=True); return
    ops = None
    for cand in (getattr(xattn_cuda, "ops", None), xattn_cuda):
        if cand is not None and hasattr(cand, "_flash_sca_last_backward_preparation"):
            ops = cand; break
    if ops is None:
        print("ROUTE-UNAVAILABLE no _flash_sca_last_backward_preparation on xattn_cuda(.ops)", flush=True); return
    try:
        d = ops._flash_sca_last_backward_preparation()
        # The ARM CLASS is part of the row, not context: `unset` and "0" can select the SAME body
        # on one tree and DIFFERENT bodies on another, so a row that does not print what the
        # environment said cannot be compared with any other row. Empty is spelled `empty` so a
        # set-but-empty variable is not read as unset.
        print("ROUTE launched=%s prepared=%s refusal=%s native_dq=%s native_key=%s generic_mask=%s"
              " arm_fwd=%s arm_bwd=%s"
              % (d.get("launched"), d.get("prepared"), d.get("refusal"), d.get("native_dq"),
                 d.get("native_key"), d.get("generic_mask"),
                 os.environ.get("XATTN_SCA_UMMA_FWD") or ("unset" if "XATTN_SCA_UMMA_FWD" not in os.environ else "empty"),
                 os.environ.get("XATTN_SCA_BWD_ARM") or ("unset" if "XATTN_SCA_BWD_ARM" not in os.environ else "empty")),
              flush=True)
    except Exception as exc:
        print("ROUTE-UNAVAILABLE diag %s: %s" % (type(exc).__name__, exc), flush=True); return
    try:
        # EVERY forward row is printed, with the whole tuple, and the mixed case is named. The
        # table is cumulative and keyed by (phase, selected, launched, state) -- it is NOT ordered,
        # and a cell can legitimately carry more than one forward row (a native-leaf refusal leaves
        # forward/umma/mma_sync/refused beside a forward/mma_sync/mma_sync/ran, and a later repeat
        # can add forward/umma/umma/ran). Printing one entry would hide that a native body also
        # ran, so nothing here is selected: every row is printed, and a cell whose forward rows
        # disagree on (selected, launched) says `FWD-MIXED yes` instead of picking a winner.
        rows = list(ops._flash_sca_native_phase_tags())
        print("PHASES " + " | ".join("%s %s %s %s" % (r[0], r[1], r[2], r[3])
                                     for r in rows if r[0] in ("dq", "dk", "dv")), flush=True)
        fwd = [r for r in rows if str(r[0]) in ("fwd", "forward")]
        print("FWD-N %d" % len(fwd), flush=True)
        for r in fwd:
            print("FWDROW " + " ".join(str(x) for x in r), flush=True)
        kinds = sorted({"%s/%s" % (str(r[1]), str(r[2])) for r in fwd})
        print("FWD-MIXED %s" % ("yes" if len(kinds) > 1 else "no"), flush=True)
        if not fwd:
            print("FWD unreported (no forward row in the tag table)", flush=True)
        else:
            print("FWD " + " ".join(kinds), flush=True)
    except Exception as exc:
        print("PHASES-UNAVAILABLE %s: %s" % (type(exc).__name__, exc), flush=True)
        print("FWD unreported (phase tags unreadable)", flush=True)

def pipe_launches():
    # An unavailable counter is not an observed zero. Fail closed with the reader's error named.
    try:
        import xattn_cuda
        ops = next(c for c in (getattr(xattn_cuda, "ops", None), xattn_cuda)
                   if c is not None and hasattr(c, "_flash_sca_native_dq_pipe_launches"))
        value = ops._flash_sca_native_dq_pipe_launches()
    except Exception as exc:
        raise RuntimeError("DQPIPE-UNAVAILABLE counter: %s: %s" % (type(exc).__name__, exc)) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("DQPIPE-UNAVAILABLE non-integer/negative count: %r" % (value,))
    return value


def run(arm, **kw):
    # The markers are what lets a FAIL name the arm and the pass: both arms share one process and
    # one sanitizer log, so the summary line alone cannot say which of the two trapped.
    print("ARM %s start" % arm, flush=True)
    y, y32, lse = xattn.ops.flash_sca_fwd(q, k, v, chunk_size=C, scale=None,
                                          high_precision_level=L, **kw)
    # Asserted in the sanitized process: y_fp32 is None at level 0 and present at levels 1-4
    # (presence against the level, not the dtype).
    assert (y32 is None) == (L == 0), "level %d returned y_fp32=%r" % (L, y32)
    saved = y if L == 0 else y32
    dq_before = pipe_launches()
    g = xattn.ops.flash_sca_bwd(dy, q, k, v, saved, lse, C, scale=None,
                                high_precision_level=L, **kw)
    torch.cuda.synchronize()
    dq_after = pipe_launches()
    dq_delta = dq_after - dq_before
    # One public backward launches at most one dQ pipe. Missing and inconsistent counts are
    # errors; an observed zero means that this backward did not take the pipe, not "generic".
    if dq_delta not in (0, 1):
        raise RuntimeError("DQPIPE-INVALID before=%d after=%d delta=%d"
                           % (dq_before, dq_after, dq_delta))
    print("DQPIPE arm=%s before=%d after=%d delta=%d"
          % (arm, dq_before, dq_after, dq_delta), flush=True)
    # EXPECT_ARMS and the arm accounting below read the `^ARM (dense|bos) ok` prefix, so it is the
    # same at every level.
    print("ARM %s ok  y=%s lse=%s dq=%s level=%d saved=%s"
          % (arm, tuple(y.shape), tuple(lse.shape), tuple(g[0].shape), L,
             "y" if L == 0 else "y_fp32"), flush=True)

for rep in range(REP):
    print("REPEAT %d/%d" % (rep + 1, REP), flush=True)
    run("dense")
    run("bos", bos_mask=bos)
route_line()
CASE
  RCM=$?
  # The sanitizer's own return code goes INTO the row's file, because summary.json must be able to
  # apply the same fail-closed predicate as the shell below: a nonzero rc after a zero-error
  # summary is a FAIL in the JSON as in the shell.
  printf 'CELL-RC %s\n' "$RCM" >> "$OUT/mc_${TAG}.out"
  CLOG=$OUT/mc_${TAG}.log
  NLINES=$(grep -c "ERROR SUMMARY" "$CLOG" 2>/dev/null)
  # The body this cell ran, read from the sanitized process's own stdout -- printed, never guessed.
  # A clean generic cell and a clean native cell must not read the same, and at levels 2-4 the
  # native leaf is not entered at all, so the string is part of the verdict line.
  ROUTE=$(grep -m1 -E '^(ROUTE|ROUTE-UNAVAILABLE) ' "$OUT/mc_${TAG}.out" 2>/dev/null || true)
  [ -n "$ROUTE" ] || ROUTE="ROUTE-UNAVAILABLE no route line in mc_${TAG}.out"
  # The forward body is the one body a default-route row can be wrong about, so it travels with the
  # route in every line that prints one -- PASS, FAIL and UNPARSED alike, and in summary.json. ALL
  # of its rows travel, not the first one: the tag table is cumulative and unordered, and a cell can
  # carry a native refusal beside a generic run.
  FWD=$(awk '/^(FWD-N|FWDROW|FWD-MIXED|FWD) /{printf "%s%s", (n++ ? " ;; " : ""), $0}' \
        "$OUT/mc_${TAG}.out" 2>/dev/null || true)
  [ -n "$FWD" ] || FWD="FWD unreported (no FWD line in mc_${TAG}.out)"
  ROUTE="${ROUTE}  ${FWD}"
  NTRAP=$(grep -c "Trace/breakpoint trap" "$CLOG" 2>/dev/null || true)
  NTRAP=${NTRAP:-0}
  # How many arms actually COMPLETED. compute-sanitizer prints `ERROR SUMMARY: 0 errors` even when
  # the driver ran nothing at all (a zero-repeat loop exits cleanly), so the summary alone cannot
  # tell a clean run from an EMPTY one -- that was an earlier vacuous-PASS finding, and this counter
  # is the second half of the fix (the first half refuses REPEATS<1 and an empty CELLS upfront).
  # Each arm that finishes prints exactly one `ARM <arm> ok` marker into the cell's own stdout
  # file; the expected count is 2 arms (dense, bos) x REPEATS_VAL.
  EXPECT_ARMS=$((2 * REPEATS_VAL))
  NOKS=$(grep -cE '^ARM (dense|bos) ok' "$OUT/mc_${TAG}.out" 2>/dev/null || true)
  NOKS=${NOKS:-0}
  # The body witness is per BACKWARD. A zero delta is the observed non-pipe count, not a guess
  # about which other leaf ran; at 128 level 0 the explicit-off row must remain distinguishable.
  # Count alone is not a completeness predicate: two dense rows and no bos row also sum to two.
  # Keep the raw count, but pass only the exact dense,bos sequence once per repeat, with no
  # malformed/extra DQPIPE row and with every delta equal to after-before. The CPU controls execute
  # this same embedded checker against duplicate, swapped, missing, malformed and wrong-delta rows.
  DQREPORT=$(python3 - "$OUT/mc_${TAG}.out" "$EXPECT_ARMS" <<'DQCHECK'
import re, sys
try:
    text = open(sys.argv[1], "r", errors="replace").read()
except Exception:
    text = ""
expect = int(sys.argv[2])
raw = re.findall(r"(?m)^DQPIPE[^\n]*$", text)
observations = re.findall(
    r"(?m)^DQPIPE arm=(dense|bos) before=([0-9]+) after=([0-9]+) delta=([01])$", text)
valid = (len(raw) == expect and len(observations) == expect and
         [o[0] for o in observations] == ["dense", "bos"] * (expect // 2) and
         all(int(after) - int(before) == int(delta)
             for arm, before, after, delta in observations))
print("%d %d %d" % (len(observations),
                    sum(int(o[3]) for o in observations), int(valid)))
DQCHECK
)
  read -r NDQWIT DQTOTAL DQVALID <<< "$DQREPORT"
  NDQWIT=${NDQWIT:-0}; DQTOTAL=${DQTOTAL:-0}; DQVALID=${DQVALID:-0}
  ROUTE="${ROUTE}  DQPIPE-WITNESS ${NDQWIT}/${EXPECT_ARMS} launches=${DQTOTAL} valid=${DQVALID}"
  # Worst over EVERY summary in the log, so log order cannot hide a failing process: a run that
  # produced two summaries (a forked child) once had its earlier, failing one dropped by `tail -1`.
  SUM=$(grep -o "ERROR SUMMARY: [0-9]* error" "$CLOG" 2>/dev/null | sort -n -k3 | tail -1)
  say "  cell=D=$D Dv=$DV  ERROR SUMMARY lines found: ${NLINES:-0}  (worst of them is the line below)"
  if [ -z "$SUM" ]; then
    say "  UNPARSED cell=D=$D Dv=$DV  no ERROR SUMMARY line -- NOT a pass"; fail=1; unparsed=1
    say "    route: ${ROUTE}  traps=${NTRAP}"
    RESULTS="$RESULTS
    (D=$D,Dv=$DV)  UNPARSED  rc=$RCM"
    continue
  fi
  NERR=$(printf '%s' "$SUM" | grep -oE "[0-9]+" | head -1)
  if [ "$RCM" -eq 0 ] && [ "$NERR" -eq 0 ] && [ "$NOKS" -eq "$EXPECT_ARMS" ] && [ "$DQVALID" -eq 1 ]; then
    npass=$((npass + 1))
    say "  PASS memcheck cell=D=$D Dv=$DV  $SUM  (sanitizer rc=0; arms completed ${NOKS}/${EXPECT_ARMS}; dQ-body witnesses ${NDQWIT}/${EXPECT_ARMS})"
    say "    route: ${ROUTE}  traps=${NTRAP}"
    RESULTS="$RESULTS
    (D=$D,Dv=$DV)  PASS      $SUM  arms ${NOKS}/${EXPECT_ARMS} dQ-body witnesses ${NDQWIT}/${EXPECT_ARMS}"
  else
    say "  FAIL memcheck cell=D=$D Dv=$DV  $SUM  (sanitizer rc=$RCM; arms completed ${NOKS}/${EXPECT_ARMS}; dQ-body witnesses ${NDQWIT}/${EXPECT_ARMS})"; fail=1
    say "    route: ${ROUTE}  traps=${NTRAP}"
    RESULTS="$RESULTS
    (D=$D,Dv=$DV)  FAIL      $SUM rc=$RCM arms ${NOKS}/${EXPECT_ARMS}"
    # The FIRST TWELVE `=========` header lines, of ANY class -- not a whitelist. Header lines are
    # the ones that do not begin with five spaces (those are the `at`/`by thread`/`Host Frame`
    # detail lines). `Trace/breakpoint trap` is a device trap and matches no memory-fault word, so
    # a whitelist prints nothing for exactly the failure this instrument was built to expose.
    say "    first header lines of ANY class (a trap is not a memory fault and will not match a memory-fault word):"
    grep -E "^========= " "$CLOG" 2>/dev/null | grep -vE "^=========     " | head -12 | sed "s/^/     /" || true
    say "    first host-side frames naming this tree (for the owner of the leaf):"
    grep -m 4 -E "host\.cc|_sca_native\.py|sliding_chunk_attention\.py" "$CLOG" 2>/dev/null | sed "s/^/     /" || true
  fi
done

# ---- the machine-readable form of this row -----------------------------------------------------
# summary.json carries every cell's numbers at fixed JSON paths. It is written BEFORE the verdict
# block, so a failing row keeps its numbers too: a failing cell's error count is evidence about the
# binary.
python3 - "$OUT" "$CELL_LIST" "$NCELLS_REQ" "$LEVEL_VAL" "$REPEATS_VAL" "$N_VAL" "$SRC_SHA" "$SRC_N" "$SOSHA" "$0" <<'SUMMARY' 2>&1 | tee -a "$LOG"
import hashlib, json, os, re, sys
out, cells, ncell, level, reps, nval, srcsha, srcn, sosha, selfpath = sys.argv[1:11]

def read(p):
    try:
        return open(p, "r", errors="replace").read()
    except Exception:
        return ""

rows, route_map = [], {}
tot_err = tot_trap = tot_noks = tot_expect = npass = 0
for cell in cells.split():
    D, DV = cell.split(":")
    tag = "D%s_Dv%s_L%s" % (D, DV, level)
    lt, ot = read(os.path.join(out, "mc_%s.log" % tag)), read(os.path.join(out, "mc_%s.out" % tag))
    sums = re.findall(r"ERROR SUMMARY: (\d+) error", lt)
    err = max((int(s) for s in sums), default=None)          # worst summary, so log order cannot hide one
    traps = lt.count("Trace/breakpoint trap")
    arms = len(re.findall(r"(?m)^ARM (?:dense|bos) ok", ot))
    expect = 2 * int(reps)
    dq_raw = re.findall(r"(?m)^DQPIPE[^\n]*$", ot)
    dq_observations = [{"arm": arm, "before": int(before), "after": int(after),
                        "delta": int(delta)} for arm, before, after, delta in re.findall(
        r"(?m)^DQPIPE arm=(dense|bos) before=([0-9]+) after=([0-9]+) delta=([01])$", ot)]
    dq_witness_complete = (len(dq_raw) == expect and len(dq_observations) == expect and
                           [o["arm"] for o in dq_observations] == ["dense", "bos"] * int(reps) and
                           all(o["delta"] == o["after"] - o["before"] for o in dq_observations))
    m = re.search(r"(?m)^(ROUTE .*|ROUTE-UNAVAILABLE .*)$", ot)
    route = m.group(1) if m else "ROUTE-UNAVAILABLE no route line"
    # The sanitized process's return code, written into this cell's own file by the shell. It is
    # part of the verdict: a zero-error summary with a nonzero rc is a failure, and a JSON that
    # dropped it would be a green wider than the driver's own line.
    mr = re.search(r"(?m)^CELL-RC (-?\d+)$", ot)
    rc = int(mr.group(1)) if mr else None
    # EVERY forward row of the cell, not the first: the table is cumulative and unordered, and a
    # mixed cell must read as mixed here rather than as whichever row was printed first.
    fwd_rows = re.findall(r"(?m)^FWDROW (.*)$", ot)
    fwd_kinds = sorted({"/".join(r.split()[1:3]) for r in fwd_rows if len(r.split()) >= 3})
    fwd_line = (re.search(r"(?m)^FWD (.*)$", ot) or [None, None])[1]
    fwd_mixed = len(fwd_kinds) > 1
    if err is None or rc is None:
        verdict = "UNPARSED"
    else:
        verdict = ("PASS" if rc == 0 and err == 0 and arms == expect
                   and dq_witness_complete else "FAIL")
    npass += 1 if verdict == "PASS" else 0
    tot_err += err or 0
    tot_trap += traps
    tot_noks += arms
    tot_expect += expect
    route_map["%s:%s" % (D, DV)] = route
    rows.append({"cell": "%s:%s" % (D, DV), "D": int(D), "Dv": int(DV), "verdict": verdict,
                 "errors": err, "traps": traps, "arms_completed": arms, "arms_expected": expect,
                 "rc": rc, "route": route, "fwd": fwd_line, "fwd_kinds": fwd_kinds,
                 "fwd_rows": fwd_rows, "fwd_mixed": fwd_mixed,
                 "dq_pipe_observations": dq_observations,
                 "dq_pipe_launches": sum(o["delta"] for o in dq_observations),
                 "dq_pipe_witness_complete": dq_witness_complete})


try:
    inst = hashlib.sha256(open(selfpath, "rb").read()).hexdigest()
except Exception:
    inst = None
so = sosha.split()[0] if re.match(r"^[0-9a-f]{64}\s", sosha) else None
mixed = ["%s" % r["cell"] for r in rows if r["fwd_mixed"]]
summary = {
    "instrument_sha256": inst, "sources_sha256": srcsha or None, "sources_files": int(srcn or 0),
    "so_sha256": so, "level": int(level), "levels": [int(level)], "dtype": "bf16",
    "deterministic": False, "n": int(nval), "repeats": int(reps),
    "cells_requested": int(ncell), "cells_passed": npass,
    "calls": tot_noks, "arms_completed": tot_noks, "arms_expected": tot_expect,
    "dq_pipe_observations": sum(len(r["dq_pipe_observations"]) for r in rows),
    "dq_pipe_launches": sum(r["dq_pipe_launches"] for r in rows),
    "dq_pipe_witness_complete": (bool(rows) and
                                 all(r["dq_pipe_witness_complete"] for r in rows)),
    # The row's own verdict, by the same fail-closed rule the shell uses: every requested cell
    # PASS (rc=0, worst ERROR SUMMARY 0, expected arms and dQ-body witnesses completed) and none
    # unparsed. A sanitizer with zero errors cannot mask an absent or inconsistent body witness.
    "verdict": ("PASS" if (npass == int(ncell)
                           and not any(r["verdict"] == "UNPARSED" for r in rows))
                else "FAIL"),
    "errors": tot_err, "traps": tot_trap, "route": route_map,
    "fwd_mixed_cells": mixed, "cells": rows,
}
path = os.path.join(out, "summary.json")
with open(path, "w") as fh:
    json.dump(summary, fh, indent=1, sort_keys=True)
print("  summary.json: %s" % path)
print("  level=%s cells=%d passed=%d errors=%d traps=%d arms=%d/%d so=%s"
      % (level, len(rows), npass, tot_err, tot_trap, tot_noks, tot_expect, so))
print("  forward rows carried per cell: %s"
      % ", ".join("%s=%s" % (r["cell"], "+".join(r["fwd_kinds"]) or "unreported") for r in rows))
if mixed:
    print("  FWD-MIXED cells (more than one forward body in one cell): %s" % " ".join(mixed))
print("  the route and EVERY forward row of every cell are in the file: a clean generic cell and a")
print("  clean native cell read differently there, which is the reason it is written, not described")
SUMMARY

say "=========== VERDICT"
say "  level=${LEVEL_VAL}  (one level per process: this summary belongs to level ${LEVEL_VAL} alone)"
say "  this row requested ${NCELLS_REQ} cell(s): $(printf '%s' "$CELL_LIST" | tr '\n' ' ')"
say "  per-cell table (this row, these bytes):$RESULTS"
if [ "$unparsed" -ne 0 ]; then
  say "  VERDICT-UNPARSED -- at least one cell produced no ERROR SUMMARY line, so the row cannot"
  say "  say whether that cell is clean or dirty. Exit 4, as the conventions above promise; this"
  say "  is never a pass and it is a different code from a measured FAIL (exit 1)."
  exit 4
fi
if [ "$fail" -ne 0 ]; then say "  VERDICT-FAIL -- the per-cell lines above name what failed, and the cell is in the table"; exit 1; fi
if [ "$npass" -ne "$NCELLS_REQ" ]; then
  say "  VERDICT-FAIL -- ${npass} of ${NCELLS_REQ} requested cells reached PASS, so a PASS for this row"
  say "  would be wider than the evidence. This guard exists because a review found the line below"
  say "  printing 'all nine cells' for a row that had driven one."
  exit 1
fi
say "  VERDICT-PASS memcheck clean at ${npass}/${NCELLS_REQ} cells driven: $(printf '%s' "$CELL_LIST" | tr '\n' ' ')"
say "  SCOPE OF THIS VERDICT, stated so it is not read wider: tool=memcheck; THIS binary (the .so"
say "  sha256 above); THESE inputs; level=${LEVEL_VAL}; N=$N_VAL; these cells. It is not a statement of"
say "  race freedom (that is --tool racecheck, not run here) and not a statement about any other"
say "  binary. The ROUTE is printed per cell above and in summary.json, and it is a record of what the"
say "  host last prepared, so a level whose body is the generic/scalar one cannot read as native here."
exit 0
