#!/bin/sh
# The seam-inertness pair, both sides on one machine and one build each.
#
#   base side = the same tree with only Backward's post-allocation schedule
#               replaced by the complete old generic schedule
#               (`seam_inertness_probe.py swap --to base`)
#
# Headers, the BackwardRequest parser, namespace aliases and diagnostic bindings
# are retained: the current host still needs their types outside the swapped span.
# This is NOT a historical checkout (a job copy has no .git). This pair forces
# the generic body on both sides; it tests that arm, not default native selection
# or a preparation refusal. Per-case phase tags must corroborate the body choice.
# Byte identity is never native-serving proof, and det-off atomic dQ bit identity
# is not a correctness requirement. Fixed FP64 qualification is separate.
#
# Both sides are built explicitly. The probe rebuilds only when a call is
# unusable, and on the base side the seam binary imports and runs fine, so a
# missing explicit build would measure one binary twice and look like a pass.
#
#   OUT=<dir> sh tests_b300/sca/seam_pair.sh
#
# Writes $OUT/seam_inertness_{stageb,base}.json and keeps the seam side at
# $OUT/host_seam.cc. Restores SOURCE on exit (including failed build/probe/compare),
# not the extension: rebuild the restored tree before other use. Retains both
# imported extension files with their SHA256; imports outside this tree fail.
set -e
OUT="${OUT:-/tmp}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export OUT
cd "$(dirname "$0")/../.."
unset XATTN_TARGET_SM
case "${XATTN_SCA_BWD_ARM-generic}" in
    generic) export XATTN_SCA_BWD_ARM=generic ;;
    *) echo "FAIL: seam_pair requires XATTN_SCA_BWD_ARM=generic (or unset)"; exit 2 ;;
esac
export MAX_JOBS=32
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/xt}"

keep_binary() {
    python - "$1" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import torch  # Load PyTorch's shared libraries before the extension.
import xattn_cuda
p = Path(xattn_cuda.__file__).resolve()
if not p.is_relative_to(Path.cwd().resolve()):
    raise RuntimeError("imported extension is outside the candidate: " + str(p))
label = sys.argv[1]
kept = Path(os.environ["OUT"]) / (label + "_xattn_cuda.so")
shutil.copy2(p, kept)
row = {"label": label, "imported_path": str(p), "kept_path": str(kept),
       "sha256": hashlib.sha256(kept.read_bytes()).hexdigest()}
(Path(os.environ["OUT"]) / (label + "_binary.json")).write_text(json.dumps(row) + "\n")
print(json.dumps(row))
PY
}

echo "== order and body-swap checks (source only, no write/build)"
python tests_b300/sca/seam_inertness_probe.py order
python tests_b300/sca/seam_inertness_probe.py order --selftest
python tests_b300/sca/seam_inertness_probe.py swap --to base --dry-run
python tests_b300/sca/pair_compare.py --selftest

echo "== seam side: build (backward arm: ${XATTN_SCA_BWD_ARM-unset})"
cp csrc/flash_sca/native/host.cc "$OUT/host_seam.cc"
trap 'cp "$OUT/host_seam.cc" csrc/flash_sca/native/host.cc' 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
python setup.py build_ext --inplace -j 32
keep_binary stageb
echo "== seam side: probe"
python tests_b300/sca/seam_inertness_probe.py probe --label stageb \
    --out "$OUT/seam_inertness_stageb.json"

echo "== base side: replace Backward schedule only; retain headers/helpers"
python tests_b300/sca/seam_inertness_probe.py swap --to base
if grep -nE 'PrepareBackwardUmma\(|PlanBackwardPhases\(|CheckBackwardPhases\(|LaunchBackwardPhases\(' \
    csrc/flash_sca/native/host.cc; then
    echo "FAIL: a phase-driver call remains in the generic base"
    exit 1
fi
cp csrc/flash_sca/native/host.cc "$OUT/host_base.cc"
python setup.py build_ext --inplace -j 32
keep_binary base
echo "== base side: probe"
python tests_b300/sca/seam_inertness_probe.py probe --label base \
    --out "$OUT/seam_inertness_base.json"

echo "== restore the seam source (extension still belongs to the last build)"
cp "$OUT/host_seam.cc" csrc/flash_sca/native/host.cc

echo "== compare the two sides (the pair rule, including the anti-vacuity rows)"
python tests_b300/sca/pair_compare.py
echo "done"
