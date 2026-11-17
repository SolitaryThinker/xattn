#!/bin/bash
# Native dQ leaf: compile the two instantiation TUs for sm_103a, nothing else.
#
# This is the build-only half of run_dq_probe.sh (which also compiles the FP64
# probe and runs it); use it when only the register/spill table is wanted.
#
#   bash csrc/flash_sca/native/probes/build.sh      # OUT defaults to ./out here
#   OUT=/some/dir bash .../build.sh
#
# Refuses to run anywhere but a tree that holds the leaf, so a stale copy cannot
# silently compile something else.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/kernel/csrc"
OUT="${OUT:-$HERE/out}"
LEAF="$SRC/flash_sca/native/umma_bwd_dq.cuh"

[ -f "$LEAF" ] || { echo "no leaf at $LEAF" >&2; exit 1; }
mkdir -p "$OUT"

NVCC=${NVCC:-nvcc}
FLAGS=(-std=c++17 -O3 -lineinfo -Xptxas -v
       -gencode arch=compute_103a,code=sm_103a
       -I "$SRC" -I "$SRC/cutlass/include"
       --expt-relaxed-constexpr -DNDEBUG)
# -c only: these TUs instantiate the launcher and the preflight, no main.
CFLAGS=(-c)

echo "== nvcc: $($NVCC --version | tail -1)"
echo "== leaf: $LEAF"
sha256sum "$LEAF"
rc=0
for tu in sca_umma_bwd_dq_d128_v128 sca_umma_bwd_dq_d256_v256; do
  src="$SRC/flash_sca/native/instantiations/$tu.cu"
  [ -f "$src" ] || { echo "missing $src" >&2; rc=1; continue; }
  echo "-- $tu"
  "$NVCC" "${FLAGS[@]}" "${CFLAGS[@]}" "$src" -o "$OUT/$tu.o" 2>&1 |
      tee "$OUT/$tu.compile.log" | grep -E 'error|Used [0-9]+ registers|bytes stack|bytes spill|bytes gmem'
  st=${PIPESTATUS[0]}
  echo "   exit=$st"
  [ "$st" -eq 0 ] || rc=1
done
echo "== DONE (exit $rc)"
exit "$rc"
