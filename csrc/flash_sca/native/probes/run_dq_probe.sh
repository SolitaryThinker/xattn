#!/bin/bash
# Build and RUN the native dQ leaf FP64 probe, and census the cubin.
#
# Torch-free: one nvcc TU (dq_probe.cu, hand-written host FP64 reference), no
# extension, no python. The leaf compiled here is the TREE's copy
# (csrc/flash_sca/native/umma_bwd_dq.cuh); its sha256 is printed, so an evidence
# row can name the exact source that produced it.
#
#   bash csrc/flash_sca/native/probes/run_dq_probe.sh   # writes $OUT/probe.run.log
#   OUT=/some/dir bash .../run_dq_probe.sh
#
# Non-zero exit = the probe failed or refused a case, or the injected-refusal
# fixture (further down) failed to refuse; the per-case table is in
# $OUT/probe.run.log.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
SRC="$ROOT/csrc"
OUT="${OUT:-$HERE/out}"
LEAF="$SRC/flash_sca/native/umma_bwd_dq.cuh"
[ -f "$LEAF" ] || { echo "no leaf at $LEAF" >&2; exit 1; }
mkdir -p "$OUT"

NVCC=${NVCC:-nvcc}
FLAGS=(-std=c++17 -O3 -lineinfo -Xptxas -v
       -gencode arch=compute_103a,code=sm_103a
       -I "$SRC" -I "$SRC/cutlass/include"
       --expt-relaxed-constexpr -DNDEBUG)

echo "== nvcc: $($NVCC --version | tail -1)"
echo "== leaf compiled in this job: $LEAF"
sha256sum "$LEAF"
# The probe calls LaunchUmmaDq<E, 128, 128> and LaunchUmmaDq<E, 256, 256>, and both try the
# pipelined dQ leaf first (umma_bwd_dq.cuh); its definitions come from the dQ-pipe
# instantiation TUs linked here, one file per element and bucket.
"$NVCC" "${FLAGS[@]}" "$HERE/dq_probe.cu" \
    "$SRC/flash_sca/native/instantiations/sca_umma_bwd_dq_pipe_bf16.cu" \
    "$SRC/flash_sca/native/instantiations/sca_umma_bwd_dq_pipe_fp16.cu" \
    "$SRC/flash_sca/native/instantiations/sca_umma_bwd_dq_pipe_d128_v128_bf16.cu" \
    "$SRC/flash_sca/native/instantiations/sca_umma_bwd_dq_pipe_d128_v128_fp16.cu" -o "$OUT/dq_probe" \
    2>&1 | tee "$OUT/probe.build.log"
st=${PIPESTATUS[0]}
echo "== build exit=$st"
if [ "$st" -ne 0 ]; then grep -E 'error|Error' "$OUT/probe.build.log" | head -30; exit 1; fi

echo "== running probe"
"$OUT/dq_probe" 2>&1 | tee "$OUT/probe.run.log"
rc=${PIPESTATUS[0]}
echo "== probe exit=$rc"

# ---------------------------------------------------- the fail path, exercised
# A green run never executes the probe's refusal path, so a defect there would ship
# silently. The same binary is run once more with DQ_PROBE_INJECT_REFUSAL set: one
# valid case gets a non-null dq_accum, the leaf refuses it, and the probe MUST exit
# nonzero. A zero here means the refusal counter or the status path is broken, and
# this script must not end green in that case.
echo "== fail-path fixture (an injected numerical-arm refusal must be nonzero)"
DQ_PROBE_INJECT_REFUSAL=1 "$OUT/dq_probe" >"$OUT/probe.inject.log" 2>&1
inj_rc=$?
tail -2 "$OUT/probe.inject.log" | sed 's/^/   /'
echo "   injected run exit=$inj_rc (expected nonzero)"
if [ "$inj_rc" -eq 0 ]; then
  echo "   FAIL: an injected numerical-arm refusal exited 0 - the verdict path is broken"
  rc=1
fi

# ---------------------------------------------------------------- native proof
# Pinned form: source sha + sass sha + elf-listing sha, plus a PER-SYMBOL
# demangled join. UTCHMMA (tcgen05.mma) must be > 0 and LDSM (mma.sync operand
# load) must be 0 in EVERY leaf kernel; LC/UTCMMA/UTCLD/OMMA are not mnemonics on
# this architecture, so counting them would measure nothing.
CUOBJ=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
NM=$(command -v nm || echo /usr/bin/nm)
SASS="$OUT/dq_probe.sass"
ELF="$OUT/elf.list"
echo "== cubin census: binary identity"
sha256sum "$OUT/dq_probe" 2>/dev/null | sed 's/^/   binary /'
"$CUOBJ" --list-elf "$OUT/dq_probe" > "$ELF" 2>&1 || true
cat "$ELF" | sed 's/^/   elf: /'
sha256sum "$ELF" 2>/dev/null | sed 's/^/   listing /'
"$CUOBJ" -sass "$OUT/dq_probe" > "$SASS" 2> "$SASS.err" || true
echo "   sass bytes $(wc -c < "$SASS")  stderr: $(head -1 "$SASS.err")"
sha256sum "$SASS" 2>/dev/null | sed 's/^/   sass  /'
echo "== whole-binary mnemonic counts"
for m in UTCHMMA LDTM STTM UTMALDG UBLKCP LDSM HMMA UTCMMA FFMA; do
  # `|| echo 0` would print a second line: grep exits 1 on no match AND prints
  # nothing, so the count is captured and defaulted instead.
  c=$(grep -cw "$m" "$SASS" 2>/dev/null); c=${c:-0}
  printf '   %-9s %s\n' "$m" "$c"
done
echo "== per-symbol join (demangled function name -> UTCHMMA / LDSM)"
awk '
  /Function : / { sym=$3 }
  /\/\*[0-9a-f]+\*\// {
    if (match($0, /UTCHMMA/)) mma[sym]++
    if (match($0, /LDSM/))    ldsm[sym]++
    seen[sym]=1
  }
  END {
    n=0
    for (s in seen) {
      if (mma[s]+0 > 0 || ldsm[s]+0 > 0) {
        printf "   %-72s UTCHMMA %4d  LDSM %d\n", s, mma[s]+0, ldsm[s]+0
        n++
      }
    }
    if (n == 0) print "   (no per-symbol join: no `Function :` markers in this dump)"
  }
' "$SASS" | sort
echo "== host launcher / preflight symbols"
"$NM" -C "$OUT/dq_probe" 2>/dev/null | grep -E 'LaunchUmmaDq|CanLaunchUmmaDq' |
    sed 's/^/   nm: /' | head -8
echo "== is the device asm guard on? (it defaults ON; a build without it emits nothing)"
grep -n 'XATTN_UMMA_DEVICE_ASM' "$SRC/flash_sca/native/umma_fwd.cuh" "$LEAF" 2>/dev/null | head -6
# The census is a report, not an alternative verdict. A probe that failed or
# refused a case must take the job's exit code with it: without the line below the
# script ends after the census and returns 0 for a red table, so a red table
# would be readable as a succeeded job. This line is the difference between a log
# and a check.
[ "${rc:-0}" -eq 0 ] || exit "$rc"
echo "== DONE"
exit 0
