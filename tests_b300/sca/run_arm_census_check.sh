#!/bin/sh
# Arm census for the native FlashSCA path: a documented arm must be read, and an
# arm that is read must be documented.
#
# WHY THIS IS A TEST AND NOT A NOTE. A defect of this kind leaves the tree
# compiling and the arm still documented, and the only symptom is a
# measurement that means something other
# than its label:
#   * XATTN_SCA_BWD (the body A/B arm) is documented in umma_bwd.h and read by
#     umma_bwd.cu and host.cc; a merge onto a base that predates that reader
#     can keep the documentation without the read, which leaves
#     the header describing an arm that nothing reads, and XATTN_SCA_BWD=0
#     becomes a variable nobody refuses;
#   * a layout parser that accepts any value beginning with 'h' invents a
#     silent third arm in an instrument built to label two;
#   * a dimension-bucket rule stated in three files instead of one header
#     drifts until a link error exposes it.
# Nothing in the build notices any of these: a getenv nobody calls and a name
# nobody reads are both silent.
#
# WHAT THIS IS AND WHAT IT IS NOT. It is a SOURCE-level census over
# `csrc/flash_sca/native/`, the directory that owns the native arm vocabulary.
# It is not a binary fact: it cannot see the shipped .so, and it must never be
# cited as evidence that a body ran. Its claims are narrow:
#   1. every arm named in DOCUMENTED below is read in that directory by a
#      literal `getenv("<name>")`, so a header may not document an arm that
#      nothing reads;
#   2. every `getenv("XATTN_...")` in that directory names an arm in
#      DOCUMENTED, so a new arm cannot enter the tree without a line here and a
#      reader someone can find.
# Per-arm read sites are printed because a second site is where the vocabulary
# drifts (the branch reads XATTN_SCA_BWD in host.cc *and* umma_bwd.cu, with two
# different meanings). The count is reported, not failed: a deliberate second
# site is a design decision this file is in no position to overrule, but it
# should never be invisible again.
#
# BOUNDARY. The table is armed to the branch that owns the native path, not to
# every branch of the repo: run against a tree that predates the vocabulary it
# goes red naming the arms that tree does not have, which is a true statement
# about the tree and not a defect in the census. Read the FAIL lines, not the
# exit code alone - the two directions mean different things ("this tree is
# missing an arm" versus "this tree reads an arm the table does not know").
#
# SCOPE, stated so the check does not overclaim: only `getenv("XATTN_...")`
# literals under csrc/flash_sca/native/ are considered. That deliberately
# excludes csrc/attention/hopper/ (XATTN_ATTENTION_VARLEN_SCHEDULER,
# XATTN_FLASH_SCA_VARLEN_SCHEDULER) and the probe-only DQ_PROBE_INJECT_REFUSAL
# arm, which is not XATTN_-prefixed and belongs to csrc/flash_sca/native/probes.
# The cost of the check is one line in a header the day someone refactors a
# literal getenv into an indirect constant: that is the moment the vocabulary
# changes, and it is the moment to look.
#
# Usage: run_arm_census_check.sh [root] [out_json]
#   root defaults to $PWD; prints `arm ...` lines and one JSON line, exit 0
#   only when both claims hold.
set -u

root=${1:-$PWD}
out=${2:-}
dir="$root/csrc/flash_sca/native"

# The arms the native path owns. Every entry has a documented meaning in a
# header of this directory and a reader in the C++ that implements it.
#
# XATTN_SCA_BWD is listed because umma_bwd.h documents it and umma_bwd.cu +
# host.cc read it. If it is ever retired, the retirement must
# delete the header's documentation, this line, and the name's read site
# together - and, if a process setting it should not silently time the default,
# put a raise on the retired name. If instead it is kept, keep the line. Either
# way this census is what makes the choice visible: drop the read and leave the
# line, and claim 1 goes red.
#
# XATTN_SCA_BWD_LEGTIME and XATTN_SCA_BWD_DIAGNOSTIC select no body. The first
# times the backward's legs (host-side, default off; umma_bwd.cu). The second
# is only reported back by the backward record (host.cc). Both are listed
# because they are read here, which is what claim 2 counts.
#
# XATTN_SCA_FWD_PIPE selects the pipelined level-0 forward at (256,256) and (128,128): unset or empty
# is on while XATTN_SCA_UMMA_FWD is unset too, "1" is on, "0" is the control arm
# (the retained unpipelined forward), anything else raises; umma_fwd_pipe_host.h documents and reads
# it, and host.cc resolves it together with XATTN_SCA_UMMA_FWD once per call.
#
# XATTN_FWD_PROBE selects no body and is read by no shipped build: its one read site
# (umma_fwd_pipe_host.h ProbeHook, which documents it) is compiled only in a probe build
# that prepends XATTN_FWD_PROBE_HOOKS to copies of the forward instantiation TUs. It is
# listed because the read is in this directory, which is what claim 2 counts.
#
# XATTN_SCA_BWD_DQ_PIPE selects the dQ body of a level-0 call at (256,256) and at (128,128): the
# pipelined dQ leaf (unset, empty or "1", the default) or DqKernel ("0", the control arm);
# bwd_pipe_host.h documents and reads it.
#
# XATTN_SCA_BWD_KEY_ARM selects the key (dK/dV) body of a level-0 (128,128) call: umma_bwd_dkdv.cuh
# reads it at two sites on every call, behind the fused M128 key leaf and its pair arms. It is not
# this file's own history, and it is listed here because claim 2 below was RED on main without it:
# the tree read an arm the table did not name, which is exactly the drift this check exists to
# catch. Adding the line is the repair the check asks for, not a waiver of it - claim 1 still
# requires the name be read, and it is (twice).
DOCUMENTED="XATTN_SCA_ORDER XATTN_SCA_LAYOUT XATTN_SCA_BWD_ARM XATTN_SCA_UMMA_FWD XATTN_SCA_BWD XATTN_SCA_BWD_LEGTIME XATTN_SCA_BWD_DIAGNOSTIC XATTN_SCA_FWD_PIPE XATTN_FWD_PROBE XATTN_SCA_BWD_DQ_PIPE XATTN_SCA_BWD_KEY_ARM"

checks=0
failures=0

fail() {
  checks=$((checks + 1))
  failures=$((failures + 1))
  echo "FAIL $1"
}

pass() {
  checks=$((checks + 1))
  echo "ok   $1"
}

if [ ! -d "$dir" ]; then
  echo "FAIL nosrc: no $dir (pass the repository root as the first argument)"
  echo "{\"checks\": 1, \"failures\": 1, \"passed\": false}"
  exit 1
fi

# Every literal getenv("XATTN_...") in the directory, as bare names, sorted.
reads=$(grep -rho 'getenv("XATTN_[A-Z0-9_]*")' "$dir" \
        | sed 's/^getenv("//; s/")$//' \
        | sort -u)

echo "dir=$dir"
echo "documented=$(echo $DOCUMENTED | wc -w) read=$(echo "$reads" | wc -w)"

# Claim 1: every documented arm is read, with its sites printed either way.
for name in $DOCUMENTED; do
  sites=$(grep -rn "getenv(\"$name\")" "$dir" | cut -d: -f1,2 | tr '\n' ' ')
  count=$(grep -rc "getenv(\"$name\")" "$dir" 2>/dev/null | awk -F: '{n += $NF} END {print n + 0}')
  echo "arm  $name sites=$count ${sites}"
  if [ "$count" -eq 0 ]; then
    fail "documented-but-unread: $name is named as an arm by the native path and read nowhere in $dir"
  else
    pass "documented-and-read: $name"
  fi
done

# Claim 2: every arm read is documented here.
for name in $reads; do
  found=0
  for known in $DOCUMENTED; do
    if [ "$name" = "$known" ]; then
      found=1
    fi
  done
  if [ "$found" -eq 0 ]; then
    fail "read-but-undocumented: $name is read in $dir and missing from DOCUMENTED in this file"
  else
    pass "read-and-documented: $name"
  fi
done

if [ "$out" != "" ]; then
  printf '{"dir": "%s", "documented": %s, "read": %s, "checks": %s, "failures": %s}\n' \
    "$dir" "$(echo $DOCUMENTED | wc -w)" "$(echo "$reads" | wc -w)" "$checks" "$failures" > "$out"
fi

if [ "$failures" -ne 0 ]; then
  echo "{\"checks\": $checks, \"failures\": $failures, \"passed\": false}"
  exit 1
fi
echo "{\"checks\": $checks, \"failures\": 0, \"passed\": true}"
exit 0
