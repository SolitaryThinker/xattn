#!/bin/sh
# Builds and runs thread_phase_tags_check.cc: host only, no GPU, no CUDA runtime.
# Usage: run_thread_phase_tags_check.sh [root] [binary]
#   root defaults to $PWD (the repository top); exit 0 only when every check holds.
set -u
root=${1:-$PWD}
bin=${2:-/tmp/thread_phase_tags_check}
${CXX:-g++} -std=c++17 -O1 -pthread -I"$root" "$root/tests_b300/sca/thread_phase_tags_check.cc" \
  -o "$bin" || exit 2
"$bin"
