#!/bin/sh
# Two-translation-unit host-only check; no CUDA toolkit or tensor library needed.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT HUP INT TERM
CXX=${CXX:-c++}
"$CXX" --version
for opt in 0 3; do
  "$CXX" -std=c++17 -O"$opt" -Wall -Wextra -Werror -pedantic -I"$ROOT/csrc" \
    -DXATTN_BUCKET_HELPER_TU -c "$HERE/dimension_bucket_check.cc" \
    -o "$BUILD/helper-$opt.o"
  "$CXX" -std=c++17 -O"$opt" -Wall -Wextra -Werror -pedantic -I"$ROOT/csrc" \
    -c "$HERE/dimension_bucket_check.cc" -o "$BUILD/main-$opt.o"
  "$CXX" "$BUILD/helper-$opt.o" "$BUILD/main-$opt.o" -o "$BUILD/check-$opt"
  printf 'O%s ' "$opt"
  "$BUILD/check-$opt"
  if [ -n "${OUT:-}" ]; then
    mkdir -p "$OUT"
    cp "$BUILD/check-$opt" "$OUT/dimension-bucket-O$opt"
  fi
done
