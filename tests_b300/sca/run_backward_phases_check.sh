#!/bin/sh
# Link the REAL host driver to test-owned registry/stage/leaf stubs only.
# No registry TU, device body, built extension, CUDA context, or GPU is needed.
# nvcc supplies CUDA headers for the ABI; neither input contains device code.
#
# Usage: sh tests_b300/sca/run_backward_phases_check.sh [output_binary]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
if [ ! -f "$ROOT/csrc/flash_sca/native/umma_bwd.cu" ]; then
  echo "cannot locate repository csrc from $HERE" >&2
  exit 2
fi
TORCH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
OUT_BIN=${1:-/tmp/backward_phases_check}
echo "root=$ROOT torch=$TORCH testing=1"
nvcc -std=c++17 -O1 -x cu \
  -DXATTN_SCA_BWD_TESTING=1 -D_GLIBCXX_USE_CXX11_ABI="$TORCH_ABI" \
  -I"$ROOT/csrc" -I"$ROOT" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
  -o "$OUT_BIN" \
  "$HERE/backward_phases_check.cu" \
  "$ROOT/csrc/flash_sca/native/umma_bwd.cu" \
  -L"$TORCH/lib" -lc10 -ltorch_cpu -Xlinker -rpath -Xlinker "$TORCH/lib"
echo "compiled=$OUT_BIN"
"$OUT_BIN"
