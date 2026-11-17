#!/bin/sh
# Build and run the stage-mask refusal probe against the launcher AS IT STANDS IN THE TREE.
#
# The launcher is csrc/flash_sca/native/dispatch.cu. Its nine (D, Dv) bucket bodies come
# from backward_stage_refusal_stubs.cu, so this needs no built extension, no CUDA context
# and no GPU: only nvcc, because the launcher is CUDA. The probe never launches a kernel -
# it asks the guards at the top of LaunchBackwardStages to refuse, and they fire before
# the bucket branch; the one mask it expects to pass (empty, no accumulator) launches nothing.
#
# A source-level fact, not a .so fact: it proves the guards in this tree, not which binary
# the package ships. Where build/lib.linux-*/xattn_cuda*.so exists, linking that instead is
# strictly stronger, and the two can differ only if the tree and the build do.
#
#   sh tests_b300/sca/run_backward_stage_refusal_check.sh [root] [out_bin]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:-$(cd "$HERE/../.." && pwd)}
OUT_BIN=${2:-$ROOT/backward_stage_refusal_check}
if [ ! -f "$ROOT/csrc/flash_sca/native/dispatch.cu" ]; then
  echo "cannot locate repository csrc from $HERE" >&2
  exit 2
fi
TORCH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
TORCH_ABI=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
echo "root=$ROOT torch=$TORCH stub_link=1"
nvcc -std=c++17 -O1 -x cu \
  -D_GLIBCXX_USE_CXX11_ABI="$TORCH_ABI" \
  -I"$ROOT/csrc" -I"$ROOT" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
  -o "$OUT_BIN" \
  "$HERE/backward_stage_refusal_check.cu" \
  "$ROOT/csrc/flash_sca/native/dispatch.cu" \
  "$HERE/backward_stage_refusal_stubs.cu" \
  -L"$TORCH/lib" -lc10 -ltorch_cpu -lc10_cuda -Xlinker -rpath -Xlinker "$TORCH/lib"
echo "compiled=$OUT_BIN"
"$OUT_BIN"
