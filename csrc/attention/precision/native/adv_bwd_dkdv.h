// Key-owned dK/dV leaf of the native SM103 advanced (level 2-4) reader
// backward.  Leaf contract: reads q/k/v/dY and the row workspace, writes
// p.dk and p.dv and nothing else (dQ and the statistics sinks belong to the
// row-owned producer, so the aggregate launcher calls both and each output
// element has exactly one writer).
#pragma once

#include "attention/precision/native/adv_bwd_native.h"

// TORCH_CHECK in the shipped build; in a standalone probe build (the leaf unit
// probe) the same failure prints and aborts, so the leaf's device code can be
// exercised without linking c10.  The condition text is the check's identity.
#ifdef XATTN_ADV_STANDALONE
#include <cstdio>
#include <cstdlib>
#define XATTN_ADV_CHECK(cond, ...)                                       \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "adv_bwd CHECK failed: %s\n", #cond);          \
      std::abort();                                                      \
    }                                                                    \
  } while (0)
#else
#include <c10/util/Exception.h>
#define XATTN_ADV_CHECK(cond, ...) TORCH_CHECK(cond, __VA_ARGS__)
#endif

namespace xattn {
namespace ops {
namespace native_adv {

void LaunchDkdv(const Params& p, cudaStream_t stream);

}  // namespace native_adv
}  // namespace ops
}  // namespace xattn
