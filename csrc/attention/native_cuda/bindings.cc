// Python bindings for the native full-causal SoftDelta reader host boundary.
//
// Registration follows the existing extension convention: the Python module is
// xattn_cuda and these functions live in its `ops` submodule, so they are
// reached as xattn_cuda.ops._native_attention_fwd / _native_attention_bwd.
// The root module registers this file through DefineNativeAttentionOps.
//
// _native_attention_level_mask reports which precision levels are implemented
// and linked for a given tensor's device. Nothing in this file may advertise a
// level that has no linked implementation.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>

#include "attention/native_cuda/reader.h"

namespace xattn {
namespace ops {

namespace {

// Level support is a property of what is actually compiled and linked into
// this extension, so it is derived from the root registration macros that are
// set atomically with the complete source lists:
//   XATTN_HAS_NATIVE_READER    - native host + bindings + forward + basic
//                                backward are linked: levels 0 and 1 exist.
//   XATTN_HAS_NATIVE_PRECISION - additionally the native RowStatistics/Main/
//                                Ordered path and the portable helpers: levels
//                                2, 3 and 4 exist.
// XATTN_HAS_SM103 alone is an architecture macro and registers nothing, so it
// contributes no bits here.
#if defined(XATTN_HAS_NATIVE_PRECISION) && !defined(XATTN_HAS_NATIVE_READER)
#error "XATTN_HAS_NATIVE_PRECISION requires XATTN_HAS_NATIVE_READER: the advanced precision path needs the native reader host and basic backward"
#endif

#ifdef XATTN_HAS_NATIVE_READER
constexpr int64_t kNativeBasicLevels = 0x3;  // bits 0, 1
#else
constexpr int64_t kNativeBasicLevels = 0x0;
#endif
#ifdef XATTN_HAS_NATIVE_PRECISION
constexpr int64_t kNativeAdvancedLevels = 0x1C;  // bits 2, 3, 4
#else
constexpr int64_t kNativeAdvancedLevels = 0x0;
#endif

constexpr int64_t kNativeReaderLevels =
    kNativeBasicLevels | kNativeAdvancedLevels;

// Support metadata, not a qualification claim: bit k means "level k has a
// linked native implementation on this device and build". The initial native
// path is Blackwell cc-10.3 only, so any other device reports 0 rather than
// advertising a path the binary cannot serve.
int64_t NativeAttentionLevelMask(const torch::Tensor& q) {
  if (!q.is_cuda()) {
    return 0;
  }
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(static_cast<c10::DeviceIndex>(q.get_device()));
  if (properties == nullptr || properties->major != 10 ||
      properties->minor != 3) {
    return 0;
  }
  return kNativeReaderLevels;
}

// Visibilities the native advanced-precision (level 2-4) backward serves on
// this device: bit 0 full causal, bit 2 sliding chunk. Window (bit 1) has no
// native leaf. Zero without the precision group or off cc-10.3.
int64_t NativePrecisionVisibilityMask(const torch::Tensor& q) {
  if (kNativeAdvancedLevels == 0 || NativeAttentionLevelMask(q) == 0) {
    return 0;
  }
  return 0x5;
}

}  // namespace

void DefineNativeAttentionOps(py::module& m) {
  m.def("_native_precision_visibility_mask", &NativePrecisionVisibilityMask,
        "Bitmask of visibilities (bit 0 full causal, bit 1 window, bit 2 sliding "
        "chunk) the native advanced-precision backward implements on this tensor's "
        "device. Support metadata only.",
        py::arg("q"));
  m.def(
       "_native_attention_level_mask", &NativeAttentionLevelMask,
       "Bitmask of native reader precision levels whose paired forward and "
       "backward implementations are linked for this tensor's device: bit k "
       "set means level k is implemented. 0 for a non-CUDA tensor or a device "
       "other than cc-10.3, and 0 for every level whose leaf is not linked. "
       "This reports support, not numerical or performance qualification.",
       py::arg("q"))
      .def(
       "_native_attention_fwd", &attention::native_cuda::ReaderFwd,
       "Native full-causal single-reader forward. Returns (y, lse): y is "
       "[B,Nq,Hq,Dv] in Q's dtype, lse is float32 [B,Hq,Nq] in natural log "
       "with -inf on empty rows. high_precision_level 0 takes no output_state; "
       "levels 1..4 require a contiguous float32 [B,Nq,Hq,Dv] output_state. "
       "Previous K/V precede current K/V in the logical key space, so "
       "k_segment_idx covers Nprev+Nk keys.",
       py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
       py::arg("prev_k") = py::none(), py::arg("prev_v") = py::none(),
       py::arg("q_segment_idx") = py::none(),
       py::arg("k_segment_idx") = py::none(), py::arg("strict_past") = false,
       py::arg("high_precision_level") = 0,
       py::arg("output_state") = py::none())
      .def(
          "_native_attention_bwd", &attention::native_cuda::ReaderBwd,
          "Native full-causal single-reader backward. Returns (dq, dk, dv, "
          "dprev_k, dprev_v) in Q's dtype; the previous-key gradients are "
          "returned only when previous K/V were supplied. Accepts "
          "high_precision_level 0 and 1 only and rejects 2..4, which belong to "
          "the precision orchestration. level must name the saved state "
          "actually passed: Q's dtype at level 0, float32 at level 1.",
          py::arg("dy"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("saved_output"), py::arg("lse"), py::arg("scale"),
          py::arg("prev_k") = py::none(), py::arg("prev_v") = py::none(),
          py::arg("q_segment_idx") = py::none(),
          py::arg("k_segment_idx") = py::none(),
          py::arg("strict_past") = false, py::arg("high_precision_level") = 0,
          py::arg("deterministic") = false);
}

}  // namespace ops
}  // namespace xattn
