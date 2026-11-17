#include "bindings.h"
#include "softdelta/gate.h"
#ifdef XATTN_HAS_SM90
#include "softdelta/bwd.h"
#include "softdelta/fwd.h"
#endif

namespace xattn {
namespace ops {

void DefineSoftDeltaOps(py::module& m) {
  m.def(
       "_softdelta_gate_fwd", &SoftDeltaGateFwd,
       "SoftDelta GQA-broadcast gate FWD: read - sigmoid(gate) * neg_read (into output when given)",
       py::arg("read"), py::arg("neg_read"), py::arg("gate"), py::arg("output") = py::none())
      .def(
          "_softdelta_gate_restore", &SoftDeltaGateRestore,
          "SoftDelta read from the gated output: y + sigmoid(gate) * neg_read, or `precomputed` "
          "unchanged with no launch",
          py::arg("y"), py::arg("neg_read"), py::arg("gate"), py::arg("precomputed") = py::none())
      .def(
          "_softdelta_gate_bwd", &SoftDeltaGateBwd,
          "SoftDelta GQA-broadcast gate BWD: (neg_read_grad, gate_grad, read); read is the "
          "restored query read when `y` is given (fused with the restore), else empty",
          py::arg("dy"), py::arg("neg_read"), py::arg("gate"), py::arg("y") = py::none())
      .def(
          "_softdelta_gate_bwd_deltas", &SoftDeltaGateBwdDeltas,
          "SoftDelta gate BWD with both reads' backward deltas instead of the restored read: "
          "(neg_read_grad, gate_grad, delta_read [B,H,L], delta_neg [B,Hkv,L]); four undefined "
          "tensors when the layout is not one it takes (the caller restores the read instead)",
          py::arg("dy"), py::arg("neg_read"), py::arg("gate"), py::arg("y"))
      .def(
          "_softdelta_merge_gradients", &SoftDeltaMergeGradients,
          "SoftDelta K/V gradient merge (in place into 16-bit read gradients)",
          py::arg("read_key_grad"), py::arg("neg_read_key_grad"),
          py::arg("dneg_q"), py::arg("read_value_grad"),
          py::arg("neg_read_value_grad"));
#ifdef XATTN_HAS_SM90
  m.def(
      "_softdelta_fwd", &SoftDeltaFwd,
      "SoftDelta fused reads FWD: (output [B, Lq, Hkv + H, V], lse [B, Hkv + H, Lq]), class-major readers; "
      "visibility 0 causal, 1 sliding window (span = window_size), 2 sliding chunk (span = chunk_size)",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("neg_q"), py::arg("scale"),
      py::arg("q_segment_idx") = py::none(), py::arg("k_segment_idx") = py::none(),
      py::arg("output_state") = py::none(), py::arg("visibility") = 0, py::arg("span") = 0,
      py::arg("reset") = false);
  m.def(
      "_softdelta_fwd_varlen", &SoftDeltaFwdVarlen,
      "SoftDelta packed-sequence reads FWD (varlen causal + varlen strict): _softdelta_fwd's readers, lse",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("neg_q"), py::arg("cu_seqlens"),
      py::arg("max_seqlen"), py::arg("scale"));
  m.def(
      "_softdelta_bwd", &SoftDeltaBwd,
      "SoftDelta fused BWD from the gated output and negative reads: (dq, dk, dv, dneg_q, gate_grad in the "
      "gate's shape); dneg_q is empty with fold_neg_q",
      py::arg("dy"), py::arg("gate"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("neg_q"),
      py::arg("y"), py::arg("neg_read"), py::arg("lse"), py::arg("scale"),
      py::arg("q_segment_idx") = py::none(), py::arg("k_segment_idx") = py::none(),
      py::arg("fold_neg_q") = false, py::arg("reader_groups") = -1,
      py::arg("visibility") = 0, py::arg("span") = 0, py::arg("reset") = false);
#endif
}

}  // namespace ops
}  // namespace xattn
