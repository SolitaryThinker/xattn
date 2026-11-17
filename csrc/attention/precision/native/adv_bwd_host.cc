// Full-causal native advanced reader backward. Operands and row workspaces
// are validated before allocation or launch. The two public precision entries
// share the same native graph and differ only in statistics accumulation.
#include "attention/precision/native/adv_bwd_native.h"
#include "attention/precision/main.h"
#include "attention/precision/ordered.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <optional>
#include <tuple>

namespace xattn {
namespace ops {
namespace native_adv {

// The public precision-repair backward (high_precision_level=1) is the
// algorithm the leaf implements as its internal level 4: dual-operand P/dS,
// range protection, probability-mass repair. Levels 2 and 3 of the leaf are no
// longer reachable from the public API.
constexpr int kRepairLevel = 4;

namespace {

// Mirrors main.cu / ordered.cu validation one-for-one, then narrows the scope
// to what the native leaf implements.  Anything outside that scope throws; no
// silent fallback and no change to the public checks the Hopper route performs.
Params Plan(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& g, const at::Tensor& delta, const at::Tensor& logz,
            const at::Tensor& stats, const at::Tensor& meta,
            const std::optional<at::Tensor>& qs, const std::optional<at::Tensor>& ks,
            const std::optional<at::Tensor>& qp, double scale, int level,
            int visibility, int span, bool strict, bool reset, bool ordered,
            const std::optional<at::Tensor>& rho_exponent,
            void* dq_ptr, void* dk_ptr, void* dv_ptr) {
  TORCH_CHECK(level >= 2 && level <= 4, "precision mainloop needs level 2, 3, or 4");
  TORCH_CHECK(visibility >= 0 && visibility <= 2, "invalid attention visibility");
  TORCH_CHECK(q.is_cuda() && q.dim() == 4, "Q must be CUDA [B,N,H,D]");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "precision mainloop requires FP16/BF16");
  for (const auto& t : {q, k, v, g}) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.dim() == 4 &&
                    t.is_contiguous() && t.scalar_type() == q.scalar_type(),
                "Q/K/V/dO must be contiguous tensors of one dtype/device");
  }
  TORCH_CHECK(q.size(0) == k.size(0) && k.sizes().slice(0, 3) == v.sizes().slice(0, 3) &&
                  q.sizes().slice(0, 3) == g.sizes().slice(0, 3) && q.size(3) == k.size(3) &&
                  g.size(3) == v.size(3),
              "incompatible precision operands");
  TORCH_CHECK(k.size(2) > 0 && q.size(2) > 0 && q.size(2) % k.size(2) == 0, "invalid GQA mapping");
  TORCH_CHECK(q.size(3) > 0 && q.size(3) <= 256 && v.size(3) > 0 && v.size(3) <= 256 &&
                  q.size(3) % 8 == 0 && v.size(3) % 8 == 0,
              "precision mainloop requires D/V multiples of 8 up to 256");
  TORCH_CHECK(std::isfinite(scale) && scale > 0., "precision mainloop requires a positive finite scale");
  TORCH_CHECK(visibility == 0 || (visibility == 1 ? span >= 0 : span > 0), "invalid local attention span");
  const int64_t count = q.size(0) * q.size(2) * q.size(1);
  for (const auto& t : {delta, logz, stats, meta}) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.is_contiguous(),
                "precision workspace must be contiguous on Q's device");
  }
  TORCH_CHECK(delta.scalar_type() == at::kFloat && logz.scalar_type() == at::kFloat &&
                  meta.scalar_type() == at::kFloat && stats.scalar_type() == at::kDouble,
              "invalid precision workspace dtype");
  TORCH_CHECK(delta.numel() == count && logz.numel() == count && meta.numel() == 6 * count &&
                  stats.numel() == 4 * count,
              "invalid precision workspace shape");
  // Same gate as the public ordered entry: levels 3/4 accumulate int64 per-row
  // grids and need the per-row exponent; level 2 has no fixed-point plane.
  TORCH_CHECK(level == 2 || !ordered || (rho_exponent && rho_exponent->is_cuda() &&
                             rho_exponent->device() == q.device() &&
                             rho_exponent->is_contiguous() &&
                             rho_exponent->scalar_type() == at::kInt &&
                             rho_exponent->numel() == count),
              "ordered levels 3/4 require an int32 exponent per row");
  TORCH_CHECK(qs.has_value() == ks.has_value(), "both segment arrays are required");
  if (qs) {
    for (int side = 0; side < 2; ++side) {
      const auto& t = side ? *ks : *qs;
      const int64_t n = side ? k.size(1) : q.size(1);
      TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.is_contiguous() &&
                      t.scalar_type() == at::kLong && t.dim() == 2 && t.size(0) == q.size(0) &&
                      t.size(1) == n,
                  "invalid mainloop segments");
    }
  }
  if (qp) {
    TORCH_CHECK(qp->is_cuda() && qp->device() == q.device() && qp->is_contiguous() &&
                    qp->scalar_type() == at::kInt && qp->numel() == q.size(0) * q.size(1),
                "invalid reset query positions");
  }
  TORCH_CHECK(!reset || visibility == 2, "reset positions only apply to sliding chunks");

  // ---- scope of the native leaf -------------------------------------------------
  TORCH_CHECK(visibility == 0 || visibility == 2,
              "native advanced reader backward implements full-causal (0) and sliding-chunk (2) "
              "visibility; window (1) is an explicit follow-up, not a silent route change");
  // Reset query positions need no plane of their own: at visibility 2 the leaf
  // admits exactly the keys between the row statistics' first and last
  // admitted key, which the reset bounds already determined.

  Params p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.dy = g.data_ptr();
  p.delta = delta.data_ptr<float>();
  p.logz = logz.data_ptr<float>();
  p.meta = meta.data_ptr<float>();
  p.stats = reinterpret_cast<double*>(stats.data_ptr());
  p.dq = dq_ptr;
  p.dk = dk_ptr;
  p.dv = dv_ptr;
  p.q_seg = qs ? qs->data_ptr<int64_t>() : nullptr;
  p.k_seg = ks ? ks->data_ptr<int64_t>() : nullptr;
  p.rho_exponent = rho_exponent ? rho_exponent->data_ptr<int32_t>() : nullptr;

  p.q_batch_stride = q.stride(0); p.q_row_stride = q.stride(1); p.q_head_stride = q.stride(2);
  p.k_batch_stride = k.stride(0); p.k_row_stride = k.stride(1); p.k_head_stride = k.stride(2);
  p.v_batch_stride = v.stride(0); p.v_row_stride = v.stride(1); p.v_head_stride = v.stride(2);
  p.dy_batch_stride = g.stride(0); p.dy_row_stride = g.stride(1); p.dy_head_stride = g.stride(2);
  p.dq_batch_stride = q.size(1) * q.size(2) * q.size(3);
  p.dq_row_stride = q.size(2) * q.size(3);
  p.dq_head_stride = q.size(3);
  p.dk_batch_stride = k.size(1) * k.size(2) * k.size(3);
  p.dk_row_stride = k.size(2) * k.size(3);
  p.dk_head_stride = k.size(3);
  p.dv_batch_stride = v.size(1) * v.size(2) * v.size(3);
  p.dv_row_stride = v.size(2) * v.size(3);
  p.dv_head_stride = v.size(3);
  p.rowstat_head_stride = q.size(1);  // planes are [B, Hq, Nq]: row stride 1

  p.B = q.size(0);
  p.Nq = q.size(1);
  p.Nk = k.size(1);
  p.Nt = k.size(1);              // K arrives with previous K/V already concatenated
  p.Hq = q.size(2);
  p.Hkv = k.size(2);
  p.D = q.size(3);
  p.Dv = v.size(3);
  p.offset = k.size(1) - q.size(1);  // right alignment
  p.qk_ratio = q.size(2) / k.size(2);
  p.visibility = visibility;
  p.span = span;
  p.reset = reset;
  p.level = level;
  // Carry the natural scale for the final dQ/dK factor and the float log2
  // scale for the shifted exponent, as separate quantities.
  p.softmax_scale = scale;
  p.softmax_scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  p.strict = strict;
  // Level 4 always splits P into two element components; levels 2/3 need it
  // exactly when the query is shorter than the key axis (right-aligned short Q).
  p.two_component_dv = (level == 4) || (k.size(1) > q.size(1));
  p.ordered = ordered;
  p.elem = (q.scalar_type() == at::kBFloat16) ? Elem::kBf16 : Elem::kFp16;
  return p;
}

// Shared tail: validate, allocate exact-width FP32 outputs, launch, return.
std::tuple<at::Tensor, at::Tensor, at::Tensor> Run(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
    const at::Tensor& delta, const at::Tensor& logz, const at::Tensor& stats,
    const at::Tensor& meta, const std::optional<at::Tensor>& qs,
    const std::optional<at::Tensor>& ks, const std::optional<at::Tensor>& qp,
    double scale, int level, int visibility, int span, bool strict, bool reset, bool ordered,
    const std::optional<at::Tensor>& rho_exponent) {
  Params p = Plan(q, k, v, g, delta, logz, stats, meta, qs, ks, qp, scale, level,
                  visibility, span, strict, reset, ordered, rho_exponent,
                  nullptr, nullptr, nullptr);
  c10::cuda::CUDAGuard guard(q.device());
  auto dq = at::empty(q.sizes(), q.options().dtype(at::kFloat));
  auto dk = at::empty(k.sizes(), q.options().dtype(at::kFloat));
  auto dv = at::empty(v.sizes(), q.options().dtype(at::kFloat));
  p.dq = dq.data_ptr();
  p.dk = dk.data_ptr();
  p.dv = dv.data_ptr();
  if (!q.numel()) {
    return {at::zeros(q.sizes(), q.options().dtype(at::kFloat)),
            at::zeros(k.sizes(), q.options().dtype(at::kFloat)),
            at::zeros(v.sizes(), q.options().dtype(at::kFloat))};
  }
  Launch(p, at::cuda::getCurrentCUDAStream(q.get_device()));
  return {dq, dk, dv};
}

}  // namespace

// Both canonical declarations are included above so binding/linkage cannot
// silently advertise a different ABI from these definitions.
}  // namespace native_adv
}  // namespace ops
}  // namespace xattn

namespace xattn::ops::attention::precision {
std::tuple<at::Tensor, at::Tensor, at::Tensor> MainBackward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
    const at::Tensor& delta, const at::Tensor& logz, const at::Tensor& stats,
    const at::Tensor& meta, const std::optional<at::Tensor>& qs,
    const std::optional<at::Tensor>& ks, const std::optional<at::Tensor>& qp,
    double scale, int visibility, int span, bool strict, bool reset) {
  return native_adv::Run(q, k, v, g, delta, logz, stats, meta, qs, ks, qp, scale,
                         native_adv::kRepairLevel, visibility, span, strict, reset, false,
                         std::nullopt);
}

namespace ordered {
std::tuple<at::Tensor, at::Tensor, at::Tensor> MainBackward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
    const at::Tensor& delta, const at::Tensor& logz, const at::Tensor& stats,
    const at::Tensor& meta, const std::optional<at::Tensor>& qs,
    const std::optional<at::Tensor>& ks, const std::optional<at::Tensor>& qp,
    double scale, int visibility, int span, bool strict, bool reset,
    const at::Tensor& rho_exponent) {
  return native_adv::Run(q, k, v, g, delta, logz, stats, meta, qs, ks, qp, scale,
                         native_adv::kRepairLevel, visibility, span, strict, reset, true,
                         rho_exponent);
}
}  // namespace ordered
}  // namespace xattn::ops::attention::precision
