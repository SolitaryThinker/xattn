// Host boundary for the native full-causal SoftDelta reader.
//
// ReaderFwd/ReaderBwd validate, normalize the scale exactly, pack strided
// inputs, concatenate previous and current K/V, and restore the public
// dtype/shape/chain-rule semantics. ReaderFwdCuda/ReaderBwdCuda are the
// contiguous-operand adapters that build the POD and call the CUDA leaf.
//
// Every rule in the validation block below is host-side and runs before any
// device pointer is formed, and every rule below is a rule of the public API,
// not a preference of this file.

#include "attention/native_cuda/reader_launch.h"
#include "attention/native_cuda/reader_params.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "attention/partition/segment_metadata.h"

namespace xattn {
namespace ops {
namespace attention {
namespace native_cuda {
namespace {

constexpr int64_t kMinLevel = 0;
constexpr int64_t kMaxLevel = 4;
constexpr uintptr_t kRequiredAlignment = 16;

struct CoreDims {
  int64_t batch;
  int64_t nq;
  int64_t nk;
  int64_t nprev;
  int64_t ntotal;
  int64_t hq;
  int64_t hkv;
  int64_t d;
  int64_t dv;
};

void CheckInputDtype(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      "native reader ", name, " must be fp16 or bf16, got ",
      tensor.scalar_type());
}

// V1-V4: Q/K/V, previous K/V and segment metadata, all on one device.
CoreDims CheckReaderCore(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx) {
  TORCH_CHECK(
      q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
      "native reader Q, K, V must be rank four [B,N,H,D]");
  TORCH_CHECK(
      q.device().type() == torch::kCUDA && k.device() == q.device() &&
          v.device() == q.device(),
      "native reader Q, K, V must be same-device CUDA tensors");
  CheckInputDtype(q, "Q");
  TORCH_CHECK(
      k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
      "native reader Q, K, V must share one dtype");

  CoreDims dims;
  dims.batch = q.size(0);
  dims.nq = q.size(1);
  dims.nk = k.size(1);
  dims.nprev = 0;
  dims.hq = q.size(2);
  dims.hkv = k.size(2);
  dims.d = q.size(3);
  dims.dv = v.size(3);

  TORCH_CHECK(
      dims.batch == k.size(0) && dims.batch == v.size(0),
      "native reader Q, K, V batch sizes must match");
  TORCH_CHECK(dims.nq > 0, "native reader current query length must be positive");
  TORCH_CHECK(dims.nk >= 0, "native reader key length must be nonnegative");
  TORCH_CHECK(dims.d > 0 && dims.dv > 0,
              "native reader head dims must be positive");
  TORCH_CHECK(dims.hq > 0 && dims.hkv > 0,
              "native reader head counts must be positive");
  TORCH_CHECK(
      dims.hq % dims.hkv == 0,
      "native reader Q head count must be a multiple of the KV head count");
  TORCH_CHECK(dims.d == k.size(3),
              "native reader Q and K head dims must match");
  // V shares K's key axis and KV head axis and differs only in value width,
  // so a short or long V is an out-of-bounds read for a leaf that indexes
  // v[..., key, head] against K's key length and head count, not a wrong
  // number. Both axes are therefore checked here, before any pointer exists.
  TORCH_CHECK(
      v.size(1) == dims.nk,
      "native reader V key length must match K: K has ", dims.nk,
      " keys, V has ", v.size(1));
  TORCH_CHECK(
      v.size(2) == dims.hkv,
      "native reader V KV head count must match K: K has ", dims.hkv,
      " KV heads, V has ", v.size(2));

  TORCH_CHECK(
      prev_k.has_value() == prev_v.has_value(),
      "native reader previous K and V must both be provided or both be None");
  if (prev_k.has_value()) {
    const torch::Tensor& pk = prev_k.value();
    const torch::Tensor& pv = prev_v.value();
    TORCH_CHECK(
        pk.dim() == 4 && pv.dim() == 4,
        "native reader previous K/V must be rank four");
    TORCH_CHECK(
        pk.device() == q.device() && pv.device() == q.device(),
        "native reader previous K/V must be on the Q device");
    TORCH_CHECK(
        pk.scalar_type() == q.scalar_type() &&
            pv.scalar_type() == q.scalar_type(),
        "native reader previous K/V must share the input dtype");
    TORCH_CHECK(
        pk.size(0) == dims.batch && pv.size(0) == dims.batch &&
            pk.size(1) == pv.size(1) && pk.size(1) > 0 &&
            pk.size(2) == dims.hkv && pv.size(2) == dims.hkv &&
            pk.size(3) == dims.d && pv.size(3) == dims.dv,
        "native reader previous K/V shapes are incompatible with Q/K/V");
    dims.nprev = pk.size(1);
  }
  dims.ntotal = dims.nprev + dims.nk;

  partition::CheckSegmentMetadata(
      q_segment_idx, k_segment_idx, dims.batch, dims.nq, dims.ntotal,
      q.device(), "native reader");
  return dims;
}

// V8: the level selects an algorithm; the state dtype is not a level.
void CheckLevel(int64_t level, bool backward) {
  TORCH_CHECK(
      level >= kMinLevel && level <= kMaxLevel,
      "native reader high_precision_level must be in [0, 4], got ", level);
  if (backward) {
    TORCH_CHECK(
        level <= 1,
        "native reader basic backward supports high_precision_level 0 and 1 "
        "only; level ", level,
        " must run through the precision orchestration, not this entry");
  }
}

// Levels 2..4 exist only when the native precision path is compiled into this
// extension, and _native_attention_level_mask reports exactly that. A direct
// call must not be able to reach a leaf at a level this build cannot serve, so
// the same fact is enforced at the entry point rather than only advertised.
void CheckCompiledLevel(int64_t level) {
#if !defined(XATTN_HAS_NATIVE_PRECISION)
  TORCH_CHECK(
      level <= 1,
      "native reader high_precision_level ", level,
      " has no compiled native precision path in this build: levels 2..4 "
      "require XATTN_HAS_NATIVE_PRECISION, which is the same condition "
      "_native_attention_level_mask reports for a tensor's device");
#endif
}

// V10: any finite scale is accepted, including zero and negative values.
void CheckScale(double scale) {
  TORCH_CHECK(
      std::isfinite(scale), "native reader scale must be a finite value, got ",
      scale);
}

// V5/V6/V7: saved state, upstream gradient and LSE sizes and widths.
void CheckSavedState(
    const torch::Tensor& state, const torch::Tensor& q, const CoreDims& dims,
    int64_t level, const char* name) {
  TORCH_CHECK(
      state.device() == q.device(), "native reader ", name,
      " must be on the Q device");
  const at::ScalarType expected =
      level == 0 ? q.scalar_type() : at::kFloat;
  TORCH_CHECK(
      state.scalar_type() == expected, "native reader ", name,
      " must be ", expected, " at high_precision_level ", level, ", got ",
      state.scalar_type(),
      "; the level must be passed explicitly and must match the saved state");
  const std::vector<int64_t> want = {dims.batch, dims.nq, dims.hq, dims.dv};
  TORCH_CHECK(
      state.sizes() == at::IntArrayRef(want), "native reader ", name,
      " must be [B, Nq, Hq, Dv] = [", want[0], ", ", want[1], ", ", want[2],
      ", ", want[3], "]");
}

void CheckLse(
    const torch::Tensor& lse, const torch::Tensor& q, const CoreDims& dims) {
  TORCH_CHECK(
      lse.device() == q.device(), "native reader LSE must be on the Q device");
  TORCH_CHECK(
      lse.scalar_type() == at::kFloat, "native reader LSE must be float32");
  const std::vector<int64_t> want = {dims.batch, dims.hq, dims.nq};
  TORCH_CHECK(
      lse.sizes() == at::IntArrayRef(want),
      "native reader LSE must be [B, Hq, Nq]");
}

// V9: the output state is written through a raw pointer, so it is checked
// completely before the leaf sees it.
void CheckOutputState(
    const c10::optional<torch::Tensor>& output_state, const torch::Tensor& q,
    const CoreDims& dims, int64_t level) {
  if (level == 0) {
    TORCH_CHECK(
        !output_state.has_value(),
        "native reader high_precision_level 0 does not accept an FP32 output "
        "state; pass the same level to forward and backward");
    return;
  }
  TORCH_CHECK(
      output_state.has_value(),
      "native reader high_precision_level ", level,
      " requires a float32 output_state [B, Nq, Hq, Dv]");
  const torch::Tensor& state = output_state.value();
  // The state type is decided by the explicit level, so a wrong state type is
  // reported as a level/state mismatch rather than as a generic dtype error: it
  // is the mistake a caller actually makes when forward and backward disagree.
  TORCH_CHECK(
      state.scalar_type() == at::kFloat,
      "native reader output_state must be float32 at high_precision_level ",
      level, ", got ", state.scalar_type(),
      "; the level is explicit and must match the saved state");
  TORCH_CHECK(
      state.device() == q.device() && state.is_contiguous(),
      "native reader output_state must be contiguous and on the Q device");
  const std::vector<int64_t> want = {dims.batch, dims.nq, dims.hq, dims.dv};
  TORCH_CHECK(
      state.sizes() == at::IntArrayRef(want),
      "native reader output_state must be [B, Nq, Hq, Dv]");
}

bool IsAligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kRequiredAlignment ==
         0;
}

// V13/V16: pack instead of rejecting, then guarantee the leaf's alignment.
torch::Tensor PackAligned(const torch::Tensor& tensor, const char* name) {
  torch::Tensor packed = tensor.is_contiguous() ? tensor : tensor.contiguous();
  if (IsAligned(packed)) {
    return packed;
  }
  // A contiguous tensor can still start at an odd storage offset; the caching
  // allocator returns an aligned block, so a fresh copy restores alignment.
  torch::Tensor aligned = at::empty_like(packed);
  aligned.copy_(packed);
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(aligned.data_ptr()) % kRequiredAlignment == 0,
      "native reader could not produce a 16-byte aligned copy of ", name);
  return aligned;
}

struct ScalePlan {
  double effective;
  bool negate_q;
  bool zero_q;
};

// The leaf consumes a positive finite scale. Negating Q and flipping the sign
// back on dQ is exact for finite fp16/bf16 inputs; zeroing Q is exact too.
//
// The finite/sign/zero decision and the value handed on are both made in double
// and stay double in the POD. Narrowing to float here would turn a finite tiny
// nonzero scale into zero, and a large finite scale into infinity, silently
// changing which branch of the public API a call takes. Cases where the float
// cast would not be representable are left visible here rather than absorbed.
ScalePlan PlanScale(double scale) {
  ScalePlan plan{scale, false, false};
  if (scale < 0.0) {
    plan.effective = -scale;
    plan.negate_q = true;
  } else if (scale == 0.0) {
    plan.effective = 1.0;
    plan.zero_q = true;
  }
  return plan;
}

// An explicit zero fill, not tensor * 0: NaN * 0 is NaN.
torch::Tensor ZeroedQ(const torch::Tensor& q) {
  torch::Tensor zeros = at::zeros_like(q);
  return zeros;
}

torch::Tensor CatPrevious(
    const c10::optional<torch::Tensor>& prev, const torch::Tensor& current) {
  if (!prev.has_value()) {
    return current;
  }
  return at::cat({prev.value(), current}, /*dim=*/1);
}

c10::optional<torch::Tensor> PackOptional(
    const c10::optional<torch::Tensor>& tensor, const char* name) {
  if (!tensor.has_value()) {
    return c10::nullopt;
  }
  return c10::make_optional(PackAligned(tensor.value(), name));
}

int64_t SliceLength(int64_t length, int64_t offset) {
  TORCH_CHECK(
      offset >= 0 && offset <= length, "native reader slice offset out of range");
  return length - offset;
}

torch::Tensor EmptyFwdOutput(const torch::Tensor& q, const CoreDims& dims) {
  std::vector<int64_t> shape = {dims.batch, dims.nq, dims.hq, dims.dv};
  return at::empty(shape, q.options());
}

torch::Tensor EmptyLse(const torch::Tensor& q, const CoreDims& dims) {
  std::vector<int64_t> shape = {dims.batch, dims.hq, dims.nq};
  // Empty rows carry -infinity, matching the production epilogue. A batch of
  // zero rows has no rows at all; the shape is all that is observed.
  return at::full(
      shape, -std::numeric_limits<double>::infinity(),
      q.options().dtype(at::kFloat));
}

}  // namespace

// The ABI declares no head-dimension bound, and this boundary does not invent
// one: bucket coverage is a build-time property of the forward and backward
// leaves (a compiled set of (D, Dv) instantiations, plus the padding the final
// default covers), so a whitelist here would reject shapes a padded leaf serves
// and would drift from the leaves it is meant to guard. What is guaranteed
// instead is that every dimension reaching a pointer is positive and internally
// consistent, and that an unsupported bucket is refused by the leaf entry
// itself, which is the only code that knows what was compiled.
ForwardResult ReaderFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool strict_past,
    int64_t high_precision_level,
    const c10::optional<torch::Tensor>& output_state) {
  const CoreDims dims = CheckReaderCore(
      q, k, v, prev_k, prev_v, q_segment_idx, k_segment_idx);
  CheckScale(scale);
  CheckLevel(high_precision_level, /*backward=*/false);
  CheckCompiledLevel(high_precision_level);
  CheckOutputState(output_state, q, dims, high_precision_level);

  // V12: no launch for an empty batch; the shapes are still correct.
  if (dims.batch == 0) {
    if (output_state.has_value()) {
      output_state.value().zero_();
    }
    return ForwardResult{EmptyFwdOutput(q, dims), EmptyLse(q, dims)};
  }

  c10::cuda::CUDAGuard device_guard(q.device());
  const ScalePlan plan = PlanScale(scale);
  const torch::Tensor q_source =
      plan.negate_q ? at::neg(q) : (plan.zero_q ? ZeroedQ(q) : q);
  const torch::Tensor q_packed = PackAligned(q_source, "Q");
  const torch::Tensor k_total = PackAligned(CatPrevious(prev_k, k), "K");
  const torch::Tensor v_total = PackAligned(CatPrevious(prev_v, v), "V");

  // Every operand the leaf receives is 16-byte aligned. The state is an output,
  // so it cannot be swapped for a private copy without losing the write: when a
  // contiguous state starts at a storage offset that breaks that alignment, the
  // leaf writes into an aligned temporary and the bytes are copied back on the
  // same stream. That adds no caller-visible contiguity rule and no
  // synchronization, and keeps one alignment invariant for every POD pointer.
  c10::optional<torch::Tensor> state_target = output_state;
  c10::optional<torch::Tensor> state_scratch;
  if (output_state.has_value() && !IsAligned(output_state.value())) {
    state_scratch = at::empty_like(output_state.value());
    state_target = state_scratch;
  }

  ForwardResult result = ReaderFwdCuda(
      q_packed, k_total, v_total, plan.effective,
      PackOptional(q_segment_idx, "q_segment_idx"),
      PackOptional(k_segment_idx, "k_segment_idx"), strict_past,
      high_precision_level, state_target);
  if (state_scratch.has_value()) {
    output_state.value().copy_(state_scratch.value(), /*non_blocking=*/true);
  }
  return result;
}

BackwardResult ReaderBwd(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& saved_output,
    const torch::Tensor& lse, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool strict_past,
    int64_t high_precision_level, bool deterministic) {
  const CoreDims dims = CheckReaderCore(
      q, k, v, prev_k, prev_v, q_segment_idx, k_segment_idx);
  CheckScale(scale);
  CheckLevel(high_precision_level, /*backward=*/true);
  CheckSavedState(
      dy, q, dims, /*level=*/0, "dy");  // dy always carries the input dtype
  TORCH_CHECK(
      dy.is_floating_point(), "native reader dy must be a floating tensor");
  CheckSavedState(
      saved_output, q, dims, high_precision_level, "saved_output");
  CheckLse(lse, q, dims);

  const ScalePlan plan = PlanScale(scale);

  if (dims.batch == 0) {
    const std::vector<int64_t> dq_shape = {dims.batch, dims.nq, dims.hq, dims.d};
    const std::vector<int64_t> dk_shape = {dims.batch, dims.nk, dims.hkv, dims.d};
    const std::vector<int64_t> dv_shape = {dims.batch, dims.nk, dims.hkv, dims.dv};
    BackwardResult empty{
        at::zeros(dq_shape, q.options()), at::zeros(dk_shape, q.options()),
        at::zeros(dv_shape, q.options()), c10::nullopt, c10::nullopt};
    if (dims.nprev > 0) {
      const std::vector<int64_t> dpk_shape = {
          dims.batch, dims.nprev, dims.hkv, dims.d};
      const std::vector<int64_t> dpv_shape = {
          dims.batch, dims.nprev, dims.hkv, dims.dv};
      std::get<3>(empty) = at::zeros(dpk_shape, q.options());
      std::get<4>(empty) = at::zeros(dpv_shape, q.options());
    }
    return empty;
  }

  c10::cuda::CUDAGuard device_guard(q.device());
  const torch::Tensor q_source =
      plan.negate_q ? at::neg(q) : (plan.zero_q ? ZeroedQ(q) : q);
  const torch::Tensor q_packed = PackAligned(q_source, "Q");
  const torch::Tensor dy_packed = PackAligned(dy, "dy");
  const torch::Tensor saved_packed = PackAligned(saved_output, "saved_output");
  const torch::Tensor lse_packed = PackAligned(lse, "lse");
  const torch::Tensor k_total = PackAligned(CatPrevious(prev_k, k), "K");
  const torch::Tensor v_total = PackAligned(CatPrevious(prev_v, v), "V");

  CudaBackwardResult total = ReaderBwdCuda(
      dy_packed, q_packed, k_total, v_total, saved_packed, lse_packed,
      plan.effective, PackOptional(q_segment_idx, "q_segment_idx"),
      PackOptional(k_segment_idx, "k_segment_idx"), strict_past,
      high_precision_level, deterministic);

  torch::Tensor dq = std::get<0>(total);
  const torch::Tensor dk_total = std::get<1>(total);
  const torch::Tensor dv_total = std::get<2>(total);

  // The public dQ carries the original-scale chain rule; dK/dV do not, because
  // (-q).k.|s| == q.k.s.
  if (plan.negate_q) {
    dq = at::neg(dq);
  } else if (plan.zero_q) {
    dq = at::zeros_like(dq);
  }

  torch::Tensor dk;
  torch::Tensor dv;
  c10::optional<torch::Tensor> dprev_k;
  c10::optional<torch::Tensor> dprev_v;
  if (dims.nprev > 0) {
    const int64_t current_keys = SliceLength(dims.nk, 0);
    dk = dk_total.narrow(1, dims.nprev, current_keys).contiguous();
    dv = dv_total.narrow(1, dims.nprev, current_keys).contiguous();
    dprev_k = dk_total.narrow(1, 0, dims.nprev).contiguous();
    dprev_v = dv_total.narrow(1, 0, dims.nprev).contiguous();
  } else {
    dk = dk_total;
    dv = dv_total;
  }

  // At original scale zero dQ, dK and dPrevK are zero mathematically; computing
  // them would rely on a finite dS, which an empty row does not guarantee.
  if (plan.zero_q) {
    dk = at::zeros_like(dk);
    if (dprev_k.has_value()) {
      dprev_k = at::zeros_like(dprev_k.value());
    }
  }

  // The gradient merge path consumes dQ, dK and dV as fresh, allocator-aligned
  // buffers, so the invariant is asserted rather than assumed: every branch
  // above either returns a fresh allocation or a fresh copy.
  TORCH_CHECK(
      IsAligned(dq) && IsAligned(dk) && IsAligned(dv),
      "native reader backward produced a gradient buffer without 16-byte "
      "alignment, which the gradient merge path requires");

  return BackwardResult{dq, dk, dv, dprev_k, dprev_v};
}

// ---------------------------------------------------------------------------
// Contiguous-operand adapters: build the platform-neutral POD and call the raw
// CUDA launch functions the leaf translation units define. These
// allocate every output, so the leaf allocates no Torch tensors, and they
// launch on Q's current stream only.
//
// A truthful availability dispatch stays false until these bodies are linked
// against real leaf code; nothing here returns a hardcoded success or no-op.
// ---------------------------------------------------------------------------

namespace {

ReaderScalarType ToReaderScalarType(at::ScalarType type) {
  if (type == at::kHalf) {
    return ReaderScalarType::kHalf;
  }
  if (type == at::kBFloat16) {
    return ReaderScalarType::kBFloat16;
  }
  TORCH_CHECK(
      false, "native reader accepts fp16 and bf16 operands only, got ", type);
}

// The adapters restate the reduced contract instead of trusting the caller: a
// direct call cannot hand the leaf a strided, misaligned or cross-device
// pointer.
void CheckPacked(const torch::Tensor& tensor, const torch::Tensor& q,
                 const char* name) {
  TORCH_CHECK(
      tensor.device() == q.device(), "native reader ", name,
      " must be on the Q device");
  TORCH_CHECK(
      tensor.is_contiguous(), "native reader ", name,
      " must be contiguous at the launch boundary");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kRequiredAlignment == 0,
      "native reader ", name, " must be 16-byte aligned at the launch boundary");
}

const int64_t* SegmentPtr(const c10::optional<torch::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return nullptr;
  }
  TORCH_CHECK(
      tensor->scalar_type() == at::kLong && tensor->is_contiguous(),
      "native reader segment metadata must be contiguous int64 at the launch "
      "boundary");
  return tensor->data_ptr<int64_t>();
}

ReaderInputs MakeInputs(
    const torch::Tensor& q, const torch::Tensor& k_total,
    const torch::Tensor& v_total, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool strict_past,
    int64_t high_precision_level) {
  ReaderInputs inputs{};
  inputs.shape.batch = q.size(0);
  inputs.shape.q_length = q.size(1);
  inputs.shape.k_length = k_total.size(1);
  inputs.shape.query_heads = q.size(2);
  inputs.shape.kv_heads = k_total.size(2);
  inputs.shape.qk_dim = q.size(3);
  inputs.shape.value_dim = v_total.size(3);
  inputs.scalar_type = ToReaderScalarType(q.scalar_type());
  inputs.q = q.data_ptr();
  inputs.k = k_total.data_ptr();
  inputs.v = v_total.data_ptr();
  inputs.q_segment_idx = SegmentPtr(q_segment_idx);
  inputs.k_segment_idx = SegmentPtr(k_segment_idx);
  inputs.scale = scale;
  inputs.strict_past = strict_past;
  inputs.high_precision_level = static_cast<int32_t>(high_precision_level);
  return inputs;
}

void CheckLeafArgs(const torch::Tensor& q, const torch::Tensor& k_total,
                   const torch::Tensor& v_total, double scale) {
  // Everything that can be decided without forming a pointer is decided first;
  // the alignment test is the last thing to look at an address.
  TORCH_CHECK(
      q.dim() == 4 && k_total.dim() == 4 && v_total.dim() == 4,
      "native reader packed operands must be rank four");
  TORCH_CHECK(
      q.size(1) > 0 && k_total.size(1) >= 0 && q.size(3) > 0 &&
          v_total.size(3) > 0,
      "native reader packed dimensions must be positive");
  TORCH_CHECK(
      q.size(0) == k_total.size(0) && q.size(0) == v_total.size(0),
      "native reader packed batch sizes must match");
  TORCH_CHECK(
      v_total.size(1) == k_total.size(1),
      "native reader packed V key length must match K");
  TORCH_CHECK(
      k_total.scalar_type() == q.scalar_type() &&
          v_total.scalar_type() == q.scalar_type(),
      "native reader K/V must share Q's dtype at the launch boundary");
  TORCH_CHECK(
      k_total.size(3) == q.size(3) && v_total.size(2) == k_total.size(2) &&
          k_total.size(2) > 0 && q.size(2) % k_total.size(2) == 0,
      "native reader K/V head shape is inconsistent with Q");
  // The host boundary has already normalized the sign and the zero case; the
  // leaf only ever sees a positive finite scale.
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "native reader leaf requires a positive finite scale, got ", scale);
  CheckPacked(q, q, "Q");
  CheckPacked(k_total, q, "K");
  CheckPacked(v_total, q, "V");
}

}  // namespace

ForwardResult ReaderFwdCuda(
    const torch::Tensor& q, const torch::Tensor& k_total,
    const torch::Tensor& v_total, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool strict_past,
    int64_t high_precision_level,
    const c10::optional<torch::Tensor>& output_state) {
  CheckLeafArgs(q, k_total, v_total, scale);
  TORCH_CHECK(
      high_precision_level >= kMinLevel && high_precision_level <= kMaxLevel,
      "native reader forward accepts high_precision_level 0..4, got ",
      high_precision_level);
  if (high_precision_level >= 1) {
    TORCH_CHECK(
        output_state.has_value(),
        "native reader high_precision_level ", high_precision_level,
        " requires a float32 output_state at the launch boundary");
    CheckPacked(output_state.value(), q, "output_state");
    TORCH_CHECK(
        output_state->scalar_type() == at::kFloat,
        "native reader output_state must be float32");
  } else {
    TORCH_CHECK(
        !output_state.has_value(),
        "native reader high_precision_level 0 must not write an FP32 state");
  }

  c10::cuda::CUDAGuard device_guard(q.device());

  const std::vector<int64_t> output_shape = {
      q.size(0), q.size(1), q.size(2), v_total.size(3)};
  torch::Tensor output = at::empty(output_shape, q.options());
  const std::vector<int64_t> lse_shape = {q.size(0), q.size(2), q.size(1)};
  torch::Tensor lse = at::empty(lse_shape, q.options().dtype(at::kFloat));

  ReaderFwdParams params{};
  params.inputs = MakeInputs(
      q, k_total, v_total, scale, q_segment_idx, k_segment_idx, strict_past,
      high_precision_level);
  params.output = output.data_ptr();
  params.output_state = output_state.has_value()
      ? output_state->data_ptr<float>()
      : nullptr;
  params.lse = lse.data_ptr<float>();

  LaunchReaderFwd(params, c10::cuda::getCurrentCUDAStream(q.device().index()).stream());
  // Turn a failed launch into an exception here rather than letting a caller
  // read uninitialized outputs: the leaf reports nothing back to the host.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ForwardResult{output, lse};
}

CudaBackwardResult ReaderBwdCuda(
    const torch::Tensor& dy, const torch::Tensor& q,
    const torch::Tensor& k_total, const torch::Tensor& v_total,
    const torch::Tensor& saved_output, const torch::Tensor& lse, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool strict_past,
    int64_t high_precision_level, bool deterministic) {
  CheckLeafArgs(q, k_total, v_total, scale);
  TORCH_CHECK(
      high_precision_level == kMinLevel || high_precision_level == 1,
      "native reader basic backward accepts high_precision_level 0 and 1 only, "
      "got ",
      high_precision_level);
  CheckPacked(dy, q, "dy");
  CheckPacked(saved_output, q, "saved_output");
  CheckPacked(lse, q, "lse");
  TORCH_CHECK(dy.scalar_type() == q.scalar_type(), "native reader dy dtype");
  TORCH_CHECK(
      saved_output.scalar_type() ==
          (high_precision_level == 0 ? q.scalar_type() : at::kFloat),
      "native reader saved_output dtype must follow the level");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "native reader lse must be fp32");
  TORCH_CHECK(
      dy.size(3) == v_total.size(3) && saved_output.size(3) == v_total.size(3),
      "native reader dy and saved_output value width must match V");

  c10::cuda::CUDAGuard device_guard(q.device());

  const std::vector<int64_t> dq_shape = {
      q.size(0), q.size(1), q.size(2), q.size(3)};
  const std::vector<int64_t> dk_shape = {
      k_total.size(0), k_total.size(1), k_total.size(2), k_total.size(3)};
  const std::vector<int64_t> dv_shape = {
      v_total.size(0), v_total.size(1), v_total.size(2), v_total.size(3)};
  torch::Tensor dq = at::empty(dq_shape, q.options());
  torch::Tensor dk = at::empty(dk_shape, q.options());
  torch::Tensor dv = at::empty(dv_shape, q.options());
  // Linear row scratch, initialized on this stream before either gradient
  // contraction. The three planes are delta, raw maximum and inverse sum.
  torch::Tensor row_scratch = at::empty(
      {3, q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));
  const int64_t row_count = q.size(0) * q.size(2) * q.size(1);

  ReaderBwdParams params{};
  params.inputs = MakeInputs(
      q, k_total, v_total, scale, q_segment_idx, k_segment_idx, strict_past,
      high_precision_level);
  params.dy = dy.data_ptr();
  params.saved_output = saved_output.data_ptr();
  params.lse = lse.data_ptr<float>();
  params.deterministic = deterministic;
  params.dq = dq.data_ptr();
  params.dk = dk.data_ptr();
  params.dv = dv.data_ptr();
  params.delta = row_scratch.data_ptr<float>();
  params.row_max = params.delta + row_count;
  params.inv_normalizer = params.row_max + row_count;

  LaunchReaderBwd(params, c10::cuda::getCurrentCUDAStream(q.device().index()).stream());
  // Same rule for backward: a bad launch must not look like a zero gradient.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return CudaBackwardResult{dq, dk, dv};
}

}  // namespace native_cuda
}  // namespace attention
}  // namespace ops
}  // namespace xattn
