// Host-side refusal probe for csrc/flash_sca/native/umma_prepare.h.
//
// This probe builds one valid fixture, breaks exactly one thing per case, and
// reports the refusal each case produced, so the predicates are measured rather
// than read. It runs the same cases with D=Dv=128 (a mounted bucket) and D=Dv=64
// (an explicitly unmounted bucket); no backward kernel is launched here.
//
// Built twice by prepare_refusal_check.py: once with
// -DXATTN_UMMA_PREPARE_FORCE_BUCKET=1, where later legs are observable for both
// buckets, and once without it. The shipped predicate must reach the same later
// legs for 128/128, but refuse every 64/64 case with kBucket before inspecting
// the deliberately broken operand. Both builds run only on capability (10, 3),
// whose check precedes the bucket check.
//
// The two modules are separate translation units of the same header. This
// two-build trick depends on the preparation staying header-only: putting it
// into a compiled translation unit would give the two builds one symbol table
// and the probe would have to become two processes.
#include <torch/extension.h>

#include <ATen/ATen.h>

#include <map>
#include <string>

#include "flash_sca/native/umma_prepare.h"

// umma_prepare.h includes the mounted-entry registry and its shared inline
// dimension_bucket.h helpers; this probe must not redefine those helpers.

namespace {

using xattn::ops::flash_sca_native::Element;
using xattn::ops::flash_sca_native::Params;
namespace bwd_host = xattn::ops::flash_sca_native::umma::bwd_host;

// B=1, Nq=Nk=128, Hq=4, Hkv=2: two 64-row query and key tiles.
// D=Dv is supplied per run: mounted 128/128 or unmounted 64/64.
constexpr int64_t kB = 1, kNq = 128, kNk = 128, kHq = 4, kHkv = 2;
constexpr int64_t kP = 64;  // previous chunk: unaligned to the 128-row key map

struct Fixture {
  at::Tensor q, k, v, dy, saved, lse, delta, dq, dk, dv, dq_accum;
  at::Tensor prev_k, prev_v, dprev_k, dprev_v;
  Params p;
};

at::Tensor BF16(std::initializer_list<int64_t> shape) {
  return at::zeros(shape, at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA));
}

at::Tensor FP32(std::initializer_list<int64_t> shape) {
  return at::zeros(shape, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
}

// with_prev=false: total == current keys, no previous plane.
// with_prev=true:  a 64-row previous chunk, so the key axis is packed into one
//                  [B, 192, Hkv, *] plane by the preparation.
Fixture MakeFixture(bool with_prev, int64_t dimension) {
  Fixture f;
  const int64_t kD = dimension, kDv = dimension;
  const int64_t total = with_prev ? kNk + kP : kNk;
  f.q = BF16({kB, kNq, kHq, kD});
  f.k = BF16({kB, kNk, kHkv, kD});
  f.v = BF16({kB, kNk, kHkv, kDv});
  f.dy = BF16({kB, kNq, kHq, kDv});
  f.saved = BF16({kB, kNq, kHq, kDv});
  f.lse = FP32({kB, kHq, kNq});
  f.delta = FP32({kB, kHq, kNq});
  f.dq = BF16({kB, kNq, kHq, kD});
  f.dk = BF16({kB, kNk, kHkv, kD});
  f.dv = BF16({kB, kNk, kHkv, kDv});
  f.dq_accum = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  if (with_prev) {
    f.prev_k = BF16({kB, kP, kHkv, kD});
    f.prev_v = BF16({kB, kP, kHkv, kDv});
    f.dprev_k = BF16({kB, kP, kHkv, kD});
    f.dprev_v = BF16({kB, kP, kHkv, kDv});
  }
  f.p.batch = kB;
  f.p.q_length = kNq;
  f.p.k_length = kNk;
  f.p.total_length = total;
  f.p.prev_length = with_prev ? kP : 0;
  f.p.query_heads = kHq;
  f.p.kv_heads = kHkv;
  f.p.qk_dim = kD;
  f.p.value_dim = kDv;
  f.p.element = Element::kBFloat16;
  return f;
}

std::string Run(Fixture& f) {
  const at::Tensor* prev_k = f.prev_k.defined() ? &f.prev_k : nullptr;
  const at::Tensor* prev_v = f.prev_v.defined() ? &f.prev_v : nullptr;
  const at::Tensor* dprev_k = f.dprev_k.defined() ? &f.dprev_k : nullptr;
  const at::Tensor* dprev_v = f.dprev_v.defined() ? &f.dprev_v : nullptr;
  auto out = bwd_host::PrepareBackwardUmma(f.q, f.k, f.v, prev_k, prev_v, f.dy, f.saved, f.lse,
                                          f.delta, f.dq, f.dk, f.dv, dprev_k, dprev_v,
                                          f.dq_accum, /*deterministic=*/false, f.p);
  return std::string(bwd_host::RefusalName(out.refusal));
}

}  // namespace

// Case name -> refusal observed. One valid fixture per case, one thing broken.
std::map<std::string, std::string> RunChecks(int64_t dimension) {
  TORCH_CHECK(dimension == 128 || dimension == 64,
              "refusal probe covers only square 128 and 64 buckets");
  const int64_t kD = dimension, kDv = dimension;
  std::map<std::string, std::string> r;
  {  // accepted when the bucket is mounted or the bucket check is forced off
    auto f = MakeFixture(false, dimension);
    r["valid"] = Run(f);
  }
  {  // unaligned previous chunk: the packed plane is built, still accepted
    auto f = MakeFixture(true, dimension);
    r["valid_packed_prev"] = Run(f);
  }
  {  // grid-order instruments belong to the mma.sync body
    auto f = MakeFixture(false, dimension);
    f.p.sorted_order = true;
    r["sorted_order"] = Run(f);
  }
  {
    auto f = MakeFixture(false, dimension);
    f.p.head_fastest = true;
    r["head_fastest"] = Run(f);
  }
  {  // LSE must be FP32 as documented, not merely resolvable
    auto f = MakeFixture(false, dimension);
    f.lse = f.lse.to(at::kBFloat16);
    r["lse_dtype"] = Run(f);
  }
  {  // LSE must be [B, Hq, Nq], not any contiguous tensor of the right numel
    auto f = MakeFixture(false, dimension);
    f.lse = FP32({kB * kHq, kNq});
    r["lse_shape"] = Run(f);
  }
  {  // delta is read with vector loads on dense rows
    auto f = MakeFixture(false, dimension);
    f.delta = FP32({kB, kHq, 2 * kNq}).narrow(2, 0, kNq);
    r["delta_not_contiguous"] = Run(f);
  }
  {  // the saved matrix is [B, Nq, Hq, Dv]
    auto f = MakeFixture(false, dimension);
    f.saved = BF16({kB, kNq, kHq, kDv / 2});
    r["saved_shape"] = Run(f);
  }
  {  // dy is a TMA source encoded from Q's element type, so its dtype matters
    auto f = MakeFixture(false, dimension);
    f.dy = f.dy.to(at::kFloat);
    r["dy_dtype"] = Run(f);
  }
  {
    auto f = MakeFixture(false, dimension);
    f.dy = BF16({kB, kNq, kHq, kDv / 2});
    r["dy_shape"] = Run(f);
  }
  {  // a present row table must have the row-tile count the leaf indexes with
    auto f = MakeFixture(false, dimension);
    f.p.tile_bounds = reinterpret_cast<int32_t*>(f.dq.data_ptr());
    f.p.tile_count = 5;  // Nq = 128 -> 2
    r["tile_count"] = Run(f);
  }
  {  // same for the key-tile table
    auto f = MakeFixture(false, dimension);
    f.p.key_tile_range = reinterpret_cast<int32_t*>(f.dq.data_ptr());
    f.p.key_tile_count = 1;  // Nk = 128 -> 2
    r["key_tile_count"] = Run(f);
  }
  {  // dQ is written whole: wrong extent is an out-of-bounds write
    auto f = MakeFixture(false, dimension);
    f.dq = BF16({kB, kNq, kHq, kD / 2});
    r["dq_shape"] = Run(f);
  }
  {  // a strided dK is not the dense plane the epilogue stores
    auto f = MakeFixture(false, dimension);
    // Narrowing a size-1 leading dim leaves the tensor contiguous (a size-1
    // dim's stride is not part of the contiguity test), so this case was
    // vacuous until the slice was taken on the head dim, where the stride
    // survives. The assertion keeps it from going vacuous again.
    f.dk = BF16({kB, kNk, kHkv, 2 * kD}).narrow(3, 0, kD);
    TORCH_CHECK(!f.dk.is_contiguous(), "dk_not_contiguous fixture is contiguous");
    r["dk_not_contiguous"] = Run(f);
  }
  {  // dy is a TMA source too: a stride the map cannot describe is refused
    auto f = MakeFixture(false, dimension);
    f.dy = BF16({kB, kNq, kHq, kDv + 4}).narrow(3, 0, kDv);
    TORCH_CHECK(!f.dy.is_contiguous(), "dy_unaligned_stride fixture is contiguous");
    // Observed as `state`, not `map_source`: `p.dy` is published to a leaf as a
    // raw dense-row pointer beside the map, so step 4 requires contiguity before
    // the map walk is reached, and the earlier refusal is the one that fires.
    r["dy_unaligned_stride"] = Run(f);
  }
  {  // a footprint-short contiguous previous plane passes every stride mirror
     // (they validate the strides a shape implies) and would then be described
     // as prev_length rows of a shorter tensor
    auto f = MakeFixture(true, dimension);
    f.prev_k = BF16({kB, kP / 2, kHkv, kD});
    TORCH_CHECK(f.prev_k.is_contiguous(), "prev_shape fixture is not contiguous");
    r["prev_shape"] = Run(f);
  }
  {  // the previous planes are read as Q's element type, like every other source
    auto f = MakeFixture(true, dimension);
    f.prev_v = FP32({kB, kP, kHkv, kDv});
    r["prev_dtype"] = Run(f);
  }
  {  // previous-plane gradients come in pairs
    auto f = MakeFixture(true, dimension);
    f.dprev_v = at::Tensor();
    r["dprev_pair"] = Run(f);
  }
  {  // ... and the pair rule is symmetric: missing k is the same refusal
    auto f = MakeFixture(true, dimension);
    f.dprev_k = at::Tensor();
    r["dprev_pair_missing_k"] = Run(f);
  }
  {  // a stride the TMA cannot describe falls back instead of throwing later
    auto f = MakeFixture(false, dimension);
    f.q = BF16({kB, kNq, kHq, kD + 4}).narrow(3, 0, kD);
    r["q_unaligned_stride"] = Run(f);
  }
  {
    auto f = MakeFixture(true, dimension);
    f.prev_k = BF16({kB, kP, kHkv, kD + 4}).narrow(3, 0, kD);
    r["prev_unaligned_stride"] = Run(f);
  }
  {  // an accumulator is folded into dQ, so it must have dQ's extent
    auto f = MakeFixture(false, dimension);
    f.dq_accum = FP32({kB, kNq - 1, kHq, kD});
    r["dq_accum_extent"] = Run(f);
  }
  {  // a HALF previous K/V pair is refused in both directions: the metadata still
     // claims a previous chunk, but `has_prev` is a conjunction downstream, so
     // without this refusal the call would be prepared as one with no previous
     // chunk and the mask would skip keys the caller believes it supplied
    auto f = MakeFixture(true, dimension);
    f.prev_v = at::Tensor();
    TORCH_CHECK(f.prev_k.defined(), "input_prev_missing_v fixture lost its prev_k");
    TORCH_CHECK(f.prev_k.numel() == kB * kP * kHkv * kD,
                "input_prev_missing_v fixture does not carry a full previous plane");
    r["input_prev_missing_v"] = Run(f);
  }
  {  // ... and the other direction, which no read of the fixture can distinguish
    auto f = MakeFixture(true, dimension);
    f.prev_k = at::Tensor();
    TORCH_CHECK(f.prev_v.defined(), "input_prev_missing_k fixture lost its prev_v");
    r["input_prev_missing_k"] = Run(f);
  }
  {  // previous planes present, no previous-plane gradients requested: accepted,
     // and deliberately so. The pair rule above is about a HALF pair of outputs; a
     // caller that wants dK/dV for the current chunk only hands in no dPrev at all,
     // and the generic body serves it. Measured, not asserted as a refusal.
    auto f = MakeFixture(true, dimension);
    f.dprev_k = at::Tensor();
    f.dprev_v = at::Tensor();
    r["dprev_both_null"] = Run(f);
  }
  return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_checks", &RunChecks, pybind11::arg("dimension") = 128,
        "refusal observed per broken-fixture case in the selected square bucket");
}
