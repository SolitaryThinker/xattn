// Native Blackwell (sm_103a) FlashSCA: launch parameters shared by the host
// boundary (host.cc) and the CUDA kernels (kernels.cuh).
//
// Visibility model. Key positions j index the total key axis
// [prev_k ; k] of length Nt = P + Nk (P = 0 without previous K/V). Query row i
// admits key j iff lo(b,i) <= j <= hi(b,i) and, when segment planes are present
// and not marked exact for that batch row, k_seg[b,j] == q_seg[b,i].
//   * Analytic bounds (row_lo == nullptr): pos = i + Nt - Nq,
//       lo = max(0, (floor(pos / C) - 1) * C), hi = pos - strict.
//   * Explicit bounds (row_lo/row_hi, int32 [B,Nq]): computed by the Python
//     boundary for segment/BOS/reset metadata. A row with lo > hi is empty.
// Every bound is clamped to [0, Nt) by the kernels. An empty row produces
// y = 0, an FP32 state of 0, LSE = -inf, and contributes nothing to any
// gradient.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "flash_sca/native/dimension_bucket.h"

namespace xattn::ops::flash_sca_native {

enum class Element : int32_t { kHalf, kBFloat16 };

struct Operand {
  const void* ptr = nullptr;
  int64_t batch_stride = 0;
  int64_t row_stride = 0;
  int64_t head_stride = 0;
};

struct Params {
  Element element = Element::kBFloat16;
  // Q [B,Nq,Hq,D]; K/V current [B,Nk,Hkv,*]; previous [B,P,Hkv,*] or absent.
  // Strides are in elements; the last dimension is contiguous and every
  // stride is a multiple of 8 elements, every base 16-byte aligned.
  Operand q, k, v, prev_k, prev_v;

  int64_t batch = 0;
  int64_t q_length = 0;     // Nq
  int64_t k_length = 0;     // Nk (current)
  int64_t prev_length = 0;  // P
  int64_t total_length = 0; // Nt = P + Nk
  int64_t query_heads = 0;
  int64_t kv_heads = 0;
  int64_t group = 1;        // Hq / Hkv
  int64_t qk_dim = 0;       // D
  int64_t value_dim = 0;    // Dv
  int64_t chunk = 1;        // C

  float scale = 1.f;        // positive, finite
  float scale_log2 = 1.f;   // scale * log2(e)
  bool strict = false;      // hi = pos - 1 in analytic mode
  bool subtract_before_scale = false;  // level >= 1 forward softmax form
  // Work-descending order (opt-in, default off): process query tiles in the
  // stable descending order of their key-tile work:
  //   rank(t) = #{t' : w(t') > w(t)} + #{t' < t : w(t') == w(t)},
  // dead tiles (no visible key) last, equal work by ascending tile. One writer
  // per output tile, so any permutation of the map is exact. This is an audit
  // and screening arm, not a claimed win: measured at the in-order layout, the
  // table is a loss on both train shapes (-7.6% / -16.0%) and neutral
  // on prefill; the table's cost is paid per CTA on every wave.
  bool sorted_order = false;
  // Whether the permutation applies on every grid (true) or only when the grid
  // is shallow enough for the per-CTA rank to be cheaper than the tail it can
  // remove (false; kOrderMaxCtas in kernels.cuh). XATTN_SCA_ORDER=sdesc is the
  // gated form, sdesc-all and dump force it on every grid.
  bool order_always = false;
  // Audit aid (opt-in, default off): print the rank -> query tile map of the
  // sorted order once per batch row, so the permutation can be checked on the
  // device (bijection, dead tiles last, ties by ascending tile) instead of by
  // reading the code. Only reachable through XATTN_SCA_ORDER=dump, never in a
  // measured arm.
  bool order_dump = false;
  // Head-fastest layout (opt-in, default off, XATTN_SCA_LAYOUT=hf): enumerate the
  // forward grid head-fastest, grid(Hq, tiles, B). Default (tf) is
  // blockIdx.x = query tile, blockIdx.y = query head, i.e. the base grid. The
  // work of a query tile carries no head term, so one [B,T] order serves both
  // layouts; with the head-fastest grid the Hq copies of tile t launch
  // consecutively, so a rank-ordered x is the LPT order of CTA space.
  bool head_fastest = false;

  // Host-prepared `UmmaInputMaps` for the native forward leaf, or nullptr when
  // the host has not prepared the call. A `const void*` on purpose: `params.h`
  // must not learn the TMA descriptor types, and the pointer is only ever
  // dereferenced by the host-side arm in `launch.cuh`, inside the call that
  // created it. It never reaches a kernel -- the arm nulls the copy it launches
  // with -- because a host address in a kernel argument would be read on the
  // device. Lifetime is the caller's `PreparedUmmaCall`; the field is cleared
  // by the host before that object goes out of scope.
  const void* umma_maps = nullptr;

  // Optional visibility metadata; see the header comment.
  const int32_t* row_lo = nullptr;
  const int32_t* row_hi = nullptr;
  const int64_t* q_segment = nullptr;  // [B,Nq]
  const int64_t* k_segment = nullptr;  // [B,Nt]
  const int32_t* segment_exact = nullptr;  // [B]; nonzero skips the key test

  // Per 64-row query tile bounds [B, ceil(Nq/64), 4] filled by the bounds
  // kernel: [0] = min lo over the entry's live rows, [1] = max hi over its live
  // rows, [2] = max lo over all rows, [3] = min hi over all rows (a dead or
  // out-of-range row counts as lo = INT_MAX, hi = -1 in the last two).
  // ABI pin, no live row in the entry: every column is dead-marked, [0] and [2]
  // = INT_MAX, [1] and [3] = -1. Hence `b[0] <= b[1]` is exactly "this entry
  // has at least one live row" (the launch order's liveness test) and
  // `b[2] <= b[3]` is exactly "every row is live", which is NOT the same
  // predicate: a partial tail entry or a segment boundary inside the tile has
  // live rows and still fails the second.
  int32_t* tile_bounds = nullptr;
  int64_t tile_count = 0;
  // Explicit bounds only: per 64-key tile, the first and last 64-row query
  // tile whose bounds intersect it [B, ceil(Nt/64), 2] (first > last when
  // none does), filled by the key-range kernel after the bounds kernel.
  int32_t* key_tile_range = nullptr;
  int64_t key_tile_count = 0;

  // Forward outputs: y [B,Nq,Hq,Dv] (input dtype), optional FP32 state of the
  // same shape, LSE [B,Hq,Nq] natural log. All contiguous.
  void* y = nullptr;
  float* y_state = nullptr;
  float* lse = nullptr;

  // Backward inputs: dy and the saved output, contiguous [B,Nq,Hq,Dv]; the
  // saved output is in the input dtype or FP32 (saved_fp32). LSE as above.
  const void* dy = nullptr;
  const void* saved = nullptr;
  bool saved_fp32 = false;
  const float* lse_in = nullptr;
  float* delta = nullptr;  // [B,Hq,Nq] scratch
  // When set (determinism not requested), the dK/dV kernels also reduce the
  // unscaled dQ into this zero-initialized FP32 [B,Nq,Hq,D] buffer with
  // unordered atomics and no dQ kernel runs; otherwise the row-owned dQ
  // kernel writes dq in a fixed order.
  float* dq_accum = nullptr;

  // Backward outputs (contiguous, input dtype): dq like Q, dk/dv like the
  // current K/V, dprev_k/dprev_v like the previous K/V.
  void* dq = nullptr;
  void* dk = nullptr;
  void* dv = nullptr;
  void* dprev_k = nullptr;
  void* dprev_v = nullptr;
  // dk/dv/dprev_k/dprev_v are FP32 instead of the input dtype (the private
  // `fp32_kv_grads` option; grouped heads only): every key writer stores the FP32
  // values its 16-bit store would round, so rounding them to the input dtype
  // gives that store's bits. dq keeps the input dtype.
  bool key_grads_fp32 = false;
};

// Which calls the native tcgen05/UMMA forward serves. Two predicates, one rule:
// host.cc's HasUmmaForwardLeaf reads IsUmmaForwardDims on the call's logical dims (whether its TMA
// maps are prepared, and whether XATTN_SCA_UMMA_FWD=1 may be honoured), and launch.cuh's
// kUmmaForwardLeaf reads IsUmmaForwardLeafBucket on the (QkBucket, ValueBucket) pair the dims
// dispatch to (whether that bucket consumes them). Every dim IsUmmaForwardDims admits buckets to a
// pair IsUmmaForwardLeafBucket admits, so a prepared call always reaches a native launcher.
constexpr bool IsUmmaForwardBucketWidth(int64_t w) { return w == 64 || w == 128 || w == 256; }

// Every (D, Dv) bucket pair of 64/128/256 has a leaf launcher
// (instantiations/sca_umma_fwd_d<D>_v<Dv>.cu, ForwardKernel<E, D, Dv>), and that launcher tries
// the pipelined body (instantiations/sca_umma_fwd_pipe_d<D>_v<Dv>.cu) first.
constexpr bool IsUmmaForwardLeafBucket(int64_t d_bucket, int64_t dv_bucket) {
  return IsUmmaForwardBucketWidth(d_bucket) && IsUmmaForwardBucketWidth(dv_bucket);
}

// The logical dims whose calls the native forward serves: each in 8..256 and a multiple of 8 (a
// TMA row of 16-bit elements is a multiple of 16 bytes). A dim that is not a bucket width is
// padded to its bucket (QkBucket / ValueBucket) by the TMA's zero fill of the last 64-column box.
constexpr bool IsUmmaForwardDims(int64_t qk_dim, int64_t value_dim) {
  return qk_dim >= 8 && qk_dim <= 256 && qk_dim % 8 == 0 && value_dim >= 8 && value_dim <= 256 &&
         value_dim % 8 == 0;
}


void LaunchTileBounds(const Params& params, cudaStream_t stream);
void LaunchKeyTileRange(Params& params, cudaStream_t stream);
void LaunchForward(const Params& params, cudaStream_t stream);
void LaunchBackward(const Params& params, cudaStream_t stream);

}  // namespace xattn::ops::flash_sca_native
