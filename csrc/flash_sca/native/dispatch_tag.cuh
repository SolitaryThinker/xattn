// Per-phase host dispatch tags for Blackwell FlashSCA.
//
// A row records a selected producer whose launch returned without an immediate
// launch error. It does not prove device completion or execution of particular
// instructions. Join it with the actual kernel symbol, same-binary cubin/SASS
// and device evidence; an unrelated UMMA symbol in the binary proves nothing
// about the producer named by a row.
//
// Scope: this host path covers level-0/1 backward and level-0..4 forward.
// Level-2..4 backward runs the separate precision backend and records there,
// not through the level-0/1 producers described below: its one launcher reports
// its three contractions as [kDq][kScalar][kScalar][kRan], [kDk][...] and
// [kDv][...], one row each after the corresponding launch returned, because the
// advanced leaves carry no matrix instruction (see Body::kScalar). Those three
// rows are the only witness that the advanced reader ran at all; they do not
// make any level-2..4 product a tensor-core body. These rows belong to a
// launcher whose leaves are scalar. A launcher that calls armed UMMA entries
// already gets one row per (phase, decision) from those entries, as
// (umma, umma, ran) or (umma, scalar, refused), so adding these kScalar rows to
// it would double-record the same decisions and stamp scalar on phases a UMMA
// leaf carried.
// The forward tag names its actual body, not the directory in which
// that implementation lives.
//
// Each backward product has one selected writer. Eligibility queries emit no
// producer rows: uncompiled/refused candidates are described by the per-call
// preparation snapshot. An eligible native dQ or key leaf records kUmma only
// after its launcher returns true; a final refusal raises without replaying a
// generic writer. Generic products record kMmaSync. Delta records kScalar.
//
// The forward's `selected` is the XATTN_SCA_UMMA_FWD decision (host.cc): "1"
// requires a native leaf and selects kUmma on every bucket this build compiles
// one for; unset and "0" select kMmaSync, so a forward `selected` is
// information and must be read as such. Unset and "0" reach the same predicate,
// the same generic body and the SAME ROW: this table is indexed by
// (phase, selected, launched, state) and carries no arm field, so both record
// [kForward][kMmaSync][kMmaSync][kRan] and no retained artifact can tell
// "whatever main does" from "forced off". Telling them apart would need an arm
// field here; it cannot be read off the tag.
// One selection precedes that decision: a level-0 call at (256,256) or (128,128) that the
// pipelined body accepts selects kUmma when XATTN_SCA_FWD_PIPE is unset, empty or "1" and
// XATTN_SCA_UMMA_FWD is unset or empty. The env precedence is a contract
// (host.cc ResolveForwardBody, resolved once per call): both unset => the
// pipelined body where it accepts the call and main's generic body elsewhere;
// XATTN_SCA_FWD_PIPE=0 => main's forward exactly; an explicit XATTN_SCA_UMMA_FWD
// "0" or "1" => that body (generic, leaf) and never the pipelined one;
// XATTN_SCA_FWD_PIPE=1 with an explicit XATTN_SCA_UMMA_FWD, or any other
// XATTN_SCA_FWD_PIPE value ("2" included), raises before anything is allocated.
//
// FORWARD ROWS AT (256,256) AND (128,128), LEVEL 0 - READ BEFORE QUOTING ONE. Both tcgen05
// forward bodies record the SAME row, [kForward][kUmma][kUmma][kRan]: the
// pipelined body (fwdpipe::PipeForwardKernel, umma_fwd_pipe.cuh) and the older
// leaf (umma::ForwardKernel, umma_fwd.cuh). No Body value separates them (the
// vocabulary names instruction classes); two host launch counts do,
// _flash_sca_native_fwd_pipe_launches and _flash_sca_native_fwd_leaf_launches.
// A forward row on a head that has the pipelined body MUST therefore carry:
//   (1) the forward tag rows, read by difference around ONE call,
//   (2) the pipe-launch delta and (3) the leaf-launch delta around that call,
//   (4) the XATTN_SCA_FWD_PIPE value and (5) the XATTN_SCA_UMMA_FWD value in force,
// and the sha256 of the extension that ran it. Without them an `umma` forward row
// at (256,256) or (128,128), level 0, names neither body. One pipe-launch count
// serves both buckets; the call's D and Dv name the bucket.
// A launch-count delta is a HOST launch witness - the launch call returned
// without an error - not device completion and not proof of which instructions
// executed. A receipt joins the source/.so sha256, the effective arm values, the
// executed body by counter delta, and a successful synchronized completion plus
// the numerical result of that same call; static SASS and generic umma tags
// alone do not name the producer of a row.
//
// States:
//   kRan: the selected producer's launch returned successfully.
//   kRefused: a pre-write refusal with fallback, recorded as ONE decision row
//     (selected=kUmma, launched=kMmaSync). Only the forward emits it, and only
//     under `XATTN_SCA_UMMA_FWD=1` or on a call the pipelined body serves at
//     (256,256) or (128,128) (whose launcher refuses when a check made before its launch call
//     fails), because a forward `selected=kUmma` requires one of the two and
//     otherwise unset or "0" never reach the native leaf to refuse it. A native
//     forward launch call (pipelined body or leaf) that reports an error is NOT
//     this row: it raises with the CUDA error, records nothing, and no other body
//     runs over its outputs.
//     A bucket this build compiles no leaf for is NOT this row: host.cc
//     raises on "1" before anything is launched when HasUmmaForwardLeaf is
//     false, so an uncovered bucket writes no row at all and is visible only as
//     that raise. The reachable refusal is a compiled-leaf call whose prepared
//     maps do not fit the leaf's column budget: the leaf reports it by returning
//     false BEFORE its first launch, and the generic body then carries the
//     phase. So this row never describes a partial launch. RECONCILING A FALLBACK: such a call writes TWO rows --
//     this decision row, then the generic body's own [kMmaSync][kMmaSync][kRan]
//     execution row, because the generic record site is shared with the plain
//     generic path. `selected` is therefore a decision only on a kRefused row:
//     count decisions from kRefused rows and executed bodies from kRan rows
//     (exactly one per body that ran), never one row per call. Backward
//     selection reports refusals in preparation diagnostics instead; it never
//     emits a phantom alternative producer.
//   kSkipped: no separate body was attempted, both body fields kNone. For the
//     generic atomic-dQ route the row-dQ stage is skipped: the key kernel plus
//     the host fold owns dQ. This does not mean dQ was omitted. A selected
//     native product whose output plane has no element (no query rows, or no
//     key rows at all) is not launched either, and its phases record kSkipped
//     too (umma_bwd.cu:LaunchLeaf): a key launcher returns true on an empty
//     key grid without launching, so its return value alone cannot back a
//     kRan row.
//
// Rows are keyed by (phase, selected, launched, state), not by order. Compare
// counter deltas for an isolated call to determine multiplicity. A failed
// launch records no success row; absence is not evidence for another body.
// Batch-zero calls record nothing.
//
// Not covered: scalar LaunchTileBounds/LaunchKeyTileRange, preparation copies,
// or the torch allocation/scale/copy kernels used by an ACTUAL atomic-dQ plan.
// Deterministic=False does not imply atomics: a selected direct plan needs none
// of those accumulator operations. Tagged phase counts therefore are not total
// kernel-launch counts. Bucket queries and cudaFuncSetAttribute launch nothing.
//
// This header is host-only; neither its counters nor their reset/read functions
// are consulted by device code.
#pragma once

#include <atomic>
#include <cstdint>

namespace xattn::ops::flash_sca_native {

// The contractions of one SCA call, in launch order within the call. dK and dV
// are separate phases even when one fused leaf computes both. A fused launch
// records both phases, one row each; a future split route must preserve the
// same one-producer obligation for each contraction.
enum class Phase : int {
  kForward = 0,  // forward: one launch over query tiles
  kDelta = 1,    // backward: row sums of dY*O (a scalar reduction, see Body)
  kDq = 2,       // backward: row-owned dQ
  kDk = 3,       // backward: key-owned dK
  kDv = 4,       // backward: key-owned dV
  kPhaseCount = 5,
};

// The body that carries a phase. kMmaSync is the generic SM80 `mma.sync` path
// (SASS: HMMA); kUmma is a tcgen05/UMMA path (SASS: UTCHMMA-class); kScalar is a
// body with no matrix instruction at all (SASS: FFMA, zero HMMA and zero
// UTCHMMA) -- `detail::DeltaKernel` is one today, and the untagged metadata
// kernels named above are others. For a join with a cubin census the mapping is
// one line: kUmma -> umma_mma, kMmaSync -> legacy_mma, kScalar -> ffma with zero
// HMMA and zero UTCHMMA. No wgmma body exists on this path, and no torch op is a
// Body. kNone appears only in a kSkipped row, in both body fields.
enum class Body : int {
  kNone = 0,
  kMmaSync = 1,
  kUmma = 2,
  kScalar = 3,
  kBodyCount = 4,
};

enum class State : int {
  kRan = 0,
  kRefused = 1,
  kSkipped = 2,
  kStateCount = 3,
};

constexpr int kPhaseCount = static_cast<int>(Phase::kPhaseCount);
constexpr int kBodyCount = static_cast<int>(Body::kBodyCount);
constexpr int kStateCount = static_cast<int>(State::kStateCount);

struct PhaseTagCounters {
  // [phase][selected][launched][state]
  std::atomic<int64_t> count[kPhaseCount][kBodyCount][kBodyCount][kStateCount];
  PhaseTagCounters() {
    for (int p = 0; p < kPhaseCount; ++p) {
      for (int s = 0; s < kBodyCount; ++s) {
        for (int l = 0; l < kBodyCount; ++l) {
          for (int t = 0; t < kStateCount; ++t) {
            count[p][s][l][t].store(0, std::memory_order_relaxed);
          }
        }
      }
    }
  }
};

// One instance per process: a static local of an inline function is a single
// object across translation units, including across the per-bucket
// instantiation files that each call the launch templates.
inline PhaseTagCounters& PhaseTags() {
  static PhaseTagCounters counters;
  return counters;
}

// The calling host thread's own copy of the same table. A launch site records
// on the host thread that enqueues its launch, and a host thread runs one call
// at a time, so the change in this table across one call belongs to that call
// alone. The process-wide table above cannot say that while other threads run
// calls concurrently. Zero-initialized as thread storage and never reset:
// ResetPhaseTags() deliberately leaves it alone, so a reader takes a difference
// across its own call and never reads an absolute.
struct ThreadPhaseTagCounts {
  int64_t count[kPhaseCount][kBodyCount][kBodyCount][kStateCount];
};

inline ThreadPhaseTagCounts& ThreadPhaseTags() {
  thread_local ThreadPhaseTagCounts counts;
  return counts;
}

// The calling thread's count of one row.
inline int64_t ThreadPhaseCount(Phase phase, Body selected, Body launched, State state) {
  return ThreadPhaseTags().count[static_cast<int>(phase)][static_cast<int>(selected)]
                                [static_cast<int>(launched)][static_cast<int>(state)];
}

// Records one launch decision. Call it exactly once per (phase, decision).
inline void RecordPhase(Phase phase, Body selected, Body launched, State state) {
  const int p = static_cast<int>(phase);
  const int s = static_cast<int>(selected);
  const int l = static_cast<int>(launched);
  const int t = static_cast<int>(state);
  PhaseTags().count[p][s][l][t].fetch_add(1, std::memory_order_relaxed);
  ++ThreadPhaseTags().count[p][s][l][t];
}

// Zeroes the process-wide table only; the per-thread copies keep counting.
inline void ResetPhaseTags() {
  PhaseTagCounters& tags = PhaseTags();
  for (int p = 0; p < kPhaseCount; ++p) {
    for (int s = 0; s < kBodyCount; ++s) {
      for (int l = 0; l < kBodyCount; ++l) {
        for (int t = 0; t < kStateCount; ++t) {
          tags.count[p][s][l][t].store(0, std::memory_order_relaxed);
        }
      }
    }
  }
}

inline const char* PhaseName(Phase phase) {
  switch (phase) {
    case Phase::kForward: return "forward";
    case Phase::kDelta: return "delta";
    case Phase::kDq: return "dq";
    case Phase::kDk: return "dk";
    case Phase::kDv: return "dv";
    default: return "unknown";
  }
}

inline const char* BodyName(Body body) {
  switch (body) {
    case Body::kNone: return "none";
    case Body::kMmaSync: return "mma_sync";
    case Body::kUmma: return "umma";
    case Body::kScalar: return "scalar";
    default: return "unknown";
  }
}

inline const char* StateName(State state) {
  switch (state) {
    case State::kRan: return "ran";
    case State::kRefused: return "refused";
    case State::kSkipped: return "skipped";
    default: return "unknown";
  }
}

}  // namespace xattn::ops::flash_sca_native
