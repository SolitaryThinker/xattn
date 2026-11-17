// Independent key-owned dK leaf (family A's split derivation of the fused dK/dV
// header), D, Dv in {64, 128, 256}: emits one cubin per (bucket pair, dtype) in the
// ordinary package build, so the dk support bit can be proven by census.
//
// The pair list here and the rows of dkdv_detail::kKeyAutoTable (in the header) must be
// EQUAL IN BOTH DIRECTIONS, and neither direction is caught by the compiler or the
// linker: a line without a row is a cubin no call can select (coverage that looks
// compiled and is unreachable), and a row without a line is a leaf whose cubin is
// emitted by whichever caller instantiates the header template instead of by this TU -
// still functional, but the provenance claim "this TU emits the pair" becomes false.
// The census that ships with a change to either side therefore prints the table's rows
// and the two TUs' lines in ONE form (dtype, D, Dv, kWrite) and diffs them both ways.
//
// What that census does NOT prove: the host
// launcher symbols are weak here, not strong, and each TU carries the whole split set
// plus the fused set - one device kernel per (pair, dtype, kWrite) the table names,
// not the number of lines this file holds - because the one table references
// KeyLaunchOne for every row it holds, so including the header instantiates all of
// them. The count is therefore evidence of what the object CONTAINS, never of what this
// file alone declared.
//
// Nine pairs, because nine is what the router can ask for: the contract's real
// dimensions round into every cell of {64,128,256} x {64,128,256}, and the backward
// router - the kForward=false instantiation of the
// templated DispatchBuckets (dispatch.cu:89-111) - routes all nine cells, including
// the three with a 64-wide axis against a wider one: (64,128), (128,64), (256,64).
#include "flash_sca/native/umma_bwd_dkdv.cuh"

XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 64, 64, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 64, 64, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 64, 128, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 64, 128, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 64, 256, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 64, 256, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 128, 64, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 128, 64, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 128, 128, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 128, 128, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 128, 256, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 128, 256, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 256, 64, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 256, 64, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 256, 128, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 256, 128, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 256, 256, 1)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 256, 256, 1)

// Per-pair resource audit, compiler-checked. BackwardDkdvPlan is keyed on (kD, kDv)
// alone - the kWrite output set changes which products are computed, never the SMEM or
// TMEM map - so the table lives in this file only; the dV TU deliberately carries no
// second copy, which would be a second place for the nine numbers to disagree.
//
// The expected bytes are the header's own formula evaluated by hand
// (kQBytes = 16384*kD/64, kG = 16384*kDv/64, kK = 8192*kD/64, kV = 8192*kDv/64, then
// the barrier/slot/envelope/segment tail and the 1024 B alignment slack), and the two
// pairs this leaf shipped with - (128,128) = 100352 and (256,256) = 198656 - are the
// figures its class comment publishes, so the formula and the shipped prose agree.
// 232448 B is the opt-in dynamic-SMEM ceiling of the target part; the worst pair clears
// it by 33 KiB.
//
// Three of the cells are the transposes of three audited ones and are asserted as
// such rather than derived afresh: SMEM is a sum over the same four operand buffers, so
// swapping D and Dv must leave it unchanged - (64,128) = (128,64) = 75776, matching
// (64,256) = (256,64) = 124928 - and a pair that broke that symmetry would fail here.
namespace key_budget {
using xattn::ops::flash_sca_native::umma::BackwardDkdvPlan;

static_assert(BackwardDkdvPlan<64, 64>::kSmemBytes == 51200, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<64, 64>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<64, 128>::kSmemBytes == 75776, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<64, 128>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<64, 256>::kSmemBytes == 124928, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<64, 256>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<128, 64>::kSmemBytes == 75776, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<128, 64>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<128, 128>::kSmemBytes == 100352, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<128, 128>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<128, 256>::kSmemBytes == 149504, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<128, 256>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<256, 64>::kSmemBytes == 124928, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<256, 64>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<256, 128>::kSmemBytes == 149504, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<256, 128>::kAllocCols == 512, "fused 512-column TMEM map");

static_assert(BackwardDkdvPlan<256, 256>::kSmemBytes == 198656, "audited key-leaf SMEM");
static_assert(BackwardDkdvPlan<256, 256>::kAllocCols == 512, "fused 512-column TMEM map");

// The ceiling, asserted on the worst of the nine rather than on a chosen one: a bucket
// that grew past it must fail here, not at the driver's launch.
static_assert(BackwardDkdvPlan<256, 256>::kSmemBytes <= 232448, "sm_103a opt-in SMEM cap");
}  // namespace key_budget
