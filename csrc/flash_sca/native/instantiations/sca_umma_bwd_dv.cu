// Independent key-owned dV leaf (family A's split derivation), D, Dv in
// {64, 128, 256}. Sibling of sca_umma_bwd_dk.cu; the fused TUs are untouched.
//
// Its pair list must equal the dK sibling's and the header's kKeyAutoTable rows in both
// directions - nine pairs, the set the router can ask for (see the dK sibling's comment
// on the three 64-wide cells). The census that proves that, and the per-pair SMEM/TMEM
// audit, live in sca_umma_bwd_dk.cu: BackwardDkdvPlan is keyed on (kD, kDv) alone, so a
// second audit here would be a second place for the same nine numbers to disagree.
#include "flash_sca/native/umma_bwd_dkdv.cuh"

XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 64, 64, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 64, 64, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 64, 128, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 64, 128, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 64, 256, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 64, 256, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 128, 64, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 128, 64, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 128, 128, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 128, 128, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 128, 256, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 128, 256, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 256, 64, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 256, 64, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 256, 128, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 256, 128, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__nv_bfloat16, 256, 256, 2)
XATTN_UMMA_BWD_KEY_INSTANTIATE(__half, 256, 256, 2)
