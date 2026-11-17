// This file is intentionally empty: the forward launch preparation lives in
// `umma_prepare.h` (`umma::fwd_host`) beside
// the backward seam's preparation, so that the two directions share one header,
// one refusal vocabulary and one set of map-source predicates instead of two
// copies of the same decisions.
//
// Nothing includes it, and it is a removal candidate.
#pragma once
