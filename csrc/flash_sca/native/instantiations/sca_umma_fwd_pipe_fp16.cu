// The pipelined forward (fwdpipe::TryLaunchPipe<E, D, Dv>) is instantiated per bucket pair, beside
// its leaf launcher, in sca_umma_fwd_d<D>_v<Dv>.cu. This TU instantiates nothing; it stays so that
// probe scripts that prepend XATTN_FWD_PROBE_HOOKS to it keep finding it.
#include "flash_sca/native/umma_fwd_pipe.cuh"
