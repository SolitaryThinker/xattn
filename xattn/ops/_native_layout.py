"""Layout normalization for native CUDA vector-load boundaries."""

import torch
from torch import Tensor


def native_contiguous(tensor: Tensor) -> Tensor:
    """Return identical values in contiguous, 16-byte-aligned storage.

    Contiguous views can retain an odd storage offset. ``contiguous()`` alone
    does not copy those views, so it is not a vector-load alignment guarantee.
    This helper changes neither dtype nor device, preserves autograd edges,
    and uses only tensor metadata and ordinary current-stream copy operations.
    Empty tensors require no aligned payload. Shape/dtype/device validation
    remains the responsibility of the calling API.
    """
    packed = tensor.contiguous()
    if packed.numel() and packed.data_ptr() % 16:
        packed = packed.clone(memory_format=torch.contiguous_format)
    return packed
