# SPDX-License-Identifier: MIT

"""Shared host-side contracts for FlyDSL BF16 decode GEMM kernels."""

from __future__ import annotations

import torch

from aiter.jit.utils.chip_info import get_gfx


def _storage_range(tensor: torch.Tensor) -> tuple[int, int]:
    begin = tensor.data_ptr()
    return begin, begin + tensor.numel() * tensor.element_size()


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    lhs_begin, lhs_end = _storage_range(lhs)
    rhs_begin, rhs_end = _storage_range(rhs)
    return lhs_begin < rhs_end and rhs_begin < lhs_end


def validate_gemm_decode_tensors(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> None:
    """Validate the packed real-tensor ABI shared by both kernel families."""
    tensors = {"A": A, "B": B, "C": C}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dim() != 2:
            raise ValueError(f"{name} must be rank 2, got rank {tensor.dim()}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must have dtype torch.bfloat16")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA/ROCm device")

    if not (1 <= M <= 4):
        raise ValueError("decode GEMM supports M in [1, 4]")
    if N <= 0 or K <= 0:
        raise ValueError("decode GEMM requires positive N and K")
    if A.device != B.device or A.device != C.device:
        raise ValueError("A, B, and C must be on the same device")

    expected_shapes = {"A": (M, K), "B": (N, K), "C": (M, N)}
    expected_strides = {"A": (K, 1), "B": (K, 1), "C": (N, 1)}
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"{name} must have shape {expected_shapes[name]}, "
                f"got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous() or tuple(tensor.stride()) != expected_strides[name]:
            raise ValueError(f"{name} must use packed row-major storage")

    if _overlaps(C, A) or _overlaps(C, B):
        raise ValueError("C must not overlap A or B")
    if get_gfx() != "gfx950":
        raise ValueError(f"decode GEMM requires gfx950, got {get_gfx()}")
