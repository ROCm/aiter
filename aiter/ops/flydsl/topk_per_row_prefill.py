# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL one-workgroup prefill TopK interface."""

from functools import lru_cache

import flydsl.compiler as flyc
import torch

from .kernels.tensor_shim import _run_compiled, ptr_arg
from .kernels.topk_per_row_prefill_one_workgroup import (
    build_topk_per_row_prefill_one_workgroup_module,
)

_MAX_BUFFER_ROW_ELEMENTS = ((1 << 32) - 1) // torch.float32.itemsize
_SUPPORTED_ARCHES = ("gfx1250",)

_TensorSignature = tuple[
    torch.Size,
    tuple[int, ...],
    torch.dtype,
    torch.device,
]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


@lru_cache(maxsize=128)
def _validate_signature(
    logits_signature: _TensorSignature,
    row_starts_signature: _TensorSignature,
    row_ends_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
) -> None:
    logits_shape, logits_stride, logits_dtype, logits_device = logits_signature
    if (
        len(logits_shape) != 2
        or logits_dtype != torch.float32
        or logits_device.type != "cuda"
    ):
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if logits_stride[1] != 1:
        raise ValueError("logits must have inner stride 1")
    if (stride0, stride1) != logits_stride:
        raise ValueError("stride0 and stride1 must match logits strides")
    if not 0 <= num_rows <= logits_shape[0]:
        raise ValueError("num_rows must be in [0, logits.shape[0]]")
    if logits_shape[1] > _MAX_BUFFER_ROW_ELEMENTS:
        raise ValueError("one logits row exceeds the AMD buffer descriptor span")
    if k <= 0:
        raise ValueError("k must be positive")

    for name, signature in (
        ("row_starts", row_starts_signature),
        ("row_ends", row_ends_signature),
    ):
        shape, tensor_stride, dtype, device = signature
        if len(shape) != 1 or dtype != torch.int32:
            raise ValueError(f"{name} must be a 1D int32 tensor")
        if tensor_stride != (1,):
            raise ValueError(f"{name} must be contiguous")
        if shape[0] < num_rows:
            raise ValueError(f"{name} does not have enough entries")
        if device != logits_device:
            raise ValueError(
                f"{name} must be on the same CUDA device as logits"
            )

    indices_shape, indices_stride, indices_dtype, indices_device = (
        indices_signature
    )
    if (
        indices_shape != (num_rows, k)
        or indices_dtype != torch.int32
    ):
        raise ValueError(
            "indices must be an int32 tensor with shape [num_rows, k]"
        )
    if indices_stride != (k, 1):
        raise ValueError("indices must be contiguous")
    if indices_device != logits_device:
        raise ValueError("indices must be on the same CUDA device as logits")


@lru_cache(maxsize=128)
def _validate_values_signature(
    values_signature: _TensorSignature,
    num_rows: int,
    k: int,
    logits_device: torch.device,
) -> None:
    shape, stride, dtype, device = values_signature
    if shape != (num_rows, k) or dtype != torch.float32:
        raise ValueError(
            "values must be a float32 tensor with shape [num_rows, k]"
        )
    if stride != (k, 1):
        raise ValueError("values must be contiguous")
    if device != logits_device:
        raise ValueError(
            "values must be on the same CUDA device as logits"
        )


def _arch_name(device: torch.device) -> str:
    return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]


def is_flydsl_top_k_per_row_prefill_supported(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None = None,
) -> bool:
    """Return whether the call can use the gfx1250 one-workgroup kernel."""
    if not isinstance(logits, torch.Tensor) or logits.device.type != "cuda":
        return False
    if _arch_name(logits.device) not in _SUPPORTED_ARCHES:
        return False
    try:
        _validate_signature(
            _tensor_signature(logits),
            _tensor_signature(row_starts),
            _tensor_signature(row_ends),
            _tensor_signature(indices),
            num_rows,
            stride0,
            stride1,
            k,
        )
        if values is not None:
            _validate_values_signature(
                _tensor_signature(values),
                num_rows,
                k,
                logits.device,
            )
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def flydsl_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    max_effective_row_len: int | None = None,
) -> None:
    """Write per-row TopK indices and optional values."""
    _validate_signature(
        _tensor_signature(logits),
        _tensor_signature(row_starts),
        _tensor_signature(row_ends),
        _tensor_signature(indices),
        num_rows,
        stride0,
        stride1,
        k,
    )
    if values is not None:
        _validate_values_signature(
            _tensor_signature(values),
            num_rows,
            k,
            logits.device,
        )
    if _arch_name(logits.device) not in _SUPPORTED_ARCHES:
        raise ValueError("FlyDSL prefill TopK currently supports gfx1250 only")
    if num_rows == 0:
        return

    if max_effective_row_len is None:
        max_effective_row_len = logits.shape[1]
    if not 0 <= max_effective_row_len <= logits.shape[1]:
        raise ValueError(
            "max_effective_row_len must be in [0, logits.shape[1]]"
        )
    block_threads = 256 if max_effective_row_len <= 4096 else 1024
    device_index = logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    backend = flyc.compile_backend_name()
    launcher = build_topk_per_row_prefill_one_workgroup_module(
        k,
        block_threads=block_threads,
        write_values=values is not None,
        stable=stable,
        device_index=device_index,
        backend=backend,
    )
    stream = torch.cuda.current_stream(logits.device)
    with torch.cuda.device(logits.device):
        _run_compiled(
            launcher,
            ptr_arg(logits),
            ptr_arg(row_starts),
            ptr_arg(row_ends),
            ptr_arg(indices),
            ptr_arg(values if values is not None else logits),
            stride0,
            num_rows,
            stream,
        )
