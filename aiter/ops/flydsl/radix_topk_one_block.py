# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL one-block radix TopK interface for gfx942, gfx950 and gfx1250."""

from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.kernels_common import get_warp_size
from .kernels.radix_topk_one_block import (
    _COMPACT_CAPACITY,
    build_radix_topk_one_block_module,
)
from .kernels.tensor_shim import _run_compiled

_MAX_BUFFER_ROW_ELEMENTS = ((1 << 32) - 1) // torch.float32.itemsize
_SUPPORTED_ARCHES = ("gfx942", "gfx950", "gfx1250")
# The measured short-row crossover on gfx1250 is between 256 and 512 rows.
_SHORT_ROWS_1024_THREAD_MAX_ROWS = 256

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
            raise ValueError(f"{name} must be on the same CUDA device as logits")

    indices_shape, indices_stride, indices_dtype, indices_device = indices_signature
    if indices_shape != (num_rows, k) or indices_dtype != torch.int32:
        raise ValueError("indices must be an int32 tensor with shape [num_rows, k]")
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
        raise ValueError("values must be a float32 tensor with shape [num_rows, k]")
    if stride != (k, 1):
        raise ValueError("values must be contiguous")
    if device != logits_device:
        raise ValueError("values must be on the same CUDA device as logits")


def _validate_call(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None,
) -> None:
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


@lru_cache(maxsize=128)
def _is_call_supported(
    logits_signature: _TensorSignature,
    row_starts_signature: _TensorSignature,
    row_ends_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values_signature: _TensorSignature | None,
) -> bool:
    logits_device = logits_signature[3]
    if logits_device.type != "cuda":
        return False
    if get_gfx() not in _SUPPORTED_ARCHES:
        return False
    try:
        _validate_signature(
            logits_signature,
            row_starts_signature,
            row_ends_signature,
            indices_signature,
            num_rows,
            stride0,
            stride1,
            k,
        )
        if values_signature is not None:
            _validate_values_signature(
                values_signature,
                num_rows,
                k,
                logits_device,
            )
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def is_radix_topk_one_block_supported(
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
    """Return whether the call can use the one-block radix kernel."""
    return _is_call_supported(
        _tensor_signature(logits),
        _tensor_signature(row_starts),
        _tensor_signature(row_ends),
        _tensor_signature(indices),
        num_rows,
        stride0,
        stride1,
        k,
        None if values is None else _tensor_signature(values),
    )


def radix_topk_one_block(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    *,
    is_decode: bool = False,
    next_n: int = 1,
) -> None:
    """Write prefill or decode TopK indices through one shared wrapper."""
    if is_decode:
        from .topk_per_row import _validate_flydsl_topk_call

        _validate_flydsl_topk_call(
            logits,
            next_n,
            row_ends,
            indices,
            num_rows,
            stride0,
            stride1,
            k,
            values,
        )
        kernel_row_starts = row_ends
    else:
        if row_starts is None:
            raise ValueError("row_starts is required for prefill")
        _validate_call(
            logits,
            row_starts,
            row_ends,
            indices,
            num_rows,
            stride0,
            stride1,
            k,
            values,
        )
        kernel_row_starts = row_starts

    arch = get_gfx()
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(
            "FlyDSL one-block radix TopK supports gfx942, gfx950 and gfx1250"
        )
    if num_rows == 0:
        return

    width = logits.shape[1]
    short_rows = width <= _COMPACT_CAPACITY
    block_threads = (
        1024 if not short_rows or num_rows <= _SHORT_ROWS_1024_THREAD_MAX_ROWS else 256
    )
    stream = torch.cuda.current_stream(logits.device)
    launcher = build_radix_topk_one_block_module(
        k,
        block_threads=block_threads,
        write_values=values is not None,
        stable=stable,
        short_rows=short_rows,
        is_decode=is_decode,
        wave_size=get_warp_size(arch),
    )
    _run_compiled(
        launcher,
        logits,
        kernel_row_starts,
        row_ends,
        indices,
        values if values is not None else logits,
        width,
        next_n,
        num_rows,
        stream,
    )
