# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL decode TopK interface."""

from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from ..kernels.kernels_common import get_warp_size
from ..kernels.tensor_shim import _run_compiled
from ..kernels.topk.radix_topk_one_block import (
    _COMPACT_CAPACITY,
    _MAX_ROW_ELEMENTS,
    build_radix_topk_one_block_module,
)
from ..kernels.topk.topk_per_row_decode import (
    build_topk_per_row_decode_module,
    topk_per_row_decode_chunks,
    topk_per_row_decode_workspace_shapes,
)
from ..kernels.topk.topk_per_row_decode_persistent import (
    build_topk_per_row_decode_one_workgroup_module,
)

# Measured crossover between the one-workgroup and multi-kernel paths.
_ONE_WORKGROUP_MAX_ROW_WIDTH = 20_000
_SHORT_ROWS_1024_THREAD_MAX_ROWS = 256


@lru_cache(maxsize=16)
def _get_cached_workspace(
    device: torch.device,
    stream_id: int,
    hist_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep scratch isolated by device, stream, and exact kernel layout."""
    return (
        torch.empty(hist_shape, device=device, dtype=torch.int32),
        torch.empty(state_shape, device=device, dtype=torch.int32),
    )


def _get_topk_workspace(
    device: torch.device,
    stream_id: int,
    hist_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Do not let graph-pool allocations escape through the process cache.
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.empty(hist_shape, device=device, dtype=torch.int32),
            torch.empty(state_shape, device=device, dtype=torch.int32),
        )
    return _get_cached_workspace(device, stream_id, hist_shape, state_shape)


def clear_topk_per_row_decode_workspace_cache() -> None:
    _get_cached_workspace.cache_clear()


@lru_cache(maxsize=128)
def _validate_topk_signature(
    logits_shape: torch.Size,
    logits_stride: tuple[int, ...],
    logits_dtype: torch.dtype,
    logits_device: torch.device,
    seq_lens_shape: torch.Size,
    seq_lens_stride: tuple[int, ...],
    seq_lens_dtype: torch.dtype,
    seq_lens_device: torch.device,
    indices_shape: torch.Size,
    indices_stride: tuple[int, ...],
    indices_dtype: torch.dtype,
    indices_device: torch.device,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
) -> None:
    if len(logits_shape) != 2 or logits_dtype != torch.float32:
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if logits_device.type != "cuda":
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if k <= 0 or k > logits_shape[1]:
        raise ValueError("k must be in the range [1, logits.shape[1]]")
    if logits_stride[1] != 1:
        raise ValueError("logits must have inner stride 1")

    rows = logits_shape[0]
    if num_rows != rows:
        raise ValueError("num_rows must equal logits.shape[0]")
    # There used to be a refusal here at 4 GiB. The kernels built one buffer
    # descriptor over the whole logits tensor, and both `num_records` and the
    # load's voffset in it are 32-bit byte quantities, so a row past 4 GiB was
    # unaddressable and read as zero -- silently wrong rather than faulting, at
    # 4096 x 262144 but not 4095 x 262144. They now slice the row before
    # building the descriptor, which puts the row's base in the descriptor's
    # 48-bit base address and leaves only an offset within the row, so the
    # tensor as a whole is no longer bounded. The refusal went with it.
    if (stride0, stride1) != logits_stride:
        raise ValueError("stride0 and stride1 must match logits strides")

    if len(seq_lens_shape) != 1 or seq_lens_dtype != torch.int32:
        raise ValueError("seq_lens must be a 1D int32 tensor")
    if seq_lens_stride != (1,):
        raise ValueError("seq_lens must be contiguous")
    if seq_lens_device != logits_device:
        raise ValueError("seq_lens must be on the same CUDA device as logits")
    if next_n <= 0:
        raise ValueError("next_n must be positive")
    required_seq_lens = (rows + next_n - 1) // next_n
    if seq_lens_shape[0] < required_seq_lens:
        raise ValueError("seq_lens does not have enough entries for logits rows")

    if indices_shape != (rows, k) or indices_dtype != torch.int32:
        raise ValueError("indices must be an int32 tensor with shape [rows, k]")
    if indices_stride != (k, 1):
        raise ValueError("indices must be contiguous")
    if indices_device != logits_device:
        raise ValueError("indices must be on the same CUDA device as logits")


@lru_cache(maxsize=128)
def _validate_values_signature(
    values_shape: torch.Size,
    values_stride: tuple[int, ...],
    values_dtype: torch.dtype,
    values_device: torch.device,
    rows: int,
    k: int,
    logits_device: torch.device,
) -> None:
    if values_dtype != torch.float32:
        raise ValueError("values must be a float32 tensor")
    if values_shape != (rows, k):
        raise ValueError("values must have shape [rows, k]")
    if values_stride != (k, 1):
        raise ValueError("values must be contiguous")
    if values_device != logits_device:
        raise ValueError("values must be on the same CUDA device as logits")


def _validate_flydsl_topk_call(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None,
) -> None:
    _validate_topk_signature(
        logits.shape,
        logits.stride(),
        logits.dtype,
        logits.device,
        seq_lens.shape,
        seq_lens.stride(),
        seq_lens.dtype,
        seq_lens.device,
        indices.shape,
        indices.stride(),
        indices.dtype,
        indices.device,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
    )
    if values is not None:
        _validate_values_signature(
            values.shape,
            values.stride(),
            values.dtype,
            values.device,
            logits.shape[0],
            k,
            logits.device,
        )


_FLYDSL_TOPK_ONE_BLOCK_ARCHES = ("gfx942", "gfx950", "gfx1250")


def _validate_radix_topk_one_block_call(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    *,
    is_decode: bool,
    next_n: int,
) -> None:
    """Raise if this call cannot run the one-block radix kernel."""
    _validate_flydsl_topk_call(
        logits, next_n, row_ends, indices, num_rows, stride0, stride1, k, values
    )
    if row_starts is not row_ends:
        _validate_flydsl_topk_call(
            logits, next_n, row_starts, indices, num_rows, stride0, stride1, k, values
        )
    if logits.shape[1] > _MAX_ROW_ELEMENTS:
        raise ValueError("one logits row exceeds the AMD buffer descriptor span")


_TensorSignature = tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


@lru_cache(maxsize=128)
def _is_flydsl_topk_call_supported(
    logits_signature: _TensorSignature,
    seq_lens_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values_signature: _TensorSignature | None,
) -> bool:
    try:
        _validate_topk_signature(
            *logits_signature,
            *seq_lens_signature,
            *indices_signature,
            next_n,
            num_rows,
            stride0,
            stride1,
            k,
        )
        if values_signature is not None:
            _validate_values_signature(
                *values_signature, logits_signature[0][0], k, logits_signature[3]
            )
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=128)
def _is_flydsl_radix_topk_one_block_call_supported(
    logits_signature: _TensorSignature,
    row_starts_signature: _TensorSignature | None,
    row_ends_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values_signature: _TensorSignature | None,
    is_decode: bool,
    arch: str,
) -> bool:
    if arch not in _FLYDSL_TOPK_ONE_BLOCK_ARCHES:
        return False
    if logits_signature[0][1] > _MAX_ROW_ELEMENTS:
        return False
    if not is_decode and row_starts_signature is None:
        return False
    if not _is_flydsl_topk_call_supported(
        logits_signature,
        row_ends_signature,
        indices_signature,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        values_signature,
    ):
        return False
    if row_starts_signature is None or row_starts_signature == row_ends_signature:
        return True
    return _is_flydsl_topk_call_supported(
        logits_signature,
        row_starts_signature,
        indices_signature,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        values_signature,
    )


def _is_flydsl_radix_topk_one_block_supported(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    *,
    is_decode: bool = False,
    next_n: int = 1,
) -> bool:
    """Return whether the one-block radix kernel supports this call."""
    return _is_flydsl_radix_topk_one_block_call_supported(
        _tensor_signature(logits),
        None if row_starts is None else _tensor_signature(row_starts),
        _tensor_signature(row_ends),
        _tensor_signature(indices),
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        None if values is None else _tensor_signature(values),
        is_decode,
        get_gfx_runtime(),
    )


def is_flydsl_top_k_per_row_decode_supported(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None = None,
) -> bool:
    """Return whether a call satisfies every FlyDSL-only precondition."""
    return _is_flydsl_topk_call_supported(
        _tensor_signature(logits),
        _tensor_signature(seq_lens),
        _tensor_signature(indices),
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        None if values is None else _tensor_signature(values),
    )


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    values: torch.Tensor | None = None,
) -> None:
    """Write per-row TopK indices using each request's effective context length."""

    _validate_flydsl_topk_call(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, k, values
    )

    rows, width = logits.shape
    arch = torch.cuda.get_device_properties(logits.device).gcnArchName
    wave_size = get_warp_size(arch)
    stream = torch.cuda.current_stream(logits.device)
    if width <= _ONE_WORKGROUP_MAX_ROW_WIDTH:
        launcher = build_topk_per_row_decode_one_workgroup_module(
            k, wave_size=wave_size, write_values=values is not None
        )
        _run_compiled(
            launcher,
            logits,
            seq_lens,
            indices,
            values if values is not None else logits,
            width,
            next_n,
            stride0,
            rows,
            stream,
        )
        return

    # One row is `chunks` blocks, so the split has to follow the row count: at
    # one row, 16 chunks leaves all but a handful of CUs idle, and at many rows
    # it makes the single-block reduce walk counters nobody needed.
    chunks = topk_per_row_decode_chunks(rows, width, wave_size)
    hist_shape, state_shape = topk_per_row_decode_workspace_shapes(rows, stable, chunks)
    partial_hist, state = _get_topk_workspace(
        logits.device, stream.cuda_stream, hist_shape, state_shape
    )

    launcher = build_topk_per_row_decode_module(
        k,
        stable,
        wave_size=wave_size,
        write_values=values is not None,
        chunks_per_row=chunks,
    )
    _run_compiled(
        launcher,
        logits,
        seq_lens,
        indices,
        values if values is not None else logits,
        partial_hist,
        state,
        width,
        next_n,
        stride0,
        rows,
        stream,
    )


def flydsl_radix_topk_one_block(
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
    """Launch the one-block radix TopK kernel for one prefill or decode call."""
    if row_starts is None:
        if not is_decode:
            raise ValueError("row_starts is required for prefill")
        row_starts = row_ends
    _validate_radix_topk_one_block_call(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride0,
        stride1,
        k,
        is_decode=is_decode,
        next_n=next_n,
    )
    if num_rows == 0:
        return

    arch = get_gfx_runtime()
    width = logits.shape[1]
    wave_size = get_warp_size(arch)
    stream = torch.cuda.current_stream(logits.device)
    short_rows = width <= _COMPACT_CAPACITY
    block_threads = (
        1024 if not short_rows or num_rows <= _SHORT_ROWS_1024_THREAD_MAX_ROWS else 256
    )
    launcher = build_radix_topk_one_block_module(
        k,
        block_threads=block_threads,
        write_values=values is not None,
        stable=stable,
        short_rows=short_rows,
        is_decode=is_decode,
        wave_size=wave_size,
        arch=arch,
    )
    _run_compiled(
        launcher,
        logits,
        row_starts,
        row_ends,
        indices,
        values if values is not None else logits,
        width,
        next_n,
        num_rows,
        stream,
    )
