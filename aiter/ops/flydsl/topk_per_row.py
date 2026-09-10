# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared FlyDSL TopK validation and one-block / multi-block dispatch."""

from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.kernels_common import get_warp_size
from .kernels.radix_topk_multi_block import (
    build_radix_topk_multi_block_module,
    radix_topk_multi_block_workspace_shapes,
)
from .kernels.radix_topk_one_block import (
    _COMPACT_CAPACITY,
    build_radix_topk_one_block_module,
)
from .kernels.tensor_shim import _run_compiled

_MAX_BUFFER_ROW_ELEMENTS = ((1 << 32) - 1) // torch.float32.itemsize
_SUPPORTED_ARCHES = ("gfx942", "gfx950", "gfx1250")
# (maximum batch size, maximum one-block row width), with inclusive bounds.
# Batch size is the number of rows being processed (including decode MTP rows).
# Match the first batch band; None is the final catch-all band. Within each
# band, wider rows use multi-block. Physical width avoids reading CUDA bounds.
_OneBlockDispatchBands = tuple[tuple[int | None, int], ...]
# Untuned arches keep the previous single 20k cutoff.
_UNTUNED_ONE_BLOCK_DISPATCH_BANDS: _OneBlockDispatchBands = ((None, 20_000),)
# gfx950 CUDAGraph A/B. Small batches cannot amortize the multi-block launch
# chain, so one-block stays wider. Batch 1-2: unordered still prefers one
# through 49152 (65536 already loses on k>=2048). Batch 3-8: all modes prefer
# one through 28672; 32768 only loses on k=2048 stable+values. Large batches
# keep one-block for occupancy.
# gfx942 CUDAGraph A/B (same protocol). Multi stays ~23-32us at batch 1-8, so
# small batches keep one-block past the tight-mode flip. Batch 1-2: unordered
# k>=2048 prefers one through 40960 (49152 already ~0.95). Batch 3-8: all
# modes prefer one through 28672; keep 32768 so unordered 8x32768 stays
# one-block. Batch 16 all-mode last width is 65536. Batch >=17 occupancy beats
# the launch chain.
_ONE_BLOCK_DISPATCH_BANDS: dict[str, _OneBlockDispatchBands] = {
    "gfx950": (
        (2, 49_152),
        (8, 32_768),
        (16, 32_768),
        (32, 32_768),
        (48, 49_152),
        (64, 65_536),
        (128, 131_072),
        (None, _MAX_BUFFER_ROW_ELEMENTS),
    ),
    "gfx942": (
        (2, 40_960),
        (8, 32_768),
        (16, 65_536),
        (None, _MAX_BUFFER_ROW_ELEMENTS),
    ),
    "gfx1250": _UNTUNED_ONE_BLOCK_DISPATCH_BANDS,
}
_SHORT_ROWS_1024_THREAD_MAX_ROWS = 256

_TensorSignature = tuple[
    torch.Size,
    tuple[int, ...],
    torch.dtype,
    torch.device,
]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


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
    return _get_cached_workspace(
        device,
        stream_id,
        hist_shape,
        state_shape,
    )


def clear_topk_per_row_workspace_cache() -> None:
    _get_cached_workspace.cache_clear()


# Preserve the existing cache-management entry point for decode callers.
clear_topk_per_row_decode_workspace_cache = clear_topk_per_row_workspace_cache


@lru_cache(maxsize=128)
def _validate_topk_signature(
    logits_signature: _TensorSignature,
    row_starts_signature: _TensorSignature | None,
    row_ends_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    values_signature: _TensorSignature | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    is_decode: bool,
    next_n: int,
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
    if logits_shape[1] > _MAX_BUFFER_ROW_ELEMENTS:
        raise ValueError("one logits row exceeds the AMD buffer descriptor span")
    if k <= 0:
        raise ValueError("k must be positive")

    if is_decode:
        if k > logits_shape[1]:
            raise ValueError("k must be in the range [1, logits.shape[1]]")
        if num_rows != logits_shape[0]:
            raise ValueError("num_rows must equal logits.shape[0]")
        if next_n <= 0:
            raise ValueError("next_n must be positive")
        required_entries = (num_rows + next_n - 1) // next_n
        row_bounds = (("seq_lens", row_ends_signature),)
    else:
        if not 0 <= num_rows <= logits_shape[0]:
            raise ValueError("num_rows must be in [0, logits.shape[0]]")
        if row_starts_signature is None:
            raise ValueError("row_starts is required for prefill")
        required_entries = num_rows
        row_bounds = (
            ("row_starts", row_starts_signature),
            ("row_ends", row_ends_signature),
        )

    for name, signature in row_bounds:
        shape, stride, dtype, device = signature
        if len(shape) != 1 or dtype != torch.int32:
            raise ValueError(f"{name} must be a 1D int32 tensor")
        if stride != (1,):
            raise ValueError(f"{name} must be contiguous")
        if shape[0] < required_entries:
            raise ValueError(f"{name} does not have enough entries")
        if device != logits_device:
            raise ValueError(f"{name} must be on the same CUDA device as logits")

    for name, signature, expected_dtype in (
        ("indices", indices_signature, torch.int32),
        ("values", values_signature, torch.float32),
    ):
        if signature is None:
            continue
        shape, stride, dtype, device = signature
        if shape != (num_rows, k) or dtype != expected_dtype:
            raise ValueError(
                f"{name} must have dtype {expected_dtype} and shape [num_rows, k]"
            )
        if stride != (k, 1):
            raise ValueError(f"{name} must be contiguous")
        if device != logits_device:
            raise ValueError(f"{name} must be on the same CUDA device as logits")


@lru_cache(maxsize=128)
def _validate_topk_call(
    logits_signature: _TensorSignature,
    row_starts_signature: _TensorSignature | None,
    row_ends_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    values_signature: _TensorSignature | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    is_decode: bool,
    next_n: int,
) -> bool:
    try:
        _validate_topk_signature(
            logits_signature,
            row_starts_signature,
            row_ends_signature,
            indices_signature,
            values_signature,
            num_rows,
            stride0,
            stride1,
            k,
            is_decode,
            next_n,
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=128)
def _is_flydsl_topk_call_supported(
    logits_signature: _TensorSignature,
    row_starts_signature: _TensorSignature | None,
    row_ends_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    values_signature: _TensorSignature | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    is_decode: bool,
    next_n: int,
    arch: str,
) -> bool:
    return arch in _SUPPORTED_ARCHES and _validate_topk_call(
        logits_signature,
        row_starts_signature,
        row_ends_signature,
        indices_signature,
        values_signature,
        num_rows,
        stride0,
        stride1,
        k,
        is_decode,
        next_n,
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
    """Return whether a decode call satisfies the FlyDSL preconditions."""
    return _is_flydsl_topk_call_supported(
        _tensor_signature(logits),
        None,
        _tensor_signature(seq_lens),
        _tensor_signature(indices),
        None if values is None else _tensor_signature(values),
        num_rows,
        stride0,
        stride1,
        k,
        True,
        next_n,
        get_gfx(),
    )


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
    """Return whether a prefill call satisfies the FlyDSL preconditions."""
    return _is_flydsl_topk_call_supported(
        _tensor_signature(logits),
        _tensor_signature(row_starts),
        _tensor_signature(row_ends),
        _tensor_signature(indices),
        None if values is None else _tensor_signature(values),
        num_rows,
        stride0,
        stride1,
        k,
        False,
        1,
        get_gfx(),
    )


def _should_use_one_block(arch: str, num_rows: int, width: int) -> bool:
    """Select one-block within the row-width limit of the matching batch band."""
    bands = _ONE_BLOCK_DISPATCH_BANDS.get(arch, _UNTUNED_ONE_BLOCK_DISPATCH_BANDS)
    for max_batch_size, max_row_width in bands:
        if max_batch_size is None or num_rows <= max_batch_size:
            return width <= max_row_width
    return False


def _flydsl_top_k_per_row(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    stable: bool,
    *,
    is_decode: bool,
    next_n: int,
) -> None:
    """Validate once, then launch either backend with the same row-bound API."""
    _validate_topk_signature(
        _tensor_signature(logits),
        None if row_starts is None else _tensor_signature(row_starts),
        _tensor_signature(row_ends),
        _tensor_signature(indices),
        None if values is None else _tensor_signature(values),
        num_rows,
        stride0,
        stride1,
        k,
        is_decode,
        next_n,
    )
    arch = get_gfx()
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError("FlyDSL TopK supports gfx942, gfx950 and gfx1250")
    if num_rows == 0:
        return

    width = logits.shape[1]
    wave_size = get_warp_size(arch)
    stream = torch.cuda.current_stream(logits.device)
    # Decode reads seq_lens through row_ends and ignores row_starts.
    kernel_row_starts = row_ends if is_decode else row_starts
    value_output = values if values is not None else logits
    if _should_use_one_block(arch, num_rows, width):
        short_rows = width <= _COMPACT_CAPACITY
        block_threads = (
            1024
            if not short_rows or num_rows <= _SHORT_ROWS_1024_THREAD_MAX_ROWS
            else 256
        )
        launcher = build_radix_topk_one_block_module(
            k,
            block_threads=block_threads,
            write_values=values is not None,
            stable=stable,
            short_rows=short_rows,
            is_decode=is_decode,
            wave_size=wave_size,
        )
        _run_compiled(
            launcher,
            logits,
            kernel_row_starts,
            row_ends,
            indices,
            value_output,
            width,
            next_n,
            num_rows,
            stream,
        )
        return

    hist_shape, state_shape = radix_topk_multi_block_workspace_shapes(num_rows, stable)
    partial_hist, state = _get_topk_workspace(
        logits.device,
        stream.cuda_stream,
        hist_shape,
        state_shape,
    )
    launcher = build_radix_topk_multi_block_module(
        k,
        stable,
        wave_size=wave_size,
        write_values=values is not None,
        is_decode=is_decode,
    )
    _run_compiled(
        launcher,
        logits,
        kernel_row_starts,
        row_ends,
        indices,
        value_output,
        partial_hist,
        state,
        width,
        next_n,
        stride0,
        num_rows,
        stream,
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
    return _flydsl_top_k_per_row(
        logits,
        None,
        seq_lens,
        indices,
        values,
        num_rows,
        stride0,
        stride1,
        k,
        stable,
        is_decode=True,
        next_n=next_n,
    )


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
) -> None:
    """Write per-row TopK indices for prefill [row_starts, row_ends) ranges."""
    return _flydsl_top_k_per_row(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride0,
        stride1,
        k,
        stable,
        is_decode=False,
        next_n=1,
    )
