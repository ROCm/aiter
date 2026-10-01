# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL decode TopK interface."""

from functools import lru_cache

import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_cu_num

from ..kernels.kernels_common import get_warp_size
from ..kernels.tensor_shim import _run_compiled
from ..kernels.topk.radix_topk_one_block import (
    _COMPACT_CAPACITY,
    _MAX_ROW_ELEMENTS,
    build_radix_topk_one_block_module,
)
from ..kernels.topk.topk_per_row_decode import (
    clear_topk_per_row_decode_workspace_cache,  # noqa: F401 -- public re-export
    launch_topk_per_row_decode,
)
from ..kernels.topk.topk_per_row_decode_persistent import (
    build_topk_per_row_decode_one_workgroup_module,
    one_workgroup_rows_are_exact,
)
from .row_lens import row_len_slot

# Widths up to this value use the persistent one-workgroup kernel.
_ONE_WORKGROUP_MAX_ROW_WIDTH = 20_000
# gfx942, k=512, width >= 8192: one scatter scan of two contiguous float4s.
# Narrower rows and every other arch keep one float4 per scan step.
_PAIR_SCATTER_MIN_WIDTH = 8192
_PAIR_SCATTER_K = 512
_SHORT_ROWS_1024_THREAD_MAX_ROWS = 256


@lru_cache(maxsize=128)
def _validate_topk_signature(
    logits_shape: torch.Size,
    logits_stride: tuple[int, ...],
    logits_dtype: torch.dtype,
    logits_device: torch.device,
    seq_lens_shape: torch.Size | None,
    seq_lens_stride: tuple[int, ...] | None,
    seq_lens_dtype: torch.dtype | None,
    seq_lens_device: torch.device | None,
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

    if next_n <= 0:
        raise ValueError("next_n must be positive")
    # A None signature is `seq_lens=None`: every row is full-width.
    if seq_lens_shape is not None:
        if len(seq_lens_shape) != 1 or seq_lens_dtype != torch.int32:
            raise ValueError("seq_lens must be a 1D int32 tensor")
        if seq_lens_stride != (1,):
            raise ValueError("seq_lens must be contiguous")
        if seq_lens_device != logits_device:
            raise ValueError("seq_lens must be on the same CUDA device as logits")
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
    seq_lens: torch.Tensor | None,
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
        *_optional_tensor_signature(seq_lens),
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


_FLYDSL_TOPK_ONE_BLOCK_ARCHES = ("gfx950", "gfx1250")


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
    width: int | None = None,
) -> None:
    """Raise if this call cannot run the one-block radix kernel."""
    _validate_flydsl_topk_call(
        logits, next_n, row_ends, indices, num_rows, stride0, stride1, k, values
    )
    if row_starts is not row_ends:
        _validate_flydsl_topk_call(
            logits, next_n, row_starts, indices, num_rows, stride0, stride1, k, values
        )
    if (logits.shape[1] if width is None else width) > _MAX_ROW_ELEMENTS:
        raise ValueError("one logits row exceeds the AMD buffer descriptor span")


_TensorSignature = tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


def _optional_tensor_signature(tensor: torch.Tensor | None) -> tuple:
    """`_tensor_signature`, or four Nones for an absent tensor."""
    return (None,) * 4 if tensor is None else _tensor_signature(tensor)


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
        get_rocm_arch(),
    )


def is_flydsl_top_k_per_row_decode_supported(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor | None,
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
        _optional_tensor_signature(seq_lens),
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
    seq_lens: torch.Tensor | None,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    values: torch.Tensor | None = None,
    *,
    row_starts: torch.Tensor | None = None,
    plane_width: int | None = None,
) -> None:
    """Write per-row TopK indices using each request's effective context length.

    ``row_starts`` enables packed rows: row r starts at row_starts[r] in the
    input's first row, while output indices remain relative to that start.
    ``plane_width`` bounds every row and selects the launch geometry.
    ``seq_lens=None`` makes every row ``plane_width`` (default: the logits
    width) long; the kernels then read no length tensor, so there is nothing
    for a captured graph to hold an address of.
    Before CUDAGraph capture, warm the split path on the capture stream so its
    stream-local arrival counter is allocated and initialized.
    """

    _validate_flydsl_topk_call(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, k, values
    )

    rows, width = logits.shape
    width = width if plane_width is None else plane_width
    if rows == 0:
        return
    # The bare arch name: the selector keys its radix layout and LDS on it.
    # FlyDSL's resolver, not aiter's: it honours FLYDSL_GPU_ARCH, which is the
    # arch this kernel is actually built for under AOT with no GPU, and falls
    # back to live hardware otherwise.
    arch = get_rocm_arch()
    wave_size = get_warp_size(arch)
    stream = torch.cuda.current_stream(logits.device)
    packed = row_starts is not None
    full_rows = seq_lens is None
    len_slot = row_len_slot(seq_lens, logits.device)
    starts = row_starts if packed else len_slot
    if width <= _ONE_WORKGROUP_MAX_ROW_WIDTH:
        pair_scatter = (
            arch.startswith("gfx942")
            and k == _PAIR_SCATTER_K
            and width >= _PAIR_SCATTER_MIN_WIDTH
        )
        launcher = build_topk_per_row_decode_one_workgroup_module(
            k,
            wave_size=wave_size,
            write_values=values is not None,
            packed_rows=packed,
            full_rows=full_rows,
            exact_rows=full_rows and one_workgroup_rows_are_exact(width),
            pair_scatter=pair_scatter,
        )
        _run_compiled(
            launcher,
            logits,
            starts,
            len_slot,
            indices,
            values if values is not None else logits,
            width,
            next_n,
            stride0,
            rows,
            stream,
        )
        return

    # Allocated width bounds every row, but a wide buffer may still hold short
    # rows: the kernel reads each row's length and splits only the long ones.
    launch_topk_per_row_decode(
        logits=logits,
        next_n=next_n,
        seq_lens=len_slot,
        row_starts=starts,
        indices=indices,
        rows=rows,
        width=width,
        k=k,
        stable=stable,
        values=values,
        arch=arch,
        wave_size=wave_size,
        num_cus=get_cu_num(),
        stream=stream,
        packed_rows=packed,
        full_rows=full_rows,
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
    plane_width: int | None = None,
    packed_rows: bool = False,
) -> None:
    """Launch the one-block radix TopK kernel for one prefill or decode call.

    With ``packed_rows`` in decode mode, row r starts at ``row_starts[r]`` in
    the input's first row, while output indices remain relative to that start.
    ``plane_width`` defaults to the logits width; it bounds every row and
    selects the same kernel variant that an unpacked plane of that width uses.
    """
    if packed_rows and (not is_decode or row_starts is None):
        raise ValueError("packed rows require decode mode and row_starts")
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
        width=plane_width,
    )
    if num_rows == 0:
        return

    arch = get_rocm_arch()
    width = logits.shape[1] if plane_width is None else plane_width
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
        packed_rows=packed_rows,
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
