# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL decode TopK interface."""

import functools

import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode import (
    build_topk_per_row_decode_module,
    topk_per_row_decode_workspace_shapes,
)
from .kernels.topk_per_row_decode_persistent import (
    create_topk_per_row_decode_tiered_kernel,
    topk_workspace_slots,
)

# The one-workgroup path wins both eager and Graph E2E at 20K, but loses to the
# multi-kernel path by 40K. Keep the cutoff at the measured crossover boundary.
_ONE_WORKGROUP_MAX_ROW_WIDTH = 20_000


@functools.cache
def _get_topk_launcher(
    rows: int,
    k: int,
    stable: bool,
):
    """Cache the dynamic-N multi-kernel launcher by compile-time dimensions."""
    return build_topk_per_row_decode_module(rows, k, stable)


@functools.cache
def _get_one_workgroup_topk_launcher(k: int):
    return create_topk_per_row_decode_tiered_kernel(
        k,
        blocks_per_row=1,
        bits_per_pass=11,
        scan_stages=2,
        tier_mode="short",
        stable=True,
    )


def _is_stream_capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _stream_key(device: torch.device) -> int:
    if _current_raw_stream is not None:
        return int(_current_raw_stream(device.index))
    return int(torch.cuda.current_stream(device).cuda_stream)


@functools.lru_cache(maxsize=16)
def _get_cached_workspace(
    device_index: int,
    stream_id: int,
    hist_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep scratch isolated by device, stream, and exact kernel layout."""
    del stream_id
    device = torch.device("cuda", device_index)
    return (
        torch.empty(hist_shape, device=device, dtype=torch.int32),
        torch.empty(state_shape, device=device, dtype=torch.int32),
    )


_persistent_workspace_cache: dict[tuple[int, int], torch.Tensor] = {}


def _get_cached_persistent_workspace(
    device_index: int,
    output_ptr: int,
) -> torch.Tensor:
    """Key scratch by static output so a warmed graph captures only TopK."""
    key = (device_index, output_ptr)
    workspace = _persistent_workspace_cache.get(key)
    if workspace is not None:
        return workspace
    workspace = torch.zeros(
        topk_workspace_slots(1, 11),
        device=torch.device("cuda", device_index),
        dtype=torch.int32,
    )
    if len(_persistent_workspace_cache) >= 16:
        _persistent_workspace_cache.pop(next(iter(_persistent_workspace_cache)))
    _persistent_workspace_cache[key] = workspace
    return workspace


def _get_persistent_workspace(
    device: torch.device,
    indices: torch.Tensor,
) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    output_ptr = indices.data_ptr()
    key = (device_index, output_ptr)
    workspace = _persistent_workspace_cache.get(key)
    if workspace is not None:
        return workspace
    if _is_stream_capturing():
        return torch.zeros(
            topk_workspace_slots(1, 11),
            device=device,
            dtype=torch.int32,
        )
    return _get_cached_persistent_workspace(device_index, output_ptr)


def _get_topk_workspace(
    device: torch.device,
    hist_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    # A graph-pool allocation must not escape capture through this process cache.
    if _is_stream_capturing():
        return (
            torch.empty(hist_shape, device=device, dtype=torch.int32),
            torch.empty(state_shape, device=device, dtype=torch.int32),
        )
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _get_cached_workspace(
        device_index,
        _stream_key(device),
        hist_shape,
        state_shape,
    )


def clear_topk_per_row_decode_workspace_cache() -> None:
    _get_cached_workspace.cache_clear()
    _persistent_workspace_cache.clear()


@functools.lru_cache(maxsize=128)
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
    if (stride0, stride1) != logits_stride:
        raise ValueError("stride0 and stride1 must match logits strides")
    if torch.cuda.get_device_properties(logits_device).warp_size != 64:
        raise ValueError("the FlyDSL decode TopK kernel requires a wave64 GPU")

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
    if indices_device != logits_device:
        raise ValueError("indices must be on the same CUDA device as logits")


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
) -> None:
    """Write per-row TopK indices using each request's effective context length.

    For output row ``r``, the valid input length is
    ``seq_lens[r // next_n] - next_n + r % next_n + 1``.
    If the valid input length is less than ``k``, the remaining output positions
    are filled with ``-1``.
    When ``stable`` is true, indices are emitted in ascending order and ties at
    the selection threshold prefer the smallest input indices.
    """
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
        indices.dtype,
        indices.device,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
    )

    rows, n = logits.shape
    use_one_workgroup = stable and n <= _ONE_WORKGROUP_MAX_ROW_WIDTH
    if use_one_workgroup:
        # One independent workgroup per row: one launch, no inter-workgroup
        # barrier, and no workspace traffic.
        launcher = _get_one_workgroup_topk_launcher(k)
        workspace = _get_persistent_workspace(logits.device, indices)
        _run_compiled(
            launcher,
            logits,
            next_n,
            seq_lens,
            indices,
            workspace,
            rows,
            stride0,
            stride1,
            torch.cuda.current_stream(),
        )
        return

    # A long-row multi-block persistent launch uses a non-cooperative grid
    # barrier. Concurrent graph replays can schedule only part of each grid and
    # deadlock all resident workgroups, so every long row uses safe launch-boundary
    # synchronization instead.
    hist_shape, state_shape = topk_per_row_decode_workspace_shapes(rows, stable)
    partial_hist, state = _get_topk_workspace(
        logits.device,
        hist_shape,
        state_shape,
    )

    launcher = _get_topk_launcher(
        rows,
        k,
        stable,
    )
    _run_compiled(
        launcher,
        logits,
        seq_lens,
        indices,
        partial_hist,
        state,
        n,
        next_n,
        stride0,
        torch.cuda.current_stream(),
    )
