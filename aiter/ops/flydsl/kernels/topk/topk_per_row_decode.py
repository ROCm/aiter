# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers and launch geometry for deadlock-free FlyDSL decode TopK."""

from functools import cache, lru_cache

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .radix_topk_one_block import build_radix_topk_one_block_module
from .topk_common import (
    _DECODE_SPLIT_PART_ELEMENTS,
    _DECODE_SPLIT_STABLE_PART_ELEMENTS,
)

_MAX_PARTS = 16
ONE_WORKGROUP_MAX_ROW_WIDTH = 24_576


def topk_per_row_decode_parts(
    rows: int,
    width: int,
    num_cus: int,
    stable: bool,
) -> int:
    part_elements = (
        _DECODE_SPLIT_STABLE_PART_ELEMENTS if stable else _DECODE_SPLIT_PART_ELEMENTS
    )
    cu_parts = 1 << (max(1, num_cus // max(1, rows)).bit_length() - 1)
    width_parts = max(1, width // part_elements)
    width_parts = 1 << (width_parts.bit_length() - 1)
    return min(_MAX_PARTS, cu_parts, width_parts)


@cache
def build_topk_per_row_decode_split_modules(
    k: int,
    wave_size: int,
    arch: str,
    stable: bool,
    write_values: bool,
    parts: int,
):
    local = build_radix_topk_one_block_module(
        k,
        block_threads=1024,
        write_values=True,
        stable=stable,
        short_rows=False,
        is_decode=True,
        decode_split=True,
        write_direct_values=write_values,
        wave_size=wave_size,
        arch=arch,
    )
    merge = build_radix_topk_one_block_module(
        k,
        block_threads=1024,
        write_values=write_values,
        stable=stable,
        short_rows=parts * k <= 4096,
        is_decode=True,
        map_indices=True,
        wave_size=wave_size,
        arch=arch,
    )
    return local, merge


@lru_cache(maxsize=16)
def _get_cached_workspace(
    device: torch.device,
    stream_id: int,
    candidate_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(candidate_shape, device=device, dtype=torch.float32),
        torch.empty(candidate_shape, device=device, dtype=torch.int32),
        torch.empty((candidate_shape[0],), device=device, dtype=torch.int32),
    )


def _get_workspace(device, stream_id, candidate_shape):
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.empty(candidate_shape, device=device, dtype=torch.float32),
            torch.empty(candidate_shape, device=device, dtype=torch.int32),
            torch.empty((candidate_shape[0],), device=device, dtype=torch.int32),
        )
    return _get_cached_workspace(device, stream_id, candidate_shape)


def clear_topk_per_row_decode_workspace_cache() -> None:
    _get_cached_workspace.cache_clear()


def launch_topk_per_row_decode_split(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    rows: int,
    k: int,
    stable: bool,
    values: torch.Tensor | None,
    arch: str,
    wave_size: int,
    num_cus: int,
    stream: torch.cuda.Stream,
) -> None:
    width = logits.shape[1]
    parts = topk_per_row_decode_parts(rows, width, num_cus, stable)
    value_output = values if values is not None else logits
    if parts == 1:
        launcher = build_radix_topk_one_block_module(
            k,
            block_threads=1024,
            write_values=values is not None,
            stable=stable,
            short_rows=False,
            is_decode=True,
            wave_size=wave_size,
            arch=arch,
        )
        _run_compiled(
            launcher,
            logits,
            seq_lens,
            seq_lens,
            indices,
            value_output,
            indices,
            indices,
            value_output,
            width,
            next_n,
            1,
            rows,
            stream,
        )
        return

    local_launcher, merge_launcher = build_topk_per_row_decode_split_modules(
        k, wave_size, arch, stable, values is not None, parts
    )
    partial_shape = (rows, parts * k)
    partial_values, partial_indices, merge_lengths = _get_workspace(
        logits.device,
        stream.cuda_stream,
        partial_shape,
    )
    local_blocks = rows * parts
    # The shared launcher uses index_labels for merge lengths and direct_* for
    # the final output on rows that dynamically select one part.
    _run_compiled(
        local_launcher,
        logits,
        seq_lens,
        seq_lens,
        partial_indices,
        partial_values,
        merge_lengths,
        indices,
        value_output,
        width,
        next_n,
        parts,
        local_blocks,
        stream,
    )
    # Here merge_lengths supplies the candidate row bounds and partial_indices
    # maps selected candidate positions back to the original column IDs.
    _run_compiled(
        merge_launcher,
        partial_values,
        merge_lengths,
        merge_lengths,
        indices,
        value_output,
        partial_indices,
        indices,
        value_output,
        parts * k,
        1,
        1,
        rows,
        stream,
    )
