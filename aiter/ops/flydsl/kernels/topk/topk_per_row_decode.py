# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers and launch geometry for deadlock-free FlyDSL decode TopK."""

from functools import cache, lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import (
    const_expr,
    gpu,
    range_constexpr,
)

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .topk_common import _row_length

_VEC = 4
_MAX_PARTS = 16
_DECODE_SPLIT_PART_ELEMENTS = 32_768
_DECODE_SPLIT_STABLE_PART_ELEMENTS = 65_536
ONE_CTA_MAX_ROW_LENGTH = 24_576


def topk_per_row_decode_parts(
    rows: int,
    width: int,
    num_cus: int,
    stable: bool,
) -> int:
    part_elements = (
        _DECODE_SPLIT_STABLE_PART_ELEMENTS if stable else _DECODE_SPLIT_PART_ELEMENTS
    )
    cu_parts = 1 << max(0, max(1, num_cus // max(1, rows)).bit_length() - 1)
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
    from .radix_topk_one_block import build_radix_topk_one_block_module

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


@cache
def build_topk_per_row_decode_geometry(k: int, stable: bool):
    aligned_k = (k + _VEC - 1) // _VEC * _VEC
    target_part_elements = (
        _DECODE_SPLIT_STABLE_PART_ELEMENTS if stable else _DECODE_SPLIT_PART_ELEMENTS
    )
    target_part_elements = max(target_part_elements, aligned_k)
    part_options = (2, 4, 8, 16)

    @flyc.jit
    def decode_geometry(
        block,
        tid,
        row_ends,
        merge_lengths,
        width,
        next_n,
        num_parts,
    ):
        zero = fx.Int32(0)
        one = fx.Int32(1)
        row = block // num_parts
        part = block % num_parts
        full_len = _row_length(row, row_ends, width, next_n)
        active_parts = one
        for i in range_constexpr(len(part_options)):
            parts = fx.Int32(part_options[i])
            target = fx.min(num_parts, parts)
            enough_work = full_len >= parts * fx.Int32(target_part_elements)
            promote = (active_parts < target) & enough_work
            active_parts = promote.select(target, active_parts)
        direct_row = active_parts == one
        if (part == zero) & (tid == zero):
            merge_lengths[row] = direct_row.select(zero, active_parts * fx.Int32(k))
        chunk = (full_len // (active_parts * fx.Int32(_VEC))) * fx.Int32(_VEC)
        row_start = part * chunk
        active = part < active_parts
        row_end = active.select(
            (part == active_parts - one).select(full_len, row_start + chunk),
            row_start,
        )
        return (
            row,
            part,
            row_start,
            row_end,
            fx.Int32(k) * part,
            active,
            direct_row,
        )

    return decode_geometry


@cache
def build_topk_per_row_decode_direct_epilogue(
    k: int,
    block_threads: int,
    write_values: bool,
):
    @flyc.jit
    def direct_epilogue(
        part,
        direct_row,
        tid,
        row_indices,
        row_values,
        direct_indices,
        direct_values,
    ):
        if direct_row & (part == 0):
            gpu.barrier()
            for step in range_constexpr((k + block_threads - 1) // block_threads):
                col = step * block_threads + tid
                if col < k:
                    direct_indices[col] = row_indices[col]
                    if const_expr(write_values):
                        direct_values[col] = row_values[col]

    return direct_epilogue


@lru_cache(maxsize=16)
def _get_cached_workspace(
    device: torch.device,
    stream_id: int,
    candidate_shape: tuple[int, ...],
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(candidate_shape, device=device, dtype=torch.float32),
        torch.empty(candidate_shape, device=device, dtype=torch.int32),
        torch.empty((rows,), device=device, dtype=torch.int32),
    )


def _get_workspace(device, stream_id, candidate_shape, rows):
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.empty(candidate_shape, device=device, dtype=torch.float32),
            torch.empty(candidate_shape, device=device, dtype=torch.int32),
            torch.empty((rows,), device=device, dtype=torch.int32),
        )
    return _get_cached_workspace(device, stream_id, candidate_shape, rows)


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
    local_launcher, merge_launcher = build_topk_per_row_decode_split_modules(
        k, wave_size, arch, stable, values is not None, parts
    )
    partial_shape = (rows, parts * k)
    partial_values, partial_indices, merge_lengths = _get_workspace(
        logits.device,
        stream.cuda_stream,
        partial_shape,
        rows,
    )
    local_blocks = rows * parts
    value_output = values if values is not None else logits
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
