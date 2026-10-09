# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-launch variable-length decode TopK.

Every CTA reads its row's length. A short row is one CTA writing the final
output. A long row is split into parts: each part publishes a local TopK,
counts in on a per-row counter, and the last part to arrive merges the
candidates in the same launch. No CTA waits on another, so the grid never needs
to be co-resident. Both the part and the merge run the one-block selector.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from ..communication_ops_utils import fence_agent_acquire, fence_agent_release
from ..kernels_common import atomic_add_i32
from .radix_topk_one_block import (
    _MAX_ROW_ELEMENTS,
    _VEC,
    build_radix_topk_one_block_body,
)
from .topk_common import _row_length

_BLOCK_THREADS = 1024
_MAX_PARTS = 16
_PART_OPTIONS = (2, 4, 8, 16)
# Elements a part of a long row aims for. Stable selection costs more per
# element, so its parts are larger.
_PART_ELEMENTS = 32_768
_STABLE_PART_ELEMENTS = 65_536


def topk_per_row_decode_parts(rows: int, width: int, num_cus: int, stable: bool) -> int:
    """Parts per long row: no more than fill the CUs or than the width can feed.

    Each row still takes as many as its own length supports, at run time.
    """
    part_elements = _STABLE_PART_ELEMENTS if stable else _PART_ELEMENTS
    cu_parts = 1 << (max(1, num_cus // max(1, rows)).bit_length() - 1)
    width_parts = max(1, width // part_elements)
    width_parts = 1 << (width_parts.bit_length() - 1)
    return min(_MAX_PARTS, cu_parts, width_parts)


@cache
def build_topk_per_row_decode_module(
    k: int,
    stable: bool,
    write_values: bool,
    *,
    wave_size: int,
    arch: str,
    packed_rows: bool = False,
):
    """Build the decode kernel; the grid is rows * num_parts CTAs."""
    # Parts always publish values for the merge; the first trip gates its value
    # writes at run time, the merge at build time.
    shared_storage, run_one_block_body = build_radix_topk_one_block_body(
        k,
        _BLOCK_THREADS,
        True,
        stable,
        False,
        wave_size=wave_size,
        arch=arch,
    )
    aligned_k = (k + _VEC - 1) // _VEC * _VEC
    part_elements = max(_STABLE_PART_ELEMENTS if stable else _PART_ELEMENTS, aligned_k)
    # The selector's copy for rows no longer than k writes the final output at
    # offset 0. Only a direct row may reach it, so a part must be longer.
    if part_elements <= k:
        raise ValueError(f"k={k} leaves a part no longer than k")

    @fx.struct
    class ArrivalStorage:
        last: fx.Array[fx.Int32, 1, 16]

    @flyc.kernel(
        name=(
            f"topk_per_row_decode_k{k}_w{wave_size}"
            f"_v{int(write_values)}_s{int(stable)}"
            f"{'_packed' if packed_rows else ''}"
        ),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def topk_per_row_decode_kernel(
        input: fx.Tensor,
        row_starts: fx.Tensor,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        value_output: fx.Tensor,
        partial_indices: fx.Tensor,
        partial_values: fx.Tensor,
        counters: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
        num_parts: fx.Int32,
    ):
        tid = fx.thread_idx.x
        zero = fx.Int32(0)
        one = fx.Int32(1)
        block = fx.Int32(fx.block_idx.x)
        row = block // num_parts
        part = block - row * num_parts

        # Row geometry, all CTA-uniform. A row takes more parts only while each
        # keeps part_elements, so every part is longer than k.
        full_len = _row_length(row, seq_lens, width, next_n)
        active_parts = one
        for i in range_constexpr(len(_PART_OPTIONS)):
            option = fx.Int32(_PART_OPTIONS[i])
            target = fx.min(num_parts, option)
            enough_work = full_len >= option * fx.Int32(part_elements)
            promote = (active_parts < target) & enough_work
            active_parts = promote.select(target, active_parts)
        direct_row = active_parts == one
        chunk = (full_len // (active_parts * fx.Int32(_VEC))) * fx.Int32(_VEC)
        row_start = part * chunk
        row_end = (part == active_parts - one).select(full_len, row_start + chunk)
        active = part < active_parts
        merge_len = active_parts * fx.Int32(k)

        read_row = fx.Int32(0) if packed_rows else row
        packed_start = row_starts[row] if packed_rows else zero
        logits_row = fx.slice(input, (read_row, None))
        final_indices = fx.slice(indices, (row, None))
        final_values = fx.slice(value_output, (row, None))
        part_indices = fx.slice(partial_indices, (row, None))
        part_values = fx.slice(partial_values, (row, None))

        # One allocation serves both trips.
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(shared_storage)
        last_part = (
            allocator.allocate(ArrivalStorage).last.peek().view(fx.make_layout(1, 1))
        )
        if tid == 0:
            last_part[0] = zero
        gpu.barrier()

        def run_first_trip():
            # A direct row writes the final output; a part writes candidates.
            part_offset = part * fx.Int32(k)
            index_iter = direct_row.select(
                fx.get_iter(final_indices),
                fx.add_offset(fx.get_iter(part_indices), part_offset),
            )
            value_iter = fx.add_offset(fx.get_iter(part_values), part_offset)
            write_row_values = ~direct_row
            if const_expr(write_values):
                value_iter = direct_row.select(fx.get_iter(final_values), value_iter)
                write_row_values = True
            run_one_block_body(
                storage,
                fx.add_offset(fx.get_iter(logits_row), packed_start + row_start),
                row_end - row_start,
                fx.make_view(index_iter, part_indices.layout),
                fx.make_view(value_iter, part_values.layout),
                row_start,
                write_row_values=write_row_values,
            )

        def arrive(last_part):
            # Every wave drains its stores before lane 0 publishes the CTA.
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                fence_agent_release()
                arrived = atomic_add_i32(counters, one, row, "agent")
                is_last = arrived == active_parts - one
                if is_last:
                    # Every part has counted in: rearm for the next launch.
                    atomic_add_i32(counters, zero - active_parts, row, "agent")
                    fence_agent_acquire()
                last_part[0] = is_last.select(one, zero)
            gpu.barrier()

        def run_merge_trip():
            labels = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.get_iter(part_indices), fx.make_layout(_MAX_ROW_ELEMENTS, 1)
                ),
                num_records_bytes=fx.Int64(merge_len) * fx.Int64(4),
            )
            run_one_block_body(
                storage,
                fx.get_iter(part_values),
                merge_len,
                final_indices,
                final_values,
                zero,
                row_labels=labels,
                write_row_values=write_values,
            )

        if active:
            run_first_trip()
        if active & ~direct_row:
            arrive(last_part)
        if last_part[0] != zero:
            run_merge_trip()

    @flyc.jit
    def launch_topk_per_row_decode_kernel(
        input: fx.Tensor,
        row_starts: fx.Tensor,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        values: fx.Tensor,
        partial_indices: fx.Tensor,
        partial_values: fx.Tensor,
        counters: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
        num_parts: fx.Int32,
        blocks: fx.Int32,
        stream: fx.Stream,
    ):
        topk_per_row_decode_kernel(
            input,
            row_starts,
            seq_lens,
            indices,
            values,
            partial_indices,
            partial_values,
            counters,
            width,
            next_n,
            num_parts,
        ).launch(grid=(blocks, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_topk_per_row_decode_kernel


_WORKSPACES_ATTR = "_aiter_flydsl_topk_decode_workspaces"


def _get_workspace(indices, rows, columns):
    """Keep workspace alive exactly as long as the graph's static output.

    CUDAGraph callers already retain their input/output tensors for replay.
    An eager warmup allocates and zeros the counters once, so later capture
    records no memset. Capture without warmup is also valid: its first
    allocation records a defensive zero-fill in that graph. Different
    candidate widths receive independent counters. Reusing one output
    concurrently from multiple streams is already invalid because the final
    indices would race.
    """
    key = columns
    workspaces = getattr(indices, _WORKSPACES_ATTR, None)
    if workspaces is not None and key in workspaces:
        return workspaces[key]
    if workspaces is None:
        workspaces = {}
        setattr(indices, _WORKSPACES_ATTR, workspaces)
    # Arrival starts at zero; the last CTA subtracts active_parts to rearm it.
    workspace = (
        torch.empty((rows, columns), device=indices.device, dtype=torch.int32),
        torch.empty((rows, columns), device=indices.device, dtype=torch.float32),
        torch.zeros((rows,), device=indices.device, dtype=torch.int32),
    )
    workspaces[key] = workspace
    return workspace


def launch_topk_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    row_starts: torch.Tensor,
    indices: torch.Tensor,
    rows: int,
    width: int,
    k: int,
    stable: bool,
    values: torch.Tensor | None,
    arch: str,
    wave_size: int,
    num_cus: int,
    stream: torch.cuda.Stream,
    packed_rows: bool,
) -> None:
    parts = topk_per_row_decode_parts(rows, width, num_cus, stable)
    launcher = build_topk_per_row_decode_module(
        k,
        stable,
        values is not None,
        wave_size=wave_size,
        arch=arch,
        packed_rows=packed_rows,
    )
    partial_indices, partial_values, counters = _get_workspace(indices, rows, parts * k)
    _run_compiled(
        launcher,
        logits,
        row_starts,
        seq_lens,
        indices,
        values if values is not None else logits,
        partial_indices,
        partial_values,
        counters,
        width,
        next_n,
        parts,
        rows * parts,
        stream,
    )
