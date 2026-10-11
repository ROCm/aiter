# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-launch variable-length decode TopK.

Every CTA reads its row's length. A short row is one CTA writing the final
output. A long row is split into parts: each part publishes a local TopK,
counts in on a per-row counter, and the last part to arrive merges the
candidates in the same launch. No CTA waits on another, so the grid never needs
to be co-resident. Both the part and the merge run the one-block selector.
"""

from functools import cache, lru_cache
from threading import Lock

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.utility.graph_alloc import persistent_alloc

from ..communication_ops_utils import (
    atomic_add_agent,
    fence_agent_acquire,
    fence_agent_release,
)
from .radix_topk_one_block import (
    _MAX_ROW_ELEMENTS,
    _VEC,
    build_radix_topk_one_block_body,
    build_radix_topk_one_block_module,
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
    """Build the decode kernel; the grid is rows * num_parts CTAs.

    gfx942 intentionally reuses the wave64 one-block body with
    _LONG_RADIX_DEFAULT. The standalone one-block arch allowlist is a
    performance routing policy, not a compatibility limit of this body.
    """
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
            input_iter = fx.add_offset(
                fx.get_iter(logits_row), packed_start + row_start
            )
            seg_len = row_end - row_start
            part_offset_bytes = fx.Int64(part * fx.Int32(k)) * fx.Int64(4)
            index_addr = direct_row.select(
                fx.Int64(ptrtoint(fx.get_iter(final_indices))),
                fx.Int64(ptrtoint(fx.get_iter(part_indices))) + part_offset_bytes,
            )
            value_addr = direct_row.select(
                fx.Int64(ptrtoint(fx.get_iter(final_values))),
                fx.Int64(ptrtoint(fx.get_iter(part_values))) + part_offset_bytes,
            )
            index_ptr = fx.inttoptr(
                fx.PointerType.get(
                    fx.Int32.ir_type,
                    address_space=fx.AddressSpace.Global,
                    alignment=4,
                ),
                index_addr,
            )
            value_ptr = fx.inttoptr(
                fx.PointerType.get(
                    fx.Float32.ir_type,
                    address_space=fx.AddressSpace.Global,
                    alignment=4,
                ),
                value_addr,
            )
            run_one_block_body(
                storage,
                input_iter,
                seg_len,
                fx.make_view(index_ptr, final_indices.layout),
                fx.make_view(value_ptr, final_values.layout),
                row_start,
                write_row_values=fx.Boolean(write_values) | ~direct_row,
            )

        def arrive(last_part):
            # Every wave drains its stores before lane 0 publishes the CTA.
            if const_expr(arch.startswith("gfx12")):
                rocdl.s_wait_storecnt(0)
            else:
                rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                fence_agent_release()
                counter_addr = fx.Int64(ptrtoint(fx.get_iter(counters))) + fx.Int64(
                    row
                ) * fx.Int64(4)
                arrived = fx.Int32(atomic_add_agent(counter_addr, one))
                is_last = arrived == active_parts - one
                if is_last:
                    # Every part has counted in: rearm for the next launch.
                    atomic_add_agent(counter_addr, zero - active_parts)
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


def _allocate_partials(device, rows, columns):
    return (
        torch.empty((rows, columns), device=device, dtype=torch.int32),
        torch.empty((rows, columns), device=device, dtype=torch.float32),
    )


def _allocate_workspace(device, rows, columns):
    return (
        *_allocate_partials(device, rows, columns),
        torch.zeros((rows,), device=device, dtype=torch.int32),
    )


@lru_cache(maxsize=16)
def _get_cached_workspace(device, stream_id, rows, columns):
    return _allocate_workspace(device, rows, columns)


_COUNTER_LOCK = Lock()
_COUNTER_RESERVES = {}
_CAPTURE_COUNTERS = []
_MAX_COUNTER_RESERVES = 16


def _reserve_capture_counter(device, rows, columns):
    """Keep one initialized counter ready for a later capture of this shape."""
    key = (device, rows, columns)
    with _COUNTER_LOCK:
        if _COUNTER_RESERVES.get(key):
            return
        with persistent_alloc(device):
            counter = torch.zeros((rows,), device=device, dtype=torch.int32)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
        _COUNTER_RESERVES.setdefault(key, []).append((counter, ready))
        if len(_COUNTER_RESERVES) > _MAX_COUNTER_RESERVES:
            del _COUNTER_RESERVES[next(iter(_COUNTER_RESERVES))]


def _claim_capture_counter(device, rows, columns):
    key = (device, rows, columns)
    with _COUNTER_LOCK:
        reserves = _COUNTER_RESERVES.get(key)
        if not reserves:
            return None
        workspace = reserves.pop()
        if not reserves:
            del _COUNTER_RESERVES[key]
        # A captured graph retains only raw pointers. Keep every claimed
        # counter alive permanently; each capture gets a distinct counter.
        _CAPTURE_COUNTERS.append(workspace)
    counter, ready = workspace
    torch.cuda.current_stream(device).wait_event(ready)
    return counter


def _get_workspace(device, stream_id, rows, columns):
    """Use cached eager storage and capture-private partials/counters."""
    if torch.cuda.is_current_stream_capturing():
        partials = _allocate_partials(device, rows, columns)
        counters = _claim_capture_counter(device, rows, columns)
        if counters is None:
            # Capture-first remains safe: this zero-fill becomes a Graph node.
            counters = torch.zeros((rows,), device=device, dtype=torch.int32)
        return (*partials, counters)
    workspace = _get_cached_workspace(device, stream_id, rows, columns)
    _reserve_capture_counter(device, rows, columns)
    return workspace


def clear_topk_per_row_decode_workspace_cache() -> None:
    _get_cached_workspace.cache_clear()
    with _COUNTER_LOCK:
        _COUNTER_RESERVES.clear()


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
    if parts == 1:
        launcher = build_radix_topk_one_block_module(
            k,
            block_threads=_BLOCK_THREADS,
            write_values=values is not None,
            stable=stable,
            short_rows=False,
            is_decode=True,
            wave_size=wave_size,
            arch=arch,
            packed_rows=packed_rows,
        )
        _run_compiled(
            launcher,
            logits,
            row_starts,
            seq_lens,
            indices,
            values if values is not None else logits,
            width,
            next_n,
            rows,
            stream,
        )
        return

    launcher = build_topk_per_row_decode_module(
        k,
        stable,
        values is not None,
        wave_size=wave_size,
        arch=arch,
        packed_rows=packed_rows,
    )
    partial_indices, partial_values, counters = _get_workspace(
        indices.device, stream.cuda_stream, rows, parts * k
    )
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
