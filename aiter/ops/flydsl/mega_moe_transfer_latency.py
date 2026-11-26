# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch the isolated MegaMoE transfer kernel with preallocated buffers."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.mega_moe_transfer_latency import (
    BLOCK_THREADS,
    ROWS,
    SCATTER_LDS_COLUMNS,
    transfer_kernel,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
)


@flyc.jit
def _launch(
    source: fx.Pointer,
    rowmap: fx.Pointer,
    times: fx.Pointer,
    arena_handle: fx.Int64,
    clock_ticks: fx.Int64,
    op: fx.Constexpr[str],
    wire_row: fx.Constexpr[int],
    slot_bytes: fx.Constexpr[int],
    world: fx.Constexpr[int],
    peers: fx.Constexpr[tuple],
    offsets: fx.Constexpr[tuple],
    samples: fx.Constexpr[int],
    instrument: fx.Constexpr[bool],
    empty: fx.Constexpr[bool],
    stream: fx.Stream,
):
    transfer_kernel(
        source,
        rowmap,
        times,
        arena_handle,
        clock_ticks,
        op,
        wire_row,
        slot_bytes,
        world,
        peers,
        offsets[0],
        offsets[1],
        offsets[2],
        samples,
        instrument,
        empty,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def launch_transfer(
    source,
    rowmap,
    times,
    *,
    arena_handle,
    op,
    hidden,
    world,
    peers,
    offsets,
    samples=1,
    instrument=True,
    empty=False,
    clock_ticks=0,
):
    if get_gfx_runtime() != "gfx1250":
        raise RuntimeError("MegaMoE transfer latency requires gfx1250")
    if op not in ("scatter", "dispatch", "clock"):
        raise ValueError(f"unknown transfer operation: {op}")
    if hidden < 512 or hidden % 32:
        raise ValueError("hidden must be a multiple of 32 and at least 512")
    if not 1 <= samples <= 4096:
        raise ValueError("samples must be in [1, 4096]")
    if len(peers) != ROWS or any(not 0 <= p < world for p in peers):
        raise ValueError("peers must contain eight valid destination ranks")
    if op == "dispatch" and len(set(peers)) != 1:
        raise ValueError("a dispatch run has one destination rank")
    if len(offsets) != 3 or any(o < 0 or o % 256 for o in offsets):
        raise ValueError("arena region offsets must be nonnegative and 256 B aligned")
    if op != "clock" and arena_handle <= 0:
        raise ValueError("a live registered CCO window is required")
    if op == "clock" and not 0 < clock_ticks <= 100_000_000:
        raise ValueError("clock probe requires clock_ticks in [1, 100000000]")
    for name, tensor, dtype in (
        ("source", source, torch.int32),
        ("rowmap", rowmap, torch.int64),
        ("times", times, torch.int64),
    ):
        if (
            not tensor.is_cuda
            or tensor.device != source.device
            or tensor.dtype != dtype
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous 1-D {dtype} CUDA tensor")
    if source.data_ptr() % 16 or rowmap.numel() != ROWS:
        raise ValueError("source must be 16 B aligned and rowmap must have eight pairs")
    if source.device.index != torch.cuda.current_device():
        raise ValueError("launch device must be the source tensor's device")
    required_times = samples * 2 if instrument and op != "clock" else 2
    if times.numel() < required_times:
        raise ValueError(f"times needs at least {required_times} int64 elements")
    wire_row = ((hidden // 2 + hidden // 32 + 127) // 128) * 128
    if wire_row > 65535:
        raise ValueError("wire row exceeds the TDM tile dimension field")
    source_bytes = (
        ROWS * wire_row if op == "dispatch" else ROWS * SCATTER_LDS_COLUMNS * 2
    )
    if op != "clock" and source.numel() * 4 != source_bytes:
        raise ValueError(f"source must contain exactly {source_bytes} staging bytes")
    lds_bytes = source_bytes + (samples * 16 if instrument else 0)
    if lds_bytes > get_lds_capacity_bytes(get_gfx_runtime()):
        raise ValueError(f"sampling requires {lds_bytes} LDS bytes")

    constexprs = (
        op,
        wire_row,
        1 << (hidden * 2 - 1).bit_length(),
        world,
        tuple(peers),
        tuple(offsets),
        samples,
        instrument,
        empty,
    )
    # Raw pointers are legal only for the validated layout/device above.
    # Each tensor ABI and every constexpr is included in the compiled cache key.
    key = (
        source.device.index,
        *((str(t.dtype), t.ndim, t.stride()) for t in (source, rowmap, times)),
        *constexprs,
    )
    _run_compiled(
        _launch,
        ptr_arg(source, fx.Int32),
        ptr_arg(rowmap, fx.Int64),
        ptr_arg(times, fx.Int64),
        arena_handle,
        clock_ticks,
        *constexprs,
        current_stream_arg(),
        specialization_key=key,
    )
