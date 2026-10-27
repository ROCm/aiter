# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device TILE_MAP for the skew-compact jagged_dense_bmm kernel.

One workgroup writes the map. It fills every row with the sentinel
``(0, bm_tiles, 0, 0)``, then walks groups in chunks of ``_CHUNK``. Each
chunk loads tile counts, runs an exclusive Hillis-Steele scan in LDS, and
scatters ``(group, local_m, seq_start, seq_end)`` at the scanned row.
``_CHUNK`` is a compile-time LDS bound; ``n_groups`` is a launch argument,
so a new batch does not rebuild the kernel.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

_THREADS = 256
_CHUNK = 256
_SCAN_SHIFTS = _CHUNK.bit_length() - 1

_DEFAULT_STREAM = fx.Stream(None)


def upper_bound_tiles(total_rows: int, n_groups: int, block_m: int) -> int:
    """Host upper bound on occupied M-tiles: ``ceil(L / BLOCK_M) + n_groups``."""
    if total_rows <= 0 or block_m <= 0 or n_groups < 0:
        raise ValueError(
            f"bad tile-map bound rows={total_rows} block_m={block_m} groups={n_groups}"
        )
    return (int(total_rows) + int(block_m) - 1) // int(block_m) + int(n_groups)


def _as_torch_stream(stream, device):
    if stream is None or (isinstance(stream, fx.Stream) and stream.value is None):
        return torch.cuda.current_stream(device)
    if isinstance(stream, fx.Stream):
        raw = stream.value
        if hasattr(raw, "cuda_stream"):
            return raw
        raise TypeError(f"unsupported stream {stream!r}")
    return stream


@flyc.kernel(name="skew_tile_map_scan", known_block_size=[_THREADS, 1, 1])
def _scan_kernel(
    seq_offsets: fx.Tensor,
    tile_map: fx.Tensor,
    n_groups: fx.Int32,
    block_m: fx.Int32,
    bm_tiles: fx.Int32,
    upper_bound: fx.Int32,
):
    tid = fx.Int32(gpu.thread_idx.x)
    seq_rsrc = buffer_ops.create_buffer_resource(seq_offsets, max_size=True)
    map_rsrc = buffer_ops.create_buffer_resource(tile_map, max_size=True)
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, _CHUNK * 3, 16])
    shared = storage.peek().view(fx.make_layout(_CHUNK * 3, 1))
    tiles_base = fx.Int32(0)
    scan_base = fx.Int32(_CHUNK)
    start_base = fx.Int32(2 * _CHUNK)

    steps = (upper_bound + fx.Int32(_THREADS - 1)) // fx.Int32(_THREADS)
    for step in range(fx.Int32(0), steps, fx.Int32(1)):
        row = step * fx.Int32(_THREADS) + tid
        if row < upper_bound:
            base = row * fx.Int32(4)
            buffer_ops.buffer_store(fx.Int32(0), map_rsrc, base)
            buffer_ops.buffer_store(bm_tiles, map_rsrc, base + fx.Int32(1))
            buffer_ops.buffer_store(fx.Int32(0), map_rsrc, base + fx.Int32(2))
            buffer_ops.buffer_store(fx.Int32(0), map_rsrc, base + fx.Int32(3))
    gpu.barrier()

    n_chunks = (n_groups + fx.Int32(_CHUNK - 1)) // fx.Int32(_CHUNK)
    for chunk_i, state in range(fx.Int32(0), n_chunks, fx.Int32(1), init=[fx.Int32(0)]):
        running = state[0]
        chunk_base = fx.Int32(chunk_i) * fx.Int32(_CHUNK)
        remain = n_groups - chunk_base
        n_this = (remain < fx.Int32(_CHUNK)).select(remain, fx.Int32(_CHUNK))

        shared[tiles_base + tid] = fx.Int32(0)
        shared[start_base + tid] = fx.Int32(0)
        gpu.barrier()
        if tid < n_this:
            group = chunk_base + tid
            seq_start = buffer_ops.buffer_load(
                seq_rsrc, group, vec_width=1, dtype=T.i32
            )
            seq_end = buffer_ops.buffer_load(
                seq_rsrc, group + fx.Int32(1), vec_width=1, dtype=T.i32
            )
            length = fx.Int32(seq_end) - fx.Int32(seq_start)
            shared[tiles_base + tid] = (length + block_m - fx.Int32(1)) // block_m
            shared[start_base + tid] = fx.Int32(seq_start)
        gpu.barrier()

        shared[scan_base + tid] = shared[tiles_base + tid]
        gpu.barrier()
        for shift in range_constexpr(_SCAN_SHIFTS):
            distance = fx.Int32(1 << shift)
            take = tid >= distance
            peer = take.select(tid - distance, tid)
            added = take.select(shared[scan_base + peer], fx.Int32(0))
            updated = shared[scan_base + tid] + added
            gpu.barrier()
            shared[scan_base + tid] = updated
            gpu.barrier()

        count = shared[tiles_base + tid]
        seq_start = shared[start_base + tid]
        inclusive = shared[scan_base + tid]
        chunk_total = shared[scan_base + n_this - fx.Int32(1)]
        gpu.barrier()

        if tid < n_this:
            group = chunk_base + tid
            seq_end = buffer_ops.buffer_load(
                seq_rsrc, group + fx.Int32(1), vec_width=1, dtype=T.i32
            )
            row0 = running + inclusive - count
            for local in range(fx.Int32(0), count, fx.Int32(1)):
                base = (row0 + local) * fx.Int32(4)
                buffer_ops.buffer_store(group, map_rsrc, base)
                buffer_ops.buffer_store(local, map_rsrc, base + fx.Int32(1))
                buffer_ops.buffer_store(seq_start, map_rsrc, base + fx.Int32(2))
                buffer_ops.buffer_store(fx.Int32(seq_end), map_rsrc, base + fx.Int32(3))
        gpu.barrier()
        _running = yield [running + chunk_total]


@flyc.jit
def _launch_scan(
    seq_offsets: fx.Tensor,
    tile_map: fx.Tensor,
    n_groups: fx.Int32,
    block_m: fx.Int32,
    bm_tiles: fx.Int32,
    upper_bound: fx.Int32,
    stream: fx.Stream = _DEFAULT_STREAM,
):
    _scan_kernel(
        seq_offsets, tile_map, n_groups, block_m, bm_tiles, upper_bound
    ).launch(grid=(1, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def build_tile_map_device_fused(
    seq_offsets: torch.Tensor,
    n_groups: int,
    total_rows: int,
    max_seq_len: int,
    block_m: int = 128,
    stream=None,
) -> tuple[torch.Tensor, int]:
    """Allocate and fill ``(ub, 4)`` int32 ``(off_b, m_idx, seq_start, seq_end)``.

    The map is new every call and is filled on ``stream``. Occupied rows
    overwrite the sentinel prefix; rows past the packed tiles stay sentinels.
    """
    ub = upper_bound_tiles(total_rows, n_groups, block_m)
    bm_tiles = (int(max_seq_len) + int(block_m) - 1) // int(block_m)
    st = _as_torch_stream(stream, seq_offsets.device)
    with torch.cuda.stream(st):
        tile_map = torch.empty((ub, 4), dtype=torch.int32, device=seq_offsets.device)
        _launch_scan(
            seq_offsets,
            tile_map,
            n_groups,
            block_m,
            bm_tiles,
            ub,
            stream=st,
        )
    return tile_map, ub
