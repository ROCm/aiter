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
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from .tensor_shim import _run_compiled

_THREADS = 256
_CHUNK = 256
_SCAN_SHIFTS = _CHUNK.bit_length() - 1

_DEFAULT_STREAM = fx.Stream(None)


@fx.struct
class _ScanStorage:
    tiles: fx.Array[fx.Int32, _CHUNK, 16]
    scan: fx.Array[fx.Int32, _CHUNK, 16]
    starts: fx.Array[fx.Int32, _CHUNK, 16]


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
    tid = fx.thread_idx.x
    # Scalar int32 records. Offsets are element indices, so these stay on buffer_ops.
    seq_rsrc = buffer_ops.create_buffer_resource(seq_offsets, max_size=True)
    map_rsrc = buffer_ops.create_buffer_resource(tile_map, max_size=True)
    lds = fx.SharedAllocator().allocate(_ScanStorage).peek()
    tiles = lds.tiles.view(fx.make_layout(_CHUNK, 1))
    scan = lds.scan.view(fx.make_layout(_CHUNK, 1))
    starts = lds.starts.view(fx.make_layout(_CHUNK, 1))
    chunk_width = fx.Int32(_CHUNK)
    zero = fx.Int32(0)
    one = fx.Int32(1)

    steps = fx.ceildiv(upper_bound, fx.Int32(_THREADS))
    for step in range(zero, steps, one):
        row = step * _THREADS + tid
        if row < upper_bound:
            base = row * 4
            buffer_ops.buffer_store(zero, map_rsrc, base)
            buffer_ops.buffer_store(bm_tiles, map_rsrc, base + 1)
            buffer_ops.buffer_store(zero, map_rsrc, base + 2)
            buffer_ops.buffer_store(zero, map_rsrc, base + 3)
    fx.gpu.barrier()

    n_chunks = fx.ceildiv(n_groups, chunk_width)
    for chunk_i, state in range(zero, n_chunks, one, init=[zero]):
        running = state[0]
        # The carried range yields an index. Narrow it before the i32 address math.
        chunk_base = fx.Int32(chunk_i) * _CHUNK
        remain = n_groups - chunk_base
        n_this = (remain < chunk_width).select(remain, chunk_width)

        tiles[tid] = zero
        starts[tid] = zero
        fx.gpu.barrier()
        if tid < n_this:
            group = chunk_base + tid
            seq_start = fx.Int32(
                buffer_ops.buffer_load(seq_rsrc, group, vec_width=1, dtype=T.i32)
            )
            seq_end = fx.Int32(
                buffer_ops.buffer_load(seq_rsrc, group + 1, vec_width=1, dtype=T.i32)
            )
            tiles[tid] = fx.ceildiv(seq_end - seq_start, block_m)
            starts[tid] = seq_start
        fx.gpu.barrier()

        scan[tid] = tiles[tid]
        fx.gpu.barrier()
        for shift in fx.range_constexpr(_SCAN_SHIFTS):
            distance = fx.Int32(1 << shift)
            take = tid >= distance
            peer = take.select(tid - distance, tid)
            updated = scan[tid] + take.select(scan[peer], zero)
            fx.gpu.barrier()
            scan[tid] = updated
            fx.gpu.barrier()

        count = tiles[tid]
        seq_start = starts[tid]
        inclusive = scan[tid]
        chunk_total = scan[n_this - 1]
        fx.gpu.barrier()

        if tid < n_this:
            group = chunk_base + tid
            seq_end = fx.Int32(
                buffer_ops.buffer_load(seq_rsrc, group + 1, vec_width=1, dtype=T.i32)
            )
            row0 = running + inclusive - count
            for local in range(zero, count, one):
                base = (row0 + local) * 4
                buffer_ops.buffer_store(group, map_rsrc, base)
                buffer_ops.buffer_store(local, map_rsrc, base + 1)
                buffer_ops.buffer_store(seq_start, map_rsrc, base + 2)
                buffer_ops.buffer_store(seq_end, map_rsrc, base + 3)
        fx.gpu.barrier()
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
        _run_compiled(
            _launch_scan,
            seq_offsets,
            tile_map,
            n_groups,
            block_m,
            bm_tiles,
            ub,
            st,
        )
    return tile_map, ub
