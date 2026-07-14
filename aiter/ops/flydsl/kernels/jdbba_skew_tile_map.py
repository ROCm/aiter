"""Device-side TILE_MAP prep for skew jdbba launches.

Builds a group-major list of ``(group_id, m_tile_idx)`` for every occupied
M-tile, padded to a host-known upper bound. Slack rows hold a sentinel so the
compact kernel early-exits without a device->host tile-count readback.

Pattern mirrors ``aiter/ops/flydsl/kernels/moe_m_tile_map.py``:

  1. Prefix: exclusive scan of occupied M-tiles per group from ``SEQ_OFFSETS``.
  2. Sentinel prefill: unused rows get ``(off_b=0, m_idx=BM_TILES)``.
  3. Scatter: one block per group writes its occupied ``(off_b, m_idx)`` rows.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops
from flydsl.expr.typing import T

BLOCK_THREADS = 256


@flyc.jit
def _emit_prefix_sum_seq(seq_rsrc, group, block_m):
    # Intentional O(n_groups) per-block scan; tiny prep kernel.
    c_one = fx.Int32(1)
    for _cur, state in range(
        fx.Index(0), fx.Index(group), fx.Index(1), init=[fx.Int32(0)]
    ):
        acc = state[0]
        cur = fx.Int32(_cur)
        s = buffer_ops.buffer_load(seq_rsrc, cur, vec_width=1, dtype=T.i32)
        s1 = buffer_ops.buffer_load(seq_rsrc, cur + c_one, vec_width=1, dtype=T.i32)
        tiles = (s1 - s + (block_m - c_one)) // block_m
        results = yield [acc + tiles]
    return results


@flyc.jit
def _scatter_group_rows(map_rsrc, grp, tid, prefix, tiles):
    c_threads = fx.Int32(BLOCK_THREADS)
    c_one = fx.Int32(1)
    c_two = fx.Int32(2)
    trips = (tiles + fx.Int32(BLOCK_THREADS - 1)) // c_threads
    for _it in range(fx.Index(0), fx.Index(trips), fx.Index(1)):
        local_tile = fx.Int32(_it) * c_threads + tid
        if local_tile < tiles:
            base = (prefix + local_tile) * c_two
            buffer_ops.buffer_store(grp, map_rsrc, base)
            buffer_ops.buffer_store(local_tile, map_rsrc, base + c_one)


@flyc.kernel(name="skew_tile_map_fused", known_block_size=[BLOCK_THREADS, 1, 1])
def _fused_kernel(
    SEQ_OFFSETS: fx.Tensor,  # (n_groups+1,) int32
    TILE_MAP: fx.Tensor,  # (upper_bound, 2) int32 -> flat 2*upper_bound
    n_groups: fx.Int32,
    block_m: fx.Int32,
):
    grp = fx.block_idx.x
    tid = fx.thread_idx.x
    seq_rsrc = buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
    map_rsrc = buffer_ops.create_buffer_resource(TILE_MAP, max_size=True)

    if grp < n_groups:
        prefix = _emit_prefix_sum_seq(seq_rsrc, grp, block_m)
        s = buffer_ops.buffer_load(seq_rsrc, grp, vec_width=1, dtype=T.i32)
        s1 = buffer_ops.buffer_load(
            seq_rsrc, grp + fx.Int32(1), vec_width=1, dtype=T.i32
        )
        tiles = (s1 - s + (block_m - fx.Int32(1))) // block_m
        _scatter_group_rows(map_rsrc, grp, tid, prefix, tiles)


@flyc.jit
def _launch_fused(
    SEQ_OFFSETS: fx.Tensor,
    TILE_MAP: fx.Tensor,
    n_groups: fx.Int32,
    block_m: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    _fused_kernel(SEQ_OFFSETS, TILE_MAP, n_groups, block_m).launch(
        grid=(n_groups, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def build_tile_map_device_fused(
    seq_offsets, n_groups, L, max_seq_len, block_m=128, stream=None
):
    """Build TILE_MAP on device. Returns ``(tile_map, upper_bound)``."""
    bm_tiles = (max_seq_len + block_m - 1) // block_m
    ub = upper_bound_tiles(L, n_groups, block_m)
    tile_map = torch.empty((ub, 2), dtype=torch.int32, device=seq_offsets.device)
    tile_map[:, 0] = 0
    tile_map[:, 1] = bm_tiles
    st = torch.cuda.current_stream() if stream is None else stream
    _launch_fused(seq_offsets, tile_map, int(n_groups), int(block_m), stream=st)
    return tile_map, ub


def upper_bound_tiles(L, n_groups, block_m):
    """Host-known upper bound on occupied M-tiles."""
    return (L + block_m - 1) // block_m + n_groups
