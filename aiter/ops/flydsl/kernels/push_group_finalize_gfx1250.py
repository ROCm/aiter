# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""push-group finalize (gfx1250): counts -> GEMM1 tile metadata.

Consumes the per-local-expert landing counts written by the push-group dispatch
kernel (``pg_running[num_local_experts]``) and emits the tile schedule the a8w4
TDM GEMM1 needs to read the fixed-slot ``[num_local_experts, CAP]`` recv grid
contiguously (no consumer-side gather):

  * ``pg_tile_row_base[meta]`` = ``le*CAP + t*tile_m``  (start row of tile ``t``
    of local expert ``le`` inside the grouped recv grid)
  * ``pg_expert_ids[meta]``    = ``rank*experts_per_rank + le`` (GLOBAL expert id)
  * ``pg_num_valid[0]``        = ``sum_le ceil(count[le]/tile_m) * tile_m``

Tail-row srcmap padding is NOT done here: the op pre-fills the whole tis region
with the drop sentinel in ``reset_push_group`` before dispatch, so any row a
token never landed in already reads sentinel (combine skips it).

Single block / single thread: ``num_local_experts`` and ``CAP/tile_m`` are small
compile-time bounds; no named barriers, no cross-CTA / cross-rank stores.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, T
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import Int32, Int64

from aiter.ops.flydsl.dispatch_combine_v2.intranode_kernels import WAVE


def _make_finalize(*, num_local_experts, cap, tile_m, rank, experts_per_rank):
    tiles_per_expert = (cap + tile_m - 1) // tile_m

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def finalize(
        pg_running_addr: Int64,
        tile_row_base_addr: Int64,
        expert_ids_addr: Int64,
        num_valid_addr: Int64,
    ):
        tid = fx.thread_idx.x
        if tid == 0:
            rsrc_run = create_buffer_resource_from_addr(pg_running_addr)
            rsrc_trb = create_buffer_resource_from_addr(tile_row_base_addr)
            rsrc_eid = create_buffer_resource_from_addr(expert_ids_addr)
            rsrc_nv = create_buffer_resource_from_addr(num_valid_addr)

            tile_ctr = arith.constant(0)
            for le in const_expr(range(num_local_experts)):
                count = buffer_load(rsrc_run, le, vec_width=1, dtype=T.i32())
                # ceil(count / tile_m); tile_m is a power of two.
                num_tiles = (count + (tile_m - 1)) // tile_m
                gid = arith.constant(rank * experts_per_rank + le)
                for t in const_expr(range(tiles_per_expert)):
                    active = t < num_tiles
                    if active:
                        buffer_store(
                            arith.constant(le * cap + t * tile_m), rsrc_trb, tile_ctr
                        )
                        buffer_store(gid, rsrc_eid, tile_ctr)
                    tile_ctr = tile_ctr + arith.select(
                        active, arith.constant(1), arith.constant(0)
                    )
            buffer_store(tile_ctr * tile_m, rsrc_nv, 0)

    @flyc.jit
    def run(
        pg_running_addr: Int64,
        tile_row_base_addr: Int64,
        expert_ids_addr: Int64,
        num_valid_addr: Int64,
        stream=fx.Stream(None),
    ):
        finalize(
            pg_running_addr,
            tile_row_base_addr,
            expert_ids_addr,
            num_valid_addr,
        ).launch(grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream)

    return run


_CACHE = {}


def launch_push_group_finalize(
    *,
    pg_running_ptr,
    tile_row_base_ptr,
    expert_ids_ptr,
    num_valid_ptr,
    num_local_experts,
    cap,
    tile_m,
    rank,
    experts_per_rank,
    stream=None,
):
    """Launch the finalize kernel. Pointers are raw i64 device addresses."""
    import torch

    key = (num_local_experts, cap, tile_m, rank, experts_per_rank)
    run = _CACHE.get(key)
    if run is None:
        run = _make_finalize(
            num_local_experts=num_local_experts,
            cap=cap,
            tile_m=tile_m,
            rank=rank,
            experts_per_rank=experts_per_rank,
        )
        _CACHE[key] = run

    st = (
        fx.Stream(stream)
        if stream is not None
        else fx.Stream(torch.cuda.current_stream())
    )
    run(
        int(pg_running_ptr),
        int(tile_row_base_ptr),
        int(expert_ids_ptr),
        int(num_valid_ptr),
        st,
    )
