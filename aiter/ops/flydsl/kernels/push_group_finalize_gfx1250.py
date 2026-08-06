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

Parallel, one thread per local expert (single block, ``num_local_experts`` <= 1024).
Two facts make this trivially parallel with no prefix-scan / no compaction:

  * The GEMM launches a FIXED grid of ``E*tiles_per_expert`` M-tiles and skips
    padding tiles via the ``expert_ids == n_experts`` sentinel, so it never reads
    ``num_valid`` and does not need the schedule compacted. We therefore write a
    fixed, NON-compacted layout: expert ``le``'s tile ``t`` lives at meta index
    ``le*tiles_per_expert + t``; unused slots keep the host-prefilled defaults
    (tile_row_base=-1, expert_ids=E skip, tile_valid=0).
  * Each expert's counter is owned by exactly one thread, so the read-then-zero
    of ``pg_running[le]`` is race-free within the kernel (cross-rank races are
    already fenced by dispatch's end barrier + combine's barrier around it).

The previous single-thread version serialized ~E*(2+3*tiles) global mem ops behind
one lane, which was pure latency (~25 us/call at E=96); this issues them across E
threads concurrently. ``num_valid`` (unused by the GEMM, kept for the unit test)
is summed via a cheap atomic_add.
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
from aiter.ops.flydsl.dispatch_combine_v2 import flydsl_prims as P


def _make_finalize(
    *, num_local_experts, cap, tile_m, rank, experts_per_rank, emit_valid_rows=False
):
    tiles_per_expert = (cap + tile_m - 1) // tile_m
    # one thread per local expert, rounded up to a whole wavefront (single block).
    block = ((num_local_experts + WAVE - 1) // WAVE) * WAVE

    @flyc.kernel(known_block_size=[block, 1, 1])
    def finalize(
        pg_running_addr: Int64,
        tile_row_base_addr: Int64,
        expert_ids_addr: Int64,
        num_valid_addr: Int64,
        tile_valid_addr: Int64,
        valid_rows_addr: Int64,
        valid_routes_addr: Int64,
    ):
        le = fx.thread_idx.x
        if le < num_local_experts:
            rsrc_run = create_buffer_resource_from_addr(pg_running_addr)
            rsrc_trb = create_buffer_resource_from_addr(tile_row_base_addr)
            rsrc_eid = create_buffer_resource_from_addr(expert_ids_addr)
            rsrc_tv = create_buffer_resource_from_addr(tile_valid_addr)

            count = buffer_load(rsrc_run, le, vec_width=1, dtype=T.i32())
            # Read-then-zero the accumulator here (NOT in a separate host reset
            # before dispatch): finalize runs after dispatch's end cross-device
            # barrier (all peer atomic_add landings visible) and before the next
            # layer's landing (gated by combine's barrier), so zeroing it now is
            # race-free. A pre-dispatch memset instead races with peers' P2P
            # atomic_add into this same counter -> lost increments -> dropped rows.
            # Each expert is owned by exactly one thread, so this is also intra-
            # kernel race-free.
            buffer_store(arith.constant(0), rsrc_run, le)
            # clamp to cap: a token that overflowed its expert's CAP never landed.
            count = arith.select(count < cap, count, arith.constant(cap))
            # ceil(count / tile_m); tile_m is a power of two.
            num_tiles = (count + (tile_m - 1)) // tile_m
            gid = le + arith.constant(rank * experts_per_rank)
            # Tile meta layout depends on the GEMM grid strategy:
            #   * compact (emit_valid_rows): this expert claims a contiguous run of
            #     compact tile indices [tile_base, tile_base+num_tiles) via the
            #     num_valid atomic (old value is rows; /tile_m is the compact tile
            #     base since every claim adds a multiple of tile_m). The GEMM then
            #     launches a grid sized to the valid-tile upper bound instead of
            #     E*tiles_per_expert, so it skips the padded tail tiles entirely
            #     (not per-tile sentinel early-exit). This is the small-bs win.
            #   * fixed (default): expert le's tile t lives at le*tiles_per_expert+t;
            #     the GEMM launches the full grid and skips padding via the sentinel.
            # num_valid ends == total_tiles*tile_m either way (host pre-zeroes it).
            if const_expr(emit_valid_rows):
                old_rows = P.atomic_add_global(num_valid_addr, num_tiles * tile_m)
                tile_base = old_rows // tile_m
            else:
                tile_base = None
            for t in const_expr(range(tiles_per_expert)):
                active = t < num_tiles
                if active:
                    if const_expr(emit_valid_rows):
                        meta = tile_base + t
                    else:
                        meta = le * tiles_per_expert + t
                    buffer_store(le * cap + (t * tile_m), rsrc_trb, meta)
                    buffer_store(gid, rsrc_eid, meta)
                    # valid rows in THIS tile = min(tile_m, count - t*tile_m); the
                    # gemm reads it as mn_oob so padding rows (>= valid) load 0
                    # instead of stale/garbage recv data (which else -> Inf/NaN
                    # that leaks through the shared LDS store into valid rows).
                    rem = count - arith.constant(t * tile_m)
                    valid = arith.select(rem < tile_m, rem, arith.constant(tile_m))
                    buffer_store(valid, rsrc_tv, meta)
            if const_expr(not emit_valid_rows):
                # num_valid is unused by the fixed-grid GEMM (sentinel skip); kept
                # correct for the unit test via a cheap atomic (host pre-zeroes it).
                P.atomic_add_global(num_valid_addr, num_tiles * tile_m)

            # Compact list of OCCUPIED fixed-slot rows (le*cap + slot for slot <
            # count), for the route-indexed preshuffle whose grid is sized to the
            # actual landed-token count instead of the full E*cap padded grid. Each
            # expert claims a contiguous [base, base+count) range via one atomic on
            # the shared cursor (== total landed after the kernel; host pre-zeroes
            # it). Order across experts is arbitrary but each row is independent
            # (src==dst per entry), so the preshuffle output is identical to the
            # masked-grid path for every row the gemm consumes.
            if const_expr(emit_valid_rows):
                rsrc_vr = create_buffer_resource_from_addr(valid_rows_addr)
                base = P.atomic_add_global(valid_routes_addr, count)
                for w in const_expr(range(cap)):
                    if w < count:
                        buffer_store(le * cap + w, rsrc_vr, base + w)

    @flyc.jit
    def run(
        pg_running_addr: Int64,
        tile_row_base_addr: Int64,
        expert_ids_addr: Int64,
        num_valid_addr: Int64,
        tile_valid_addr: Int64,
        valid_rows_addr: Int64,
        valid_routes_addr: Int64,
        stream=fx.Stream(None),
    ):
        finalize(
            pg_running_addr,
            tile_row_base_addr,
            expert_ids_addr,
            num_valid_addr,
            tile_valid_addr,
            valid_rows_addr,
            valid_routes_addr,
        ).launch(grid=(1, 1, 1), block=[block, 1, 1], stream=stream)

    return run


_CACHE = {}


def launch_push_group_finalize(
    *,
    pg_running_ptr,
    tile_row_base_ptr,
    expert_ids_ptr,
    num_valid_ptr,
    tile_valid_ptr,
    num_local_experts,
    cap,
    tile_m,
    rank,
    experts_per_rank,
    valid_rows_ptr=None,
    valid_routes_ptr=None,
    stream=None,
):
    """Launch the finalize kernel. Pointers are raw i64 device addresses.

    When ``valid_rows_ptr``/``valid_routes_ptr`` are provided the kernel also emits
    a compact list of occupied fixed-slot rows (+ their total count) so the
    route-indexed preshuffle can size its grid to the real landed-token count
    instead of the full ``E*cap`` padded grid. ``valid_routes_ptr`` must be a
    host-pre-zeroed ``(1,)`` int32 cursor.
    """
    import torch

    emit_valid_rows = valid_rows_ptr is not None and valid_routes_ptr is not None
    key = (num_local_experts, cap, tile_m, rank, experts_per_rank, emit_valid_rows)
    run = _CACHE.get(key)
    if run is None:
        run = _make_finalize(
            num_local_experts=num_local_experts,
            cap=cap,
            tile_m=tile_m,
            rank=rank,
            experts_per_rank=experts_per_rank,
            emit_valid_rows=emit_valid_rows,
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
        int(tile_valid_ptr),
        int(valid_rows_ptr) if emit_valid_rows else 0,
        int(valid_routes_ptr) if emit_valid_rows else 0,
        st,
    )
