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

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T, Int32

BLOCK_THREADS = 256


@functools.lru_cache(maxsize=None)
def _build_scatter_launcher():
    """Scatter kernel: one block per group writes (off_b, m_idx) into TILE_MAP.

    Mirrors ``moe_m_tile_map``'s map kernel but stores two int32 per row.
    """

    @flyc.kernel(name="skew_tile_map_scatter", known_block_size=[BLOCK_THREADS, 1, 1])
    def scatter_kernel(
        PREFIX: fx.Tensor,  # (n_groups+1,) int32 exclusive scan of occupied M-tiles
        TILE_MAP: fx.Tensor,  # (upper_bound, 2) int32 -> flat 2*upper_bound
        n_groups: Int32,
    ):
        i32 = T.i32
        grp = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        prefix_rsrc = buffer_ops.create_buffer_resource(PREFIX, max_size=True)
        map_rsrc = buffer_ops.create_buffer_resource(TILE_MAP, max_size=True)

        grp_valid = arith.cmpi(CmpIPredicate.ult, grp, ArithValue(n_groups))
        if_grp = scf.IfOp(grp_valid)
        with ir.InsertionPoint(if_grp.then_block):
            prefix = buffer_ops.buffer_load(prefix_rsrc, grp, vec_width=1, dtype=i32)
            next_prefix = buffer_ops.buffer_load(
                prefix_rsrc, grp + arith.constant(1, type=i32), vec_width=1, dtype=i32
            )
            tiles = next_prefix - prefix
            c_threads = arith.constant(BLOCK_THREADS, type=i32)
            c2 = arith.constant(2, type=i32)

            # Runtime loop: trip count varies per group.
            tiles_idx = arith.index_cast(T.index, tiles)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            trips = (tiles_idx + arith.index(BLOCK_THREADS - 1)) // arith.index(
                BLOCK_THREADS
            )

            loop = scf.ForOp(c0, trips, c1)
            loop_ip = ir.InsertionPoint(loop.body)
            loop_ip.__enter__()
            it = arith.index_cast(i32, loop.induction_variable)
            local_tile = it * c_threads + tid
            tile_ok = arith.cmpi(CmpIPredicate.ult, local_tile, tiles)
            if_tile = scf.IfOp(tile_ok)
            with ir.InsertionPoint(if_tile.then_block):
                row = prefix + local_tile
                base = row * c2
                buffer_ops.buffer_store(grp, map_rsrc, base)
                buffer_ops.buffer_store(
                    local_tile, map_rsrc, base + arith.constant(1, type=i32)
                )
                scf.YieldOp([])
            scf.YieldOp([])
            loop_ip.__exit__(None, None, None)
            scf.YieldOp([])

    @flyc.jit
    def launch_scatter(
        PREFIX: fx.Tensor,
        TILE_MAP: fx.Tensor,
        n_groups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        gx = arith.index_cast(T.index, n_groups)
        scatter_kernel(PREFIX, TILE_MAP, n_groups).launch(
            grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch_scatter


def _emit_prefix_sum_seq(seq_rsrc, group, block_m):
    """Exclusive prefix of occupied M-tiles for groups ``[0, group)``.

    Mirrors ``moe_m_tile_map._emit_prefix_sum`` but derives tile counts from
    ``SEQ_OFFSETS``.
    """
    i32 = T.i32
    c0 = arith.constant(0, index=True)
    c1 = arith.constant(1, index=True)
    grp_idx = arith.index_cast(T.index, group)
    init_acc = arith.constant(0, type=i32)
    loop = scf.ForOp(c0, grp_idx, c1, [init_acc])
    loop_ip = ir.InsertionPoint(loop.body)
    loop_ip.__enter__()
    cur = arith.index_cast(i32, loop.induction_variable)
    acc = ArithValue(loop.inner_iter_args[0])
    s = buffer_ops.buffer_load(seq_rsrc, cur, vec_width=1, dtype=i32)
    s1 = buffer_ops.buffer_load(
        seq_rsrc, cur + arith.constant(1, type=i32), vec_width=1, dtype=i32
    )
    mb = ArithValue(s1) - ArithValue(s)
    tiles = (mb + arith.constant(block_m - 1, type=i32)) // arith.constant(
        block_m, type=i32
    )
    scf.YieldOp([acc + tiles])
    loop_ip.__exit__(None, None, None)
    return ArithValue(loop.results[0])


@functools.lru_cache(maxsize=None)
def _build_fused_launcher(block_m):
    """Fused prefix + scatter in one kernel launch. Caller prefills sentinels."""

    @flyc.kernel(name="skew_tile_map_fused", known_block_size=[BLOCK_THREADS, 1, 1])
    def fused_kernel(
        SEQ_OFFSETS: fx.Tensor,  # (n_groups+1,) int32
        TILE_MAP: fx.Tensor,  # (upper_bound, 2) int32 -> flat 2*upper_bound
        n_groups: Int32,
    ):
        i32 = T.i32
        grp = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        seq_rsrc = buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
        map_rsrc = buffer_ops.create_buffer_resource(TILE_MAP, max_size=True)

        grp_valid = arith.cmpi(CmpIPredicate.ult, grp, ArithValue(n_groups))
        if_grp = scf.IfOp(grp_valid)
        with ir.InsertionPoint(if_grp.then_block):
            prefix = _emit_prefix_sum_seq(seq_rsrc, grp, block_m)
            s = buffer_ops.buffer_load(seq_rsrc, grp, vec_width=1, dtype=i32)
            s1 = buffer_ops.buffer_load(
                seq_rsrc, grp + arith.constant(1, type=i32), vec_width=1, dtype=i32
            )
            mb = ArithValue(s1) - ArithValue(s)
            tiles = (mb + arith.constant(block_m - 1, type=i32)) // arith.constant(
                block_m, type=i32
            )

            c_threads = arith.constant(BLOCK_THREADS, type=i32)
            c2 = arith.constant(2, type=i32)
            tiles_idx = arith.index_cast(T.index, tiles)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            trips = (tiles_idx + arith.index(BLOCK_THREADS - 1)) // arith.index(
                BLOCK_THREADS
            )

            loop = scf.ForOp(c0, trips, c1)
            loop_ip = ir.InsertionPoint(loop.body)
            loop_ip.__enter__()
            it = arith.index_cast(i32, loop.induction_variable)
            local_tile = it * c_threads + tid
            tile_ok = arith.cmpi(CmpIPredicate.ult, local_tile, tiles)
            if_tile = scf.IfOp(tile_ok)
            with ir.InsertionPoint(if_tile.then_block):
                row = ArithValue(prefix) + local_tile
                base = row * c2
                buffer_ops.buffer_store(grp, map_rsrc, base)
                buffer_ops.buffer_store(
                    local_tile, map_rsrc, base + arith.constant(1, type=i32)
                )
                scf.YieldOp([])
            scf.YieldOp([])
            loop_ip.__exit__(None, None, None)
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        SEQ_OFFSETS: fx.Tensor,
        TILE_MAP: fx.Tensor,
        n_groups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = arith.index_cast(T.index, n_groups)
        fused_kernel(SEQ_OFFSETS, TILE_MAP, n_groups).launch(
            grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch_fused


def build_tile_map_device_fused(
    seq_offsets, n_groups, L, max_seq_len, block_m=128, stream=None
):
    """Build TILE_MAP on device. Returns ``(tile_map, upper_bound)``."""
    bm_tiles = (max_seq_len + block_m - 1) // block_m
    ub = upper_bound_tiles(L, n_groups, block_m)
    tile_map = torch.empty((ub, 2), dtype=torch.int32, device=seq_offsets.device)
    tile_map[:, 0] = 0
    tile_map[:, 1] = bm_tiles
    launch = _build_fused_launcher(block_m)
    st = torch.cuda.current_stream() if stream is None else stream
    launch(seq_offsets, tile_map, int(n_groups), stream=st)
    return tile_map, ub


def build_prefix(seq_offsets, n_groups, block_m):
    """Exclusive scan of occupied M-tiles per group. Returns int32 ``(n_groups+1,)``."""
    so = seq_offsets.to(torch.int64)
    mb = so[1:] - so[:-1]
    nt = ((mb + (block_m - 1)) // block_m).clamp(min=0)
    prefix = torch.zeros(n_groups + 1, dtype=torch.int32, device=seq_offsets.device)
    prefix[1:] = nt.cumsum(0).to(torch.int32)
    return prefix


def upper_bound_tiles(L, n_groups, block_m):
    """Host-known upper bound on occupied M-tiles."""
    return (L + block_m - 1) // block_m + n_groups


def build_tile_map_device(
    seq_offsets, n_groups, L, max_seq_len, block_m=128, stream=None
):
    """Build TILE_MAP via torch prefix + scatter. Returns ``(tile_map, ub, prefix)``."""
    bm_tiles = (max_seq_len + block_m - 1) // block_m
    ub = upper_bound_tiles(L, n_groups, block_m)
    prefix = build_prefix(seq_offsets, n_groups, block_m)

    # Sentinel prefill before scatter overwrites occupied rows.
    tile_map = torch.empty((ub, 2), dtype=torch.int32, device=seq_offsets.device)
    tile_map[:, 0] = 0
    tile_map[:, 1] = bm_tiles

    launch = _build_scatter_launcher()
    st = torch.cuda.current_stream() if stream is None else stream
    launch(prefix, tile_map, int(n_groups), stream=st)
    return tile_map, ub, prefix
