"""On-device TILE_MAP prep for the skew compacted-static-grid lever.

Replaces the host oracle (skew_compact_proxy.build_tile_map) with a real device
path that produces the group-major TILE_MAP = list of (off_b, m_idx) for every
occupied M-tile, padded to a host-known upper bound with a sentinel that makes
the compact jdbba kernel early-exit. NO device->host readback of the tile count.

Design (mirrors aiter/ops/flydsl/kernels/moe_m_tile_map.py, a proven pattern):
  1. Prefix (device, no host sync): from SEQ_OFFSETS (int32, (n_groups+1,)),
       M_b     = so[b+1]-so[b]
       nt_b    = ceil(M_b/BLOCK_M).clamp(min=0)          # occupied M-tiles
       PREFIX  = [0, *cumsum(nt_b)]  (int32, group-major exclusive scan)
     Pure torch on the GPU stream: no .cpu()/.item(). n_groups is host-known
     (it is the static group count), so no data-dependent host readback.
  2. Sentinel prefill (device, no host sync): TILE_MAP (upper_bound, 2) int32
     filled with (off_b=0, m_idx=BM_TILES). BM_TILES=ceil(max_seq_len/BLOCK_M) so
     start_m = BM_TILES*BLOCK_M >= M_b for group 0 -> the compact kernel's
     `if start_m < M_b` guard skips every slack row.
  3. Scatter (FlyDSL kernel, grid=(n_groups,1,1), one block per group): block b
     loads PREFIX[b], PREFIX[b+1] -> tiles=diff, then threads stride over
     local in [0,tiles) writing TILE_MAP[(PREFIX[b]+local)*2 + 0] = b (off_b),
     TILE_MAP[(PREFIX[b]+local)*2 + 1] = local (m_idx). Empty groups have
     tiles=0 (skipped). Group-major is implicit from the group-major PREFIX.

The host launches the compact kernel with grid_x = upper_bound (host-known:
ceil(L/BLOCK_M)+n_groups, L=seq_offsets[-1] already known to allocate the packed
output); the slack rows (<= n_groups) early-exit via the sentinel.
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
    """FlyDSL scatter kernel: one block per group, scatters (off_b, m_idx) rows
    into TILE_MAP at PREFIX[b]+local. Mirrors moe_m_tile_map's map kernel but
    writes TWO int32 per compact row instead of one packed tile id."""

    @flyc.kernel(name="skew_tile_map_scatter", known_block_size=[BLOCK_THREADS, 1, 1])
    def scatter_kernel(
        PREFIX: fx.Tensor,    # (n_groups+1,) int32 exclusive scan of occupied M-tiles
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
            tiles = next_prefix - prefix  # occupied M-tiles of this group
            c_threads = arith.constant(BLOCK_THREADS, type=i32)
            c2 = arith.constant(2, type=i32)

            # Runtime trip count = ceil(tiles / BLOCK_THREADS); tiles varies per
            # block (per group), so this is a runtime scf.ForOp bound.
            tiles_idx = arith.index_cast(T.index, tiles)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            trips = (tiles_idx + arith.index(BLOCK_THREADS - 1)) // arith.index(BLOCK_THREADS)

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
                buffer_ops.buffer_store(grp, map_rsrc, base)                              # off_b
                buffer_ops.buffer_store(local_tile, map_rsrc, base + arith.constant(1, type=i32))  # m_idx
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
    """On-device exclusive prefix = sum_{g in [0,group)} ceil(M_g/block_m), where
    M_g = SEQ_OFFSETS[g+1]-SEQ_OFFSETS[g]. Runtime scf.ForOp over [0,group).
    Mirrors moe_m_tile_map._emit_prefix_sum but derives tiles from seq offsets."""
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
    s1 = buffer_ops.buffer_load(seq_rsrc, cur + arith.constant(1, type=i32), vec_width=1, dtype=i32)
    mb = ArithValue(s1) - ArithValue(s)
    tiles = (mb + arith.constant(block_m - 1, type=i32)) // arith.constant(block_m, type=i32)
    scf.YieldOp([acc + tiles])
    loop_ip.__exit__(None, None, None)
    return ArithValue(loop.results[0])


@functools.lru_cache(maxsize=None)
def _build_fused_launcher(block_m):
    """Single-kernel prep: prefix (on-device per-block scan of SEQ_OFFSETS) +
    scatter, one FlyDSL launch, NO torch prefix. grid=(n_groups,1,1). Sentinel
    slack is prefilled by the caller (torch, device) before launch."""

    @flyc.kernel(name="skew_tile_map_fused", known_block_size=[BLOCK_THREADS, 1, 1])
    def fused_kernel(
        SEQ_OFFSETS: fx.Tensor,  # (n_groups+1,) int32
        TILE_MAP: fx.Tensor,     # (upper_bound, 2) int32 -> flat 2*upper_bound
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
            prefix = _emit_prefix_sum_seq(seq_rsrc, grp, block_m)  # start row of this group
            s = buffer_ops.buffer_load(seq_rsrc, grp, vec_width=1, dtype=i32)
            s1 = buffer_ops.buffer_load(
                seq_rsrc, grp + arith.constant(1, type=i32), vec_width=1, dtype=i32
            )
            mb = ArithValue(s1) - ArithValue(s)
            tiles = (mb + arith.constant(block_m - 1, type=i32)) // arith.constant(block_m, type=i32)

            c_threads = arith.constant(BLOCK_THREADS, type=i32)
            c2 = arith.constant(2, type=i32)
            tiles_idx = arith.index_cast(T.index, tiles)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            trips = (tiles_idx + arith.index(BLOCK_THREADS - 1)) // arith.index(BLOCK_THREADS)

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
                buffer_ops.buffer_store(local_tile, map_rsrc, base + arith.constant(1, type=i32))
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


def build_tile_map_device_fused(seq_offsets, n_groups, L, max_seq_len, block_m=128, stream=None):
    """Single-FlyDSL-launch prep (+ one torch sentinel fill). No torch prefix,
    no device->host readback. Returns (tile_map (ub,2) int32, ub)."""
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
    """Device exclusive-scan of occupied M-tiles per group. Pure torch on the
    GPU stream (no host sync). Returns int32 (n_groups+1,)."""
    so = seq_offsets.to(torch.int64)
    mb = so[1:] - so[:-1]
    nt = ((mb + (block_m - 1)) // block_m).clamp(min=0)
    prefix = torch.zeros(n_groups + 1, dtype=torch.int32, device=seq_offsets.device)
    prefix[1:] = nt.cumsum(0).to(torch.int32)
    return prefix


def upper_bound_tiles(L, n_groups, block_m):
    """Host-known upper bound on occupied M-tiles: sum_b ceil(M_b/BM) < L/BM +
    n_groups. L is already host-known (caller must know it to allocate output)."""
    return (L + block_m - 1) // block_m + n_groups


def build_tile_map_device(seq_offsets, n_groups, L, max_seq_len, block_m=128, stream=None):
    """Full on-device prep. Returns (tile_map (ub,2) int32, ub, prefix).

    ub is used as grid_x (total_occ_tiles) for the compact kernel; slack rows
    hold the sentinel and early-exit. No device->host readback."""
    bm_tiles = (max_seq_len + block_m - 1) // block_m
    ub = upper_bound_tiles(L, n_groups, block_m)
    prefix = build_prefix(seq_offsets, n_groups, block_m)

    # Sentinel prefill (device): (off_b=0, m_idx=BM_TILES) -> compact kernel
    # early-exits on every row the scatter does not overwrite.
    tile_map = torch.empty((ub, 2), dtype=torch.int32, device=seq_offsets.device)
    tile_map[:, 0] = 0
    tile_map[:, 1] = bm_tiles

    launch = _build_scatter_launcher()
    st = torch.cuda.current_stream() if stream is None else stream
    launch(prefix, tile_map, int(n_groups), stream=st)
    return tile_map, ub, prefix
