# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MoE route-gather + e8m0-scale preshuffle kernel (FlyDSL).

Background
----------
The grouped a8w4 stage1 path needs the per-token MXFP8 e8m0 scale both
*route-gathered* into the grouped per-expert layout and *preshuffled* into the
WMMA layout the masked grouped GEMM consumes. Previously this was two passes:

    1. scatter-copy   a1_scale_token_u8[tok] -> a1_scale_raw[e, m]   (row-major)
    2. preshuffle     a1_scale_raw -> grouped_a1_scale              (torch permute)

where ``preshuffle`` (``_grouped_a8w4_preshuffle_e8m0_scale``) produces the
N_Pack=2 layout::

    g = scale.view(E, -1, wmma_rep//2, 2, 16, k_groups, k_wmma_steps, 2, 2)
    g = g.permute(0, 1, 4, 5, 6, 2, 7, 8, 3).contiguous()
    grouped = g.reshape(E, max_m // wmma_rep, k_scale * wmma_rep)

Each 4-byte DWORD packs two consecutive wm tiles (N_Pack=2) with NP-innermost
byte interleaving: [asm0_k0, asm1_k0, asm0_k1, asm1_k1].
kgrp=0 and kgrp=1 are separated by 4 bytes (SCALES_PER_WMMA), which matches
the existing ``lane_kgrp * SCALES_PER_WMMA`` LDS offset.  The WMMA instruction
uses ``scaleBType = wm % 2`` (opsel) to select even/odd bytes.

This kernel fuses the two passes: it gathers each token's scale row and writes
it straight into the preshuffled layout, dropping the intermediate
``a1_scale_raw`` buffer and the separate permute launch.

Layout / index math
-------------------
``Ws = k_scale = model_dim // 32`` scale bytes per row. ``src_dwords = Ws // 4``
(one dword = 4 K-scale bytes = kgrp×kpack grid).

One work item == one ``(lane, sd, wm_pair)`` where ``wm_pair`` indexes pairs of
consecutive wm tiles:

    out_row   = tile*16 + lane
    dst_base  = e*(max_m*src_dwords) + out_row*(src_dwords*wmma_rep)
    dst_dword = dst_base + sd*wmma_rep + wm_pair*2   (kgrp=0 at +0, kgrp=1 at +1)
    grow0 = ... + wm_pair*2*16 + lane              # wm=even
    grow1 = grow0 + 16                              # wm=odd
    src0 = src[token(grow0), sd]; src1 = src[token(grow1), sd]
    store [kgrp0_dword, kgrp1_dword] at dst_dword  (dwordx2)

Grid  : (tiles_per_expert, E, 1)   -- tiles_per_expert = max_m // (wmma_rep*16)
Block : (BLOCK_THREADS, 1, 1)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import buffer_ops, vector

BLOCK_THREADS = 256


def _emit_preshuffle_dword(gather, map_rsrc, src_rsrc, grow, sd, c_src_dwords, c0):
    """Emit the load of one preshuffled source dword (grouped row ``grow``, scale
    dword ``sd``).

    This is a plain (non-``@flyc.kernel``) helper on purpose: the build-time
    ``gather`` branch lives here, NOT inside the kernel body, so the kernel AST
    rewriter never turns it into device control flow. ``gather=True`` indirects
    through ``rows_to_tokens`` (padding -> 0); ``gather=False`` reads the grouped
    row directly (identity, pure preshuffle).
    """
    i32 = T.i32
    if gather:
        srow = ArithValue(
            buffer_ops.buffer_load(map_rsrc, grow, vec_width=1, dtype=i32)
        )
        valid = arith.cmpi(CmpIPredicate.sge, srow, c0)
        # Clamp offset in-bounds when padding, then zero the result.
        src_off = arith.select(valid, srow * c_src_dwords + sd, c0)
        v_raw = buffer_ops.buffer_load(src_rsrc, src_off, vec_width=1, dtype=i32)
        return arith.select(valid, v_raw, c0)
    src_off = grow * c_src_dwords + sd
    return buffer_ops.buffer_load(src_rsrc, src_off, vec_width=1, dtype=i32)


def build_moe_scatter_copy_preshuffle_scale_module(
    row_bytes: int, wmma_rep: int, scale_k_per_tile: int, gather: bool = True
):
    """Return a JIT launcher for scale WMMA preshuffle, with optional route-gather.

    Parameters
    ----------
    row_bytes : int          scale bytes per row (``Ws = K // 32``).
    wmma_rep : int           ``warp_tile_m // 16``.
    scale_k_per_tile : int   ``tile_k // 32`` (scale bytes per k-tile).
    gather : bool            if True (stage1), gather each grouped row from a
                             source token via ``rows_to_tokens`` (-1 => pad to 0);
                             if False (stage2), the source is already grouped
                             row-major so the grouped row maps to itself (pure
                             preshuffle, like the old torch permute but in-kernel).

    Launcher signature::

        gather=True:  (src, dst, rows_to_tokens, max_m, E, tiles_per_expert, stream=...)
        gather=False: (src, dst, max_m, E, tiles_per_expert, stream=...)

    ``src`` is the scale viewed (num_src, row_bytes) uint8; ``dst`` is the
    preshuffled output viewed (E*(max_m//wmma_rep), row_bytes*wmma_rep) uint8;
    ``rows_to_tokens`` is int32 (E*max_m,) grouped row -> token (-1 skip).
    """
    assert row_bytes > 0 and row_bytes % 4 == 0, "scale row must be dword-aligned"
    assert wmma_rep >= 2 and wmma_rep % 2 == 0, "wmma_rep must be even (>= 2)"
    assert scale_k_per_tile % 4 == 0, "scale_k_per_tile must be a multiple of 4"
    assert row_bytes % scale_k_per_tile == 0, "scale_k_per_tile must divide row"

    # Compile-time tile geometry (mirrors _grouped_a8w4_preshuffle_e8m0_scale).
    #
    # NP-innermost layout: each DWORD packs 2 wm tiles (N_Pack=2).  Layout per
    # lane16 row (innermost first): N_Pack(2) x kpack(2) x kgrp(2) x wm_pair x kws x kg
    # kgrp=0/1 are separated by SCALES_PER_WMMA=4 bytes.
    #
    # One work item == one (lane, sd, wm_pair): gathers wm_even and wm_odd source
    # dwords and stores 2 output dwords (kgrp=0 and kgrp=1).
    src_dwords = row_bytes // 4               # k_groups * k_wmma_steps (dwords/row)
    rows_per_tile = wmma_rep * 16             # grouped rows per row-tile
    dpr = src_dwords * wmma_rep               # dst dwords per output row
    wm_pairs = wmma_rep // 2
    # Each work item writes 2 dwords (dwordx2 store).
    units_per_tile = 16 * src_dwords * wm_pairs

    _g = "g" if gather else "p"
    module_name = (
        f"moe_scatter_preshuffle_scale_b{row_bytes}_r{wmma_rep}_k{scale_k_per_tile}_{_g}"
    )

    @flyc.kernel(name=module_name)
    def scatter_preshuffle_kernel(
        src: fx.Tensor,  # (num_src, row_bytes) uint8
        dst: fx.Tensor,  # (E*(max_m//wmma_rep), row_bytes*wmma_rep) uint8
        rows_to_tokens: fx.Tensor,  # (E*max_m,) int32  -- -1 = skip (gather only)
        max_m: Int32,
    ):
        i32 = T.i32
        vec2_ty = ir.VectorType.get([2], i32)

        tile = ArithValue(fx.block_idx.x)
        e = ArithValue(fx.block_idx.y)
        tid = ArithValue(fx.thread_idx.x)
        max_m_i32 = ArithValue(max_m)

        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_src_dwords = arith.constant(src_dwords, type=i32)
        c_dpr = arith.constant(dpr, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_wm_pairs = arith.constant(wm_pairs, type=i32)
        c_units = arith.constant(units_per_tile, type=i32)
        c16 = arith.constant(16, type=i32)
        c0 = arith.constant(0, type=i32)
        c_0xFF = arith.constant(0xFF, type=i32)
        c_0xFF00 = arith.constant(0xFF00, type=i32)
        c_8i = arith.constant(8, type=i32)
        c_16i = arith.constant(16, type=i32)

        # Per-tile bases
        row_base = e * max_m_i32 + tile * c_rows_per_tile
        expert_dword_base = e * (max_m_i32 * c_src_dwords)
        tile_out_row0 = tile * c16

        map_rsrc = buffer_ops.create_buffer_resource(rows_to_tokens, max_size=True)
        src_rsrc = buffer_ops.create_buffer_resource(src, max_size=True)
        dst_rsrc = buffer_ops.create_buffer_resource(dst, max_size=True)

        for it in range_constexpr((units_per_tile + BLOCK_THREADS - 1) // BLOCK_THREADS):
            unit = tid + arith.constant(it * BLOCK_THREADS, type=i32)
            u_ok = arith.cmpi(CmpIPredicate.ult, unit, c_units)
            _if_u = scf.IfOp(u_ok)
            with ir.InsertionPoint(_if_u.then_block):
                # Decode (lane, sd, wm_pair) — wm_pair innermost for coalescing.
                wm_pair = unit % c_wm_pairs
                t2 = unit // c_wm_pairs
                sd = t2 % c_src_dwords
                lane = t2 // c_src_dwords
                out_row = tile_out_row0 + lane

                # dst offset: sd*wmma_rep + wm_pair*2 (kgrp=0 and kgrp=1 adjacent)
                dst_off = (
                    expert_dword_base
                    + out_row * c_dpr
                    + sd * c_wmma_rep
                    + wm_pair * arith.constant(2, type=i32)
                )

                # Gather source dwords for wm=even (wm_pair*2) and wm=odd (wm_pair*2+1).
                grow0 = row_base + wm_pair * arith.constant(2 * 16, type=i32) + lane
                grow1 = grow0 + c16
                src0 = _emit_preshuffle_dword(
                    gather, map_rsrc, src_rsrc, grow0, sd, c_src_dwords, c0
                )
                src1 = _emit_preshuffle_dword(
                    gather, map_rsrc, src_rsrc, grow1, sd, c_src_dwords, c0
                )

                # NP-innermost byte interleave into two output DWORDs:
                #   kgrp=0: [src0[0], src1[0], src0[1], src1[1]]  = [asm0k0, asm1k0, asm0k1, asm1k1]
                #   kgrp=1: [src0[2], src1[2], src0[3], src1[3]]  = [asm0k2, asm1k2, asm0k3, asm1k3]
                dst_kgrp0 = arith.ori(
                    arith.ori(arith.andi(src0, c_0xFF),
                              arith.shli(arith.andi(src1, c_0xFF), c_8i)),
                    arith.ori(arith.shli(arith.andi(src0, c_0xFF00), c_8i),
                              arith.shli(arith.andi(src1, c_0xFF00), c_16i)),
                )
                s0h = arith.shrui(src0, c_16i)
                s1h = arith.shrui(src1, c_16i)
                dst_kgrp1 = arith.ori(
                    arith.ori(arith.andi(s0h, c_0xFF),
                              arith.shli(arith.andi(s1h, c_0xFF), c_8i)),
                    arith.ori(arith.shli(arith.andi(s0h, c_0xFF00), c_8i),
                              arith.shli(arith.andi(s1h, c_0xFF00), c_16i)),
                )

                vec = vector.from_elements(vec2_ty, [dst_kgrp0, dst_kgrp1])
                buffer_ops.buffer_store(vec, dst_rsrc, dst_off)
                scf.YieldOp([])

    if gather:

        @flyc.jit
        def launch_scatter_preshuffle(
            src: fx.Tensor,
            dst: fx.Tensor,
            rows_to_tokens: fx.Tensor,
            max_m: fx.Int32,
            E: fx.Int32,
            tiles_per_expert: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                pass

            idx_tiles = arith.index_cast(T.index, tiles_per_expert)
            idx_e = arith.index_cast(T.index, E)
            launcher = scatter_preshuffle_kernel(src, dst, rows_to_tokens, max_m)
            launcher.launch(
                grid=(idx_tiles, idx_e, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_scatter_preshuffle

    @flyc.jit
    def launch_preshuffle(
        src: fx.Tensor,
        dst: fx.Tensor,
        max_m: fx.Int32,
        E: fx.Int32,
        tiles_per_expert: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_tiles = arith.index_cast(T.index, tiles_per_expert)
        idx_e = arith.index_cast(T.index, E)
        # rows_to_tokens is unused when gather=False; pass src as a placeholder.
        launcher = scatter_preshuffle_kernel(src, dst, src, max_m)
        launcher.launch(
            grid=(idx_tiles, idx_e, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_preshuffle
