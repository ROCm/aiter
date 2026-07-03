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

where ``preshuffle`` (``_grouped_a8w4_preshuffle_e8m0_scale``) is the reshape::

    g = scale.view(E, -1, wmma_rep, 16, k_groups, k_wmma_steps, 4)
    g = g.permute(0, 1, 3, 4, 5, 2, 6).contiguous()
    grouped = g.reshape(E, max_m // wmma_rep, k_scale * wmma_rep)

i.e. for a source byte at ``(w, lane, kg, ks, kw)`` inside one row-tile of
``(wmma_rep, 16)`` rows it lands at ``(lane, kg, ks, w, kw)``. The permute is
*tile-local*: nothing crosses a ``wmma_rep*16`` row boundary.

This kernel fuses the two passes: it gathers each token's scale row and writes
it straight into the preshuffled layout, dropping the intermediate
``a1_scale_raw`` buffer and the separate permute launch.

Layout / index math
-------------------
``Ws = k_scale = model_dim // 32`` scale bytes per row. The scale row is copied
as dword (4-byte) units: ``src_dwords = Ws // 4 = k_groups * k_wmma_steps``,
where the innermost 4 (``kw``) is exactly one dword and is contiguous in *both*
source and destination.

The permute only relocates the ``wmma_rep`` axis next to the trailing ``4``, so
in the output the innermost ``(wmma_rep, 4)`` is a *contiguous* ``wmma_rep*4``-
byte block (``wmma_rep`` dwords). We give that whole block to one thread: it
gathers the ``wmma_rep`` source dwords (one per source token-row, the only
scattered part) into a register vector and issues a single ``dwordx{wmma_rep}``
store (a 16B ``dwordx4`` when ``wmma_rep == 4`` -- the widest/fastest op). The
permute is thus done in registers; the store is fully vectorized.

One thread-block handles one row-tile (``wmma_rep*16`` grouped rows) of one
expert -- a 2D grid ``(tiles_per_expert, E)``. One work item == one
``(lane, sd)`` with ``sd = kg*k_wmma_steps + ks`` in ``[0, src_dwords)``:

    out_row   = tile*16 + lane
    dst_dword = e*(max_m*src_dwords) + out_row*(src_dwords*wmma_rep) + sd*wmma_rep
    for w in range(wmma_rep):                      # the contiguous wmma_rep axis
        grow  = e*max_m + tile*(wmma_rep*16) + w*16 + lane
        srow  = rows_to_tokens[grow]               # source token (-1 => padding)
        vec[w] = (srow >= 0) ? src[srow, sd] : 0   # 0 for padding lanes
    store vec  (wmma_rep dwords) at dst_dword

Padding lanes are written as 0 (matching the zero-init reference / harmless to
the masked GEMM, which never reads padding rows). The whole output is written
once -- same write volume as the old ``contiguous()`` permute it replaces.

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

from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc, MOE_KERNARG_PRELOAD_COUNT

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
    assert wmma_rep >= 1, "wmma_rep must be >= 1"
    assert scale_k_per_tile % 4 == 0, "scale_k_per_tile must be a multiple of 4"
    assert row_bytes % scale_k_per_tile == 0, "scale_k_per_tile must divide row"

    # Compile-time tile geometry (mirrors _grouped_a8w4_preshuffle_e8m0_scale).
    src_dwords = row_bytes // 4  # k_groups * k_wmma_steps (dwords/row)
    rows_per_tile = wmma_rep * 16  # grouped rows per row-tile
    dpr = src_dwords * wmma_rep  # dst dwords per output row
    # The contiguous (wmma_rep, 4) dst block is wmma_rep dwords. Buffer ops cap at
    # dwordx4 (128b), so store it in chunks of `store_vw` (largest of {4,2,1}
    # dividing wmma_rep); wmma_rep in {1,2,4} is a single chunk, 8 -> two dwordx4.
    if wmma_rep % 4 == 0:
        store_vw = 4
    elif wmma_rep % 2 == 0:
        store_vw = 2
    else:
        store_vw = 1
    n_chunks = wmma_rep // store_vw
    # One work item == one (lane, sd, chunk): it writes `store_vw` contiguous dst
    # dwords as one dwordx{store_vw} store.
    units_per_tile = 16 * src_dwords * n_chunks

    _g = "g" if gather else "p"
    module_name = f"moe_scatter_preshuffle_scale_b{row_bytes}_r{wmma_rep}_k{scale_k_per_tile}_{_g}"

    @flyc.kernel(name=module_name)
    def scatter_preshuffle_kernel(
        src: fx.Pointer,  # (num_src, row_bytes) uint8
        dst: fx.Pointer,  # (E*(max_m//wmma_rep), row_bytes*wmma_rep) uint8
        rows_to_tokens: fx.Pointer,  # (E*max_m,) int32  -- -1 = skip (gather only)
        max_m: Int32,
    ):
        i32 = T.i32
        vec_ty = ir.VectorType.get([store_vw], i32) if store_vw > 1 else None

        tile = ArithValue(fx.block_idx.x)
        e = ArithValue(fx.block_idx.y)
        tid = ArithValue(fx.thread_idx.x)
        max_m_i32 = ArithValue(max_m)

        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_src_dwords = arith.constant(src_dwords, type=i32)
        c_dpr = arith.constant(dpr, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_n_chunks = arith.constant(n_chunks, type=i32)
        c_store_vw = arith.constant(store_vw, type=i32)
        c_units = arith.constant(units_per_tile, type=i32)
        c16 = arith.constant(16, type=i32)
        c0 = arith.constant(0, type=i32)

        # Per-tile bases (runtime e/tile/max_m, compile-time geometry).
        # grouped src-row base of this tile's first row.
        row_base = e * max_m_i32 + tile * c_rows_per_tile
        # dst dword base of this expert+tile (out_row 0 of the tile).
        # expert stride (dwords) = max_m * src_dwords; tile adds 16 out-rows.
        expert_dword_base = e * (max_m_i32 * c_src_dwords)
        tile_out_row0 = tile * c16

        # Created unconditionally (no in-body `if`): for gather=False the launcher
        # passes a placeholder for rows_to_tokens and the helper never reads it.
        map_rsrc = ptr_rsrc(rows_to_tokens)
        src_rsrc = ptr_rsrc(src)
        dst_rsrc = ptr_rsrc(dst)

        for it in range_constexpr(
            (units_per_tile + BLOCK_THREADS - 1) // BLOCK_THREADS
        ):
            unit = tid + arith.constant(it * BLOCK_THREADS, type=i32)
            u_ok = arith.cmpi(CmpIPredicate.ult, unit, c_units)
            _if_u = scf.IfOp(u_ok)
            with ir.InsertionPoint(_if_u.then_block):
                # Decode (lane, sd, chunk) with chunk innermost so consecutive
                # threads write consecutive dst dwords (coalesced).
                chunk = unit % c_n_chunks
                t2 = unit // c_n_chunks
                sd = t2 % c_src_dwords
                lane = t2 // c_src_dwords
                out_row = tile_out_row0 + lane
                w_base = chunk * c_store_vw  # first wmma_rep row of this chunk
                # col = sd*wmma_rep + chunk*store_vw  (+ j for j in [0, store_vw))
                dst_off = expert_dword_base + out_row * c_dpr + sd * c_wmma_rep + w_base

                # Collect this chunk's store_vw dwords (one per wmma_rep row). The
                # gather/identity choice is resolved in the plain helper, so no
                # build-time `if` appears inside this rewritten kernel body.
                elems = []
                for j in range_constexpr(store_vw):
                    grow = (
                        row_base + (w_base + arith.constant(j, type=i32)) * c16 + lane
                    )
                    elems.append(
                        _emit_preshuffle_dword(
                            gather, map_rsrc, src_rsrc, grow, sd, c_src_dwords, c0
                        )
                    )

                if store_vw == 1:
                    buffer_ops.buffer_store(elems[0], dst_rsrc, dst_off)
                else:
                    vec = vector.from_elements(vec_ty, elems)
                    buffer_ops.buffer_store(vec, dst_rsrc, dst_off)
                scf.YieldOp([])

    if gather:

        @flyc.jit
        def launch_scatter_preshuffle(
            src: fx.Pointer,
            dst: fx.Pointer,
            rows_to_tokens: fx.Pointer,
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

        launch_scatter_preshuffle.compile_hints = {
            "llvm_options": {
                "amdgpu-kernarg-preload": True,
                "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
            },
        }
        return launch_scatter_preshuffle

    @flyc.jit
    def launch_preshuffle(
        src: fx.Pointer,
        dst: fx.Pointer,
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

    launch_preshuffle.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_preshuffle
