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
    g = g.permute(0, 1, 4, 5, 2, 3, 6).contiguous()
    grouped = g.reshape(E, max_m // wmma_rep, k_scale * wmma_rep)

i.e. for a source byte at ``(w, lane, kg, ks, kw)`` inside one row-tile of
``(wmma_rep, 16)`` rows it lands at ``(kg, ks, w, lane, kw)``. The permute is
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

The output's innermost ``(16, 4)`` is one contiguous M16 scale block. Adjacent
threads write adjacent row dwords, so stores are coalesced and the GEMM can load
two adjacent M16 blocks with one full-wave LDS instruction.

One thread-block handles one row-tile (``wmma_rep*16`` grouped rows) of one
expert -- a 2D grid ``(tiles_per_expert, E)``. One work item is one
``(sd, w, lane)`` destination dword:

    grow      = e*max_m + tile*(wmma_rep*16) + w*16 + lane
    srow      = rows_to_tokens[grow]               # source token (-1 => padding)
    value     = (srow >= 0) ? src[srow, sd] : 0    # 0 for padding lanes
    dst_dword = tile_base + (sd*wmma_rep + w)*16 + lane

Padding lanes are written as 0 (matching the zero-init reference / harmless to
the masked GEMM, which never reads padding rows). The whole output is written
once -- same write volume as the old ``contiguous()`` permute it replaces.

Grid  : (tiles_per_expert, E, 1)   -- tiles_per_expert = max_m // (wmma_rep*16)
Block : (BLOCK_THREADS, 1, 1)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_buf_tensor,
)

BLOCK_THREADS = 256


def _emit_preshuffle_dword(gather, map_p, src_p, grow, sd, src_dwords):
    """Emit the load of one preshuffled source dword (grouped row ``grow``, scale
    dword ``sd``).

    This is a plain (non-``@flyc.kernel``) helper on purpose: the build-time
    ``gather`` branch lives here, NOT inside the kernel body, so the kernel AST
    rewriter never turns it into device control flow. ``gather=True`` indirects
    through ``rows_to_tokens`` (padding -> 0); ``gather=False`` reads the grouped
    row directly (identity, pure preshuffle).
    """
    if gather:
        srow = map_p[grow]
        # -1 marks a padding row, so the test is signed.
        valid = fx.Int32(srow) >= fx.Int32(0)
        # Clamp offset in-bounds when padding, then zero the result.
        src_off = valid.select(fx.Uint32(srow) * src_dwords + sd, fx.Uint32(0))
        v_raw = src_p[src_off]
        return valid.select(fx.Uint32(v_raw), fx.Uint32(0))
    return src_p[grow * src_dwords + sd]


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
    units_per_tile = 16 * src_dwords * wmma_rep

    _g = "g" if gather else "p"
    module_name = f"moe_scatter_preshuffle_scale_b{row_bytes}_r{wmma_rep}_k{scale_k_per_tile}_{_g}"

    @flyc.kernel(name=module_name)
    def scatter_preshuffle_kernel(
        src: fx.Pointer,  # (num_src, row_bytes) uint8
        dst: fx.Pointer,  # (E*(max_m//wmma_rep), row_bytes*wmma_rep) uint8
        rows_to_tokens: fx.Pointer,  # (E*max_m,) int32  -- -1 = skip (gather only)
        max_m: Int32,
    ):
        """Write scales as ``(E, M//(wmma_rep*16), K//128, wmma_rep, 16, 4)``."""
        # Uint32: every value here is a non-negative count/index, so `<`, `%` and
        # `//` lower to ult/remui/divui rather than their signed forms.
        tile = fx.Uint32(fx.block_idx.x)
        e = fx.Uint32(fx.block_idx.y)
        tid = fx.Uint32(fx.thread_idx.x)
        max_m_i32 = fx.Uint32(max_m)

        # Per-tile bases (runtime e/tile/max_m, compile-time geometry).
        # grouped src-row base of this tile's first row.
        row_base = e * max_m_i32 + tile * rows_per_tile
        # dst dword base of this expert+tile.
        # expert stride (dwords) = max_m * src_dwords; each tile owns one full
        # row-tile block of units_per_tile = 16 * src_dwords * wmma_rep dwords.
        expert_dword_base = e * (max_m_i32 * src_dwords)
        tile_dword_base = expert_dword_base + tile * units_per_tile

        # Created unconditionally (no in-body `if`): for gather=False the launcher
        # passes a placeholder for rows_to_tokens and the helper never reads it.
        map_p = ptr_buf_tensor(rows_to_tokens)
        src_p = ptr_buf_tensor(src)
        dst_p = ptr_buf_tensor(dst)

        for it in range_constexpr(
            (units_per_tile + BLOCK_THREADS - 1) // BLOCK_THREADS
        ):
            unit = tid + it * BLOCK_THREADS
            if unit < fx.Uint32(units_per_tile):
                lane = unit % 16
                t2 = unit // 16
                w = t2 % wmma_rep
                sd = t2 // wmma_rep
                grow = row_base + w * 16 + lane
                value = _emit_preshuffle_dword(
                    gather, map_p, src_p, grow, sd, src_dwords
                )
                dst_off = tile_dword_base + (sd * wmma_rep + w) * 16 + lane
                dst_p[dst_off] = value

    if gather:

        @flyc.jit
        def launch_scatter_preshuffle(
            src: fx.Pointer,
            dst: fx.Pointer,
            rows_to_tokens: fx.Pointer,
            max_m: fx.Int32,
            E: fx.Int32,
            tiles_per_expert: fx.Int32,
            stream: fx.Stream,
        ):
            launcher = scatter_preshuffle_kernel(src, dst, rows_to_tokens, max_m)
            launcher.launch(
                grid=(fx.Int64(tiles_per_expert), fx.Int64(E), 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        launch_scatter_preshuffle.compile_hints = {
            "llvm_options": {
                "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
                "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
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
        stream: fx.Stream,
    ):
        # rows_to_tokens is unused when gather=False; pass src as a placeholder.
        launcher = scatter_preshuffle_kernel(src, dst, src, max_m)
        launcher.launch(
            grid=(fx.Int64(tiles_per_expert), fx.Int64(E), 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_preshuffle.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_preshuffle
