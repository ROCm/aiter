# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped contiguous-M MoE GEMM for gfx1250 with a fused MXFP8 shared expert.

One launch runs the routed experts (MXFP4 weights, the caller's activation
format) and the shared expert (the last expert id, ``n_experts - 1``), which is
A8W8: MXFP8 weights and MXFP8 activations, both in their own tensors. The grid
is the shared expert's tiles followed by the routed grid of
``launch_gemm_a8w4_tdm``. Both kinds of tile run a GEMM body built by
``build_gemm_a8w4_tdm_task``: the routed body, or the MXFP8 one
(``a_is_fp4=0``, ``w_is_fp8=1``) for a shared tile.

The shared expert is a dense GEMM over the tokens: its operands are per-token
buffers (row = token), so its tiles are its own -- a tile taller than the
routed one reads the shared weights fewer times. Only its final (stage-2)
output goes into the contiguous-M layout, at the rows the route kernel gave
that expert, which the route keeps in token order. A shared tile keeps the
routed tile's workgroup size (same number of waves) and LDS residency.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T

from aiter.jit.utils.lds_capacity import get_lds_capacity_bytes

from .kernels_common import ceildiv
from .mxfp4_preshuffle_gfx1250_tdm import (
    build_gemm_a8w4_tdm_task,
    launch_gemm_a8w4_tdm,
)

WAVE = 32
# M tiles per swizzle group; must match build_gemm_a8w4_tdm_task's mapping so a
# routed tile keeps its place in the routed grid.
TILES_PER_GROUP = 16


# Widest N one wave of a shared (MXFP8) tile computes. A wave keeps one k128 of
# B in registers (16 dwords per 16 columns, two pipeline slots) next to its
# accumulators. Measured on MiniMax-M3.1 TP4 (MI455X) at 256 tokens: 64 columns
# per wave is 3 us slower than 128 (twice the tiles); a wave of 32 or 64 rows
# instead of 16 is 7 us slower (82 -> 89 us), so a taller tile takes more waves
# along M, never taller waves.
_MAX_SHARED_WARP_N = 128


def shared_tile_candidates(
    *,
    K,
    n_total,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    stage1_quant_out,
    shared_tile_m,
):
    """Shared-tile shapes ``(m_warp, n_warp, tile_n, tile_k, num_buffers)``,
    preferred first.

    A shape keeps the routed workgroup's wave count and its wave's rows (the
    A-scale buffer is laid out per routed wave), so only the wave grid changes:
    more waves along M make the tile taller, up to ``shared_tile_m`` rows --
    every token has a row of the shared expert, and a taller tile reads its
    weights fewer times. Its N is the widest that stages no more weight bytes
    per k128 than a routed tile (an MXFP8 row is twice an MXFP4 one), or the
    narrowest the waves allow.
    """
    waves = m_warp * n_warp
    unit_rows = tile_m // m_warp
    # A wave's N must hold whole MX blocks of the fused stage-1 output (64
    # gate/up columns), else whole N32 B-scale super-rows.
    n_unit = 64 if stage1_quant_out else 32
    shapes = []
    for sh_m_warp in (8, 4, 2, 1):
        if waves % sh_m_warp or (
            sh_m_warp > 1 and sh_m_warp * unit_rows > shared_tile_m
        ):
            continue
        sh_n_warp = waves // sh_m_warp
        step = sh_n_warp * n_unit
        widths = [
            w
            for w in range(step, n_total + 1, step)
            if n_total % w == 0 and w // sh_n_warp <= _MAX_SHARED_WARP_N
        ]
        if not widths:
            continue
        narrow = [w for w in widths if 2 * w <= tile_n]
        sh_tile_n = narrow[-1] if narrow else widths[0]
        for sh_tile_k in dict.fromkeys((min(tile_k, 256), 128)):
            if K % sh_tile_k:
                continue
            for sh_buffers in range(min(num_buffers, K // sh_tile_k), 1, -1):
                shapes.append((sh_m_warp, sh_n_warp, sh_tile_n, sh_tile_k, sh_buffers))
            if K // sh_tile_k == 1:
                shapes.append((sh_m_warp, sh_n_warp, sh_tile_n, sh_tile_k, 1))
    return shapes


def _pick_shared_body(build_body, routed_arena, shapes, min_residency):
    """First shape that keeps enough workgroups resident, else the best.

    LDS bounds the workgroups a CU runs at once with large tiles, and the
    kernel's arena is the larger of the two bodies'. ``min_residency`` is how
    many workgroups per CU the launch can use at all (its grid over the CUs):
    a decode GEMM1 has fewer workgroups than CUs, and there a shared tile is
    better off with the routed tile's pipeline depth than with a small arena.
    """
    lds_bytes = get_lds_capacity_bytes("gfx1250")
    if routed_arena > lds_bytes:
        raise ValueError(
            f"routed tile needs {routed_arena} B of LDS, above the {lds_bytes} B "
            "a workgroup can have"
        )
    needed = min(lds_bytes // routed_arena, max(1, min_residency))
    best = None
    for shape in shapes:
        built = build_body(1, *shape)
        residency = lds_bytes // max(routed_arena, int(built[4]["arena"]))
        if residency >= needed:
            return shape, built
        if residency and (best is None or residency > best[0]):
            best = (residency, shape, built)
    if best is None:
        raise ValueError("no shared-expert tile fits the routed workgroup")
    return best[1], best[2]


@flyc.jit
def launch_gemm_a8w4_tdm_shared_w8(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_c_shared: fx.Pointer,
    arg_a_shared: fx.Pointer,
    arg_b_shared: fx.Pointer,
    arg_scale_a_shared: fx.Pointer,
    arg_scale_b_shared: fx.Pointer,
    arg_quant_scale_shared: fx.Pointer,
    i32_m: fx.Int32,
    i32_shared_rows: fx.Int32,
    i32_shared_cap: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    n_total: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    out_is_f16: Constexpr[int],
    num_buffers: Constexpr[int],
    a_is_fp4: Constexpr[int],
    arg_m_tile_map: fx.Pointer,
    n_experts: Constexpr[int],
    stage1_act: Constexpr[int],
    f32_swiglu_limit: fx.Float32,
    row_align: Constexpr[int],
    shared_tile_m: Constexpr[int],
    min_residency: Constexpr[int],
    stage1_quant_out: Constexpr[int] = 0,
    quant_wmma_rep: Constexpr[int] = 1,
    arg_quant_scale: fx.Pointer = None,
    next_stage_prefetch: Constexpr[int] = 0,
    num_waves_per_tensor_tdm: Constexpr[int] = 2,
    tdm_b_th: Constexpr[int] = 0,
    f32_situ_beta: fx.Float32 = 1.0,
    f32_situ_linear_beta: fx.Float32 = 1.0,
    rolled_drain: Constexpr[int] = 0,
):
    """Launch the grouped GEMM with the A8W8 shared expert's tiles fused in.

    ``arg_b`` / ``arg_scale_b`` hold the routed experts (ids below
    ``n_experts - 1``). The ``*_shared`` operands hold the shared expert alone:
    its MXFP8 weights in the routed tensors' preshuffled layouts (one byte per
    weight), and its MXFP8 activations as per-token rows (``i32_shared_cap``
    rows allocated, ``i32_shared_rows`` at most in use) with the E8M0 scales in
    the layout a routed tile reads. Stage 1 (``stage1_quant_out``) writes the
    shared expert's MXFP8 output and scales to ``arg_c_shared`` /
    ``arg_quant_scale_shared``, again per token. Stage 2 writes its rows into
    ``arg_c`` where the contiguous layout holds the shared expert: from the
    last routed expert's end rounded up to ``row_align``, in token order.
    ``N`` must equal ``n_total``. ``shared_tile_m`` caps the shared tile's rows;
    ``min_residency`` is the workgroups per CU the grid can fill (see
    ``_pick_shared_body``).
    """
    assert (
        not stage1_act or stage1_quant_out
    ), "stage 1 of the fused shared expert quantizes in its epilogue"
    lds_holder = [None]
    meta_holders = ([None], [None])

    def build_body(shared, body_m_warp, body_n_warp, body_tile_n, body_tile_k, nb):
        return build_gemm_a8w4_tdm_task(
            K=K,
            tile_m=(tile_m // m_warp) * body_m_warp,
            tile_n=body_tile_n,
            tile_k=body_tile_k,
            m_warp=body_m_warp,
            n_warp=body_n_warp,
            out_is_f16=out_is_f16,
            num_buffers=nb,
            a_is_fp4=0 if shared else a_is_fp4,
            n_experts=n_experts,
            stage1_act=stage1_act,
            has_bias=0,
            stage1_quant_out=stage1_quant_out,
            quant_wmma_rep=quant_wmma_rep,
            cluster_n=1,
            cluster_m=1,
            next_stage_prefetch=next_stage_prefetch,
            num_waves_per_tensor_tdm=num_waves_per_tensor_tdm,
            tdm_as_in_prologue=0,
            tdm_b_th=tdm_b_th,
            enable_ep_scatter=0,
            ep_arena_handle=0,
            ep_combine_input_offset=0,
            ep_slot_stride_bytes=0,
            ep_destination_stride=0,
            ep_world_size=0,
            ep_quant_bits=0,
            row_major_ascale=0,
            a_row_stride_bytes=0,
            a_scale_row_stride_bytes=0,
            lds_soa_load_interleave=0,
            # Nonzero builds the inlinable body; this kernel has no worker loop.
            persistent_workers=1,
            rolled_drain=rolled_drain,
            a_preshuffled=0,
            m_align=0,
            explicit_tile=1,
            lds_base=lds_holder,
            tile_meta=meta_holders[shared],
            w_is_fp8=shared,
            single_item=1,
        )

    routed_task, block, routed_name, _, routed_layout = build_body(
        0, m_warp, n_warp, tile_n, tile_k, min(num_buffers, max(1, K // tile_k))
    )
    routed_arena = int(routed_layout["arena"])
    shared_shape, shared_built = _pick_shared_body(
        build_body,
        routed_arena,
        shared_tile_candidates(
            K=K,
            n_total=n_total,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            num_buffers=num_buffers,
            stage1_quant_out=stage1_quant_out,
            shared_tile_m=shared_tile_m,
        ),
        min_residency,
    )
    shared_task, shared_block, shared_name, _, shared_layout = shared_built
    assert shared_block == block, "both bodies must use the same workgroup"
    sh_tile_m = (tile_m // m_warp) * shared_shape[0]
    sh_tile_n = shared_shape[2]
    sh_n_tiles = n_total // sh_tile_n
    arena_bytes = max(routed_arena, int(shared_layout["arena"]))
    kname = (
        routed_name
        + "_sh"
        + shared_name.split("_e")[0].replace("mxfp4_tdm_", "")
        + f"_ra{row_align}"
    )

    def set_lds_base(ptr):
        lds_holder[0] = ptr

    def set_tile_meta(body, *meta):
        # A call, not a subscript store: a store inside a traced branch would
        # make the holder a branch result.
        meta_holders[body][0] = meta

    @flyc.kernel(name=kname, known_block_size=[block, 1, 1])
    def kernel(
        arg_m_tile_map: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_shared_rows: fx.Int32,
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_quant_scale: fx.Pointer,
        arg_c_shared: fx.Pointer,
        arg_a_shared: fx.Pointer,
        arg_b_shared: fx.Pointer,
        arg_scale_a_shared: fx.Pointer,
        arg_scale_b_shared: fx.Pointer,
        arg_quant_scale_shared: fx.Pointer,
        i32_shared_cap: fx.Int32,
        f32_swiglu_limit: fx.Float32,
        f32_situ_beta: fx.Float32,
        f32_situ_linear_beta: fx.Float32,
    ):
        tid = fx.thread_idx.x
        lane = tid % WAVE
        # One arena for both bodies: a kernel owns a single SharedAllocator.
        set_lds_base(fx.SharedAllocator(static=True).allocate(arena_bytes)._ptr)
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        tile_map = fx.recast_iter(i32_ptr, arg_m_tile_map)
        valid_rows = tile_map[n_experts - 1]
        # The shared expert's rows end the layout; routed tiles stop before.
        routed_rows = fx.Int32(0)
        if const_expr(n_experts > 1):
            routed_rows = tile_map[n_experts - 2]
        shared_grid = ceildiv(i32_shared_rows, sh_tile_m) * sh_n_tiles
        bid = fx.block_idx.x
        if bid < shared_grid:
            # Shared tiles lead the grid: they are the longest items. Their
            # rows are tokens; only the stage-2 output is in the layout.
            sh_m = bid // sh_n_tiles
            sh_n = bid - sh_m * sh_n_tiles
            shared_start = ((routed_rows + (row_align - 1)) // row_align) * row_align
            blk_m = sh_m * sh_tile_m
            shared_valid = valid_rows - shared_start
            if blk_m < shared_valid:
                if const_expr(stage1_quant_out):
                    set_tile_meta(1, fx.Int32(0), shared_valid - blk_m)
                    shared_task(
                        arg_c_shared,
                        arg_a_shared,
                        arg_b_shared,
                        arg_scale_a_shared,
                        arg_scale_b_shared,
                        arg_m_tile_map,
                        arg_a,
                        arg_quant_scale_shared,
                        arg_c,
                        i32_shared_cap,
                        i32_n,
                        f32_swiglu_limit,
                        f32_situ_beta,
                        f32_situ_linear_beta,
                        sh_m,
                        sh_n,
                    )
                else:
                    set_tile_meta(1, fx.Int32(0), shared_valid - blk_m, shared_start)
                    shared_task(
                        arg_c,
                        arg_a_shared,
                        arg_b_shared,
                        arg_scale_a_shared,
                        arg_scale_b_shared,
                        arg_m_tile_map,
                        arg_a,
                        arg_quant_scale,
                        arg_c,
                        i32_shared_cap,
                        i32_n,
                        f32_swiglu_limit,
                        f32_situ_beta,
                        f32_situ_linear_beta,
                        sh_m,
                        sh_n,
                    )
        else:
            # The DeepGEMM contiguous-M swizzle of build_gemm_a8w4_tdm_task.
            rid = bid - shared_grid
            total_n_tiles = ceildiv(i32_n, tile_n)
            total_m_tiles = ceildiv(i32_m, tile_m)
            blocks_per_group = total_n_tiles * TILES_PER_GROUP
            group = rid // blocks_per_group
            group_first_tile = group * TILES_PER_GROUP
            in_group = rid - group * blocks_per_group
            rem_tiles = total_m_tiles - group_first_tile
            group_tiles = (rem_tiles < TILES_PER_GROUP).select(
                rem_tiles, TILES_PER_GROUP
            )
            n_tile = in_group // group_tiles
            m_tile = group_first_tile + (in_group - n_tile * group_tiles)
            blk_m = m_tile * tile_m
            if blk_m < routed_rows:
                # psum is sorted: the owning expert is the count of experts
                # ending at or before blk_m, one strided load per lane.
                psum_hits = fx.Int32(0)
                for i in range_constexpr(ceildiv(n_experts, WAVE)):
                    e_idx = lane + i * WAVE
                    if const_expr((i + 1) * WAVE <= n_experts):
                        end_e = tile_map[e_idx]
                        psum_hits = psum_hits + (end_e <= blk_m).select(1, 0)
                    else:
                        e_ok = e_idx < n_experts
                        end_e = tile_map[e_ok.select(e_idx, n_experts - 1)]
                        psum_hits = psum_hits + (e_ok & (end_e <= blk_m)).select(1, 0)
                for sh in range_constexpr(int.bit_length(WAVE) - 1):
                    psum_hits = psum_hits + psum_hits.shuffle_xor(
                        fx.Int32(1 << sh), fx.Int32(WAVE)
                    )
                expert = rocdl.readfirstlane(T.i32, psum_hits)
                set_tile_meta(0, expert, tile_map[expert] - blk_m)
                routed_task(
                    arg_c,
                    arg_a,
                    arg_b,
                    arg_scale_a,
                    arg_scale_b,
                    arg_m_tile_map,
                    arg_a,
                    arg_quant_scale,
                    arg_c,
                    i32_m,
                    i32_n,
                    f32_swiglu_limit,
                    f32_situ_beta,
                    f32_situ_linear_beta,
                    m_tile,
                    n_tile,
                )

    if arg_quant_scale is None:
        arg_quant_scale = arg_c
    grid_x = ceildiv(i32_shared_rows, sh_tile_m) * sh_n_tiles + ceildiv(
        i32_m, tile_m
    ) * ceildiv(N, tile_n)
    kernel(
        arg_m_tile_map,
        i32_m,
        N,
        i32_shared_rows,
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        arg_quant_scale,
        arg_c_shared,
        arg_a_shared,
        arg_b_shared,
        arg_scale_a_shared,
        arg_scale_b_shared,
        arg_quant_scale_shared,
        i32_shared_cap,
        f32_swiglu_limit,
        f32_situ_beta,
        f32_situ_linear_beta,
    ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)


launch_gemm_a8w4_tdm_shared_w8.compile_hints["llvm_options"] = dict(
    launch_gemm_a8w4_tdm.compile_hints["llvm_options"]
)
