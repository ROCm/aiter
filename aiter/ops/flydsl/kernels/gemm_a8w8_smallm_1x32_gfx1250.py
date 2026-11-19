# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native gfx1250 16x{32,128} MXFP8 GEMM for small M.

This is intentionally separate from the four-wave compute kernel.  One
workgroup owns one or four complete n32k4 scale groups, double-buffers only A
and B through TDM, and reads both E8M0 scale words directly into registers.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, llvm, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from .gemm_common_gfx1250 import make_lds_copy_ops, pipeline_fence
from .kernels_common import format_kernel_name
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

TILE_M = 16
TILE_K = 128
WAVE = 32
LDS_PAD_A = 16
NUM_BUFFERS = 2


@flyc.jit
def launch_gemm_a8w8_smallm_1x32(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_workspace: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldc: fx.Int32,
    num_waves: Constexpr[int],
    split_k: Constexpr[int],
    tile_n: Constexpr[int] = 32,
):
    """Launch a 16x{32,128}x128 MXFP8 kernel.

    ``split_k`` workgroups for one output tile are one hardware cluster.  Every
    rank publishes an FP32 tile, then rank zero sums ranks in increasing order
    and writes BF16.  The fixed order and the cluster barrier make graph replay
    independent of workgroup arrival order.
    """

    assert tile_n in (32, 128), f"tile_n must be 32 or 128, got {tile_n}"
    assert num_waves in (1, 2, 4), f"num_waves must be 1/2/4, got {num_waves}"
    assert tile_n // 16 % num_waves == 0, (
        f"tile_n/16 must be divisible by num_waves, got {tile_n}/{num_waves}"
    )
    assert split_k in (1, 2, 4, 8), f"split_k must be 1/2/4/8, got {split_k}"
    block = num_waves * WAVE
    n_rep = (tile_n // 16) // num_waves

    a_lds_row = TILE_K + LDS_PAD_A
    stage_a = TILE_M * a_lds_row
    b_lds_row = TILE_K * 16
    stage_b = (tile_n // 16) * b_lds_row
    stage_pitch = ((stage_a + stage_b + 1023) // 1024) * 1024
    arena_bytes = NUM_BUFFERS * stage_pitch
    tile_elems = TILE_M * tile_n

    kernel_name = format_kernel_name(
        f"gemm_a8w8_mx32_smallm_1x32_t{TILE_M}x{tile_n}x{TILE_K}"
        f"_w{num_waves}_nb{NUM_BUFFERS}_sk{split_k}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_workspace: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % WAVE
        lane16 = lane % 16
        k_group = lane // 16
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        bid_m, bid_n, k_rank = fx.block_idx
        block_m = bid_m * TILE_M
        block_n = bid_n * tile_n
        k_tiles = i32_k // (split_k * TILE_K)
        k_tile_base = k_rank * k_tiles

        arena = fx.SharedAllocator(static=False)
        arena.allocate(arena_bytes)
        lds = arena.base_ptr

        def _view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def _global_view(base, offset, shape, stride):
            return _view(fx.add_offset(base, offset), shape, stride)

        def _slot_ptr(slot):
            return fx.add_offset(lds, slot * stage_pitch)

        g_a_base = fx.recast_iter(fx.Int8, arg_a)
        g_b_base = fx.recast_iter(fx.Int8, arg_b)
        k64 = fx.Int64(i32_k)
        lda64 = fx.Int64(i32_lda)
        a_offset = fx.Int64(block_m) * lda64
        b_offset = fx.Int64(block_n // 16) * k64 * 16

        a_tile = (TILE_M, TILE_K)
        a_box_stride = (a_lds_row, 1)
        g_a = _global_view(
            g_a_base,
            a_offset,
            (TILE_M, i32_k),
            (lda64, 1),
        )
        atom_a, coord_a = rocdl.cdna5.make_tiled_tdm_atom(
            rocdl.TensorLoad(),
            g_a,
            fx.make_layout(a_tile, a_box_stride),
            a_tile,
            num_warps=1,
        )
        block_a = fx.zipped_divide(coord_a, a_tile)[None, (0, None)]

        b_tile = (tile_n // 16, TILE_K * 16)
        b_box_stride = (b_lds_row, 1)
        g_b = _global_view(
            g_b_base,
            b_offset,
            (tile_n // 16, i32_k * 16),
            (k64 * 16, 1),
        )
        atom_b, coord_b = rocdl.cdna5.make_tiled_tdm_atom(
            rocdl.TensorLoad(),
            g_b,
            fx.make_layout(b_tile, b_box_stride),
            b_tile,
            num_warps=1,
        )
        block_b = fx.zipped_divide(coord_b, b_tile)[None, (0, None)]

        def _tdm_copy(atom, block_coord, dst, tile, stride, stage_bytes, src_tile):
            source, target = rocdl.cdna5.tdm_partition(
                atom,
                0,
                fx.make_layout(1, 1),
                _view(dst, (tile, 1), (stride, stage_bytes)),
                block_coord,
            )
            fx.copy(atom, target[None, src_tile], source[None, 0])

        def issue(slot, local_k_tile):
            global_k_tile = k_tile_base + local_k_tile
            slot_ptr = _slot_ptr(slot)
            if wave == 0:
                _tdm_copy(
                    atom_a,
                    block_a,
                    slot_ptr,
                    a_tile,
                    a_box_stride,
                    stage_a,
                    global_k_tile,
                )
            b_owner = 1 if const_expr(num_waves == 2) else 0
            if wave == b_owner:
                _tdm_copy(
                    atom_b,
                    block_b,
                    fx.add_offset(slot_ptr, stage_a),
                    b_tile,
                    b_box_stride,
                    stage_b,
                    global_k_tile,
                )

        lds_load_b128, _ = make_lds_copy_ops(128)

        def _join_fragment(parts):
            lo = parts[0].shuffle(parts[1], list(range(8)))
            hi = parts[2].shuffle(parts[3], list(range(8)))
            return lo.shuffle(hi, list(range(16)))

        def load_a(slot_base):
            offset = fx.Int64(lane16 * a_lds_row + k_group * 16)
            return _join_fragment(
                [
                    Vec(lds_load_b128(slot_base, offset + j * 32))
                    for j in range_constexpr(4)
                ]
            )

        def load_b(slot_base, local_n):
            n_block = wave * n_rep + local_n
            offset = fx.Int64(
                stage_a + n_block * b_lds_row + k_group * 256 + lane16 * 16
            )
            return _join_fragment(
                [
                    Vec(lds_load_b128(slot_base, offset + j * 512))
                    for j in range_constexpr(4)
                ]
            )

        def _rmem_fragment(value):
            fragment = fx.make_rmem_tensor(16, fx.Int32)
            fragment.store(value)
            return fragment

        scale_a_i32 = fx.recast_iter(fx.Int32, arg_scale_a)
        scale_b_i32 = fx.recast_iter(fx.Int32, arg_scale_b)
        k_scale_words = i32_k // TILE_K

        def load_scale_a(global_k_tile):
            # Four consecutive E8M0 K/32 scales are one WMMA scale word.
            index = (
                fx.Int64(block_m + lane16) * fx.Int64(k_scale_words)
                + fx.Int64(global_k_tile)
            )
            return fx.Int32(fx.ptr_load(scale_a_i32 + index))

        def load_scale_b(global_k_tile, local_n):
            # n32k4 storage has one independent pack per 32 output columns.
            # A 128-column tile spans four packs, whose K strides are not
            # adjacent in memory; split the relative column into pack/lane.
            column = wave * (n_rep * 16) + local_n * 16 + lane16
            pack = column // 32
            pack_lane = column % 32
            index = (
                fx.Int64(block_n // 32 + pack) * (fx.Int64(i32_k) // 4)
                + fx.Int64(global_k_tile) * 32
                + fx.Int64(pack_lane)
            )
            return fx.Int32(fx.ptr_load(scale_b_i32 + index))

        mma = fx.make_mma_atom(
            rocdl.WMMAScale(
                16,
                16,
                128,
                fx.Float8E4M3FN,
                fx.Float8E4M3FN,
                fx.Float32,
            )
        )
        accumulators = [
            fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_rep)
        ]
        for accumulator in accumulators:
            accumulator.store(fx.full(8, 0.0, fx.Float32))

        issue(0, fx.Int32(0))
        for local_k_tile in range(k_tiles):
            slot = local_k_tile % NUM_BUFFERS
            pipeline_fence(outstanding=0, use_cluster=False)
            next_k_tile = local_k_tile + 1
            if next_k_tile < k_tiles:
                issue(next_k_tile % NUM_BUFFERS, next_k_tile)

            slot_base = fx.Int64(fx.ptrtoint(_slot_ptr(slot)))
            global_k_tile = k_tile_base + local_k_tile
            activation = _rmem_fragment(load_a(slot_base))
            scale_a = load_scale_a(global_k_tile)
            for local_n in range_constexpr(n_rep):
                weight = _rmem_fragment(load_b(slot_base, local_n))
                scale_b = load_scale_b(global_k_tile, local_n)
                accumulator = accumulators[local_n]
                fx.gemm(
                    mma,
                    accumulator,
                    weight,
                    activation,
                    accumulator,
                    scale_a=scale_b,
                    scale_b=scale_a,
                )

        ldc64 = fx.Int64(i32_ldc)
        out = fx.recast_iter(fx.BFloat16, arg_c)

        def fragment_position(local_n):
            column = (
                wave * (n_rep * 16) + local_n * 16 + k_group * 8
            )
            return lane16, column

        if const_expr(split_k == 1):
            for local_n in range_constexpr(n_rep):
                row, column = fragment_position(local_n)
                output_offset = (
                    fx.Int64(block_m + row) * ldc64 + fx.Int64(block_n + column)
                )
                fx.ptr_store(
                    accumulators[local_n].load().to(fx.BFloat16),
                    out + output_offset,
                )
        else:
            workspace = fx.recast_iter(fx.Float32, arg_workspace)
            tile_index = bid_m * (i32_n // tile_n) + bid_n
            partial_base = (
                (fx.Int64(tile_index) * split_k + fx.Int64(k_rank)) * tile_elems
            )
            for local_n in range_constexpr(n_rep):
                row, column = fragment_position(local_n)
                fx.ptr_store(
                    accumulators[local_n].load(),
                    workspace + partial_base + row * tile_n + column,
                )

            llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
            cluster.cluster_barrier()
            llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)

            if k_rank == 0:
                tile_base = fx.Int64(tile_index) * split_k * tile_elems
                for local_n in range_constexpr(n_rep):
                    row, column = fragment_position(local_n)
                    offset = row * tile_n + column
                    total = fx.ptr_load(
                        workspace + tile_base + offset,
                        result_type=T.vec(8, T.f32),
                    )
                    for rank in range_constexpr(1, split_k):
                        total = total + fx.ptr_load(
                            workspace
                            + tile_base
                            + rank * tile_elems
                            + offset,
                            result_type=T.vec(8, T.f32),
                        )
                    output_offset = (
                        fx.Int64(block_m + row) * ldc64
                        + fx.Int64(block_n + column)
                    )
                    fx.ptr_store(total.to(fx.BFloat16), out + output_offset)

    kernel(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        arg_workspace,
        i32_m,
        i32_n,
        i32_k,
        i32_lda,
        i32_ldc,
        value_attrs={
            "rocdl.cluster_dims": (
                f"1,1,{split_k}" if split_k > 1 else None
            )
        },
    ).launch(
        grid=(i32_m // TILE_M, i32_n // tile_n, split_k),
        block=(block, 1, 1),
        stream=stream,
        cluster=(1, 1, split_k) if split_k > 1 else None,
    )


launch_gemm_a8w8_smallm_1x32.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
