# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B008
# The FlyDSL body uses a typed Stream default for its tracing semantics.

"""gfx942 FP8 blockscale GEMM: 2x4 wave grid, A and B staged through LDS, pf1.

C[M, N] = A[M, K] @ B[N, K]^T with FP8 E4M3FNUZ A and B, FP32 block scales
ScaleA[K/128, M] (x_scale transposed) and ScaleB[ceil(N/128), K/128], and BF16
or FP16 output. B is plain [N, K] or aiter shuffle_weight(B, layout=(16, 16)).
Needs N % 16 == 0, K % 128 == 0 and K >= 128; any M >= 1 works: the raw-buffer
range checks zero rows and columns past M/N, rows past M are outside the C
descriptor and columns past N store at voffset 0x7FFFFFFF, which is dropped.

One 256x256 tile per workgroup of 512 threads, 64 KiB LDS. wave_m = wave // 4
owns 128 rows, wave_n = wave % 4 owns 64 columns. A and B are both copied
cooperatively into 32 KiB of LDS each (four 64-row packets). B operands are
read once per K block; A packet p is read in stage p. Each packet of the next
K block is loaded during this block and written to LDS one stage later
(one-stage load lead, "pf1"). Waves 4-7 trail waves 0-3 by one barrier.

The main loop is ported unchanged from PyHIP
src/contrib/flydsl/gemm_a8w8_blockscale_gfx942_2x4_lds_pf1.py, the FlyDSL form
of PyHIP's gemm_fp8_gfx942_4x2_lds.py (which names the grid in gfx950's (N, M)
order): tile order, LDS layout and swizzle, load/ds_write/ds_read order, every
wait, barrier, sched_barrier and s_setprio, and the MFMA/dequant order are
PyHIP's. Only the helpers are inlined and the kernel symbol is named:
PyHIP's BufferTensor.load is _buffer_load_dwordx4 below, the same
buffer_load_dwordx4 with byte voffset and soffset. The raw rocdl/llvm calls
below are kept for that reason; each names the stable API it stands in for.

The interface is the gfx950 8-wave kernel's (gemm_a8w8_blockscale_8wave.py):
compile_gemm_fp8_gfx942_2x4_lds_pf1(TILE_M, TILE_N, TILE_K, N, K, preshuffle_b)
returns launch_gemm(A, B, C, scaleA, scaleB, M, stream), which takes tensors
and rebases A, B and C per tile in i64; C's dtype selects the epilogue.

At K = 128 (one K block) the two inline-asm s_buffer_load_dword of the next
block's B scales still load values nothing reads (PyHIP design note F3).
PyHIP's build loads both into one SGPR, which its ISA audit flags as a WAW;
here FlyDSL 0.5.3 gives them separate SGPRs and the audit passes.

The epilogue is aiter's: it rounds the FP32 results to BF16 or FP16 to
nearest even, as the gfx950 8-wave kernel (gemm_a8w8_blockscale_8wave.py)
does. PyHIP instead truncates to BF16 and has no FP16 output, so BF16 results
differ from PyHIP's in the last bit wherever rounding goes up.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.kernels_common import ceildiv

TILE = 256
TILE_K = 128
PACKET_ROWS = 64
PACKET_DWORDS = PACKET_ROWS * TILE_K // 4


def _lds_dwords(row, column):
    # XOR swizzle of the 16B column keeps MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


def _tb_swizzle(block_id, M, N, tile_m, tile_n, m01=8, group_num=8):
    """WG id -> (block_m, block_n), as pyhip.contrib.common.loaders.tb_swizzle."""
    m0 = (M + tile_m - 1) // tile_m
    n0 = (N + tile_n - 1) // tile_n
    group_size = (m0 * n0 + group_num - 1) // group_num
    big_group_num = group_num - (group_size * group_num - m0 * n0)
    group_id_y = block_id // group_num
    group_id_x = block_id - group_id_y * group_num
    remap = group_id_x * group_size + group_id_y
    remap = (group_id_x > big_group_num).select(
        remap + (big_group_num - group_id_x), remap
    )
    idx_m0 = remap // n0
    idx_n0 = remap - idx_m0 * n0
    m0_mod_m01 = m0 - m0 // m01 * m01
    m01_adapt = (idx_m0 < m0 - m0_mod_m01).select(fx.Int32(m01), m0_mod_m01)
    idx_m00 = idx_m0 // m01
    local = idx_n0 + (idx_m0 - idx_m00 * m01) * n0
    block_n = local // m01_adapt
    return local - block_n * m01_adapt + idx_m00 * m01, block_n


def _buffer_load_dwordx4(rsrc, voffset_bytes, soffset_bytes):
    """One buffer_load_dwordx4 ... offen of four i32 at byte voffset + soffset."""
    # Raw load, not fx.copy(BufferCopy128b): a copy atom addresses a layout
    # coordinate, while this pipeline issues PyHIP's precomputed tile-relative
    # voffset (row offsets, range checked so rows past M/N read zero) with the
    # K-block offset in soffset; the raw op keeps PyHIP's instruction.
    return Vec(
        rocdl.raw_ptr_buffer_load(
            T.i32x4,
            rsrc,
            fx.Int32(voffset_bytes).ir_value(),
            fx.Int32(soffset_bytes).ir_value(),
            0,
        )
    )


def _s_buffer_load_f32(rsrc, offset_bytes):
    """Scalar s_buffer_load_dword hidden from LLVM; wait lgkmcnt(0) before use."""
    # Inline asm, not tensor_shim.buf_scalar_load: the tracked SMEM intrinsic
    # lacks a memory operand in this LLVM and gets conservative waits. Every use
    # follows the pipeline's own lgkmcnt(0) waits.
    value = llvm.inline_asm(
        T.f32,
        [rsrc, fx.Int32(offset_bytes).ir_value()],
        "s_buffer_load_dword $0, $1, $2",
        "=s,s,s",
        has_side_effects=True,
    )
    return fx.Float32(value)


def compile_gemm_fp8_gfx942_2x4_lds_pf1(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    preshuffle_b=False,
):
    """Compile the 2x4-LDS pf1 blockscale kernel for one (N, K, B layout).

    ``launch_gemm(A, B, C, scaleA, scaleB, M, stream)`` takes A[M, K] and B
    (plain [N, K] or shuffle_weight(B, (16, 16))) as FP8 bytes, C[M, N] in BF16
    or FP16, and FP32 ScaleA[K/128, M] and ScaleB[ceil(N/128), K/128], all
    contiguous.
    """
    assert (TILE_M, TILE_N, TILE_K) == (256, 256, 128)
    assert N % 16 == 0 and K >= TILE_K and K % TILE_K == 0
    KB = K // TILE_K
    scale_rows_b = (N + 127) // 128
    b_k_bytes = TILE_K * 16 if preshuffle_b else TILE_K

    @fx.struct
    class SharedTiles:
        a: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]
        b: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]

    @flyc.kernel(
        name=f"gemm_a8w8_blockscale_gfx942_2x4_lds_pf1_ps{int(preshuffle_b)}",
        known_block_size=[512, 1, 1],
    )
    def gemm_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argC: fx.Tensor,
        argScaleA: fx.Tensor,
        argScaleB: fx.Tensor,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        # No fx op makes a value wave-uniform: readfirstlane keeps wave_m/wave_n
        # in SGPRs, so the staggered `if wave_m == ...` barriers branch on scalars.
        wave_m = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
        wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave % 4).ir_value()))
        block_m, block_n = _tb_swizzle(fx.block_idx.x, M, N, TILE, TILE)
        row_base = block_m * TILE
        col_base = block_n * TILE
        rows_left = M - row_base
        rows = (rows_left < TILE).select(rows_left, TILE)
        cols_left = N - col_base
        cols = (cols_left < TILE).select(cols_left, TILE)

        # Rebase A, B and C in i64 per tile, in their own element types (no
        # recast claims more alignment than a view has); all descriptor
        # offsets below are tile-local bytes.
        a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(argA), fx.Int64(row_base) * K),
                fx.make_layout(TILE * K, 1),
            ),
            max_size=False,
            num_records_bytes=rows * K,
        )
        b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(argB), fx.Int64(col_base) * K),
                fx.make_layout(TILE * K, 1),
            ),
            max_size=False,
            num_records_bytes=cols * K,
        )
        c_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(argC), fx.Int64(row_base) * N),
                fx.make_layout(TILE * N, 1),
            ),
            max_size=False,
            num_records_bytes=rows * (N * 2),
        )
        scale_a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(argScaleA), fx.make_layout(M * KB, 1)),
            max_size=False,
            num_records_bytes=M * (KB * 4),
        )
        scale_b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(argScaleB), fx.make_layout(scale_rows_b * KB, 1)),
            max_size=False,
            num_records_bytes=scale_rows_b * KB * 4,
        )
        a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_tensor))
        b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_tensor))
        scale_a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_a_tensor))
        scale_b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_b_tensor))
        c_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(c_tensor))

        # Row offsets stay in voffset so raw-buffer range checks zero rows/columns
        # past M/N.
        copy_row = tid // 8
        copy_col = tid % 8
        a_voffsets = [
            (copy_row + packet * PACKET_ROWS) * K + copy_col * 16
            for packet in range_constexpr(4)
        ]
        a_write = _lds_dwords(copy_row, copy_col)
        if const_expr(preshuffle_b):
            b_voffsets = [
                (wave // 2 + packet * 4) * (16 * K) + wave % 2 * 1024 + lane * 16
                for packet in range_constexpr(4)
            ]
            b_write = _lds_dwords(wave // 2 * 16 + lane % 16, wave % 2 * 4 + lane // 16)
        else:
            b_voffsets = a_voffsets
            b_write = a_write
        local_row = wave_m * 16 + lane % 16
        a_read = [
            [
                _lds_dwords(local_row + rep_m * 32, lane // 16 + half_k * 4)
                for half_k in range_constexpr(2)
            ]
            for rep_m in range_constexpr(2)
        ]
        b_read = [
            [
                _lds_dwords(
                    wave_n * 16 + rep_n * 64 + lane % 16, lane // 16 + half_k * 4
                )
                for half_k in range_constexpr(2)
            ]
            for rep_n in range_constexpr(2)
        ]
        last_row = M - 1
        scale_a_voffsets = []
        for packet in range_constexpr(4):
            for rep_m in range_constexpr(2):
                row = row_base + local_row + (packet * PACKET_ROWS + rep_m * 32)
                # Rows past M are never stored; clamping keeps the K-block soffset
                # in bounds.
                scale_a_voffsets.append((row < last_row).select(row, last_row) * 4)
        scale_b_offsets = [
            col_base // 128 * (KB * 4),
            (col_base // 128 + 1) * (KB * 4),
        ]
        last_scale_b = (N - 1) // 128 * (KB * 4)
        scale_b_offsets[1] = (scale_b_offsets[1] < last_scale_b).select(
            scale_b_offsets[1], last_scale_b
        )

        lds = fx.SharedAllocator().allocate(SharedTiles).peek()

        def lds_view(base, dwords):
            return fx.make_view(base + dwords, fx.make_layout(4, 1))

        def write_a(packet):
            lds_view(lds.a.ptr, a_write + packet * PACKET_DWORDS).store(
                a_stage[None, packet].load()
            )

        def write_b(packet):
            lds_view(lds.b.ptr, b_write + packet * PACKET_DWORDS).store(
                b_stage[None, packet % 2].load()
            )

        def read_a(packet, rep_m, half_k):
            return lds_view(
                lds.a.ptr, a_read[rep_m][half_k] + packet * PACKET_DWORDS
            ).load()

        def read_b(side, rep_n, half_k):
            return lds_view(
                lds.b.ptr, b_read[rep_n][half_k] + side * 2 * PACKET_DWORDS
            ).load()

        def load_scale_a(k_block, packet, rep_m):
            # Raw load for the same reason as _buffer_load_dwordx4: a clamped
            # per-lane row voffset with the K-block offset in soffset.
            value = rocdl.raw_ptr_buffer_load(
                T.f32,
                scale_a_rsrc,
                fx.Int32(scale_a_voffsets[packet * 2 + rep_m]).ir_value(),
                fx.Int32(k_block * M * 4).ir_value(),
                fx.Int32(0).ir_value(),
            )
            return Vec.from_elements([fx.Float32(value)], fx.Float32)

        def barrier():
            # A bare s_barrier between scheduling fences, as in PyHIP: the
            # pipeline issues its own lgkmcnt/vmcnt waits, and gpu.barrier()
            # adds workgroup-scope memory fences that reschedule this pipeline
            # (its final ISA then differs from PyHIP's).
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # a_stage: four A packets; b_stage: two B slots.
        # b_ops column side * 4 + rep_n * 2 + half_k.
        a_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
        b_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
        b_ops = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
        # scale_a column slot * 2 + rep_m (slot = packet % 2);
        # scale_b column slot * 2 + side (slot = block % 2).
        scale_a = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        scale_b = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        # accum column (packet * 2 + side) * 4 + rep_m * 2 + rep_n;
        # partial column rep_m * 2 + rep_n.
        accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
        partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
        partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

        def load_scale_b(slot, k_block):
            for side in range_constexpr(2):
                scale_b[None, slot * 2 + side].store(
                    Vec.from_elements(
                        [
                            _s_buffer_load_f32(
                                scale_b_rsrc, scale_b_offsets[side] + k_block * 4
                            )
                        ],
                        fx.Float32,
                    )
                )

        def run_k(slot, k_block):
            # The last block re-fetches clamped blocks so it stays inside the loop body.
            k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
            for packet in range_constexpr(4):
                if const_expr(packet == 0):
                    write_a(3)
                else:
                    write_a(packet - 1)
                    write_b(packet - 1)
                    if const_expr(packet == 3):
                        write_b(3)
                rocdl.sched_barrier(0)
                if const_expr(packet == 0):
                    for side in range_constexpr(2):
                        for rep_n in range_constexpr(2):
                            for half_k in range_constexpr(2):
                                b_ops[None, side * 4 + rep_n * 2 + half_k].store(
                                    read_b(side, rep_n, half_k)
                                )
                a_ops = [
                    [
                        Vec(read_a(packet, rep_m, half_k)).bitcast(fx.Int64)
                        for half_k in range_constexpr(2)
                    ]
                    for rep_m in range_constexpr(2)
                ]
                rocdl.sched_barrier(0)
                for rep_m in range_constexpr(2):
                    if const_expr(packet < 3):
                        scale_a[None, (packet + 1) % 2 * 2 + rep_m].store(
                            load_scale_a(k_block, packet + 1, rep_m)
                        )
                    else:
                        scale_a[None, rep_m].store(load_scale_a(k_next, 0, rep_m))
                a_stage[None, packet].store(
                    _buffer_load_dwordx4(a_rsrc, a_voffsets[packet], k_next * TILE_K)
                )
                if const_expr(packet < 3):
                    b_stage[None, packet % 2].store(
                        _buffer_load_dwordx4(
                            b_rsrc, b_voffsets[packet], k_next * b_k_bytes
                        )
                    )
                if const_expr(packet == 2):
                    b_stage[None, 1].store(
                        _buffer_load_dwordx4(b_rsrc, b_voffsets[3], k_next * b_k_bytes)
                    )
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                # No fx API sets wave priority: this group's MFMA phase runs at
                # priority 3 against the other group's memory phase.
                rocdl.s_setprio(3)
                if const_expr(packet == 0):
                    # Next block's B scales; consumed after several lgkmcnt(0) waits.
                    load_scale_b(1 - slot, k_next)
                    rocdl.sched_barrier(0)
                for side in range_constexpr(2):
                    previous_packet = (packet - 1) % 4 if side == 0 else packet
                    for pair_begin in range_constexpr(0, 4, 2):
                        rep_m = pair_begin // 2
                        for k_step in range_constexpr(4):
                            half_k, word = divmod(k_step, 2)
                            for rep_n in range_constexpr(2):
                                partial_index = pair_begin + rep_n
                                index = pair_begin * 4 + k_step * 2 + rep_n
                                b_value = Vec(
                                    b_ops[None, side * 4 + rep_n * 2 + half_k].load()
                                ).bitcast(fx.Int64)[word]
                                accumulator = (
                                    Vec.filled(4, 0.0, fx.Float32)
                                    if k_step == 0
                                    else Vec(partial[None, partial_index].load())
                                )
                                # Raw MFMA, not an MMA atom with fx.gemm: one
                                # instruction at a time on i64 words of the LDS
                                # fragments with A and B swapped (D[n][m]), each
                                # followed by one scale FMA under sched_barrier.
                                partial[None, partial_index].store(
                                    Vec(
                                        rocdl.mfma_f32_16x16x32_fp8_fp8(
                                            T.f32x4,
                                            [
                                                b_value,
                                                a_ops[rep_m][half_k][word],
                                                accumulator,
                                                0,
                                                0,
                                                0,
                                            ],
                                        )
                                    )
                                )
                                # Keep MFMA -> FMA order: the FMA's partial then has
                                # two MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                if const_expr(index == 0 or index == 8):
                                    scale = (
                                        Vec(
                                            scale_a[
                                                None, packet % 2 * 2 + index // 8
                                            ].load()
                                        )[0]
                                        * Vec(scale_b[None, slot * 2 + side].load())[0]
                                    )
                                    partial_scale[None, index // 8].store(
                                        Vec.from_elements([scale], fx.Float32)
                                    )
                                if const_expr(index < 8):
                                    source, source_scale = 2 + index // 4, 1
                                    column = (
                                        previous_packet * 2 + 1 - side
                                    ) * 4 + source
                                else:
                                    source, source_scale = (index - 8) // 4, 0
                                    column = (packet * 2 + side) * 4 + source
                                target = accum[None, column]
                                element = index % 4
                                target[element] = fx.fma(
                                    Vec(partial[None, source].load())[element],
                                    Vec(partial_scale[None, source_scale].load())[0],
                                    target[element],
                                )
                                rocdl.sched_barrier(0)
                rocdl.s_setprio(0)
                barrier()

        for packet in range_constexpr(4):
            a_stage[None, packet].store(
                _buffer_load_dwordx4(a_rsrc, a_voffsets[packet], 0)
            )
        for pair_begin in range_constexpr(0, 4, 2):
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                b_stage[None, packet % 2].store(
                    _buffer_load_dwordx4(b_rsrc, b_voffsets[packet], 0)
                )
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                write_b(packet)
            rocdl.sched_barrier(0)
        for rep_m in range_constexpr(2):
            scale_a[None, rep_m].store(load_scale_a(0, 0, rep_m))
        load_scale_b(0, 0)
        for packet in range_constexpr(4):
            write_a(packet)
        accum.fill(0)
        # The first stage retires these zeros as the "previous" side.
        for index in range_constexpr(4):
            partial[None, index].store(Vec.filled(4, 0.0, fx.Float32))
        for index in range_constexpr(2):
            partial_scale[None, index].store(Vec.filled(1, 0.0, fx.Float32))
        rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3
        # compute.
        if wave_m == 1:
            barrier()

        # Two blocks per iteration alternate the B-scale slots; odd KB peels block 0.
        if const_expr(KB % 2):
            run_k(0, fx.Int32(0))
        for pair in range(KB // 2):
            k_block = fx.Int32(pair) * 2 + KB % 2
            run_k(KB % 2, k_block)
            run_k(1 - KB % 2, k_block + 1)

        # The final side's partials 2-3 have no later MFMA shadow.
        final_scale = Vec(partial_scale[None, 1].load())[0]
        for source in range_constexpr(2, 4):
            target = accum[None, (3 * 2 + 1) * 4 + source]
            values = Vec(partial[None, source].load())
            for element in range_constexpr(4):
                target[element] = fx.fma(values[element], final_scale, target[element])
        if wave_m == 0:
            barrier()

        # ---- epilogue store ----
        out_dtype = argC.dtype
        assert out_dtype in (fx.BFloat16, fx.Float16)
        output_col = col_base + wave_n * 16 + lane // 16 * 4
        for packet in range_constexpr(4):
            for side in range_constexpr(2):
                for rep_m in range_constexpr(2):
                    for rep_n in range_constexpr(2):
                        column = output_col + (side * 128 + rep_n * 64)
                        row = local_row + (packet * PACKET_ROWS + rep_m * 32)
                        offset = (column < N).select(
                            row * (N * 2) + column * 2, fx.Int32(0x7FFFFFFF)
                        )
                        values = Vec(
                            accum[
                                None,
                                (packet * 2 + side) * 4 + rep_m * 2 + rep_n,
                            ].load()
                        )
                        # Round to nearest even, as the gfx950 8-wave kernel's
                        # epilogue does; gfx942 has no v_cvt_pk_bf16_f32, so
                        # LLVM expands the BF16 conversion.
                        packed = values.to(out_dtype).bitcast(fx.Int32)
                        # Raw store, not fx.copy(BufferCopy64b): columns past N
                        # need the explicit out-of-range voffset 0x7FFFFFFF.
                        rocdl.raw_ptr_buffer_store(
                            packed.ir_value(),
                            c_rsrc,
                            offset.ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(T.i32, 0),
                        )

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        scaleA: fx.Tensor,
        scaleB: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Keep the dequant FMAs scalar: v_pk_fma_f32 pairs would break the
        # one-FMA-per-MFMA interleave.
        value_attrs = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}
        gemm_kernel(A, B, C, scaleA, scaleB, M, value_attrs=value_attrs).launch(
            grid=(ceildiv(M, TILE_M) * ceildiv(N, TILE_N), 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch_gemm
