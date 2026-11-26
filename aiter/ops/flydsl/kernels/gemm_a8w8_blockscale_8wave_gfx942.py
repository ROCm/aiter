# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B008
# The FlyDSL body uses a typed Stream default for its tracing semantics.
"""gfx942 FP8 GEMM: C = A @ B.T, FP32 accumulation and BF16/FP16 output.

The blockscale contract is ScaleA[KB, M] and ScaleB[ceil(N/128), KB], KB=K/128,
with FP8 E4M3FNUZ A[M, K] and B[N, K]; ScaleA is x_scale transposed.
The kernel uses a 256x256 WG tile, eight waves on a 2x4 grid, and 64 KiB LDS.
Optional preshuffle_b consumes aiter.shuffle_weight(B, layout=(16, 16)) directly;
only the B load and LDS write offsets change, both scale layouts stay unchanged.
Needs N % 16 == 0, K % 128 == 0 and K >= 128; any M >= 1 works: the raw-buffer
range checks zero rows and columns past M/N, rows past M are outside the C
descriptor and columns past N store at voffset 0x7FFFFFFF, which is dropped.

Only the 2x4-LDS pf1 pipeline is supported. wave_m = wave_id // 4 owns 128
rows, wave_n = wave_id % 4 owns 64 columns. gfx942 has no 16-byte global-to-LDS
load, so A and B are staged through VGPRs (buffer_load_dwordx4, ds_write_b128)
into 32 KiB of LDS each: one K tile as four 64-row packets, XOR-swizzled.
B operands are read once per K tile; A packet p is read in stage p. Each packet
of the next K tile is loaded during this tile and written to LDS one stage
later (one-stage load lead, "pf1"). Waves 4-7 trail waves 0-3 by one barrier,
and each compute phase runs at s_setprio 3. ScaleA uses per-lane buffer loads;
ScaleB retains its inline-asm scalar buffer loads.

Names and helpers follow the gfx950 8-wave kernel
(gemm_a8w8_blockscale_8wave_gfx950.py). Global A/B loads go through
_buffer_load_dwordx4 below: one buffer_load_dwordx4 with a byte voffset and
soffset. Each raw rocdl/llvm call names the stable API it stands in for and
why that API is not used.

At K = 128 (one K tile) the two inline-asm s_buffer_load_dword of the next
tile's B scales still load values nothing reads. FlyDSL 0.5.3 gives the two
loads separate SGPRs, so they do not write the same register.

The epilogue rounds the FP32 results to BF16 or FP16 to nearest even, as the
gfx950 8-wave kernel does.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.kernels_common import ceildiv


def _buffer_resource(tensor, num_records_bytes):
    buffer = rocdl.make_buffer_tensor(tensor, num_records_bytes=num_records_bytes)
    return rocdl.get_buffer_rsrc(fx.get_iter(buffer))


def _buffer_load_dwordx4(rsrc, voffset_bytes, soffset_bytes):
    """One buffer_load_dwordx4 ... offen of four i32 at byte voffset + soffset."""
    # Raw load, not fx.copy(BufferCopy128b): a copy atom addresses a layout
    # coordinate, while this pipeline issues a precomputed tile-relative
    # voffset (row offsets, range checked so rows past M/N read zero) with the
    # K-tile offset in soffset, as a single buffer_load_dwordx4.
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
    # Inline asm, as gfx950's load_scale_b_inline, not tensor_shim.buf_scalar_load:
    # the tracked SMEM intrinsic lacks a memory operand in this LLVM and gets
    # conservative waits. Every use follows the pipeline's own lgkmcnt(0) waits.
    value = _llvm.inline_asm(
        T.f32,
        [rsrc, fx.Int32(offset_bytes).ir_value()],
        "s_buffer_load_dword $0, $1, $2",
        "=s,s,s",
        has_side_effects=True,
    )
    return fx.Float32(value)


def compile_gemm_fp8_blockscale_gfx942(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    preshuffle_b=False,
):
    """Compile the 2x4-LDS pf1 blockscale kernel with VGPR-staged A/B loads.

    The pipeline has a one-stage A/B load lead, ScaleA loads one stage ahead,
    inline-asm scalar ScaleB loads one K tile ahead, and one scalar FMA per
    MFMA. Each compute phase raises s_setprio to 3; waves 4-7 trail waves
    0-3 by one barrier. preshuffle_b uses the FP8 (16, 16) weight shuffle; only
    its B load and LDS write offsets differ from the plain path.

    The interface is the gfx950 8-wave kernel's: ``launch_gemm(A, B, C, scaleA,
    scaleB, M, stream)`` takes A[M, K] and B (plain [N, K] or
    shuffle_weight(B, (16, 16))) as FP8 bytes, C[M, N] in BF16 or FP16, and
    FP32 ScaleA[K/128, M] and ScaleB[ceil(N/128), K/128], all contiguous. The
    kernel rebases A, B and C per tile in i64; C's dtype selects the epilogue.
    """
    assert (TILE_M, TILE_N, TILE_K) == (256, 256, 128)
    assert N % 16 == 0 and K >= TILE_K and K % TILE_K == 0
    BLOCK_K = TILE_K
    # A and B tiles are four LDS packets of 64 rows x 128 bytes (8 KiB);
    # stage p of a K tile computes A packet p.
    PACKET_ROWS = 64
    NUM_PACKETS = TILE_M // PACKET_ROWS
    PACKET_DWORDS = PACKET_ROWS * BLOCK_K // 4
    # Per packet, each wave repeats its 16x16 MFMA tile twice along M (32 rows
    # apart) and, per 128-column half, twice along N (64 columns apart).
    PHASE_M_REP = 2
    N_REP = 2
    num_tiles = K // BLOCK_K
    # ScaleA is [scaleA_stride, M] and ScaleB [ceil(N/128), scaleA_stride].
    scaleA_stride = K // 128
    # B bytes per K tile: 128 in a plain row, 2 KiB in shuffle_weight's layout.
    b_tile_k_bytes = BLOCK_K * 16 if preshuffle_b else BLOCK_K

    @flyc.jit
    def get_pids_942(pid, M, M01, GROUP_NUM):
        # Tile-id swizzle, the counterpart of get_pids_950: pids with equal
        # pid % GROUP_NUM take one contiguous range of tile ids (cf. NUM_XCDS),
        # and each band of M01 M tiles is walked M first (cf. GROUP_SIZE_M).
        num_pid_m = (M + TILE_M - 1) // TILE_M
        num_pid_n = ceildiv(N, TILE_N)
        group_size = (num_pid_m * num_pid_n + GROUP_NUM - 1) // GROUP_NUM
        big_group_num = GROUP_NUM - (group_size * GROUP_NUM - num_pid_m * num_pid_n)
        group_id_y = pid // GROUP_NUM
        group_id_x = pid - group_id_y * GROUP_NUM
        remap = group_id_x * group_size + group_id_y
        remap = (group_id_x > big_group_num).select(
            remap + (big_group_num - group_id_x), remap
        )
        idx_m0 = remap // num_pid_n
        idx_n0 = remap - idx_m0 * num_pid_n
        m0_mod_m01 = num_pid_m - num_pid_m // M01 * M01
        m01_adapt = (idx_m0 < num_pid_m - m0_mod_m01).select(fx.Int32(M01), m0_mod_m01)
        idx_m00 = idx_m0 // M01
        local = idx_n0 + (idx_m0 - idx_m00 * M01) * num_pid_n
        pid_n = local // m01_adapt
        return local - pid_n * m01_adapt + idx_m00 * M01, pid_n

    def _lds_dwords(row, dwordx4_chunk):
        # XOR swizzle of the 16-byte chunk keeps MFMA-pattern ds_read_b128
        # conflict free.
        return row * (BLOCK_K // 4) + ((row ^ dwordx4_chunk) & 7) * 4

    # A and B tiles use 32 KiB each (one K tile); 64 KiB in total.
    @fx.struct
    class LDS:
        a: fx.Array[fx.Int32, NUM_PACKETS * PACKET_DWORDS, 16]
        b: fx.Array[fx.Int32, NUM_PACKETS * PACKET_DWORDS, 16]

    @flyc.kernel(
        name=f"gemm_a8w8_blockscale_8wave_gfx942_ps{int(preshuffle_b)}_BM{TILE_M}_BN{TILE_N}_BK{TILE_K}",
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
        lane_id = tid % 64
        wave_id = tid // 64
        # No fx op makes a value wave-uniform: readfirstlane keeps wave_m/wave_n
        # in SGPRs, so the staggered `if wave_m == ...` barriers branch on scalars.
        wave_m = fx.Int32(rocdl.readfirstlane(T.i32, (wave_id // 4).ir_value()))
        wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave_id % 4).ir_value()))
        bid_x, bid_y = get_pids_942(fx.block_idx.x, M, 8, 8)
        row_base = bid_x * TILE_M
        col_base = bid_y * TILE_N
        a_rows_left = M - row_base
        a_rows = (a_rows_left < TILE_M).select(a_rows_left, TILE_M)
        b_rows_left = N - col_base
        b_rows = (b_rows_left < TILE_N).select(b_rows_left, TILE_N)

        # Rebase A, B and C in i64 per tile before building buffer descriptors,
        # in their own element types: no recast claims more alignment than a
        # view has. All load/store offsets below are local to this WG; the
        # ScaleA/ScaleB descriptors cover the whole matrices.
        a_iter = fx.add_offset(fx.get_iter(argA), fx.Int64(row_base) * K)
        a_rsrc = _buffer_resource(
            fx.make_view(a_iter, fx.make_layout(TILE_M * K, 1)),
            num_records_bytes=a_rows * K,
        )
        b_iter = fx.add_offset(fx.get_iter(argB), fx.Int64(col_base) * K)
        b_rsrc = _buffer_resource(
            fx.make_view(b_iter, fx.make_layout(TILE_N * K, 1)),
            num_records_bytes=b_rows * K,
        )
        c_iter = fx.add_offset(fx.get_iter(argC), fx.Int64(row_base) * N)
        c_store_rsrc = _buffer_resource(
            fx.make_view(c_iter, fx.make_layout(TILE_M * N, 1)),
            num_records_bytes=a_rows * (N * 2),
        )
        scaleA_rsrc = _buffer_resource(
            argScaleA, num_records_bytes=M * (scaleA_stride * 4)
        )
        scaleB_rsrc = _buffer_resource(
            argScaleB, num_records_bytes=ceildiv(N, 128) * scaleA_stride * 4
        )

        ### Per-lane global and LDS offsets.
        # g2r mapping: 512 threads = 64 rows x 8 16-byte chunks per packet.
        # Row offsets stay in voffset so raw-buffer range checks zero rows past
        # M/N; the K-tile offset goes in soffset.
        a_lane_row = tid // 8
        a_lane_chunk = tid % 8
        a_src_offsets = [
            (a_lane_row + packet * PACKET_ROWS) * K + a_lane_chunk * 16
            for packet in range_constexpr(NUM_PACKETS)
        ]
        ldsA_wr = _lds_dwords(a_lane_row, a_lane_chunk)
        if const_expr(preshuffle_b):
            # shuffle_weight's [N/16, K/128, 4k2, 2k1, 16n0, 16k0] layout: each
            # wave copies a contiguous 16x64-byte block per packet.
            b_src_offsets = [
                (wave_id // 2 + packet * 4) * (16 * K)
                + wave_id % 2 * 1024
                + lane_id * 16
                for packet in range_constexpr(NUM_PACKETS)
            ]
            ldsB_wr = _lds_dwords(
                wave_id // 2 * 16 + lane_id % 16, wave_id % 2 * 4 + lane_id // 16
            )
        else:
            b_src_offsets = a_src_offsets
            ldsB_wr = ldsA_wr
        # MFMA operand rows: lane_id % 16 selects the row, lane_id // 16 the
        # 16-byte chunk of each 64-byte K half. A and B share this K order,
        # which leaves each K128 dot product unchanged.
        local_row = wave_m * 16 + lane_id % 16
        ldsA_rd = [
            [
                _lds_dwords(local_row + m0 * 32, lane_id // 16 + half_k * 4)
                for half_k in range_constexpr(2)
            ]
            for m0 in range_constexpr(PHASE_M_REP)
        ]
        ldsB_rd = [
            [
                _lds_dwords(
                    wave_n * 16 + n0 * 64 + lane_id % 16, lane_id // 16 + half_k * 4
                )
                for half_k in range_constexpr(2)
            ]
            for n0 in range_constexpr(N_REP)
        ]
        ### ScaleA rows of this lane and the tile's two ScaleB rows.
        last_row = M - 1
        scaleA_src_offsets = []
        for packet in range_constexpr(NUM_PACKETS):
            for m0 in range_constexpr(PHASE_M_REP):
                scale_row = row_base + local_row + (packet * PACKET_ROWS + m0 * 32)
                # Rows past M are never stored; clamping keeps the K-tile soffset
                # in bounds.
                scaleA_src_offsets.append(
                    (scale_row < last_row).select(scale_row, last_row) * 4
                )
        scaleB_offsets = [
            col_base // 128 * (scaleA_stride * 4),
            (col_base // 128 + 1) * (scaleA_stride * 4),
        ]
        last_scaleB_offset = (N - 1) // 128 * (scaleA_stride * 4)
        scaleB_offsets[1] = (scaleB_offsets[1] < last_scaleB_offset).select(
            scaleB_offsets[1], last_scaleB_offset
        )

        ### Allocate LDS.
        lds = fx.SharedAllocator().allocate(LDS).peek()

        ### Register tensors.
        # a_prefetch_reg: four A packets; b_prefetch_reg: two B slots (packet % 2).
        # Both hold the next K tile in VGPRs until its LDS packet is free.
        # frag_B column quadrant_n * 4 + n0 * 2 + half_k: all B operands of a
        # K tile, read in stage 0.
        a_prefetch_reg = fx.make_rmem_tensor(
            fx.make_layout((4, NUM_PACKETS), (1, 4)), fx.Int32
        )
        b_prefetch_reg = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
        frag_B = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
        # mfma_scaleA column slot * 2 + m0 (slot = packet % 2);
        # mfma_scaleB column slot * 2 + quadrant_n (slot = K tile % 2).
        mfma_scaleA = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        mfma_scaleB = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        # frag_C column (packet * 2 + quadrant_n) * 4 + m0 * 2 + n0: the wave's
        # 128x64 C tile. frag_P column m0 * 2 + n0: one 16-f32 FIFO of unscaled
        # K128 partials. fifo_dq_scale column m0: ScaleA * ScaleB of FIFO pair m0.
        frag_C = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
        frag_P = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
        fifo_dq_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

        def lds_view(base, dwords):
            return fx.make_view(base + dwords, fx.make_layout(4, 1))

        def lds_wr_A(packet):
            lds_view(lds.a.ptr, ldsA_wr + packet * PACKET_DWORDS).store(
                a_prefetch_reg[None, packet].load()
            )

        def lds_wr_B(packet):
            lds_view(lds.b.ptr, ldsB_wr + packet * PACKET_DWORDS).store(
                b_prefetch_reg[None, packet % 2].load()
            )

        def lds_rd_B():
            # Both 128-column B halves (two packets each) of the K tile.
            for quadrant_n in range_constexpr(2):
                for n0 in range_constexpr(N_REP):
                    for half_k in range_constexpr(2):
                        frag_B[None, quadrant_n * 4 + n0 * 2 + half_k].store(
                            lds_view(
                                lds.b.ptr,
                                ldsB_rd[n0][half_k] + quadrant_n * 2 * PACKET_DWORDS,
                            ).load()
                        )

        def lds_rd_A(packet):
            # A packet p as i64 MFMA operand words: frag_A[m0][half_k][word].
            return [
                [
                    Vec(
                        lds_view(
                            lds.a.ptr, ldsA_rd[m0][half_k] + packet * PACKET_DWORDS
                        ).load()
                    ).bitcast(fx.Int64)
                    for half_k in range_constexpr(2)
                ]
                for m0 in range_constexpr(PHASE_M_REP)
            ]

        def load_A(packet, ki):
            a_prefetch_reg[None, packet].store(
                _buffer_load_dwordx4(a_rsrc, a_src_offsets[packet], ki * BLOCK_K)
            )

        def load_B(packet, ki):
            b_prefetch_reg[None, packet % 2].store(
                _buffer_load_dwordx4(b_rsrc, b_src_offsets[packet], ki * b_tile_k_bytes)
            )

        def load_scale_a(packet, m0, ki):
            # Raw load for the same reason as _buffer_load_dwordx4: a clamped
            # per-lane row voffset with the K-tile offset in soffset.
            value = rocdl.raw_ptr_buffer_load(
                T.f32,
                scaleA_rsrc,
                fx.Int32(scaleA_src_offsets[packet * PHASE_M_REP + m0]).ir_value(),
                fx.Int32(ki * M * 4).ir_value(),
                0,
            )
            return Vec.from_elements([fx.Float32(value)], fx.Float32)

        def load_scale_b(slot, ki):
            # The two ScaleB values (quadrant_n 0/1) of K tile ki, into SGPRs.
            for quadrant_n in range_constexpr(2):
                mfma_scaleB[None, slot * 2 + quadrant_n].store(
                    Vec.from_elements(
                        [
                            _s_buffer_load_f32(
                                scaleB_rsrc, scaleB_offsets[quadrant_n] + ki * 4
                            )
                        ],
                        fx.Float32,
                    )
                )

        def barrier():
            # A bare s_barrier between scheduling fences: the pipeline issues
            # its own lgkmcnt/vmcnt waits, and gpu.barrier() adds workgroup-scope
            # memory fences that reschedule this pipeline.
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def begin_compute_phase():
            # The stage's ds_write/ds_read complete before the barrier. No fx API
            # sets wave priority: the compute phase runs at priority 3 against
            # the other wave group's memory phase.
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            barrier()
            rocdl.s_setprio(3)

        def end_compute_phase():
            rocdl.s_setprio(0)
            barrier()

        def do_gemm(packet, tick, frag_A):
            # One compute phase: A packet p against both B halves, 32 MFMA
            # (2 quadrant_n x PHASE_M_REP x 4 K steps x N_REP), A and B swapped
            # (D[n][m]). A half-phase is one quadrant_n (16 MFMA). frag_P holds
            # its unscaled K128 partials; each MFMA is followed by one FP32 FMA
            # that retires one element of an older partial into frag_C:
            #   MFMA 0-7 (m0 = 0) retire FIFO pair 1 (partials 2-3) of the
            #   previous half-phase before m0 = 1 overwrites it;
            #   MFMA 8-15 (m0 = 1) retire this half-phase's pair 0.
            # The last half-phase's pair 1 is drained after the K loop.
            # Compiler-only sched_barrier fences keep 1 MFMA -> 1 FMA; the
            # launch's -packed-fp32-ops keeps the FMAs scalar.
            for quadrant_n in range_constexpr(2):
                b_slot = tick * 2 + quadrant_n
                previous_packet = (
                    (packet - 1) % NUM_PACKETS if quadrant_n == 0 else packet
                )
                for m0 in range_constexpr(PHASE_M_REP):
                    a_slot = packet % 2 * 2 + m0
                    for k_step in range_constexpr(4):
                        half_k, word = divmod(k_step, 2)
                        for n0 in range_constexpr(N_REP):
                            p_col = m0 * N_REP + n0
                            mfma_idx = m0 * 8 + k_step * 2 + n0
                            b_value = Vec(
                                frag_B[None, quadrant_n * 4 + n0 * 2 + half_k].load()
                            ).bitcast(fx.Int64)[word]
                            accumulator = (
                                Vec.filled(4, 0.0, fx.Float32)
                                if k_step == 0
                                else Vec(frag_P[None, p_col].load())
                            )
                            # Raw MFMA, not an MMA atom with fx.gemm: one
                            # instruction at a time on i64 words of the LDS
                            # fragments with A and B swapped (D[n][m]), each
                            # followed by one scale FMA under sched_barrier.
                            frag_P[None, p_col].store(
                                Vec(
                                    rocdl.mfma_f32_16x16x32_fp8_fp8(
                                        T.f32x4,
                                        [
                                            b_value,
                                            frag_A[m0][half_k][word],
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
                            if const_expr(mfma_idx == 0 or mfma_idx == 8):
                                dq_scale_a = Vec(mfma_scaleA[None, a_slot].load())[0]
                                dq_scale_b = Vec(mfma_scaleB[None, b_slot].load())[0]
                                dq_scale = dq_scale_a * dq_scale_b
                                fifo_dq_scale[None, m0].store(
                                    Vec.from_elements([dq_scale], fx.Float32)
                                )
                            if const_expr(m0 == 0):
                                retire_col, retire_pair = 2 + mfma_idx // 4, 1
                                c_col = (
                                    previous_packet * 2 + 1 - quadrant_n
                                ) * 4 + retire_col
                            else:
                                retire_col, retire_pair = (mfma_idx - 8) // 4, 0
                                c_col = (packet * 2 + quadrant_n) * 4 + retire_col
                            sub_frag_C = frag_C[None, c_col]
                            elem = mfma_idx % 4
                            sub_frag_C[elem] = fx.fma(
                                Vec(frag_P[None, retire_col].load())[elem],
                                Vec(fifo_dq_scale[None, retire_pair].load())[0],
                                sub_frag_C[elem],
                            )
                            rocdl.sched_barrier(0)

        def run_k_tile(tick, ki):
            # One K128 tile k in four stages; stage p computes A packet p.
            # t = tick (k % 2) and u = tock select the ScaleB slots; SA = ScaleA
            # (packet), SB = ScaleB. Each stage's memory phase writes the packets
            # staged one stage earlier (pf1), reads its operands and issues its
            # loads; then its compute phase runs do_gemm.
            # | Stage | ds_write (tile)    | ds_read | loads issued (tile)          |
            # |-------|--------------------|---------|------------------------------|
            # | 0     | A3 (k)             | B, A0   | SA1 (k), A0 B0 (k+1)         |
            # | 1     | A0 B0 (k+1)        | A1      | SA2 (k), A1 B1 (k+1)         |
            # | 2     | A1 B1 (k+1)        | A2      | SA3 (k), A2 B2 B3 (k+1)      |
            # | 3     | A2 B2 B3 (k+1)     | A3      | SA0 (k+1), A3 (k+1)          |
            # Every stage: ds_write -> ds_read -> loads -> lgkmcnt(0) ->
            # s_barrier -> s_setprio 3 -> do_gemm -> s_setprio 0 -> s_barrier.
            # Stage 0's compute phase first loads SB (u, k+1) into two SGPRs.
            # The last tile re-fetches clamped tiles so it stays inside the loop body.
            ki_next = (ki + 1 < num_tiles).select(ki + 1, num_tiles - 1)
            tock = 1 - tick
            for packet in range_constexpr(NUM_PACKETS):
                if const_expr(packet == 0):
                    lds_wr_A(3)
                else:
                    lds_wr_A(packet - 1)
                    lds_wr_B(packet - 1)
                    if const_expr(packet == 3):
                        lds_wr_B(3)
                rocdl.sched_barrier(0)
                if const_expr(packet == 0):
                    lds_rd_B()
                frag_A = lds_rd_A(packet)
                rocdl.sched_barrier(0)
                for m0 in range_constexpr(PHASE_M_REP):
                    if const_expr(packet < 3):
                        mfma_scaleA[None, (packet + 1) % 2 * 2 + m0].store(
                            load_scale_a(packet + 1, m0, ki)
                        )
                    else:
                        mfma_scaleA[None, m0].store(load_scale_a(0, m0, ki_next))
                load_A(packet, ki_next)
                if const_expr(packet < 3):
                    load_B(packet, ki_next)
                if const_expr(packet == 2):
                    load_B(3, ki_next)
                begin_compute_phase()
                if const_expr(packet == 0):
                    # Next tile's B scales; consumed after several lgkmcnt(0) waits.
                    load_scale_b(tock, ki_next)
                    rocdl.sched_barrier(0)
                do_gemm(packet, tick, frag_A)
                end_compute_phase()

        ### Prologue: K tile 0 and its scales; zero C and the FIFO.
        for packet in range_constexpr(NUM_PACKETS):
            load_A(packet, 0)
        for pair_begin in range_constexpr(0, NUM_PACKETS, 2):
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                load_B(packet, 0)
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                lds_wr_B(packet)
            rocdl.sched_barrier(0)
        for m0 in range_constexpr(PHASE_M_REP):
            mfma_scaleA[None, m0].store(load_scale_a(0, m0, 0))
        load_scale_b(0, 0)
        for packet in range_constexpr(NUM_PACKETS):
            lds_wr_A(packet)
        frag_C.fill(0)
        # The first stage retires these zeros as the "previous" half-phase.
        frag_P.fill(0)
        fifo_dq_scale.fill(0)
        rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        # Offset the two groups of four waves by one stage: waves 4-7 trail by
        # one barrier, so each group's memory phase overlaps the other group's
        # compute phase. The lower group closes this unmatched barrier after the
        # final FIFO drain.
        if wave_m == 1:
            barrier()

        # Two K tiles per iteration alternate the ScaleB slots (tick/tock); an
        # odd num_tiles peels tile 0.
        if const_expr(num_tiles % 2):
            run_k_tile(0, fx.Int32(0))
        for kidx in range(num_tiles // 2):
            kiter = fx.Int32(kidx) * 2 + num_tiles % 2
            run_k_tile(num_tiles % 2, kiter)
            run_k_tile(1 - num_tiles % 2, kiter + 1)

        # Drain the final FIFO pair: partials 2-3 of the last half-phase
        # (packet 3, quadrant_n 1) have no later MFMA shadow.
        dq_scale = Vec(fifo_dq_scale[None, 1].load())[0]
        for p_col in range_constexpr(2, 4):
            sub_frag_C = frag_C[None, ((NUM_PACKETS - 1) * 2 + 1) * 4 + p_col]
            partial = Vec(frag_P[None, p_col].load())
            for elem in range_constexpr(4):
                sub_frag_C[elem] = fx.fma(partial[elem], dq_scale, sub_frag_C[elem])
        if wave_m == 0:
            barrier()

        # ---- epilogue store ----
        out_dtype = argC.dtype
        assert out_dtype in (fx.BFloat16, fx.Float16)
        # In physical C, lane_id % 16 selects a row of the MFMA tile and
        # lane_id // 16 four consecutive columns in that row.
        output_col = col_base + wave_n * 16 + lane_id // 16 * 4
        for packet in range_constexpr(NUM_PACKETS):
            for quadrant_n in range_constexpr(2):
                for m0 in range_constexpr(PHASE_M_REP):
                    for n0 in range_constexpr(N_REP):
                        col = output_col + (quadrant_n * (TILE_N // 2) + n0 * 64)
                        row = local_row + (packet * PACKET_ROWS + m0 * 32)
                        # N % 16 == 0: each four-element store is fully in or out.
                        byte_offset = (col < N).select(
                            row * (N * 2) + col * 2, fx.Int32(0x7FFFFFFF)
                        )
                        acc = Vec(
                            frag_C[
                                None,
                                (packet * 2 + quadrant_n) * 4 + m0 * N_REP + n0,
                            ].load()
                        )
                        # Round to nearest even, as the gfx950 8-wave kernel's
                        # epilogue does; gfx942 has no v_cvt_pk_bf16_f32, so
                        # LLVM expands the BF16 conversion.
                        packed = acc.to(out_dtype).bitcast(fx.Int32)
                        # Raw store, not fx.copy(BufferCopy64b): columns past N
                        # need the explicit out-of-range voffset 0x7FFFFFFF.
                        # vector<2xi32> -> buffer_store_dwordx2: C[row, col:col+4].
                        rocdl.raw_ptr_buffer_store(
                            packed.ir_value(),
                            c_store_rsrc,
                            byte_offset.ir_value(),
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
