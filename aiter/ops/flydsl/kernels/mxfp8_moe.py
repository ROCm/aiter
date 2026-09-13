# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 MXFP8/A8W4 MoE prefill with GPU-resident routing.

MXFP8 uses 256x256 eight-wave tiles; A8W4 uses 128x256 four-wave tiles.
Weights retain the G1U1 16x64 preshuffle and E8M0 scale layout. Stage 1
fuses SwiGLU (MXFP8) or SiLU and output quantization (A8W4). Stage 2 writes
sorted BF16 partials for the shared FP32 routing reduction.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .gemm_a8w8_8wave import (
    _xcd_swizzle_any,
    ceildiv,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    make_g2s_loader,
    make_s2r_loader,
    run_8wave_pipeline,
)
from .kernels_common import get_warp_size
from .mfma_preshuffle_pipeline import split_row_major_2d
from .mxfp8_moe_utils import (
    _store_factory,
    make_mx_mfma,
    make_mx_pipeline_mma,
    make_scale_preshuffled_s2r,
)


def compile_mxfp8_moe_gemm(
    *,
    K,
    stage,
    b_dtype="fp8",
    b_k=None,
    xcd_swizzle=1,
    swiglu_limit=None,
    persistent_tiles=0,
):
    """Compile a tuned grouped tile; K is the padded A/scale stride.

    Stage 1 gathers A by row_map. Stage 2 consumes sorted A and physical
    weight stride b_k. Live rows stay on the GPU; c_m bounds the allocation.
    Persistent2/4 retain B across adjacent M256 tiles for FP8 K384 only.
    """
    assert str(get_rocm_arch()).startswith("gfx950") and get_warp_size() == 64
    assert stage in (1, 2) and b_dtype in ("fp8", "fp4")
    assert K >= 256 and K % 256 == 0
    b_k = K if b_k is None else b_k
    assert 256 <= b_k <= K and b_k % 128 == 0
    assert persistent_tiles in (0, 2, 4)
    if persistent_tiles:
        assert stage == 2 and b_dtype == "fp8" and K == 512 and b_k == 384
        return _compile_persistent_gemm(
            xcd_swizzle=xcd_swizzle, m_tiles=persistent_tiles
        )

    num_waves = 4 if b_dtype == "fp4" else 8
    threads = num_waves * 64
    BLOCK_M, BLOCK_N, BLOCK_K = (128 if num_waves == 4 else 256), 256, 128
    logical_k = K if stage == 1 else b_k
    K_ITERS = logical_k // BLOCK_K
    b_pack = 2 if b_dtype == "fp4" else 1
    N_TILES_A, N_TILES_B = BLOCK_M // 64, BLOCK_N // (num_waves * 16)
    LDS_BLOCK_M, LDS_BLOCK_N = BLOCK_M // 2, BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // (num_waves * 8)
    N_LDS_STEPS_B = LDS_BLOCK_N // (num_waves * 8 * b_pack)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K // b_pack
    A_GRP_ROWS, B_GRP_ROWS = N_TILES_A * 16, N_TILES_B * 16
    if swiglu_limit is None:
        swiglu_limit = 0.0 if b_dtype == "fp4" else 7.0
    store_factory = _store_factory(
        activation=stage == 1,
        transpose=stage == 2,
        activation_type="silu" if b_dtype == "fp4" else "swiglu",
        swiglu_limit=swiglu_limit,
        fuse_quant=stage == 1 and b_dtype == "fp4",
    )

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(
        name="a8w4_gemm_4wave" if num_waves == 4 else None,
        known_block_size=[threads, 1, 1],
    )
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        expert_ids: fx.Tensor,
        row_map: fx.Tensor,
        valid_rows: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // (num_waves // 2)
        wave_n = wave_id % (num_waves // 2)
        # Partition actual work across XCDs, excluding the routing allocation tail.
        live_rows = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, valid_rows[0]))
        m_blocks = ceildiv(live_rows, BLOCK_M)
        if fx.block_idx.x < m_blocks * n_blocks:
            if const_expr(xcd_swizzle > 0):
                block_m, block_n = _xcd_swizzle_any(m_blocks, n_blocks, xcd_swizzle)
                simple_m, simple_n = split_row_major_2d(fx.block_idx.x, n_blocks)
                use_simple = m_blocks * n_blocks < 1024
                block_m = use_simple.select(simple_m, block_m)
                block_n = use_simple.select(simple_n, block_n)
            else:
                block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)
            b_row = block_n * BLOCK_N
            expert = rocdl.readfirstlane(fx.Int32.ir_type, expert_ids[block_m])
            b_row += fx.Int32(expert) * c_n

            A0_gl_offset = (block_m * BLOCK_M) * K
            A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
            B_K_STEP = 2048 // b_pack
            B0_gl_offset = b_row * (b_k // b_pack)
            B1_gl_offset = (b_row + LDS_BLOCK_N) * (b_k // b_pack)

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_STEPS_A, preshuffled=False
            )
            if const_expr(b_dtype == "fp4"):
                gl_off_b = [
                    (wave_id + step * num_waves) * (b_k // 2 * 16)
                    + lane_id % 16 * 16
                    + lane_id // 16 * 256
                    for step in range_constexpr(N_LDS_STEPS_B)
                ]
            else:
                gl_off_b = compute_global_swizzle(
                    lane_id, wave_id, b_k, N_LDS_STEPS_B, preshuffled=True
                )

            gl_off_a1 = gl_off_a
            if const_expr(stage == 1):
                offsets = []
                for half in range_constexpr(2):
                    part = []
                    for offset in gl_off_a:
                        row = block_m * BLOCK_M + half * LDS_BLOCK_M + offset // K
                        source_row = row_map[row]
                        part.append(source_row * K + offset % K)
                    offsets.append(part)
                gl_off_a, gl_off_a1 = offsets
                A0_gl_offset = A1_gl_offset = fx.Int32(0)

            mfma = make_mx_mfma(N_TILES_A, N_TILES_B, b_dtype)

            a_g2s = make_g2s_loader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            a1_g2s = make_g2s_loader(a_div, gl_off_a1, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = make_g2s_loader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = make_s2r_loader(wave_m, N_TILES_A)
            b_s2r = make_s2r_loader(wave_n, N_TILES_B, packed_fp4=b_dtype == "fp4")
            scratch = [
                a_cur0.ptr,
                a_cur1.ptr,
                a_next0.ptr,
                a_next1.ptr,
                b_cur0.ptr,
                b_cur1.ptr,
                b_next0.ptr,
                b_next1.ptr,
            ]
            store_c = store_factory(
                C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, scratch
            )

            a_sc = make_scale_preshuffled_s2r(A_scale, c_m, K, N_TILES_A)
            b_scale_rows = fx.size(B_scale.shape).unpack() // (K // 32)
            b_sc = make_scale_preshuffled_s2r(B_scale, b_scale_rows, K, N_TILES_B)
            # 16-row tile index of each wave's first row, per LDS half.
            a_base16 = [
                (block_m * BLOCK_M + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
                for h in range_constexpr(2)
            ]
            b_base16 = [
                (b_row + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16
                for h in range_constexpr(2)
            ]
            SCALE_LOADS_PER_PAIR = 2 * (N_TILES_A // 2) + 2 * (N_TILES_B // 2)
            assert SCALE_LOADS_PER_PAIR == 6, (
                f"scale prefetch issues {SCALE_LOADS_PER_PAIR} VMEM loads per K-pair, not the 6 the "
                "wait_barrier counts below were validated against; re-check them against the ISA"
            )

            pipeline_mma = make_mx_pipeline_mma(
                mfma, a_sc, b_sc, a_base16, b_base16, K_ITERS
            )
            pipeline_mma.prefetch(0)
            b_g2s.load(b_cur0, B0_gl_offset)
            a_g2s.load(a_cur0, A0_gl_offset)
            b_g2s.load(b_cur1, B1_gl_offset)
            a1_g2s.load(a_cur1, A1_gl_offset)
            if wave_m == 1:
                rocdl.s_barrier()

            c00_frag, c01_frag, c10_frag, c11_frag = run_8wave_pipeline(
                lds,
                (a_g2s, a1_g2s),
                b_g2s,
                a_s2r,
                b_s2r,
                (A0_gl_offset, A1_gl_offset),
                (B0_gl_offset, B1_gl_offset),
                pipeline_mma,
                k_iters=K_ITERS,
                b_k_step=B_K_STEP,
                b_preshuffled=True,
                # Grouped/gathered A must complete before LDS buffer rotation.
                loop_wait_count=N_LDS_STEPS_B,
                # With no main loop, the final B prefetch needs its VMEM fence.
                tail_a1_fence=K_ITERS == 2,
            )

            # Rejoin the staggered M wave groups before reusing LDS for the epilogue.
            if wave_m == 0:
                rocdl.s_barrier()
            wave_n_offset = wave_n * (N_TILES_B * 16)
            wave_m_offset = wave_m * (N_TILES_A * 16)
            base_row = block_m * BLOCK_M + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset

            store_c(c00_frag, base_row + 0, base_col + 0)
            store_c(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
            store_c(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
            store_c(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dynamic_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        expert_ids: fx.Tensor,
        row_map: fx.Tensor,
        valid_rows: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            expert_ids,
            row_map,
            valid_rows,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{threads},{threads}",
            },
        ).launch(
            grid=(ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N), 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch_dynamic_gemm


def _run_resident_b_pipeline(
    b_tiles0,
    b_tiles1,
    a_cur0,
    a_cur1,
    a_next0,
    a_next1,
    a_g2s,
    a_s2r,
    b_s2r,
    a_offsets,
    mma,
    store_c,
    base_row,
    base_col,
):
    count = mma.n_tiles_a * mma.n_tiles_b
    c00, c01, c10, c11 = ([mma.zero_value] * count for _ in range(4))
    # Emit the final group stores as soon as their accumulators retire.
    for k in range_constexpr(3):
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        rocdl.s_barrier()
        if const_expr(k < 2):
            mma.prefetch(k + 1)
            a_g2s.load(a_next0, a_offsets[0] + (k + 1) * 128)
            a_g2s.load(a_next1, a_offsets[1] + (k + 1) * 128)
        a0 = a_s2r.load(a_cur0)
        b0 = b_s2r.load(b_tiles0[k], preshuffled=True)
        rocdl.s_setprio(1)
        c00 = mma.call(a0, b0, c00, k=k, a_half=0, b_half=0, set_prio=False)
        if const_expr(k == 2):
            store_c(c00, base_row, base_col)
        b1 = b_s2r.load(b_tiles1[k], preshuffled=True)
        c01 = mma.call(a0, b1, c01, k=k, a_half=0, b_half=1, set_prio=False)
        if const_expr(k == 2):
            store_c(c01, base_row, base_col + 128)
        a1 = a_s2r.load(a_cur1)
        c10 = mma.call(a1, b0, c10, k=k, a_half=1, b_half=0, set_prio=False)
        if const_expr(k == 2):
            store_c(c10, base_row + 128, base_col)
        c11 = mma.call(a1, b1, c11, k=k, a_half=1, b_half=1, set_prio=False)
        if const_expr(k == 2):
            store_c(c11, base_row + 128, base_col + 128)
        rocdl.s_setprio(0)
        if const_expr(k < 2):
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()


def _load_b_if_needed(condition, b_g2s, bs0, bs1, b_row, b_k):
    def issue():
        for k in range_constexpr(3):
            b_g2s.load(bs0[k], b_row * b_k + k * 2048)
            b_g2s.load(bs1[k], (b_row + 128) * b_k + k * 2048)

    @flyc.jit
    def dispatch():
        if condition:
            issue()

    dispatch()


def _compile_persistent_gemm(*, xcd_swizzle, m_tiles):
    """K384 down projection with resident B and per-tile expert/row guards."""
    K, b_k = 512, 384
    BLOCK_M = BLOCK_N = 256
    LDS_BLOCK_M = LDS_BLOCK_N = 128
    N_TILES_A, N_TILES_B = 4, 2
    N_LDS_STEPS_A = N_LDS_STEPS_B = 2
    A_GRP_ROWS, B_GRP_ROWS = 64, 32
    a_lds_size = b_lds_size = 16384
    store_factory = _store_factory(transpose=True)
    resident_pipeline = _run_resident_b_pipeline
    load_resident_b = _load_b_if_needed
    make_pipeline_mma = make_mx_pipeline_mma

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_extra0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_extra1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(
        name=f"mxfp8_moe_gemm2_persistent{m_tiles}", known_block_size=[512, 1, 1]
    )
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        expert_ids: fx.Tensor,
        row_map: fx.Tensor,
        valid_rows: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # Partition actual work across XCDs, excluding the routing allocation tail.
        live_rows = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, valid_rows[0]))
        m_blocks = ceildiv(live_rows, m_tiles * BLOCK_M)
        if fx.block_idx.x < m_blocks * n_blocks:
            if const_expr(xcd_swizzle > 0):
                block_m, block_n = _xcd_swizzle_any(m_blocks, n_blocks, xcd_swizzle)
                simple_m, simple_n = split_row_major_2d(fx.block_idx.x, n_blocks)
                use_simple = m_blocks * n_blocks < 1024
                block_m = use_simple.select(simple_m, block_m)
                block_n = use_simple.select(simple_n, block_n)
            else:
                block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)
            a_div = fx.logical_divide(
                make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
            )
            b_div = fx.logical_divide(
                make_fp8_buffer_tensor(B_T, F8_IR_t), fx.make_layout(1, 1)
            )
            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_STEPS_A, preshuffled=False
            )
            gl_off_b = compute_global_swizzle(
                lane_id, wave_id, b_k, N_LDS_STEPS_B, preshuffled=True
            )
            a_g2s = make_g2s_loader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = make_g2s_loader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = make_s2r_loader(wave_m, N_TILES_A)
            b_s2r = make_s2r_loader(wave_n, N_TILES_B)
            b_tiles0 = [b_cur0, b_next0, lds.B_extra0]
            b_tiles1 = [b_cur1, b_next1, lds.B_extra1]
            mfma = make_mx_mfma(N_TILES_A, N_TILES_B)
            a_sc = make_scale_preshuffled_s2r(A_scale, c_m, K, N_TILES_A)
            b_scale_rows = fx.size(B_scale.shape).unpack() // (K // 32)
            b_sc = make_scale_preshuffled_s2r(B_scale, b_scale_rows, K, N_TILES_B)

            def process_m_tile(m_repeat, previous_expert):
                bm = block_m * m_tiles + m_repeat
                expert = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, expert_ids[bm]))
                b_row = block_n * BLOCK_N + expert * c_n
                load_resident_b(
                    expert != previous_expert, b_g2s, b_tiles0, b_tiles1, b_row, b_k
                )
                a_offsets = [bm * BLOCK_M * K, (bm * BLOCK_M + LDS_BLOCK_M) * K]
                a_g2s.load(a_cur0, a_offsets[0])
                a_g2s.load(a_cur1, a_offsets[1])
                a_base16 = [
                    (bm * BLOCK_M + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
                    for h in range_constexpr(2)
                ]
                b_base16 = [
                    (b_row + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16
                    for h in range_constexpr(2)
                ]
                pipeline_mma = make_pipeline_mma(
                    mfma, a_sc, b_sc, a_base16, b_base16, 3
                )
                pipeline_mma.prefetch(0)
                # K2 uses the original A-current fields. The K1 fields have
                # retired at its opening barrier and are wave-private scratch;
                # B stays resident for the next M tile.
                scratch = [a_next0.ptr + i * 4096 for i in range_constexpr(4)]
                scratch += [a_next1.ptr + i * 4096 for i in range_constexpr(4)]
                store_c = store_factory(
                    C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, scratch
                )
                base_row = bm * BLOCK_M + wave_m * A_GRP_ROWS
                base_col = block_n * BLOCK_N + wave_n * B_GRP_ROWS
                resident_pipeline(
                    b_tiles0,
                    b_tiles1,
                    a_cur0,
                    a_cur1,
                    a_next0,
                    a_next1,
                    a_g2s,
                    a_s2r,
                    b_s2r,
                    a_offsets,
                    pipeline_mma,
                    store_c,
                    base_row,
                    base_col,
                )

            process_m_tile(0, fx.Int32(-1))
            for repeat in range_constexpr(1, m_tiles):
                if (block_m * m_tiles + repeat) * BLOCK_M < live_rows:
                    previous_expert = fx.Int32(
                        rocdl.readfirstlane(
                            fx.Int32.ir_type,
                            expert_ids[block_m * m_tiles + repeat - 1],
                        )
                    )
                    process_m_tile(repeat, previous_expert)

    @flyc.jit
    def launch_dynamic_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        expert_ids: fx.Tensor,
        row_map: fx.Tensor,
        valid_rows: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            expert_ids,
            row_map,
            valid_rows,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(
            grid=(ceildiv(c_m, m_tiles * BLOCK_M) * ceildiv(c_n, BLOCK_N), 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch_dynamic_gemm


def compile_mxfp8_moe_sort_input_scale(*, K: int, topk: int):
    """Sort per-token E8M0 bytes and unpack routes in one pass.

    Source scales use unshuffled [tokens, K // 32] bytes. Output uses the
    GEMM shuffled layout, viewed as int32 for coalesced stores. Padding scales
    are 127 and padding row-map entries are -1. Rows and the GPU valid count
    must be padded to 32; rows beyond the valid count remain untouched.
    """
    groups = K // 32
    assert K > 0 and K % 256 == 0 and topk > 0

    @flyc.kernel(name=f"mxfp8_moe_sort_input_scale_k{K}", known_block_size=[256, 1, 1])
    def kernel(
        source: fx.Tensor,
        output: fx.Tensor,
        ids: fx.Tensor,
        row_map: fx.Tensor,
        valid_rows: fx.Tensor,
        tokens: fx.Int32,
    ):
        lane = fx.thread_idx.x % 64
        wave = fx.thread_idx.x // 64
        row = fx.block_idx.x * 32 + lane % 32
        kg_base = wave * 8
        if row < valid_rows[0]:
            src = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(source, max_size=False),
                fx.make_layout(4, 1),
            )
            load = fx.make_copy_atom(rocdl.BufferCopy32b(), fx.Uint8)
            a = ids[row]
            ta = a & 0xFFFFFF
            valid = (ta < tokens) & (((a >> 24) & 0xFF) < topk)
            for step in range_constexpr((groups + 31) // 32):
                kg = kg_base + step * 32
                if kg < groups:
                    index = valid.select(
                        (ta * groups + kg + lane // 32 * 4) // 4, fx.Int32(-1)
                    )
                    reg = fx.make_rmem_tensor(4, fx.Uint8)
                    fx.copy(load, fx.slice(src, (None, index)), reg)
                    # Each lane loads four adjacent scales for one row.
                    # The wave then transposes them into the four-byte shuffled
                    # word: [row, row+16] x [group, group+4].
                    value = valid.select(reg.load().bitcast(fx.Int32)[0], 0x7F7F7F7F)
                    shift = lane // 16 * 8
                    s0 = (fx.gpu.shuffle_idx(value, lane % 16, 64) >> shift) & 255
                    s1 = (fx.gpu.shuffle_idx(value, lane % 16 + 16, 64) >> shift) & 255
                    s2 = (fx.gpu.shuffle_idx(value, lane % 16 + 32, 64) >> shift) & 255
                    s3 = (fx.gpu.shuffle_idx(value, lane % 16 + 48, 64) >> shift) & 255
                    word = fx.block_idx.x * groups * 8 + fx.thread_idx.x + step * 256
                    output[word] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)
            if fx.thread_idx.x < 32:
                row_map[row] = valid.select(ta, fx.Int32(-1))

    @flyc.jit
    def launch(
        source: fx.Tensor,
        output: fx.Tensor,
        ids: fx.Tensor,
        row_map: fx.Tensor,
        valid_rows: fx.Tensor,
        tokens: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(source, output, ids, row_map, valid_rows, tokens).launch(
            grid=((rows + 31) // 32, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def compile_mxfp8_moe_quant(
    *, K: int, gather: bool, scatter_scale_topk: int = 0, dynamic_rows=False
):
    """Fuse BF16 gather, K padding and quantization into the GEMM scale layout.

    ``row_map`` contains source row indices and -1 for expert padding. With
    ``gather=False`` normally quantizes already-sorted stage-1 output. With
    ``scatter_scale_topk > 0``, quantize source tokens once and scatter their
    scales using the inverse map ``row_map[tokens, topk]``; initialize padded
    scale rows to 127 before launching.
    """
    assert K > 0 and K % 32 == 0
    assert not (gather and scatter_scale_topk)
    kp = (K + 255) // 256 * 256
    groups = kp // 32

    @flyc.kernel(
        name=f"mxfp8_moe_quant_k{K}_gather{int(gather)}", known_block_size=[256, 1, 1]
    )
    def kernel(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        row_map: fx.Tensor,
        rows: fx.Int32,
        valid_rows: fx.Tensor,
    ):
        inp = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(x, max_size=False), fx.make_layout(8, 1)
        )
        out = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(y, max_size=False), fx.make_layout(16, 1)
        )
        scales = fx.rocdl.make_buffer_tensor(scale, max_size=False)
        load = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        store = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int8)
        group = fx.block_idx.x * 256 + fx.thread_idx.x
        row, kg = group // groups, group % groups
        limit = valid_rows[0] if dynamic_rows else rows
        if row < limit:
            src_row = row_map[row] if gather else row
            valid = (src_row >= 0) & (kg < K // 32)
            values = []
            amax = fx.Float32(1e-30)
            for chunk in range_constexpr(4):
                offset = valid.select(src_row * (K // 8) + kg * 4 + chunk, fx.Int32(-1))
                reg = fx.make_rmem_tensor(8, fx.BFloat16)
                fx.copy(load, fx.slice(inp, (None, offset)), reg)
                v = reg.load().to(fx.Float32)
                amax = fx.max(amax, fmath.absf(v).reduce(ReductionOp.MAX))
                values.append(v)
            bits = (amax * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
            exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(fx.Int32)
            inv = ((254 - exponent) << 23).bitcast(fx.Float32)
            # [row/32, kg/8, kg%4, row%16, kg/4%2, row/16%2]
            for slot in range_constexpr(max(1, scatter_scale_topk)):
                sr = (
                    row_map[row * scatter_scale_topk + slot]
                    if scatter_scale_topk
                    else row
                )
                scale_index = (
                    (sr // 32 * (kp // 256) + kg // 8) * 64 + kg % 4 * 16 + sr % 16
                ) * 4
                scale_index += kg // 4 % 2 * 2 + sr // 16 % 2
                scales[(sr >= 0).select(scale_index, fx.Int32(-1))] = exponent.to(
                    fx.Uint8
                )
            for half in range_constexpr(2):
                words = []
                for word in range_constexpr(4):
                    v = values[half * 2 + word // 2]
                    start = word % 2 * 4
                    packed = rocdl.cvt_pk_fp8_f32(
                        T.i32, v[start] * inv, v[start + 1] * inv, fx.Int32(0), 0
                    )
                    packed = rocdl.cvt_pk_fp8_f32(
                        T.i32, v[start + 2] * inv, v[start + 3] * inv, packed, 1
                    )
                    words.append(packed)
                reg = fx.make_rmem_tensor(16, fx.Int8)
                reg.store(Vec.from_elements(words, fx.Int32).bitcast(fx.Int8))
                fx.copy(store, reg, fx.slice(out, (None, group * 2 + half)))

    @flyc.jit
    def launch(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        row_map: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(x, y, scale, row_map, rows, row_map).launch(
            grid=((rows * groups + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_dynamic(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        row_map: fx.Tensor,
        rows: fx.Int32,
        valid_rows: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(x, y, scale, row_map, rows, valid_rows).launch(
            grid=((rows * groups + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_dynamic if dynamic_rows else launch


def compile_mxfp8_moe_reduce(*, N: int, topk: int, sorted_weights=False):
    """Gather sorted down projections and sum routing-weighted top-k in FP32."""
    assert N > 0 and N % 8 == 0 and topk > 0

    @flyc.kernel(name=f"mxfp8_moe_reduce_n{N}_topk{topk}", known_block_size=[256, 1, 1])
    def kernel(
        x: fx.Tensor,
        y: fx.Tensor,
        inverse: fx.Tensor,
        weights: fx.Tensor,
        tokens: fx.Int32,
    ):
        inp = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(x, max_size=False), fx.make_layout(8, 1)
        )
        out = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(y, max_size=False), fx.make_layout(8, 1)
        )
        wr = fx.rocdl.make_buffer_tensor(weights, max_size=False)
        atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        linear = fx.block_idx.x * 256 + fx.thread_idx.x
        token, col = linear // (N // 8), linear % (N // 8) * 8
        if token < tokens:
            acc = Vec.filled(8, 0.0, fx.Float32)
            for slot in range_constexpr(topk):
                index = token * topk + slot
                row = inverse[index]
                weight_index = row if sorted_weights else index
                weight = wr[weight_index]
                reg = fx.make_rmem_tensor(8, fx.BFloat16)
                # Missing routes use negative offsets, beyond the descriptor even for buffers >2 GiB.
                fx.copy(atom, fx.slice(inp, (None, row * (N // 8) + col // 8)), reg)
                acc = acc + reg.load().to(fx.Float32) * weight
            reg = fx.make_rmem_tensor(8, fx.BFloat16)
            reg.store(acc.to(fx.BFloat16))
            fx.copy(atom, reg, fx.slice(out, (None, linear)))

    @flyc.jit
    def launch(
        x: fx.Tensor,
        y: fx.Tensor,
        inverse: fx.Tensor,
        weights: fx.Tensor,
        tokens: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(x, y, inverse, weights, tokens).launch(
            grid=((tokens * (N // 8) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


def compile_mxfp8_moe_unpack_routes(*, topk: int, dynamic_rows=False):
    """Convert the existing MoE sorter's packed IDs into GEMM routing maps.

    The caller supplies the actual padded row count from the sorter. Its upper
    eight bits encode the top-k slot, and its lower 24 bits encode the token.
    In dynamic mode, ``rows`` is the grid upper bound and an additional GPU
    valid-row tensor follows ``tokens`` in the launcher arguments.
    """

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel(
        packed: fx.Tensor,
        row_map: fx.Tensor,
        inverse: fx.Tensor,
        rows: fx.Int32,
        tokens: fx.Int32,
        valid_rows: fx.Tensor,
    ):
        row = fx.block_idx.x * 256 + fx.thread_idx.x
        limit = valid_rows[0] if dynamic_rows else rows
        if row < limit:
            value = packed[row]
            token, slot = value & 0xFFFFFF, (value >> 24) & 0xFF
            valid = (token < tokens) & (slot < topk)
            row_map[row] = valid.select(token, fx.Int32(-1))
            if valid:
                inverse[token * topk + slot] = row

    @flyc.jit
    def launch(
        packed: fx.Tensor,
        row_map: fx.Tensor,
        inverse: fx.Tensor,
        rows: fx.Int32,
        tokens: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(packed, row_map, inverse, rows, tokens, packed).launch(
            grid=((rows + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_dynamic(
        packed: fx.Tensor,
        row_map: fx.Tensor,
        inverse: fx.Tensor,
        rows: fx.Int32,
        tokens: fx.Int32,
        valid_rows: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(packed, row_map, inverse, rows, tokens, valid_rows).launch(
            grid=((rows + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_dynamic if dynamic_rows else launch
