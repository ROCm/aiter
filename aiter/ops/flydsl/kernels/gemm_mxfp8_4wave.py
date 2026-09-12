# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Four-wave 128x256 A8W4 MoE GEMM for gfx950, using the shared K-loop."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
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
from .gemm_mxfp8_8wave import (
    make_mx_mfma,
    make_mx_pipeline_mma,
    make_scale_preshuffled_s2r,
)
from .kernels_common import get_warp_size
from .mfma_preshuffle_pipeline import split_row_major_2d

BLOCK_K = 128
LDS_LIMIT_BYTES = 160 * 1024


def compile_mxfp8_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 128,
    BLOCK_N: int = 256,
    xcd_swizzle: int = 0,
    store_factory,
    logical_k: int | None = None,
    gather_a: bool = False,
    expert_block_m: int | None = None,
    b_k: int | None = None,
):
    """Compile the four-wave grouped tile with GPU-resident row bounds.

    Two M waves and two N waves share the existing staggered K-loop. The
    four-wave DMA geometry stages 64 A rows and 128 packed B rows per half,
    using 64 KiB LDS. Each wave owns 32 output activation columns, so its
    fused SiLU/FP8 epilogue can quantize complete groups without another wave.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith(
        "gfx950"
    ), f"A8W4 four-wave GEMM requires gfx950 (CDNA4), got {arch}"
    assert (
        get_warp_size() == 64
    ), f"A8W4 four-wave GEMM assumes wave64, got {get_warp_size()}"

    assert (BLOCK_M, BLOCK_N) == (128, 256), "four-wave A8W4 requires tile128x256"
    expert_block_m = BLOCK_M if expert_block_m is None else expert_block_m
    assert expert_block_m % BLOCK_M == 0
    assert (
        K % 256 == 0
    ), f"K must be a multiple of 256 (MX scale chunk staging), got {K}"

    logical_k = K if logical_k is None else logical_k
    assert 256 <= logical_k <= K and logical_k % BLOCK_K == 0
    b_k = K if b_k is None else b_k
    assert logical_k <= b_k <= K and b_k % 128 == 0
    b_pack = 2
    K_ITERS = logical_k // BLOCK_K
    # Scale words are addressed by K-pair, so K must contain a whole number of
    # them (K % 256 above already guarantees it).

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 64

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 32
    N_LDS_STEPS_B = LDS_BLOCK_N // (32 * b_pack)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K // b_pack

    A_GRP_ROWS = N_TILES_A * 16  # rows one wave_m owns inside one LDS half
    B_GRP_ROWS = N_TILES_B * 16

    lds_bytes = 4 * (a_lds_size + b_lds_size)
    assert (
        lds_bytes <= LDS_LIMIT_BYTES
    ), f"LDS {lds_bytes} B exceeds {LDS_LIMIT_BYTES} B"

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

    @flyc.kernel(name="a8w4_gemm_4wave", known_block_size=[256, 1, 1])
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
        wave_m = wave_id // 2
        wave_n = wave_id % 2
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
            expert = rocdl.readfirstlane(
                fx.Int32.ir_type, expert_ids[block_m // (expert_block_m // BLOCK_M)]
            )
            b_row += fx.Int32(expert) * c_n

            A0_gl_offset = (block_m * BLOCK_M) * K
            A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
            B_K_STEP = 1024
            B0_gl_offset = b_row * (b_k // b_pack)
            B1_gl_offset = (b_row + LDS_BLOCK_N) * (b_k // b_pack)

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_STEPS_A, preshuffled=False
            )
            gl_off_b = [
                (wave_id + step * 4) * (b_k // 2 * 16)
                + lane_id % 16 * 16
                + lane_id // 16 * 256
                for step in range_constexpr(N_LDS_STEPS_B)
            ]

            gl_off_a1 = gl_off_a
            if const_expr(gather_a):
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

            mfma = make_mx_mfma(N_TILES_A, N_TILES_B, "fp4")

            a_g2s = make_g2s_loader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            a1_g2s = make_g2s_loader(a_div, gl_off_a1, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = make_g2s_loader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = make_s2r_loader(wave_m, N_TILES_A)
            b_s2r = make_s2r_loader(wave_n, N_TILES_B, packed_fp4=True)
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
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=(ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_dynamic_gemm
