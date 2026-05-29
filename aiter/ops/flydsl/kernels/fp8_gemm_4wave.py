
"""4-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_4wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1).

Global IO, scale loads, and bf16 stores go through the layout API
(``fx.rocdl.make_buffer_tensor`` + ``fx.copy`` with ``BufferCopyLDS128b``
/ ``BufferCopy{16,32,128}b``). MFMAs use ``fly.mma_atom_call_ssa`` so
the chained Vec(4, f32) accumulator stays on AGPR. The XOR swizzle and
the 8-buffer LDS pipeline ping-pong are kept as direct arithmetic to
preserve the original kernel's interleaved-cluster scheduling.

Optional B preshuffle uses the same on-disk layout as
``preshuffle_gemm_v2`` / ``shuffle_weight((16, 16))``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr
from .fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    StoreC,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)


def _divmod(a, b):
    return (a // b, a % b)


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x

    num_wg = num_pid_m * num_pid_n

    if num_wg <= SWIZZLE_THRESHOLD or num_wg % NUM_XCDS != 0:
        return _divmod(wgid, num_pid_n)

    intra_xcd, xcd = _divmod(wgid, NUM_XCDS)
    wgid = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = _divmod(wgid, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


def compile_fp8_gemm_4w(
    *,
    M: int,
    N: int,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    use_xcd_remap: bool = True,
    b_preshuffled: bool = False,
):
    # MFMA atom is 16x16x128; 4 waves in a 2x2 config require BLOCK >= 64.
    BLOCK_K = 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    assert M >= 1 and N >= 1
    assert BLOCK_M >= 64 and BLOCK_M % 64 == 0 and BLOCK_N >= 64 and BLOCK_N % 64 == 0
    assert K % BLOCK_K == 0

    N_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
    K_ITERS = K // BLOCK_K
    # Number of 16-row 16x128 tiles per wave per A/B partition.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    _use_interleaved_block = BLOCK_M == 256 and BLOCK_N == 256

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

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

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

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

        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle((M + BLOCK_M - 1) // BLOCK_M, N_BLOCKS)
        else:
            tile_i, tile_j = _divmod(fx.block_idx.x, N_BLOCKS)

        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K
        A_K_STEP = BLOCK_K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        def _compute_lds_swizzle(s2r, preshuffled=False):
            lds_swz = []
            for row_offset in range_constexpr(s2r.n_tiles):
                row = s2r.wave_idx * (s2r.n_tiles * 16) + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    if const_expr(preshuffled):
                        swz.append((row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128)
                    else:
                        r, c = swizzle_128(row, col)
                        swz.append(r * BLOCK_K + c)
                lds_swz.append(swz)
            return lds_swz

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        def _interleaved_cluster(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            rt_dst = []

            c[mfma.idx(0, 0)] = mfma.call_one(a, b, c, 0, 0)
            c[mfma.idx(0, 1)] = mfma.call_one(a, b, c, 0, 1)

            lds_swz = _compute_lds_swizzle(s2r, preshuffled=lds_src_preshuffled)
            g2s.load_one(lds_dst, k_offset, 0)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[0][0])

            c[mfma.idx(0, 2)] = mfma.call_one(a, b, c, 0, 2)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[0][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(0, 3)] = mfma.call_one(a, b, c, 0, 3)

            g2s.load_one(lds_dst, k_offset, 1)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[1][0])

            c[mfma.idx(1, 0)] = mfma.call_one(a, b, c, 1, 0)
            c[mfma.idx(1, 1)] = mfma.call_one(a, b, c, 1, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[1][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(1, 2)] = mfma.call_one(a, b, c, 1, 2)
            c[mfma.idx(1, 3)] = mfma.call_one(a, b, c, 1, 3)

            g2s.load_one(lds_dst, k_offset, 2)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[2][0])

            c[mfma.idx(2, 0)] = mfma.call_one(a, b, c, 2, 0)
            c[mfma.idx(2, 1)] = mfma.call_one(a, b, c, 2, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[2][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(2, 2)] = mfma.call_one(a, b, c, 2, 2)
            c[mfma.idx(2, 3)] = mfma.call_one(a, b, c, 2, 3)

            g2s.load_one(lds_dst, k_offset, 3)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[3][0])

            c[mfma.idx(3, 0)] = mfma.call_one(a, b, c, 3, 0)
            c[mfma.idx(3, 1)] = mfma.call_one(a, b, c, 3, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[3][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(3, 2)] = mfma.call_one(a, b, c, 3, 2)
            c[mfma.idx(3, 3)] = mfma.call_one(a, b, c, 3, 3)

            return c, rt_dst

        def _compute_cluster(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            g2s.load(lds_dst, k_offset)
            rt_dst = s2r.load(lds_src, preshuffled=lds_src_preshuffled)
            c = mfma.call(a, b, c)
            return c, rt_dst

        def _compute_block(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(
                    lds_dst,
                    g2s,
                    k_offset,
                    s2r,
                    lds_src,
                    a,
                    b,
                    c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )
            else:
                return _compute_cluster(
                    lds_dst,
                    g2s,
                    k_offset,
                    s2r,
                    lds_src,
                    a,
                    b,
                    c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )

        # Each wave handles 2x2 64x64 sub-tiles of the output.
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled)

        a_g2s = G2SLoader(ga_div, gl_off_a, N_TILES_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(gb_div, gl_off_b, N_TILES_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)
        store_c = StoreC(A_scale, B_scale, C, M, N, mfma.idx, N_TILES_A, N_TILES_B)

        # Prologue: 8-buffer LDS pipeline pre-fill.
        a_g2s.load(a_cur0, A0_gl_offset + 0 * A_K_STEP)
        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * A_K_STEP)

        a_g2s.load(a_next0, A0_gl_offset + 1 * A_K_STEP)
        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next1, A1_gl_offset + 1 * A_K_STEP)

        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))

        a0_frag = a_s2r.load(a_cur0)

        wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))

        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)

        for k in range_constexpr(K_ITERS - 2):
            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c00_frag, b1_frag = _compute_block(
                a_cur0,
                a_g2s,
                A0_gl_offset + (k + 2) * A_K_STEP,
                b_s2r,
                b_cur1,
                a0_frag,
                b0_frag,
                c00_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            c01_frag, a1_frag = _compute_block(
                b_cur0,
                b_g2s,
                B0_gl_offset + (k + 2) * B_K_STEP,
                a_s2r,
                a_cur1,
                a0_frag,
                b1_frag,
                c01_frag,
            )

            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c10_frag, a0_frag = _compute_block(
                b_cur1,
                b_g2s,
                B1_gl_offset + (k + 2) * B_K_STEP,
                a_s2r,
                a_next0,
                a1_frag,
                b0_frag,
                c10_frag,
            )

            c11_frag, b0_frag = _compute_block(
                a_cur1,
                a_g2s,
                A1_gl_offset + (k + 2) * A_K_STEP,
                b_s2r,
                b_next0,
                a1_frag,
                b1_frag,
                c11_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 2.
        wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        a1_frag = a_s2r.load(a_cur1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0_frag = a_s2r.load(a_next0)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 1.
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        wait_barrier(0)
        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        a1_frag = a_s2r.load(a_cur1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        grid_x = ((M + BLOCK_M - 1) // BLOCK_M) * N_BLOCKS
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
