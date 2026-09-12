# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave MXFP8/A8W4 matmul for AMD CDNA4 (gfx950 / MI355X)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from ..gemm_a8w8_8wave import (
    G2SLoader,
    S2RLoader,
    _xcd_swizzle_any,
    ceildiv,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from ..kernels_common import get_warp_size
from ..mfma_preshuffle_pipeline import split_row_major_2d

# 1 block = 512 threads = 8 waves (2 in M x 4 in N); the LDS budget below only
# closes for this shape, see ``compile_mxfp8_gemm_8w``.
BLOCK_K = 128
LDS_LIMIT_BYTES = 160 * 1024


class ScalePreshuffledS2R:
    """Coalesced reader for ``shuffle_scale_w4``-packed E8M0 -- no LDS staging."""

    def __init__(self, scale_arg, rows, K, n_tiles):
        assert n_tiles % 2 == 0, "shuffle_scale_w4 pairs tiles two at a time"
        self.n_pairs = n_tiles // 2
        self.k1_stride = K // 256  # i32 groups of 64 per 32-row super-row
        self.lane = fx.thread_idx.x % 64
        # Same byte count as the raw layout, just permuted.
        t_i8 = fx.rocdl.make_buffer_tensor(
            scale_arg,
            max_size=False,
            num_records_bytes=fx.Int64(rows) * fx.Int64(K // 32),
        )
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=fx.rocdl.TargetAddressSpace.BufferDesc,
            alignment=4,
        )
        iter_i32 = fx.recast_iter(i32_ptr, fx.get_iter(t_i8))
        n_i32 = fx.Int32(rows) * fx.Int32(K // 128)
        self.g_div = fx.logical_divide(
            fx.Tensor(fx.make_view(iter_i32, fx.make_layout(n_i32, 1))),
            fx.make_layout(1, 1),
        )
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        # Two register sets: the caller prefetches the next K-pair while the
        # current one is still feeding MFMAs.
        self.regs = [
            [
                fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                for _ in range_constexpr(self.n_pairs)
            ]
            for _ in range_constexpr(2)
        ]

    def read(self, row_base16, k):
        """``n_tiles`` scale operands for K-step ``k``; tiles of a pair share one."""
        k1 = k // 2  # compile-time; k % 2 is the k_pack the opsel encodes
        regs = self.regs[k1 % 2]
        words = []
        for p in range_constexpr(self.n_pairs):
            # row_base16 is even (wave offsets are multiples of 32 rows), so the
            # 32-row super-row is row_base16 // 2 + p and n_pack is the tile parity.
            n1 = row_base16 // 2 + p
            base = fx.rocdl.readfirstlane(
                fx.Int32.ir_type, (n1 * self.k1_stride + k1) * 64
            )
            fx.copy(
                self.atom,
                fx.slice(self.g_div, (None, fx.Int32(base) + self.lane)),
                regs[p],
            )
            w = fx.Int32(regs[p].load()[0])
            words += [w, w]
        return words


class PackedFp4S2R(S2RLoader):
    """Read a packed 16x128 FP4 tile from the standard AITER weight layout."""

    def load(self, lds_src, preshuffled=True):
        fragments = []
        for i in range_constexpr(self.n_tiles):
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            offset = row // 16 * 1024 + row % 16 * 16 + self.lane_id // 16 * 256
            fragments.append(self._vec_load_16xf8(lds_src, offset).bitcast(fx.Int32))
        return fragments


class MxMfma:
    """16x16x128 scaled MFMA with per-tile packed scales and byte selectors.

    One ``(opsel_a, opsel_b)`` atom per byte pair: in the ``shuffle_scale_w4``
    layout a single i32 carries the E8M0 of two 16-row tiles x two K-steps, and
    the byte is picked by ``opsel`` -- a compile-time atom field -- so the hot
    loop emits no byte-select instructions at all.
    """

    def __init__(self, n_tiles_a, n_tiles_b, b_dtype="fp8"):
        # opsel = k_pack * 2 + tile_in_pair, so both operands share k_pack.
        self.atoms = {
            (kp * 2 + ia, kp * 2 + jb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16,
                    16,
                    128,
                    fx.Float8E4M3FN,
                    fx.Float4E2M1FN if b_dtype == "fp4" else fx.Float8E4M3FN,
                    opsel_a=kp * 2 + ia,
                    opsel_b=kp * 2 + jb,
                )
            )
            for kp in range_constexpr(2)
            for ia in range_constexpr(2)
            for jb in range_constexpr(2)
        }
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.b_words = 4 if b_dtype == "fp4" else 8

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _operand(self, value, words=8):
        frag = fx.make_rmem_tensor(words, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _accum(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def _atom_for(self, k_pack, i, j):
        return self.atoms[(k_pack * 2 + i % 2, k_pack * 2 + j % 2)]

    def call(self, a, b, c, sa, sb, *, k_pack, set_prio=True):
        assert len(a) == self.n_tiles_a and len(sa) == self.n_tiles_a
        assert len(b) == self.n_tiles_b and len(sb) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        a_frags = [self._operand(a[i]) for i in range_constexpr(self.n_tiles_a)]
        b_frags = [
            self._operand(b[j], self.b_words) for j in range_constexpr(self.n_tiles_b)
        ]
        c_frags = [
            self._accum(c[i]) for i in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                atom = self._atom_for(k_pack, i, j)
                fx.gemm(
                    atom, cf, a_frags[i], b_frags[j], cf, scale_a=sa[i], scale_b=sb[j]
                )
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [
            c_frags[i].load().ir_value()
            for i in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]


def compile_mxfp8_gemm_8w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    b_preshuffled: bool = False,
    xcd_swizzle: int = 0,
    grouped: bool = False,
    store_factory,
    logical_k: int | None = None,
    gather_a: bool = False,
    expert_block_m: int | None = None,
    b_k: int | None = None,
    dynamic_rows: bool = False,
    b_dtype: str = "fp8",
):
    """Build an FP8-activation launcher with FP8 or packed FP4 weights.

    ``grouped`` accepts expert-sorted, 256-row-padded A and one expert ID per
    M tile. B and its scales contain consecutive experts. The grouped launcher
    takes ``expert_ids`` and ``row_map`` after the two scale tensors; the dense
    API is unchanged. ``gather_a`` reads A using the sorted-to-source row map.
    ``store_factory`` lets MoE reuse the compute pipeline with a fused epilogue.
    ``b_dtype="fp4"`` uses E2M1 pairs in the standard 16x64-byte weight layout.
    ``b_k`` is the logical weight stride before FP4 packing and can differ from
    the padded A/scale stride.
    ``dynamic_rows`` adds a GPU valid-row tensor after ``row_map``; c_m remains
    the allocation/grid upper bound and inactive CTAs never read expert IDs.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith(
        "gfx950"
    ), f"MXFP8 8-wave GEMM requires gfx950 (CDNA4), got {arch}"
    assert (
        get_warp_size() == 64
    ), f"MXFP8 8-wave GEMM assumes wave64, got {get_warp_size()}"

    assert (BLOCK_M, BLOCK_N) in (
        (256, 256),
        (128, 512),
    ), "supported tiles: 256x256 and 128x512"
    expert_block_m = BLOCK_M if expert_block_m is None else expert_block_m
    assert expert_block_m % BLOCK_M == 0
    assert (
        K % 256 == 0
    ), f"K must be a multiple of 256 (MX scale chunk staging), got {K}"

    logical_k = K if logical_k is None else logical_k
    assert 256 <= logical_k <= K and logical_k % BLOCK_K == 0
    b_k = K if b_k is None else b_k
    assert logical_k <= b_k <= K and b_k % 64 == 0
    assert not dynamic_rows or grouped
    assert b_dtype in ("fp8", "fp4")
    assert b_dtype != "fp4" or b_preshuffled
    b_pack = 2 if b_dtype == "fp4" else 1
    assert b_k % (64 * b_pack) == 0
    K_ITERS = logical_k // BLOCK_K
    # Scale words are addressed by K-pair, so K must contain a whole number of
    # them (K % 256 above already guarantees it).

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // (64 * b_pack)

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

    @flyc.kernel(known_block_size=[512, 1, 1])
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
        live_rows = c_m
        if const_expr(dynamic_rows):
            live_rows = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, valid_rows[0]))
        m_blocks = ceildiv(live_rows, BLOCK_M)
        active = fx.Int32(1)
        if const_expr(dynamic_rows):
            active = fx.block_idx.x < m_blocks * n_blocks
        if active:
            if const_expr(xcd_swizzle > 0):
                block_m, block_n = _xcd_swizzle_any(m_blocks, n_blocks, xcd_swizzle)
                simple_m, simple_n = split_row_major_2d(fx.block_idx.x, n_blocks)
                use_simple = m_blocks * n_blocks < 1024
                if const_expr(not grouped):
                    use_simple = use_simple | (m_blocks * n_blocks % 8 != 0)
                block_m = use_simple.select(simple_m, block_m)
                block_n = use_simple.select(simple_n, block_n)
            else:
                block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)
            b_row = block_n * BLOCK_N
            if const_expr(grouped):
                expert = rocdl.readfirstlane(
                    fx.Int32.ir_type, expert_ids[block_m // (expert_block_m // BLOCK_M)]
                )
                b_row += fx.Int32(expert) * c_n

            A0_gl_offset = (block_m * BLOCK_M) * K
            A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
            B_K_STEP = ((2 * 1024) if b_preshuffled else BLOCK_K) // b_pack
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
                    (wave_id + step * 8) * (b_k // 2 * 16)
                    + lane_id % 16 * 16
                    + lane_id // 16 * 256
                    for step in range_constexpr(N_LDS_STEPS_B)
                ]
            else:
                gl_off_b = compute_global_swizzle(
                    lane_id, wave_id, b_k, N_LDS_STEPS_B, preshuffled=b_preshuffled
                )

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

            mfma = MxMfma(N_TILES_A, N_TILES_B, b_dtype)

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            a1_g2s = G2SLoader(a_div, gl_off_a1, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            b_s2r = (PackedFp4S2R if b_dtype == "fp4" else S2RLoader)(wave_n, N_TILES_B)
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

            a_sc = ScalePreshuffledS2R(A_scale, c_m, K, N_TILES_A)
            b_scale_rows = (
                fx.size(B_scale.shape).unpack() // (K // 32) if grouped else c_n
            )
            b_sc = ScalePreshuffledS2R(B_scale, b_scale_rows, K, N_TILES_B)
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

            def scale_prefetch(sc, k0):
                if const_expr(k0 >= K_ITERS or k0 // 2 in sc):
                    return sc
                words = {
                    w: (a_sc if w[0] == "a" else b_sc).read(
                        (a_base16 if w[0] == "a" else b_base16)[int(w[1])], k0
                    )
                    for w in ("a0", "a1", "b0", "b1")
                }
                return {**sc, k0 // 2: words}

            def scale_read(sc, k, which):
                """The prefetched E8M0 operands for K-step ``k``."""
                return sc[k // 2][which]

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            sc_pf = scale_prefetch({}, 0)

            b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
            a1_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

            if wave_m == 1:
                rocdl.s_barrier()

            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

            b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
            a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            for k in range_constexpr(K_ITERS - 2):
                if const_expr(k % 2 == 1):
                    sc_pf = scale_prefetch(sc_pf, k + 1)
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
                a0_frag = a_s2r.load(a_cur0)
                a1_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                rocdl.s_barrier()

                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
                rocdl.s_barrier()

                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                rocdl.s_barrier()

                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)

                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
                # Complete the A prefetches before rotating grouped/gathered LDS
                # buffers, leaving only the final B half (two loads) outstanding.
                # The dense vmcnt(6) allowance races on expert/padding boundaries.
                wait_barrier(
                    N_LDS_STEPS_B if grouped else 2 * N_LDS_STEPS_A + N_LDS_STEPS_B
                )

                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)

                # Swap cur and next
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            # Step k = K_ITERS - 2
            # Odd logical K (e.g. 384 with physical stride 512) needs the final
            # scale pair even though the main loop never reaches its odd prefetch.
            sc_pf = scale_prefetch(sc_pf, K_ITERS - 1)
            k = K_ITERS - 2
            sa0 = scale_read(sc_pf, k, "a0")
            sb0 = scale_read(sc_pf, k, "b0")
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

            sb1 = scale_read(sc_pf, k, "b1")
            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

            sa1 = scale_read(sc_pf, k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            # Main loop prefetches a_next1 one step behind; issue the final
            # K_ITERS - 1 tile here, otherwise c10 / c11 read stale A1 data.
            a1_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
            if const_expr(grouped and K_ITERS == 2):
                # With no main-loop iteration, the final B prefetch has not
                # passed its VMEM fence before the staggered S2R read.
                rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)

            b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
            rocdl.s_barrier()

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)
            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            # Step k = K_ITERS - 1
            k = K_ITERS - 1
            sa0 = scale_read(sc_pf, k, "a0")
            sb0 = scale_read(sc_pf, k, "b0")
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

            sb1 = scale_read(sc_pf, k, "b1")
            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

            sa1 = scale_read(sc_pf, k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(
                a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2, set_prio=False
            )
            c11_frag = mfma.call(
                a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2, set_prio=False
            )
            rocdl.s_setprio(0)
            rocdl.s_barrier()

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
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            A,
            A,
            A,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    @flyc.jit
    def launch_grouped_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        expert_ids: fx.Tensor,
        row_map: fx.Tensor,
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
            A,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(
            grid=(ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N), 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

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
            grid=(ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N), 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    if dynamic_rows:
        return launch_dynamic_gemm
    return launch_grouped_gemm if grouped else launch_gemm
