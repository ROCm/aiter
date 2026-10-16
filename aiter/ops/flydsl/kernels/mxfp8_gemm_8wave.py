# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""8-wave MXFP8 matmul (per-1x32 E8M0 scales) for AMD CDNA4 (gfx950 only).

Ported from FlyDSL ``kernels/gemm/mxfp8_gemm_8wave.py`` (incl. ROCm/FlyDSL#1275,
row-major scales). The K pipeline is the a8w8 ptpc 8-wave one
(``gemm_a8w8_8wave``), whose loaders it shares; the MFMA carries the E8M0
scales, so the epilogue only converts. ``MxMfma`` issues one scaled atom per
16x16 tile with ``scale_a=`` / ``scale_b=``, the form the pinned FlyDSL takes.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    S2RLoader,
    _xcd_swizzle_any,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d

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
            address_space=rocdl.TargetAddressSpace.BufferDesc,
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
            base = rocdl.readfirstlane(
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


class ScaleRawLds:
    """Row-major ``[rows, K // 32]`` E8M0 (no host shuffle), staged through LDS."""

    CHUNK = 4
    BYTES = 2 * 256 * 16

    def __init__(
        self,
        A_scale,
        B_scale,
        K,
        block_m,
        block_n,
        wave_id,
        a_row,
        b_row,
        lds,
        fp8_ir_t,
    ):
        """``a_row`` / ``b_row``: block-local first row of this wave's tiles; ``lds``: two chunk slots."""
        self.row_bytes = K // 32
        lane = fx.thread_idx.x % 64
        self.wave_id = wave_id
        self.lds = lds
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
        self.LdsPtr_t = fx.PointerType.get(fp8_ir_t, 2, 512)
        self.srcs = []
        for arg, block in ((A_scale, block_m), (B_scale, block_n)):
            src = fx.logical_divide(
                make_fp8_buffer_tensor(arg, fp8_ir_t), fx.make_layout(1, 1)
            )
            offs = []
            for step in range_constexpr(2):
                word = step * 512 + wave_id * 64 + lane  # 4 words per row
                offs.append((block * 256 + word // 4) * self.row_bytes + (word % 4) * 4)
            self.srcs.append((src, offs))
        # One per-lane base per operand; every read adds only a constant, which
        # folds into the ds_read offset as long as the buffer sits below 64 KB.
        lane_off = (lane % 16) * 16 + lane // 16
        self.bases = [
            fx.add_offset(lds.ptr, fx.make_int_tuple(op * 4096 + row * 16 + lane_off))
            for op, row in ((0, a_row), (1, b_row))
        ]

    def dma(self, chunk):
        """Stage chunk ``chunk``; the last chunk may run past K (those bytes are never read)."""
        base_i32 = fx.Int32(fx.ptrtoint(self.lds.ptr))
        for op in range_constexpr(2):
            src, offs = self.srcs[op]
            for step in range_constexpr(2):
                dst_off = (
                    chunk % 2 * self.BYTES
                    + op * 4096
                    + step * 2048
                    + self.wave_id * 256
                )
                dst = fx.make_view(
                    fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(dst_off)),
                    fx.make_layout(1, 1),
                )
                fx.copy(
                    self.atom,
                    fx.slice(src, (None, fx.Int32(offs[step]))),
                    dst,
                    soffset=fx.Int32(chunk * 16),
                )

    def read(self, op, half_row, n_tiles, k):
        """MFMA scale operands of ``n_tiles`` tiles, ``half_row`` rows past this wave's first row."""
        words = []
        for t in range_constexpr(n_tiles):
            off = (
                k // self.CHUNK % 2 * self.BYTES
                + (half_row + t * 16) * 16
                + (k % self.CHUNK) * 4
            )
            v = fx.make_view(
                fx.recast_iter(
                    fx.Uint8, fx.add_offset(self.bases[op], fx.make_int_tuple(off))
                ),
                fx.make_layout(1, 1),
            )
            words.append(fx.Int32(v.load()[0]))
        return words


class MxMfma:
    """n_tiles_a x n_tiles_b scaled 16x16x128 MFMAs, one E8M0 word per tile."""

    def __init__(self, n_tiles_a, n_tiles_b, paired_tiles=True):
        # paired_tiles: shuffle_scale_w4 words, byte = k_pack * 2 + tile parity;
        # otherwise ScaleRawLds words, byte 0.
        self.paired_tiles = paired_tiles
        self.atoms = {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, fx.Float8E4M3FN, opsel_a=osa, opsel_b=osb
                )
            )
            for osa in range(4)
            for osb in range(4)
        }
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _frag(self, value, n, dtype):
        frag = fx.make_rmem_tensor(n, dtype)
        frag.store(Vec(value))
        return frag

    def _opsel(self, k_pack, tile):
        return k_pack * 2 + tile % 2 if self.paired_tiles else 0

    def call(self, a, b, c, sa, sb, *, k_pack, set_prio=True):
        assert len(a) == len(sa) == self.n_tiles_a
        assert len(b) == len(sb) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        a_frags = [self._frag(v, 8, fx.Int32) for v in a]
        b_frags = [self._frag(v, 8, fx.Int32) for v in b]
        c_frags = [self._frag(v, 4, fx.Float32) for v in c]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(
                    self.atoms[(self._opsel(k_pack, i), self._opsel(k_pack, j))],
                    cf,
                    a_frags[i],
                    b_frags[j],
                    cf,
                    scale_a=sa[i],
                    scale_b=sb[j],
                )
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [cf.load().ir_value() for cf in c_frags]


class SwappedMxMfma:
    """``MxMfma`` with A/B -- and their scales and opsels -- exchanged.

    ``B^T A^T = (A B)^T``, so every accumulator comes out transposed and a lane
    ends up holding four consecutive *columns* instead of four rows. That is
    what lets the epilogue store 64 bits at a time; see ``StoreBf16Swapped``.

    The inner atom is built with the tile counts swapped, since its "A" is our
    B -- its own asserts and ``idx`` then line up without further adjustment.
    ``idx`` forwards swapped so callers keep indexing in (M-tile, N-tile) order.
    """

    def __init__(self, n_tiles_a, n_tiles_b, paired_tiles=True):
        self._inner = MxMfma(n_tiles_b, n_tiles_a, paired_tiles=paired_tiles)
        self.zero_value = self._inner.zero_value
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return self._inner.idx(j, i)

    def call(self, a, b, c, sa, sb, *, k_pack, set_prio=True):
        return self._inner.call(b, a, c, sb, sa, k_pack=k_pack, set_prio=set_prio)


def _raw(v):
    """Unwrap a DSL value to its raw ``ir.Value``, for ops that take one."""
    return v.ir_value() if hasattr(v, "ir_value") else v


def _permlane16_swap(x, y):
    """gfx950 ``v_permlane16_swap_b32``: exchange 16-lane rows between two VGPRs.

    Verified on hardware rather than assumed::

        vdst_new = [X.r0, Y.r0, X.r2, Y.r2]
        vsrc_new = [X.r1, Y.r1, X.r3, Y.r3]

    Position *within* a row is preserved, so a lane keeps its ``lane % 16`` and
    therefore its C row.
    """
    st = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = fx.rocdl.permlane16_swap(st, _raw(x), _raw(y), False, True)
    i32 = ir.IntegerType.get_signless(32)
    return (
        fx.Int32(
            _llvm_d.ExtractValueOp(i32, res, ir.DenseI64ArrayAttr.get([0])).result
        ),
        fx.Int32(
            _llvm_d.ExtractValueOp(i32, res, ir.DenseI64ArrayAttr.get([1])).result
        ),
    )


def _ds_bpermute(value, src_lane):
    """Pull ``value`` from ``src_lane``. Byte-addressed, hence the <<2."""
    i32 = ir.IntegerType.get_signless(32)
    return fx.Int32(
        fx.rocdl.ds_bpermute(i32, _raw(fx.Int32(src_lane) * fx.Int32(4)), _raw(value))
    )


class StoreBf16Swapped:
    """C store for an A/B-swapped MFMA: 64 bits per lane instead of 16.

    After the swap lane ``l`` holds ``D[l%16][4*(l/16)+k]`` -- four consecutive
    columns, i.e. eight contiguous bytes of a row-major C. The unswapped layout
    gives a lane four consecutive *rows* of one column, which can only be
    written one bf16 at a time.

    The accumulators are already scaled by the MFMA, so the epilogue is a plain
    fp32 -> bf16 convert.
    """

    def __init__(self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        gC = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=c_rows * c_cols * 2
        )
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        self.reg_bf16_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)

    def store(self, c_frag, base_row, base_col):
        lane = self.lane_id
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + (lane // 16) * 4
                # A lane's four columns are consecutive, so one predicate covers
                # them; out-of-range lands on the descriptor's OOB index.
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                vals = [vec_f32[k].to(fx.BFloat16) for k in range_constexpr(4)]
                idx = (col + 3 < self.c_cols).select(row * self.c_cols + col, oob)
                fx.memref_store_vec(
                    Vec.from_elements(vals, fx.BFloat16), self.reg_bf16_4
                )
                fx.copy(
                    self.out_atom_4,
                    self.reg_bf16_4,
                    fx.slice(self.c_div, (None, fx.Int32(idx))),
                )


class StoreBf16Permlane(StoreBf16Swapped):
    """One cross-lane step on top, giving 16B per lane and 64B rows.

    After the swap a lane owns four consecutive columns of one N-tile, i.e.
    eight bytes, and the two N-tiles it holds are 16 columns apart -- so a row
    still only gets 32 bytes per instruction. Two ``permlane16_swap`` fix that
    exactly, with no ``ds_bpermute`` and no LDS::

        (A, B) = permlane16_swap(tile0.d0, tile1.d0)
        (C, D) = permlane16_swap(tile0.d1, tile1.d1)
        lane group g stores (A, C, B, D)

    which lands g = 0,1,2,3 on columns 0-7, 16-23, 8-15, 24-31. The four groups
    together cover columns 0..31 contiguously: 64 bytes per row, from 16 bytes
    per lane. The column permutation is absorbed into the address, so it is
    free.
    """

    _lane_transpose = False

    def store(self, c_frag, base_row, base_col):
        assert self.n_tiles_b == 2, (
            "the permlane mapping pairs exactly two N-tiles (BLOCK_N == 256); "
            f"got n_tiles_b={self.n_tiles_b}"
        )
        lane = self.lane_id
        grp = lane // 16
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            dwords = []
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                packed = Vec.from_elements(
                    [vec_f32[k].to(fx.BFloat16) for k in range_constexpr(4)],
                    fx.BFloat16,
                ).bitcast(fx.Int32)
                dwords.append((packed[0], packed[1]))
            a, b = _permlane16_swap(dwords[0][0], dwords[1][0])
            c, d = _permlane16_swap(dwords[0][1], dwords[1][1])
            if const_expr(self._lane_transpose):
                # The permlane stage leaves a row's four 8-column chunks in
                # lanes 16 apart, so a 16-lane group touches 16 rows at 16 bytes
                # each. Transposing the lane index -- lane l' = 4r'+q' pulls
                # from the lane holding (row r', chunk q') -- puts adjacent
                # lanes on one row, so lanes 0-3 write 64 contiguous bytes and a
                # group covers 4 rows instead of 16.
                #
                # chunk q' lives at column q'*8, and the permlane stage put
                # column (g%2)*16 + (g//2)*8 in group g, so g is the swap of
                # q's two bits: 0,1,2,3 -> 0,2,1,3.
                q = lane % 4
                src_lane = ((q % 2) * 2 + q // 2) * 16 + lane // 4
                a, c, b, d = (_ds_bpermute(v, src_lane) for v in (a, c, b, d))
                row = base_row + ti * 16 + lane // 4
                col = base_col + q * 8
            else:
                col = base_col + (grp % 2) * 16 + (grp // 2) * 8
            out8 = Vec.from_elements([a, c, b, d], fx.Int32).bitcast(fx.BFloat16)
            oob = fx.Int32(self.c_rows * self.c_cols)
            idx = (col + 7 < self.c_cols).select(row * self.c_cols + col, oob)
            fx.memref_store_vec(out8, self.reg_bf16_8)
            fx.copy(
                self.out_atom_8,
                self.reg_bf16_8,
                fx.slice(self.c_div, (None, fx.Int32(idx))),
            )


class StoreBf16LaneTranspose(StoreBf16Permlane):
    """``StoreBf16Permlane`` plus the ds_bpermute lane transpose."""

    _lane_transpose = True


def compile_mxfp8_gemm_8w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    b_preshuffled: bool = True,
    scale_preshuffled: bool = False,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
):
    """Build the MXFP8 launcher.

    ``scale_preshuffled=True`` takes ``shuffle_scale_w4``-permuted A/B scales;
    ``False`` takes them row-major ``[rows, K // 32]`` as the quantizer emits them.
    """
    # A 256x256 tile already spends 128 KB of the 160 KB LDS; every other
    # BLOCK_M/BLOCK_N either overflows it or breaks the 2-threads-per-scale-row
    # staging split. Kept as an assert rather than dead generality.
    assert (
        BLOCK_M == 256 and BLOCK_N == 256
    ), "MXFP8 8-wave is specialized to a 256x256 block tile"
    assert K % BLOCK_K == 0, f"K must be a multiple of {BLOCK_K}, got {K}"

    K_ITERS = K // BLOCK_K
    # Only the preshuffled reader addresses scale words by K-pair (k1 = k // 2),
    # so only it needs a whole number of pairs. ScaleRawLds stages CHUNK steps
    # at a time and its dma() already tolerates a partial last chunk.
    if scale_preshuffled:
        assert K % 256 == 0, (
            f"preshuffled scales are addressed by K-pair, so K must be a "
            f"multiple of 256, got {K}"
        )
        assert K_ITERS % 2 == 0

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    A_GRP_ROWS = N_TILES_A * 16  # rows one wave_m owns inside one LDS half
    B_GRP_ROWS = N_TILES_B * 16

    lds_bytes = 4 * (a_lds_size + b_lds_size) + (
        0 if scale_preshuffled else 2 * ScaleRawLds.BYTES
    )
    assert (
        lds_bytes <= LDS_LIMIT_BYTES
    ), f"LDS {lds_bytes} B exceeds {LDS_LIMIT_BYTES} B"

    _layout_tag = "bpreshuffle" if b_preshuffled else "rowmajor"
    _scale_tag = "" if scale_preshuffled else "_row"
    _kname = (
        f"flydsl_{_layout_tag}_mxfp8_8w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_B16_"
        f"{waves_per_eu}x{xcd_swizzle}{_scale_tag}_k{K}"
    )

    @fx.struct
    class SharedStorage:
        if not scale_preshuffled:
            S_lds: fx.Array[fx.Float8E4M3FN, 2 * ScaleRawLds.BYTES, 16]
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(name=_kname, known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
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
        if const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                ceildiv(c_m, BLOCK_M), n_blocks, wgm=xcd_swizzle
            )
        else:
            block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
        )
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled
        )

        mfma = SwappedMxMfma(N_TILES_A, N_TILES_B, paired_tiles=scale_preshuffled)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        store_c = StoreBf16LaneTranspose(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        if const_expr(scale_preshuffled):
            a_sc = ScalePreshuffledS2R(A_scale, c_m, K, N_TILES_A)
            b_sc = ScalePreshuffledS2R(B_scale, c_n, K, N_TILES_B)
        else:
            raw_sc = ScaleRawLds(
                A_scale,
                B_scale,
                K,
                block_m,
                block_n,
                wave_id,
                wave_m * A_GRP_ROWS,
                wave_n * B_GRP_ROWS,
                lds.S_lds,
                F8_IR_t,
            )
            n_chunks = (K_ITERS + ScaleRawLds.CHUNK - 1) // ScaleRawLds.CHUNK
        # 16-row tile index of each wave's first row, per LDS half.
        a_base16 = [
            (block_m * BLOCK_M + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
            for h in range_constexpr(2)
        ]
        b_base16 = [
            (block_n * BLOCK_N + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16
            for h in range_constexpr(2)
        ]
        SCALE_LOADS_PER_PAIR = 2 * (N_TILES_A // 2) + 2 * (N_TILES_B // 2)
        assert SCALE_LOADS_PER_PAIR == 6, (
            f"scale prefetch issues {SCALE_LOADS_PER_PAIR} VMEM loads per K-pair, not the 6 the "
            "wait_barrier counts below were validated against; re-check them against the ISA"
        )

        def scale_prefetch(sc, k0):
            if const_expr(not scale_preshuffled or k0 >= K_ITERS or k0 // 2 in sc):
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
            if const_expr(scale_preshuffled):
                words = sc[k // 2][which]
            else:
                op, h = int(which[0] == "b"), int(which[1])
                half_row = h * (LDS_BLOCK_N if op else LDS_BLOCK_M)
                words = raw_sc.read(op, half_row, N_TILES_B if op else N_TILES_A, k)
            return words

        def scale_chunk_dma(k):
            # Issued after step k's first barrier: chunk k // CHUNK - 1 (same slot)
            # is then finished by both wave_m groups, and the step's closing
            # wait_barrier drains this DMA long before chunk k // CHUNK + 1 is read.
            chunk = k // ScaleRawLds.CHUNK + 1
            if const_expr(
                not scale_preshuffled
                and k % ScaleRawLds.CHUNK == 0
                and k > 0
                and chunk < n_chunks
            ):
                raw_sc.dma(chunk)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        sc_pf = scale_prefetch({}, 0)
        if const_expr(not scale_preshuffled):
            for chunk in range_constexpr(min(2, n_chunks)):
                raw_sc.dma(chunk)

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

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
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()
            scale_chunk_dma(k)

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

            sb1 = scale_read(sc_pf, k, "b1")
            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

            sa1 = scale_read(sc_pf, k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            # The wave_m == 1 group runs one barrier behind, so this barrier of
            # it pairs with the other group's end-of-step wait, after which that
            # group reads step k + 1's b_cur0 / a_cur0. Those were loaded in step
            # k - 1, and the end-of-step 2 * A + B leaves them in flight: drain
            # everything but this step's three loads and step k - 1's b_cur1.
            wait_barrier(2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        sa0 = scale_read(sc_pf, k, "a0")
        sb0 = scale_read(sc_pf, k, "b0")
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        # The peeled steps issue no loads before reading step K_ITERS - 1's
        # tiles, which the lagging group loaded in step K_ITERS - 3: drain them
        # here, the barrier the other group's next reads pair with.
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

        sb1 = scale_read(sc_pf, k, "b1")
        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

        sa1 = scale_read(sc_pf, k, "a1")
        a1_frag = a_s2r.load(a_cur1)
        # Main loop prefetches a_next1 one step behind; issue the final
        # K_ITERS - 1 tile here, otherwise c10 / c11 read stale A1 data.
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
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

        # Accumulators are already scaled by the MFMA: convert and store.
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

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
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
