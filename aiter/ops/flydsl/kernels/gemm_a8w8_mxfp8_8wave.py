# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8-wave mxfp8 matmul for CDNA4 (gfx950): 1x32 ue8m0 scales, bf16 out.

``v_mfma_scale_f32_16x16x128_f8f6f4`` takes e8m0 scales as instruction
operands, so a 32-wide ue8m0 GEMM needs no dequantisation arithmetic at all --
no second accumulator, no running rescale, no division. One MFMA consumes a
whole K=128 step with its four 32-blocks already dequantised in hardware. That
is what separates this from the 128-wide blockscale kernels, whose promote and
rescale chain exists precisely because a 128-wide block cannot be expressed
this way.

The mainloop is ``gemm_a8w8_8wave``'s, with the scale operands threaded into
the four MFMA calls. Two things around it are not, and both are about
*addresses* rather than arithmetic:

**A/B are swapped.** ``fx.gemm(atom, c, b, a, c)`` computes ``B^T A^T =
(A B)^T``, which moves a lane's four accumulator values from four consecutive
rows to four consecutive columns. In a row-major C that is 8 contiguous bytes
where the unswapped layout can only ever emit ``buffer_store_short``. Two
cross-lane stages on top of that take a row to 64 contiguous bytes. Measured
38.30 -> 35.74us on the GEMM alone.

**The scales are K-block major and packed four to a dword.** A block group's
sixteen lanes want sixteen consecutive rows of one K block, so K-block major
makes that one coalesced access where the quantiser's own ``[M, K/32]`` row
major makes it sixteen -- same instruction count, 3.2x the cache accesses,
+50 to +67% on the whole GEMM. And a lane's four A tiles differ only by sixteen
rows and share the K block, so packing their four bytes into one dword lets
``opsel_b`` pick between them for free: -5 to -7.5%, with VALU flat. CK does
the same in ``preShuffleScaleBuffer_gfx950``.

Ported from mori's ``mori.ops.gemm_ar.kernels_fused``, minus the all-reduce
epilogue it exists to serve.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    _xcd_swizzle_any,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d

BLOCK_K = 128

#: ue8m0 block size along K, and along N for B. Fixed by the checkpoint format
#: and by the instruction.
MXFP8_BLOCK = 32

#: Rows one packed A-scale group spans: four M tiles of sixteen rows each.
A_SCALE_GROUP_M = 64


class _SwappedMfma:
    """``Mfma16x16x128`` with the A/B operands exchanged at every call.

    Thin on purpose: the accumulator registers, the atom and ``zero_value`` are
    the wrapped object's, only the operand order and the ``idx`` argument order
    change. ``idx(ti, tj)`` here forwards to ``idx(tj, ti)`` so the store can
    keep indexing in (M-tile, N-tile) order.
    """

    def __init__(self, inner):
        self._inner = inner
        self.zero_value = inner.zero_value

    def idx(self, i, j):
        return self._inner.idx(j, i)

    def call(self, a, b, c, *, set_prio=True, scale_a=None, scale_b=None):
        # The scales follow their operands: with the exchange the instruction's
        # A is our B, so its scale_a must be our B scale. The lane mapping is
        # unaffected -- row is still ``lane % 16`` and the 32-block still
        # ``lane // 16`` -- only which matrix that row indexes changes.
        return self._inner.call(
            b, a, c, set_prio=set_prio, scale_a=scale_b, scale_b=scale_a
        )


class _Mxfp8ScaleK:
    """The 32-wide ue8m0 scales, fed to the MFMA as operands.

    Lane mapping: lane ``16*s + r`` supplies the ue8m0 scale of 32-block ``s``
    of row ``r``, at op_sel 0. So for K step ``ks`` this lane wants block
    ``4*ks + lane//16`` of row ``lane % 16``.

    Scales arrive as int32 with the exponent byte in the low 8 bits rather than
    as packed uint8. The MFMA's scale operand is a 32-bit register read at
    op_sel 0, so the widening is free at the instruction, and it buys a plain
    dword load here instead of sub-dword addressing.
    """

    BLOCK = MXFP8_BLOCK

    def __init__(self, A_scale, B_scale, m, n, k, *, n_tiles_a, n_tiles_b):
        # Four bytes to a dword, so exactly four M tiles. BLOCK_M >= 256
        # already forces it at the call site.
        if n_tiles_a != 4:
            raise ValueError(
                f"packed A scales need exactly 4 M tiles (BLOCK_M=256), got "
                f"n_tiles_a={n_tiles_a}"
            )
        self.kb_count = k // self.BLOCK
        self.m = m
        self.n_groups = n // self.BLOCK
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.lane = fx.thread_idx.x % 64
        a_bytes = m * self.kb_count  # one byte per scale, [K/32, M/64, 16, 4]
        b_bytes = self.n_groups * self.kb_count * 4
        gSA = fx.rocdl.make_buffer_tensor(
            A_scale, max_size=False, num_records_bytes=a_bytes
        )
        gSB = fx.rocdl.make_buffer_tensor(
            B_scale, max_size=False, num_records_bytes=b_bytes
        )
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        self.reg_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)

    def _load1(self, div, index):
        fx.copy(self.atom_1, fx.slice(div, (None, fx.Int32(index))), self.reg_1)
        return Vec(fx.memref_load_vec(self.reg_1))[0]

    def _kb(self, ks):
        """This lane's 32-block for K step ``ks``: block ``4*ks + lane//16``."""
        return fx.Int32(ks * 4) + self.lane // 16

    def a_scales(self, base_row, ks):
        """Per M-tile A scale operand, from ``[K/32, M/64, 16, 4]`` bytes.

        The four M tiles of one lane sit in one dword, because they differ only
        by 16 rows and share the K block. One load feeds all four MFMAs and
        ``opsel_b`` picks the byte, so the four returned entries are
        deliberately the same register. Sixteen lanes of a block group still
        read sixteen consecutive dwords, so this keeps the coalescing and
        divides the A-scale load count by four. base_row is a multiple of 64
        and M a multiple of 4, so both shifts are exact.
        """
        kb = self._kb(ks)
        v = self._load1(
            self.sa_div,
            kb * fx.Int32(self.m // 4) + base_row // fx.Int32(4) + self.lane % 16,
        )
        return [v for _ in range_constexpr(self.n_tiles_a)]

    def b_scales(self, base_col, ks):
        """Per N-tile B scale operand. ``B_scale`` is K-block major, ``[K/32, N/32]``.

        A 16-column tile never straddles a 32-column group (``base_col`` is a
        multiple of 16), so the group index is constant across the tile's rows:
        all sixteen lanes of a block group read the same address and the load
        is a broadcast.
        """
        col = base_col + self.lane % 16
        kb = self._kb(ks)
        return [
            self._load1(
                self.sb_div,
                kb * fx.Int32(self.n_groups) + (col + tj * 16) // fx.Int32(self.BLOCK),
            )
            for tj in range_constexpr(self.n_tiles_b)
        ]

    def step(self, base_row, base_col, ks, lds_block_m, lds_block_n):
        """The A/B scale operands of K step ``ks`` for the four LDS halves.

        Returned in the order the mainloop consumes them: ``(a0, a1, b0, b1)``,
        pairing as c00=(a0,b0), c01=(a0,b1), c10=(a1,b0), c11=(a1,b1). A method
        rather than a closure: FlyDSL rewrites the AST of every ``def`` nested
        in a kernel, which turns captured instances into locals.
        """
        return (
            self.a_scales(base_row + 0 * lds_block_m, ks),
            self.a_scales(base_row + 1 * lds_block_m, ks),
            self.b_scales(base_col + 0 * lds_block_n, ks),
            self.b_scales(base_col + 1 * lds_block_n, ks),
        )


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


class _Mxfp8StoreC:
    """C store for an A/B-swapped MFMA, with no scaling left to do.

    The mainloop applied the ue8m0 scales at the instruction, so the epilogue
    is a plain fp32 -> bf16 convert -- unlike the ptpc ``StoreC`` this does not
    subclass, which carries two fp32 scale buffers and loads from them per
    tile.

    After the swap lane ``l`` holds ``D[l%16][4*(l/16)+k]``: 4 consecutive
    columns, i.e. 8 contiguous bytes in a row-major C.
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
                # BLOCK_N divides N in every shape this builds for, so the four
                # columns are either all in range or all out; one predicate.
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


class _PermlaneStoreC(_Mxfp8StoreC):
    """One cross-lane step on top, giving 16B per lane and 64B rows.

    After the swap a lane owns 4 consecutive columns of one N-tile, i.e. 8
    bytes, and the two N-tiles it holds are 16 columns apart -- so a row still
    only gets 32 bytes per instruction. Two ``permlane16_swap`` fix that
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
                # gcnasm's second stage, as one ds_bpermute per dword. The
                # permlane stage leaves a row's four 8-column chunks in lanes
                # 16 apart, so a 16-lane group touches 16 rows at 16 bytes
                # each. Transposing the lane index -- lane l' = 4r'+q' pulls
                # from the lane holding (row r', chunk q') -- puts adjacent
                # lanes on one row, so lanes 0-3 write 64 contiguous bytes and
                # a group covers 4 rows instead of 16.
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


class _LaneTransposeStoreC(_PermlaneStoreC):
    """``_PermlaneStoreC`` plus gcnasm's ds_bpermute lane transpose."""

    _lane_transpose = True


def compile_mxfp8_gemm_8w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    permlane: bool = True,
    lane_transpose: bool = True,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
):
    """Compile the 8-wave mxfp8 GEMM for one K and tile.

    ``A_scale`` is the packed K-block-major int32 buffer
    ``aiter.ops.shuffle.shuffle_mxfp8_a_scale`` emits; ``B_scale`` is the
    checkpoint's ``[N/32, K/32]`` ue8m0 bytes transposed to K-block major and
    widened to int32. ``B_T`` is ``shuffle_weight(w, (16, 16))``.
    """
    assert BLOCK_M == 256, (
        f"BLOCK_M={BLOCK_M}: the packed A scale puts a lane's four 16-row M "
        "tiles in one dword, which is exactly BLOCK_M=256"
    )
    # BLOCK_N's constraint belongs to the *store*, not the mainloop: the
    # mainloop builds N_TILES_B = BLOCK_N//128 accumulators and is happy with
    # any count, while the permlane store pairs exactly two N-tiles. So
    # permlane needs BLOCK_N == 256 -- an equality, not a floor. Written as
    # ">= 256" it would let BLOCK_N=512 through to an assert several frames
    # deeper, and the wide tile is only chosen once the grid is large enough,
    # so such a build would serve small batches correctly and die when one grew.
    if permlane:
        assert BLOCK_N == 256, (
            f"BLOCK_N={BLOCK_N}: permlane's store pairs exactly two N-tiles, "
            "so it needs BLOCK_N == 256 (pass permlane=False for other widths)"
        )
    else:
        assert BLOCK_N >= 128, f"BLOCK_N={BLOCK_N} is below 128"
    assert BLOCK_N % 128 == 0, f"BLOCK_N={BLOCK_N}"
    assert lane_transpose <= permlane, "lane_transpose builds on permlane"
    assert K % BLOCK_K == 0, f"K={K} must be a multiple of {BLOCK_K}"
    assert K // BLOCK_K >= 2, (
        f"K={K}: the mainloop prefetches a second K block and runs two tail "
        f"steps, so K/{BLOCK_K} must be at least 2"
    )

    K_ITERS = K // BLOCK_K
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

    _kname = (
        f"flydsl_mxfp8_8w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_B16_"
        f"{'P' if permlane else ''}{'T' if lane_transpose else ''}"
        f"{waves_per_eu}x{xcd_swizzle}_k{K}"
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
        B_K_STEP = 2 * 1024
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
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=True
        )

        # Tile counts swap with the operands so Mfma's own asserts and its
        # idx() line up; the store then addresses the accumulator as
        # idx(tj, ti). With the swap the instruction's B is our A, so the
        # per-tile opsel that selects a packed A byte is opsel_b on the raw
        # atom.
        mfma = _SwappedMfma(
            Mfma16x16x128(N_TILES_B, N_TILES_A, opsel_b_per_tile=True)
        )

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        if const_expr(lane_transpose):
            store_c = _LaneTransposeStoreC(
                C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(permlane):
            store_c = _PermlaneStoreC(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)
        else:
            store_c = _Mxfp8StoreC(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        msk = _Mxfp8ScaleK(
            A_scale,
            B_scale,
            c_m,
            c_n,
            K,
            n_tiles_a=N_TILES_A,
            n_tiles_b=N_TILES_B,
        )
        base_row_pre = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        base_col_pre = block_n * BLOCK_N + wave_n * (N_TILES_B * 16)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        # Opens a deliberate half-wave stagger: waves 4-7 take one extra
        # barrier, and since s_barrier is a counting rendezvous, waves 0-3 run
        # one phase ahead from here on -- which is what the double-buffered
        # mainloop wants. Closed before the epilogue.
        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            msa0, msa1, msb0, msb1 = msk.step(
                base_row_pre, base_col_pre, k, LDS_BLOCK_M, LDS_BLOCK_N
            )
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(
                a0_frag, b0_frag, c00_frag, scale_a=msa0, scale_b=msb0
            )

            b1_frag = b_s2r.load(b_cur1, preshuffled=True)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(
                a0_frag, b1_frag, c01_frag, scale_a=msa0, scale_b=msb1
            )

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(
                a1_frag, b0_frag, c10_frag, scale_a=msa1, scale_b=msb0
            )

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            # Letting 2 * A + B loads stay outstanding here crosses the barrier
            # with LDS writes still in flight that the next iteration reads.
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B - 1)

            c11_frag = mfma.call(
                a1_frag, b1_frag, c11_frag, scale_a=msa1, scale_b=msb1
            )

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        msa0, msa1, msb0, msb1 = msk.step(
            base_row_pre, base_col_pre, k, LDS_BLOCK_M, LDS_BLOCK_N
        )
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, scale_a=msa0, scale_b=msb0)

        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, scale_a=msa0, scale_b=msb1)

        a1_frag = a_s2r.load(a_cur1)
        # Main loop prefetches a_next1 one step behind; issue the final
        # K_ITERS - 1 tile here, otherwise c10 / c11 read stale A1 data.
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, scale_a=msa1, scale_b=msb0)

        b0_frag = b_s2r.load(b_next0, preshuffled=True)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, scale_a=msa1, scale_b=msb1)
        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        msa0, msa1, msb0, msb1 = msk.step(
            base_row_pre, base_col_pre, k, LDS_BLOCK_M, LDS_BLOCK_N
        )
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, scale_a=msa0, scale_b=msb0)

        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, scale_a=msa0, scale_b=msb1)

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(
            a1_frag, b0_frag, c10_frag, set_prio=False, scale_a=msa1, scale_b=msb0
        )
        c11_frag = mfma.call(
            a1_frag, b1_frag, c11_frag, set_prio=False, scale_a=msa1, scale_b=msb1
        )
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Close the half-wave barrier pairing the prologue opened, so the two
        # halves are level again before the epilogue. gcnasm closes it at the
        # same point (opus_gemm_a2a_lsa, quad-subtile template). Nothing here
        # observes the imbalance today, but leaving the counts unbalanced makes
        # the next thing added after the epilogue silently wrong.
        if wave_m == 0:
            rocdl.s_barrier()

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
