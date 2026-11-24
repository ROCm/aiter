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
import flydsl.expr.arith as farith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.dpp_utils import dpp_xor_f32
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


def barrier_if_zero(sgpr_value):
    """``s_barrier`` only in waves whose (wave-uniform, SGPR) value is 0.

    Inline asm rather than ``if``: a branch splits the block while every
    accumulator is live, and the scalar ``scf.if`` alone raised the WKV
    (K=6144) kernel's scratch spills 8 -> 20 and its time by ~2% (MI355X). The
    branch skips the one 4-byte ``s_barrier``.
    """
    _llvm.inline_asm(
        res=None,
        operands_=[sgpr_value],
        asm_string="s_cmp_lg_u32 $0, 0\ns_cbranch_scc1 1\ns_barrier",
        constraints="s,~{scc}",
        has_side_effects=True,
    )


class ScalePreshuffledS2R:
    """Coalesced reader for ``shuffle_scale_w4``-packed E8M0 -- no LDS staging."""

    def __init__(self, scale_arg, rows, K, n_tiles):
        assert n_tiles % 2 == 0, "shuffle_scale_w4 pairs tiles two at a time"
        self.n_pairs = n_tiles // 2
        self.k1_stride = K // 256  # i32 groups of 64 per 32-row super-row
        self.lane = fx.thread_idx.x % 64
        # The layout covers the rows padded to 256 (what the launcher validates):
        # a row in the last, partial 32-row group sits past rows * K / 32 bytes.
        rows = ceildiv(rows, 256) * 256
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
    """Unshuffled E8M0 scales (no host shuffle), staged through LDS a chunk of
    ``CHUNK`` K-steps at a time.

    Per operand: a 1x32 scale is row-major ``[rows, K // 32]`` (16 bytes of a row
    per chunk; a 32x32 B ``[N // 32, K // 32]`` repeats a row over its 32 N-rows);
    a 1x128 A is the blockscale column-major ``[K // 128, M]`` (a chunk is CHUNK
    columns of the block's 256 rows); a 128x128 B is ``[N // 128, K // 128]``, one
    row per 128-row LDS half (a chunk is a dword of each). The 1x128 A goes a
    byte per lane into dword slots. Every operand issues two copies per chunk,
    so the pipeline's vmcnt waits stay as validated.
    """

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
        b_rows_per=1,
        a_col_major=False,
        b_block128=False,
        c_m=None,
        k_byte_off=None,
    ):
        """``a_row`` / ``b_row``: block-local first row of this wave's tiles; ``lds``: two chunk slots.

        ``b_rows_per`` N-rows share one B scale row: 1 for ``[N, K // 32]``, 32
        for a 32x32 ``[N // 32, K // 32]`` -- the staged chunk is the 1x32 one.
        ``a_col_major`` / ``b_block128`` select the 1x128 / 128x128 forms (``c_m``
        is M, the column stride of the former). ``k_byte_off``: a split-K
        segment's first scale byte of a row (row-major 1x32 / 32x32 forms only).
        """
        assert k_byte_off is None or not (a_col_major or b_block128)
        self.k_byte_off = k_byte_off
        self.row_bytes = K // 32
        self.k128 = K // 128
        self.a_col = a_col_major
        self.b_blk = b_block128
        self.c_m = c_m
        lane = fx.thread_idx.x % 64
        self.wave_id = wave_id
        self.lds = lds
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
        self.LdsPtr_t = fx.PointerType.get(fp8_ir_t, 2, 512)
        self.srcs = []
        for op, (arg, block, rows_per) in enumerate(
            ((A_scale, block_m, 1), (B_scale, block_n, b_rows_per))
        ):
            if const_expr(op == 0 and a_col_major):
                # Copied a byte per lane into a dword slot of its own (as bmm's
                # byte-copied panels): the column offsets c * M are not dword
                # aligned, and a dword copy ending past the tensor would read
                # zero whole, valid bytes too. The buffer bounds are exact.
                src = fx.rocdl.get_buffer_rsrc(
                    fx.get_iter(fx.rocdl.make_buffer_tensor(arg, max_size=False))
                )
            else:
                src = fx.logical_divide(
                    make_fp8_buffer_tensor(arg, fp8_ir_t), fx.make_layout(1, 1)
                )
            offs = []
            for step in range_constexpr(2):
                word = step * 512 + wave_id * 64 + lane
                if const_expr(op == 0 and a_col_major):
                    # A chunk is CHUNK columns of the block's 256 rows: 1024 bytes,
                    # two copies of 512 lanes, slot w = (column, row).
                    offs.append(word // 256 * c_m + block * 256 + word % 256)
                elif const_expr(op == 1 and b_block128):
                    # Slot w holds half w % 2's row; only slots 0 and 1 are read.
                    offs.append((block * 2 + word % 2) * self.k128)
                else:
                    row = block * 256 + word // 4  # 4 words per row
                    if const_expr(rows_per > 1):
                        row = row // rows_per
                    offs.append(row * self.row_bytes + (word % 4) * 4)
            self.srcs.append((src, offs))
        # One per-lane base per operand; every read adds only a constant, which
        # folds into the ds_read offset as long as the buffer sits below 64 KB.
        lane_off = (lane % 16) * 16 + lane // 16
        self.bases = [
            fx.add_offset(
                lds.ptr,
                fx.make_int_tuple(
                    (a_row + lane % 16) * 4 if a_col_major else a_row * 16 + lane_off
                ),
            ),
            fx.add_offset(
                lds.ptr,
                fx.make_int_tuple(4096 if b_block128 else 4096 + b_row * 16 + lane_off),
            ),
        ]

    def _soffset(self, op, chunk):
        if const_expr(op == 0 and self.a_col):
            return fx.Int32(chunk * self.CHUNK) * self.c_m
        if const_expr(op == 1 and self.b_blk):
            # The dword never runs past the row's end (K // 128 >= 4).
            return fx.Int32(min(chunk * self.CHUNK, self.k128 - 4))
        if const_expr(self.k_byte_off is not None):
            return fx.Int32(chunk * 16) + self.k_byte_off
        return fx.Int32(chunk * 16)

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
                lds_dst = fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(dst_off))
                if const_expr(op == 0 and self.a_col):
                    rocdl.buffer_load_to_lds(
                        src,
                        fx.to_llvm_ptr(lds_dst),
                        fx.Int32(offs[step]).ir_value(),
                        size_bytes=1,
                        soffset=self._soffset(op, chunk).ir_value(),
                    )
                    continue
                dst = fx.make_view(lds_dst, fx.make_layout(1, 1))
                fx.copy(
                    self.atom,
                    fx.slice(src, (None, fx.Int32(offs[step]))),
                    dst,
                    soffset=self._soffset(op, chunk),
                )

    def _byte(self, op, off):
        v = fx.make_view(
            fx.recast_iter(
                fx.Uint8, fx.add_offset(self.bases[op], fx.make_int_tuple(off))
            ),
            fx.make_layout(1, 1),
        )
        return fx.Int32(v.load()[0])

    def read(self, op, half_row, n_tiles, k):
        """MFMA scale operands of ``n_tiles`` tiles, ``half_row`` rows past this wave's first row."""
        slot = k // self.CHUNK % 2 * self.BYTES
        if const_expr(op == 1 and self.b_blk):
            # One block per 128-row half: every tile and lane shares its byte.
            chunk = k // self.CHUNK
            start = min(chunk * self.CHUNK, self.k128 - 4)
            v = self._byte(op, slot + half_row // 128 * 4 + (k - start))
            return [v] * n_tiles
        words = []
        for t in range_constexpr(n_tiles):
            if const_expr(op == 0 and self.a_col):
                words.append(
                    self._byte(
                        op, slot + ((k % self.CHUNK) * 256 + half_row + t * 16) * 4
                    )
                )
            else:
                off = slot + (half_row + t * 16) * 16 + (k % self.CHUNK) * 4
                words.append(self._byte(op, off))
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
        self.zero_value = fx.full(4, 0.0, fx.Float32)
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


class StoreBf16:
    """Converts the already-scaled fp32 accumulators to bf16 and stores them."""

    def __init__(self, C, c_rows, c_cols, ldc, c_idx_fn, n_tiles_a, n_tiles_b):
        # ldc: C's row stride in elements (>= c_cols), so C may be a column
        # slice of a wider row-major matrix. Rows past c_rows land at or beyond
        # c_rows * ldc, which the descriptor drops.
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.ldc = ldc
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        gC = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=c_rows * ldc * 2
        )
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def store(self, c_frag, base_row, base_col):
        # MFMA 16x16 C layout: lane l holds rows (l // 16) * 4 + i, column l % 16.
        # The lane's element index is computed once; a row's offset
        # (ti * 16 + i) * ldc is wave-uniform (SALU), so each element costs one
        # add. A column past c_cols moves the lane's base to c_rows * ldc, which
        # every row offset keeps out of the descriptor's range.
        oob = fx.Int32(self.c_rows * self.ldc)
        lane_col = base_col + self.lane_id % 16
        lane_base = (base_row + (self.lane_id // 16) * 4) * self.ldc + lane_col
        for tj in range_constexpr(self.n_tiles_b):
            col_valid = (lane_col + tj * 16) < self.c_cols
            base_tj = fx.Int32(col_valid.select(lane_base + tj * 16, oob))
            for ti in range_constexpr(self.n_tiles_a):
                # Four rows converted together: two packed cvts, not four.
                acc = Vec(c_frag[self.c_idx_fn(ti, tj)]).to(fx.BFloat16)
                for i in range_constexpr(4):
                    fx.memref_store_vec(
                        fx.full(1, acc[i], fx.BFloat16), self.reg_bf16_1
                    )
                    row_off = fx.Int32((ti * 16 + i) * self.ldc)
                    fx.copy(
                        self.out_atom_1,
                        self.reg_bf16_1,
                        fx.slice(self.c_div, (None, base_tj + row_off)),
                    )


class StoreSwigluMx8W4:
    """gate_up epilogue: SwiGLU-OAI and the MXFP8 quant of the down projection's
    input, with its e8m0 scales in the shuffle_scale_w4 layout.

    The launch points the B tile's two 128-row halves at gate rows [128 t, +128) and
    the matching up rows [I + 128 t, +128) of the [gate | up] weight, so a lane holds
    the gate and the up of the same output elements. Each element gets the bytes of
    the bf16 GEMM output followed by SwiGLU-OAI in this order: g = min(gate, L),
    u = clamp(up, -L, L), (beta + u) * (g * (1 / (exp2(alpha * -g * log2 e) + 1))),
    rounded to bf16; then the MXFP8 quant of each 32-column group: code =
    ceil_pow2(max(amax, 1e-10) * (1 / 448)), q = clamp(x * 2^(127 - code), +-448) to
    e4m3. Output: one byte buffer, e4m3 [c_m, I] row-major, then the scales of
    [pad256(c_m), I / 32].
    Same bytes, fewer instructions: 1 / y as v_rcp plus one Newton step for y < 2^126
    (the correctly rounded quotient there) and 0 above (gate <= -51.3: the element's
    e4m3 byte is the same signed zero and cannot set its group's amax); raw
    v_exp_f32 (llvm.exp2 for inputs >= -126, which the gate clamp keeps); the up clamp
    as one v_med3_f32 (LLVM would narrow the clamp of a bf16-exact value to bf16);
    the group amax as an integer max of |x| bit patterns over a 16-lane DPP row.
    """

    def __init__(
        self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, alpha, beta, limit
    ):
        assert n_tiles_b == 2
        self.c_rows = c_rows
        self.n_out = c_cols // 2
        self.q_bytes = c_rows * self.n_out
        self.nb256 = self.n_out // 256
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.alpha = alpha
        self.beta = beta
        self.limit = limit
        rows_pad = (c_rows + 255) // 256 * 256
        gC = fx.rocdl.make_buffer_tensor(
            C,
            max_size=False,
            num_records_bytes=self.q_bytes + rows_pad * (self.n_out // 32),
        )
        self.rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(gC))

    def _swiglu(self, g_raw, u_raw):
        from flydsl.expr.typing import T

        F = fx.Float32
        lim = F(self.limit)
        g = g_raw.minimumf(lim)
        t = F(self.alpha) * (-g)
        m = t * F(1.4426950408889634)
        e = F(fmath.exp2(m, fastmath="afn"))
        y = e + F(1.0)
        r0 = F(rocdl.rcp(T.f32, y.ir_value()))
        err = F(fmath.fma(-y, r0, F(1.0)))
        sig = F(fmath.fma(err, r0, r0))
        sig = (y < F(2.0**126)).select(sig, F(0.0))
        gs = g * sig
        u = F(
            rocdl.fmed3(
                T.f32, u_raw.ir_value(), F(-self.limit).ir_value(), lim.ir_value()
            )
        )
        return (F(self.beta) + u) * gs

    def _store(self, value, off):
        from flydsl._mlir import ir
        from flydsl.expr.typing import T

        rocdl.raw_ptr_buffer_store(
            value.ir_value(),
            self.rsrc,
            off.ir_value(),
            fx.Int32(0).ir_value(),
            aux=ir.IntegerAttr.get(T.i32, 0),
        )

    def store(self, c_gate, c_up, base_row, block_n, wave_n):
        """``base_row``: first row of this wave's 64-row slice (a multiple of 64)."""
        from flydsl.expr.typing import T

        F = fx.Float32
        lane_c = self.lane_id % 16
        lane_g = self.lane_id // 16
        odd = (self.lane_id % 2) == 1
        oob = fx.Int32(0x7FFFFFF0)
        out_col_base = block_n * 128 + wave_n * 32
        lane_row0 = base_row + lane_g * 4
        q_lane_base = (
            lane_row0 * self.n_out + out_col_base + odd.select(lane_c + 15, lane_c)
        )
        # w4: group 4 block_n + wave_n -> 256-col block block_n // 2; row r -> 32-row
        # block r // 32, word wave_n * 16 + r % 16, byte block_n % 2 * 2 + r // 16 % 2.
        s_lane_base = (
            self.q_bytes
            + (base_row // 32 * self.nb256 + block_n // 2) * 256
            + (wave_n * 16 + lane_g * 4) * 4
            + (block_n % 2) * 2
        )
        for ti in range_constexpr(self.n_tiles_a):
            xs = []
            for tj in range_constexpr(2):
                g4 = Vec(c_gate[self.c_idx_fn(ti, tj)]).to(fx.BFloat16).to(F)
                u4 = Vec(c_up[self.c_idx_fn(ti, tj)]).to(fx.BFloat16).to(F)
                vals = [self._swiglu(F(g4[i]), F(u4[i])) for i in range_constexpr(4)]
                xs.append(Vec.from_elements(vals, F).to(fx.BFloat16).to(F))
            for i in range_constexpr(4):
                row = lane_row0 + ti * 16 + i
                x0 = F(xs[0][i])
                x1 = F(xs[1][i])
                # amax over the 32-column group: |x| bit patterns, max over the 16-lane
                # DPP row (row_ror folds into v_max_u32_dpp), then the 1e-10 floor.
                absmask = fx.Int32(0x7FFFFFFF)
                mi = fx.Int32(
                    farith.maxui(
                        (x0.bitcast(fx.Int32) & absmask).ir_value(),
                        (x1.bitcast(fx.Int32) & absmask).ir_value(),
                    )
                )
                for ror in (8, 4, 2, 1):
                    rot = fx.Int32(
                        rocdl.update_dpp(
                            fx.Int32.ir_type,
                            mi.ir_value(),
                            mi.ir_value(),
                            0x120 + ror,
                            0xF,
                            0xF,
                            True,
                        )
                    )
                    mi = fx.Int32(farith.maxui(mi.ir_value(), rot.ir_value()))
                floor_bits = F(1e-10).bitcast(fx.Int32)
                amax = fx.Int32(
                    farith.maxui(mi.ir_value(), floor_bits.ir_value())
                ).bitcast(F)
                bits = (amax * F(1.0 / 448.0)).bitcast(fx.Int32)
                code = (bits >> fx.Int32(23)) + (
                    (bits & fx.Int32(0x7FFFFF)) != fx.Int32(0)
                ).select(fx.Int32(1), fx.Int32(0))
                rexp = fx.Int32(254) - code
                rexp = (rexp > fx.Int32(0)).select(rexp, fx.Int32(0))
                recip = (rexp << fx.Int32(23)).bitcast(F)
                y0 = (x0 * recip).maximumf(F(-448.0)).minimumf(F(448.0))
                y1 = (x1 * recip).maximumf(F(-448.0)).minimumf(F(448.0))
                n0 = F(dpp_xor_f32(y0, 1))
                n1 = F(dpp_xor_f32(y1, 1))
                lo = odd.select(n1, y0)
                hi = odd.select(y1, n0)
                pk = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, lo, hi, fx.Int32(0), False))
                row_ok = row < self.c_rows
                q_off = row_ok.select(
                    q_lane_base + fx.Int32((ti * 16 + i) * self.n_out), oob
                )
                s_off = s_lane_base + fx.Int32(
                    (ti // 2) * self.nb256 * 256 + i * 4 + ti % 2
                )
                self._store(fx.Int16(pk.ir_value()), q_off)
                s_off = (row_ok & (lane_c == 0)).select(s_off, oob)
                self._store(fx.Int8(code.ir_value()), s_off)


class StoreF32Slab:
    """Stores the fp32 accumulators of split-K segment ``seg`` into slab ``seg``
    of a contiguous ``[k_split, c_rows, c_cols]`` fp32 workspace."""

    def __init__(self, W, k_split, seg, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.slab_row0 = seg * c_rows
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.n_elems = fx.Int32(k_split) * c_rows * c_cols
        gW = fx.rocdl.make_buffer_tensor(
            W, max_size=False, num_records_bytes=self.n_elems * 4
        )
        self.w_div = fx.logical_divide(gW, fx.make_layout(1, 1))
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

    def store(self, c_frag, base_row, base_col):
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    r = row + i
                    # Rows past c_rows would land in the next slab: drop them
                    # (and column overflow) out of bounds.
                    ok = (r < self.c_rows) & (col < self.c_cols)
                    idx = (self.slab_row0 + r) * self.c_cols + col
                    fx.memref_store_vec(fx.full(1, vec_f32[i], fx.Float32), self.reg)
                    fx.copy(
                        self.atom,
                        self.reg,
                        fx.slice(
                            self.w_div, (None, fx.Int32(ok.select(idx, self.n_elems)))
                        ),
                    )


def compile_mxfp8_gemm_8w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    b_preshuffled: bool = True,
    scale_preshuffled: bool = False,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    b_scale_block: str = "1x32",
    a_scale_block: str = "1x32",
    k_split: int = 1,
    epilogue: str = "bf16",
    swiglu: tuple | None = None,
):
    """Build the MXFP8 launcher.

    ``scale_preshuffled=True`` takes ``shuffle_scale_w4``-permuted A/B scales;
    ``False`` takes them row-major ``[rows, K // 32]`` as the quantizer emits them.
    ``b_scale_block="32x32"`` (row-major only) takes B's as ``[N // 32, K // 32]``,
    ``"128x128"`` as ``[N // 128, K // 128]``; ``a_scale_block="1x128"`` takes A's
    as the blockscale column-major ``[K // 128, M]`` (both unshuffled only).

    ``k_split > 1`` builds the split-K tail kernel: workgroup ``t * k_split + s``
    runs K-segment ``s`` (``K // k_split`` deep) of output tile ``t`` (row-major
    tile order, no XCD swizzle) and writes its fp32 partial to slab ``s`` of
    ``C``, a contiguous ``[k_split, c_m, c_n]`` fp32 workspace (ldc unused).

    ``epilogue="swiglu_mx8_w4"`` builds the gate_up GEMM of a SwiGLU-OAI MLP
    (``swiglu`` = (alpha, beta, limit)): B is the unpermuted ``[gate | up]``
    weight (c_n = 2I rows), each block tile computes 128 gate and the matching
    128 up columns, and the epilogue writes the e4m3 activation of the down
    projection plus its shuffle_scale_w4 e8m0 scales (StoreSwigluMx8W4).
    """
    assert b_scale_block in (
        "1x32",
        "32x32",
        "128x128",
    ), f"b_scale_block {b_scale_block}"
    assert a_scale_block in ("1x32", "1x128"), f"a_scale_block {a_scale_block}"
    assert not scale_preshuffled or (a_scale_block, b_scale_block) == (
        "1x32",
        "1x32",
    ), "a 32x32 / 1x128 / 128x128 scale is unshuffled only"
    assert (
        b_scale_block != "128x128" or K >= 512
    ), "a 128x128 B scale needs K >= 512 (a dword of K // 128 per chunk)"
    # A 256x256 tile already spends 128 KB of the 160 KB LDS; every other
    # BLOCK_M/BLOCK_N either overflows it or breaks the 2-threads-per-scale-row
    # staging split. Kept as an assert rather than dead generality.
    assert (
        BLOCK_M == 256 and BLOCK_N == 256
    ), "MXFP8 8-wave is specialized to a 256x256 block tile"
    assert (
        K % 256 == 0
    ), f"K must be a multiple of 256 (MX scale chunk staging), got {K}"

    assert (
        k_split >= 1 and K % (k_split * 256) == 0
    ), f"k_split {k_split} must leave K-segments of whole 256-K scale chunks"
    assert k_split == 1 or (
        not scale_preshuffled and a_scale_block == "1x32" and b_scale_block != "128x128"
    ), "a split-K segment takes row-major 1x32 A and 1x32 / 32x32 B scales"
    assert k_split == 1 or xcd_swizzle == 0, "split-K tiles run in row-major order"
    assert epilogue in ("bf16", "swiglu_mx8_w4"), epilogue
    SWIGLU = epilogue == "swiglu_mx8_w4"
    assert not SWIGLU or (
        k_split == 1 and scale_preshuffled and swiglu is not None
    ), "SwiGLU + MXFP8 (w4) epilogue: preshuffled scales, unsplit, (alpha, beta, L)"
    K_ITERS = K // BLOCK_K // k_split
    # Scale words are addressed by K-pair, so K must contain a whole number of
    # them (K % 256 above already guarantees it).
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
    if a_scale_block == "1x128":
        _scale_tag += "_xb1x128"
    if b_scale_block != "1x32":
        _scale_tag += f"_wb{b_scale_block}"
    _kname = (
        f"flydsl_{_layout_tag}_mxfp8_8w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_B16_"
        f"{waves_per_eu}x{xcd_swizzle}{_scale_tag}_k{K}"
        f"{f'_ks{k_split}' if k_split > 1 else ''}"
    )
    if SWIGLU:
        _kname += "_swiglu_mx8w4_" + "_".join(
            f"{v:g}".replace(".", "p").replace("-", "m") for v in swiglu
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
        ldc: fx.Int32,
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
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        if const_expr(k_split > 1):
            seg = fx.Int32(fx.block_idx.x) % k_split
            block_m, block_n = split_row_major_2d(
                fx.Int32(fx.block_idx.x) // k_split, n_blocks
            )
            k0 = seg * K_ITERS  # first K-step of this segment
            a_k_off = k0 * BLOCK_K
            b_k_off = k0 * B_K_STEP
        elif const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                ceildiv(c_m, BLOCK_M), n_blocks, wgm=xcd_swizzle
            )
        else:
            block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        if const_expr(SWIGLU):
            # LDS half 0 <- gate rows [128 block_n, +128), half 1 <- the matching up
            B0_gl_offset = (block_n * LDS_BLOCK_N) * K
            B1_gl_offset = (c_n // 2 + block_n * LDS_BLOCK_N) * K
        else:
            B0_gl_offset = (block_n * BLOCK_N) * K
            B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K
        if const_expr(k_split > 1):
            A0_gl_offset = A0_gl_offset + a_k_off
            A1_gl_offset = A1_gl_offset + a_k_off
            B0_gl_offset = B0_gl_offset + b_k_off
            B1_gl_offset = B1_gl_offset + b_k_off

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

        mfma = MxMfma(N_TILES_A, N_TILES_B, paired_tiles=scale_preshuffled)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        if const_expr(SWIGLU):
            store_c = StoreSwigluMx8W4(
                C,
                c_m,
                c_n,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                swiglu[0],
                swiglu[1],
                swiglu[2],
            )
        elif const_expr(k_split > 1):
            store_c = StoreF32Slab(
                C, k_split, seg, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        else:
            store_c = StoreBf16(C, c_m, c_n, ldc, mfma.idx, N_TILES_A, N_TILES_B)

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
                b_rows_per=32 if b_scale_block == "32x32" else 1,
                a_col_major=a_scale_block == "1x128",
                b_block128=b_scale_block == "128x128",
                c_m=c_m,
                # A scale row holds K // 32 bytes: 4 per 128-K step.
                k_byte_off=k0 * (BLOCK_K // 32) if k_split > 1 else None,
            )
            n_chunks = (K_ITERS + ScaleRawLds.CHUNK - 1) // ScaleRawLds.CHUNK
        # 16-row tile index of each wave's first row, per LDS half.
        a_base16 = [
            (block_m * BLOCK_M + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
            for h in range_constexpr(2)
        ]
        if const_expr(SWIGLU):
            b_base16 = [
                (h * (c_n // 2) + block_n * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16
                for h in range_constexpr(2)
            ]
        else:
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
        # The wave_m == 1 group took one extra barrier in the prologue, so this
        # one would have no partner: it would hold that group until the other
        # group's waves exit, serializing the two epilogues (ATT: ~5k cycles a
        # tile). Only the leading group pairs it; no LDS access follows.
        barrier_if_zero(rocdl.readfirstlane(fx.Int32.ir_type, wave_m))

        # Accumulators are already scaled by the MFMA: convert and store.
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        if const_expr(SWIGLU):
            # gate in c00 / c10 (LDS half 0), the matching up in c01 / c11
            store_c.store(c00_frag, c01_frag, base_row + 0, block_n, wave_n)
            store_c.store(c10_frag, c11_frag, base_row + LDS_BLOCK_M, block_n, wave_n)
        else:
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
        ldc: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N) * k_split
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            ldc,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
