# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 NVFP4 paged-attention tile with wave-owned tokens (one query tile).

NVFP4 K/V per (page, KV head): token-major [128, D/2] e2m1 payload (element 2j in
the low nibble of byte j) and [128, D/16] E4M3 block scales. Wave w owns tokens
[32 w, 32 w + 32) of every page, all 16 query rows and every output dim, so the
page loop has no CTA barrier and no LDS exchange:

* each wave DMAs its own 32 tokens (payload and scale rows, K and V) into a
  wave-private ring of ``nvfp4_buffers`` slots and waits only on its own copies;
* QK is ``v_wmma_scale16_f32_16x16x128_f8f6f4``: A is the raw e2m1 K rows with
  their E4M3 scale rows, read from LDS as the cache stores them; B is the query
  split once per task into FP8 terms (two for BF16, three for FP16) whose sum is
  the query to within 2^-21 of each 16-dim block's maximum, with power-of-two
  E8M0 block scales. No K widening;
* V is dequantized and transposed by the same instruction: A is the raw e2m1 V
  rows with their scale rows, B a constant selector (1.0 where k = n mod 16)
  whose E4M3 block scales are 1.0 for one 16-dim block and 0 elsewhere, so the
  accumulator holds V[token, 16 b + n] * scale exactly, eight tokens of one dim a
  lane: the BF16 PV operand after one packing conversion;
* softmax is online per wave; P never leaves registers (the QK accumulator is
  PV's B operand); after the loop the four waves merge (max, sum, output) once
  through LDS.

The result is the same BF16-operand PV as the CTA-shared pipeline, with QK in
FP32 over the exact query and K.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T

from ..utils import rcp_f32
from .op_gemm import Gfx1250Gemm
from .op_lds import Gfx1250Memory, _gptr
from .op_merge import Gfx1250PartitionMerge
from .traits import KV_COMPUTE_BLOCK, LOG2E

# Wave slot of one page: K payload [32, 64], V payload, K scales [32, 8], V scales.
# Every plane starts 256-byte aligned for the tensor DMA.
_TOKENS = 32
_ROW = 64
_SROW = 8
_PLANE_K = 0
_PLANE_V = _TOKENS * _ROW
_PLANE_KS = 2 * _TOKENS * _ROW
_PLANE_VS = _PLANE_KS + 256
WAVE_SLOT_BYTES = _PLANE_VS + 256
# Post-loop merge scratch (aliases the slots): per wave 8 dim tiles x 32 lanes x
# 8 FP32 outputs, then per (wave, row) the maximum and the sum.
_MERGE_OUT = 8 * 32 * 32
MERGE_BYTES = 4 * _MERGE_OUT + 4 * 16 * 8
# E4M3 1.0, the selector's block scale.
_E4M3_ONE = 0x38


def _attr(text):
    return ir.Attribute.parse(text)


def _wmma_scale16(a, b, c, scale_a, scale_b, fmt_a, fmt_b, scale_fmt_b="e4m3"):
    """16x16x128 f8f6f4 WMMA with one scale per 16 contraction values: E4M3 for
    the e2m1 A operand, ``scale_fmt_b`` for B (the assembler takes E4M3 block
    scales on e2m1 data only; the FP8 query uses E8M0).

    Raw ROCDL: FlyDSL's ``WMMAScale(block_size=16)`` atom hard-codes E8M0 scale
    formats, and NVFP4 needs E4M3."""
    return fx.Vector(
        fx.rocdl.wmma_scale16_f32_16x16x128_f8f6f4(
            fx.Vector.make_type(8, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(c),
            fx.as_ir_value(scale_a),
            fx.as_ir_value(scale_b),
            fmtA=_attr(f"#rocdl<matrix_format {fmt_a}>"),
            fmtB=_attr(f"#rocdl<matrix_format {fmt_b}>"),
            modC=_attr("#rocdl<wmma_c_modifier none>"),
            scaleAType=_attr("#rocdl<wmma_matrix_scale row0>"),
            fmtScaleA=_attr("#rocdl<wmma_matrix_scale_format e4m3>"),
            scaleBType=_attr("#rocdl<wmma_matrix_scale row0>"),
            fmtScaleB=_attr(f"#rocdl<wmma_matrix_scale_format {scale_fmt_b}>"),
            reuseA=False,
            reuseB=False,
        ).result
    )


def _f32_bits(value):
    return fx.Vector.from_elements([fx.Float32(value)], dtype=fx.Float32).bitcast(
        fx.Int32
    )[0]


def _pow2(k):
    """2^k as FP32 for an Int32 k in [-126, 127]."""
    return fx.Vector.from_elements([fx.Int32((k + 127) << 23)], dtype=fx.Int32).bitcast(
        fx.Float32
    )[0]


def _e8m0_pow2(k):
    """E8M0 byte of 2^k."""
    return fx.Int32(k + 127)


class Gfx1250Nvfp4WavePipeline:
    """One query tile (<= 16 rows); every wave owns 32 tokens of each page."""

    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        assert self.t.head_dim == 128 and self.t.block_size == 128
        assert self.t.m_tiles == 1
        self.mem = Gfx1250Memory(ctx)
        self.merge = Gfx1250PartitionMerge(ctx, self.mem)

    def _tile(self, vh):
        """The 16-dim output tile this wave stores for ``vh``: tiles 4 vh + w
        (dims 64 vh + 16 w), or under the MXFP8 output 2 w + vh, so each wave
        holds whole 32-dim groups (dims 32 w + 16 vh)."""
        if const_expr(self.t.quant_mx):
            return self.ctx.warp * 2 + vh
        return vh * 4 + self.ctx.warp

    def _dim(self, vh):
        """First of the 8 dims lane (row, half) stores for ``vh``."""
        ctx = self.ctx
        if const_expr(self.t.quant_mx):
            return ctx.warp * 32 + vh * 16 + ctx.half * 8
        return vh * 64 + ctx.warp * 16 + ctx.half * 8

    def _slot(self, slot):
        return (slot * self.t.NWARP + self.ctx.warp) * WAVE_SLOT_BYTES

    @flyc.jit
    def _stage(self, token_base, slot):
        """DMA this wave's 32 tokens of the page at ``token_base`` into ``slot``.

        Rows at or past the context (and pages past it, read as the row's first
        block) arrive zero-filled, so they score and weigh zero."""
        ctx = self.ctx
        safe = (token_base < ctx.planned_context).select(token_base, fx.Int32(0))
        table = fx.recast_iter(fx.Int32, ctx.block_tables_ptr)
        page = fx.Int32(table[ctx.planned_seq * ctx.max_blocks_per_seq + safe // 128])
        first = token_base + ctx.warp * _TOKENS
        valid = fx.max(
            fx.Int32(0), fx.min(fx.Int32(_TOKENS), ctx.planned_context - first)
        )
        base = self._slot(slot)
        # tdm_copy addresses in BF16 elements: every offset here is even.
        payload = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * (128 * _ROW // 2) + fx.Int64(
            ctx.warp * (_TOKENS * _ROW // 2)
        )
        scales = (
            fx.Int64(page) * fx.Int64(ctx.stride_ks_block)
            + fx.Int64(ctx.kv_h) * fx.Int64(ctx.stride_ks_head)
        ) // 2 + fx.Int64(ctx.warp * (_TOKENS * _SROW // 2))
        for ptr, plane in (
            (ctx.key_cache_ptr, _PLANE_K),
            (ctx.value_cache_ptr, _PLANE_V),
        ):
            self.mem.tdm_copy(
                _gptr(ptr, fx.BFloat16, 16),
                payload,
                base + plane,
                (_TOKENS * _ROW // 2,),
                (1,),
                (valid * (_ROW // 2),),
                8,
                8,
            )
        for ptr, plane in (
            (ctx.key_scale_ptr, _PLANE_KS),
            (ctx.value_scale_ptr, _PLANE_VS),
        ):
            self.mem.tdm_copy(
                _gptr(ptr, fx.BFloat16, 16),
                scales,
                base + plane,
                (_TOKENS * _SROW // 2,),
                (1,),
                (valid * (_SROW // 2),),
                8,
                8,
            )

    def _rows(self, base, plane, n):
        """A operand of token tile ``n``: e2m1 rows (lane16 = token) and the
        token's eight E4M3 block scales."""
        ctx = self.ctx
        token = n * 16 + ctx.lane16
        row = base + plane + token * _ROW + ctx.half * 16
        words = []
        for j in range_constexpr(2):
            frag = self.mem.load(row + j * 32, fx.Int32, 4)
            words.extend([frag[i] for i in range_constexpr(4)])
        scale_plane = _PLANE_KS if plane == _PLANE_K else _PLANE_VS
        scale = self.mem.load(base + scale_plane + token * _SROW, fx.Int64, 1)[0]
        return fx.Vector.from_elements(words, dtype=fx.Int32), scale

    def _query(self):
        """This lane's query row as FP8 terms for the B operand of K128 WMMA.

        Lane (row lane16, half) holds dims 16 (2 j + half) + [0, 16), j < 4: four
        whole 16-dim scale blocks. Term i scales block b by 2^(E_b - 7 - 4 i),
        E_b = floor(log2(max |q|)), so the first term holds each block's maximum
        in [128, 256] and every later term the previous residual (exact in FP32)
        in at most 128: two terms reproduce BF16 and three FP16 to within 2^-21
        of the block maximum. Returns per term (operand, i64 scales)."""
        ctx, t = self.ctx, self.t
        query = fx.recast_iter(t.q_dtype, ctx.query_ptr)
        # Padding rows (lane16 >= rows) read row 0 and are zeroed: their scores
        # are masked and never stored.
        live = ctx.lane16 < t.rows
        row = fx.min(ctx.lane16, fx.Int32(t.rows - 1))
        qbase = (
            fx.Int64(ctx.planned_seq) * t.query_length
            + ctx.query_begin
            + row // t.query_group_size
        ) * ctx.stride_q_row + (
            ctx.kv_h * t.query_group_size + row % t.query_group_size
        ) * ctx.stride_q_head
        blocks = []
        for j in range_constexpr(4):
            values = fx.Vector(
                fx.ptr_load(
                    fx.add_offset(query, qbase + j * 32 + ctx.half * 16),
                    result_type=fx.Vector.make_type(16, t.q_dtype),
                )
            ).to(fx.Float32)
            keep = fx.Vector.from_elements(
                [live.select(fx.Float32(1.0), fx.Float32(0.0))], dtype=fx.Float32
            ).broadcast_to(16)
            blocks.append(values * keep)
        terms = 3 if t.query_dtype == "f16" else 2
        operands = [[] for _ in range_constexpr(terms)]
        scale_words = [fx.Int32(0) for _ in range_constexpr(terms)]
        for j in range_constexpr(4):
            amax = fmath.absf(blocks[j]).reduce(ReductionOp.MAX)
            exponent = ((_f32_bits(amax) >> 23) & 0xFF) - 127
            residual = blocks[j]
            for i in range_constexpr(terms):
                # Zero blocks (exponent -127) clamp to any scale; nonzero
                # FP32 exponents keep k well inside 2^k's normal range.
                k = fx.max(
                    fx.min(exponent - (7 + 4 * i), fx.Int32(120)), fx.Int32(-120)
                )
                scaled = residual * fx.Vector.from_elements(
                    [_pow2(-k)], dtype=fx.Float32
                ).broadcast_to(16)
                words = []
                for h in range_constexpr(2):
                    pair = self.mem.fp8(
                        fx.Vector.from_elements(
                            [scaled[h * 8 + e] for e in range_constexpr(8)],
                            dtype=fx.Float32,
                        )
                    )
                    words.extend([pair[0], pair[1]])
                    if const_expr(i + 1 < terms):
                        back = fx.Vector(
                            fx.rocdl.cvt_scale_pk8_f32_fp8(
                                T.vec(8, T.f32),
                                fx.as_ir_value(
                                    fx.Vector.from_elements(
                                        [pair[0], pair[1]], dtype=fx.Int32
                                    )
                                ),
                                fx.as_ir_value(fx.Int32(0x7F7F7F7F)),
                                0,
                            )
                        )
                        if const_expr(h == 0):
                            low = back
                        else:
                            high = back
                operands[i].extend(words)
                scale_words[i] = scale_words[i] | (_e8m0_pow2(k) << (8 * j))
                if const_expr(i + 1 < terms):
                    quantized = fx.Vector.from_elements(
                        [low[e] for e in range_constexpr(8)]
                        + [high[e] for e in range_constexpr(8)],
                        dtype=fx.Float32,
                    )
                    residual = residual - quantized * fx.Vector.from_elements(
                        [_pow2(k)], dtype=fx.Float32
                    ).broadcast_to(16)
        result = []
        lower = ctx.half == 0
        for i in range_constexpr(terms):
            mine = scale_words[i]
            other = fx.Int32(mine.shuffle_xor(16, t.WAVE))
            # Byte b of the row's scales is block b: even blocks from half 0.
            even = lower.select(mine, other)
            odd = lower.select(other, mine)
            lo = (
                (even & 0xFF)
                | ((odd & 0xFF) << 8)
                | ((even & 0xFF00) << 8)
                | ((odd & 0xFF00) << 16)
            )
            hi = (
                ((even >> 16) & 0xFF)
                | (((odd >> 16) & 0xFF) << 8)
                | (((even >> 24) & 0xFF) << 16)
                | (((odd >> 24) & 0xFF) << 24)
            )
            scale = fx.Vector.from_elements([lo, hi], dtype=fx.Int32).bitcast(fx.Int64)[
                0
            ]
            result.append((fx.Vector.from_elements(operands[i], dtype=fx.Int32), scale))
        return result

    def _selector(self):
        """B = 1.0 (e2m1 0b0010) where k = n mod 16: dword w of lane n holds
        k mod 16 in [8 (w % 2), 8 (w % 2) + 8)."""
        ctx = self.ctx
        n = ctx.lane16
        one = fx.Int32(2) << ((n % 8) * 4)
        even = (n < 8).select(one, fx.Int32(0))
        odd = (n >= 8).select(one, fx.Int32(0))
        return fx.Vector.from_elements([even, odd] * 4, dtype=fx.Int32)

    @flyc.jit
    def run(self):
        ctx, t = self.ctx, self.t
        ctx.tid = fx.Int32(gpu.thread_id("x"))
        ctx.warp, ctx.lane = ctx.tid // t.WAVE, ctx.tid % t.WAVE
        ctx.lane16, ctx.half = ctx.lane % 16, ctx.lane // 16
        ctx.kv_h = fx.Int32(gpu.block_id("y")) // t.query_splits
        ctx.query_begin = (fx.Int32(gpu.block_id("y")) % t.query_splits) * (
            t.query_length // t.query_splits
        )
        self.mem.allocate()
        nb = t.nvfp4_buffers
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        # The first pages' DMA overlaps the query split.
        self._stage(begin, fx.Int32(0))
        for ahead in range_constexpr(1, nb - 1):
            if begin + ahead * 128 < end:
                self._stage(begin + ahead * 128, fx.Int32(ahead))
        q_terms = self._query()
        selector = self._selector()
        # Block scales of the selector: 1.0 (E4M3) on block b only. They must
        # reach the WMMA in VGPRs: a uniform i64 is emitted as an inline constant
        # (block 0's 0x38) or an SGPR pair, and the GPU reads either wrongly
        # (probes: VGPR scales exact for every block; inline 56 corrupts block
        # 0, SGPR pairs every block). OR a lane-varying zero that the compiler
        # cannot fold: lane + context is a nonnegative Int32.
        lane_zero = fx.Int64((ctx.lane + ctx.planned_context) >> 31)
        selector_scales = [
            fx.Int64(_E4M3_ONE << (8 * b)) | lane_zero for b in range_constexpr(8)
        ]
        row = ctx.lane16
        upper = (
            ctx.planned_context
            - t.query_length
            + ctx.query_begin
            + row // t.query_group_size
            + 1
        )
        initial = [fx.Float32(float("-inf")), fx.Float32(0.0)]
        initial.extend(
            [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range_constexpr(8)]
        )
        zeros = fx.Vector.filled(8, 0.0, fx.Float32)
        for token_i, state in range(begin, end, 128, init=initial):
            token_base = fx.Int32(token_i)
            page = (token_base - begin) // 128
            slot = page % nb
            # This wave's copies of the page are the oldest in flight.
            ahead = token_base + (nb - 1) * 128
            # Four copies a page; the pages issued after this one may stay in flight.
            if const_expr(nb == 2):
                tdm_ops.tensor_wait(0)
            elif const_expr(nb == 3):
                if token_base + 128 < end:
                    tdm_ops.tensor_wait(4)
                else:
                    tdm_ops.tensor_wait(0)
            else:
                if token_base + 256 < end:
                    tdm_ops.tensor_wait(8)
                else:
                    if token_base + 128 < end:
                        tdm_ops.tensor_wait(4)
                    else:
                        tdm_ops.tensor_wait(0)
            if ahead < end:
                self._stage(ahead, (page + nb - 1) % nb)
            base = self._slot(slot)
            units = []
            v_tiles = []
            for n in range_constexpr(2):
                k_rows, k_scale = self._rows(base, _PLANE_K, n)
                scores = zeros
                for operand, scale in q_terms:
                    scores = _wmma_scale16(
                        k_rows,
                        operand,
                        scores,
                        k_scale,
                        scale,
                        "fp4_e2m1",
                        "fp8_e4m3",
                        "e8",
                    )
                for r in range_constexpr(8):
                    token = token_base + ctx.warp * _TOKENS + n * 16 + ctx.half * 8 + r
                    visible = (token < upper) & (row < t.rows)
                    units.append(
                        visible.select(
                            scores[r] * fx.Float32(t.softmax_scale * LOG2E),
                            fx.Float32(float("-inf")),
                        )
                    )
                v_rows, v_scale = self._rows(base, _PLANE_V, n)
                v_tiles.append(
                    [
                        _wmma_scale16(
                            v_rows,
                            selector,
                            zeros,
                            v_scale,
                            selector_scales[b],
                            "fp4_e2m1",
                            "fp4_e2m1",
                        )
                        for b in range_constexpr(8)
                    ]
                )
            scores = fx.Vector.from_elements(units, dtype=fx.Float32)
            tile_max = scores.reduce(ReductionOp.MAX)
            tile_max = fx.maxnumf(tile_max, tile_max.shuffle_xor(16, t.WAVE))
            old_max = state[0]
            new_max = fx.maxnumf(old_max, tile_max)
            safe_max = (new_max > fx.Float32(float("-inf"))).select(
                new_max, fx.Float32(0.0)
            )
            corr = fx.Float32(fx.exp2(old_max - safe_max, fastmath="afn"))
            probs = fx.Vector(
                fx.exp2(
                    scores
                    - fx.Vector.from_elements(
                        [safe_max], dtype=fx.Float32
                    ).broadcast_to(16),
                    fastmath="afn",
                )
            )
            tile_sum = probs.reduce(ReductionOp.ADD)
            tile_sum = tile_sum + tile_sum.shuffle_xor(16, t.WAVE)
            # The QK accumulators are PV's B operand: lane (row, half) holds
            # tokens 8 half + [0, 8) and 16 + 8 half + [0, 8) of this wave.
            p_operand = probs.to(fx.BFloat16)
            next_state = [new_max, state[1] * corr + tile_sum]
            factor = fx.Vector.from_elements([corr], dtype=fx.Float32).broadcast_to(8)
            for b in range_constexpr(8):
                v_operand = fx.Vector.from_elements(
                    [v_tiles[0][b][e] for e in range_constexpr(8)]
                    + [v_tiles[1][b][e] for e in range_constexpr(8)],
                    dtype=fx.Float32,
                ).to(fx.BFloat16)
                next_state.append(
                    Gfx1250Gemm.mma(
                        v_operand, p_operand, state[2 + b] * factor, 32, True
                    )
                )
            final = yield next_state
        # Merge the four waves' (maximum, sum, output) through the idle slots.
        tdm_ops.tensor_wait(0)
        gpu.barrier()
        for b in range_constexpr(8):
            self.mem.store(
                ctx.warp * _MERGE_OUT + (b * 32 + ctx.lane) * 32,
                final[2 + b],
                fx.Float32,
            )
        stats = 4 * _MERGE_OUT
        if ctx.half == 0:
            self.mem.store(
                stats + (ctx.warp * 16 + ctx.lane16) * 8,
                fx.Vector.from_elements([final[0], final[1]], dtype=fx.Float32),
                fx.Float32,
            )
        gpu.barrier()
        wave_stats = [
            self.mem.load(stats + (w * 16 + ctx.lane16) * 8, fx.Float32, 2)
            for w in range_constexpr(t.NWARP)
        ]
        row_max = wave_stats[0][0]
        for w in range_constexpr(1, t.NWARP):
            row_max = fx.maxnumf(row_max, wave_stats[w][0])
        safe_max = (row_max > fx.Float32(float("-inf"))).select(
            row_max, fx.Float32(0.0)
        )
        weights = [
            fx.Float32(fx.exp2(wave_stats[w][0] - safe_max, fastmath="afn"))
            for w in range_constexpr(t.NWARP)
        ]
        denom = fx.Float32(0.0)
        for w in range_constexpr(t.NWARP):
            denom = denom + weights[w] * wave_stats[w][1]
        merged = [row_max, denom]
        for vh in range_constexpr(2):
            b = self._tile(vh)
            out = fx.Vector.filled(8, 0.0, fx.Float32)
            for w in range_constexpr(t.NWARP):
                out = out + self.mem.load(
                    w * _MERGE_OUT + (b * 32 + ctx.lane) * 32, fx.Float32, 8
                ) * fx.Vector.from_elements(
                    [weights[w]], dtype=fx.Float32
                ).broadcast_to(
                    8
                )
            merged.append(out)
        self._store(merged)

    @flyc.jit
    def _store(self, final):
        """The CTA-shared BF16 pipeline's epilogue: wave w stores dims
        ``_dim(vh)`` + [0, 8) of row lane16, and with ``quant_output``
        its FP8 output (``Gfx1250OutputQuant.store_cta``) from the same values."""
        ctx, t = self.ctx, self.t
        slot = fx.Int64(ctx.kv_h) * fx.Int64(ctx.capacity) + fx.Int64(ctx.task_slot)

        def store(direct):
            pout = fx.recast_iter(
                t.q_dtype,
                ctx.output_ptr if t.merged_output and direct else ctx.pout_ptr,
            )
            pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
            psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
            row = ctx.lane16
            output_row = ctx.query_begin * t.query_group_size + row
            partial_row = slot * (t.rows * t.query_splits) + output_row
            denom = final[1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            if row < t.rows:
                for vh in range_constexpr(2):
                    d = self._dim(vh)
                    value = (
                        final[2 + vh]
                        * fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(
                            8
                        )
                    ).to(t.q_dtype)
                    offset = (
                        self.merge.output_offset(output_row, d)
                        if direct
                        else partial_row * t.head_dim + d
                    )
                    if const_expr(t.merge_partitions and not direct):
                        self.merge.publish(fx.add_offset(pout, offset), value)
                    else:
                        fx.ptr_store(value, fx.add_offset(pout, offset))
                if const_expr(not direct):  # noqa: SIM102
                    if ctx.warp == 0 and ctx.half == 0:
                        row_max = final[0] * fx.Float32(1.0 / LOG2E)
                        if const_expr(t.merge_partitions):
                            self.merge.publish(
                                fx.add_offset(pmax, partial_row), row_max
                            )
                            self.merge.publish(fx.add_offset(psum, partial_row), denom)
                        else:
                            pmax[partial_row] = row_max
                            psum[partial_row] = denom

        def store_quantized():
            # The FP8 epilogue of the CTA-shared pipelines: one query token,
            # lane16 picks its row (GQA head), warp, half and vh its dims, so
            # this CTA holds all of the token's values.
            denom = final[1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            norm = fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(8)
            values, offsets = [], []
            for vh in range_constexpr(2):
                values.append((final[2 + vh] * norm).to(t.q_dtype))
                offsets.append(self.merge.output_offset(ctx.lane16, self._dim(vh)))
            # One 16-row tile; the four waves (parts) split its dims.
            ctx.quant.store_cta(
                self.mem,
                ctx.planned_seq,
                ctx.lane,
                ctx.warp,
                t.NWARP,
                [(ctx.lane16, values, offsets)],
                fx.recast_iter(
                    t.q_dtype,
                    ctx.output_ptr if t.merged_output else ctx.pout_ptr,
                ),
            )

        if const_expr(t.merged_output):
            if ctx.task_count == 1:
                if const_expr(t.quant_output):
                    store_quantized()
                else:
                    store(True)
            else:
                store(False)
                if const_expr(t.merge_partitions):
                    self.merge.arrive()
        elif const_expr(t.quant_output):
            store_quantized()
        else:
            store(t.direct_output)
