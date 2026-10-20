# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors


"""Wave32 staging, online softmax and output accumulation for PA decode."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp

from ..utils import rcp_f32
from .op_gemm_gfx1250 import Gfx1250Gemm
from .op_lds_gfx1250 import Gfx1250Memory
from .traits import KV_COMPUTE_BLOCK, LOG2E


class Gfx1250Pipeline:
    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)

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
        ctx.n_kv = fx.Int32(gpu.grid_dim.y) // t.query_splits
        self.mem.allocate()
        if const_expr(t.q_tiles == t.m_tiles):
            self.mem.stage_query()
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        self.mem.stage_kv(begin, fx.Int32(0))
        # Per-M-tile state: maximum, denominator, then wave-owned D fragments.
        vh_count = t.head_dim // 64
        initial = []
        for m in range_constexpr(t.m_tiles):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [
                    fx.Vector.filled(8, 0.0, fx.Float32)
                    for vh in range_constexpr(vh_count)
                ]
            )
        state_stride = 2 + vh_count
        if const_expr(t.packed_lds):
            initial.append(fx.Float32(1.0))
        resident_q = (
            not t.bf16 and (t.head_dim == 128 or t.packed_lds) and t.m_tiles == 1
        )
        if const_expr(resident_q):
            gpu.barrier()
            q_operands = [
                self.mem.operand(d * 128, t.q_stride, 128, ctx.lane16, ctx.half)
                for d in range_constexpr(t.head_dim // 128)
            ]
            q_scale = fx.Float32(
                self.mem.load(ctx.lane16 * t.q_stride + t.head_dim, fx.Float32)[0]
            )
        for token_i, state in range(begin, end, t.TOKENS, init=initial):
            token_base = fx.Int32(token_i)
            buffer = ((token_base - begin) // t.TOKENS) % t.buffers
            if const_expr(t.use_tdm):
                tdm_ops.tensor_wait(0)
                if const_expr(t.async_page):
                    fx.rocdl.s_wait_asynccnt(0)
            else:
                fx.rocdl.s_wait_asynccnt(0)
            gpu.barrier()
            kv_off = t.q_bytes + buffer * t.kv_stride
            if const_expr(t.buffers == 2):  # noqa: SIM102
                if token_base + t.TOKENS < end:
                    self.mem.stage_kv(token_base + t.TOKENS, 1 - buffer)
            vnorm = fx.Float32(1.0)
            vfactor = fx.Float32(1.0 if t.bf16 else 1.0 / t.FP8_MAX)
            if const_expr(t.per_token_kv):
                # Every wave reduces the tile's scales, avoiding a CTA barrier.
                scale_base = kv_off + t.k_bytes + t.v_bytes + t.TOKENS * 4
                vmax = fx.maxnumf(
                    fx.Float32(self.mem.load(scale_base + ctx.lane * 4, fx.Float32)[0]),
                    fx.Float32(
                        self.mem.load(scale_base + (ctx.lane + 32) * 4, fx.Float32)[0]
                    ),
                )
                if const_expr(t.TOKENS == 128):
                    vmax = fx.maxnumf(
                        vmax,
                        fx.Float32(
                            self.mem.load(scale_base + (ctx.lane + 64) * 4, fx.Float32)[
                                0
                            ]
                        ),
                    )
                    vmax = fx.maxnumf(
                        vmax,
                        fx.Float32(
                            self.mem.load(scale_base + (ctx.lane + 96) * 4, fx.Float32)[
                                0
                            ]
                        ),
                    )
                for shift in (16, 8, 4, 2, 1):
                    vmax = fx.maxnumf(vmax, vmax.shuffle_xor(shift, t.WAVE))
                vfactor = vmax * fx.Float32(1.0 / t.FP8_MAX)
                if const_expr(t.packed_lds):
                    # Empty planned tails carry no new mass. Keep the prior
                    # scale so their zero V contribution preserves the output.
                    vfactor = (vfactor > 0.0).select(vfactor, fx.Float32(state[-1]))
                vnorm = fx.Float32(rcp_f32(fx.maxnumf(vfactor, fx.Float32(1e-20))))
            elif const_expr(not t.bf16):
                vs = fx.recast_iter(fx.Float32, ctx.value_scale_ptr)
                vfactor = fx.Float32(vs[0]) * fx.Float32(1.0 / t.FP8_MAX)
                if const_expr(t.packed_lds):
                    # The fused accumulator carries unscaled V until epilogue.
                    # A zero scalar V scale must produce zero without 0 * inf.
                    vnorm = fx.Float32(
                        rcp_f32((vfactor != 0.0).select(vfactor, fx.Float32(1.0)))
                    )
            next_state = []
            for m in range_constexpr(t.m_tiles):
                if const_expr(t.q_tiles != t.m_tiles):
                    self.mem.stage_query(m)
                    gpu.barrier()
                q_tile = m if t.q_tiles == t.m_tiles else 0
                row = m * 16 + ctx.lane16
                qscale = fx.Float32(1.0)
                if const_expr(resident_q):
                    qscale = q_scale
                elif const_expr(not t.bf16):
                    qscale = fx.Float32(
                        self.mem.load(
                            (q_tile * 16 + ctx.lane16) * t.q_stride + t.head_dim,
                            fx.Float32,
                        )[0]
                    )
                scale = qscale * fx.Float32(t.softmax_scale * LOG2E)
                if const_expr(not t.per_token_kv and not t.bf16):
                    ks = fx.recast_iter(fx.Float32, ctx.key_scale_ptr)
                    scale = scale * fx.Float32(ks[0])
                upper = (
                    ctx.planned_context
                    - t.query_length
                    + ctx.query_begin
                    + row // t.query_group_size
                    + 1
                )
                units, values = [], []
                qk_k = 32 if t.bf16 else (128 if t.head_dim % 128 == 0 else 64)
                for n in range_constexpr(t.TOKENS // 64):
                    scores = fx.Vector.filled(8, 0.0, fx.Float32)
                    for d in range_constexpr(t.head_dim // qk_k):
                        if const_expr(t.packed_lds):
                            if const_expr(t.TOKENS == 128):
                                a = self.mem.page_k_operand(
                                    kv_off + d * 128 * 128,
                                    ctx.warp * 32 + n * 16,
                                    ctx.lane16,
                                    ctx.half,
                                )
                            else:
                                a = self.mem.packed_operand(
                                    kv_off + ctx.warp * t.head_dim * 16 + d * qk_k * 16,
                                    16,
                                    qk_k,
                                    ctx.lane16,
                                    ctx.half,
                                )
                        else:
                            a = self.mem.operand(
                                kv_off
                                + ctx.warp * 16 * t.q_stride
                                + d * qk_k * t.elem_bytes,
                                t.q_stride,
                                qk_k,
                                ctx.lane16,
                                ctx.half,
                            )
                        if const_expr(resident_q):
                            b = q_operands[d]
                        else:
                            b = self.mem.operand(
                                q_tile * 16 * t.q_stride + d * qk_k * t.elem_bytes,
                                t.q_stride,
                                qk_k,
                                ctx.lane16,
                                ctx.half,
                            )
                        scores = Gfx1250Gemm.mma(a, b, scores, qk_k, t.bf16)
                    if const_expr(t.per_token_kv):
                        scale_token = ctx.warp * (t.TOKENS // 4) + n * 16 + ctx.half * 8
                        ks = self.mem.load(
                            kv_off + t.k_bytes + t.v_bytes + scale_token * 4,
                            fx.Float32,
                            8,
                        )
                        vs = self.mem.load(
                            kv_off
                            + t.k_bytes
                            + t.v_bytes
                            + (t.TOKENS + scale_token) * 4,
                            fx.Float32,
                            8,
                        )
                        values.extend([vs[r] for r in range_constexpr(8)])
                    for r in range_constexpr(8):
                        token = ctx.warp * (t.TOKENS // 4) + n * 16 + ctx.half * 8 + r
                        value = scores[r] * scale
                        if const_expr(t.per_token_kv):
                            value = value * fx.Float32(ks[r])
                        visible = (token_base + token < upper) & (row < t.rows)
                        if const_expr(t.sliding_window > 0):
                            visible = visible & (
                                token_base + token >= upper - t.sliding_window
                            )
                        units.append(visible.select(value, fx.Float32(float("-inf"))))
                scores = fx.Vector.from_elements(units, dtype=fx.Float32)
                if const_expr(t.per_token_kv):
                    vscales = fx.Vector.from_elements(values, dtype=fx.Float32)
                local_max = scores.reduce(ReductionOp.MAX)
                local_max = fx.maxnumf(local_max, local_max.shuffle_xor(16, t.WAVE))
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + (ctx.lane16 * t.NWARP + ctx.warp) * 4,
                        local_max,
                        fx.Float32,
                    )
                gpu.barrier()
                tile_max = self.mem.load(
                    t.stats_offset + ctx.lane16 * t.NWARP * 4, fx.Float32, t.NWARP
                ).reduce(ReductionOp.MAX)
                old_max = state[m * state_stride]
                new_max = fx.maxnumf(old_max, tile_max)
                safe_max = (new_max > fx.Float32(float("-inf"))).select(
                    new_max, fx.Float32(0.0)
                )
                # Approximate exponentials avoid the library underflow fixup.
                # Keep infinity semantics for masked rows and empty history.
                corr = fx.Float32(fx.exp2(old_max - safe_max, fastmath="afn"))
                probs = fx.Vector(
                    fx.exp2(
                        scores
                        - fx.Vector.from_elements(
                            [fx.Float32(safe_max)], dtype=fx.Float32
                        ).broadcast_to(t.TOKENS // 8),
                        fastmath="afn",
                    )
                )
                local_sum = probs.reduce(ReductionOp.ADD)
                local_sum = local_sum + local_sum.shuffle_xor(16, t.WAVE)
                pscale = []
                for r in range_constexpr(t.TOKENS // 8):
                    if const_expr(t.per_token_kv):
                        vs = fx.Float32(vscales[r])
                        pscale.append(probs[r] * vs * vnorm)
                    else:
                        pscale.append(
                            probs[r] * fx.Float32(1.0 if t.bf16 else t.FP8_MAX)
                        )
                for n in range_constexpr(t.TOKENS // 64):
                    pvalues = fx.Vector.from_elements(
                        [pscale[n * 8 + r] for r in range_constexpr(8)],
                        dtype=fx.Float32,
                    )
                    p = pvalues.to(fx.BFloat16) if t.bf16 else self.mem.fp8(pvalues)
                    self.mem.store(
                        t.p_offset
                        + ctx.lane16 * t.v_stride
                        + (ctx.warp * (t.TOKENS // 4) + n * 16 + ctx.half * 8)
                        * t.elem_bytes,
                        p,
                        fx.BFloat16 if t.bf16 else fx.Int32,
                    )
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + 256 + (ctx.lane16 * t.NWARP + ctx.warp) * 4,
                        local_sum,
                        fx.Float32,
                    )
                # Prefetch V operands before waiting for cross-wave P stores.
                pv_k = 32 if t.bf16 else t.TOKENS
                if const_expr(t.packed_lds):
                    v_operands = [
                        [
                            self.mem.packed_operand(
                                kv_off + t.k_bytes + (vh * 64 + ctx.warp * 16) * 16,
                                t.head_dim,
                                t.TOKENS,
                                ctx.lane16,
                                ctx.half,
                            )
                        ]
                        for vh in range_constexpr(vh_count)
                    ]
                else:
                    v_operands = [
                        [
                            self.mem.operand(
                                kv_off
                                + t.k_bytes
                                + (vh * 64 + ctx.warp * 16) * t.v_stride
                                + d * pv_k * t.elem_bytes,
                                t.v_stride,
                                pv_k,
                                ctx.lane16,
                                ctx.half,
                            )
                            for d in range_constexpr(t.TOKENS // pv_k)
                        ]
                        for vh in range_constexpr(vh_count)
                    ]
                gpu.barrier()
                denom = state[m * state_stride + 1] * corr + self.mem.load(
                    t.stats_offset + 256 + ctx.lane16 * t.NWARP * 4, fx.Float32, t.NWARP
                ).reduce(ReductionOp.ADD)
                next_state.extend([new_max, denom])
                p_operands = [
                    self.mem.operand(
                        t.p_offset + d * pv_k * t.elem_bytes,
                        t.v_stride,
                        pv_k,
                        ctx.lane16,
                        ctx.half,
                    )
                    for d in range_constexpr(t.TOKENS // pv_k)
                ]
                for vh in range_constexpr(vh_count):
                    # Packed outputs remain normalized by the previous V
                    # factor. Rescale before WMMA to fuse its accumulation.
                    out = (
                        state[m * state_stride + 2 + vh]
                        * fx.Vector.from_elements(
                            [corr * state[-1] * vnorm], dtype=fx.Float32
                        ).broadcast_to(8)
                        if t.packed_lds
                        else fx.Vector.filled(8, 0.0, fx.Float32)
                    )
                    for d in range_constexpr(t.TOKENS // pv_k):
                        out = Gfx1250Gemm.mma(
                            v_operands[vh][d], p_operands[d], out, pv_k, t.bf16
                        )
                    if const_expr(t.packed_lds):
                        next_state.append(out)
                    else:
                        next_state.append(
                            state[m * state_stride + 2 + vh]
                            * fx.Vector.from_elements(
                                [fx.Float32(corr)], dtype=fx.Float32
                            ).broadcast_to(8)
                            + out
                            * fx.Vector.from_elements(
                                [fx.Float32(vfactor)], dtype=fx.Float32
                            ).broadcast_to(8)
                        )
                # The next iteration's barrier retires the final M-tile reads
                # before a double buffer is reused. Intra-iteration reuse and
                # the single-buffer staging path still need an explicit barrier.
                if const_expr(m + 1 < t.m_tiles or t.buffers == 1):
                    gpu.barrier()
            if const_expr(t.buffers == 1):  # noqa: SIM102
                if token_base + t.TOKENS < end:
                    self.mem.stage_kv(token_base + t.TOKENS, fx.Int32(0))
            if const_expr(t.packed_lds):
                next_state.append(vfactor)
            final = yield next_state
        pout = fx.recast_iter(t.q_dtype, ctx.pout_ptr)
        pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
        psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
        slot = fx.Int64(ctx.kv_h) * fx.Int64(gpu.grid_dim.x) + fx.Int64(
            gpu.block_id("x")
        )
        for m in range_constexpr(t.m_tiles):
            row = m * 16 + ctx.lane16
            output_row = ctx.query_begin * t.query_group_size + row
            partial_row = slot * (t.rows * t.query_splits) + output_row
            denom = final[m * state_stride + 1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            if row < t.rows:
                for vh in range_constexpr(vh_count):
                    d = vh * 64 + ctx.warp * 16 + ctx.half * 8
                    value = (
                        final[m * state_stride + 2 + vh]
                        * fx.Vector.from_elements(
                            [fx.Float32(inv * final[-1] if t.packed_lds else inv)],
                            dtype=fx.Float32,
                        ).broadcast_to(8)
                    ).to(t.q_dtype)
                    fx.ptr_store(
                        value,
                        fx.add_offset(
                            pout,
                            (
                                (
                                    fx.Int64(ctx.planned_seq) * t.query_length
                                    + output_row // t.query_group_size
                                )
                                * ctx.stride_q_row
                                + (
                                    ctx.kv_h * t.query_group_size
                                    + output_row % t.query_group_size
                                )
                                * ctx.stride_q_head
                                + d
                                if t.direct_output
                                else partial_row * t.head_dim + d
                            ),
                        ),
                    )
                if const_expr(not t.direct_output):  # noqa: SIM102
                    if ctx.warp == 0 and ctx.half == 0:
                        pmax[partial_row] = final[m * state_stride] * fx.Float32(
                            1.0 / LOG2E
                        )
                        psum[partial_row] = denom
