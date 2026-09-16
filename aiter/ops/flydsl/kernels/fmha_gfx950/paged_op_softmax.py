# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Paged causal masking, online softmax and lazy rescaling."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels.fmha_gfx950.common import _read_exec_i64
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
    DualwaveFp8KernelContext,
    _exp2_score_slice,
    _pack_p_v8_slices,
    _reduction_pair,
    _safe_l_inv,
    _scale_o_accs,
    _scale_sub_score_pair,
    _score_lists_to_vecs,
    _score_pair_max,
    _score_pair_sum,
    _score_pair_to_lists,
    _tree_reduce,
    _v_p_to_vec32,
    _v_vec32_to_p,
)


class DualwaveFp8SoftmaxHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def reduce_max_pair(self, v_s_a, v_s_b):
        values = [self.c_neg_inf]
        for scores in (*v_s_a, *v_s_b):
            values += [Vec(scores)[i] for i in range_constexpr(16)]
        local_max = _tree_reduce(values, fx.maxnumf)
        lhs, rhs = _reduction_pair(local_max)
        return fx.maxnumf(lhs, rhs)

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        # Vectorized-K sigma arranges wide FP8 scores in four-token groups
        # (0..3, 8..11, ...), matching dense FP8 but not vectorized BF16.
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        # q_row_i32 is set by init_q_row (called after helper construction), so read
        # it from the live ctx.
        rel_lo_i32 = fx.Int32(
            self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        for r in range_constexpr(16):
            threshold = (r // 4) * 8 + r % 4
            s_lo[r] = (rel_lo_i32 < threshold).select(self.c_neg_inf, s_lo[r])
            s_hi[r] = (rel_hi_i32 < threshold).select(self.c_neg_inf, s_hi[r])

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            kv_end_pos = self.traits.BLOCK_N

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos):
            s_lo, s_hi = v_s
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 <= fx.Int32(
                kv_end_pos - 1
            ):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def causal_mask_pair_if_needed(self, v_s_a, v_s_b, tile_a):
        """Causal-mask a BN128 tile pair under one scalar-uniform branch.

        Branching per sub-tile would put two scf.if regions in the body on every
        iteration, splitting it into five basic blocks and blocking the QK/softmax/PV
        interleave the sched_group_barriers ask for. Branching once on the pair's end
        position keeps every strictly-below-diagonal pair a single straight-line block.
        Masking sub-tile `a` inside the taken branch even when only `b` needs it is a
        no-op beyond the VALU compares, and happens in at most one pair per q-block.

        This replaces seq_pad_mask_if_needed: with delta = seqlen_kv - seqlen_q the
        largest key any row may attend to is seqlen_kv - 1, so every padding column is
        already strictly above the diagonal.
        """
        traits = self.traits
        kv_end_pos = (tile_a + fx.Index(2)) * traits.BLOCK_N

        @flyc.jit
        def _run(v_s_a, v_s_b, tile_a=tile_a, kv_end_pos=kv_end_pos):
            a_lo, a_hi = v_s_a
            b_lo, b_hi = v_s_b
            # The inclusive last key fits int32 even when the exclusive end is 2**31.
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 <= fx.Int32(
                kv_end_pos - 1
            ):
                a_l, a_h = self.v_s_vec_to_lists(v_s_a)
                self._causal_mask_inplace((a_l, a_h), tile_a)
                a_lo, a_hi = _score_lists_to_vecs((a_l, a_h))
                b_l, b_h = self.v_s_vec_to_lists(v_s_b)
                self._causal_mask_inplace((b_l, b_h), tile_a + fx.Index(1))
                b_lo, b_hi = _score_lists_to_vecs((b_l, b_h))
            return a_lo, a_hi, b_lo, b_hi

        a_lo, a_hi, b_lo, b_hi = _run(v_s_a, v_s_b)
        return (a_lo, a_hi), (b_lo, b_hi)

    def _seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s_lists
        col_base = fx.Int32(tile_idx * traits.BLOCK_N) + fx.Int32(
            self.lane_div_32
        ) * fx.Int32(4)
        for r in range_constexpr(16):
            thr = (r // 4) * 8 + (r % 4)
            col_lo = col_base + fx.Int32(thr)
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * self.traits.BLOCK_N
            if fx.Int32(kv_tile_end) > self.seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def reduce_max(self, v_s):
        return _score_pair_max(v_s, self.c_neg_inf, self.fm_fast)

    def max2(self, a, b):
        return fx.maxnumf(a, b)

    def floor_masked_max(self, row_max):
        return fx.maxnumf(row_max, self.c_neg_floor)

    # log2 of e4m3's largest finite value, 448.
    _P_HEADROOM_LOG2 = math.log2(448.0)

    def sub_m(self, v_s, row_max):
        # P is cast to e4m3, whose smallest subnormal is 2**-9, so a softmax
        # over thousands of keys loses its tail to flush-to-zero -- while l_row,
        # summed before the cast, still counts it. Scaling P up first uses the
        # format's whole range; l_row scales with it, so the output is unchanged
        # apart from the tail that survives. Free: it rides the FMA's addend.
        #
        # Available headroom is bounded by how large exp2 gets: the lazy path
        # holds the running max until a tile exceeds it by RESCALE_THRESHOLD, so
        # exp2 <= 2**THRESHOLD there; the eager path rebases every tile.
        headroom = self._P_HEADROOM_LOG2
        if const_expr(self.traits.DUALWAVE_SWP_LAZY_RESCALE):
            headroom -= self.traits.DUALWAVE_SWP_RESCALE_THRESHOLD
        bias = fx.Float32(headroom) if headroom > 0.0 else None
        return _scale_sub_score_pair(
            v_s,
            row_max,
            self.c_logit_scale,
            self.c_zero_f,
            self.fm_fast,
            bias,
            explicit_rounding=self.traits.PAGED,
        )

    def exp2(self, v_s, start, length):
        return _exp2_score_slice(v_s, start, length)

    def tile_sum(self, v_p):
        return _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def reduce_sum(self, l_row, v_p):
        return l_row + self.tile_sum(v_p)

    def cast_p(self, v_p):
        # Pack the finished softmax probabilities into v8 bf16 P packs for PV.
        return _pack_p_v8_slices(self.traits, v_p, self.bf16_trunc_pack_v8)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits, self.fm_fast)

    def scale_v_p(self, v_p, scale_scalar):
        # P is v8 bf16 (HIPREC): ext to f32, scale, repack bf16.
        p_lo, p_hi = v_p
        out_lo, out_hi = [], []
        for src, dst in ((p_lo, out_lo), (p_hi, out_hi)):
            for pk in src:
                f32 = Vec(
                    llvm.FPExtOp(
                        Vec.make_type(8, fx.Float32), as_mlir_value(pk)
                    ).result,
                    (8,),
                    fx.Float32,
                )
                scaled = [fx.Float32(f32[i]) * scale_scalar for i in range(8)]
                dst.append(self.bf16_trunc_pack_v8(scaled))
        return out_lo, out_hi

    def anchor_v_o(self, v_o):
        return self.preserve_accumulators(v_o)

    def anchor_scalar_f32(self, x):
        return llvm.intr_arithmetic_fence(fx.as_ir_value(x))

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        row_max = fx.maxnumf(m_row, m_tile_max)
        diff_scaled = (m_row - row_max) * self.c_logit_scale
        rescale = rocdl.exp2(T.f32, as_mlir_value(diff_scaled))
        return row_max, rescale

    def apply_l_rescale(self, l_row, rescale):
        return l_row * rescale

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        self.scale_o(v_o, corr)
        v_o = self.anchor_v_o(v_o)
        v_p = self.scale_v_p(v_p, corr)
        l_row = self.apply_l_rescale(l_row, corr)
        return v_o, m_new, l_row, v_p

    def v_p_to_vec32(self, v_p):
        # P packs are (p_lo[0..1], p_hi[0..1]) v8 bf16; concat into one v32 SSA value
        # for the scf.if loop-carry.
        return _v_p_to_vec32(v_p)

    def v_vec32_to_p(self, v_p_all):
        return _v_vec32_to_p(self.traits, v_p_all, elem_dtype=self.p_elem)

    def _lazy_correction(self, v_o, m_row, m_tile_max):
        """Monotonic: a downward rebase gives corr > 1 and repeated ones overflow."""
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        return (
            m_new,
            corr,
            [as_mlir_value(scaled_accs[i]) for i in range(self.traits.D_CHUNKS)],
        )

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max, v_p):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = (fx.Int64(ballot) == fx.Int64(_read_exec_i64())).ir_value()
            all_below = llvm.intr_expect(all_below, fx.Boolean(True).ir_value())

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            vp_out = self.v_p_to_vec32(v_p)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                vp_out = self.v_p_to_vec32(self.scale_v_p(v_p, corr))
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out, self.v_vec32_to_p(vp_out))

        return _run(v_o, m_row, l_row, m_tile_max, v_p)

    def lazy_correct_o(self, v_o, m_row, l_row, m_tile_max):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = (fx.Int64(ballot) == fx.Int64(_read_exec_i64())).ir_value()
            all_below = llvm.intr_expect(all_below, fx.Boolean(True).ir_value())

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out)

        return _run(v_o, m_row, l_row, m_tile_max)
