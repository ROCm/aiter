# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Paged causal masking, online softmax and lazy rescaling."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.fmha_gfx950.common import _read_exec_i64
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
    DualwaveFp8KernelContext,
    _exp2_score_slice,
    _reduction_pair,
    _safe_l_inv,
    _scale_o_accs,
    _scale_sub_score_pair,
    _score_lists_to_vecs,
    _score_pair_sum,
    _score_pair_to_lists,
    _tree_reduce,
)


class DualwaveFp8SoftmaxHelper(DualwaveFp8KernelContext):
    def reduce_max_pair(self, v_s_a, v_s_b):
        values = [self.c_neg_inf]
        for scores in (*v_s_a, *v_s_b):
            values += [Vec(scores)[i] for i in range_constexpr(16)]
        local_max = _tree_reduce(values, fx.maxnumf)
        lhs, rhs = _reduction_pair(local_max)
        return fx.maxnumf(lhs, rhs)

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

    def causal_mask_pair_if_needed(self, v_s_a, v_s_b, tile_a):
        """Mask both halves with one branch to preserve QK/softmax/PV scheduling.

        Bottom-right causal masking also excludes padded KV positions. Keep
        the inclusive-last-key comparison below to support the int32 limit.
        """
        traits = self.traits
        kv_end_pos = (tile_a + 2) * traits.BLOCK_N

        @flyc.jit
        def _run(v_s_a, v_s_b, tile_a=tile_a, kv_end_pos=kv_end_pos):
            a_lo, a_hi = v_s_a
            b_lo, b_hi = v_s_b
            # The inclusive last key fits int32 even when the exclusive end is 2**31.
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 <= fx.Int32(
                kv_end_pos - 1
            ):
                a_l, a_h = _score_pair_to_lists(v_s_a)
                self._causal_mask_inplace((a_l, a_h), tile_a)
                a_lo, a_hi = _score_lists_to_vecs((a_l, a_h))
                b_l, b_h = _score_pair_to_lists(v_s_b)
                self._causal_mask_inplace((b_l, b_h), tile_a + 1)
                b_lo, b_hi = _score_lists_to_vecs((b_l, b_h))
            return a_lo, a_hi, b_lo, b_hi

        a_lo, a_hi, b_lo, b_hi = _run(v_s_a, v_s_b)
        return (a_lo, a_hi), (b_lo, b_hi)

    def floor_masked_max(self, row_max):
        return fx.maxnumf(row_max, self.c_neg_floor)

    def sub_m(self, v_s, row_max):
        # Use E4M3's exponent headroom to retain small probabilities before
        # the FP8 cast. The normalization sum carries the same factor; LSE
        # removes it. Lazy rescaling reserves room for RESCALE_THRESHOLD.
        headroom = self.traits.P_HEADROOM_LOG2
        bias = fx.Float32(headroom) if headroom > 0.0 else None
        return _scale_sub_score_pair(
            v_s,
            row_max,
            self.c_logit_scale,
            self.c_zero_f,
            bias,
        )

    def exp2(self, v_s, start):
        return _exp2_score_slice(v_s, start)

    def tile_sum(self, v_p):
        return _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def reduce_sum(self, l_row, v_p):
        return l_row + self.tile_sum(v_p)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits)

    def anchor_v_o(self, v_o):
        return self.preserve_accumulators(v_o)

    def anchor_scalar_f32(self, x):
        return llvm.intr_arithmetic_fence(fx.as_ir_value(x))

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        m_row = fx.Float32(m_row)
        row_max = fx.maxnumf(m_row, m_tile_max)
        diff_scaled = (m_row - row_max) * self.c_logit_scale
        rescale = fx.exp2(diff_scaled, fastmath="afn").ir_value()
        return row_max, rescale

    def apply_l_rescale(self, l_row, rescale):
        return l_row * rescale

    def _lazy_correction(self, v_o, m_row, m_tile_max):
        """Monotonic: a downward rebase gives corr > 1 and repeated ones overflow."""
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        return (
            m_new,
            corr,
            [fx.as_ir_value(scaled_accs[i]) for i in range(self.traits.D_CHUNKS)],
        )

    def lazy_correct_o(self, v_o, m_row, l_row, m_tile_max):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, fx.as_ir_value(below))
            all_below = (fx.Int64(ballot) == fx.Int64(_read_exec_i64())).ir_value()
            all_below = llvm.intr_expect(all_below, fx.Boolean(True).ir_value())

            o_out = [
                fx.as_ir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = fx.as_ir_value(m_row)
            l_out = fx.as_ir_value(l_row)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    fx.as_ir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                l_out = fx.as_ir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out)

        return _run(v_o, m_row, l_row, m_tile_max)
