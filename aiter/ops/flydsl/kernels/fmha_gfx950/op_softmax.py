# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Online softmax, with causal / seq-pad masking."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import as_ir_value as as_mlir_value
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.fmha_gfx950.pipeline import (
    DualwaveFp8KernelContext,
    _anchor_scalar_f32,
    _anchor_v_o,
    _apply_dualwave_causal_mask_pair,
    _attn_mask_vec2_imm,
    _exp2_score_slice,
    _lane_pair_reduce,
    _p_headroom_log2,
    _read_exec_i64,
    _reduce_score_pair,
    _scale_o_accs,
    _scale_sub_score_pair,
    _score_lists_to_vecs,
    _score_pair_max,
    _score_pair_sum,
    _score_pair_to_lists,
)


class DualwaveFp8SoftmaxHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _attn_mask_vec2_imm(
        self, rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
    ):
        return _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
        )

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        lane_off, pair_thresholds = self.score_mask_columns()
        lane_off_i32 = fx.Int32(self.lane_div_32) * lane_off
        # q_row_i32 is set by init_q_row (called after helper construction), so read
        # it from the live ctx.
        rel_lo_i32 = fx.Int32(
            self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi_i32 = rel_lo_i32 - 32
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
        _apply_dualwave_causal_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds)
        _apply_dualwave_causal_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds)

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
        kv_end_pos = (tile_a + 2) * traits.BLOCK_N

        @flyc.jit
        def _run(v_s_a, v_s_b, tile_a=tile_a, kv_end_pos=kv_end_pos):
            a_lo, a_hi = v_s_a
            b_lo, b_hi = v_s_b
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                a_l, a_h = self.v_s_vec_to_lists(v_s_a)
                self._causal_mask_inplace((a_l, a_h), tile_a)
                a_lo, a_hi = _score_lists_to_vecs((a_l, a_h))
                b_l, b_h = self.v_s_vec_to_lists(v_s_b)
                self._causal_mask_inplace((b_l, b_h), tile_a + 1)
                b_lo, b_hi = _score_lists_to_vecs((b_l, b_h))
            return a_lo, a_hi, b_lo, b_hi

        a_lo, a_hi, b_lo, b_hi = _run(v_s_a, v_s_b)
        return (a_lo, a_hi), (b_lo, b_hi)

    def causal_mask_tile_if_needed(self, v_s, tile_idx):
        @flyc.jit
        def _run(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_end_pos = (tile_idx + 1) * self.traits.BLOCK_N
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s, tile_idx)

    def causal_mask_diag(self, v_s, tile_idx, hi_vis):
        """causal_mask_tile_if_needed for a tile whose high strip may be skipped: without
        hi_vis the high strip already holds the floor, whose P is exactly 0."""
        traits = self.traits

        @flyc.jit
        def _run(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_end_pos = (tile_idx + 1) * traits.BLOCK_N
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                kv_start_i32 = fx.Int32(tile_idx * traits.BLOCK_N)
                lane_off, thresholds = self.score_mask_columns()
                rel_lo_i32 = fx.Int32(
                    self.ctx_ref.q_row_i32
                    + self.delta_i32
                    - kv_start_i32
                    - fx.Int32(self.lane_div_32) * lane_off
                )
                neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
                _apply_dualwave_causal_mask_pair(
                    lo_list, rel_lo_i32, neg_inf_i32, thresholds
                )
                s_lo = Vec.from_elements(lo_list, fx.Float32).ir_value()
                if hi_vis:
                    _apply_dualwave_causal_mask_pair(
                        hi_list, rel_lo_i32 - 32, neg_inf_i32, thresholds
                    )
                    s_hi = Vec.from_elements(hi_list, fx.Float32).ir_value()
            return s_lo, s_hi

        return _run(v_s, tile_idx)

    def _seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s_lists
        kv_tile_start = tile_idx * traits.BLOCK_N
        lane_off, pair_thresholds = self.score_mask_columns()
        col_base = fx.Int32(kv_tile_start) + fx.Int32(self.lane_div_32) * lane_off
        for r in range_constexpr(16):
            thr = pair_thresholds[r // 2][r % 2]
            col_lo = col_base + fx.Int32(thr)
            col_hi = col_lo + 32
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx=None):
        if tile_idx is None:
            tile_idx = 0

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + 1) * self.traits.BLOCK_N
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
        headroom = _p_headroom_log2(self.traits)
        bias = fx.Float32(headroom) if headroom > 0.0 else None
        return _scale_sub_score_pair(
            v_s, row_max, self.c_logit_scale, self.c_zero_f, self.fm_fast, bias
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
        return _anchor_v_o(self.traits, v_o)

    def anchor_scalar_f32(self, x):
        return _anchor_scalar_f32(x)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        row_max = fx.maxnumf(m_row, m_tile_max)
        diff_scaled = (m_row - row_max) * self.c_logit_scale
        rescale = fx.math.exp2(fx.Float32(diff_scaled), fastmath="afn").ir_value()
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
            [as_mlir_value(scaled_accs[i]) for i in range(self.traits.D_CHUNKS)],
        )

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

    # conventional_bn64 softmax. The loop carries nsm = bias - scale * m instead of the row max m, and l as this lane's
    # half-row sum (both lane halves of a row rescale by the same factor, so the halves meet once in the epilogue).
    # The default body and _store_lse keep the m-carry above.

    def nsm_init(self):
        """Huge finite offset (m at the floor): the first tile always rebases, a keyless row keeps P = exp2(-inf) = 0,
        and no -inf/inf pair meets in the FMA."""
        return fx.Float32(3.0e38)

    def half_row_max(self, v_s):
        """This lane's half-row max: "every lane within the threshold" is the same test as with the full row max."""
        reducer = lambda a, b, _fm: fx.maxnumf(a, b)
        return fx.maxnumf(
            _reduce_score_pair(v_s, self.c_neg_inf, reducer, self.fm_fast),
            self.c_neg_floor,
        )

    def _rebase_nsm(self, v_o, nsm, l_row, m_half):
        """Rebase to the whole row's max, so both lane halves of a row agree on the offset. Monotonic (nsm only falls):
        corr <= 1."""
        m_row_tile = _lane_pair_reduce(m_half, lambda a, b, _fm: fx.maxnumf(a, b), None)
        bias = fx.Float32(_p_headroom_log2(self.traits))
        nsm_new = fx.minnumf(
            fx.Float32(nsm), bias - fx.Float32(m_row_tile) * self.c_logit_scale
        )
        corr = fx.math.exp2(fx.Float32(nsm_new - nsm), fastmath="afn").ir_value()
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        return (
            [
                as_mlir_value(scaled_accs[dc])
                for dc in range_constexpr(self.traits.D_CHUNKS)
            ],
            nsm_new,
            l_row * corr,
        )

    def rebase_nsm(self, v_o, nsm, l_row, m_half):
        """Eager rescale (no lazy threshold): rebase every tile."""
        return self._rebase_nsm(v_o, nsm, l_row, m_half)

    def lazy_correct_o_nsm(self, v_o, nsm, l_row, m_half):
        """lazy_correct_o on the nsm carry: the test is one FMA, scale * m_tile + nsm <= thr + bias."""
        limit = fx.Float32(
            self.traits.DUALWAVE_SWP_RESCALE_THRESHOLD + _p_headroom_log2(self.traits)
        )

        @flyc.jit
        def _run(v_o, nsm, l_row, m_half):
            t = fx.Float32(m_half) * self.c_logit_scale + nsm
            # Finite clamped max and finite offset: t is never NaN, so "no lane above" reads straight off VCC.
            above = t > limit
            ballot = rocdl.ballot(T.i64, as_mlir_value(above))
            all_below = (fx.Int64(ballot) == fx.Int64(0)).ir_value()
            all_below = llvm.intr_expect(all_below, fx.Boolean(True).ir_value())

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            n_out = as_mlir_value(nsm)
            l_out = as_mlir_value(l_row)
            if fx.Boolean(all_below):
                pass
            else:
                scaled, nsm_new, l_new = self._rebase_nsm(v_o, nsm, l_row, m_half)
                o_out = [
                    as_mlir_value(scaled[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                l_out = as_mlir_value(l_new)
                n_out = self.anchor_scalar_f32(nsm_new)
            return (o_out, n_out, l_out)

        return _run(v_o, nsm, l_row, m_half)

    def sub_nsm(self, v_s, nsm, scalar):
        """scale * s + nsm. scalar: one v_fma_f32 per score (SLP is off for this variant), which co-issues under the
        MFMAs where a packed FMA waits for the matrix pipe."""
        s_lo, s_hi = v_s
        if const_expr(not scalar):
            scale_v = fx.full(16, fx.Float32(self.c_logit_scale), fx.Float32)
            nsm_v = fx.full(16, fx.Float32(nsm), fx.Float32)
            lo = fx.fma(Vec(s_lo), scale_v, nsm_v, fastmath=self.fm_fast)
            hi = fx.fma(Vec(s_hi), scale_v, nsm_v, fastmath=self.fm_fast)
            return as_mlir_value(lo), as_mlir_value(hi)
        out = []
        for vec in (s_lo, s_hi):
            vals = [
                fx.fma(
                    fx.Float32(Vec(vec)[r]),
                    fx.Float32(self.c_logit_scale),
                    fx.Float32(nsm),
                )
                for r in range_constexpr(16)
            ]
            out.append(Vec.from_elements(vals, fx.Float32).ir_value())
        return out[0], out[1]

    def half_row_sum(self, l_row, v_p, scalar):
        """l_row plus this lane's sum of P. scalar: an explicit pairwise tree of v_add_f32 (SLP off). The anchor keeps
        the sum, and with it P's fp32 copy, ahead of the V reads; without the per-tile lane swap pinning it, shuffled
        spills 8-11 VGPRs."""
        p_lo, p_hi = _score_lists_to_vecs(v_p)
        if const_expr(not scalar):
            tile = (Vec(p_lo) + Vec(p_hi)).reduce(
                "add", init_val=self.c_zero_f, fastmath=self.fm_fast
            )
            return _anchor_scalar_f32(l_row + tile)
        cur = [fx.Float32(Vec(p_lo)[i]) + fx.Float32(Vec(p_hi)[i]) for i in range(16)]
        while len(cur) > 1:
            h = len(cur) // 2
            cur = [cur[i] + cur[i + h] for i in range(h)]
        return _anchor_scalar_f32(l_row + cur[0])

    def finish_half_row_sum(self, l_row):
        return _lane_pair_reduce(l_row, lambda a, b, _fm: a + b, None)
