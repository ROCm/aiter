# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""PA score masking, online softmax, and BF16/FP8 probability staging."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import ReductionOp, T

from ..utils import rcp_f32
from .traits import MFMA_ACC_ELEMS, MFMA_MNK, WAVE


class PaDecodeSoftmax:
    def __init__(self, ctx, lds, kv, gemm):
        self.ctx = ctx
        self.traits = ctx.traits
        self.lds = lds
        self.kv = kv
        self.gemm = gemm

    def init_mask(self):
        self._ct = [
            fx.Vector.from_elements(
                [float(a * MFMA_MNK + r) for r in range_constexpr(4)]
            )
            for a in range_constexpr(self.traits.NCHUNK)
        ]

    def score_mask(self, a, upper, lower):
        valid = self._ct[a] < upper
        if const_expr(self.traits.sliding_window > 0):
            valid = valid & (self._ct[a] >= lower)
        return valid

    def scale_mask(self, tile_valid, window_left):
        ctx_thr = fx.full(
            4,
            tile_valid.to(fx.Float32)
            - fx.Int32(
                self.ctx.warp * self.traits.TOK_PER_WARP + self.ctx.rgroup * 4
            ).to(fx.Float32),
            fx.Float32,
        )
        window_scale_thr = None
        if const_expr(self.traits.sliding_window > 0):
            # Only the MTP window union may set the shared FP8 scale;
            # invisible large scales could underflow visible probabilities.
            first_visible = (window_left - (self.traits.query_length - 1)).to(
                fx.Float32
            )
            window_scale_thr = fx.full(
                4,
                first_visible
                - fx.Int32(
                    self.ctx.warp * self.traits.TOK_PER_WARP + self.ctx.rgroup * 4
                ).to(fx.Float32),
                fx.Float32,
            )
        zero4_scale = fx.full(4, 0.0, fx.Float32)
        return ctx_thr, window_scale_thr, zero4_scale

    def mask_v_scale(self, vec, a, bounds):
        upper, lower, zero = bounds
        return self.score_mask(a, upper, lower).select(vec, zero)

    def lmax_offset(self, m):
        return (
            self.traits.sLmax_off
            + m * MFMA_MNK * self.traits.NWARP_PAD * self.traits.f32
        )

    def lsum_offset(self, m):
        return (
            self.traits.sLsum_off
            + m * MFMA_MNK * self.traits.NWARP_PAD * self.traits.f32
        )

    def p_offset(self, m):
        return self.traits.sP_off + m * MFMA_MNK * self.traits.SP_ROW_BYTES

    def row_thresholds(self, tile_valid, window_left, m):
        n_valid_tile = (tile_valid - self.ctx.causal_offset[m]).to(fx.Float32)
        base_tok_f = fx.Int32(
            self.ctx.warp * self.traits.TOK_PER_WARP + self.ctx.rgroup * 4
        ).to(fx.Float32)
        thr = fx.full(4, n_valid_tile - base_tok_f, fx.Float32)
        window_thr = None
        if const_expr(self.traits.sliding_window > 0):
            # Subtract before converting to avoid rounding large edges.
            first_valid_tile = (window_left - self.ctx.causal_offset[m]).to(fx.Float32)
            window_thr = fx.full(4, first_valid_tile - base_tok_f, fx.Float32)
        return thr, window_thr

    @flyc.jit
    def multi_scores(self, frag_Ss, scale, k_scale_shared, tile_valid, window_left, m):
        thr, window_thr = self.row_thresholds(tile_valid, window_left, m)
        neg4 = fx.full(4, float("-inf"), fx.Float32)

        # Scale finite logits before selecting -inf to avoid 0 * -inf
        # for zero-Q rows. The stored row max is already scaled.
        scale_b = fx.full(4, scale, fx.Float32)
        if const_expr(self.traits.FP8_MTP4_LEAN):
            # The positive row scale applies in pack_fused_lean's exp2.
            scaled_frags = [
                frag_Ss[a] * k_scale_shared[a]
                for a in range_constexpr(self.traits.NCHUNK)
            ]
        elif const_expr(self.traits.per_token_kv):
            scaled_frags = [
                frag_Ss[a] * k_scale_shared[a] * scale_b
                for a in range_constexpr(self.traits.NCHUNK)
            ]
        else:
            scaled_frags = [
                frag_Ss[a] * scale_b for a in range_constexpr(self.traits.NCHUNK)
            ]

        if const_expr(self.traits.MTP4_FUSED):
            # Interior tiles need no mask for any of the four queries.
            masked_all = fx.Vector.from_elements(
                [
                    scaled_frags[a][r]
                    for a in range_constexpr(self.traits.NCHUNK)
                    for r in range_constexpr(MFMA_ACC_ELEMS)
                ],
                dtype=fx.Float32,
            )
            needs_mask = (
                tile_valid < self.traits.TILE_TOK + self.traits.query_length - 1
            )
            if const_expr(self.traits.sliding_window > 0):
                needs_mask = needs_mask | (window_left > 0)
            if needs_mask:
                masked_all = fx.Vector.from_elements(
                    [
                        self.score_mask(a, thr, window_thr).select(
                            scaled_frags[a], neg4
                        )[r]
                        for a in range_constexpr(self.traits.NCHUNK)
                        for r in range_constexpr(MFMA_ACC_ELEMS)
                    ],
                    dtype=fx.Float32,
                )
            masked_chunks = [
                fx.Vector.from_elements(
                    [
                        masked_all[a * MFMA_ACC_ELEMS + r]
                        for r in range_constexpr(MFMA_ACC_ELEMS)
                    ],
                    dtype=fx.Float32,
                )
                for a in range_constexpr(self.traits.NCHUNK)
            ]
        else:
            masked_chunks = [
                self.score_mask(a, thr, window_thr).select(scaled_frags[a], neg4)
                for a in range_constexpr(self.traits.NCHUNK)
            ]
        return masked_chunks

    def stage_row_max(self, masked_chunks, base, scale, scaled):
        pm = fx.Float32(float("-inf"))
        for a in range_constexpr(self.traits.NCHUNK):
            if const_expr(scaled):
                pm = fx.maxnumf(
                    pm,
                    masked_chunks[a].reduce(ReductionOp.MAX, fastmath=self.ctx.fm_nnan),
                    fastmath=self.ctx.fm_nnan,
                )
            else:
                pm = fx.maxnumf(pm, masked_chunks[a].reduce(ReductionOp.MAX))
        for sh in (16, 32):
            if const_expr(scaled):
                pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE), fastmath=self.ctx.fm_nnan)
            else:
                pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE))
        self.lds.store_wave(
            base,
            self.ctx.lane16,
            self.ctx.warp,
            pm if const_expr(scaled) else pm * scale,
        )  # redundant across the 4 lanes sharing this qhead

    def stage_v_scale_max(self, v_scales, bounds):
        pv_max = fx.Float32(0.0)
        for a in range_constexpr(self.traits.NCHUNK):
            pv_max = fx.maxnumf(
                pv_max,
                self.mask_v_scale(v_scales[a], a, bounds).reduce(ReductionOp.MAX),
            )
        for sh in (16, 32):
            pv_max = fx.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
        self.lds.store_wave(self.traits.sVScaleMax_off, 0, self.ctx.warp, pv_max)

    def v_scale_normalization(self):
        parts = self.lds.load_wave(self.traits.sVScaleMax_off, 0)
        if const_expr(self.traits.FP8_MTP4_LEAN):
            v_max_global = self.fused_max(
                [parts[i] for i in range_constexpr(self.traits.NWARP)]
            )
        else:
            v_max_global = parts.reduce(ReductionOp.MAX)
        v_max_scaled = v_max_global * fx.Float32(1.0 / self.traits.FP8_MAX)
        v_max_safe = v_max_scaled + fx.Float32(1e-8 / self.traits.FP8_MAX)
        norm_factor = fx.Float32(rcp_f32(v_max_safe))
        norm_factor_b = fx.full(4, norm_factor, fx.Float32)
        return v_max_scaled, norm_factor_b

    @flyc.jit
    def pack_fused_lean(self, masked_chunks_saved, ostate, cur_kv_buf, norm_b, scales):
        """pack_fused for unscaled scores: exp2(s * scale - max) is one FMA,
        v_scale * norm is shared by the M-tiles and Lsum adds packed pairs."""
        t = self.traits
        m_new_saved = []
        scale_saved = []
        safe_saved = []
        for m in range_constexpr(t.M_TILES):
            parts = self.lds.load_wave(self.lmax_offset(m), self.ctx.lane16)
            tile_max = self.fused_max([parts[i] for i in range_constexpr(t.NWARP)])
            m_new = fx.maxnumf(
                ostate[t.m_slot(m)], tile_max * scales[m], fastmath=self.ctx.fm_nnan
            )
            m_new_saved.append(m_new)
            # Empty rows need exponent reference 0 to avoid -inf-(-inf).
            safe_max = (m_new > self.ctx.NEG_INF).select(m_new, self.ctx.ZERO_F)
            safe_saved.append(fx.full(4, fx.Float32(safe_max), fx.Float32))
            scale_saved.append(fx.full(4, scales[m], fx.Float32))
        ls_saved = [None] * t.M_TILES
        for a in range_constexpr(t.NCHUNK):
            v_sc = self.kv.load_scale(t.sVScale_off, a, cur_kv_buf)
            vn = v_sc * norm_b
            for m in range_constexpr(t.M_TILES):
                Pa = fx.Vector(
                    fx.exp2(
                        masked_chunks_saved[m][a] * scale_saved[m] - safe_saved[m],
                        fastmath="fast",
                    )
                )
                ls_saved[m] = Pa if const_expr(a == 0) else ls_saved[m] + Pa
                word = self.gemm.fp8_words(Pa * vn)[0]
                p_off = (
                    t.sP_off
                    + m * MFMA_MNK * t.SP_ROW_BYTES
                    + self.ctx.lane16 * t.SP_ROW_BYTES
                    + self.ctx.warp * t.TOK_PER_WARP
                    + self.ctx.rgroup * 4
                    + a * (MFMA_MNK // 4) * t.f32
                )
                self.lds.store(
                    p_off, fx.Int32, fx.Vector.from_elements([word], dtype=fx.Int32)
                )
        ls = self.rgroup_fold4(
            [
                self.fused_sum(
                    [ls_saved[m][r] for r in range_constexpr(MFMA_ACC_ELEMS)]
                )
                for m in range_constexpr(t.M_TILES)
            ],
            lambda a, b: a + b,
        )
        self.lds.store_wave(
            t.sLsum_off + self.rgroup_fold_m() * (MFMA_MNK * t.NWARP_PAD * t.f32),
            self.ctx.lane16,
            self.ctx.warp,
            ls,
        )
        return m_new_saved

    def single_scores(
        self,
        frag_Ss,
        scale,
        tile_valid,
        window_left,
        cur_kv_buf,
        bounds,
        m,
        masked=True,
    ):
        """``masked=False``: the tile holds no token past any row's bound, so
        the score mask is skipped (single-query decode, interior tiles)."""
        thr, window_thr = self.row_thresholds(tile_valid, window_left, m)
        neg4 = fx.full(
            4,
            float("-inf") if self.traits.M1_SCALE_BEFORE_MASK else -1e30,
            fx.Float32,
        )
        # As in Phase A, scale before -inf masking to avoid zero-Q NaNs.
        scale_b = None
        if const_expr(self.traits.M1_SCALE_BEFORE_MASK):
            scale_b = fx.full(4, scale, fx.Float32)
        v_scale_vecs = None
        if const_expr(self.traits.per_token_kv):
            v_scale_vecs = []
            scaled_frags = []
            masked_chunks = []
            for a in range_constexpr(self.traits.NCHUNK):
                k_scale_vec, v_scale_vec = self.kv.load_kv_scales(a, cur_kv_buf)
                v_scale_vecs.append(v_scale_vec)
                scaled_frag = frag_Ss[a] * k_scale_vec
                if const_expr(self.traits.M1_SCALE_BEFORE_MASK):
                    scaled_frag = scaled_frag * scale_b
                    masked_chunks.append(
                        self.score_mask(a, thr, window_thr).select(scaled_frag, neg4)
                        if const_expr(masked)
                        else scaled_frag
                    )
                else:
                    scaled_frags.append(scaled_frag)
        else:
            if const_expr(self.traits.M1_SCALE_BEFORE_MASK):
                masked_chunks = [
                    (
                        self.score_mask(a, thr, window_thr).select(
                            frag_Ss[a] * scale_b, neg4
                        )
                        if const_expr(masked)
                        else frag_Ss[a] * scale_b
                    )
                    for a in range_constexpr(self.traits.NCHUNK)
                ]
            else:
                scaled_frags = frag_Ss
        if const_expr(not self.traits.M1_SCALE_BEFORE_MASK):
            masked_chunks = [
                (
                    self.score_mask(a, thr, window_thr).select(scaled_frags[a], neg4)
                    if const_expr(masked)
                    else scaled_frags[a]
                )
                for a in range_constexpr(self.traits.NCHUNK)
            ]
        # pass 1: per-warp max for this qhead
        self.stage_row_max(
            masked_chunks,
            self.lmax_offset(m),
            scale,
            self.traits.M1_SCALE_BEFORE_MASK,
        )
        if const_expr(self.traits.per_token_kv):
            self.stage_v_scale_max(v_scale_vecs, bounds)
        return masked_chunks, v_scale_vecs

    @flyc.jit
    def pack_fused(self, masked_chunks_saved, ostate, cur_kv_buf, norm_factor_b):
        # Private P/Lsum slots share one barrier without overwrite races.
        m_new_saved = []
        safe_max_saved = []
        for m in range_constexpr(self.traits.M_TILES):
            tile_max = self.lds.load_wave(self.lmax_offset(m), self.ctx.lane16).reduce(
                ReductionOp.MAX, fastmath=self.ctx.fm_nnan
            )
            m_new = (
                tile_max
                if const_expr(self.traits.single_tile_plan)
                else fx.maxnumf(
                    ostate[self.traits.m_slot(m)], tile_max, fastmath=self.ctx.fm_nnan
                )
            )
            m_new_saved.append(m_new)
            safe_max = (m_new > self.ctx.NEG_INF).select(m_new, self.ctx.ZERO_F)
            safe_max_saved.append(fx.full(4, fx.Float32(safe_max), fx.Float32))
        ls_saved = [self.ctx.ZERO_F for _ in range_constexpr(self.traits.M_TILES)]
        # Share one scale fragment across queries, not the full scale tile.
        for a in range_constexpr(self.traits.NCHUNK):
            v_sc = self.kv.load_scale(self.traits.sVScale_off, a, cur_kv_buf)
            for m in range_constexpr(self.traits.M_TILES):
                Pa = fx.Vector(
                    fx.exp2(
                        masked_chunks_saved[m][a] - safe_max_saved[m],
                        fastmath="fast",
                    )
                )
                ls_saved[m] = ls_saved[m] + Pa.reduce(ReductionOp.ADD)
                p_scaled = Pa * v_sc * norm_factor_b
                word = self.gemm.fp8_words(p_scaled)[0]
                p_off = (
                    self.traits.sP_off
                    + m * MFMA_MNK * self.traits.SP_ROW_BYTES
                    + self.ctx.lane16 * self.traits.SP_ROW_BYTES
                    + self.ctx.warp * self.traits.TOK_PER_WARP
                    + self.ctx.rgroup * 4
                    + a * (MFMA_MNK // 4) * self.traits.f32
                )
                self.lds.store(
                    p_off,
                    fx.Int32,
                    fx.Vector.from_elements([word], dtype=fx.Int32),
                )
        for m in range_constexpr(self.traits.M_TILES):
            ls = ls_saved[m]
            for sh in (16, 32):
                ls = ls + ls.shuffle_xor(sh, WAVE)
            if self.ctx.rgroup == 0:
                self.lds.store_wave(
                    self.traits.sLsum_off
                    + m * MFMA_MNK * self.traits.NWARP_PAD * self.traits.f32,
                    self.ctx.lane16,
                    self.ctx.warp,
                    ls,
                )
        return m_new_saved

    # ---- Fused BF16 multi-query tiles (traits.BF16_MTP_FUSED) ----
    # Scores stay unscaled until exp2(s * scale - max): a positive scale
    # commutes with max, and the multiply folds into the exponent's FMA.

    @staticmethod
    def _tree(vals, op):
        # Fixed pairwise order: short dependency chains, deterministic sums.
        while len(vals) > 1:
            nxt = [op(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    def fused_max(self, vals):
        # maxnum lowers to v_max/v_max3; vector.reduce(MAX) to cmp/select.
        return self._tree(
            vals, lambda a, b: fx.maxnumf(a, b, fastmath=self.ctx.fm_nnan)
        )

    def fused_sum(self, vals):
        return self._tree(vals, lambda a, b: a + b)

    def _lane_swap(self, x, y, span):
        """v_permlane{16,32}_swap of f32 ``x``, ``y``: afterwards each lane
        holds, across the pair, its own value and the one ``span`` lanes away."""
        swap = fx.rocdl.permlane32_swap if span == 32 else fx.rocdl.permlane16_swap
        pair = swap(
            llvm_d.StructType.get_literal([T.i32, T.i32]),
            fx.as_ir_value(x.bitcast(fx.Int32)),
            fx.as_ir_value(y.bitcast(fx.Int32)),
            False,
            False,
        )
        return [
            fx.Int32(llvm_d.extractvalue(T.i32, pair, [i])).bitcast(fx.Float32)
            for i in (0, 1)
        ]

    def rgroup_fold4(self, vals, op):
        """Fold four per-lane values over the four rgroups in three swaps.

        rgroup r ends with the fold of ``vals[self.rgroup_fold_m()]``.
        """
        a = self._lane_swap(vals[0], vals[1], 32)
        b = self._lane_swap(vals[2], vals[3], 32)
        c = self._lane_swap(op(a[0], a[1]), op(b[0], b[1]), 16)
        return op(c[0], c[1])

    def rgroup_fold_m(self):
        return (self.ctx.rgroup & 1) * 2 + (self.ctx.rgroup >> 1)

    def fused_max2(self, a, b):
        return fx.maxnumf(a, b, fastmath=self.ctx.fm_nnan)

    def lane_row_max(self, chunks):
        """This lane's max over one M-tile's raw scores."""
        return self.fused_max(
            [
                chunks[a][r]
                for a in range_constexpr(self.traits.NCHUNK)
                for r in range_constexpr(MFMA_ACC_ELEMS)
            ]
        )

    def stage_max4(self, lane_maxes):
        """Publish four M-tiles' row maxima, one M-tile per rgroup."""
        pm = self.rgroup_fold4(lane_maxes, self.fused_max2)
        self.lds.store_wave(
            self.traits.sLmax_off
            + self.rgroup_fold_m()
            * (MFMA_MNK * self.traits.NWARP_PAD * self.traits.f32),
            self.ctx.lane16,
            self.ctx.warp,
            pm,
        )

    def _fused_mask_all(self, frags, tile_valid):
        neg4 = fx.full(4, float("-inf"), fx.Float32)
        out = []
        for m in range_constexpr(self.traits.M_TILES):
            thr, _ = self.row_thresholds(tile_valid, None, m)
            for a in range_constexpr(self.traits.NCHUNK):
                masked = self.score_mask(a, thr, None).select(frags[m][a], neg4)
                out.extend([masked[r] for r in range_constexpr(MFMA_ACC_ELEMS)])
        return out

    @flyc.jit
    def fused_scores(self, frags, tile_valid):
        """Raw scores, -inf past each row's causal bound.

        With TILE_TOK + query_length - 1 or more valid tokens every row sees
        the whole tile, so one uniform branch skips the selects on all such
        tiles, including most tasks' last tile.
        """
        t = self.traits
        flat = fx.Vector.from_elements(
            [
                frags[m][a][r]
                for m in range_constexpr(t.M_TILES)
                for a in range_constexpr(t.NCHUNK)
                for r in range_constexpr(MFMA_ACC_ELEMS)
            ],
            dtype=fx.Float32,
        )
        if tile_valid < t.TILE_TOK + t.query_length - 1:
            flat = fx.Vector.from_elements(
                self._fused_mask_all(frags, tile_valid), dtype=fx.Float32
            )
        per_m = t.NCHUNK * MFMA_ACC_ELEMS
        return [
            [
                fx.Vector.from_elements(
                    [
                        flat[m * per_m + a * MFMA_ACC_ELEMS + r]
                        for r in range_constexpr(MFMA_ACC_ELEMS)
                    ],
                    dtype=fx.Float32,
                )
                for a in range_constexpr(t.NCHUNK)
            ]
            for m in range_constexpr(t.M_TILES)
        ]

    def fused_stage_max(self, chunks, m):
        pm = self.fused_max(
            [
                chunks[a][r]
                for a in range_constexpr(self.traits.NCHUNK)
                for r in range_constexpr(MFMA_ACC_ELEMS)
            ]
        )
        for sh in (16, 32):
            pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE), fastmath=self.ctx.fm_nnan)
        # Raw max; redundant across the 4 lanes sharing this row.
        self.lds.store_wave(self.lmax_offset(m), self.ctx.lane16, self.ctx.warp, pm)

    @flyc.jit
    def pack_fused_bf16(self, masked_chunks, m, m_prev):
        """BF16 P and Lsum of M-tile ``m`` into its private slots.

        Returns the M-tile's new max and the correction of its carried
        output/denominator.
        """
        scale_b = fx.full(4, self.ctx.scale_qk, fx.Float32)
        parts = self.lds.load_wave(self.lmax_offset(m), self.ctx.lane16)
        tile_max = self.fused_max(
            [parts[i] for i in range_constexpr(self.traits.NWARP)]
        )
        m_new = fx.maxnumf(
            m_prev, tile_max * self.ctx.scale_qk, fastmath=self.ctx.fm_nnan
        )
        # Empty rows need exponent reference 0 to avoid -inf-(-inf).
        safe_max = (m_new > self.ctx.NEG_INF).select(m_new, self.ctx.ZERO_F)
        safe_b = fx.full(4, fx.Float32(safe_max), fx.Float32)
        p_off0 = (
            self.p_offset(m)
            + self.ctx.lane16 * self.traits.SP_ROW_BYTES
            + (self.ctx.warp * self.traits.TOK_PER_WARP + self.ctx.rgroup * 4)
            * self.traits.QP_ELEM_BYTES
        )
        p_vals = []
        for a in range_constexpr(self.traits.NCHUNK):
            Pa = fx.Vector(
                fx.exp2(masked_chunks[a] * scale_b - safe_b, fastmath="fast")
            )
            p_vals.extend([Pa[r] for r in range_constexpr(MFMA_ACC_ELEMS)])
            self.lds.store(p_off0 + a * MFMA_MNK * 2, fx.BFloat16, Pa.to(fx.BFloat16))
        ls = self.fused_sum(p_vals)
        for sh in (16, 32):
            ls = ls + ls.shuffle_xor(sh, WAVE)
        if self.ctx.rgroup == 0:
            self.lds.store_wave(self.lsum_offset(m), self.ctx.lane16, self.ctx.warp, ls)
        # Empty history contributes exp2(-inf - safe_max) = 0.
        corr = fx.Float32(fx.exp2(m_prev - safe_max, fastmath="fast"))
        return m_new, corr

    def fused_row_sum(self, m):
        parts = self.lds.load_wave(self.lsum_offset(m), self.ctx.lane16)
        return self.fused_sum([parts[i] for i in range_constexpr(self.traits.NWARP)])

    @flyc.jit
    def pack_multi(
        self,
        masked_chunks,
        m,
        m_prev,
        p_base,
        lsum_base,
        v_scale_shared,
        cur_kv_buf,
        norm_factor_b,
    ):
        tile_max = self.lds.load_wave(self.lmax_offset(m), self.ctx.lane16).reduce(
            ReductionOp.MAX, fastmath=self.ctx.fm_nnan
        )
        m_new = (
            tile_max
            if const_expr(self.traits.single_tile_plan)
            else fx.maxnumf(m_prev, tile_max, fastmath=self.ctx.fm_nnan)
        )
        # Empty rows need exponent reference 0 to avoid -inf-(-inf).
        safe_max = (m_new > self.ctx.NEG_INF).select(m_new, self.ctx.ZERO_F)
        m_new_b = fx.full(4, fx.Float32(safe_max), fx.Float32)
        ls = fx.Float32(0.0)
        words = []
        for a in range_constexpr(self.traits.NCHUNK):
            Pa = fx.Vector(fx.exp2(masked_chunks[a] - m_new_b, fastmath="fast"))
            ls = ls + Pa.reduce(ReductionOp.ADD)
            if const_expr(self.traits.is_bf16_kv):
                p_scaled = Pa
            elif const_expr(self.traits.per_token_kv):
                v_sc = (
                    self.kv.load_scale(self.traits.sVScale_off, a, cur_kv_buf)
                    if const_expr(self.traits.M_TILES >= 4)
                    else v_scale_shared[a]
                )
                p_scaled = Pa * v_sc * norm_factor_b
            else:
                p_scaled = Pa * fx.full(4, self.traits.FP8_MAX, fx.Float32)
            words.append(
                p_scaled.to(fx.BFloat16)
                if const_expr(self.traits.is_bf16_kv)
                else self.gemm.fp8_words(p_scaled)[0]
            )

        p_off0 = (
            p_base
            + self.ctx.lane16 * self.traits.SP_ROW_BYTES
            + (self.ctx.warp * self.traits.TOK_PER_WARP + self.ctx.rgroup * 4)
            * self.traits.QP_ELEM_BYTES
        )
        # Scatter P words in the interleave expected by PV reads.
        for a in range_constexpr(self.traits.NCHUNK):
            if const_expr(self.traits.is_bf16_kv):
                self.lds.store(p_off0 + a * MFMA_MNK * 2, fx.BFloat16, words[a])
            else:
                self.lds.store(
                    p_off0 + a * MFMA_MNK,
                    fx.Int32,
                    fx.Vector.from_elements([words[a]], dtype=fx.Int32),
                )
        for sh in (16, 32):
            ls = ls + ls.shuffle_xor(sh, WAVE)
        # Empty history must contribute exp2(-inf-safe_max)=0;
        # replacing m_prev with 0 can overflow for negative logits.
        corr_reg = (
            self.ctx.ZERO_F
            if const_expr(self.traits.single_tile_plan)
            else fx.Float32(fx.exp2(m_prev - safe_max, fastmath="fast"))
        )
        if self.ctx.rgroup == 0:
            self.lds.store_wave(lsum_base, self.ctx.lane16, self.ctx.warp, ls)
        return m_new, corr_reg

    @flyc.jit
    def pack_single(
        self,
        masked_chunks,
        m_prev,
        scale,
        v_scale_vecs,
        cur_kv_buf,
        norm_factor_b,
        m,
        masked=True,
    ):
        # pass 2: global max over warps -> exp -> fp8 P pack (-> sP) -> sum.
        # M-tile m owns its Lmax, P and Lsum slots.
        if const_expr(self.traits.M1_SCALE_BEFORE_MASK):
            tile_max = self.lds.load_wave(self.lmax_offset(m), self.ctx.lane16).reduce(
                ReductionOp.MAX, fastmath=self.ctx.fm_nnan
            )
            m_new = (
                tile_max
                if const_expr(self.traits.single_tile_plan)
                else fx.maxnumf(m_prev, tile_max, fastmath=self.ctx.fm_nnan)
            )
        else:
            tile_max = self.lds.load_wave(self.lmax_offset(m), self.ctx.lane16).reduce(
                ReductionOp.MAX
            )
            m_new = (
                tile_max
                if const_expr(self.traits.single_tile_plan)
                else fx.maxnumf(m_prev, tile_max)
            )
        # Keep empty partitions' persisted max at -inf, but exponentiate
        # relative to 0 to avoid -inf-(-inf) NaNs in P and corrections.
        softmax_max = m_new
        if const_expr(self.traits.M1_SCALE_BEFORE_MASK):
            softmax_max = (m_new > self.ctx.NEG_INF).select(m_new, self.ctx.ZERO_F)
        m_new_b = fx.full(4, fx.Float32(softmax_max), fx.Float32)
        ls = fx.Float32(0.0)
        words = []
        if const_expr(not self.traits.M1_SCALE_BEFORE_MASK):
            zero4_p = fx.full(4, 0.0, fx.Float32)
        for a in range_constexpr(self.traits.NCHUNK):
            if const_expr(self.traits.M1_SCALE_BEFORE_MASK):
                Pa = fx.Vector(fx.exp2(masked_chunks[a] - m_new_b, fastmath="fast"))
            elif const_expr(not masked):
                Pa = fx.Vector(
                    fx.exp2(masked_chunks[a] * scale - m_new_b, fastmath="fast")
                )
            else:
                # Finite mask sentinels require an explicit zero probability.
                valid_a = masked_chunks[a] > fx.full(4, -1e29, fx.Float32)
                Pa = valid_a.select(
                    fx.Vector(
                        fx.exp2(
                            masked_chunks[a] * scale - m_new_b,
                            fastmath="fast",
                        )
                    ),
                    zero4_p,
                )
            ls = ls + Pa.reduce(ReductionOp.ADD)
            if const_expr(self.traits.is_bf16_kv):
                p_scaled = Pa
            elif const_expr(self.traits.per_token_kv):
                v_scale_this = (
                    self.kv.load_scale(self.traits.sVScale_off, a, cur_kv_buf)
                    if const_expr(self.traits.head_dim == 64)
                    else v_scale_vecs[a]
                )
                p_scaled = Pa * v_scale_this * norm_factor_b
            else:
                p_scaled = Pa * fx.full(4, self.traits.FP8_MAX, fx.Float32)
            words.append(
                p_scaled.to(fx.BFloat16)
                if const_expr(self.traits.is_bf16_kv)
                else self.gemm.fp8_words(p_scaled)[0]
            )
        p_off0 = (
            self.p_offset(m)
            + self.ctx.lane16 * self.traits.SP_ROW_BYTES
            + (self.ctx.warp * self.traits.TOK_PER_WARP + self.ctx.rgroup * 4)
            * self.traits.QP_ELEM_BYTES
        )
        for a in range_constexpr(self.traits.NCHUNK):
            if const_expr(self.traits.is_bf16_kv):
                self.lds.store(p_off0 + a * MFMA_MNK * 2, fx.BFloat16, words[a])
            else:
                self.lds.store(
                    p_off0 + a * MFMA_MNK,
                    fx.Int32,
                    fx.Vector.from_elements([words[a]], dtype=fx.Int32),
                )
        if const_expr(self.traits.head_dim == 64):
            fx.rocdl.sched_dswr(self.traits.NCHUNK)
        for sh in (16, 32):
            ls = ls + ls.shuffle_xor(sh, WAVE)
        corr_reg = (
            self.ctx.ZERO_F
            if const_expr(self.traits.single_tile_plan)
            else fx.Float32(fx.exp2(m_prev - softmax_max, fastmath="fast"))
        )
        if self.ctx.rgroup == 0:
            self.lds.store_wave(self.lsum_offset(m), self.ctx.lane16, self.ctx.warp, ls)
        return m_new, corr_reg
