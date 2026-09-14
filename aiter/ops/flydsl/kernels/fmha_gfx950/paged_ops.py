# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
    DualwaveFp8KernelContext,
    _anchor_scalar_f32,
    _anchor_v_p,
    _apply_dualwave_causal_mask_pair,
    _attn_mask_vec2_imm,
    _causal_pair_thresholds,
    _exp2_score_slice,
    _pack_p_v8_slices,
    _read_exec_i64,
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


class DualwaveFp8GemmHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)
        if const_expr(self.traits.PAGED):
            self.fp8_mma = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, fx.Float8E4M3FN)
            )

    def _pack_p_fp8(self, f32):
        # P words must follow vectorized K's four-token groups for paged P*V.
        packed = self._pack_fp8_i32x8(f32)
        if const_expr(self.traits.PAGED):
            words = Vec(packed, (8,), fx.Int32)
            return Vec.from_elements(
                [words[i] for i in (0, 2, 1, 3, 4, 6, 5, 7)],
                fx.Int32,
            ).ir_value()
        return packed

    def _mfma_acc_fp8_wide(self, a_i32x8, b_i32x8, c_v16):
        if const_expr(self.traits.PAGED):
            a = fx.make_rmem_tensor(8, fx.Int32)
            b = fx.make_rmem_tensor(8, fx.Int32)
            c = fx.make_rmem_tensor(16, fx.Float32)
            a.store(Vec(a_i32x8))
            b.store(Vec(b_i32x8))
            c.store(Vec(c_v16))
            fx.gemm(
                self.fp8_mma,
                c,
                a,
                b,
                c,
                scale_a=fx.Int32(0x7F7F7F7F),
                scale_b=fx.Int32(0x7F7F7F7F),
            )
            return c.load().ir_value()
        # Wide fp8 QK: mfma_scale (32x32x64) with unit E8M0 scales, i32x8 operands.
        return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            self.v16f32_type,
            [
                as_mlir_value(a_i32x8),
                as_mlir_value(b_i32x8),
                as_mlir_value(c_v16),
                0,
                0,
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
            ],
        )

    def _pack_fp8_i32x8(self, f32_vals):
        c0 = llvm.mlir_poison(T.i32)
        words = []
        for g in range_constexpr(8):
            base = g * 4
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base]),
                as_mlir_value(f32_vals[base + 1]),
                c0,
                0,
            )
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base + 2]),
                as_mlir_value(f32_vals[base + 3]),
                w,
                1,
            )
            words.append(fx.Int32(w))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _v_concat_i32x8(self, v_v, dc):
        words = []
        for ks in range_constexpr(4):
            v2 = Vec.from_elements([fx.Int64(v_v[ks][dc])], fx.Int64).bitcast(fx.Int32)
            words.append(fx.Int32(v2[0]))
            words.append(fx.Int32(v2[1]))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _load_q_wide_lds(self):
        traits = self.traits
        q_row_in_block = self.ctx_ref.q_row_in_block
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = (
                q_row_in_block * fx.Index(traits.HEAD_DIM) + fx.Index(ws * 64) + d_base
            )
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs

    def _load_q_wide_global(self):
        """Pull this lane's Q operands straight from global into VGPRs."""
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            elem = self.global_idx_q(self.ctx_ref.q_row, fx.Index(ws * 64) + d_base)
            lo = self.buffer_load_128(elem)
            hi = self.buffer_load_128(elem + fx.Index(16))
            packs.append(Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs

    def load_q_wide(self):
        if const_expr(self.traits.QLDS):
            return self._load_q_wide_lds()
        return self._load_q_wide_global()

    def qk(self, v_k, q_wide=None):
        traits = self.traits
        k_lo, k_hi = v_k
        q_all_wide = self._load_q_wide_lds() if q_wide is None else q_wide
        v_s_lo = self.c_zero_v16f32
        v_s_hi = self.c_zero_v16f32
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            q_w = q_all_wide[ws]
            v_s_lo = self._mfma_acc_fp8_wide(k_lo[ws], q_w, v_s_lo)
            v_s_hi = self._mfma_acc_fp8_wide(k_hi[ws], q_w, v_s_hi)
        n_ds = const_expr(traits.HEAD_DIM // 64 * 4)
        n_mfma = const_expr(traits.HEAD_DIM // 64 * 2)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, 12)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, n_mfma - 1, 12)
        return (v_s_lo, v_s_hi)

    def cast_p_fp8_direct(self, v_p):
        lo_partial_list, hi_full = v_p
        f32 = []
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [hi_full[p_base + s] for s in range_constexpr(8)]
        return self._pack_p_fp8(f32)

    def _pv_fp8_direct(self, p_fp8, v_v, v_o):
        v_o = self.preserve_accumulators(v_o)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_op = self._v_concat_i32x8(v_v, dc)
            v_o[dc] = self._mfma_acc_fp8_wide(v_op, p_fp8, v_o[dc])
        return v_o

    def pv(self, v_p, v_v, v_o):
        return self._pv_fp8_direct(v_p, v_v, v_o)


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
        # Vectorized-K sigma arranges wide FP8 scores in four-token groups
        # (0..3, 8..11, ...), matching dense FP8 but not vectorized BF16.
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        # q_row_i32 is set by init_q_row (called after helper construction), so read
        # it from the live ctx.
        rel_lo_i32 = fx.Int32(
            self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
        pair_thresholds = _causal_pair_thresholds(False)
        if const_expr(self.traits.PAGED):
            for r in range_constexpr(16):
                threshold = (r // 4) * 8 + r % 4
                s_lo[r] = (rel_lo_i32 < threshold).select(self.c_neg_inf, s_lo[r])
                s_hi[r] = (rel_hi_i32 < threshold).select(self.c_neg_inf, s_hi[r])
        else:
            _apply_dualwave_causal_mask_pair(
                s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds
            )
            _apply_dualwave_causal_mask_pair(
                s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds
            )

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            kv_end_pos = self.traits.BLOCK_N

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos):
            s_lo, s_hi = v_s
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
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
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
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
    _P_HEADROOM_LOG2 = 8.807354922057604

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

    def anchor_v_p(self, v_p):
        return _anchor_v_p(self.traits, v_p, elem_dtype=self.p_elem)

    def anchor_v_o(self, v_o):
        return self.preserve_accumulators(v_o)

    def anchor_scalar_f32(self, x):
        if const_expr(self.traits.PAGED):
            return llvm.intr_arithmetic_fence(fx.as_ir_value(x))
        return _anchor_scalar_f32(x)

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


class DualwaveFp8StoreHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _o_pack_2dw(self, v_o, dc, store_group):
        r_base = store_group * 4
        if const_expr(self.traits.PAGED):
            values = Vec(v_o[dc])
            packed = (
                values.shuffle(values, list(range(r_base, r_base + 4)))
                .to(fx.BFloat16)
                .bitcast(fx.Int32)
            )
            return packed[0].ir_value(), packed[1].ir_value()
        lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
        hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
        return lo, hi

    def _swap_half_partner(self, dw):
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        swapped = rocdl.permlane32_swap(
            pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
        )
        lo_res = llvm.extractvalue(T.i32, swapped, [0])
        hi_res = llvm.extractvalue(T.i32, swapped, [1])
        return (self.lane_div_32 != fx.Index(0)).select(lo_res, hi_res)

    def _packed_o_128_dwords(self, v_o, dc, g):
        is_hi_half = self.lane_div_32 != fx.Index(0)
        d0_a, d1_a = self._o_pack_2dw(v_o, dc, 2 * g)
        d0_b, d1_b = self._o_pack_2dw(v_o, dc, 2 * g + 1)
        y0_a, y1_a = self._swap_half_partner(d0_a), self._swap_half_partner(d1_a)
        y0_b, y1_b = self._swap_half_partner(d0_b), self._swap_half_partner(d1_b)
        w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
        w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
        w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
        w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
        return w0, w1, w2, w3

    def _packed_o_128_vec(self, v_o, dc, g):
        return Vec.from_elements(
            [fx.Int32(w) for w in self._packed_o_128_dwords(v_o, dc, g)], fx.Int32
        )

    def store_final_o(self, v_o, q_row):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_o(q_row, d_col)
                self.buffer_store_128(o_pack, o_global)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        m_row = fx.Float32(m_row) * self.c_logit_scale
        split_z = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
        o_part_row_base = (
            (split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v
            + q_row
        ) * (self.traits.HEAD_DIM_V // 2)
        grid_z = fx.Index(gpu.grid_dim.z)
        mrow_base = (
            grid_z
            * self.traits.NUM_HEADS_Q
            * self.seq_len_v
            * (self.traits.HEAD_DIM_V // 2)
        )
        lrow_base = mrow_base + grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v
        ml_row_idx = (
            split_z * self.traits.NUM_HEADS_Q + self.q_head_idx
        ) * self.seq_len_v + q_row

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < self.seq_len_v:
                for dc in range_constexpr(self.traits.D_CHUNKS):
                    for g in range_constexpr(2):
                        dw_col = (
                            dc * (self.traits.D_CHUNK // 2)
                            + (2 * g + self.lane_div_32) * 4
                        )
                        self.ws_store_quad_i32(
                            self._packed_o_128_dwords(v_o, dc, g),
                            o_part_row_base + dw_col,
                        )
                if self.lane < fx.Index(32):
                    self.ws_store_f32(m_row, mrow_base + ml_row_idx)
                    self.ws_store_f32(l_row, lrow_base + ml_row_idx)

        _store_splitk_partial_if_qrow()

    def store_empty_split(self):
        @flyc.jit
        def _store_empty_split():
            if self.max_num_tiles < self.split_t0 + fx.Index(4):
                q_row_e = self.q_start + self.wave_q_offset + self.lane_mod_32
                split_z_e = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
                o_row_base_e = (
                    (split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx)
                    * self.seq_len_v
                    + q_row_e
                ) * (self.traits.HEAD_DIM_V // 2)
                grid_z_e = fx.Index(gpu.grid_dim.z)
                mrow_base_e = (
                    grid_z_e
                    * self.traits.NUM_HEADS_Q
                    * self.seq_len_v
                    * (self.traits.HEAD_DIM_V // 2)
                )
                lrow_base_e = (
                    mrow_base_e + grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v
                )
                ml_row_e = (
                    split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx
                ) * self.seq_len_v + q_row_e
                if q_row_e < self.seq_len_v:
                    c_zero_i = fx.Int32(0)
                    for dc in range_constexpr(self.traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            dw_col = (
                                dc * (self.traits.D_CHUNK // 2)
                                + (2 * g + self.lane_div_32) * 4
                            )
                            self.ws_store_quad_i32(
                                [c_zero_i, c_zero_i, c_zero_i, c_zero_i],
                                o_row_base_e + dw_col,
                            )
                    if self.lane < fx.Index(32):
                        self.ws_store_f32(fx.Float32(-1e30), mrow_base_e + ml_row_e)
                        self.ws_store_f32(self.c_zero_f, lrow_base_e + ml_row_e)

        _store_empty_split()
