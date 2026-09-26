# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Paged BF16 output packing and optional FP32 log-sum-exp stores."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.fmha_gfx950.common import LN2
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
    DualwaveFp8KernelContext,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_store


class DualwaveFp8StoreHelper(DualwaveFp8KernelContext):
    def _o_pack_2dw(self, v_o, dc, store_group):
        r_base = store_group * 4
        values = Vec(v_o[dc])
        packed = (
            values.shuffle(values, list(range(r_base, r_base + 4)))
            .to(fx.BFloat16)
            .bitcast(fx.Int32)
        )
        return packed[0].ir_value(), packed[1].ir_value()

    def _swap_half_partner(self, dw):
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        swapped = rocdl.permlane32_swap(pair_i32_ty, dw, dw, False, False)
        lo_res = llvm.extractvalue(T.i32, swapped, [0])
        hi_res = llvm.extractvalue(T.i32, swapped, [1])
        return (self.lane_div_32 != 0).select(lo_res, hi_res)

    def _packed_o_128_dwords(self, v_o, dc, g):
        is_hi_half = self.lane_div_32 != 0
        d0_a, d1_a = self._o_pack_2dw(v_o, dc, 2 * g)
        d0_b, d1_b = self._o_pack_2dw(v_o, dc, 2 * g + 1)
        y0_a, y1_a = self._swap_half_partner(d0_a), self._swap_half_partner(d1_a)
        y0_b, y1_b = self._swap_half_partner(d0_b), self._swap_half_partner(d1_b)
        w0 = is_hi_half.select(y0_b, d0_a)
        w1 = is_hi_half.select(y1_b, d1_a)
        w2 = is_hi_half.select(d0_b, y0_a)
        w3 = is_hi_half.select(d1_b, y1_a)
        return w0, w1, w2, w3

    def _packed_o_128_vec(self, v_o, dc, g):
        return Vec.from_elements(
            [fx.Int32(w) for w in self._packed_o_128_dwords(v_o, dc, g)], fx.Int32
        )

    def store_final_o(self, v_o, q_row, m_row, l_row):
        # Page-1 V192 fills the VGPR budget. Finish LSE before packing O so
        # its normalization state does not overlap the output temporaries.
        if const_expr(self.traits.LSE_BEFORE_O):
            self.store_lse(q_row, m_row, l_row)
        if const_expr(self.traits.GUARD_OUTPUT_ROWS):
            live_row = q_row < self.seqlen_q_v
            end_elem = self.q_tok_end * self.stride_o_n_v
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_o(q_row, d_col)
                # Inactive rows can wrap a 32-bit byte offset before the buffer
                # bound is checked. The exact descriptor end is always OOB.
                if const_expr(self.traits.GUARD_OUTPUT_ROWS):
                    o_global = live_row.select(o_global, end_elem)
                self.buffer_store_128(o_pack, o_global)
        if const_expr(self.traits.RETURN_LSE and not self.traits.LSE_BEFORE_O):
            self.store_lse(q_row, m_row, l_row)

    def store_lse(self, q_row, m_row, l_row):
        # l_row includes the power-of-two headroom used for FP8 probabilities.
        # Avoid log(0) and -inf arithmetic for empty/fully masked rows, even
        # under the attention kernel's fast-math compilation options.
        with fx.fastmath(None):
            nonempty = fx.Float32(l_row) > self.c_zero_f
            safe_m = nonempty.select(fx.Float32(m_row), self.c_zero_f)
            safe_l = nonempty.select(fx.Float32(l_row), fx.Float32(1.0))
            value = (
                safe_m * self.c_logit_scale * fx.Float32(LN2)
                + fx.log(safe_l)
                - fx.Float32(self.traits.P_HEADROOM_LOG2 * LN2)
            )
            value = nonempty.select(value, self.c_neg_inf)

        # One lane per logical query row. Mask before narrowing the index so
        # loose launch bounds cannot wrap into another request/head's LSE.
        total_q = fx.Int64(self.lse_stride_h)
        end = total_q * self.traits.NUM_HEADS_Q
        index = fx.Int64(self.q_head_idx) * total_q + self.q_tok_base + q_row
        writer = (q_row < self.seqlen_q_v) & (self.lane < 32)
        index = writer.select(index, end)
        target = fx.rocdl.make_buffer_tensor(
            self.LSE, num_records_bytes=(end * 4).ir_value()
        )
        buf_copy_store(target, fx.Int32(index), value, fx.Float32)
