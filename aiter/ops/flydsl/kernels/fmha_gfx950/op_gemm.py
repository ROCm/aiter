# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""The two GEMMs: QK^T and PV."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels.fmha_gfx950.pipeline import (
    DualwaveFp8KernelContext,
    _anchor_v_o,
)


class DualwaveFp8GemmHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _mfma_acc_fp8_wide(self, a_i32x8, b_i32x8, c_v16):
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
            v2 = Vec(
                llvm.bitcast(self.v2i32_type, as_mlir_value(v_v[ks][dc])),
                (2,),
                fx.Int32,
            )
            words.append(fx.Int32(v2[0]))
            words.append(fx.Int32(v2[1]))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _load_q_wide_lds(self):
        traits = self.traits
        q_row_in_block = self.ctx_ref.q_row_in_block
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = q_row_in_block * traits.HEAD_DIM + (ws * 64) + d_base
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs

    def _load_q_wide_global(self):
        """Pull this lane's Q operands straight from global into VGPRs (head_dim > 128)."""
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            elem = self.global_idx_q(self.ctx_ref.q_row, (ws * 64) + d_base)
            lo = self.buffer_load_128(elem)
            hi = self.buffer_load_128(elem + 16)
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
        return self._pack_fp8_i32x8(f32)

    def _pv_fp8_direct(self, p_fp8, v_v, v_o):
        v_o = _anchor_v_o(self.traits, v_o)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_op = self._v_concat_i32x8(v_v, dc)
            v_o[dc] = self._mfma_acc_fp8_wide(v_op, p_fp8, v_o[dc])
        return v_o

    def pv(self, v_p, v_v, v_o):
        return self._pv_fp8_direct(v_p, v_v, v_o)
