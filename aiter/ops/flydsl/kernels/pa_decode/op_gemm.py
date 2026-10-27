# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8/BF16 operand conversion and QK/PV MFMA/WMMA operations."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from ..tensor_shim import _to_raw
from .traits import MFMA_ACC_ELEMS


class PaDecodeGemm:
    def __init__(self, traits, policy):
        self.traits = traits
        self.policy = policy

    def mfma(self, a_ops, b_ops, a_base, b_base, k_packs, acc):
        if const_expr(self.traits.is_bf16_kv):
            for pack in range_constexpr(k_packs):
                # Keep four BF16 values packed in i64 through the pipeline,
                # exposing the instruction's i16x4 operands only here.
                a_pack = fx.Vector.from_elements(
                    [a_ops[a_base + pack]], dtype=fx.Int64
                ).bitcast(fx.Int16)
                b_pack = fx.Vector.from_elements(
                    [b_ops[b_base + pack]], dtype=fx.Int64
                ).bitcast(fx.Int16)
                acc = fx.rocdl.mfma_f32_16x16x16bf16_1k(
                    T.f32x4, [a_pack, b_pack, acc, 0, 0, 0]
                )
        elif const_expr(
            self.traits.WIDE_FP8_MFMA and k_packs % self.traits.PACKS_PER_MFMA == 0
        ):
            for inst in range_constexpr(k_packs // self.traits.PACKS_PER_MFMA):
                a_pack = fx.Vector.from_elements(
                    [
                        a_ops[a_base + inst * self.traits.PACKS_PER_MFMA + i]
                        for i in range_constexpr(self.traits.PACKS_PER_MFMA)
                    ],
                    dtype=fx.Int64,
                ).bitcast(fx.Int32)
                b_pack = fx.Vector.from_elements(
                    [
                        b_ops[b_base + inst * self.traits.PACKS_PER_MFMA + i]
                        for i in range_constexpr(self.traits.PACKS_PER_MFMA)
                    ],
                    dtype=fx.Int64,
                ).bitcast(fx.Int32)
                acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4,
                    [
                        a_pack,
                        b_pack,
                        acc,
                        0,
                        0,
                        0,
                        fx.Int32(0x7F7F7F7F),
                        0,
                        fx.Int32(0x7F7F7F7F),
                    ],
                )
        else:
            for pack in range_constexpr(k_packs):
                acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                    T.f32x4,
                    [a_ops[a_base + pack], b_ops[b_base + pack], acc, 0, 0, 0],
                )
        return acc

    def fp8_words(self, vf32):
        n = vf32.shape[0]
        words = []
        for i in range_constexpr(n // 4):
            b = i * 4
            lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b], vf32[b + 1], 0, False)
            words.append(
                fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b + 2], vf32[b + 3], lo, True)
            )
        return fx.Vector.from_elements(words, dtype=fx.Int32)

    def fp8_residual(self, scaled, word):
        # Retain probability-packing error in a second FP8 term. Its magnitude
        # is at most 16 (OCP) or 8 (FNUZ); scaling by 16 fits either format.
        lo = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, word, False))
        hi = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, word, True))
        rounded = fx.Vector.from_elements(
            [lo[0], lo[1], hi[0], hi[1]], dtype=fx.Float32
        )
        return self.fp8_words((scaled - rounded) * fx.full(4, 16.0, fx.Float32))

    def qk(self, k_ops, q_ops, m):
        frag_Ss = []
        for a in range_constexpr(self.traits.NCHUNK):
            acc = fx.full(MFMA_ACC_ELEMS, 0.0, fx.Float32)
            if const_expr(a < self.policy.qk_chunks):
                acc = self.mfma(
                    k_ops,
                    q_ops,
                    a * self.traits.N_SUBCHUNKS,
                    m * self.traits.N_SUBCHUNKS,
                    self.traits.N_SUBCHUNKS,
                    acc,
                )
            frag_Ss.append(fx.Vector(acc))
        return frag_Ss

    def pv(self, v_ops, p_ops, residual_ops=None):
        k_packs = self.policy.pv_packs
        acc = fx.full(MFMA_ACC_ELEMS, 0.0, fx.Float32)
        if const_expr(residual_ops is not None):
            acc = self.mfma(v_ops, residual_ops, 0, 0, k_packs, acc)
            acc = fx.Vector(acc) * fx.full(MFMA_ACC_ELEMS, 1.0 / 16, fx.Float32)
        acc = self.mfma(v_ops, p_ops, 0, 0, k_packs, acc)
        return fx.Vector(acc)


class Gfx1250Gemm:
    @staticmethod
    def mma(a, b, acc, k, bf16=False):
        if const_expr(bf16):
            result = fx.rocdl.wmma_f32_16x16x32_bf16(
                fx.Vector.make_type(8, fx.Float32),
                _to_raw(a),
                _to_raw(b),
                _to_raw(acc),
                modC=ir.Attribute.parse("#rocdl<wmma_c_modifier none>"),
                reuseA=False,
                reuseB=False,
            ).result
        elif const_expr(k == 128):
            result = fx.rocdl.wmma_f32_16x16x128_fp8_fp8(
                fx.Vector.make_type(8, fx.Float32), a, b, acc
            )
        else:
            result = fx.rocdl.wmma_f32_16x16x64_fp8_fp8(
                fx.Vector.make_type(8, fx.Float32),
                _to_raw(a),
                _to_raw(b),
                _to_raw(acc),
                modC=ir.Attribute.parse("#rocdl<wmma_c_modifier none>"),
                reuseA=False,
                reuseB=False,
            ).result
        return fx.Vector(result)
