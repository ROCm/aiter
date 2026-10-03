# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8 operand conversion and QK/PV MFMA operations."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .traits import MFMA_ACC_ELEMS


class PaDecodeGemm:
    def __init__(self, traits):
        self.traits = traits

    def mfma(self, a_ops, b_ops, a_base, b_base, k_packs, acc):
        if const_expr(self.traits.WIDE_FP8_MFMA):
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

    def qk(self, k_ops, q_ops, m):
        frag_Ss = []
        for a in range_constexpr(self.traits.NCHUNK):
            acc = fx.Vector.filled(MFMA_ACC_ELEMS, 0.0, fx.Float32)
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

    def pv(self, v_ops, p_ops):
        acc = fx.Vector.filled(MFMA_ACC_ELEMS, 0.0, fx.Float32)
        acc = self.mfma(v_ops, p_ops, 0, 0, self.traits.NVOPS, acc)
        return fx.Vector(acc)
