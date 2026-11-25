# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8/BF16 operand conversion and QK/PV MFMA/WMMA operations."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .traits import MFMA_ACC_ELEMS

# e8m0 byte 127 (2**0) in all four scale bytes: the scaled MFMA applies no scaling.
UNIT_E8M0_SCALES = 0x7F7F7F7F


class PaDecodeGemm:
    def __init__(self, traits):
        self.traits = traits

    def mfma(self, a_ops, b_ops, a_base, b_base, k_packs, acc):
        if const_expr(self.traits.PV_BF16):
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
        elif const_expr(self.traits.WIDE_FP8_MFMA):
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
                        fx.Int32(UNIT_E8M0_SCALES),
                        0,
                        fx.Int32(UNIT_E8M0_SCALES),
                    ],
                )
        else:
            for pack in range_constexpr(k_packs):
                acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                    T.f32x4,
                    [a_ops[a_base + pack], b_ops[b_base + pack], acc, 0, 0, 0],
                )
        return acc

    def mfma_bf16_k32(self, a_ops, b_ops, a_base, b_base, k_packs, acc):
        """gfx950 K32 BF16 MFMA over pairs of K16 packs.

        A and B pair the same two packs, so the shared K permutation leaves the
        dot product unchanged and halves the MFMA count.
        """
        for inst in range_constexpr(k_packs // 2):
            a_pack = fx.Vector.from_elements(
                [a_ops[a_base + 2 * inst], a_ops[a_base + 2 * inst + 1]],
                dtype=fx.Int64,
            ).bitcast(fx.BFloat16)
            b_pack = fx.Vector.from_elements(
                [b_ops[b_base + 2 * inst], b_ops[b_base + 2 * inst + 1]],
                dtype=fx.Int64,
            ).bitcast(fx.BFloat16)
            acc = fx.rocdl.mfma_f32_16x16x32_bf16(
                T.f32x4, [a_pack, b_pack, acc, 0, 0, 0]
            )
        return acc

    def qk_bf16_k32(self, k_ops, q_ops, a):
        """Scores of K chunk ``a`` against one M-tile's N_SUBCHUNKS Q packs."""
        acc = fx.full(MFMA_ACC_ELEMS, 0.0, fx.Float32)
        acc = self.mfma_bf16_k32(
            k_ops,
            q_ops,
            a * self.traits.N_SUBCHUNKS,
            0,
            self.traits.N_SUBCHUNKS,
            acc,
        )
        return fx.Vector(acc)

    def pv_bf16_k32(self, v_ops, p_ops, acc):
        """Accumulate PV into ``acc`` (the rescaled running output)."""
        return fx.Vector(self.mfma_bf16_k32(v_ops, p_ops, 0, 0, self.traits.NVOPS, acc))

    def fp8_words(self, vf32):
        n = vf32.shape[0]
        words = []
        for i in range_constexpr(n // 4):
            b = i * 4
            # The second conversion writes the high half, so on gfx950 the
            # first one's old word is undef rather than a zeroed register.
            old = llvm.mlir_undef(T.i32) if const_expr(self.traits.IS_GFX950) else 0
            lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b], vf32[b + 1], old, False)
            words.append(
                fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b + 2], vf32[b + 3], lo, True)
            )
        return fx.Vector.from_elements(words, dtype=fx.Int32)

    def qk(self, k_ops, q_ops, m):
        frag_Ss = []
        for a in range_constexpr(self.traits.NCHUNK):
            acc = fx.full(MFMA_ACC_ELEMS, 0.0, fx.Float32)
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

    def qk_mxfp4(self, k_raw, k_scales, q_op, q_scale_byte):
        """QK of one MXFP4 K tile: per 16-token chunk one scaled MFMA, A = packed
        E2M1 K (lane: 16 B = one E8M0 group, its scale byte), B = E4M3 Q (``q_op``)
        scaled by the e8m0 byte ``q_scale_byte``, so the f32 accumulator never exceeds
        the logit by more than the 2x left in the mantissa of the query scale."""
        frag_Ss = []
        for a in range_constexpr(self.traits.NCHUNK):
            a_k = fx.Vector.from_elements(
                [fx.Int32(k_raw[a * 5 + 1 + i]) for i in range_constexpr(4)]
                + [fx.Int32(0) for _ in range_constexpr(4)],
                dtype=fx.Int32,
            )
            acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [
                    a_k,
                    q_op,
                    fx.full(MFMA_ACC_ELEMS, 0.0, fx.Float32),
                    4,
                    0,
                    0,
                    k_scales[a],
                    0,
                    q_scale_byte,
                ],
            )
            frag_Ss.append(fx.Vector(acc))
        return frag_Ss

    def pv(self, v_ops, p_ops):
        acc = fx.full(MFMA_ACC_ELEMS, 0.0, fx.Float32)
        acc = self.mfma(v_ops, p_ops, 0, 0, self.traits.NVOPS, acc)
        return fx.Vector(acc)


class Gfx1250Gemm:
    # V_WMMA_SCALE_F32_16X16X128_F8F6F4 operand formats.
    FMT_E4M3, FMT_E2M1 = 0, 4

    @staticmethod
    def mma_mx(a, b, scale_a, scale_b, fmt_b):
        """One scaled 16x16x128 WMMA from zero: A is E2M1 (8 dwords a lane), B is
        E4M3 (16 dwords) or E2M1 (8 dwords). Lane (r, h) holds row/column r, K bytes
        [16 h, +16) and [32 + 16 h, +16) of a 64-byte E2M1 row (a 128-byte E4M3 row
        keeps the half-wave's alternating 16-byte chunks). Lanes 0-15 supply the
        scales: byte b of row/column r's dword is the E8M0 of K 32 b .. 32 b + 31."""
        # Raw op, as in ``mma``: operands are register vectors, while the
        # WMMAScale atom's fx.gemm takes rmem tensors.
        return fx.Vector(
            fx.rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                fx.Vector.make_type(8, fx.Float32),
                a,
                b,
                fx.Vector.filled(8, 0.0, fx.Float32),
                scale_a,
                scale_b,
                fmtA=Gfx1250Gemm.FMT_E2M1,
                fmtB=fmt_b,
            )
        )

    @staticmethod
    def mma(a, b, acc, k, bf16=False):
        if const_expr(bf16):
            result = fx.rocdl.wmma_f32_16x16x32_bf16(
                fx.Vector.make_type(8, fx.Float32),
                fx.as_ir_value(a),
                fx.as_ir_value(b),
                fx.as_ir_value(acc),
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
                fx.as_ir_value(a),
                fx.as_ir_value(b),
                fx.as_ir_value(acc),
                modC=ir.Attribute.parse("#rocdl<wmma_c_modifier none>"),
                reuseA=False,
                reuseB=False,
            ).result
        return fx.Vector(result)
