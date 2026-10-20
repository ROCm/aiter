# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors


"""FP8/BF16 QK and PV WMMA operations for gfx1250 PA decode."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr

from ..tensor_shim import _to_raw


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
