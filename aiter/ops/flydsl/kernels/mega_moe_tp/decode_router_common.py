# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared helpers: bf16x3 weight split, the DPP butterfly, the bit casts and the Quark-even e8m0 / FP4 / FP8-amax arithmetic of the decode-shape (1-128 row) MoE router chain.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4`` (``router_gemm.py``, ``router_topk.py``,
``common/loaders.py``, ``i3072/gemm1_prefill_h6144_i3072_persistent.py``, ``mxfp8/gemm1_prefill_h6144_i3072.py``),
Apache-2.0, "Copyright contributors to the amd-kernels project".
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import T as _T

PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _pk_bf16(a, b):
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _split3(w):
    """4 fp32 (i32 bits) -> three terms of 2 bf16-pair dwords: RN(w), RN(w - w0), RN(w - w0 - w1), exact remainders."""
    parts = [[], [], []]
    for i in range_constexpr(2):
        a = fx.Int32(w[2 * i]).bitcast(fx.Float32)
        b = fx.Int32(w[2 * i + 1]).bitcast(fx.Float32)
        for p in range_constexpr(3):
            pk = _pk_bf16(a, b)
            parts[p].append(pk)
            if p < 2:
                a = a - (pk << 16).bitcast(fx.Float32)
                b = b - (pk & -65536).bitcast(fx.Float32)
    return parts


# one token (wave) per workgroup spreads the partial loads over more CUs: 172 rows 3.75 -> 3.33 us after the router
NW = 1


NEG_LOG2E = -1.44269504088896340736


# AITER's wave_reduce on gfx9: quad_perm [1,0,3,2], quad_perm [2,3,0,1],
# row_half_mirror, row_mirror, row_bcast:15, row_bcast:31
DPP_TREE = (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143)


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dpp(v, ctrl):
    """v_mov_b32_dpp, all rows / banks: an invalid source lane keeps its value (the lane-63 result never reads one)."""
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, _ir(v), _ir(v), ctrl, 0xF, 0xF, False))


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _cvt_pk_fp4(old, a, b, scale_f32, sel):
    """v_cvt_scalef32_pk_fp4_f32: two f32 / scale -> 2 e2m1 into byte ``sel``."""
    return fx.Int32(
        _llvm.call_intrinsic(
            _T.i32,
            "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
            [
                fx.as_ir_value(old),
                fx.as_ir_value(a),
                fx.as_ir_value(b),
                fx.as_ir_value(scale_f32),
                fx.as_ir_value(fx.Int32(sel)),
            ],
            [],
            [],
        )
    )


def _e8m0_even_e2m1(amax):
    """Quark FP4 even scale of a nonnegative f32 amax, exponent byte >= 1: exponent(amax + half an e2m1 mantissa ulp,
    0x200000) - 2, the subtraction folded into the bias (a signed shift for tiny inputs).
    """
    t = (_bits(amax) - fx.Int32(0xE00000)) >> 23
    return (t > fx.Int32(1)).select(t, fx.Int32(1))


def _fmax3_abs(a, b, c):
    """Max of 3 absolute values by v_max3_f32 with |.| source modifiers (asm: the MLIR bridge lacks amdgcn.fmax3)."""
    return fx.Float32(
        _llvm.inline_asm(
            _T.f32,
            [fx.as_ir_value(fx.Float32(v)) for v in (a, b, c)],
            "v_max3_f32 $0, |$1|, |$2|, |$3|",
            "=v,v,v,v",
            has_side_effects=False,
        )
    )
