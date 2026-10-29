# SPDX-License-Identifier: MIT
"""FP8 / MX arithmetic of the FlyDSL mono kernels (``common/mx.py``,
``stages/moe.py``: ``code_ceil``, ``rcp_pow2``, ``vllm_mx_code``), in Gluon.
An E8M0 code is a biased exponent; ``pow2(code)`` is the scale it stands for."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

FP8_MAX = gl.constexpr(448.0)
FLT_MIN = gl.constexpr(1.1754943508222875e-38)


@gluon.jit
def clamp_fp8(v):
    return gl.maximum(gl.minimum(v, FP8_MAX), -FP8_MAX)


@gluon.jit
def code_ceil(v):
    """Biased exponent of the smallest power of two >= v (v >= 0, finite),
    clamped to [1, 254]."""
    bits = v.to(gl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + gl.where((bits & 0x7FFFFF) != 0, 1, 0)
    return gl.maximum(gl.minimum(e, 254), 1)


@gluon.jit
def rcp_pow2(code):
    """1 / pow2(code), exact (code in [1, 253])."""
    return ((254 - code) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def div_rn(x, d, r):
    """x / d rounded to nearest (bit-identical to IEEE division), given
    r = RN(1 / d) (Markstein)."""
    q = x * r
    return gl.fma(gl.fma(-q, d, x), r, q)


@gluon.jit
def vllm_mx_code(amax):
    """vLLM ``mxfp8_e4m3_quantize``'s code of a 32-group: the smallest power of
    two >= RN(max(amax, FLT_MIN) / 448), clamped to [0, 254]."""
    a = gl.maximum(amax, FLT_MIN)
    q = div_rn(a, FP8_MAX, 1.0 / 448.0)
    bits = q.to(gl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + gl.where((bits & 0x7FFFFF) != 0, 1, 0)
    sub = gl.where(bits > 0x400000, 1, 0)
    e = gl.where(bits < 0x800000, sub, e)
    return gl.minimum(gl.maximum(e, 0), 254)


@gluon.jit
def fp8_pack4(a, b, c, d):
    """Four f32 (already in range) -> one dword of four E4M3 bytes, RNE
    (v_cvt_pk_fp8_f32, the second pair into the high word)."""
    return gl.inline_asm_elementwise(
        "v_cvt_pk_fp8_f32 $0, $1, $2\nv_cvt_pk_fp8_f32 $0, $3, $4 op_sel:[0,0,1]",
        "=&v,v,v,v,v",
        [a, b, c, d],
        gl.uint32,
        True,
        1,
    )
