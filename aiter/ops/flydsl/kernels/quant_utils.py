# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared MX-format quantization IR helpers for FlyDSL kernels.

These functions emit MLIR/LLVM IR for the per-block E8M0 scale calculation
and the f32 -> fp4 (e2m1) conversion. They are *IR builders* -- you must
call them inside an active ``InsertionPoint`` (i.e. while the FlyDSL DSL
is mid-build of a kernel function), and they emit the same arith / LLVM
ops the kernels would otherwise emit inline.

The four E8M0 scale-rounding modes mirror PyTorch torchao's
``ScaleCalculationMode`` (``torchao/prototype/mx_formats/config.py``) 1:1
and are dtype-agnostic across the whole MX format family
(mxfp4 / mxfp6 / mxfp8 / mxint8) -- only the ``target_max_pow2`` /
``max_pos`` / ``mbits`` constants differ between dtypes. This matches both
PyTorch torchao's and the HIP-side ``MxScaleRoundMode`` design (see
``csrc/include/mx_quant_utils.h`` and ``aiter.ops.quant.MxScaleRoundMode``).

See ``aiter/utility/fp4_utils.py`` for cross-stack naming (PyTorch torchao /
NV Triton / DSv4 / FlashInfer / AMD Quark) and CPU torch reference
implementations.

This module imports only the *bare-int* mirrors
(:class:`MxScaleRoundModeInt` / :class:`MxDtypeInt`) from
:mod:`aiter.utility.mx_types`; the pybind11 :class:`MxScaleRoundMode` /
:class:`MxDtype` from the same module are loaded lazily and would
trigger a ``module_aiter_core`` JIT build on attribute access, which is
incompatible with wheel ``PREBUILD_KERNELS`` (FlyDSL AOT runs before the
HIP modules are built; see ``setup.py``). Callers may still pass either
view to :func:`emit_mx_e8m0_scale` -- both round-trip through
``int(...)``.
"""

from __future__ import annotations

from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl.expr.arith import CmpIPredicate

# Bare-int mirrors of MxScaleRoundMode / MxDtype (mx_quant_utils.h). Same
# numeric values as the pybind11 enum classes, but importing them is
# JIT-free, which is required at FlyDSL AOT time.
from aiter.utility.mx_types import (
    MxDtypeInt as _D,
    MxScaleRoundModeInt as _M,
    MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE,
)

# Per-MX-dtype constants. Tuple form: (target_max_pow2, max_pos_inv_f32_bits, mbits)
# - target_max_pow2        = log2(largest pow2 <= max_normal(dtype))
# - max_pos_inv_f32_bits   = bit pattern of fp32(1.0 / max_normal(dtype)),
#                            used by RoundUp's ceil_pow2(amax / max_pos)
# - mbits                  = mantissa bits of the target dtype (EVEN only)
# Dict keyed by ``int(MxDtype.X)`` so callers may pass either bare int or
# pybind enum interchangeably.
_DTYPE_CFG = {
    _D.FP4_E2M1: (2, 0x3E2AAAAB, 1),  # max_pos=6.0,    1/6.0    ~= 0.16666667
    _D.FP8_E4M3: (
        8,
        0x3B124925,
        3,
    ),  # max_pos=448.0,  1/448.0  ~= 0.00223214 (OCP / H100 / gfx950+)
    _D.FP8_E4M3_FNUZ: (
        7,
        0x3B888889,
        3,
    ),  # max_pos=240.0,  1/240.0  ~= 0.00416667 (AMD gfx942)
}


def emit_mx_e8m0_scale(
    local_max,
    *,
    mode: int = _DEFAULT_MODE,
    dtype: int = _D.FP4_E2M1,
):
    """Emit IR computing the E8M0 block scale for an MX format.

    FlyDSL IR-builder analogue of PyTorch torchao ``to_mx(scaling_mode,
    elem_dtype)`` and the CPU torch ref
    :func:`aiter.utility.fp4_utils.f32_to_mx_e8m0_scale`. The four
    rounding formulas (FLOOR / RCEIL / CEIL / EVEN) are dtype-agnostic;
    ``dtype`` only selects ``target_max_pow2`` / ``max_pos`` / ``mbits``
    constants from :data:`_DTYPE_CFG`.

    See ``csrc/include/mx_quant_utils.h`` (``MxScaleRoundMode`` /
    ``MxDtype``) and :mod:`aiter.utility.mx_types` for the four formulas
    and cross-stack mapping (PyTorch torchao / NV / DSv4 / FlashInfer /
    AMD Quark naming).

    Args:
        local_max: f32 IR value, the (warp-reduced) ``max(|x|)`` of one
            block. Caller is responsible for the per-block reduction.
        mode: ``MxScaleRoundMode`` value -- accepts either the bare-int
            mirror :class:`aiter.utility.mx_types.MxScaleRoundModeInt`
            (recommended for FlyDSL kernel definitions) or the pybind11
            enum :class:`aiter.utility.mx_types.MxScaleRoundMode` (from
            user-facing code paths). Default ``RoundUp`` (industry
            consensus for MXFP4 and MXFP8).
        dtype: ``MxDtype`` value -- bare-int mirror or pybind11 enum.
            Default ``FP4_E2M1``.

    Returns:
        e8m0_biased: i32 IR value in the range ``[0, 0xFF]``. The caller
        derives ``quant_scale = (254 - e8m0_biased) << 23`` (bitcast to
        f32) for the multiplicative quant scale, and stores
        ``e8m0_biased`` as a ``uint8`` in the per-block scale tensor.
    """
    # Normalise int / pybind enum into a plain int -- pybind11 enum classes
    # don't auto-compare equal to ``int`` (unlike ``IntEnum``).
    mode_int = int(mode)
    dtype_int = int(dtype)
    if dtype_int not in _DTYPE_CFG:
        raise ValueError(
            f"emit_mx_e8m0_scale: unsupported dtype {dtype!r}; "
            f"supported: {list(_DTYPE_CFG)}"
        )
    target_max_pow2, max_pos_inv_bits, mbits = _DTYPE_CFG[dtype_int]

    c0_i32 = arith.constant(0, type=T.i32)
    c1_i32 = arith.constant(1, type=T.i32)
    c23_i32 = arith.constant(23, type=T.i32)
    c0xFF_i32 = arith.constant(0xFF, type=T.i32)  # E8M0 exponent mask
    c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=T.i32)  # f32 mantissa mask
    target_max_pow2_i32 = arith.constant(target_max_pow2, type=T.i32)

    def _clamp_u8(x):
        # Defensive clamp into the E8M0 storage range [0, 0xFF]. Pathological
        # inputs (denormals, fp32 inf, mantissa bump from 0xFF -> 0x100) can
        # otherwise corrupt the stored uint8.
        return arith.minsi(arith.maxsi(x, c0_i32), c0xFF_i32)

    if mode_int == _M.RoundUp:
        # ceil_pow2(amax / max_pos): multiply by reciprocal of max_pos to get
        # the working value, then bump the exponent if any mantissa bit is
        # set. Bit-equivalent to HIP ``aiter::fp_f32_to_e8m0_scale<RoundUp,
        # FP4_E2M1>`` and to PyTorch torchao ``_to_mx_rceil`` (modulo
        # the GPU-vs-CPU fp32 ULP boundary effects documented in the PR).
        c_inv_max_pos = arith.constant(max_pos_inv_bits, type=T.i32)
        inv_max_pos_f32 = c_inv_max_pos.bitcast(T.f32)
        working = local_max * inv_max_pos_f32
        working_i32 = working.bitcast(T.i32)
        mantissa = working_i32 & c0x7FFFFF_i32
        biased_exp = (working_i32 >> c23_i32) & c0xFF_i32
        mant_nonzero = arith.cmpi(CmpIPredicate.ne, mantissa, c0_i32)
        exp_field = arith.select(
            mant_nonzero,
            biased_exp + c1_i32,
            biased_exp,
        )
        return _clamp_u8(exp_field)

    if mode_int == _M.RoundDown:
        # floor_pow2(amax) / 2^target_max_pow2: drop the f32 mantissa, then
        # subtract target_max_pow2 from the biased exponent.
        amax_i32 = local_max.bitcast(T.i32)
        biased_exp = (amax_i32 >> c23_i32) & c0xFF_i32
        return _clamp_u8(biased_exp - target_max_pow2_i32)

    if mode_int == _M.Ceil:
        # ceil_pow2(amax) / 2^target_max_pow2: same as RoundDown but bump
        # the exponent if any mantissa bit is set.
        amax_i32 = local_max.bitcast(T.i32)
        mantissa = amax_i32 & c0x7FFFFF_i32
        biased_exp = (amax_i32 >> c23_i32) & c0xFF_i32
        mant_nonzero = arith.cmpi(CmpIPredicate.ne, mantissa, c0_i32)
        biased_exp_bumped = arith.select(
            mant_nonzero,
            biased_exp + c1_i32,
            biased_exp,
        )
        return _clamp_u8(biased_exp_bumped - target_max_pow2_i32)

    if mode_int == _M.Even:
        # round_pow2_special(amax) / 2^target_max_pow2: add a half-step at
        # the "(mbits+1)-th-from-top" mantissa bit, then drop all mantissa
        # bits. ``val_to_add = 1 << (23 - mbits - 1)`` so that the carry
        # propagates exactly when amax >= 1.5 * 2^k (mbits=1, FP4) or
        # 1.0625 * 2^k (mbits=3, FP8 e4m3) etc. -- mantissa-precision-
        # aware ties-to-even on the power-of-2 lattice.
        val_to_add = 1 << (23 - mbits - 1)
        c_val_add = arith.constant(val_to_add, type=T.i32)
        c_sign_exp_mask = arith.constant(0xFF800000, type=T.i32)
        amax_i32 = local_max.bitcast(T.i32)
        amax_rounded = (amax_i32 + c_val_add) & c_sign_exp_mask
        biased_exp = (amax_rounded >> c23_i32) & c0xFF_i32
        return _clamp_u8(biased_exp - target_max_pow2_i32)

    raise ValueError(
        f"emit_mx_e8m0_scale: unknown mode int {mode_int} for {mode!r} "
        f"(expected one of MxScaleRoundModeInt: "
        f"RoundDown={_M.RoundDown}, RoundUp={_M.RoundUp}, "
        f"Even={_M.Even}, Ceil={_M.Ceil})"
    )


def emit_f32_to_e2m1(qx_f32):
    """Convert a scaled f32 value to FP4 (E2M1) as a 4-bit unsigned nibble.

    Matches:
    - CPU Python ref ``aiter.utility.fp4_utils.f32_to_mxfp4`` (round-to-
      nearest-even normal/denormal/saturate paths)
    - HIP gfx950 HW builtin ``v_cvt_pk_fp4_*`` (exact RNE)
    - HIP gfx942 SW fallback ``even_round_e2m1`` (algorithmically equivalent
      RHA; can differ from the CPU ref by <=1 ULP at FP4 round thresholds
      due to GPU vs CPU fp32 computation order, see PR notes)

    Args:
        qx_f32: f32 IR value, the already-scaled value ``act * quant_scale``.

    Returns:
        e2m1: i32 IR value with the 4-bit nibble in the low bits (sign at
        bit 3, magnitude at bits 0-2). Pack two nibbles into a byte for
        FP4x2 storage.
    """
    c1_i32 = arith.constant(1, type=T.i32)
    c22_i32 = arith.constant(22, type=T.i32)
    c28_i32 = arith.constant(28, type=T.i32)
    c0x7_i32 = arith.constant(0x7, type=T.i32)
    c0x80000000_i32 = arith.constant(0x80000000, type=T.i32)
    c0x7FFFFFFF_i32 = arith.constant(0x7FFFFFFF, type=T.i32)
    c0x3F800000_i32 = arith.constant(0x3F800000, type=T.i32)  # 1.0f
    c0x40C00000_i32 = arith.constant(0x40C00000, type=T.i32)  # 6.0f
    c0x4A800000_i32 = arith.constant(0x4A800000, type=T.i32)  # denorm bias
    c0xC11FFFFF_i32 = arith.constant(0xC11FFFFF, type=T.i32)  # normal bias

    qx = qx_f32.bitcast(T.i32)
    s = qx & c0x80000000_i32
    qx_abs = qx & c0x7FFFFFFF_i32
    denormal_mask = arith.cmpi(CmpIPredicate.ult, qx_abs, c0x3F800000_i32)
    normal_mask = arith.andi(
        arith.cmpi(CmpIPredicate.ult, qx_abs, c0x40C00000_i32),
        arith.cmpi(CmpIPredicate.uge, qx_abs, c0x3F800000_i32),
    )

    denorm_f32 = qx_abs.bitcast(T.f32) + c0x4A800000_i32.bitcast(T.f32)
    denormal_x = denorm_f32.bitcast(T.i32) - c0x4A800000_i32

    mant_odd = (qx_abs >> c22_i32) & c1_i32
    normal_x = qx_abs + c0xC11FFFFF_i32 + mant_odd
    normal_x = normal_x >> c22_i32

    e2m1 = arith.select(normal_mask, normal_x, c0x7_i32)
    e2m1 = arith.select(denormal_mask, denormal_x, e2m1)
    return (s >> c28_i32) | e2m1
