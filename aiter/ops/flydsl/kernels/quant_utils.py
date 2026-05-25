# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared FP4 / FP8 quantization IR helpers for FlyDSL kernels.

These functions emit MLIR/LLVM IR for the per-block E8M0 scale calculation
and the f32 -> fp4 (e2m1) conversion. They are *IR builders* -- you must
call them inside an active ``InsertionPoint`` (i.e. while the FlyDSL DSL
is mid-build of a kernel function), and they emit the same arith / LLVM
ops the kernels would otherwise emit inline.

The four FP4 scale-rounding modes mirror the HIP-side ``MxScaleRoundMode``
enum in ``csrc/include/fp4_quant_utils.h``. Currently only RoundUp (the
NV / DSv4 / FlashInfer industry default, ``ceil_pow2(amax/6)``) is needed
by the FlyDSL fp4 paths -- the other modes (RoundDown / Even / Ceil) live
on the HIP path and are exposed via :class:`aiter.ops.quant.MxScaleRoundMode`.

See ``csrc/kernels/quant.md`` "Cross-Stack Mode Alignment Reference" for the
full table mapping these to PyTorch torchao / NV Triton / DSv4 / FlashInfer
/ AMD Quark naming, and ``aiter/utility/fp4_utils.py`` for the CPU torch
reference implementations.
"""

from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl.expr.arith import CmpIPredicate


def emit_fp4_e8m0_scale_round_up(local_max):
    """FP4 (E2M1) NV ROUND_UP block-scale: ``ceil_pow2(amax / 6)``.

    Industry default used by NV Triton, DeepSeek-V4 Pro, FlashInfer, and
    PyTorch torchao ``RCEIL`` -- guarantees ``scale * 6 >= amax`` so the FP4
    max-normal (6.0) can always represent the group's amax (0% max-value
    clipping). Bit-for-bit equivalent to:

    - HIP ``aiter::fp4_f32_to_e8m0_scale`` (``csrc/include/fp4_quant_utils.h``)
    - Python ref ``aiter.utility.fp4_utils.f32_to_e8m0_rceil(amax, max_pos=6)``

    Args:
        local_max: f32 IR value, the (warp-reduced) ``max(|x|)`` of a 32-elem
            block. Caller is responsible for the per-block reduction.

    Returns:
        e8m0_biased: i32 IR value in the range ``[0, 0xFF]``. The caller
        derives ``quant_scale = (254 - e8m0_biased) << 23`` (bitcast to f32)
        for the multiplicative quant scale, and stores ``e8m0_biased`` as a
        ``uint8`` in the per-block scale tensor.
    """
    c0_i32 = arith.constant(0, type=T.i32)
    c1_i32 = arith.constant(1, type=T.i32)
    c23_i32 = arith.constant(23, type=T.i32)
    c0xFF_i32 = arith.constant(0xFF, type=T.i32)  # E8M0 exponent mask
    c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=T.i32)  # f32 mantissa mask
    # 1.0f / 6.0f = 0x3E2AAAAB (round-to-nearest f32 bit pattern). Multiplying
    # by this is fp32-cheaper than dividing by 6.0f and matches the HIP
    # helper's constant.
    c0x3E2AAAAB_i32 = arith.constant(0x3E2AAAAB, type=T.i32)

    inv6_f32 = c0x3E2AAAAB_i32.bitcast(T.f32)
    amax_div6 = local_max * inv6_f32
    amax_div6_i32 = amax_div6.bitcast(T.i32)
    mantissa = amax_div6_i32 & c0x7FFFFF_i32
    exp_field_raw = (amax_div6_i32 >> c23_i32) & c0xFF_i32
    # ceil-to-power-of-2 == "if any mantissa bit is set, bump exponent +1".
    mant_nonzero = arith.cmpi(CmpIPredicate.ne, mantissa, c0_i32)
    exp_field = arith.select(
        mant_nonzero,
        exp_field_raw + c1_i32,
        exp_field_raw,
    )
    return arith.maxsi(exp_field, c0_i32)


def emit_fp8_e8m0_scale_round_up_legacy(local_max, *, headroom_i32):
    """Legacy aiter Group D FP8 e8m0 scale: ``round_pow2_1.5(amax) / 2^headroom``.

    Round-half-up to nearest power of two: if the f32 mantissa MSB is set,
    bump the exponent by 1; otherwise drop the mantissa. This is the
    pre-PR aiter behaviour for the FP8 e4m3 activation path. **Not** the
    OCP floor that ``aiter.utility.fp4_utils.f32_to_e8m0`` Python helper now
    does -- the FP8 path was deliberately *not* migrated to NV ROUND_UP in
    this PR (see ``csrc/kernels/quant.md`` for the rationale).

    Args:
        local_max: f32 IR value, the (warp-reduced) ``max(|x|)`` of the block.
        headroom_i32: i32 IR value = ``log2(max_pow2(target_dtype))``. For
            FP8 e4m3 with ``max_pow2 = 256 = 2^8`` use ``headroom_i32 = 8``.
            (Caller-provided so this stays dtype-agnostic.)

    Returns:
        e8m0_biased: i32 IR value in the range ``[0, 0xFF]``.
    """
    c0_i32 = arith.constant(0, type=T.i32)
    c23_i32 = arith.constant(23, type=T.i32)
    c0x400000_i32 = arith.constant(0x400000, type=T.i32)  # 0.5 * 2^23
    c0xFF800000_i32 = arith.constant(0xFF800000, type=T.i32)  # sign+exp mask

    max_i32 = local_max.bitcast(T.i32)
    max_rounded = (max_i32 + c0x400000_i32) & c0xFF800000_i32
    exp_field = max_rounded >> c23_i32
    return arith.maxsi(exp_field - headroom_i32, c0_i32)


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
