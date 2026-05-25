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
``csrc/include/fp4_quant_utils.h`` and ``aiter.ops.quant.MxScaleRoundMode``).

See ``csrc/kernels/quant.md`` "Cross-Stack Mode Alignment Reference" for the
full table mapping these to PyTorch torchao / NV Triton / DSv4 / FlashInfer
/ AMD Quark naming, and ``aiter/utility/fp4_utils.py`` for the CPU torch
reference implementations.
"""

from enum import IntEnum

from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl.expr.arith import CmpIPredicate


class MxScaleRoundMode(IntEnum):
    """E8M0 block-scale rounding mode for any MX format.

    Re-defined here (rather than imported from ``aiter.ops.quant``) so the
    FlyDSL kernel build path stays self-contained -- ``aiter.ops.quant``
    transitively pulls in heavy dependencies. Values **must** stay 1:1
    with the HIP-side ``enum class MxScaleRoundMode`` in
    ``csrc/include/fp4_quant_utils.h`` and the Python
    ``aiter.ops.quant.MxScaleRoundMode`` ``IntEnum``.
    """

    RoundDown = 0  # torchao FLOOR  (OCP / NV ROUND_DOWN)
    RoundUp = 1  # torchao RCEIL  (NV ROUND_UP / DSv4 Pro / FlashInfer) -- DEFAULT
    Even = 2  # torchao EVEN   (Quark EVEN)
    Ceil = 3  # torchao CEIL


class MxDtype(IntEnum):
    """Element dtype tag passed to :func:`emit_mx_e8m0_scale`.

    Each tag selects a small bundle of dtype-specific constants
    (``target_max_pow2``, ``max_pos`` reciprocal, ``mbits``) used by the
    four rounding modes; the formulas themselves are dtype-agnostic.
    """

    FP4_E2M1 = 0  # max_pos = 6.0,    target_max_pow2 = 2 (= log2(4)), mbits = 1
    FP8_E4M3 = 1  # max_pos = 448.0,  target_max_pow2 = 8 (= log2(256)), mbits = 3
    # FP8_E5M2 / FP6_* -- reserved; add when a FlyDSL caller needs them.


# Per-MX-dtype constants. Tuple form: (target_max_pow2, max_pos_inv_f32_bits, mbits)
# - target_max_pow2 = log2(largest pow2 <= max_normal(dtype))
# - max_pos_inv_f32_bits = bit pattern of fp32(1.0 / max_normal(dtype))
# - mbits = mantissa bits of the target dtype (used only by EVEN mode)
_DTYPE_CFG = {
    MxDtype.FP4_E2M1: (2, 0x3E2AAAAB, 1),  # 1/6.0 ~= 0.16666667
    MxDtype.FP8_E4M3: (8, 0x3B125641, 3),  # 1/448.0 ~= 0.00223214
}


def emit_mx_e8m0_scale(
    local_max,
    *,
    mode: MxScaleRoundMode = MxScaleRoundMode.RoundUp,
    dtype: MxDtype = MxDtype.FP4_E2M1,
):
    """Emit IR computing the E8M0 block scale for an MX format.

    Mirrors PyTorch torchao ``to_mx(scaling_mode=..., elem_dtype=...)``
    semantics 1:1: the four rounding formulas (FLOOR / RCEIL / CEIL / EVEN)
    are identical across MX dtypes, and the ``mode`` argument selects which
    formula to emit. ``dtype`` only chooses ``target_max_pow2`` /
    ``max_pos`` / ``mbits`` constants. Both ``mode`` and ``dtype`` are
    Python compile-time values (FlyDSL kernel parameters), so the branch
    is resolved before any IR is emitted -- there is no runtime cost.

    Modes (with cross-stack equivalences; see ``csrc/kernels/quant.md``):

    - ``RoundDown`` / torchao ``FLOOR``  -- ``floor_pow2(amax) / 2^target_max_pow2``
      (OCP MX spec; pre-PR aiter Python ref default).
    - ``RoundUp``   / torchao ``RCEIL``  -- ``ceil_pow2(amax / max_pos)``.
      *Default*: matches NV ROUND_UP / DSv4 Pro / FlashInfer / NV cuBLAS
      ``cvt.rp.satfinite.ue8m0x2.f32`` / DeepSeek-V3.1 ``scale_fmt=ue8m0``.
    - ``Even``      / torchao ``EVEN``   -- ``floor_pow2(round_pow2_special(amax))
      / 2^target_max_pow2`` with ``val_to_add = 1 << (23 - mbits - 1)``
      (AMD Quark default; SGLang-on-ROCm dynamic; vLLM Quark QDQ).
    - ``Ceil``      / torchao ``CEIL``   -- ``ceil_pow2(amax) / 2^target_max_pow2``
      (torchao only; coarser grid than RCEIL, no Quark/NV name).

    Args:
        local_max: f32 IR value, the (warp-reduced) ``max(|x|)`` of one
            block. Caller is responsible for the per-block reduction.
        mode: :class:`MxScaleRoundMode`. Default ``RoundUp`` (industry
            consensus for MXFP4 and MXFP8).
        dtype: :class:`MxDtype`. Default ``FP4_E2M1``.

    Returns:
        e8m0_biased: i32 IR value in the range ``[0, 0xFF]``. The caller
        derives ``quant_scale = (254 - e8m0_biased) << 23`` (bitcast to
        f32) for the multiplicative quant scale, and stores
        ``e8m0_biased`` as a ``uint8`` in the per-block scale tensor.
    """
    if dtype not in _DTYPE_CFG:
        raise ValueError(
            f"emit_mx_e8m0_scale: unsupported dtype {dtype!r}; "
            f"supported: {list(_DTYPE_CFG)}"
        )
    target_max_pow2, max_pos_inv_bits, mbits = _DTYPE_CFG[dtype]

    c0_i32 = arith.constant(0, type=T.i32)
    c1_i32 = arith.constant(1, type=T.i32)
    c23_i32 = arith.constant(23, type=T.i32)
    c0xFF_i32 = arith.constant(0xFF, type=T.i32)  # E8M0 exponent mask
    c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=T.i32)  # f32 mantissa mask
    target_max_pow2_i32 = arith.constant(target_max_pow2, type=T.i32)

    if mode == MxScaleRoundMode.RoundUp:
        # ceil_pow2(amax / max_pos): multiply by reciprocal of max_pos to get
        # the working value, then bump the exponent if any mantissa bit is
        # set. Bit-equivalent to HIP ``aiter::fp4_f32_to_e8m0_scale`` (when
        # dtype=FP4_E2M1) and to PyTorch torchao ``_to_mx_rceil`` (modulo
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
        return arith.maxsi(exp_field, c0_i32)

    if mode == MxScaleRoundMode.RoundDown:
        # floor_pow2(amax) / 2^target_max_pow2: drop the f32 mantissa, then
        # subtract target_max_pow2 from the biased exponent.
        amax_i32 = local_max.bitcast(T.i32)
        biased_exp = (amax_i32 >> c23_i32) & c0xFF_i32
        return arith.maxsi(biased_exp - target_max_pow2_i32, c0_i32)

    if mode == MxScaleRoundMode.Ceil:
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
        return arith.maxsi(biased_exp_bumped - target_max_pow2_i32, c0_i32)

    if mode == MxScaleRoundMode.Even:
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
        return arith.maxsi(biased_exp - target_max_pow2_i32, c0_i32)

    raise ValueError(
        f"emit_mx_e8m0_scale: unknown MxScaleRoundMode {mode!r} "
        f"(expected one of {list(MxScaleRoundMode)})"
    )


def emit_fp4_e8m0_scale_round_up(local_max):
    """FP4 NV ROUND_UP block-scale -- thin alias for callers / git history.

    Equivalent to::

        emit_mx_e8m0_scale(local_max,
                           mode=MxScaleRoundMode.RoundUp,
                           dtype=MxDtype.FP4_E2M1)

    See :func:`emit_mx_e8m0_scale` for the full mode/dtype semantics. New
    code should prefer the generic entry point so the choice of mode and
    dtype is explicit at the call site.
    """
    return emit_mx_e8m0_scale(
        local_max,
        mode=MxScaleRoundMode.RoundUp,
        dtype=MxDtype.FP4_E2M1,
    )


def emit_fp8_e8m0_scale_round_up_legacy(local_max, *, headroom_i32):
    """Legacy aiter Group D FP8 e8m0 scale: ``round_pow2_1.5(amax) / 2^headroom``.

    **Not** part of the PyTorch torchao 4-mode family. This is an aiter-
    specific historical formula (round-half-up on the f32 mantissa MSB)
    used by the FlyDSL FP8 activation path pre-this-PR. The FP8 path was
    deliberately not migrated to NV ROUND_UP in this PR to keep a8w4
    sweep results stable; aligning it to ``emit_mx_e8m0_scale(mode=RoundUp,
    dtype=MxDtype.FP8_E4M3)`` (which matches NV TransformerEngine MXFP8 /
    DeepSeek-V3.1 ``scale_fmt=ue8m0``) is a follow-up PR.

    Numerically this is equivalent to ``emit_mx_e8m0_scale(mode=Even, ...)``
    with a non-physical ``mbits=0`` and a runtime ``headroom_i32``; the
    standalone helper is kept for clarity.

    Args:
        local_max: f32 IR value, the (warp-reduced) ``max(|x|)`` of the block.
        headroom_i32: i32 IR value = ``log2(max_pow2(target_dtype))``. For
            FP8 e4m3 with ``max_pow2 = 256 = 2^8`` use ``headroom_i32 = 8``.

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
