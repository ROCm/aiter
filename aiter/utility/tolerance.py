# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Derive checkAllclose tolerances from the dtype error model.

The tolerance for comparing a kernel against a higher-precision reference is
determined by three things, none of which are constants across a test file:

  * the precision the math is actually carried out in (the *weakest* link in a
    quantized pipeline, not the output dtype),
  * the accumulator precision and the reduction length ``K``,
  * the magnitude of the values being compared.

This mirrors Composable Kernel's ``get_relative_threshold`` /
``get_absolute_threshold``, which is the existing mantissa-derived tolerance in
the ROCm stack, and extends it to the FP8/FP4 formats AITER uses.

Scope: float compute dtypes only. Integer quantization containers (int8, and
the int4 weight path) are deliberately not modelled here. Their error is set by
the scale the values were mapped into rather than by a mantissa width, and a
single rounding of that step under-charges the error measured on hardware by
more than an order of magnitude once cancellation compounds across a reduction.
Since signal and rounding error both grow as ``sqrt(K)`` in a random-sign dot
product, the shortfall is not reduction depth and no closed form in ``K``
recovers it. Those paths keep their existing hand-set tolerances.

Typical use in an op_test::

    from aiter.utility.tolerance import tolerance_for

    rtol, atol = tolerance_for(ref, compute_dtype=quant_dtype, num_accumulations=K)
    checkAllclose(ref, out, rtol=rtol, atol=atol, msg=msg)
"""

import math

import torch

__all__ = [
    "derive_tolerance",
    "tolerance_for",
    "unit_roundoff",
]

# Explicit stored mantissa bits, excluding the implicit leading 1. The unit
# roundoff of a float format is u = 2**-mant * 0.5 -- half a ULP at magnitude 1.
_MANTISSA_BITS = {
    torch.float64: 52,
    torch.float32: 23,
    torch.float16: 10,
    torch.bfloat16: 7,
    torch.float8_e4m3fn: 3,
    torch.float8_e4m3fnuz: 3,
    torch.float8_e5m2: 2,
    torch.float8_e5m2fnuz: 2,
}

for _name, _bits in (("float4_e2m1fn_x2", 1), ("float8_e8m0fnu", 0)):
    _dtype = getattr(torch, _name, None)
    if _dtype is not None:
        _MANTISSA_BITS[_dtype] = _bits


def unit_roundoff(dtype):
    """Largest relative representation error of ``dtype``.

    Float formats: ``2**-mant * 0.5``.

    Integer formats return ``0.0``: an integer holds no rounding error of its
    own, which is the right answer for an index, a counter, or an exact
    integer-valued output. It is *not* the right answer for an integer used as
    a quantization container, where the error comes from the scale rather than
    from the dtype; see the module docstring.
    """
    if dtype in _MANTISSA_BITS:
        return math.ldexp(0.5, -_MANTISSA_BITS[dtype])
    if not dtype.is_floating_point:
        return 0.0
    raise ValueError(f"No error model for dtype {dtype!r}")


def derive_tolerance(
    compute_dtype,
    output_dtype,
    acc_dtype=torch.float32,
    num_accumulations=1,
    max_value=1.0,
    safety_factor=1.0,
):
    """Return ``(rtol, atol)`` for one comparison.

    Args:
        compute_dtype: float dtype the math is carried out in. For a quantized
            pipeline this is the quantized dtype, not the bf16 output -- the
            weakest link sets the error. Integer quantization containers are
            out of scope and are rejected; see the module docstring.
        output_dtype: dtype the result is stored as.
        acc_dtype: accumulator dtype. MFMA GEMM accumulates in fp32.
        num_accumulations: reduction length ``K``. Scales the accumulator term
            only; with an fp32 accumulator the quantization term dominates for
            any realistic ``K``.
        max_value: magnitude of the largest reference value, used to place
            ``atol`` in the right binary exponent band.
        safety_factor: multiplier >= 1 for datapaths that are not
            round-to-nearest deterministic, e.g. stochastic rounding or an
            autotuner that reorders the reduction.

    Returns:
        ``(rtol, atol)``, both non-negative. An exact (integer) pipeline
        returns ``(0.0, 0.0)``.

    Raises:
        ValueError: if ``compute_dtype`` is an integer quantization container.
    """
    if num_accumulations < 1:
        raise ValueError("num_accumulations must be >= 1")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1; it may only loosen")

    u_output = unit_roundoff(output_dtype)

    # An integer output holds no rounding error no matter what the pipeline
    # did on the way, so the comparison is bit-exact.
    if u_output == 0.0:
        return 0.0, 0.0

    if not compute_dtype.is_floating_point:
        raise ValueError(
            f"no validated error model for integer quantization container "
            f"{compute_dtype!r}; keep the hand-set tolerance for that path"
        )

    u_compute = unit_roundoff(compute_dtype)
    u_acc = unit_roundoff(acc_dtype)

    rtol = safety_factor * max(u_compute, u_output, u_acc * num_accumulations)

    if max_value == 0.0 or not math.isfinite(max_value):
        expo = 0
    else:
        expo = math.floor(math.log2(abs(max_value)))
    scale = math.ldexp(1.0, expo)

    # The output-cast term uses a full ULP rather than half: hardware and
    # software rounding of the same fp32 value to bf16 can legitimately land on
    # adjacent codes at a tie.
    atol = (
        safety_factor
        * scale
        * max(
            u_compute,
            u_output * 2.0,
            u_acc * num_accumulations,
        )
    )
    return rtol, atol


def tolerance_for(
    reference,
    compute_dtype,
    acc_dtype=torch.float32,
    num_accumulations=1,
    safety_factor=1.0,
):
    """:func:`derive_tolerance` with ``output_dtype`` and ``max_value`` read off
    the reference tensor."""
    finite = torch.isfinite(reference)
    if finite.any():
        max_value = float(reference[finite].abs().max().item())
    else:
        max_value = 1.0
    return derive_tolerance(
        compute_dtype=compute_dtype,
        output_dtype=reference.dtype,
        acc_dtype=acc_dtype,
        num_accumulations=num_accumulations,
        max_value=max_value,
        safety_factor=safety_factor,
    )
