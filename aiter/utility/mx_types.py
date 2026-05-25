# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single source of truth for MX-format scale rounding mode and dtype enums.

These enums are imported by:
  - ``aiter.ops.quant`` (Python user-facing quant ops)
  - ``aiter.utility.fp4_utils`` (CPU torch reference quantizers)
  - ``aiter.ops.flydsl.kernels.quant_utils`` (FlyDSL IR builders)

Defined in ``aiter.utility`` (which has no aiter-internal dependencies) so
every layer above can import without triggering a circular import. The
integer values match the HIP-side ``enum class MxScaleRoundMode`` in
``csrc/include/mx_quant_utils.h`` 1:1; whenever the C++ enum changes,
update this file in the same PR.

See ``csrc/kernels/quant.md`` "Cross-Stack Mode Alignment Reference" for
the full table mapping these to PyTorch torchao / NV Triton / DSv4 /
FlashInfer / AMD Quark naming.
"""

from enum import IntEnum


class MxScaleRoundMode(IntEnum):
    """E8M0 block-scale rounding mode for any MX format.

    Applies to the whole MX format family (mxfp4 / mxfp6 / mxfp8 / mxint8) --
    PyTorch torchao's ``ScaleCalculationMode`` is dtype-agnostic in the same
    way: the formulas (FLOOR / RCEIL / CEIL / EVEN) are identical, only the
    ``max_pos`` / ``max_pow2`` / ``mbits`` constants are dtype-specific
    (e.g. 6 vs 448 for FP4 vs FP8 e4m3).

    Names follow AMD Quark's ``RoundMode``; the equivalent torchao
    ``ScaleCalculationMode`` is shown in the comments. ``IntEnum`` values
    are interchangeable with bare ``int`` -- existing callers passing
    ``round_mode=1`` keep working unchanged (``MxScaleRoundMode.RoundUp ==
    1`` and ``isinstance(MxScaleRoundMode.RoundUp, int)`` are both
    ``True``).
    """

    RoundDown = 0  # torchao FLOOR  (OCP / NV ROUND_DOWN)
    RoundUp = 1  # torchao RCEIL  (NV ROUND_UP / DSv4 Pro / FlashInfer) -- DEFAULT
    Even = 2  # torchao EVEN   (Quark EVEN)
    Ceil = 3  # torchao CEIL   (no Quark / NV equivalent)


class MxDtype(IntEnum):
    """Element dtype tag selecting MX-format-specific constants.

    Used by ``f32_to_mx_e8m0_scale`` (CPU torch ref) and
    ``emit_mx_e8m0_scale`` (FlyDSL IR builder) to look up the
    dtype-specific ``target_max_pow2`` / ``max_pos`` / ``mbits`` constants
    that the four rounding modes need. Add new entries here when a caller
    needs FP6_* or MX-INT8 support.
    """

    FP4_E2M1 = 0  # max_pos = 6.0,    target_max_pow2 = 2 (= log2(4)),    mbits = 1
    FP8_E4M3 = 1  # max_pos = 448.0,  target_max_pow2 = 8 (= log2(256)),  mbits = 3
    # FP8_E5M2 / FP6_E2M3 / FP6_E3M2 / MX_INT8 -- reserved.
