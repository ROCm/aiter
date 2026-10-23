# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness oracle of the gfx942 GEMM, shared by its tests and its tuner."""

import torch

# f32 summation-order noise, as a fraction of the largest output.
ACCUM_TOL = 1e-5


def error_ratio(y: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst elementwise error of ``y`` against the f32 oracle ``ref``, in units
    of its tolerance: correct when <= 1.

    The tolerance is one bf16 ulp for bf16 output (none for fp32), plus
    ``ACCUM_TOL`` times the largest output. A correctly rounded bf16 result is
    within half an ulp; the other half lets a different summation order flip
    the rounding.
    """
    err = (y.float() - ref).abs()
    noise = ACCUM_TOL * ref.abs().max()
    if y.dtype == torch.bfloat16:
        # bf16 keeps 8 significant bits: one ulp is 2^-7 of the leading power of two.
        exponent = torch.floor(torch.log2(ref.abs().clamp_min(1e-30)))
        tol = torch.exp2(exponent - 7) + noise
    else:
        tol = noise
    return (err / tol).max().item()
