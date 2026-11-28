# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-element error bounds for kernels checked against an aligned reference.

A reference is *aligned* when it consumes exactly the operand values the kernel
consumes (same quantized codes, same scales). The kernel and the reference then
differ only by floating-point rounding, which can be bounded element by element
from the operands, instead of being covered by a hand-picked rtol/atol.
"""

import logging
import math
from typing import NamedTuple

import torch

logger = logging.getLogger("aiter")

# Higham & Mary, "A New Approach to Probabilistic Rounding Error Analysis",
# SIAM J. Sci. Comput. 41(5), 2019, Thm 3.1: if rounding errors are independent
# and mean-zero, a length-K inner product has |error| <= g * sum|x_k * w_k| with
# g = exp(lam * sqrt(K) * u + K * u**2 / (1 - u)) - 1, with probability at least
# 1 - 2K * exp(-lam**2 * (1 - u)**2 / 2). lam = 10 makes the failure probability
# below 1e-16 per inner product for K <= 2**16. The model fails for constant
# operands and for long nonnegative sums whose terms drop below half an ulp of
# the partial sum (paper sec. 4.2: from K ~ 1e6 in fp32).
_LAMBDA = 10.0

# Hoeffding: independent, mean-zero errors e_i with |e_i| <= bound_i satisfy
# |sum_i e_i * sign(ref_i)| <= sqrt(2 ln(2 / delta) * sum_i bound_i**2) with
# probability at least 1 - delta. Errors that may share one sign in every
# element (a per-tensor scale rounded once) add their full sum on top. A kernel
# whose outputs are all off by the same relative error s moves the sum by
# s * sum_i |ref_i|.
_BIAS_DELTA = 1e-16


class ErrorBound(NamedTuple):
    total: torch.Tensor  # allowed |out - ref| per element
    common: torch.Tensor  # the part of total that may have one sign everywhere


def unit_roundoff(dtype: torch.dtype) -> float:
    """Unit roundoff u = 2**-(mantissa bits + 1) of a floating-point dtype."""
    return torch.finfo(dtype).eps / 2


def output_rounding(ref: torch.Tensor) -> torch.Tensor:
    """The part of a bound spent on rounding kernel and reference to ref.dtype.

    u_out per result in the normal range, half the smallest subnormal per
    result below it.
    """
    u_out = unit_roundoff(ref.dtype)
    tiny = torch.finfo(ref.dtype).tiny * torch.finfo(ref.dtype).eps
    return 2 * u_out / (1 - u_out) * ref.abs().float() + tiny


def gemm_error_bound(x: torch.Tensor, w: torch.Tensor, ref: torch.Tensor) -> ErrorBound:
    """Allowed |out - ref| for each element of an fp8 GEMM out = x @ w.T.

    x is (M, K) and w is (N, K), the dequantized operands the reference
    consumes; ref is the reference output after its cast to the output dtype.
    Assumes the kernel multiplies the fp8 codes exactly, accumulates and
    applies the scales in fp32, and rounds to nearest into ref.dtype. The two
    results then differ by

    * fp32 accumulation, in a different order, on each side (the reference's
      product roundings fall inside the same Thm 3.1 constant),
    * two fp32 roundings per side applying the scales,
    * rounding each result to ref.dtype.

    The first two scale with the operand mass sum_k |x_k * w_k|, not with
    |ref|: a cancelling sum keeps the rounding error of its large terms. The
    scale roundings are shared by every element using that scale, so they are
    also returned as the common part.
    """
    k = x.shape[-1]
    u_acc = unit_roundoff(torch.float32)
    u_out = unit_roundoff(ref.dtype)
    mass = x.abs().float() @ w.abs().float().t()
    both_sides = (1 + u_out) * 2 * mass
    accumulate = math.expm1(_LAMBDA * math.sqrt(k) * u_acc + k * u_acc**2 / (1 - u_acc))
    common = 2 * u_acc / (1 - 2 * u_acc) * both_sides
    return ErrorBound(accumulate * both_sides + common + output_rounding(ref), common)


def check_error_bound(
    ref: torch.Tensor,
    out: torch.Tensor,
    bound: "torch.Tensor | ErrorBound",
    msg: str = "",
) -> float:
    """Compare out to ref element by element against bound, then in aggregate.

    Fails if any |out - ref| exceeds its bound, or if the signed errors share
    a direction more than independent rounding allows (a scale or rounding
    bias too small to show in any one element). Returns the larger of the two
    ratios; a value above 1 raises AssertionError. Elements equal to ref
    (including equal infinities) pass; NaN in out fails. A plain tensor bound
    has no common part.
    """
    bound, common = bound if isinstance(bound, ErrorBound) else (bound, None)
    assert (
        out.shape == ref.shape == bound.shape
    ), f"{msg}shapes differ: out {out.shape}, ref {ref.shape}, bound {bound.shape}"
    if ref.numel() == 0:
        return 0.0
    signed = out.float() - ref.float()
    signed.masked_fill_(out == ref, 0)
    ratio = (signed.abs() / bound).nan_to_num(nan=math.inf, posinf=math.inf)
    flat = int(ratio.argmax())
    worst = ratio.flatten()[flat].item()
    idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), ratio.shape))
    over = int((ratio > 1).sum())
    ref_w = ref.flatten()[flat : flat + 1]
    detail = (
        f"max err/bound {worst:.3g} at {idx}: "
        f"ref {ref_w.item():.6g}, out {out.flatten()[flat].item():.6g}, "
        f"bound {bound.flatten()[flat].item():.3g} "
        f"(output rounding {output_rounding(ref_w).item():.3g}); "
        f"{over} of {ratio.numel()} elements over"
    )
    if worst > 1:
        logger.error(f"{msg}[error bound \033[31mfailed!\033[0m] {detail}")
        raise AssertionError(f"{msg}{detail}")

    drift = (signed.double() * ref.double().sign()).sum().item()
    spread = (bound.double() ** 2).sum().item()
    limit = math.sqrt(2 * math.log(2 / _BIAS_DELTA) * spread)
    if common is not None:
        limit += common.double().sum().item()
    bias = abs(drift) / limit if limit > 0 else 0.0
    mass = ref.double().abs().sum().item()
    detail += (
        f"; bias {bias:.3g} of limit "
        f"(relative bias {drift / mass if mass > 0 else 0.0:+.2g})"
    )
    if bias > 1:
        logger.error(f"{msg}[error bound \033[31mfailed!\033[0m] {detail}")
        raise AssertionError(f"{msg}{detail}")
    logger.info(f"{msg}[error bound \033[32mpassed~\033[0m] {detail}")
    return max(worst, bias)
