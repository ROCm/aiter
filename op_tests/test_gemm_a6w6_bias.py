# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Contract for the optional bias folded into the A6W6 store epilogue.

The perf side of ``gemm_a6w6`` lives in ``test_gemm_a6w6.py``; what needs asserting here is
narrower and does not show up as an accuracy number. The bias is passed to the asm through
two kernarg slots the fp6 kernel used to ignore -- the pointer in ``ptr_C`` and a **byte
length** in ``stride_C1`` -- and the kernel bounds-checks the load against that length
instead of branching. Three consequences of that design are worth pinning:

- Length zero has to be exactly the old unbiased kernel, because that is what almost every
  MXFP6 GEMM in a model runs. One code object serves both paths, so "no bias" is not a
  separate binary that could be verified by inspection.
- A bias shorter than the padded N has to be safe. ``gemm_a6w6`` rounds N up to a 256-wide
  tile and slices the result back, so a length-N bias is *deliberately* short of the launch
  width and the out-of-range columns must read zero rather than reading off the end.
- A dropped bias would look like a correct unbiased GEMM rather than an error, which no
  allclose against a biased reference would catch cleanly. So it is checked against the
  unbiased result directly.
"""

import pytest
import torch

import aiter
import aiter.ops.gemm_op_a6w6 as mxfp6
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950" or not mxfp6._HAS_TRITON,
    reason="gfx950 hardware FP6 conversion and Triton are required",
)

# One aligned shape and one that pads in all three dimensions, which is the case where the
# bias is shorter than the width the kernel actually launches.
SHAPES = [(256, 512, 256), (257, 513, 129)]


def _operands(m: int, n: int, k: int):
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    xq, xs = mxfp6.quant_mxfp6_gemm(x)
    wq, ws = mxfp6.quant_mxfp6_gemm(w)
    return xq, wq, xs, ws


@pytest.mark.parametrize("m, n, k", SHAPES)
def test_bias_none_is_the_unbiased_kernel(m: int, n: int, k: int):
    """``bias=None`` must be bitwise identical to not passing one at all.

    Bitwise and not allclose: this is the path taken by every MXFP6 GEMM that has no bias,
    and the claim being made about the swapped code objects is that they are byte-for-byte
    the old behaviour there, not merely close to it.
    """
    xq, wq, xs, ws = _operands(m, n, k)
    want = aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k)
    got = aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k, bias=None)
    assert torch.equal(want, got)


@pytest.mark.parametrize("m, n, k", SHAPES)
def test_zero_bias_does_not_shift_the_result(m: int, n: int, k: int):
    """A real (non-null) pointer to zeros must also leave the result untouched.

    This separates the two halves of the design: the test above exercises the null pointer,
    this one exercises the loaded-and-added path with a value that cannot move the sum. It
    is the check that catches an epilogue that adds the wrong lane, or that mixes up the
    bf16-to-f32 conversion, since either would perturb a zero bias.
    """
    xq, wq, xs, ws = _operands(m, n, k)
    zeros = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    want = aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k)
    got = aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k, bias=zeros)
    assert torch.equal(want, got)


@pytest.mark.parametrize("m, n, k", SHAPES)
def test_bias_is_added_per_column(m: int, n: int, k: int):
    """The epilogue result must match adding the bias afterwards, to within the rounding.

    The two are not bitwise equal, and the difference is the point rather than a defect: the
    separate pass rounds the GEMM to bf16 and then adds in bf16, rounding twice, while the
    epilogue adds into the f32 accumulator and rounds once. They can therefore differ by the
    last bf16 step, with the epilogue the more accurate of the two.

    The bound is one ulp of the *pre-add* magnitudes, not of the result. Where the bias
    nearly cancels the accumulator the result is small and its own ulp is tiny, while the
    quantity actually discarded by the early rounding is set by the operands' size -- so a
    bound on the result's ulp would fail on exactly the elements it should permit.
    """
    xq, wq, xs, ws = _operands(m, n, k)
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")

    unbiased = aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k)
    separate = unbiased + bias
    fused = aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k, bias=bias)

    # A dropped bias is the failure this file exists to catch, and it would pass any
    # comparison against the unbiased result being "close". Assert the shift happened.
    assert not torch.equal(fused, unbiased)

    def ulp(t: torch.Tensor) -> torch.Tensor:
        exponent = torch.floor(torch.log2(t.float().abs().clamp_min(1e-30)))
        return torch.exp2(exponent - 7.0)  # bf16 stores 7 mantissa bits

    diff = (fused.float() - separate.float()).abs()
    assert bool((diff <= ulp(unbiased) + ulp(separate)).all()), (
        f"max deviation {diff.max().item():.3e} exceeds the double-rounding bound"
    )


def test_bias_length_must_match_n():
    """Rejected in Python, because the kernel cannot tell a wrong length from a short one.

    The asm takes a byte length and treats anything past it as zero, which is what makes the
    padded-N case safe -- and is exactly why a bias that is genuinely the wrong length would
    be silently zero-extended instead of faulting.
    """
    m, n, k = SHAPES[0]
    xq, wq, xs, ws = _operands(m, n, k)
    for length in (n - 1, n + 1):
        bias = torch.zeros(length, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match=rf"bias must have length N={n}"):
            aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k, bias=bias)


def test_bias_must_be_bf16():
    """The epilogue reads 16-bit elements and widens them by shifting, so dtype is load-bearing.

    An fp32 bias would be read as two bf16 lanes per element, which is not a tolerance
    question but garbage, and the stride of the load makes it unrecoverable downstream.
    """
    m, n, k = SHAPES[0]
    xq, wq, xs, ws = _operands(m, n, k)
    bias = torch.zeros(n, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="bias must be BFloat16"):
        aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k, bias=bias)


def test_bias_must_be_contiguous_1d():
    """A strided bias would be read as if packed, since the asm gets a pointer and a length."""
    m, n, k = SHAPES[0]
    xq, wq, xs, ws = _operands(m, n, k)
    strided = torch.zeros(2 * n, dtype=torch.bfloat16, device="cuda")[::2]
    assert strided.numel() == n and not strided.is_contiguous()
    with pytest.raises(RuntimeError, match="bias must be a contiguous 1D tensor"):
        aiter.gemm_a6w6(xq, wq, xs, ws, m, n, k, bias=strided)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
