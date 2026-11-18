# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The A6W6 GEMM as a fused-producer host (``gemm_a6w6_4wave_host``).

The plain projection is checked against the standalone A6W6 GEMM bit for bit. The bias,
Q/K prep and maxima epilogues are checked against an FP32 PyTorch reference of the
dequantized MXFP6 operands, and the persistent grid against one block per workgroup.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.gemm_op_a6w6 import _ceil, dequant_mxfp6_torch, quant_mxfp6_torch

HEAD = 128
EPS = 1e-6


def _is_gfx950():
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="the A6W6 host needs gfx950")


def _case(m, n, k, bias, hadamard, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
    gamma = 1 + 0.1 * torch.randn(n, device="cuda", generator=gen)
    angles = torch.rand((m, 64), device="cuda", generator=gen) * 6.28
    return {
        "x": x,
        "w": w,
        "bias": torch.randn(n, device="cuda", generator=gen) * 0.1 if bias else None,
        # The caller folds the Hadamard's 128^-0.5 into the norm weights.
        "gamma": gamma * HEAD**-0.5 if hadamard else gamma,
        "cos": angles.cos().contiguous(),
        "sin": angles.sin().contiguous(),
    }


def _dequantized(t):
    """The FP32 values the MFMA multiplies: Hadamard-rotated, MXFP6-rounded, K padded.
    The rotation cancels in the product."""
    t = F.pad(t, (0, _ceil(t.shape[1], 128) - t.shape[1]))
    return dequant_mxfp6_torch(*quant_mxfp6_torch(t))


def _hadamard(x):
    """The epilogue's per-head Hadamard: three butterflies over a lane's 8 channels, then
    four across the head's 16 lanes, pairing lanes i ^ 1, i ^ 2, i ^ 7 and i ^ 8."""
    v = x.reshape(*x.shape[:-1], 16, 8)
    k = torch.arange(8, device=x.device)
    for shift in (1, 2, 4):
        partner = v[..., k ^ shift]
        v = torch.where((k & shift).bool(), partner - v, v + partner)
    lane = torch.arange(16, device=x.device)
    for mask, bit in ((1, 1), (2, 2), (7, 4), (8, 8)):
        sign = torch.where((lane & bit).bool(), -1.0, 1.0)[:, None]
        v = v[..., lane ^ mask, :] + sign * v
    return v.reshape(x.shape)


def _reference(case, spec=None, bias=False):
    """Stored projection, per-head sums of squares and per-head maxima of the stored
    output."""
    acc = F.linear(_dequantized(case["x"]), _dequantized(case["w"]))
    if (spec.bias if spec is not None else bias) and case["bias"] is not None:
        acc = acc + case["bias"]
    m, n = acc.shape
    y = acc.view(m, n // HEAD, HEAD)
    sq = (y * y).sum(-1)
    if spec is not None:
        if spec.norm == "rms_per_head":
            y = y * torch.rsqrt(sq / HEAD + EPS)[..., None]
        if spec.gammas:
            y = y * case["gamma"].view(n // HEAD, HEAD)
        cos, sin = case["cos"][:, None, :], case["sin"][:, None, :]
        if spec.rope == "interleaved":
            even, odd = y[..., 0::2], y[..., 1::2]
            y = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
            y = y.flatten(-2)
        elif spec.rope == "rotate_half":
            low, high = y[..., :64], y[..., 64:]
            y = torch.cat((low * cos - high * sin, high * cos + low * sin), -1)
        if spec.hadamard:
            y = _hadamard(y)
    stored = y.reshape(m, n).to(torch.bfloat16)
    maxima = stored.float().abs().view(m, n // HEAD, HEAD).amax(-1)
    return stored, sq, maxima


def _launch(case, spec=None, *, bias=False, amax=False, num_cu=0):
    """The host GEMM into NaN-filled outputs, so unwritten elements fail."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a6w6_4wave_host import (
        compile_a6w6_host_4w,
        host_weight_order,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    x, w = case["x"], case["w"]
    (m, k), n = x.shape, w.shape[0]
    a, a_scale = aiter.quant_mxfp6_gemm(x)
    b, b_scale = aiter.quant_mxfp6_gemm(w[host_weight_order(n).to(w.device)])
    with_bias = spec.bias if spec is not None else bias
    columns = [torch.ones(n, device="cuda")]
    if with_bias:
        columns.append(case["bias"])
    if spec is not None and spec.gammas:
        columns.append(case["gamma"])
    columns = torch.cat(columns)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    partials = torch.full((2, m, n // HEAD), float("nan"), device="cuda")
    kernel = compile_a6w6_host_4w(
        K=_ceil(k, 128), num_cu=num_cu, prep=spec, bias=bias, amax=amax, eps=EPS
    )
    _run_compiled(
        kernel,
        a.view(-1),
        b.view(-1),
        out.view(-1),
        a_scale.view(-1),
        b_scale.view(-1),
        columns,
        case["cos"].view(-1),
        case["sin"].view(-1),
        partials.view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()
    return out, partials


def _close(expected, actual, rtol, label, outliers=0.0):
    """``actual`` within ``rtol`` of ``expected``, but for a fraction ``outliers``, and
    within 1e-3 RMS. The kernel and PyTorch add the K products in different orders: an
    FP32 result near a BF16 rounding boundary, or small after the Hadamard's sums,
    lands a step or two away."""
    e, a = expected.float(), actual.float()
    bad = ~torch.isclose(a, e, rtol=rtol, atol=1e-6)
    assert (
        bad.float().mean().item() <= outliers
    ), f"{label}: {int(bad.sum())} of {bad.numel()} outside rtol {rtol}"
    rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    assert rms <= 1e-3, f"{label}: RMS error {rms:.3g}"


@pytest.mark.parametrize(
    "shape",
    [(256, 256, 128), (300, 512, 1000), (1024, 768, 1280), (9472, 15360, 5120)],
)
@pytest.mark.parametrize("persistent", [False, True])
@torch.no_grad()
def test_plain_matches_standalone_a6w6(shape, persistent):
    m, n, k = shape
    case = _case(m, n, k, False, False, seed=m + n + k)
    a, a_scale = aiter.quant_mxfp6_gemm(case["x"])
    b, b_scale = aiter.quant_mxfp6_gemm(case["w"])
    expected = aiter.gemm_a6w6(a, b, a_scale, b_scale, m, n, k)
    out, _ = _launch(case, num_cu=get_cu_num() if persistent else 0)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


PREP_CASES = [
    (257, 512, 512, norm, rope, hadamard, True, False)
    for norm, rope, hadamard in itertools.product(
        ("none", "rms_per_head", "rms_across_heads"),
        ("none", "interleaved", "rotate_half"),
        (True, False),
    )
] + [
    # Without a bias, then persistent at Wan's 40-head and FLUX.3's 24-head projections.
    (257, 512, 512, "rms_across_heads", "interleaved", True, False, False),
    (257, 512, 512, "rms_per_head", "rotate_half", True, False, False),
    (257, 512, 512, "rms_per_head", "interleaved", True, False, False),
    (1184, 5120, 5120, "rms_across_heads", "interleaved", True, True, True),
    (1184, 5120, 5120, "rms_per_head", "rotate_half", True, True, True),
    (576, 3072, 3072, "rms_per_head", "interleaved", True, False, True),
]


@pytest.mark.parametrize("m,n,k,norm,rope,hadamard,bias,persistent", PREP_CASES)
@torch.no_grad()
def test_prep_matches_pytorch(m, n, k, norm, rope, hadamard, bias, persistent):
    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import PrepSpec

    spec = PrepSpec(norm, rope, bias, hadamard)
    case = _case(m, n, k, bias, hadamard, seed=m + n + k)
    stored, sq, maxima = _reference(case, spec)
    out, partials = _launch(case, spec, num_cu=get_cu_num() if persistent else 0)
    _close(stored, out, 2**-7, f"{spec}: output", outliers=1e-3)
    _close(maxima, partials[1], 2**-7, f"{spec}: head maxima", outliers=1e-3)
    if spec.norm == "rms_across_heads":
        _close(sq, partials[0], 1e-3, f"{spec}: head sums of squares", outliers=1e-3)


@pytest.mark.parametrize("bias,amax", [(True, False), (False, True), (True, True)])
@torch.no_grad()
def test_bias_and_maxima_match_pytorch(bias, amax):
    m, n, k = 300, 512, 1000
    case = _case(m, n, k, bias, False, seed=11)
    stored, _, maxima = _reference(case, bias=bias)
    out, partials = _launch(case, bias=bias, amax=amax)
    _close(stored, out, 2**-7, "output", outliers=1e-3)
    if amax:
        _close(maxima, partials[1], 2**-7, "head maxima", outliers=1e-3)


@torch.no_grad()
def test_persistent_matches_one_block_per_workgroup():
    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import PrepSpec

    spec = PrepSpec("rms_across_heads", "interleaved", True, True)
    case = _case(1184, 5120, 5120, True, True, seed=3)
    one = _launch(case, spec)
    persistent = _launch(case, spec, num_cu=get_cu_num())
    for a, b in zip(one, persistent):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
