# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_qk_rmsnorm with fp8 per-group (32 / 64 / 128) E8M0 scales, against a torch
reference: its bf16 norms are within 1 bf16 ulp of an fp32 RMSNorm, and its scales, and
its fp8 on finite groups, are byte for byte the RoundUp E8M0 quant of its own bf16 norm
(scale ceil_pow2(amax / fp8 max), amax skipping NaN, capped at 2^127). Both also equal
per_group_quant_hip(scale_type=fp8_e8m0) of that norm, except its group-32 transposed
scale."""

import sys

import pytest
import torch

import aiter
from aiter import QuantType, dtypes
from aiter.ops.quant import per_group_quant_hip

# (q width, kv width): Kimi-K3 MLA q_lora 1536 / kv_lora 512, and widths across
# the group-quant dispatch (128: its fp8-only branch).
WIDTHS = [(1536, 512), (128, 128), (512, 512), (1024, 512), (4096, 512), (8192, 512)]
# decode and prefill row counts
ROWS = [1, 15, 135, 1009, 4096, 16384]
SCALE_DTYPES = [dtypes.fp8_e8m0, torch.uint8]
# per_1x128, and the smaller groups of the per_1x32 MXFP8 path (reduce threads 2-16)
GROUP_SIZES = [32, 64, 128]
FP8_MAX = torch.finfo(dtypes.fp8).max
EPS = 1e-6


def _inputs(m, n1, n2, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda")
    q = (r(m, n1) * 3).bfloat16()
    if m >= 2:
        q[0, :128] = 0  # an all-zero group
        q[1, 128:256] *= 2.0**20
    qw = r(n1).bfloat16()
    if n1 > 128:
        # an Inf group in every row: the 2^127 scale cap (a zero input there gives
        # NaN, which the amax skips)
        qw[-128:] = float("inf")
    return q, qw, (r(m, n2) * 3).bfloat16(), r(n2).bfloat16(), (r(m, n1) * 3).bfloat16()


def _fused_e8m0(
    q,
    qw,
    k,
    kw,
    transpose_scale,
    scale_dtype,
    q_norm=None,
    res=None,
    gemma=False,
    gs=128,
):
    m, n1 = q.shape
    g = n1 // gs
    q8 = torch.empty((m, n1), dtype=dtypes.fp8, device="cuda")
    s = torch.empty(
        (g, m) if transpose_scale else (m, g), dtype=scale_dtype, device="cuda"
    ).view(m, g)
    k_out = torch.empty_like(k)
    res_out = torch.empty_like(q) if res is not None else None
    aiter.fused_qk_rmsnorm(
        q_out_quantized=q8,
        q_out_scale=s,
        q=q,
        q_weight=qw,
        q_epsilon=EPS,
        q_out_unquantized=q_norm,
        k_out=k_out,
        q_res_out=res_out,
        k=k,
        k_weight=kw,
        k_epsilon=EPS,
        q_residual=res,
        gemma_norm=gemma,
        quant_type=QuantType.per_1x128,
        group_size=gs,
        transpose_scale=transpose_scale,
    )
    return q8.view(torch.uint8), s.view(torch.uint8), k_out, res_out


def _ref_norm(x, w, gemma=False):
    x = x.float()
    w = w.float() + 1 if gemma else w.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w).bfloat16()


def _ref_e8m0_quant(x, gs):
    """RoundUp E8M0 quant of bf16 x per gs: (fp8 bytes, scale bytes)."""
    xg = x.float().unflatten(-1, (-1, gs))
    # the amax skips NaN; the kernel's amax floor
    amax = xg.abs().masked_fill(xg.isnan(), 0).amax(-1).clamp(min=1e-10)
    t = amax * torch.tensor(1 / FP8_MAX, dtype=torch.float32)
    mant, e = torch.frexp(t)  # t = mant * 2^e, mant in [0.5, 1)
    e = torch.where(mant == 0.5, e - 1, e)  # ceil_pow2(t) = 2^e
    byte = torch.where(torch.isinf(t), 254, (e + 127).clamp(max=254))  # cap 2^127
    scale = torch.exp2((byte - 127).float()).unsqueeze(-1)
    q8 = (xg / scale).clamp(-FP8_MAX, FP8_MAX).to(dtypes.fp8).flatten(-2)
    return q8.view(torch.uint8), byte.to(torch.uint8)


def _ulps(a, b):
    nan = a.isnan()
    assert torch.equal(nan, b.isnan())
    # bf16 bits as integers ordered like the values (+-0 -> 0)
    a, b = (x.masked_fill(nan, 0).view(torch.int16).int() for x in (a, b))
    a, b = (torch.where(x < 0, -(x & 0x7FFF), x) for x in (a, b))
    return (a - b).abs().max().item()


def _check_quant(q8, s_raw, q_norm, transpose_scale, gs=128):
    m, n1 = q_norm.shape
    # transpose_scale stores the (m, g) scale column-major
    s = s_raw.view(-1, m).t() if transpose_scale else s_raw
    # the Inf columns (the last 128, n1 > 128) take the 2^127 scale; their fp8 is not
    # compared: rcp(2^127) flushes to 0, so the kernel writes their +-Inf and NaN as
    # -fp8 max and its finite elements as 0 (as with FP32 scales)
    fin = slice(None, n1 - 128 if n1 > 128 else n1)
    q8_ref, s_ref = _ref_e8m0_quant(q_norm, gs)
    assert torch.equal(s, s_ref)
    assert torch.equal(q8[:, fin], q8_ref[:, fin])
    if n1 > 128:
        assert (s[:, -128 // gs :] == 254).all()

    if transpose_scale and gs == 32:
        # per_group_quant_hip's group-32 transposed E8M0 scale is not a valid scale
        # here (its bytes are not E8M0 exponents of this data even at m == 1); the
        # torch reference above covers this case
        return
    q8_hip, s_hip = per_group_quant_hip(
        q_norm,
        quant_dtype=dtypes.fp8,
        group_size=gs,
        transpose_scale=transpose_scale,
        scale_type=dtypes.fp8_e8m0,
    )
    assert torch.equal(s_raw, s_hip.view(torch.uint8))
    assert torch.equal(q8[:, fin], q8_hip.view(torch.uint8)[:, fin])


@pytest.mark.parametrize("gs", GROUP_SIZES)
@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize(("n1", "n2"), WIDTHS)
def test_fused_qk_rmsnorm_e8m0(n1, n2, m, transpose_scale, scale_dtype, gs):
    q, qw, k, kw, _ = _inputs(m, n1, n2, m + n1)
    q8, s, k_out, _ = _fused_e8m0(q, qw, k, kw, transpose_scale, scale_dtype, gs=gs)
    # also writing the bf16 norm changes none of the outputs
    q_norm = torch.empty_like(q)
    outs = _fused_e8m0(q, qw, k, kw, transpose_scale, scale_dtype, q_norm, gs=gs)
    assert all(map(torch.equal, outs[:3], (q8, s, k_out)))

    assert _ulps(q_norm, _ref_norm(q, qw)) <= 1
    assert _ulps(k_out, _ref_norm(k, kw)) <= 1
    _check_quant(q8, s, q_norm, transpose_scale, gs)


@pytest.mark.parametrize("gs", GROUP_SIZES)
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("m", [1, 135, 4096])
@pytest.mark.parametrize(("n1", "n2"), [(1536, 512), (4096, 512)])
def test_fused_qk_rmsnorm_e8m0_residual_gemma(
    n1, n2, m, transpose_scale, residual, gemma, gs
):
    q, qw, k, kw, res = _inputs(m, n1, n2, m + n1 + 1)
    res = res if residual else None
    q_norm = torch.empty_like(q)
    q8, s, k_out, res_out = _fused_e8m0(
        q, qw, k, kw, transpose_scale, dtypes.fp8_e8m0, q_norm, res, gemma, gs
    )
    x = q.float() + res.float() if residual else q
    if residual:
        assert torch.equal(res_out, x.bfloat16())
    assert _ulps(q_norm, _ref_norm(x, qw, gemma)) <= 1
    assert _ulps(k_out, _ref_norm(k, kw, gemma)) <= 1
    _check_quant(q8, s, q_norm, transpose_scale, gs)


if __name__ == "__main__":
    # CI runs op_tests as scripts (.github/scripts/aiter_test.sh)
    sys.exit(pytest.main([__file__, "-q"]))
