# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_qk_rmsnorm with fp8 per-1x128 E8M0 scales: its fp8 and scales are
per_group_quant_hip(scale_type=fp8_e8m0) of its own bf16 norm, and that norm is
within 1 bf16 ulp of the unquantized fused_qk_rmsnorm."""

import pytest
import torch

import aiter
from aiter import QuantType, dtypes
from aiter.ops.quant import per_group_quant_hip

# (q width, kv width): Kimi-K3 MLA q_lora 1536 / kv_lora 512, and widths across
# the group-quant dispatch (128: its fp8-only branch).
WIDTHS = [(1536, 512), (128, 128), (512, 512), (1024, 512), (4096, 512), (8192, 512)]
# decode and prefill row counts; >= 4096 takes the unquantized kernel's 4-row path
ROWS = [1, 15, 135, 1009, 4096, 16384]


def _inputs(m, n1, n2, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda")
    q = (r(m, n1) * 3).bfloat16()
    if m >= 2:
        q[0, :128] = 0  # an all-zero group
        q[1, 128:256] *= 2.0**20
    qw = r(n1).bfloat16()
    if n1 > 128:
        qw[-128:] = float("inf")  # an Inf group in every row: the 2^127 scale cap
    return q, qw, (r(m, n2) * 3).bfloat16(), r(n2).bfloat16()


def _fused_e8m0(q, qw, k, kw, transpose_scale, q_out_unquantized=None):
    m, n1 = q.shape
    g = n1 // 128
    q8 = torch.empty((m, n1), dtype=dtypes.fp8, device="cuda")
    s = torch.empty(
        (g, m) if transpose_scale else (m, g), dtype=dtypes.fp8_e8m0, device="cuda"
    ).view(m, g)
    k_out = torch.empty_like(k)
    aiter.fused_qk_rmsnorm(
        q_out_quantized=q8,
        q_out_scale=s,
        q=q,
        q_weight=qw,
        q_epsilon=1e-6,
        q_out_unquantized=q_out_unquantized,
        k_out=k_out,
        k=k,
        k_weight=kw,
        k_epsilon=1e-6,
        quant_type=QuantType.per_1x128,
        group_size=128,
        transpose_scale=transpose_scale,
    )
    return q8.view(torch.uint8), s.view(torch.uint8), k_out


def _ulps(a, b):
    # bf16 bits as integers ordered like the values (+-0 -> 0)
    a, b = (x.view(torch.int16).int() for x in (a, b))
    a, b = (torch.where(x < 0, -(x & 0x7FFF), x) for x in (a, b))
    return (a - b).abs().max().item()


@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize(("n1", "n2"), WIDTHS)
def test_fused_qk_rmsnorm_e8m0(n1, n2, m, transpose_scale):
    q, qw, k, kw = _inputs(m, n1, n2, m + n1)
    q8, s, k_out = _fused_e8m0(q, qw, k, kw, transpose_scale)
    # also writing the bf16 norm changes none of the outputs
    q_norm = torch.empty_like(q)
    outs = _fused_e8m0(q, qw, k, kw, transpose_scale, q_norm)
    assert all(map(torch.equal, outs, (q8, s, k_out)))

    # the quant is exactly the standalone E8M0 quant of that norm
    q8_ref, s_ref = per_group_quant_hip(
        q_norm,
        quant_dtype=dtypes.fp8,
        group_size=128,
        transpose_scale=transpose_scale,
        scale_type=dtypes.fp8_e8m0,
    )
    assert torch.equal(s, s_ref.view(torch.uint8))
    # the Inf group's fp8 is this kernel's conversion (0xFE, as with FP32 scales), where
    # per_group_quant_hip writes NaN: compare the finite groups
    fin = slice(None, n1 - 128 if n1 > 128 else n1)
    assert torch.equal(q8[:, fin], q8_ref.view(torch.uint8)[:, fin])

    # the norms are within 1 bf16 ulp of the unquantized kernel's
    q_ref, k_ref = torch.empty_like(q), torch.empty_like(k)
    aiter.fused_qk_rmsnorm(
        q_out_quantized=q_ref,
        q=q,
        q_weight=qw,
        q_epsilon=1e-6,
        k_out=k_ref,
        k=k,
        k_weight=kw,
        k_epsilon=1e-6,
        quant_type=QuantType.No,
    )
    assert _ulps(q_norm, q_ref) <= 1
    assert _ulps(k_out, k_ref) <= 1
