# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_qk_rmsnorm with fp8 per-1x128 E8M0 scales: byte-identical to the
unquantized fused_qk_rmsnorm + per_group_quant_hip(scale_type=fp8_e8m0)."""

import pytest
import torch

import aiter
from aiter import QuantType, dtypes
from aiter.ops.quant import per_group_quant_hip

# (q width, kv width): Kimi-K3 MLA q_lora 1536 / kv_lora 512, and the other
# supported widths of the unquantized kernel's table.
WIDTHS = [(1536, 512), (512, 512), (1024, 512), (4096, 512), (8192, 512)]
# decode and prefill row counts; >= 4096 takes the unquantized kernel's 4-row path
ROWS = [1, 15, 135, 1009, 4096, 16384]


def _inputs(m, n1, n2, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda")
    q = (r(m, n1) * 3).bfloat16()
    if m >= 2:
        q[0, :128] = 0  # an all-zero group
        q[1, 128:256] *= 2.0**20
    return q, r(n1).bfloat16(), (r(m, n2) * 3).bfloat16(), r(n2).bfloat16()


@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize(("n1", "n2"), WIDTHS)
def test_fused_qk_rmsnorm_e8m0_matches_unquantized_then_quant(
    n1, n2, m, transpose_scale
):
    q, qw, k, kw = _inputs(m, n1, n2, m + n1)
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
    q8_ref, s_ref = per_group_quant_hip(
        q_ref,
        quant_dtype=dtypes.fp8,
        group_size=128,
        transpose_scale=transpose_scale,
        scale_type=dtypes.fp8_e8m0,
    )

    g = n1 // 128
    q8 = torch.empty((m, n1), dtype=dtypes.fp8, device="cuda")
    s = torch.empty(
        (g, m) if transpose_scale else (m, g), dtype=dtypes.fp8_e8m0, device="cuda"
    )
    s = s.view(m, g)
    k_out = torch.empty_like(k)
    aiter.fused_qk_rmsnorm(
        q_out_quantized=q8,
        q_out_scale=s,
        q=q,
        q_weight=qw,
        q_epsilon=1e-6,
        k_out=k_out,
        k=k,
        k_weight=kw,
        k_epsilon=1e-6,
        quant_type=QuantType.per_1x128,
        group_size=128,
        transpose_scale=transpose_scale,
    )

    assert torch.equal(k_out, k_ref)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    assert torch.equal(q8.view(torch.uint8), q8_ref.view(torch.uint8))


def test_fused_qk_rmsnorm_e8m0_rejects_unsupported_width():
    m, n1 = 8, 6144
    q, qw, k, kw = _inputs(m, n1, 512, 0)
    with pytest.raises(Exception, match="4096 < n <= 6144"):
        aiter.fused_qk_rmsnorm(
            q_out_quantized=torch.empty((m, n1), dtype=dtypes.fp8, device="cuda"),
            q_out_scale=torch.empty(
                (m, n1 // 128), dtype=dtypes.fp8_e8m0, device="cuda"
            ),
            q=q,
            q_weight=qw,
            q_epsilon=1e-6,
            k_out=torch.empty_like(k),
            k=k,
            k_weight=kw,
            k_epsilon=1e-6,
            quant_type=QuantType.per_1x128,
            group_size=128,
        )
