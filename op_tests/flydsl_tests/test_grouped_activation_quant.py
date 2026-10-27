# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone GUGU activation + MX quant + payload/scale preshuffle checks."""

import pytest
import torch

from aiter import dtypes, get_gfx
from aiter.ops.flydsl.moe_common import apply_gate_up
from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle
from aiter.utility import fp4_utils


@pytest.mark.parametrize("activation", [1, 2, 3])
@pytest.mark.parametrize("k", [1024, 3072])
@pytest.mark.parametrize("rep", [1, 8])
@pytest.mark.parametrize("mode,apre", [("fp4", True), ("fp4", False), ("fp8", False)])
def test_activation_quant(activation, k, rep, mode, apre):
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("requires gfx1250")
    torch.manual_seed(42)
    rows = 16 * rep
    x = (torch.randn((1, rows, 2 * k), device="cuda") * 2).to(torch.bfloat16)
    gate, up = x[..., 0::2].float(), x[..., 1::2].float()
    act = {1: "silu", 2: "swiglu", 3: "situv2"}[activation]
    limit = 7.0 if activation == 2 else float("inf")
    expected = apply_gate_up(
        gate,
        up,
        act,
        swiglu_limit=limit,
        situ_beta=2.0,
        situ_linear_beta=3.0,
    ).to(torch.bfloat16)
    kwargs = {"wmma_rep": rep, "quant_mode": mode, "payload_preshuffled": apre}
    # Guard the outputs to detect the most dangerous failure: treating 2*K
    # raw columns as K activated columns and overrunning payload/scale buffers.
    nbytes = rows * (k // 2 if mode == "fp4" else k)
    guard = 64
    pbuf = torch.full((nbytes + 2 * guard,), 0xA5, dtype=torch.uint8, device="cuda")
    sbuf = torch.full(
        (rows * k // 32 + 2 * guard,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    payload, scale = flydsl_moe_fused_quant_preshuffle(
        x,
        1,
        rows,
        activation=activation,
        swiglu_limit=limit,
        situ_beta=2.0,
        situ_linear_beta=3.0,
        out_payload=pbuf[guard:-guard],
        out_scale=sbuf[guard:-guard],
        **kwargs,
    )
    ref_payload, ref_scale = flydsl_moe_fused_quant_preshuffle(
        expected, 1, rows, **kwargs
    )

    def decode(p, s):
        if apre:
            p = p.reshape(rows // 16, k // 32, 16, 16).permute(0, 2, 1, 3).contiguous()
        s = s.reshape(rows // (16 * rep), k // 128, rep, 16, 4)
        s = s.permute(0, 2, 3, 1, 4).contiguous().reshape(rows, k // 32)
        values = (
            fp4_utils.mxfp4_to_f32(p.reshape(rows, k // 2))
            if mode == "fp4"
            else p.view(dtypes.fp8).float().reshape(rows, k)
        )
        return values * torch.exp2(s.float() - 127).repeat_interleave(32, dim=-1)

    actual, reference = decode(payload, scale), decode(ref_payload, ref_scale)
    # Tanh-based GEMM activation and the torch reference can straddle a BF16
    # rounding boundary; compare dequantized values rather than packed bytes.
    rel_l2 = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
    assert rel_l2.item() < 0.025
    for buf in (pbuf, sbuf):
        assert (buf[:guard] == 0xA5).all().item()
        assert (buf[-guard:] == 0xA5).all().item()


@pytest.mark.parametrize("act", ["silu", "swiglu", "situv2"])
@pytest.mark.parametrize(
    "data_format,apre", [("a4w4", "1"), ("a4w4", "0"), ("a8w4", "0")]
)
def test_moe_with_standalone_activation(act, data_format, apre, monkeypatch):
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("requires gfx1250")
    from op_tests.flydsl_tests.test_flydsl_grouped_gemm import (
        _ACT_BY_NAME,
        run_moe,
        set_data_format,
    )

    monkeypatch.setenv("AITER_FLYDSL_DISABLE_GEMM1_ACT", "1")
    # Verify ACT alone forces separate quant; no REQUANT override is needed.
    monkeypatch.setenv("AITER_FLYDSL_DISABLE_GEMM1_REQUANT", "0")
    monkeypatch.setenv("AITER_GROUPED_A4W4_A_PRESHUFFLE", apre)
    monkeypatch.setenv("AITER_USE_GROUPED_GEMM", "1")
    set_data_format(data_format)
    run_moe(
        data_format,
        tokens=32,
        experts=4,
        topk=2,
        model_dim=1024,
        inter_dim=1024,
        activation=_ACT_BY_NAME[act],
        use_bias=False,
        check_aot_cache=False,
        data_init="uniform",
        scale_init="uniform",
        situ_beta=2.0,
        situ_linear_beta=3.0,
    )
