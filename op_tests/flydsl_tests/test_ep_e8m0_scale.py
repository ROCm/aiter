# SPDX-License-Identifier: MIT
"""Typed E8M0 scales from EP must preserve exponent bytes in gfx1250 routing."""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes, get_hip_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
from aiter.ops.flydsl import moe_kernels


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="Requires gfx1250",
)
@pytest.mark.parametrize("quant_mode", ["fp8", "fp4"])
@pytest.mark.parametrize("typed_scale", [False, True])
def test_ep_typed_e8m0_routed_bytes(monkeypatch, quant_mode, typed_scale):
    torch.manual_seed(17)
    tokens, hidden_dim, experts, topk = 17, 5120, 4, 2
    x = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    payload, scales = get_hip_quant(QuantType.per_1x32)(
        x,
        quant_dtype=dtypes.fp8 if quant_mode == "fp8" else dtypes.fp4x2,
        scale_type=torch.float8_e8m0fnu,
    )
    scales = scales if typed_scale else scales.view(torch.uint8)
    ids = (
        (
            torch.arange(tokens, device="cuda")[:, None]
            + torch.arange(topk, device="cuda")[None, :]
        )
        .remainder(experts)
        .to(torch.int32)
    )
    real_quant = moe_kernels.flydsl_moe_fused_quant_preshuffle
    captured = []

    class ReachedGemmInput(Exception):
        pass

    def quant_and_stop(*args, **kwargs):
        # Run the real route-indexed GPU scatter/preshuffle, stopping before
        # GEMM so its byte input can be checked against the original wire.
        out_payload, out_scale = real_quant(*args, **kwargs)
        torch.cuda.synchronize()
        captured.append((kwargs, out_payload, out_scale))
        raise ReachedGemmInput

    monkeypatch.setattr(
        moe_kernels, "flydsl_moe_fused_quant_preshuffle", quant_and_stop
    )
    with pytest.raises(ReachedGemmInput):
        grouped._grouped_a8w4_tdm_moe(
            payload,
            None,
            None,
            torch.ones(tokens, topk, device="cuda"),
            ids,
            E=experts,
            model_dim=hidden_dim,
            inter_dim=256,
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            w1_scale=None,
            w2_scale=None,
            bias1=None,
            bias2=None,
            swiglu_limit=0,
            doweight_stage1=False,
            expert_mask=torch.ones(experts, device="cuda", dtype=torch.int32),
            num_local_tokens=torch.tensor([tokens], device="cuda", dtype=torch.int32),
            data_format="fp4" if quant_mode == "fp4" else "a8w4",
            a1_scale=scales,
        )
    kwargs, grouped_payload, grouped_scale = captured[0]
    rows = kwargs["topids_to_rows"].long().reshape(-1)
    source_rows = torch.arange(tokens, device="cuda").repeat_interleave(topk)
    assert (rows >= 0).all()
    torch.testing.assert_close(
        grouped_payload.reshape(-1, payload.shape[-1])[rows],
        payload.view(torch.uint8)[source_rows],
        rtol=0,
        atol=0,
    )
    rep, width = kwargs["wmma_rep"], hidden_dim // 32
    base = (rows // (rep * 16)) * (width // 4 * rep * 16) + rows % (rep * 16)
    offsets = base[:, None] + torch.arange(width // 4, device="cuda")[None, :] * (
        rep * 16
    )
    actual = (
        grouped_scale.reshape(-1)
        .view(torch.int32)[offsets]
        .contiguous()
        .view(torch.uint8)
    )
    torch.testing.assert_close(
        actual, scales.view(torch.uint8)[source_rows], rtol=0, atol=0
    )
