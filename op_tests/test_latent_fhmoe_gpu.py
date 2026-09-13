# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact-dimension MI355X correctness test for Kimi-K3 latent FHMoE."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.latent_fhmoe import latent_fhmoe
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale, shuffle_weight

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    denom = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float(error / denom)


def _situv2(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate.float() / 4.0) * torch.sigmoid(gate.float())) * (
        25.0 * torch.tanh(up.float() / 25.0)
    )


def _routed_aiter_reference(
    routed_input,
    routed_w1,
    routed_w2,
    routed_s1,
    routed_s2,
    topk_weight,
    topk_ids,
):
    m, topk = topk_ids.shape
    block_m = 32
    max_sorted = topk_ids.numel() + routed_w1.shape[0] * block_m - topk
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=routed_input.device)
    sorted_weights = torch.empty(
        max_sorted, dtype=torch.float32, device=routed_input.device
    )
    sorted_experts = torch.empty(
        (max_sorted + block_m - 1) // block_m,
        dtype=torch.int32,
        device=routed_input.device,
    )
    num_valid = torch.empty(2, dtype=torch.int32, device=routed_input.device)
    zero_out = torch.empty_like(routed_input)
    flydsl_moe_sorting_fwd(
        topk_ids,
        topk_weight,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid,
        zero_out,
        routed_w1.shape[0],
        block_m,
        None,
        None,
    )
    a1, a1_scale = aiter.fused_dynamic_mxfp8_quant_moe_sort(
        routed_input,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid,
        token_num=m,
        topk=topk,
        block_size=block_m,
        sorted_weights=sorted_weights,
    )
    a2, a2_scale = flydsl_moe_stage1(
        a1,
        routed_w1,
        sorted_ids,
        sorted_experts,
        num_valid,
        topk=topk,
        tile_m=32,
        tile_n=128,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="fp8",
        act="situv2",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        w1_scale=routed_s1,
        a1_scale=a1_scale,
        persist_m=1,
        gate_mode="separated",
    )
    return flydsl_moe_stage2(
        a2,
        routed_w2,
        sorted_ids,
        sorted_experts,
        num_valid,
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=128,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=routed_s2,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        persist=False,
    )


@pytest.mark.parametrize(
    # FlyDSL specializes runtime integer arguments in its process-local cache.
    # Run M=8 in a fresh pytest process instead of mixing specializations.
    "m",
    [8] if os.environ.get("AITER_K3_LATENT_M8", "0") == "1" else [1],
)
def test_k3_latent_fhmoe_exact_dimensions(
    monkeypatch: pytest.MonkeyPatch, m: int, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("AITER_SITUV2_A8W4", "1")
    torch.manual_seed(7 + m)
    device = torch.device("cuda")
    experts = 1
    topk = 1

    routed_input = torch.randn((m, 3584), device=device, dtype=torch.bfloat16) * 0.02
    shared_input = torch.randn((m, 7168), device=device, dtype=torch.bfloat16) * 0.02
    raw_routed_w1 = (
        torch.randn((experts, 768, 3584), device=device, dtype=torch.bfloat16) * 0.01
    )
    raw_routed_w2 = (
        torch.randn((experts, 3584, 384), device=device, dtype=torch.bfloat16) * 0.01
    )
    routed_w1, routed_s1 = per_1x32_f4_quant(raw_routed_w1)
    routed_w2, routed_s2 = per_1x32_f4_quant(raw_routed_w2)
    routed_w1 = shuffle_weight(routed_w1, layout=(16, 16))
    routed_w2 = shuffle_weight(routed_w2, layout=(16, 16))
    routed_s1 = shuffle_scale(routed_s1.view(-1, routed_s1.shape[-1])).reshape(
        experts, 768, 112
    )
    routed_s2 = shuffle_scale(routed_s2.view(-1, routed_s2.shape[-1])).reshape(
        experts, 3584, 16
    )

    raw_shared_w1 = (
        torch.randn((1, 1536, 7168), device=device, dtype=torch.bfloat16) * 0.01
    )
    raw_shared_w2 = (
        torch.randn((1, 7168, 768), device=device, dtype=torch.bfloat16) * 0.01
    )
    shared_w1 = shuffle_weight(raw_shared_w1, layout=(16, 16))
    shared_w2 = shuffle_weight(raw_shared_w2, layout=(16, 16))
    topk_ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    topk_weight = torch.ones((m, topk), dtype=torch.float32, device=device)

    routed_actual, shared_actual = latent_fhmoe(
        routed_input,
        routed_w1,
        routed_w2,
        routed_s1,
        routed_s2,
        topk_weight,
        topk_ids,
        shared_input,
        shared_w1,
        shared_w2,
    )
    torch.cuda.synchronize()

    routed_ref = _routed_aiter_reference(
        routed_input,
        routed_w1,
        routed_w2,
        routed_s1,
        routed_s2,
        topk_weight,
        topk_ids,
    )
    routed_gate, routed_up = F.linear(routed_input, raw_routed_w1[0]).chunk(2, dim=-1)
    routed_torch_ref = F.linear(
        _situv2(routed_gate, routed_up).to(torch.bfloat16), raw_routed_w2[0]
    )
    gate, up = F.linear(shared_input, raw_shared_w1[0]).chunk(2, dim=-1)
    shared_inter = _situv2(gate, up).to(torch.bfloat16)
    shared_ref = F.linear(shared_inter, raw_shared_w2[0])

    routed_error = _rel_l2(routed_actual, routed_ref)
    routed_torch_error = _rel_l2(routed_actual, routed_torch_ref)
    shared_error = _rel_l2(shared_actual, shared_ref)
    with capsys.disabled():
        print(
            "K3 latent correctness: "
            f"routed norm={routed_actual.float().norm().item():.3e}/"
            f"{routed_ref.float().norm().item():.3e}, "
            f"shared norm={shared_actual.float().norm().item():.3e}/"
            f"{shared_ref.float().norm().item():.3e}, "
            f"errors={routed_error:.3e}/{routed_torch_error:.3e}/{shared_error:.3e}"
        )
    assert torch.isfinite(routed_actual).all()
    assert torch.isfinite(shared_actual).all()
    assert routed_error <= 1e-3, f"routed AITER rel-L2: {routed_error:.3e}"
    assert shared_error <= 3e-2, f"shared rel-L2: {shared_error:.3e}"

    for _ in range(2):
        latent_fhmoe(
            routed_input,
            routed_w1,
            routed_w2,
            routed_s1,
            routed_s2,
            topk_weight,
            topk_ids,
            shared_input,
            shared_w1,
            shared_w2,
        )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        latent_fhmoe(
            routed_input,
            routed_w1,
            routed_w2,
            routed_s1,
            routed_s2,
            topk_weight,
            topk_ids,
            shared_input,
            shared_w1,
            shared_w2,
        )
    end.record()
    end.synchronize()
    with capsys.disabled():
        print(
            f"K3 latent FHMoE M={m}: {start.elapsed_time(end) / 10:.3f} ms, "
            f"routed rel-L2={routed_error:.3e}, shared rel-L2={shared_error:.3e}"
        )
