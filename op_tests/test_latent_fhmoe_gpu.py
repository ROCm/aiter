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
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)

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
        gate_mode="interleave",
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


def _install_latent_tuning_config(
    monkeypatch: pytest.MonkeyPatch, *, force: bool = False
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    stage1_value = os.environ.get("AITER_K3_LATENT_STAGE1_CONFIG")
    stage2_value = os.environ.get("AITER_K3_LATENT_STAGE2_CONFIG")
    if not force and stage1_value is None and stage2_value is None:
        return

    import importlib

    latent_impl = importlib.import_module("aiter.ops.flydsl.latent_fhmoe")
    from aiter.ops.flydsl.kernels import fhmoe as fhmoe_kernels

    stage1_config = tuple(
        int(value)
        for value in (
            stage1_value or "32,64,256,4,4,0"
        ).split(",")
    )
    stage2_config = tuple(
        int(value)
        for value in (
            stage2_value or "32,256,128,4,32,0,0"
        ).split(",")
    )
    assert len(stage1_config) == 6
    assert len(stage2_config) == 7
    monkeypatch.setattr(latent_impl, "_LATENT_BLOCK_M", stage1_config[0])
    original_stage1 = fhmoe_kernels.compile_mixed_latent_fhmoe_gemm1
    original_stage2 = fhmoe_kernels.compile_mixed_latent_fhmoe_gemm2
    monkeypatch.setattr(
        fhmoe_kernels,
        "compile_mixed_latent_fhmoe_gemm1",
        lambda *, experts, topk: original_stage1(
            experts=experts,
            topk=topk,
            tile_m=stage1_config[0],
            tile_n=stage1_config[1],
            tile_k=stage1_config[2],
            persist_m=stage1_config[3],
            waves_per_eu=stage1_config[4] or None,
            xcd_swizzle=stage1_config[5],
        ),
    )
    monkeypatch.setattr(
        fhmoe_kernels,
        "compile_mixed_latent_fhmoe_gemm2",
        lambda *, experts, topk: original_stage2(
            experts=experts,
            topk=topk,
            tile_m=stage2_config[0],
            tile_n=stage2_config[1],
            tile_k=stage2_config[2],
            persist_m=stage2_config[3],
            sort_block_m=stage2_config[4] or stage1_config[0],
            waves_per_eu=stage2_config[5] or None,
            xcd_swizzle=stage2_config[6],
        ),
    )
    return stage1_config, stage2_config


@pytest.mark.parametrize(
    # FlyDSL specializes runtime integer arguments in its process-local cache.
    # Run each M in a fresh pytest process instead of mixing specializations.
    "m",
    [
        int(
            os.environ.get(
                "AITER_K3_LATENT_M",
                "8" if os.environ.get("AITER_K3_LATENT_M8", "0") == "1" else "1",
            )
        )
    ],
)
def test_k3_latent_fhmoe_exact_dimensions(
    monkeypatch: pytest.MonkeyPatch, m: int, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("AITER_SITUV2_A8W4", "1")
    _install_latent_tuning_config(monkeypatch)
    torch.manual_seed(7 + m)
    device = torch.device("cuda")
    graph_mode = os.environ.get("AITER_K3_LATENT_GRAPH", "0") == "1"
    experts = 896 if graph_mode else 1
    topk = 16 if graph_mode else 1

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
    routed_w1 = shuffle_weight_a16w4(routed_w1, 16, True)
    routed_w2 = shuffle_weight_a16w4(routed_w2, 16, False)
    routed_s1 = shuffle_scale_a16w4(
        routed_s1.view(-1, routed_s1.shape[-1]), experts, True
    ).reshape(
        experts, 768, 112
    )
    routed_s2 = shuffle_scale_a16w4(
        routed_s2.view(-1, routed_s2.shape[-1]), experts, False
    ).reshape(
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
    if experts >= m * topk:
        topk_ids = torch.randperm(experts, dtype=torch.int32, device=device)[
            : m * topk
        ].reshape(m, topk)
    else:
        topk_ids = torch.randint(
            experts, (m, topk), dtype=torch.int32, device=device
        )
    topk_weight = torch.full(
        (m, topk), 1.0 / topk, dtype=torch.float32, device=device
    )

    args = (
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
    if graph_mode:
        latent_fhmoe(*args)
        torch.cuda.synchronize()
        # Capture with maximally duplicated routing, then replay the same graph
        # against a different live routing pattern. This matches vLLM capture
        # warmup followed by the first speculative decode and catches kernels
        # that accidentally retain routing values instead of tensor addresses.
        topk_ids.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            routed_actual, shared_actual = latent_fhmoe(*args)
        topk_ids.copy_(
            (
                torch.arange(m * topk, dtype=torch.int32, device=device)
                * (experts // (m * topk))
            ).reshape(m, topk)
        )
        scratch = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        scratch.fill_(0xA5)
        del scratch
        graph.replay()
    else:
        routed_actual, shared_actual = latent_fhmoe(*args)
    torch.cuda.synchronize()

    if experts == 1:
        routed_gate, routed_up = F.linear(routed_input, raw_routed_w1[0]).chunk(
            2, dim=-1
        )
        routed_torch_ref = F.linear(
            _situv2(routed_gate, routed_up).to(torch.bfloat16), raw_routed_w2[0]
        )
    routed_ref = _routed_aiter_reference(
        routed_input,
        routed_w1,
        routed_w2,
        routed_s1,
        routed_s2,
        topk_weight,
        topk_ids,
    )
    if experts != 1:
        routed_torch_ref = routed_ref
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
    routed_tolerance = 1e-2 if graph_mode else 1e-3
    assert routed_error <= routed_tolerance, (
        f"routed AITER rel-L2: {routed_error:.3e}"
    )
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


@pytest.mark.skipif(
    os.environ.get("AITER_K3_LATENT_LIVE_SHAPE", "0") != "1",
    reason="large 896-expert K3 shape is opt-in",
)
def test_k3_latent_fhmoe_live_tp8_shape(monkeypatch: pytest.MonkeyPatch):
    """Smoke the exact TP8/EP1 IX decode domain without allocating BF16 experts."""
    tuning_config = _install_latent_tuning_config(monkeypatch, force=True)
    assert tuning_config is not None
    stage1_config, stage2_config = tuning_config
    torch.manual_seed(23)
    device = torch.device("cuda")
    m, experts, topk = 8, 896, 16

    routed_input = torch.randn((m, 3584), device=device, dtype=torch.bfloat16) * 0.02
    shared_input = torch.randn((m, 7168), device=device, dtype=torch.bfloat16) * 0.02
    routed_w1 = torch.zeros(
        (experts, 768, 1792), device=device, dtype=torch.uint8
    ).view(aiter.dtypes.fp4x2)
    routed_w2 = torch.zeros(
        (experts, 3584, 192), device=device, dtype=torch.uint8
    ).view(aiter.dtypes.fp4x2)
    routed_s1 = torch.full(
        (experts, 768, 112), 0x7F, device=device, dtype=torch.uint8
    )
    routed_s2 = torch.full(
        (experts, 3584, 16), 0x7F, device=device, dtype=torch.uint8
    )
    raw_shared_w1 = (
        torch.randn((1, 1536, 7168), device=device, dtype=torch.bfloat16) * 0.01
    )
    raw_shared_w2 = (
        torch.randn((1, 7168, 768), device=device, dtype=torch.bfloat16) * 0.01
    )
    shared_w1 = shuffle_weight(raw_shared_w1, layout=(16, 16))
    shared_w2 = shuffle_weight(raw_shared_w2, layout=(16, 16))
    topk_ids = torch.arange(topk, dtype=torch.int32, device=device).repeat(m, 1)
    topk_weight = torch.full(
        (m, topk), 1.0 / topk, dtype=torch.float32, device=device
    )

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
    gate, up = F.linear(shared_input, raw_shared_w1[0]).chunk(2, dim=-1)
    shared_ref = F.linear(
        _situv2(gate, up).to(torch.bfloat16), raw_shared_w2[0]
    )
    shared_error = _rel_l2(shared_actual, shared_ref)
    print(
        "K3 live TP8 shape: "
        f"routed norm={routed_actual.float().norm().item():.3e}, "
        f"shared rel-L2={shared_error:.3e}"
    )
    assert torch.count_nonzero(routed_actual) == 0
    assert torch.isfinite(shared_actual).all()
    assert shared_error <= 3e-2

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
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
    for _ in range(10):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    print(
        f"K3 live TP8 graph: {start.elapsed_time(end) / 100:.3f} ms, "
        f"stage1={stage1_config}, stage2={stage2_config}"
    )
