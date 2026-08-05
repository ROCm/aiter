# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe
from aiter.fused_moe import _maybe_force_atomic_flydsl_stage2

REDUCE_KERNEL = "flydsl_moe2_afp4_wfp4_bf16_t128x256x128_reduce"
ATOMIC_KERNEL = "flydsl_moe2_afp4_wfp4_bf16_t128x256x128_atomic"


def test_force_atomic_stage2_uses_available_variant(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda name: {"mode": "atomic"} if name == ATOMIC_KERNEL else None,
    )

    assert _maybe_force_atomic_flydsl_stage2(REDUCE_KERNEL) == ATOMIC_KERNEL


def test_force_atomic_stage2_keeps_reduce_when_variant_is_unavailable(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: None,
    )

    assert _maybe_force_atomic_flydsl_stage2(REDUCE_KERNEL) == REDUCE_KERNEL


@pytest.mark.parametrize(
    "kernel_name",
    [
        "",
        "flydsl_moe2_afp4_wfp4_bf16_t128x256x128_atomic",
        "cktile_moe2_afp4_wfp4_bf16_reduce",
    ],
)
def test_force_atomic_stage2_ignores_ineligible_kernels(monkeypatch, kernel_name):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: pytest.fail("ineligible kernels must not be looked up"),
    )

    assert _maybe_force_atomic_flydsl_stage2(kernel_name) == kernel_name


def test_force_reduce_takes_precedence(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: pytest.fail("force-reduce must prevent atomic lookup"),
    )

    assert _maybe_force_atomic_flydsl_stage2(REDUCE_KERNEL) == REDUCE_KERNEL


def test_tuned_lookup_threads_atomic_mode_to_moe_sorting(monkeypatch):
    token = 2
    model_dim = 4096
    inter_dim = 1536
    num_experts = 32
    topk = 7
    activation = aiter.ActivationType.Silu
    q_type = aiter.QuantType.per_1x32
    q_dtype = aiter.dtypes.fp4x2
    stage1 = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2"
    stage2_reduce = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2"
    stage2_atomic = stage2_reduce.replace("_reduce", "_atomic")
    key = (
        "gfx950",
        256,
        token,
        model_dim,
        inter_dim,
        num_experts,
        topk,
        activation,
        str(torch.bfloat16),
        str(q_dtype),
        str(q_dtype),
        str(q_type),
        True,
        False,
    )
    config = {
        "block_m": 32,
        "ksplit": 0,
        "kernelName1": stage1,
        "kernelName2": stage2_reduce,
        "run_1stage": False,
    }

    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe, "cfg_2stages", ({key: config}, {}))
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda name: {"mode": "atomic"} if name == stage2_atomic else None,
    )
    fused_moe.get_2stage_cfgs.cache_clear()

    metadata = fused_moe.get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        num_experts,
        topk,
        torch.bfloat16,
        q_dtype,
        q_dtype,
        q_type,
        True,
        activation,
        False,
        0,
        0,
    )

    selected_stage2 = metadata.stage2.keywords["kernelName"]
    assert selected_stage2 == stage2_atomic
    assert not fused_moe.stage2_uses_route_reduce(metadata.stage2)

    recorded = {}

    def record_sort(*args, **kwargs):
        recorded.update(kwargs)
        return ("sorted",)

    monkeypatch.setattr(fused_moe, "_moe_sorting_impl", record_sort)
    result = fused_moe.moe_sorting(
        torch.zeros((token, topk), dtype=torch.int64),
        torch.zeros((token, topk), dtype=torch.float32),
        num_experts,
        model_dim,
        torch.bfloat16,
        metadata.block_m,
        None,
        None,
        accumulate=not fused_moe.stage2_uses_route_reduce(metadata.stage2),
        output_aux=metadata.output_aux,
    )

    assert result == ("sorted",)
    assert recorded["accumulate"] is True
    assert recorded["output_aux"] is False
    fused_moe.get_2stage_cfgs.cache_clear()
