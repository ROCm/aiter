# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side contract tests for the Kimi-K3 latent FHMoE vertical slice."""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.latent_fhmoe import (
    K3_SHARED_MODEL_DIM,
    _validate_latent_fhmoe_contract,
    latent_fhmoe_fake,
)
from aiter.ops.flydsl.latent_fhmoe import LatentFHMoELayout, integrated_route_domain


def _meta_contract(topk: int = 2, experts: int = 2) -> dict:
    device = torch.device("meta")
    m = 1
    return {
        "routed_input": torch.empty((m, 3584), dtype=torch.bfloat16, device=device),
        "routed_w1": torch.empty(
            (experts, 768, 1792), dtype=dtypes.fp4x2, device=device
        ),
        "routed_w2": torch.empty(
            (experts, 3584, 192), dtype=dtypes.fp4x2, device=device
        ),
        "routed_w1_scale": torch.empty(
            (experts, 768, 112), dtype=torch.uint8, device=device
        ),
        "routed_w2_scale": torch.empty(
            (experts, 3584, 16), dtype=torch.uint8, device=device
        ),
        "topk_weight": torch.empty((m, topk), dtype=torch.float32, device=device),
        "topk_ids": torch.empty((m, topk), dtype=torch.int32, device=device),
        "shared_input": torch.empty((m, 7168), dtype=torch.bfloat16, device=device),
        "shared_w1": torch.empty((1, 1536, 7168), dtype=torch.bfloat16, device=device),
        "shared_w2": torch.empty((1, 7168, 768), dtype=torch.bfloat16, device=device),
        "activation": aiter.ActivationType.Situv2,
        "quant_type": aiter.QuantType.per_1x32,
        "beta": 4.0,
        "linear_beta": 25.0,
    }


def test_k3_latent_contract_and_fake_outputs(monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module("aiter.latent_fhmoe")

    monkeypatch.setattr(module, "get_gfx", lambda: "gfx950")
    args = _meta_contract()
    _validate_latent_fhmoe_contract(**args)

    fake_args = dict(args)
    fake_args["activation"] = aiter.ActivationType.Situv2.value
    fake_args["quant_type"] = aiter.QuantType.per_1x32.value
    routed, shared = latent_fhmoe_fake(**fake_args)
    assert routed.shape == (1, 3584)
    assert shared.shape == (1, K3_SHARED_MODEL_DIM)
    assert routed.dtype == shared.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("beta", 1.0, "beta=4"),
        ("activation", aiter.ActivationType.Silu, "requires SiTUv2"),
        (
            "shared_w2",
            torch.empty((1, 3584, 768), dtype=torch.bfloat16, device="meta"),
            "Expected shared_w2 shape",
        ),
    ),
)
def test_k3_latent_contract_rejects_ambiguous_variants(
    monkeypatch: pytest.MonkeyPatch, field: str, value, message: str
):
    module = importlib.import_module("aiter.latent_fhmoe")

    monkeypatch.setattr(module, "get_gfx", lambda: "gfx950")
    args = _meta_contract()
    args[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_latent_fhmoe_contract(**args)


def test_k3_integrated_domain_has_one_shared_task_per_token():
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.75, 0.25], [0.4, 0.6]], dtype=torch.float32)
    all_ids, all_weights, shared_id = integrated_route_domain(ids, weights, 2)

    assert shared_id == 2
    assert all_ids.tolist() == [[0, 1, 2], [1, 0, 2]]
    assert torch.equal(all_weights[:, :2], weights)
    assert torch.equal(all_weights[:, 2], torch.ones(2))
    assert LatentFHMoELayout().max_inter_dim == 768
    assert LatentFHMoELayout().max_model_dim == 7168


def test_latent_api_is_separate_and_returns_two_outputs():
    from aiter.latent_fhmoe import latent_fhmoe, latent_fhmoe_

    assert aiter.kimi_k3_latent_fhmoe is latent_fhmoe
    assert "shared_input" in inspect.signature(latent_fhmoe).parameters
    schema = torch.ops.aiter.latent_fhmoe_.default._schema
    assert len(schema.returns) == 2
    assert "shared_input" in {argument.name for argument in schema.arguments}
    assert callable(latent_fhmoe_)


def test_common_kernel_builds_bf16_dual_mfma_launchers():
    from aiter.ops.flydsl.kernels.fhmoe import (
        compile_mixed_latent_fhmoe_gemm1,
        compile_mixed_latent_fhmoe_gemm2,
    )

    assert callable(compile_mixed_latent_fhmoe_gemm1(experts=2, topk=2))
    assert callable(compile_mixed_latent_fhmoe_gemm2(experts=2, topk=2))
