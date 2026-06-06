# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for FlyDSL stage2 per-slot output."""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility.fp4_utils import moe_mxfp4_sort


HAS_GPU = torch.cuda.is_available()
try:
    from aiter.ops.flydsl.utils import is_flydsl_available

    HAS_FLYDSL = bool(is_flydsl_available())
except Exception:
    HAS_FLYDSL = False

requires_flydsl = pytest.mark.skipif(
    not (HAS_GPU and HAS_FLYDSL), reason="needs FlyDSL + GPU"
)

Q_TYPE = QuantType.per_1x32
Q_DTYPE = dtypes.fp4x2


def _make_stage2_data(
    *,
    token=32,
    model_dim=512,
    inter_dim=256,
    experts=8,
    topk=4,
    block_m=32,
    dtype=dtypes.bf16,
    device="cuda",
    seed=0,
    topk_ids=None,
    topk_weights=None,
    expert_mask=None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    inp = torch.randn((token, model_dim), dtype=dtype, device=device) / 10
    w1 = torch.randn((experts, inter_dim * 2, model_dim), dtype=dtype, device=device)
    w1 = w1 / 10
    w2 = torch.randn((experts, model_dim, inter_dim), dtype=dtype, device=device)
    w2 = w2 / 10

    if topk_ids is None:
        score = torch.randn((token, experts), dtype=dtype, device=device)
        topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    else:
        topk_ids = topk_ids.to(device=device, dtype=dtypes.i32).contiguous()
        topk_weights = topk_weights.to(device=device, dtype=dtypes.fp32).contiguous()

    topk_ids = topk_ids.to(dtype=dtypes.i32).contiguous()
    topk_weights = topk_weights.to(dtype=dtypes.fp32).contiguous()

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE)
    w1_qt = w1_qt.view(experts, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(experts, model_dim, inter_dim // 2)

    a1_qt, a1_scale = torch_quant(inp, quant_dtype=Q_DTYPE)
    stage1 = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=Q_TYPE,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=False,
    )

    a2_qt, a2_scale = torch_quant(stage1, quant_dtype=Q_DTYPE)
    a2_qt = a2_qt.view(token, topk, -1)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        dtype,
        block_m,
        expert_mask=expert_mask,
    )
    a2_scale_sort = moe_mxfp4_sort(
        a2_scale[: token * topk, :].view(token, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )

    return {
        "a2_qt": a2_qt,
        "a2_scale_sort": a2_scale_sort,
        "w2_qt_shuf": shuffle_weight_a16w4(w2_qt, 16, False),
        "w2_scale_shuf": shuffle_scale_a16w4(w2_scale, experts, False),
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk_weights": topk_weights,
        "dtype": dtype,
        "token": token,
        "model_dim": model_dim,
        "topk": topk,
        "block_m": block_m,
    }


def _run_stage2(data, *, sorted_weights=None, return_per_slot=False, out=None):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    out_dtype = "bf16" if data["dtype"] == dtypes.bf16 else "f16"
    return flydsl_moe_stage2(
        inter_states=data["a2_qt"],
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        out=out,
        topk=data["topk"],
        tile_m=data["block_m"],
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype=out_dtype,
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=data["a2_scale_sort"],
        sorted_weights=sorted_weights,
        return_per_slot=return_per_slot,
    )


def _assert_close_pct(actual, expected, *, atol=1.5, rtol=0.08, pass_pct=95.0):
    actual_f = actual.float()
    expected_f = expected.float()
    close = torch.isclose(actual_f, expected_f, atol=atol, rtol=rtol)
    pct = close.float().mean().item() * 100
    max_delta = (actual_f - expected_f).abs().max().item()
    assert pct >= pass_pct, (
        f"only {pct:.1f}% close at atol={atol}, rtol={rtol}; "
        f"max_delta={max_delta:.4f}"
    )


@requires_flydsl
def test_flydsl_stage2_return_per_slot_reconstructs_weighted_sum():
    data = _make_stage2_data(seed=1)

    combined = _run_stage2(data, sorted_weights=data["sorted_weights"])
    per_slot = _run_stage2(data, return_per_slot=True)
    torch.cuda.synchronize()

    assert per_slot.shape == (data["token"], data["topk"], data["model_dim"])
    assert per_slot.dtype == data["dtype"]
    assert per_slot.is_contiguous()

    weights = data["topk_weights"].to(torch.float32).unsqueeze(-1)
    reconstructed = (per_slot.float() * weights).sum(dim=1)
    _assert_close_pct(reconstructed, combined)


@requires_flydsl
def test_flydsl_stage2_return_per_slot_zeroes_unrouted_slots():
    token, topk, experts, model_dim = 32, 4, 8, 512
    device = "cuda"
    topk_ids = torch.ones((token, topk), dtype=dtypes.i32, device=device)
    topk_ids[:, 0] = 0
    topk_weights = torch.full(
        (token, topk), 1.0 / topk, dtype=dtypes.fp32, device=device
    )
    expert_mask = torch.zeros((experts,), dtype=dtypes.i32, device=device)
    expert_mask[0] = 1

    data = _make_stage2_data(
        token=token,
        topk=topk,
        experts=experts,
        model_dim=model_dim,
        seed=2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        expert_mask=expert_mask,
    )
    out = torch.empty(
        (token, topk, model_dim), dtype=data["dtype"], device=device
    ).fill_(float("nan"))

    per_slot = _run_stage2(data, return_per_slot=True, out=out)
    torch.cuda.synchronize()

    assert per_slot.data_ptr() == out.data_ptr()
    assert torch.isfinite(per_slot).all()
    assert (per_slot[:, 0, :] != 0).any()
    assert (per_slot[:, 1:, :] == 0).all()
