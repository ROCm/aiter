# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused gfx942 packed-int4 FlyDSL MoE stage-1 correctness regressions."""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import per_1x32_i4_quant
from aiter.ops.shuffle import (
    pack_int8_to_packed_int4,
    shuffle_scale_for_int4,
    shuffle_weight,
)


_SKIP_GFX942_FLYDSL = pytest.mark.skipif(
    get_gfx() != "gfx942" or not is_flydsl_available(),
    reason="gfx942 FlyDSL required",
)


@_SKIP_GFX942_FLYDSL
def test_flydsl_stage1_a16wi4_situv2():
    """Compare direct-store packed-int4 SiTUv2 stage1 against torch."""
    token, model_dim, inter_dim, experts, topk, block_m = 16, 512, 256, 8, 2, 16
    beta, linear_beta = 0.5, 2.0
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    hidden = torch.randn((token, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w1 = (
        torch.randn(
            (experts, inter_dim * 2, model_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10
    )
    w2 = (
        torch.randn(
            (experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda"
        )
        / 10
    )
    scores = torch.randn((token, experts), dtype=torch.bfloat16, device="cuda")
    topk_weights, topk_ids = fused_topk(hidden, scores, topk, True)
    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16, block_m
    )

    w1_qt, w1_scale = per_1x32_i4_quant(w1)
    w2_qt, _ = per_1x32_i4_quant(w2)
    w1_qt = w1_qt.view(dtypes.i4x2)
    w2_qt = w2_qt.view(dtypes.i4x2)
    reference = torch_moe_stage1(
        hidden,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )

    w1_shuffled = pack_int8_to_packed_int4(
        shuffle_weight(w1_qt.view(dtypes.i8), (16, 16))
    ).view(experts, inter_dim * 2, model_dim // 2)
    w1_shuffled = w1_shuffled.view(dtypes.i4x2)
    w1_scale_shuffled = shuffle_scale_for_int4(w1_scale, group_size=32).view(-1)

    actual = flydsl_moe_stage1(
        a=hidden,
        w1=w1_shuffled,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=128,
        a_dtype="bf16",
        b_dtype="int4",
        out_dtype="bf16",
        act="situv2",
        situ_beta=beta,
        situ_linear_beta=linear_beta,
        w1_scale=w1_scale_shuffled,
        a1_scale=None,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, atol=0.2, rtol=0.1)
