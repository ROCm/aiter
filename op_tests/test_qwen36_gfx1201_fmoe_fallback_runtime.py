# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs, get_padded_M
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.enum import ActivationType, QuantType
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.moe_sorting_opus import moe_sorting_opus_get_workspace_size
from aiter.ops.shuffle import shuffle_weight


pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx1201",
    reason="gfx1201 Qwen3.6 BF16 original fallback coverage",
)


def _make_balanced_topk(token: int, topk: int, expert: int):
    rows = torch.arange(token, device="cuda", dtype=torch.int32)[:, None]
    slots = torch.arange(topk, device="cuda", dtype=torch.int32)[None, :]
    topk_ids = (rows * topk + slots) % expert
    topk_weights = torch.rand((token, topk), dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.contiguous(), topk_weights.contiguous()


def test_qwen36_gfx1201_bf16_original_fallback_4096_bucket_runs():
    torch.manual_seed(4815)
    token = 2050
    model_dim = 2048
    inter_dim = 128
    expert = 256
    topk = 8

    assert get_cu_num() == 64
    assert get_padded_M(token) == 4096
    assert moe_sorting_opus_get_workspace_size(token, expert, topk, 0, 0) >= 0

    metadata = get_2stage_cfgs(
        get_padded_M(token),
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.bf16,
        QuantType.No,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.SEPARATED.value,
        opus_weights_shuffled=True,
        input_dtype=dtypes.bf16,
    )
    assert metadata.block_m == 64
    assert metadata.stage1.keywords["kernelName"].startswith("moe_ck2stages_gemm1_")
    assert metadata.stage2.keywords["kernelName"].startswith("moe_ck2stages_gemm2_")

    hidden_states = torch.randn(
        (token, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w1 = torch.randn(
        (expert, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn(
        (expert, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda"
    )
    topk_ids, topk_weight = _make_balanced_topk(token, topk, expert)

    actual = fused_moe(
        hidden_states,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        gate_mode=GateMode.SEPARATED.value,
    )

    assert actual.shape == (token, model_dim)
    assert actual.dtype == dtypes.bf16
    assert torch.isfinite(actual).all()
