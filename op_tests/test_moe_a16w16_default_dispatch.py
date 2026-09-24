# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression for A16W16 CK MoE shards aligned to 64, but not 128 (e.g. 320)."""

import pytest
import torch
import torch.nn.functional as F

from aiter.fused_moe import fused_moe, get_2stage_cfgs
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="Exercises the A16W16 CK default dispatch",
)


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # Explicitly bypass CSV tuning: a tuned kernel must not hide this bug.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _check_moe(tokens, model_dim, inter_dim, experts, topk, block_m, dtype):
    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device="cuda", dtype=dtype)
    w1 = (
        torch.randn(experts, 2 * inter_dim, model_dim, device="cuda", dtype=dtype)
        * 0.02
    )
    w2 = torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=dtype) * 0.02
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    ids = ids.to(torch.int32)
    actual = fused_moe(
        x, shuffle_weight(w1), shuffle_weight(w2), weights, ids, block_size_M=block_m
    )
    torch.cuda.synchronize()
    # Independent reference over ALL tokens and experts; accumulate in FP32.
    expected = torch.zeros(tokens, model_dim, device="cuda", dtype=torch.float32)
    for expert in range(experts):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1[expert].float()).chunk(2, dim=-1)
        hidden = (F.silu(gate) * up).to(dtype)
        y = F.linear(hidden.float(), w2[expert].float())
        expected.index_add_(0, rows, y * weights[rows, slots, None])
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("inter_dim", [192, 256, 320, 384, 448])
@pytest.mark.parametrize("block_m", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a16w16_ck_default_alignment(inter_dim, block_m, dtype):
    # 320/448 regress the missing alignment check; 192/256/384 retain old paths.
    _check_moe(33, 256, inter_dim, 8, 2, block_m, dtype)


@pytest.mark.parametrize("tokens", [4, 512, 8192])
def test_qwen_mtp_tp2_ep1_default_dispatch(tokens):
    # Qwen3.8-Flash-Next MTP TP=2/EP=1 shape, without a block-size override.
    _check_moe(tokens, 2560, 320, 512, 10, None, torch.bfloat16)
