# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression for the CK 2-stage MoE heuristic dispatch on an inter_dim that is
64- but not 128-aligned (e.g. the 320/448 shards produced by TP splitting).

All ck2stages instances are built with GemmSpecialization::Default, and
DeviceMoeGemm::IsSupportedArgument() rejects `N % NPerBlock != 0` /
`K % KPerBlock != 0` regardless of the specialization, so the heuristic must
fall back to a narrower tile for those shapes instead of handing CK a problem
it cannot divide (which surfaces as an opaque
"device_gemm ... does not support this GEMM problem").
"""

import pytest
import torch
import torch.nn.functional as F

from aiter import dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="ck2stages moe requires a ROCm device",
)


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # A tuned CSV row must not be able to hide a broken default heuristic.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _torch_moe(x, w1, w2, topk_weights, topk_ids):
    """Silu g1u1 MoE in fp32, independent of the kernel under test."""
    token, model_dim = x.shape
    out = torch.zeros((token, model_dim), dtype=torch.float32, device=x.device)
    for e in range(w1.shape[0]):
        rows, slots = torch.where(topk_ids == e)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1[e].float()).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, w2[e].float())
        out.index_add_(0, rows, y * topk_weights[rows, slots, None].float())
    return out


def _check_moe(token, model_dim, inter_dim, expert, topk, block_m):
    torch.manual_seed(0)
    dtype = dtypes.bf16
    x = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, expert), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    ref = _torch_moe(x, w1, w2, topk_weights, topk_ids)
    out = fused_moe(
        x,
        shuffle_weight(w1),
        shuffle_weight(w2),
        topk_weights,
        topk_ids,
        block_size_M=block_m,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    checkAllclose(
        ref,
        out.float(),
        rtol=0.05,
        atol=0.05,
        msg=f"inter_dim={inter_dim} block_m={block_m}",
    )


# 320/448 are 64- but not 128-aligned and regress both the stage1 (N) and the
# stage2 (K) tile choice; 256/384 keep the wide-tile path alive.
@pytest.mark.parametrize("inter_dim", [256, 320, 384, 448])
@pytest.mark.parametrize("block_m", [32, 64, 128, 256])
def test_a16w16_dim_alignment(inter_dim, block_m):
    _check_moe(65, 512, inter_dim, 8, 2, block_m)


@pytest.mark.parametrize("token", [4, 1024])
def test_a16w16_tp_sharded_shape(token):
    # inter_dim=640 split by TP=2, block_m left to the default heuristic.
    _check_moe(token, 2560, 320, 16, 4, None)
