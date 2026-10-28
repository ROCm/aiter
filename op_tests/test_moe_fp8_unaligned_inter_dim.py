# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test per-tensor FP8 CK MoE shards whose inter_dim is not a multiple of 256."""

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950",
    reason="Exercises the gfx950 per-tensor FP8 CK stage-2 K padding",
)

# FP8 activations alone give about 0.05 relative error. A wrong kernel is far above it.
RELATIVE_ERROR_TOLERANCE = 0.1


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # Bypass the tuned CSV so that a tuned kernel cannot hide a dispatch bug.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def compute_relative_error(tokens, model_dim, inter_dim, experts, topk, block_m):
    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device="cuda", dtype=dtypes.bf16) / 10
    w1 = (
        torch.randn(experts, 2 * inter_dim, model_dim, device="cuda", dtype=dtypes.bf16)
        / 10
    )
    w2 = (
        torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=dtypes.bf16)
        / 10
    )
    # One scale per expert is per-tensor quantization.
    w1_q, w1_scale = aiter.pertoken_quant(w1.view(experts, -1), quant_dtype=dtypes.fp8)
    w2_q, w2_scale = aiter.pertoken_quant(w2.view(experts, -1), quant_dtype=dtypes.fp8)
    w1_q, w2_q = w1_q.view(w1.shape), w2_q.view(w2.shape)
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    actual = fused_moe(
        x,
        shuffle_weight(w1_q, layout=(16, 16)),
        shuffle_weight(w2_q, layout=(16, 16)),
        weights,
        ids.to(torch.int32),
        activation=ActivationType.Silu,
        quant_type=QuantType.per_Tensor,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_size_M=block_m,
    )
    torch.accelerator.synchronize()
    # Compute the reference from the dequantized weights in FP32.
    w1_dq = w1_q.float() * w1_scale.view(experts, 1, 1)
    w2_dq = w2_q.float() * w2_scale.view(experts, 1, 1)
    expected = torch.zeros(tokens, model_dim, device="cuda", dtype=torch.float32)
    for expert in range(experts):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1_dq[expert]).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, w2_dq[expert])
        expected.index_add_(0, rows, y * weights[rows, slots, None])
    assert torch.isfinite(actual).all(), "The fused MoE output has NaN or Inf values."
    return ((actual.float() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("tokens", [33])
@pytest.mark.parametrize("model_dim", [4096])
@pytest.mark.parametrize("inter_dim", [256, 512, 1024])
@pytest.mark.parametrize("experts", [32])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("block_m", [32, 64, 128])
def test_per_tensor_fp8_aligned_inter_dim(
    tokens, model_dim, inter_dim, experts, topk, block_m
):
    # These shapes are multiples of 256, so they never needed padding.
    err = compute_relative_error(tokens, model_dim, inter_dim, experts, topk, block_m)
    assert (
        err < RELATIVE_ERROR_TOLERANCE
    ), f"The relative error {err:.4f} is above {RELATIVE_ERROR_TOLERANCE}."


@pytest.mark.parametrize("tokens", [33])
@pytest.mark.parametrize("model_dim", [4096])
@pytest.mark.parametrize("inter_dim", [64, 192, 384, 640, 896])
@pytest.mark.parametrize("experts", [32])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("block_m", [32, 64, 128])
def test_per_tensor_fp8_unaligned_inter_dim(
    tokens, model_dim, inter_dim, experts, topk, block_m
):
    # None of these shapes is a multiple of 256, the K of the largest stage-2 tile.
    # 384 is the TP4 shard of Qwen3-235B-A22B-FP8, which could not be served before.
    err = compute_relative_error(tokens, model_dim, inter_dim, experts, topk, block_m)
    assert (
        err < RELATIVE_ERROR_TOLERANCE
    ), f"The relative error {err:.4f} is above {RELATIVE_ERROR_TOLERANCE}."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
