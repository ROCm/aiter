# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Default-dispatch regressions for CK MoE.

gfx942 A16W16 covers shards aligned to 64 but not 128 (for example 320).
gfx950 fp8 stage2 covers inter_dim 384 (K=128) and 512 (K=256), and rejects
sizes that 128 does not divide.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter import QuantType
from aiter import dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs, torch_moe
from aiter.ops.quant import pertoken_quant
from aiter.ops.shuffle import shuffle_weight

_ROCM = torch.cuda.is_available() and bool(torch.version.hip)
_ARCH = torch.cuda.get_device_properties(0).gcnArchName if _ROCM else ""
_GFX942 = _ROCM and "gfx942" in _ARCH
_GFX950 = _ROCM and "gfx950" in _ARCH


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


@pytest.mark.skipif(
    not _GFX942, reason="Exercises the gfx942 A16W16 CK default dispatch"
)
@pytest.mark.parametrize("inter_dim", [192, 256, 320, 384, 448])
@pytest.mark.parametrize("block_m", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a16w16_ck_default_alignment(inter_dim, block_m, dtype):
    # 320/448 regress the missing alignment check; 192/256/384 retain old paths.
    _check_moe(33, 256, inter_dim, 8, 2, block_m, dtype)


@pytest.mark.skipif(
    not _GFX942, reason="Exercises the gfx942 A16W16 CK default dispatch"
)
@pytest.mark.parametrize("tokens", [4, 512, 8192])
def test_qwen_mtp_tp2_ep1_default_dispatch(tokens):
    # Qwen3.8-Flash-Next MTP TP=2/EP=1 shape, without a block-size override.
    _check_moe(tokens, 2560, 320, 512, 10, None, torch.bfloat16)


def _fp8_weight(weight, quant_type):
    # per_Tensor is one scale per expert. per_Token is one scale per row.
    source = (
        weight.view(weight.shape[0], -1)
        if quant_type == QuantType.per_Tensor
        else weight
    )
    quantized, scale = pertoken_quant(source, quant_dtype=aiter_dtypes.fp8)
    return quantized.view(weight.shape), scale


def _rows(scale, rows):
    # torch_moe applies one scale per weight row.
    if scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.shape[1] == rows:
        return scale
    return scale.expand(-1, rows, 1).contiguous()


def _check_gfx950_fp8(inter_dim, block_m, quant_type):
    torch.manual_seed(0)
    tokens, experts, topk, model_dim = 16, 8, 2, 896
    hidden = torch.randn(tokens, model_dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(
        experts, inter_dim * 2, model_dim, device="cuda", dtype=torch.bfloat16
    )
    w2 = torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=torch.bfloat16)
    w1 = w1 * 0.02
    w2 = w2 * 0.02
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    ids = ids.to(torch.int32)
    w1q, w1_scale = _fp8_weight(w1, quant_type)
    w2q, w2_scale = _fp8_weight(w2, quant_type)
    # This kernel reads the shuffled weight layout. The reference keeps the
    # original layout and the same scales.
    actual = fused_moe(
        hidden,
        shuffle_weight(w1q),
        shuffle_weight(w2q),
        weights,
        ids,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_size_M=block_m,
        dtype=torch.bfloat16,
    ).clone()
    expected = torch_moe(
        hidden,
        w1q,
        w2q,
        weights,
        ids,
        fc1_scale=_rows(w1_scale, w1q.shape[1]),
        fc2_scale=_rows(w2_scale, w2q.shape[1]),
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.02)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize("block_m", [32, 64])
@pytest.mark.parametrize("quant_type", [QuantType.per_Tensor, QuantType.per_Token])
def test_gfx950_fp8_stage2_k_alignment(inter_dim, block_m, quant_type):
    # 384 selects the K=128 tile. 512 selects K=256 for block_m 32 and 64.
    _check_gfx950_fp8(inter_dim, block_m, quant_type)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [192, 320])
def test_gfx950_fp8_stage2_rejects_non_128(inter_dim):
    with pytest.raises(RuntimeError, match="multiple of"):
        _check_gfx950_fp8(inter_dim, 32, QuantType.per_Tensor)
