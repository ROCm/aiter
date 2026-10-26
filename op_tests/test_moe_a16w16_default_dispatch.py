# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Default-dispatch regressions for CK MoE.

gfx942 A16W16 covers shards aligned to 64 but not 128 (for example 320).
gfx950 fp8 stage2 covers shuffled inter_dim 384 (K=128) and 512 (K=256),
rejects sizes that 128 does not divide, and rejects plain fp8 weights unless
both w1 and w2 are shuffled. gfx950 int8 uses that same stage2 K selection.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter import QuantType
from aiter import dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs, torch_moe
from aiter.ops.moe_op import ck_moe_stage2_fwd
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


def _quantized_weight(weight, quant_type, quant_dtype):
    # per_Tensor is one scale per expert. per_Token is one scale per row.
    source = (
        weight.view(weight.shape[0], -1)
        if quant_type == QuantType.per_Tensor
        else weight
    )
    quantized, scale = pertoken_quant(source, quant_dtype=quant_dtype)
    return quantized.view(weight.shape), scale


def _rows(scale, rows):
    # torch_moe applies one scale per weight row.
    if scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.shape[1] == rows:
        return scale
    return scale.expand(-1, rows, 1).contiguous()


def _gfx950_fp8_inputs(inter_dim, quant_type, quant_dtype=aiter_dtypes.fp8):
    torch.manual_seed(0)
    tokens, experts, topk, model_dim = 16, 8, 2, 896
    hidden = torch.randn(tokens, model_dim, device="cuda", dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            experts, inter_dim * 2, model_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    w2 = (
        torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=torch.bfloat16)
        * 0.02
    )
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    ids = ids.to(torch.int32)
    w1q, w1_scale = _quantized_weight(w1, quant_type, quant_dtype)
    w2q, w2_scale = _quantized_weight(w2, quant_type, quant_dtype)
    return hidden, w1q, w2q, w1_scale, w2_scale, weights, ids


def _call_gfx950_fp8(inter_dim, block_m, quant_type, *, shuffle, a1_scale, inputs=None):
    hidden, w1q, w2q, w1_scale, w2_scale, weights, ids = (
        inputs if inputs is not None else _gfx950_fp8_inputs(inter_dim, quant_type)
    )
    if shuffle:
        w1q_in, w2q_in = shuffle_weight(w1q), shuffle_weight(w2q)
    else:
        w1q_in, w2q_in = w1q, w2q
    return fused_moe(
        hidden,
        w1q_in,
        w2q_in,
        weights,
        ids,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        block_size_M=block_m,
        dtype=torch.bfloat16,
    )


def _check_gfx950_fp8(inter_dim, block_m, quant_type, quant_dtype=aiter_dtypes.fp8):
    inputs = _gfx950_fp8_inputs(inter_dim, quant_type, quant_dtype)
    hidden, w1q, w2q, w1_scale, w2_scale, weights, ids = inputs
    # This kernel reads the shuffled weight layout. The reference keeps the
    # original layout and the same scales.
    actual = _call_gfx950_fp8(
        inter_dim,
        block_m,
        quant_type,
        shuffle=True,
        a1_scale=None,
        inputs=inputs,
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


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize(
    ("quant_type", "static_scale"),
    [
        (QuantType.per_Token, False),
        (QuantType.per_Tensor, True),
        (QuantType.per_Tensor, False),
    ],
)
def test_gfx950_fp8_rejects_unshuffled_weights(inter_dim, quant_type, static_scale):
    # Issue 6452 passes unshuffled weights: per-token, per-tensor with a static
    # activation scale, and per-tensor with a dynamic activation scale.
    a1_scale = (
        torch.tensor([1.0], dtype=torch.float32, device="cuda")
        if static_scale
        else None
    )
    with pytest.raises(RuntimeError, match="shuffle_weight"):
        _call_gfx950_fp8(inter_dim, None, quant_type, shuffle=False, a1_scale=a1_scale)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("shuffle_w1", [True, False])
def test_gfx950_fp8_rejects_partially_shuffled_weights(shuffle_w1):
    # Either mixed pair must fail. is_shuffled on only one weight used to
    # satisfy the guard and let the other stage read the wrong layout.
    hidden, w1q, w2q, w1_scale, w2_scale, weights, ids = _gfx950_fp8_inputs(
        384, QuantType.per_Tensor
    )
    w1_in = shuffle_weight(w1q) if shuffle_w1 else w1q
    w2_in = w2q if shuffle_w1 else shuffle_weight(w2q)
    with pytest.raises(RuntimeError, match="both w1 and w2"):
        fused_moe(
            hidden,
            w1_in,
            w2_in,
            weights,
            ids,
            quant_type=QuantType.per_Tensor,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_size_M=32,
            dtype=torch.bfloat16,
        )


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 int8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize("block_m", [32, 64])
def test_gfx950_int8_stage2_k_alignment(inter_dim, block_m):
    # int8 shares a8w8_gfx950 with fp8. 384 selects K=128. 512 selects K=256.
    _check_gfx950_fp8(
        inter_dim, block_m, QuantType.per_Tensor, quant_dtype=aiter_dtypes.i8
    )


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 int8 CK default dispatch")
def test_gfx950_int8_stage2_rejects_non_128():
    # Int8 fused_moe stage 1 is the ASM kernel, and every gfx950 int8 stage-1
    # tile is 128 or 256, so width 192 fails there and never reaches stage 2.
    # The K check added in this change lives in the shared CK stage-2 dispatch.
    inter_dim, model_dim, tokens, experts = 192, 128, 4, 1
    inter = torch.empty((tokens, inter_dim), dtype=torch.int8, device="cuda")
    w1 = torch.empty(
        (experts, inter_dim * 2, model_dim), dtype=torch.int8, device="cuda"
    )
    w2 = torch.empty((experts, model_dim, inter_dim), dtype=torch.int8, device="cuda")
    sorted_ids = torch.zeros(32, dtype=torch.int32, device="cuda")
    sorted_expert_ids = torch.zeros(4, dtype=torch.int32, device="cuda")
    num_valid = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.empty((tokens, model_dim), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of"):
        ck_moe_stage2_fwd(
            inter,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid,
            out,
            1,
            block_m=32,
            quant_type=QuantType.per_Tensor,
        )
