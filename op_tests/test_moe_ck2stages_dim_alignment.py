# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression for the CK 2-stage MoE heuristic dispatch.

Covers an inter_dim that is 64- but not 128-aligned (e.g. the 320/448 shards
produced by TP splitting) and a model_dim with the same alignment (stage2 N).

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

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight


def _ck_default_arch() -> bool:
    if not torch.cuda.is_available() or not torch.version.hip:
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    return arch in ("gfx942", "gfx950")


def _arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


pytestmark = pytest.mark.skipif(
    not _ck_default_arch(),
    reason="CK 2-stage A16W16/A8W8 default heuristics are generated for gfx942 and gfx950",
)


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # A tuned CSV row must not be able to hide a broken default heuristic.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _torch_moe(x, w1, w2, topk_weights, topk_ids):
    """Silu g1u1 MoE. Stage1 is rounded back to the kernel dtype before stage2."""
    token, model_dim = x.shape
    dtype = x.dtype
    out = torch.zeros((token, model_dim), dtype=torch.float32, device=x.device)
    for e in range(w1.shape[0]):
        rows, slots = torch.where(topk_ids == e)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1[e].float()).chunk(2, dim=-1)
        hidden = (F.silu(gate) * up).to(dtype)
        y = F.linear(hidden.float(), w2[e].float())
        out.index_add_(0, rows, y * topk_weights[rows, slots, None].float())
    return out


def _check_moe(token, model_dim, inter_dim, expert, topk, block_m):
    torch.manual_seed(0)
    dtype = dtypes.bf16
    x = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    # Same scale as test_moe_a16w16_default_dispatch.py, so rtol/atol 0.02/0.002
    # compares the bf16 stage1 round-trip instead of fp32 accumulation noise.
    w1 = (
        torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
        * 0.02
    )
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype, device="cuda") * 0.02
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
    torch.testing.assert_close(
        out.float(),
        ref,
        rtol=0.02,
        atol=0.002,
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


@pytest.mark.parametrize("block_m", [64, 128, 256])
def test_a16w16_model_dim_not_128_aligned(block_m):
    # 2880 % 128 != 0 and 2880 % 64 == 0. inter_dim 256 is 128-aligned, so a
    # stage2 heuristic that only looks at inter_dim still picks NPerBlock=128.
    _check_moe(8, 2880, 256, 4, 2, block_m)


def _quant_weights(w, quant_type, quant_dtype):
    if quant_type == QuantType.per_Tensor:
        flat, scale = aiter.pertoken_quant(
            w.view(w.shape[0], -1), quant_dtype=quant_dtype
        )
        return flat.view(w.shape), scale
    torch_quant = aiter.get_torch_quant(quant_type)
    return torch_quant(w, quant_dtype=quant_dtype)


def _check_a8w8(token, model_dim, inter_dim, expert, topk, block_m, quant_type):
    torch.manual_seed(0)
    dtype = dtypes.bf16
    quant_dtype = dtypes.fp8
    x = torch.randn((token, model_dim), dtype=dtype, device="cuda") * 0.02
    w1 = (
        torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
        * 0.02
    )
    w2 = (
        torch.randn((expert, model_dim, inter_dim), dtype=dtype, device="cuda") * 0.02
    )
    score = torch.randn((token, expert), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    w1_qt, w1_scale = _quant_weights(w1, quant_type, quant_dtype)
    w2_qt, w2_scale = _quant_weights(w2, quant_type, quant_dtype)
    torch_quant = aiter.get_torch_quant(quant_type)
    a1_qt, a1_scale = torch_quant(x, quant_dtype=quant_dtype)
    stage1 = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=quant_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=False,
    )
    a2_qt, a2_scale = torch_quant(stage1, quant_dtype=quant_dtype)
    ref = torch_moe_stage2(
        a2_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=quant_type,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )
    return fused_moe(
        x,
        shuffle_weight(w1_qt),
        shuffle_weight(w2_qt),
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=quant_type,
        block_size_M=block_m,
    ), ref


def _gfx950_a8w8_k_align(block_m):
    # A8W8_gemm2_gfx950: block_m 32/64 require inter_dim % 256, else % 128.
    return 256 if block_m in (32, 64) else 128


@pytest.mark.parametrize("quant_type", [QuantType.per_Tensor, QuantType.per_Token])
@pytest.mark.parametrize("inter_dim", [256, 320])
@pytest.mark.parametrize("block_m", [32, 128])
def test_a8w8_dim_alignment(quant_type, inter_dim, block_m):
    # per-token fp8 with inter_dim % 128 != 0 is sent to the asm 1-stage
    # kernel, so it never reaches the CK heuristic this test covers.
    if quant_type == QuantType.per_Token and inter_dim % 128 != 0:
        pytest.skip("per-token fp8 with an unaligned inter_dim uses asm 1-stage")
    # Keep per-token on the CK path: token > 16 also selects asm 1-stage.
    token = 4 if quant_type == QuantType.per_Token else 33
    if (
        _arch() == "gfx950"
        and quant_type == QuantType.per_Tensor
        and inter_dim % _gfx950_a8w8_k_align(block_m) != 0
    ):
        align = _gfx950_a8w8_k_align(block_m)
        with pytest.raises(RuntimeError, match=rf"inter_dim % {align} == 0"):
            _check_a8w8(token, 256, inter_dim, 8, 2, block_m, quant_type)
        return
    out, ref = _check_a8w8(token, 256, inter_dim, 8, 2, block_m, quant_type)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        rtol=0.05,
        atol=0.05,
        msg=f"a8w8 {quant_type} inter_dim={inter_dim} block_m={block_m}",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
