# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Default dispatch of gfx950 fp8 block-scale MoE with few rows per expert.

Without a tuned row, per_1x128 took the 1-stage asm kernel for every
token > 32. When token * topk // expert is at most 8, most of each expert's
32-row asm tile is padding, and the CK 2-stage path with block_m 16 is
faster. The heuristic now picks that path for those shapes and keeps the asm
kernel for larger ones. Other archs are unchanged.
"""

import sys

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType
from aiter import dtypes as aiter_dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

_ROCM = torch.cuda.is_available() and bool(torch.version.hip)
_ARCH = torch.cuda.get_device_properties(0).gcnArchName if _ROCM else ""
_GFX950 = _ROCM and "gfx950" in _ARCH


def _clear_dispatch_caches():
    for name in ("chip_info", "aiter.jit.utils.chip_info"):
        mod = sys.modules.get(name)
        if mod is not None:
            mod.get_gfx_custom_op_core.cache_clear()
            mod.get_gfx.cache_clear()
    get_2stage_cfgs.cache_clear()


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # Bypass the tuned CSVs: this is about the default heuristic.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    _clear_dispatch_caches()
    yield
    _clear_dispatch_caches()


def _dispatch(token, model_dim, inter_dim, expert, topk):
    return get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        torch.bfloat16,
        aiter_dtypes.fp8,
        aiter_dtypes.fp8,
        QuantType.per_1x128,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
    )


# (token, model_dim, inter_dim, expert, topk, run_1stage, block_m)
_CASES = [
    # token <= 32 already used the 2-stage path.
    (32, 3072, 1024, 256, 8, False, 16),
    # 2, 4 and 8 rows per expert: 2-stage, block_m 16.
    (64, 3072, 1024, 256, 8, False, 16),
    (128, 3072, 1024, 256, 8, False, 16),
    (256, 3072, 1024, 256, 8, False, 16),
    # 16 and 128 rows per expert: unchanged, 1-stage asm.
    (512, 3072, 1024, 256, 8, True, 32),
    (4096, 3072, 1024, 256, 8, True, 32),
    # The threshold is on rows per expert, not on token: with topk 4, 512
    # tokens are still 8 rows per expert.
    (512, 7168, 2048, 256, 4, False, 16),
    (1024, 7168, 2048, 256, 4, True, 32),
    # 10 experts of 512: 5 rows at 256 tokens, 10 at 512.
    (256, 2560, 640, 512, 10, False, 16),
    (512, 2560, 640, 512, 10, True, 32),
]


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 asm dispatch")
@pytest.mark.parametrize(
    "token,model_dim,inter_dim,expert,topk,run_1stage,block_m", _CASES
)
def test_gfx950_blockscale_default_dispatch(
    token, model_dim, inter_dim, expert, topk, run_1stage, block_m
):
    metadata = _dispatch(token, model_dim, inter_dim, expert, topk)
    assert metadata.run_1stage == run_1stage
    assert metadata.block_m == block_m


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 asm dispatch")
def test_blockscale_default_dispatch_other_arch_unchanged(monkeypatch):
    # The crossover was measured on gfx950 only. Elsewhere 64 tokens with 2
    # rows per expert still take the asm kernel.
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: "gfx942")
    if not aiter.fmoe_fp8_blockscale_g1u1_has_kernel(
        ActivationType.Silu.value, False, 32, 1024, 3072
    ):
        pytest.skip("no 1-stage blockscale kernel for this shape")
    metadata = _dispatch(64, 3072, 1024, 256, 8)
    assert metadata.run_1stage
    assert metadata.block_m == 32


def _quant_128x128(weight):
    experts, dim1, dim2 = weight.shape
    blocks = weight.view(experts, dim1 // 128, 128, dim2 // 128, 128)
    blocks = blocks.permute(0, 1, 3, 2, 4).contiguous().view(experts, -1, 128 * 128)
    qt, scale = aiter.pertoken_quant(blocks, quant_dtype=aiter_dtypes.fp8)
    qt = qt.view(experts, dim1 // 128, dim2 // 128, 128, 128)
    qt = qt.permute(0, 1, 3, 2, 4).contiguous().view(experts, dim1, dim2)
    return qt, scale.view(experts, dim1 // 128, dim2 // 128)


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 asm dispatch")
@pytest.mark.parametrize("token,run_1stage", [(64, False), (256, False), (512, True)])
def test_gfx950_blockscale_few_rows_matches_reference(token, run_1stage):
    model_dim, inter_dim, expert, topk = 1024, 512, 64, 2
    assert _dispatch(token, model_dim, inter_dim, expert, topk).run_1stage == (
        run_1stage
    )
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(token, model_dim, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(expert, 2 * inter_dim, model_dim, device=dev) / 10
    w2 = torch.randn(expert, model_dim, inter_dim, device=dev) / 10
    w1q, w1s = _quant_128x128(w1.to(torch.bfloat16))
    w2q, w2s = _quant_128x128(w2.to(torch.bfloat16))
    score = torch.randn(token, expert, device=dev, dtype=torch.bfloat16)
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    out = fused_moe(
        x,
        shuffle_weight(w1q, layout=(16, 16)),
        shuffle_weight(w2q, layout=(16, 16)),
        topk_weights,
        topk_ids,
        w1_scale=fp4_utils.e8m0_shuffle(w1s),
        w2_scale=fp4_utils.e8m0_shuffle(w2s),
        quant_type=QuantType.per_128x128,
        activation=ActivationType.Silu,
    )

    a1, a1_scale = aiter.pertoken_quant(
        x.view(token, -1, 128), quant_dtype=aiter_dtypes.fp8
    )
    out1 = torch_moe_stage1(
        a1.view(token, model_dim),
        w1q,
        w2q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_128x128,
        a1_scale=a1_scale.squeeze(-1),
        w1_scale=w1s,
    )
    a2, a2_scale = aiter.pertoken_quant(
        out1.view(token, -1, 128), quant_dtype=aiter_dtypes.fp8
    )
    ref = torch_moe_stage2(
        a2.view(token, topk, -1),
        w1q,
        w2q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_128x128,
        w2_scale=w2s,
        a2_scale=a2_scale.view(token, topk, -1),
    )
    assert torch.isfinite(out).all()
    assert _cos_diff(out, ref) < 1e-3
