# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE a8w4 (fp8 activation, fp4 weight, GUI shuffle) regression tests.

Covers stage2 tile_k auto-resolve for non-256-aligned inter_dim (e.g. DSV4
inter=640) and FlyDSL stage2 / E2E with GUI preshuffle on gfx950.

Usage:
    pytest op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -q
    pytest op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -k tile_k
"""

from __future__ import annotations

import os

import pytest
import torch

from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import (
    pick_flydsl_stage2_tile_k,
    resolve_flydsl_stage2_tile_k,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.test_common import checkAllclose

Q_TYPE = QuantType.per_1x32

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() not in ("gfx950",) or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)


def _inter_pad(inter_dim: int) -> int:
    return ((inter_dim + 255) // 256 * 256) - inter_dim


def _stage1_tile_k(model_dim: int) -> int:
    return 512 if (model_dim % 512 == 0) else 256


def _check_close(ref, out, label, atol=1.0, rtol=0.05, max_err_ratio=0.05):
    assert not out.isnan().any(), f"{label}: output has NaN"
    assert not out.isinf().any(), f"{label}: output has Inf"
    err = checkAllclose(ref, out, msg=label, atol=atol, rtol=rtol)
    assert (
        err == 0 or err <= max_err_ratio
    ), f"{label}: checkAllclose failed (err={err}, max={max_err_ratio})"


def _generate_a8w4_gui_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    seed: int = 0,
    dtype=torch.bfloat16,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inter_pad = _inter_pad(inter_dim)

    inp = torch.randn(token, model_dim, dtype=dtype, device="cuda") / 4
    w1 = torch.randn(E, inter_dim * 2, model_dim, dtype=dtype, device="cuda") / 4
    w2 = torch.randn(E, model_dim, inter_dim, dtype=dtype, device="cuda") / 4
    if inter_pad:
        w1[:, -inter_pad:, :] = 0
        w1[:, inter_dim - inter_pad : inter_dim, :] = 0
        w2[:, :, -inter_pad:] = 0

    score = torch.randn(token, E, dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    a_q, a_scale = per_1x32_f8_scale_f8_quant(
        inp, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(E, model_dim, inter_dim // 2)

    ref_stage1 = torch_moe_stage1(
        a_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=Q_TYPE,
        a1_scale=a_scale,
        w1_scale=w1_scale,
    )
    ref_stage2 = torch_moe_stage2(
        ref_stage1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )

    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        ref_stage1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_q = a2_q.view(token, topk, inter_dim)

    a_scale_sort = mxfp4_moe_sort_fwd(
        a_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=model_dim,
    )
    w1_shuf = shuffle_weight_a16w4(w1_q, 16, True)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, True)
    w2_shuf = shuffle_weight_a16w4(w2_q, 16, False)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)
    a2_scale_sort = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=inter_dim,
    )

    return dict(
        inter_pad=inter_pad,
        topk=topk,
        a_q=a_q,
        a_scale_sort=a_scale_sort,
        w1_shuf=w1_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_shuf=w2_shuf,
        w2_scale_shuf=w2_scale_shuf,
        a2_q=a2_q,
        a2_scale_sort=a2_scale_sort,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        ref_stage1=ref_stage1,
        ref_stage2=ref_stage2,
        token=token,
        inter_dim=inter_dim,
        model_dim=model_dim,
    )


@pytest.fixture(autouse=True)
def _a8w4_env():
    old_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    old_aot = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    yield
    if old_bound is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old_bound
    if old_aot is None:
        os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    else:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = old_aot


def test_pick_flydsl_stage2_tile_k():
    assert pick_flydsl_stage2_tile_k(256) == 256
    assert pick_flydsl_stage2_tile_k(512) == 256
    assert pick_flydsl_stage2_tile_k(640) == 128
    assert pick_flydsl_stage2_tile_k(384) == 128
    assert pick_flydsl_stage2_tile_k(896) == 128
    assert pick_flydsl_stage2_tile_k(1024) == 256
    assert resolve_flydsl_stage2_tile_k(640, 256) == 128
    assert resolve_flydsl_stage2_tile_k(256, 256) == 256
    assert resolve_flydsl_stage2_tile_k(512, 128) == 128


@pytest.mark.parametrize(
    "inter_dim,seed",
    [
        pytest.param(256, 101, id="i256"),
        pytest.param(384, 102, id="i384"),
        pytest.param(640, 0, id="i640_dsv4"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_a8w4_gui(inter_dim, seed):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    token, model_dim, E, topk, block_m = 16, 512, 8, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, E, topk, block_m, seed=seed
    )
    out = flydsl_moe_stage2(
        inter_states=data["a2_q"],
        w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=data["a2_scale_sort"],
        sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    _check_close(data["ref_stage2"], out, f"stage2_a8w4_gui_i{inter_dim}")


@pytest.mark.parametrize("inter_dim", [640])
@_SKIP_GFX950_FLYDSL
def test_flydsl_e2e_a8w4_gui(inter_dim):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    token, model_dim, E, topk, block_m, seed = 16, 512, 8, 2, 32, 0
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, E, topk, block_m, seed=seed
    )
    stage1_out = flydsl_moe_stage1(
        a=data["a_q"],
        w1=data["w1_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=_stage1_tile_k(model_dim),
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        act="swiglu",
        gate_mode="interleave",
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a_scale_sort"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        stage1_out, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_q = a2_q.view(token, topk, inter_dim)
    a2_scale_sort = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=data["sorted_ids"],
        num_valid_ids=data["num_valid_ids"],
        token_num=token,
        cols=inter_dim,
    )
    out = flydsl_moe_stage2(
        inter_states=a2_q,
        w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=a2_scale_sort,
        sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    _check_close(data["ref_stage2"], out, f"e2e_a8w4_gui_i{inter_dim}")
