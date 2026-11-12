# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MoE xcd_swizzle regression on gfx950.

Lives in ``tests/`` because CI runs ``python3 -m pytest -q tests`` on pull requests to
main, while ``op_tests/flydsl_tests`` is not part of the sharded op-test plan.
Its data generator and tolerance check are the a8w4 op test's.
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from op_tests.flydsl_tests.test_flydsl_moe import (
    _check_close,
    _generate_a8w4_gui_data,
    _stage1_tile_k,
)


def test_flydsl_moe_xcd_swizzle_uneven_grid():
    """xcd_swizzle on a 2 x 18 grid in both stages: 36 workgroups, not a multiple
    of the 8 XCDs. Every sorted block holds a routed row, so a tile the remap
    skips leaves NaN in stage 1 and a tile it runs twice doubles a stage-2 sum."""
    # The running GPU, not GPU_ARCHS: get_gfx() takes the last GPU_ARCHS entry.
    if get_gfx_runtime() != "gfx950":
        pytest.skip("gfx950 FlyDSL required")
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    token, model_dim, inter_dim, E, topk, block_m = 9, 512, 256, 32, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, E, topk, block_m, distinct_experts=True
    )
    stage1_out = torch.full(
        (token, topk, inter_dim), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    flydsl_moe_stage1(
        a=data["a_q"],
        w1=data["w1_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        out=stage1_out,
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
        xcd_swizzle=4,
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
        xcd_swizzle=4,
    )
    torch.cuda.synchronize()
    _check_close(data["ref_stage1"], stage1_out, "xcd4_uneven_grid_stage1")
    _check_close(data["ref_stage2"], out, "xcd4_uneven_grid_stage2")
