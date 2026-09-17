# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.fused_moe import _resolve_tuning_topk
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

_DSV41_STAGE1 = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw2_fp8"


def test_ep_tuning_topk_preserves_routes_without_fake_expert():
    assert _resolve_tuning_topk(6, is_ep=True, ep_has_fake_expert=False) == 6
    assert _resolve_tuning_topk(7, is_ep=True, ep_has_fake_expert=True) == 6
    assert _resolve_tuning_topk(6, is_ep=False) == 6


def test_a8w4_stage1_runtime_contract_uses_real_scales():
    params = get_flydsl_kernel_params(_DSV41_STAGE1)
    assert params is not None
    assert params["a_dtype"] == "fp8"
    assert params["b_dtype"] == "fp4"
    assert params["out_dtype"] == "fp8"
    assert params["gate_mode"] == "interleave"
    assert params.get("a_scale_one", False) is False
