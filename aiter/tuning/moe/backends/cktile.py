# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CK-tile candidates of the fused MoE tuner's two-stage path."""

import torch

from aiter import dtypes
from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2
from aiter.test_common import cosine_diff_compare
from aiter.tuning.moe.compare import run_torch_moe_stage1, run_torch_moe_stage2
from aiter.tuning.moe.inputs import generate_data_2stages


def cktile_moe_stage1_out(
    a1_fp8,
    w1_qt_shffle_ck,
    w2_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w1_scale_aiter,
    bias,
    dtype,
    topk,
    blockM,
    act_type,
):
    M_sorted = sorted_ids.shape[0]
    model_dim = a1_fp8.shape[1]
    a1_scale = torch.ones(
        (M_sorted, model_dim // 32), dtype=dtypes.fp8_e8m0, device=a1_fp8.device
    )
    return cktile_moe_stage1(
        a1_fp8,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        None,
        topk,
        blockM,
        a1_scale=a1_scale,
        w1_scale=w1_scale_aiter.view(dtypes.fp8_e8m0),
        sorted_weights=sorted_weights,
        bias1=bias,
        activation=act_type,
        split_k=1,
        dtype=dtype,
    )


def cktile_moe_stage2_out(
    a2_qt,
    w1_qt_shffle_ck,
    w2_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w2_scale_aiter,
    a2_scale_sort,
    bias,
    dtype,
    topk,
    blockM,
    act_type,
):
    token_num = a2_qt.shape[0]
    model_dim = w2_qt_shffle_ck.shape[1]
    out = torch.zeros(
        (token_num, model_dim),
        dtype=dtype,
        device=a2_qt.device,
    )
    return cktile_moe_stage2(
        a2_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        w2_scale=w2_scale_aiter.view(dtypes.fp8_e8m0),
        a2_scale=a2_scale_sort,
        block_m=blockM,
        activation=act_type,
        sorted_weights=sorted_weights,
        bias2=bias,
    )


def _gen_2stages_task_cktile(info, blockMs):
    """A8W4 (fp8 activation + fp4 weight + per_1x32) uses cktile path."""
    tasks_ck = []
    (
        _gfx,
        _cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        doweight_stage1,
    ) = info

    _gen_data_args_s1 = (
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        doweight_stage1,
    )
    _gen_data_args_s2 = (
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        doweight_stage1,
    )

    for blockM in blockMs:
        if blockM not in [32, 64] or not use_g1u1:
            continue

        cktile_s1_name = f"cktile_a8w4_bm{blockM}"
        tasks_ck.append(
            (
                (info, "stage1", cktile_s1_name, blockM),
                generate_data_2stages,
                (*_gen_data_args_s1, blockM, 1),
                cktile_moe_stage1_out,
                (
                    [
                        "a1_qt_fp8_cast",
                        "w1_qt_shffle_ck",
                        "w2_qt_shffle_ck",
                        "sorted_ids",
                        "sorted_expert_ids",
                        "sorted_weights",
                        "num_valid_ids",
                        "w1_scale_aiter",
                        "bias",
                    ],
                    dtype,
                    topk,
                    blockM,
                    act_type,
                ),
                {},
                run_torch_moe_stage1,
                (
                    [
                        "a1_qt",
                        "w1_qt",
                        "w2_qt",
                        "topk_weights",
                        "topk_ids",
                        "a1_scale",
                        "w1_scale",
                        "sorted_ids",
                        "num_valid_ids",
                        "bias",
                    ],
                    dtype,
                    act_type,
                    q_type,
                    doweight_stage1,
                    topk,
                    blockM,
                ),
                {},
                (None),
                0.01,
                0.01,
                cosine_diff_compare,
            )
        )

        cktile_s2_name = f"cktile_a8w4_bm{blockM}"
        tasks_ck.append(
            (
                (info, "stage2", cktile_s2_name, blockM),
                generate_data_2stages,
                (*_gen_data_args_s2, blockM, 2),
                cktile_moe_stage2_out,
                (
                    [
                        "a2_qt",
                        "w1_qt_shffle_ck",
                        "w2_qt_shffle_ck",
                        "sorted_ids",
                        "sorted_expert_ids",
                        "sorted_weights",
                        "num_valid_ids",
                        "w2_scale_aiter",
                        "a2_scale_mxfp4_sort",
                        "bias",
                    ],
                    dtype,
                    topk,
                    blockM,
                    act_type,
                ),
                {},
                run_torch_moe_stage2,
                (
                    [
                        "ref1_bf16",
                        "w1_qt",
                        "w2_qt",
                        "topk_weights",
                        "topk_ids",
                        "a2_scale_none",
                        "w2_scale",
                        "bias",
                    ],
                    dtype,
                    q_type,
                    doweight_stage1,
                ),
                {},
                (None),
                0.01,
                0.01,
                cosine_diff_compare,
            )
        )

    return tasks_ck
