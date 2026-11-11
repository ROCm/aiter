# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus candidates of the fused MoE tuner's two-stage path."""

from aiter import ActivationType, QuantType, dtypes
from aiter.test_common import cosine_diff_compare
from aiter.tuning.moe.compare import (
    run_torch_moe_stage1,
    run_torch_moe_stage2,
    sorted_stage1_cosine_diff_compare,
)
from aiter.tuning.moe.inputs import (
    generate_data_2stages,
    generate_opus_sorted_stage2_data,
)

# Lives in csrc/, so the repo root must be on sys.path, as for the tuner itself.
from csrc.opus_moe.opus_moe_common import (
    OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT,
    get_opus_a8w4_stage2_kernels,
    opus_a8w4_stage1_instances_for_shape,
)


def run_opus_stage1_out(
    a1_qt,
    w1_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    w1_scale_aiter,
    a1_scale_e8m0_one_sort,
    bias,
    topk,
    kparams,
    blockM,
    act_type,
    output_sorted=False,
):
    from aiter.ops.opus.moe_stage1_a8w4 import opus_moe_stage1_a8w4_fwd

    out, _out_scale = opus_moe_stage1_a8w4_fwd(
        a1_qt,
        w1_qt_shffle_ck,
        a1_scale_e8m0_one_sort,
        w1_scale_aiter,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk=topk,
        block_m=blockM,
        kernelName=str(kparams["kernelName"]),
        activation=act_type,
        inter_dim_pad=0,
        bias=bias,
        output_sorted=output_sorted,
    )
    return out


def run_opus_stage2_out(
    a2_qt,
    w2_qt,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w2_scale,
    a2_scale,
    moe_buf,
    bias,
    dtype,
    topk,
    kparams,
    blockM,
    q_type,
    act_type,
):
    # opus a8w4 stage2 (gfx950 decode family). kparams from
    # get_opus_a8w4_stage2_kernels: kid + kernel_block_m (B_M) + route_out.
    # The tuner's blockM is the moe_sorting block_m (== the kid's
    # SORT_BLOCK_M); the kernel's own B_M is kparams["kernel_block_m"].
    from aiter.ops.opus.moe_stage2_a8w4 import (
        opus_moe_stage2_a8w4_fwd,
        stage2_launch_config,
    )

    block_n = int(kparams.get("kernel_block_n"))
    if block_n <= 0 or w2_qt.shape[1] % block_n != 0:
        raise ValueError(
            "unsupported Opus A8W4 stage2 block_n: "
            f"model_dim={w2_qt.shape[1]}, logical_inter_dim={a2_qt.shape[-1]}, "
            f"expert={w2_qt.shape[0]}, topk={topk}, "
            f"block_n={kparams.get('kernel_block_n')}"
        )
    kid = kparams["kid"]
    # w2_qt / w2_scale here are ALREADY the a16w4 MFMA-tile shuffle:
    # gen_opus_2stages_task feeds "w2_qt_shffle_ck" (shuffle_weight_a16w4)
    # and "w2_scale_aiter" (shuffle_scale_a16w4). opus consumes them as-is.
    w2_use = w2_qt
    w2_scale_opus = w2_scale
    launch = stage2_launch_config(kid)
    if launch.sort_block_m != int(blockM):
        raise ValueError(
            f"Opus A8W4 stage2 kid {kid} requires sort block_m="
            f"{launch.sort_block_m}, got {blockM}"
        )
    if not launch.route_out:
        moe_buf.zero_()
    return opus_moe_stage2_a8w4_fwd(
        a2_qt,
        w2_use,
        a2_scale,
        w2_scale_opus,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        launch=launch,
        out=moe_buf,
        inter_dim_pad=0,
        token_num=int(moe_buf.shape[0]),
        topk=int(topk),
    )


def gen_opus_2stages_task(
    info,
    blockMs,
):
    # Opus A8W4 candidates share CK's a16w4 MFMA shuffle.
    tasks_opus = []
    (
        gfx,
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

    k_step_packed = (
        OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT.bk_logical
        // OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT.fp4_values_per_byte
    )
    # gfx950 + a8w4 (fp8 act / fp4 weight / per_1x32) + g1u1 + supported
    # inter dim. Stage1 kids are filtered by shape and block_m below.
    if not (
        gfx == "gfx950"
        and use_g1u1
        and q_type == QuantType.per_1x32
        and q_dtype_a == dtypes.fp8
        and q_dtype_w == dtypes.fp4x2
        and inter_dim % k_step_packed == 0
    ):
        return tasks_opus

    stage1_insts = opus_a8w4_stage1_instances_for_shape(
        model_dim=model_dim,
        logical_inter_dim=inter_dim,
        inter_dim_pad=0,
    )

    stage1_supported = (
        dtype == dtypes.bf16
        and act_type
        in (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)
        and not doweight_stage1
        and bool(stage1_insts)
    )

    s1_ref_args = (
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
        32,
        False,
        True,
    )

    # fp8-activation stage2 ref (shape-level, shared by all opus kids): torch
    # stage2 on the bf16 stage1 output, sliced by opus_eff to match runtime effective-K.
    s2_ref_args = (
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
    )
    run_keys = [
        "a2_qt",
        "w2_qt_shffle_ck",
        "sorted_ids",
        "sorted_expert_ids",
        "sorted_weights",
        "num_valid_ids",
        "w2_scale_aiter",
        "a2_scale_mxfp4_sort",
        "moe_buf",
        "bias",
    ]
    sorted_run_keys = ["a2_qt_sorted", *run_keys[1:]]

    for blockM in blockMs:
        if blockM not in (16, 32, 64, 128):
            continue
        if stage1_supported:
            for inst in stage1_insts:
                if int(inst.block_m) != int(blockM):
                    continue
                s1_params = {
                    "kernelName": inst.profile_name,
                }
                tasks_opus.append(
                    (
                        (info, "stage1", inst.profile_name, blockM),
                        generate_data_2stages,
                        (
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
                            blockM,
                            1,
                            True,
                        ),
                        run_opus_stage1_out,
                        (
                            [
                                "a1_qt_fp8_cast",
                                "w1_qt_shffle_ck",
                                "sorted_ids",
                                "sorted_expert_ids",
                                "num_valid_ids",
                                "w1_scale_aiter",
                                "a1_scale_e8m0_one_sort",
                                "bias",
                            ],
                            topk,
                            s1_params,
                            blockM,
                            act_type,
                        ),
                        {},
                        run_torch_moe_stage1,
                        s1_ref_args[:-3] + (blockM, False, True),
                        {},
                        None,
                        0.01,
                        0.01,
                        cosine_diff_compare,
                    )
                )
                tasks_opus.append(
                    (
                        (info, "stage1", inst.profile_name, blockM, 0, 1),
                        generate_data_2stages,
                        (
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
                            blockM,
                            1,
                            True,
                        ),
                        run_opus_stage1_out,
                        (
                            [
                                "a1_qt_fp8_cast",
                                "w1_qt_shffle_ck",
                                "sorted_ids",
                                "sorted_expert_ids",
                                "num_valid_ids",
                                "w1_scale_aiter",
                                "a1_scale_e8m0_one_sort",
                                "bias",
                            ],
                            topk,
                            s1_params,
                            blockM,
                            act_type,
                            True,
                        ),
                        {},
                        run_torch_moe_stage1,
                        s1_ref_args[:-3] + (blockM, False, True),
                        {"output_sorted": True},
                        None,
                        0.01,
                        0.01,
                        sorted_stage1_cosine_diff_compare,
                    )
                )
        for kname, kparams in get_opus_a8w4_stage2_kernels(token=token).items():
            # tuner blockM is the moe_sorting block_m; valid only when it
            # equals the kid's SORT_BLOCK_M.
            if kparams["sort_block_m"] != blockM:
                continue
            kparams = dict(kparams)
            block_n = int(kparams.get("kernel_block_n"))
            if block_n <= 0 or model_dim % block_n != 0:
                continue
            gen_args = (
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
                blockM,
                2,
                True,
            )
            run_args = (run_keys, dtype, topk, kparams, blockM, q_type, act_type)
            tasks_opus.append(
                (
                    (info, "stage2", kname, blockM),
                    generate_data_2stages,
                    gen_args,
                    run_opus_stage2_out,
                    run_args,
                    {},
                    run_torch_moe_stage2,
                    s2_ref_args,
                    {},
                    None,
                    0.01,
                    0.01,
                    cosine_diff_compare,
                )
            )
            tasks_opus.append(
                (
                    (
                        info,
                        "stage2",
                        kname.replace("opus_moe2_", "opus_moe2_layout_", 1),
                        blockM,
                        0,
                        1,
                    ),
                    generate_opus_sorted_stage2_data,
                    gen_args,
                    run_opus_stage2_out,
                    (
                        sorted_run_keys,
                        dtype,
                        topk,
                        kparams,
                        blockM,
                        q_type,
                        act_type,
                    ),
                    {},
                    run_torch_moe_stage2,
                    s2_ref_args,
                    {},
                    None,
                    0.01,
                    0.01,
                    cosine_diff_compare,
                )
            )

    return tasks_opus
