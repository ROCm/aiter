# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ASM stage-1 candidates of the fused MoE tuner's two-stage path."""

import torch

from aiter import QuantType, dtypes
from aiter.fused_moe import asm_stage1
from aiter.jit.core import get_asm_dir
from aiter.tuning.moe.backends.common import _is_tune_excluded_kernel, get_kernels_dict
from aiter.tuning.moe.compare import run_torch_moe_stage1_ref
from aiter.tuning.moe.inputs import generate_asm_stage1


def run_asm_stage1(
    input,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out,
    a1_scale,
    w1_scale,
    topk,
    block_m,
    kernelName,
    ksplit,
    activation,
    quant_type,
    doweight_stage1,
):
    if not doweight_stage1:
        sorted_weights = None
    asm_stage1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        block_m,
        kernelName,
        ksplit,
        activation,
        quant_type,
        a1_scale,
        w1_scale,
        sorted_weights,
    )
    return out


def gen_2stages_asm1_task(key, blockMs):
    info = key
    tasks = []
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
    kernels_list_csv = f"{get_asm_dir()}/fmoe_2stages/fmoe_stage1_bf16_pertoken{{quantDtype}}{{extraInfo}}_g1u1.csv"
    extraInfo = ""
    if q_type == QuantType.per_1x128:
        extraInfo += "_blockscale"
    if doweight_stage1:
        extraInfo += "_doweight"

    if q_dtype_a == dtypes.i8:
        quantDtype = "Int8"
    elif q_dtype_a == dtypes.fp8:
        quantDtype = "Fp8"
    else:
        quantDtype = ""
    asm_kernels = get_kernels_dict(
        kernels_list_csv.format(quantDtype=quantDtype, extraInfo=extraInfo)
    )
    for blockM in blockMs:
        # per_1x32 + fp4x2 is a8w4 (MX-FP8 act + MX-FP4 weight); no ASM kernel exists
        # for this combo -- the pertokenFp8 CSV only covers per_Token quant.
        if (
            use_g1u1
            and q_dtype_w != torch.int4
            and not (q_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp4x2)
        ):
            for el in asm_kernels.get(blockM, []):
                if _is_tune_excluded_kernel(el):
                    continue
                tasks.append(
                    (
                        (info, "stage1", el, blockM),  # tag
                        generate_asm_stage1,
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
                        ),
                        run_asm_stage1,  # func
                        (
                            [
                                "a1_qt",
                                "w1_qt_shffle",
                                "w2_qt_shffle",
                                "sorted_ids",
                                "sorted_expert_ids",
                                "sorted_weights",
                                "num_valid_ids",
                                "out1",
                                "a1_scale_t",
                                "w1_scale",
                            ],
                            topk,
                            blockM,
                            el,
                            0,
                            act_type,
                            q_type,
                            doweight_stage1,
                        ),
                        {},
                        run_torch_moe_stage1_ref,
                        (
                            [
                                "a1_qt",
                                "w1_qt",
                                "w2_qt",
                                "topk_weights",
                                "topk_ids",
                                "a1_scale",
                                "w1_scale",
                            ],
                            dtype,
                            act_type,
                            q_type,
                            doweight_stage1,
                            topk,
                        ),
                        {},
                        (None),
                        0.01,
                        0.01,
                        True,
                    )
                )
    return tasks
