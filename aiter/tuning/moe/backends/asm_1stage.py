# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ASM one-stage candidates of the fused MoE tuner."""

import os

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import AsmMoeMode, torch_moe_2stages, torch_moe_blockscale
from aiter.jit.core import get_asm_dir
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import _run_moe_reduction
from aiter.tuning.moe.backends.common import get_kernels_dict
from aiter.tuning.moe.inputs import _GEN_DATA_1STAGE_KEYS, generate_data_1stage


def _manifest_flat_by_kernel(df: pd.DataFrame) -> dict:
    """Map names to ASM mode: 0 sorted, 1 no_sort, 2 emsort, 3 post_reduce."""
    if "flat" not in df.columns:
        return {}
    return dict(
        zip(df["knl_name"], df["flat"].fillna(AsmMoeMode.SORTED.value).astype(int))
    )


# do weight at stage1
def run_1stage_fmoe_g1u1_tkw1(
    hidden_states,
    a1,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    a1_scale,
    w1_scale,
    w2_scale,
    fc2_smooth_scale=None,
    quant_type=QuantType.No,
    isG1U1=False,
    activation=ActivationType.Silu,
    kernel_name="",
    topk=2,
    dtype=dtypes.bf16,
):
    moe_buf = torch.zeros(
        (a1.shape[0], a1.shape[1]),
        dtype=dtype,
        device="cuda",
    )
    aiter.fmoe_g1u1_tkw1(
        moe_buf,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        w1_scale,
        w2_scale,
        kernel_name,
        fc2_smooth_scale=fc2_smooth_scale,
        activation=activation,
    )
    return moe_buf


def run_1stage_fmoe_fp8_blockscale_g1u1(
    hidden_states,
    a1,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    a1_scale,
    w1_scale,
    w2_scale,
    fc2_smooth_scale=None,
    quant_type=QuantType.No,
    isG1U1=False,
    activation=ActivationType.Silu,
    kernel_name="",
    topk=2,
    dtype=dtypes.bf16,
):
    moe_buf = torch.zeros(
        (a1.shape[0], a1.shape[1]),
        dtype=dtype,
        device="cuda",
    )
    aiter.fmoe_fp8_blockscale_g1u1(
        moe_buf,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        w1_scale,
        w2_scale,
        kernel_name,
        fc2_smooth_scale=fc2_smooth_scale,
        activation=activation,
    )
    return moe_buf


def run_1stage_fmoe_g1u1(
    hidden_states,
    a1,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    a1_scale,
    w1_scale,
    w2_scale,
    fc2_smooth_scale=None,
    quant_type=QuantType.No,
    isG1U1=False,
    activation=ActivationType.Silu,
    kernel_name="",
    topk=2,
    dtype=dtypes.bf16,
    flat: AsmMoeMode | int = AsmMoeMode.SORTED,
):
    post_reduce = AsmMoeMode(flat) is AsmMoeMode.POST_REDUCE
    token_num, model_dim = hidden_states.shape
    moe_buf = (torch.empty if post_reduce else torch.zeros)(
        (token_num, model_dim) if post_reduce else (a1.shape[0], a1.shape[1]),
        dtype=dtype,
        device=a1.device,
    )
    if post_reduce:
        partials = torch.empty(
            (token_num, topk, model_dim), dtype=dtype, device=a1.device
        )
        producer_output = partials.view(token_num, -1)
    else:
        producer_output = moe_buf
    aiter.fmoe_g1u1(
        producer_output,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        w1_scale,
        w2_scale,
        kernel_name,
        fc2_smooth_scale=fc2_smooth_scale,
        activation=activation,
    )
    if post_reduce:
        _run_moe_reduction(
            partials, moe_buf, token_num, topk, model_dim, streaming_loads=True
        )
    return moe_buf


def get_1stage_fmoe_func(quant_type, q_dtype_a, activation, isG1U1, doweight_stage1):
    fmoe_func = None
    if (
        quant_type == QuantType.No
        and activation == ActivationType.Silu
        and not isG1U1
        or quant_type == QuantType.per_1x32
    ):
        print("not support No Quant Silu G1U0 1 stage or per_1x32 quant tuning!")
    else:
        if quant_type == QuantType.per_1x128:
            fmoe_func = run_1stage_fmoe_fp8_blockscale_g1u1
        elif (q_dtype_a == dtypes.fp8) and doweight_stage1:
            fmoe_func = run_1stage_fmoe_g1u1_tkw1
        elif isG1U1:
            fmoe_func = run_1stage_fmoe_g1u1

    return fmoe_func


def get_1stage_file_info(q_type, q_dtype_a, doweight_stage1):
    if get_gfx() == "gfx950":
        extraInfo_1stage = ""
        if q_dtype_a == dtypes.i8:
            quantDtype = "Int8"
        elif q_dtype_a == dtypes.fp8:
            quantDtype = "Fp8"
        else:
            quantDtype = ""
        if doweight_stage1:
            extraInfo_1stage = "_tkw1"
        if q_type == QuantType.No:
            quantDtype_1stage = "noquant"
        elif q_type == QuantType.per_1x128:
            quantDtype_1stage = "blockscale" + quantDtype
        elif q_type == QuantType.per_1x32:
            quantDtype_1stage = "pertoken" + "MXfp4"
        else:
            quantDtype_1stage = "pertoken" + quantDtype
        return quantDtype_1stage, extraInfo_1stage
    elif get_gfx() == "gfx942":
        extraInfo_1stage = ""
        if q_dtype_a == dtypes.i8:
            quantDtype = "Int8"
        elif q_dtype_a == dtypes.fp8:
            quantDtype = "Fp8"
        else:
            quantDtype = ""
        if doweight_stage1:
            extraInfo_1stage = "_tkw1"
        if q_type == QuantType.No:
            quantDtype_1stage = "noquant"
        elif q_type == QuantType.per_1x128:
            quantDtype_1stage = "blockscale" + quantDtype
        else:
            quantDtype_1stage = "pertoken" + quantDtype
        return quantDtype_1stage, extraInfo_1stage


def gen_1stage_asm_task(key):
    task_1stage = []
    info = key
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
    ## asm moe 1 stage tuning
    get_gfx()
    key = (act_type, q_type, dtype, q_dtype_a, q_dtype_w, use_g1u1)
    acti_dir = ""
    if act_type == ActivationType.Silu:
        acti_dir = "silu"
    elif act_type == ActivationType.Gelu:
        acti_dir = "gelu"
    up = 1 if use_g1u1 else 0
    quantDtype_1stage, extraInfo_1stage = get_1stage_file_info(
        q_type, q_dtype_a, doweight_stage1
    )
    kernels_list_csv_1stage = f"{get_asm_dir()}/fmoe/{acti_dir}/fmoe_bf16_{{quantDtype_1stage}}_g1u{up}_{acti_dir}{{extraInfo_1stage}}.csv"
    asm_kernels_1stage = {}
    asm_1stage_csv_path = ""
    if (
        q_type != QuantType.No
        and q_type != QuantType.per_Tensor
        and q_dtype_w != torch.int4
    ):
        asm_1stage_csv_path = kernels_list_csv_1stage.format(
            quantDtype_1stage=quantDtype_1stage,
            extraInfo_1stage=extraInfo_1stage,
        )
        asm_kernels_1stage = get_kernels_dict(
            asm_1stage_csv_path,
            key=["subGU_m", "subGU_n", "smf"],
        )
    asm_1stage_flat = {}
    if asm_1stage_csv_path and os.path.exists(asm_1stage_csv_path):
        _df = pd.read_csv(asm_1stage_csv_path)
        asm_1stage_flat = _manifest_flat_by_kernel(_df)
    fmoe_func = (
        get_1stage_fmoe_func(q_type, q_dtype_a, act_type, use_g1u1, doweight_stage1)
        if q_type != QuantType.per_1x32
        else None
    )
    if fmoe_func is None and q_type != QuantType.per_1x32:
        return task_1stage
    for tile_m, tile_n, smf in asm_kernels_1stage:
        if inter_dim % tile_n != 0 or smf != 0:
            continue

        for el in asm_kernels_1stage.get((tile_m, tile_n, 0), []):
            # Only no_sort/emsort consume raw topk. Post-reduce candidates
            # time the sorted-input producer and its reduction together.
            flat_flag = int(asm_1stage_flat.get(el, AsmMoeMode.SORTED))
            asm_mode = AsmMoeMode(flat_flag)
            if asm_mode is AsmMoeMode.POST_REDUCE:
                if (
                    q_type != QuantType.per_1x32
                    or q_dtype_a != dtypes.fp4x2
                    or q_dtype_w != dtypes.fp4x2
                    or dtype != dtypes.bf16
                    or inter_dim != tile_n
                ):
                    continue
                candidate_func = run_1stage_fmoe_g1u1
                _data_idx = [0, 1, 2, 3, 4, 5, 6, 7, 9, 19, 20, 17]
            elif fmoe_func is None:
                continue
            elif not asm_mode.requires_host_sort:
                candidate_func = fmoe_func
                _data_idx = [0, 1, 2, 3, 15, 14, 15, 15, 18, 10, 11, 17]
            else:
                candidate_func = fmoe_func
                _data_idx = [0, 1, 2, 3, 4, 5, 6, 7, 18, 10, 11, 17]
            _data_names = [_GEN_DATA_1STAGE_KEYS[i] for i in _data_idx]
            task_1stage.append(
                (
                    (info, "asm_1stage", el, tile_m, flat_flag),
                    generate_data_1stage,
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
                        tile_m,
                    ),
                    candidate_func,
                    (
                        _data_names,
                        q_type,
                        use_g1u1,
                        act_type,
                        el,
                        topk,
                        dtype,
                    ),
                    ({"flat": asm_mode} if asm_mode is AsmMoeMode.POST_REDUCE else {}),
                    (
                        torch_moe_blockscale
                        if q_type == QuantType.per_1x128
                        else torch_moe_2stages
                    ),
                    (
                        (
                            [
                                "a1_qt",
                                "w1_qt",
                                "w2_qt",
                                "topk_weights",
                                "topk_ids",
                                "a1_scale",
                                "w1_scale",
                                "w2_scale",
                            ],
                            None,
                            (128, 128),
                            dtype,
                        )
                        if q_type == QuantType.per_1x128
                        else (
                            [
                                "a1_qt",
                                "w1_qt",
                                "w2_qt",
                                "topk_weights",
                                "topk_ids",
                                (
                                    "a1_scale_t"
                                    if asm_mode is AsmMoeMode.POST_REDUCE
                                    else "a1_scale"
                                ),
                                "w1_scale",
                                "w2_scale",
                            ],
                            dtype,
                            act_type,
                            q_type,
                            doweight_stage1,
                        )
                    ),
                    {},
                    (None),
                    0.01,
                    1,
                    True,
                )
            )

    # xbf16: benchmark blockscaleBf16 kernels (bf16 input, kernel-internal quant)
    if (
        q_type == QuantType.per_1x128
        and q_dtype_a == dtypes.fp8
        and get_gfx() == "gfx950"
    ):
        xbf16_csv = kernels_list_csv_1stage.format(
            quantDtype_1stage="blockscaleBf16",
            extraInfo_1stage=extraInfo_1stage,
        )
        xbf16_kernels = get_kernels_dict(xbf16_csv, key=["subGU_m", "subGU_n", "smf"])
        xbf16_flat = {}
        if os.path.exists(xbf16_csv):
            _df = pd.read_csv(xbf16_csv)
            xbf16_flat = _manifest_flat_by_kernel(_df)
        for tile_m, tile_n, smf in xbf16_kernels:
            if inter_dim % tile_n != 0 or smf != 0:
                continue
            for el in xbf16_kernels.get((tile_m, tile_n, 0), []):
                # xbf16: internal quant; only no_sort/emsort take raw topk.
                flat_flag = int(xbf16_flat.get(el, AsmMoeMode.SORTED))
                asm_mode = AsmMoeMode(flat_flag)
                if asm_mode is AsmMoeMode.POST_REDUCE:
                    continue
                if not asm_mode.requires_host_sort:
                    _data_idx = [0, 0, 2, 3, 15, 14, 15, 15, 18, 10, 11, 17]
                else:
                    _data_idx = [0, 0, 2, 3, 4, 5, 6, 7, 18, 10, 11, 17]
                _data_names = [_GEN_DATA_1STAGE_KEYS[i] for i in _data_idx]
                task_1stage.append(
                    (
                        (info, "asm_1stage_xbf16", el, tile_m, flat_flag),
                        generate_data_1stage,
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
                            tile_m,
                        ),
                        fmoe_func,
                        (
                            _data_names,
                            q_type,
                            use_g1u1,
                            act_type,
                            el,
                            topk,
                            dtype,
                        ),
                        {},
                        (torch_moe_blockscale),
                        (
                            [
                                "a1_qt",
                                "w1_qt",
                                "w2_qt",
                                "topk_weights",
                                "topk_ids",
                                "a1_scale",
                                "w1_scale",
                                "w2_scale",
                            ],
                            None,
                            (128, 128),
                            dtype,
                        ),
                        {},
                        (None),
                        0.01,
                        1,
                        True,
                    )
                )

    return task_1stage
