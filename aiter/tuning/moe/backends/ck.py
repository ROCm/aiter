# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CK candidates of the fused MoE tuner's two-stage path."""

import sys

import torch

import aiter
from aiter import (
    ActivationType,
    QuantType,
    ck_moe_stage1_fwd,
    ck_moe_stage2_fwd,
    dtype2str_dict,
    dtypes,
)
from aiter.jit.core import AITER_CSRC_DIR
from aiter.ops.moe_limits import (
    RUNTIME_SILU_LIMIT_QUANT_TYPES,
    normalize_swiglu_limit,
)
from aiter.test_common import limit_cosine_diff_compare
from aiter.tuning.moe.backends.cktile import _gen_2stages_task_cktile
from aiter.tuning.moe.backends.common import _is_tune_excluded_kernel
from aiter.tuning.moe.compare import run_torch_moe_stage1, run_torch_moe_stage2
from aiter.tuning.moe.inputs import generate_data_2stages

sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list


def ck_moe_stage1_fwd_out(
    a1_qt,
    w1_qt_shffle_ck,
    w2_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w1_scale,
    a1_scale,
    dtype,
    topk,
    kernelName,
    blockM,
    q_type,
    act_type,
    splitk=0,
    use_nt=False,
    swiglu_limit=None,
):
    inter_dim = w1_qt_shffle_ck.shape[1] // 2
    token_num = a1_qt.shape[0]
    swiglu_limit = normalize_swiglu_limit(swiglu_limit)
    if swiglu_limit is not None:
        if splitk > 1 or "Splitk" in kernelName:
            raise ValueError("Runtime SiLU clamp tuning requires non-split CK")
        if (
            q_type not in RUNTIME_SILU_LIMIT_QUANT_TYPES
            or act_type != ActivationType.Silu
        ):
            raise ValueError(
                "Runtime clamp tuning requires per_1x128 / per_Token / "
                "per_Tensor SiLU"
            )
    is_splitk = q_type == QuantType.per_1x128 and splitk > 1
    if is_splitk:
        sorted_size = min(token_num * topk * blockM, sorted_ids.shape[0])
        tmp_out = torch.empty(
            (sorted_size, w1_qt_shffle_ck.shape[1]),
            dtype=dtypes.fp32,
            device=a1_qt.device,
        )
    else:
        out = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=a1_qt.device,
        )
        tmp_out = out
    ck_moe_stage1_fwd(
        a1_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        topk,
        kernelName,
        w1_scale,
        a1_scale,
        blockM,
        sorted_weights,
        q_type,
        act_type,
        splitk if is_splitk else 0,
        use_non_temporal_load=use_nt,
        dst_type=dtype if is_splitk else None,
        swiglu_limit=float(swiglu_limit or 0.0),
    )
    if is_splitk:
        out = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=a1_qt.device,
        )
        valid_out = tmp_out[: token_num * topk, :]
        if act_type == ActivationType.Silu or (
            isinstance(act_type, str) and "silu" in act_type.lower()
        ):
            aiter.silu_and_mul(out, valid_out.view(dtypes.fp32))
        else:
            aiter.gelu_and_mul(out, valid_out.view(dtypes.fp32))
    if q_type == QuantType.per_1x128:
        quant_func = aiter.get_hip_quant(q_type)
        a2, _a2_scale = quant_func(
            out,
            quant_dtype=a1_qt.dtype,
        )
        out = a2
    return out


def ck_moe_stage2_fwd_out(
    a2_qt,
    w1_qt_shffle_ck,
    w2_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w2_scale,
    a2_scale,
    dtype,
    topk,
    kernelName,
    blockM,
    q_type,
    act_type,
    use_nt=False,
):
    model_dim = w2_qt_shffle_ck.shape[1]
    token_num = a2_qt.shape[0]

    out = torch.zeros(
        (token_num, model_dim),
        dtype=dtype,
        device=a2_qt.device,
    )
    return ck_moe_stage2_fwd(
        a2_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        w2_scale,
        a2_scale,
        blockM,
        sorted_weights,
        q_type,
        act_type,
        use_non_temporal_load=use_nt,
    )


def gen_2stages_task(key, blockMs, swiglu_limit):
    info = key
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

    _is_a8w4 = (
        q_dtype_a == dtypes.fp8
        and q_dtype_w == dtypes.fp4x2
        and q_type == QuantType.per_1x32
    )

    if _is_a8w4:
        return _gen_2stages_task_cktile(info, blockMs)

    if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp8:
        return tasks_ck

    # CK kernels don't support a16wi4 (per_1x32 + i4x2); skip to FlyDSL path
    if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
        return tasks_ck

    # CK gemm1 codegen has no b16 x fp4x2 combo (get_gemm1_kernels_list
    # raises "Unsupported data type combination: b16, fp4x2"); a16w4
    # (bf16 activation x mxfp4 weight) is FlyDSL-only -> skip CK.
    if (
        q_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
        and q_dtype_a == dtypes.bf16
    ):
        return tasks_ck

    # CK2stages codegen does not support SwiGLU activation. GPT-OSS MXFP4
    # cases are covered by FlyDSL (or the a8w4 CK-Tile path above).
    if (
        act_type == ActivationType.Swiglu
        and q_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
    ):
        return tasks_ck

    _, ck_stage1_kernels = get_gemm1_kernels_list(
        dtype2str_dict[q_dtype_a],
        dtype2str_dict[q_dtype_w],
        dtype2str_dict[dtype],
        False,
        int(q_type),
        str(act_type).split(".")[-1].lower(),
        doweight_stage1,
        True,  # bpreshuffle
    )

    is_fp8_blockscale = (
        q_type == QuantType.per_1x128
        and q_dtype_a == dtypes.fp8
        and q_dtype_w == dtypes.fp8
    )
    splitk_list = []
    if is_fp8_blockscale:
        tilek = 128
        for _sk in range(2, 9):
            if (model_dim % _sk == 0) and ((model_dim // _sk) % tilek == 0):
                splitk_list.append(_sk)
        if splitk_list:
            _, _ck_stage1_splitk_kernels = get_gemm1_kernels_list(
                dtype2str_dict[q_dtype_a],
                dtype2str_dict[q_dtype_w],
                dtype2str_dict[dtype],
                False,
                int(q_type),
                str(act_type).split(".")[-1].lower(),
                doweight_stage1,
                True,  # bpreshuffle
                splitk=True,
            )

    _, ck_stage2_kernels = get_gemm2_kernels_list(
        dtype2str_dict[q_dtype_a],
        dtype2str_dict[q_dtype_w],
        dtype2str_dict[dtype],
        False,
        int(q_type),
        not doweight_stage1,
        True,  # bpreshuffle
    )
    # only the blockscale kernels read the hint
    nt_values = (False, True) if q_type == QuantType.per_1x128 else (False,)
    clamp_limit = (
        swiglu_limit
        if q_type in RUNTIME_SILU_LIMIT_QUANT_TYPES
        and q_dtype_a == dtypes.fp8
        and q_dtype_w == dtypes.fp8
        else None
    )
    stage1_kwargs = {"swiglu_limit": clamp_limit} if clamp_limit else {}
    for blockM in blockMs:
        if blockM in [16, 32, 64, 128] and use_g1u1:
            for kernel, nt in [
                (k, n)
                for k in ck_stage1_kernels.values()
                if k.MPerBlock == blockM
                for n in nt_values
            ]:
                tasks_ck.append(
                    (
                        (info, "stage1", kernel.name, blockM, 0, 0, nt),  # tag
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
                        ),
                        ck_moe_stage1_fwd_out,  # func
                        (
                            [
                                "a1_qt",
                                "w1_qt_shffle_ck",
                                "w2_qt_shffle_ck",
                                "sorted_ids",
                                "sorted_expert_ids",
                                "sorted_weights",
                                "num_valid_ids",
                                "w1_scale_aiter",
                                "a1_scale_fp4_sort",
                            ],
                            dtype,
                            topk,
                            kernel.name,
                            blockM,
                            q_type,
                            act_type,
                        ),
                        {"use_nt": nt, **stage1_kwargs},
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
                        stage1_kwargs,
                        (None),
                        0.01,
                        0.01,
                        limit_cosine_diff_compare if clamp_limit else None,
                    )
                )

            for kernel, nt in [
                (k, n)
                for k in ck_stage2_kernels.values()
                if k.MPerBlock == blockM and not _is_tune_excluded_kernel(k.name)
                for n in nt_values
            ]:
                s2_ref_args = (
                    [
                        "a2_qt",
                        "w1_qt",
                        "w2_qt",
                        "topk_weights",
                        "topk_ids",
                        "a2_scale",
                        "w2_scale",
                        "bias",
                    ],
                    dtype,
                    q_type,
                    doweight_stage1,
                )
                tasks_ck.append(
                    (
                        (info, "stage2", kernel.name, blockM, 0, 0, nt),  # tag
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
                            2,
                        ),
                        ck_moe_stage2_fwd_out,  # func
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
                            ],
                            dtype,
                            topk,
                            kernel.name,
                            blockM,
                            q_type,
                            act_type,
                        ),
                        {"use_nt": nt},
                        run_torch_moe_stage2,
                        s2_ref_args,
                        {},
                        (None),
                        0.01,
                        0.01,
                        limit_cosine_diff_compare if clamp_limit else None,
                    )
                )
    return tasks_ck
