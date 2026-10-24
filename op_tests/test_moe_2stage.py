# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import functools
import gc
import itertools
import logging
import os
from contextlib import ExitStack, nullcontext

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.aot.flydsl.common import override_env, run_only_env
from aiter.fhmoe_contract import FHMOE_BUFFER_MAX_BYTES
from aiter.fused_moe import (
    _flydsl_route_output_nbytes,
    _is_flydsl_stage2,
    _stage2_runtime_uses_route_reduce,
    cktile_moe_stage1,
    cktile_moe_stage2,
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    resolve_activation_dtype,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.int4_utils import (
    convert_int8_to_uint32_int4,
    rearrange_4bit_elements,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.opus.moe_stage2_a8w4 import _route_workspace_token_capacity
from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_i4_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

try:
    from tuned_op_bench_utils import append_tuned_op_bench_rows
except ModuleNotFoundError as e:
    if e.name != "tuned_op_bench_utils":
        raise
    from op_tests.tuned_op_bench_utils import append_tuned_op_bench_rows


from aiter.ops.shuffle import (
    pack_int8_to_packed_int4,
    shuffle_scale_a16w4,
    shuffle_scale_for_int4,
    shuffle_weight,
    shuffle_weight_a16w4,
)

logger = logging.getLogger(__name__)

torch.int4 = getattr(torch, "int4", torch.uint32)
torch.set_default_device("cuda")
AITER_MOE_EXPERT_BALANCE = (
    os.environ.get("AITER_MOE_EXPERT_BALANCE", "False").lower() == "true"
)


# Force topk to activate only the first n experts (ids 0..n-1). 0 = unset.
# Takes precedence over AITER_MOE_EXPERT_BALANCE when set (> 0).
def parse_num_expert_activated():
    try:
        val = int(os.environ.get("AITER_MOE_NUM_EXPERT_ACTIVATED", "0"))
    except ValueError:
        raise ValueError("AITER_MOE_NUM_EXPERT_ACTIVATED must be an integer")
    if val < 0:
        raise ValueError(f"AITER_MOE_NUM_EXPERT_ACTIVATED must be >= 0, got {val}")
    return val


AITER_MOE_NUM_EXPERT_ACTIVATED = parse_num_expert_activated()


@benchmark()
def test_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    actType,
    gateMode,
    qType,
    AQDType,
    WQDType,
    use_g1u1=False,
    doweight_stage1=False,
    hidden_pad=0,
    intermediate_pad=0,
    preshuffle=True,
    strict_accuracy=True,
    check_aot_cache=True,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    kernel_bench=False,
    disable_stage1_bias=False,
    disable_stage2_bias=False,
    ref_dtype="bf16",
    gate_up_layout: bool | None = None,
):
    if get_gfx() not in ["gfx950"] and qType in [aiter.QuantType.per_1x32]:
        return
    if actType == aiter.ActivationType.Situv2:
        beta = DEFAULT_SITUV2_BETA if beta is None else float(beta)
        linear_beta = (
            DEFAULT_SITUV2_LINEAR_BETA if linear_beta is None else float(linear_beta)
        )
    torch_quant = aiter.get_torch_quant(qType)
    # mxfp8 (a8w8): per-1x32 e8m0 microscale on both fp8 activation and fp8 weight.
    is_mxfp8 = (
        qType == aiter.QuantType.per_1x32
        and AQDType == dtypes.fp8
        and WQDType == dtypes.fp8
    )
    input = torch.randn((token, model_dim), dtype=dtype)
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
        if hidden_pad != 0:
            w1[:, :, -hidden_pad:] = 0
        if intermediate_pad != 0:
            w1[:, -intermediate_pad:, :] = 0
            w1[:, inter_dim - intermediate_pad : inter_dim, :] = 0
        exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype), -1.0, 1.0)
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype)
        exp_bias1 = torch.clamp(torch.randn((E * inter_dim), dtype=dtype), -1.0, 1.0)
    if disable_stage1_bias:
        exp_bias1 = None
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    if intermediate_pad != 0:
        w2[:, :, -intermediate_pad:] = 0
    if hidden_pad != 0:
        w2[:, -hidden_pad:, :] = 0
    exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype), -1.0, 1.0)
    if disable_stage2_bias:
        exp_bias2 = None
    if AITER_MOE_NUM_EXPERT_ACTIVATED > 0:
        # Highest priority: activate n randomly-chosen experts (NOT the first n);
        # the other E-n experts are masked to -inf. Load is spread evenly across
        # the n active experts by round-robin (balanced), so all n are used.
        n_act = AITER_MOE_NUM_EXPERT_ACTIVATED
        if n_act < topk or n_act > E or n_act > token * topk:
            raise ValueError(
                f"AITER_MOE_NUM_EXPERT_ACTIVATED={n_act} is invalid: must be "
                f"in [topk={topk}, min(E={E}, token*topk={token * topk})]"
            )
        sel = torch.randperm(E)[:n_act]  # random active expert ids
        score = torch.full((token, E), float("-inf"), dtype=dtype)
        slot = torch.arange(token * topk) % n_act  # round-robin over active set
        rows = torch.arange(token).repeat_interleave(topk)
        score[rows, sel[slot]] = 1.0
    elif AITER_MOE_EXPERT_BALANCE:
        score = torch.zeros((token, E), dtype=dtype)
        start_col = 0
        end_col = topk
        for token_id in range(token):
            score[token_id, start_col:end_col] = 1.0
            start_col = end_col % E
            end_col = start_col + topk
    else:
        score = torch.randn((token, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if qType == aiter.QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=WQDType)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=WQDType)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
    elif qType == aiter.QuantType.per_Token and WQDType == torch.int4:  # int4 w quant
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    elif (
        qType == aiter.QuantType.per_1x32 and WQDType == dtypes.i4x2
    ):  # a16wi4: int4 weights, bf16 activations
        w1_qt, w1_scale = per_1x32_i4_quant(w1)
        w1_qt = w1_qt.view(dtypes.i4x2)
        w2_qt, w2_scale = per_1x32_i4_quant(w2)
        w2_qt = w2_qt.view(dtypes.i4x2)
    elif is_mxfp8:  # mxfp8: fp8 weights, per-1x32 e8m0 scale
        w1_qt, w1_scale = per_1x32_f8_scale_f8_quant(
            w1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        w2_qt, w2_scale = per_1x32_f8_scale_f8_quant(
            w2, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    elif qType == aiter.QuantType.per_128x128:

        def weight_per_128x128_quant(weight, quant_dtype):
            E, dim1, dim2 = weight.shape
            weight_blocks = weight.view(
                E, dim1 // 128, 128, dim2 // 128, 128
            )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_blocks = weight_blocks.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_blocks = weight_blocks.view(
                E, -1, 128 * 128
            )  # [E, num_blocks, 128*128]
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_blocks, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(
                E, dim1 // 128, dim2 // 128, 128, 128
            )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_qt = weight_qt.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
            weight_scale = weight_scale.view(
                E, dim1 // 128, dim2 // 128
            )  # [E, num_blocks_dim1, num_blocks_dim2]
            return weight_qt, weight_scale

        w1_qt, w1_scale = weight_per_128x128_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = weight_per_128x128_quant(w2, quant_dtype=WQDType)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)

    if qType == aiter.QuantType.per_1x32 and WQDType not in (
        dtypes.i4x2,
        dtypes.fp8,
    ):
        # fp4x2 weights pack 2 nibbles/byte -> halved last dim. fp8 (mxfp8) and
        # i4x2 keep their stored shape.
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
    else:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)

    # Match fused_moe's runtime activation dtype. SiTUv2 can be requested as
    # a16w4 by the caller but dispatched as a8w4 on gfx950.
    reference_aq_dtype = AQDType
    if actType == aiter.ActivationType.Situv2:
        runtime_aq_dtype = _runtime_situv2_mxfp4_q_dtype_a(qType, WQDType)
        if runtime_aq_dtype is not None:
            reference_aq_dtype = runtime_aq_dtype

    # Quant-ing a
    if qType == aiter.QuantType.per_128x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=AQDType
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    elif (
        qType == aiter.QuantType.per_1x32
        and reference_aq_dtype == dtypes.fp8
        and WQDType == dtypes.fp4x2
    ):
        a1_qt, a1_scale = per_1x32_f8_scale_f8_quant(
            input, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    elif (
        (
            qType == aiter.QuantType.per_1x32
            and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
            and WQDType == dtypes.fp4x2
        )
        or is_mxfp8
        or qType == aiter.QuantType.per_1x32
        and WQDType == dtypes.i4x2
    ):  # a16w4 & a8w4 & mxfp8
        a1_qt = input.to(dtypes.bf16)
        a1_scale = None
    else:
        a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)

    # bias dtype convert
    if (
        qType == aiter.QuantType.per_1x32
        and reference_aq_dtype == dtypes.bf16
        and WQDType == dtypes.fp4x2
        and actType in (aiter.ActivationType.Situv2, aiter.ActivationType.Relu2)
    ):  # a16w4 SiTUv2 / Relu2: no per-expert bias supported by these kernels.
        # Key on reference_aq_dtype (runtime dispatch), not the declared AQDType:
        # a SiTUv2 case declared a8w4/a4w4 still runs as a16w4 without the env opt-in.
        exp_bias1 = exp_bias2 = None
        exp_bias1_aiter = exp_bias2_aiter = None
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a8w4 (fp8 A) mxfp4
        exp_bias1_aiter = None if exp_bias1 is None else exp_bias1.to(dtypes.fp32)
        exp_bias2_aiter = None if exp_bias2 is None else exp_bias2.to(dtypes.fp32)
    elif (
        qType == aiter.QuantType.per_1x32 and WQDType == dtypes.i4x2
    ):  # a16wi4: no bias
        exp_bias1_aiter = exp_bias1 = None
        exp_bias2_aiter = exp_bias2 = None
    else:
        exp_bias1_aiter = exp_bias1 = None
        exp_bias2_aiter = exp_bias2 = None

    # pre-shuffle
    w1_scale_aiter = w1_scale
    w2_scale_aiter = w2_scale
    if qType == aiter.QuantType.per_1x32 and WQDType == dtypes.i4x2:  # a16wi4
        w1_qt_aiter = pack_int8_to_packed_int4(
            shuffle_weight(w1_qt_aiter.view(dtypes.i8), (16, 16))
        )
        w1_qt_aiter = w1_qt_aiter.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2).view(
            dtypes.i4x2
        )
        w2_qt_aiter = pack_int8_to_packed_int4(
            shuffle_weight(w2_qt_aiter.view(dtypes.i8), (16, 16))
        )
        w2_qt_aiter = w2_qt_aiter.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2).view(
            dtypes.i4x2
        )
        # groupwise scale: [E, K//32, N] bf16 -> shuffle and flatten for kernel
        w1_scale_aiter = (
            shuffle_scale_for_int4(w1_scale, group_size=32).view(-1).contiguous()
        )
        w2_scale_aiter = (
            shuffle_scale_for_int4(w2_scale, group_size=32).view(-1).contiguous()
        )
    elif WQDType == torch.int4:  # int4 w quant (a8w4)
        w1_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w1_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w2_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w2_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4 / a8w4
        # a16w4 (bf16/fp16 act) uses standard GGUU (gate_up=False), matching main;
        # a8w4 (fp8 act) keeps the gate/up-interleaved GUGU (gate_up=True).
        _w1_gu = (AQDType == dtypes.fp8) if gate_up_layout is None else gate_up_layout
        w1_qt_aiter = shuffle_weight_a16w4(w1_qt_aiter, 16, _w1_gu)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, E, _w1_gu)
        w2_qt_aiter = shuffle_weight_a16w4(w2_qt_aiter, 16, False)
        w2_scale_aiter = shuffle_scale_a16w4(w2_scale, E, False)
    elif is_mxfp8:  # mxfp8 (a8w8): gate-up interleaved fp8 weight + e8m0 scale
        w1_qt_aiter = shuffle_weight_a16w4(w1_qt_aiter, 16, True)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, E, True)
        w2_qt_aiter = shuffle_weight_a16w4(w2_qt_aiter, 16, False)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    elif WQDType != dtypes.fp4x2 or preshuffle:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    else:
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

    # # ######################## stage 1 start ###########
    stage1_ref_dtype = dtype
    if (
        actType == aiter.ActivationType.Swiglu
        and qType == aiter.QuantType.per_1x32
        and WQDType == dtypes.fp4x2
    ):
        runtime_aq_dtype = _runtime_swiglu_mxfp4_q_dtype_a(
            token, actType, gateMode, qType, AQDType, WQDType
        )
        if runtime_aq_dtype == dtypes.fp4x2:
            metadata = get_2stage_cfgs(
                get_padded_M(token),
                model_dim,
                inter_dim,
                E,
                topk,
                dtype,
                runtime_aq_dtype,
                WQDType,
                qType,
                w1.shape[1] == (inter_dim * 2),
                actType,
                doweight_stage1,
                hidden_pad,
                intermediate_pad,
                getattr(w1_qt_aiter, "is_shuffled", False)
                or getattr(w2_qt_aiter, "is_shuffled", False),
                gateMode,
            )
            if metadata.fuse_quant == "fp4":
                # Fused Swiglu MXFP4 quantizes the f32 activation directly.
                # Keep the torch reference at f32 until the quantization step.
                stage1_ref_dtype = dtypes.fp32

    # --ref-dtype fp32: keep the reference intermediate in fp32 so the a2 mxfp4
    # amax is taken over fp32 (not bf16). This mirrors the fused asm kernel, which
    # computes amax on the in-register fp32 silu result (no bf16 round-trip that
    # the 2-stage flydsl/torch path does via its bf16 a2 buffer).
    if ref_dtype == "fp32":
        stage1_ref_dtype = dtypes.fp32

    out1_ref = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=stage1_ref_dtype,
        activation=actType,
        quant_type=qType,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        w1_bias=exp_bias1,
        doweight=doweight_stage1,
        swiglu_limit=swiglu_limit,
        # SiTUv2 defaults were resolved above; other activations ignore these values.
        situ_beta=1.0 if beta is None else float(beta),
        situ_linear_beta=1.0 if linear_beta is None else float(linear_beta),
    )

    # ######################## stage 2 start ###########
    if qType == aiter.QuantType.per_128x128:
        a2_qt, a2_scale = aiter.pertoken_quant(
            out1_ref.view(token, -1, 128), quant_dtype=AQDType
        )
        a2_scale = a2_scale.view(token, topk, -1)
    elif (
        qType == aiter.QuantType.per_1x32
        and reference_aq_dtype == dtypes.fp8
        and WQDType == dtypes.fp4x2
    ):
        a2_qt, a2_scale = per_1x32_f8_scale_f8_quant(
            out1_ref, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ) or is_mxfp8:  # a16w4 & a8w4 & mxfp8
        a2_qt = out1_ref
        a2_scale = None
    elif (
        qType == aiter.QuantType.per_1x32 and WQDType == dtypes.i4x2
    ):  # a16wi4: bf16 pass-through
        a2_qt = out1_ref
        a2_scale = None
    else:
        a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=AQDType)
    a2_qt = a2_qt.view(token, topk, -1)

    out2_ref = torch_moe_stage2(
        a2_qt,
        w1_qt,  # E, inter_dim*2, model_dim
        w2_qt,  # E, model_dim, inter_dim
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=qType,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        w2_bias=exp_bias2,
        doweight=not doweight_stage1,
    )

    # ######################## stage 2 end ###########
    _fused_moe_kwargs = {
        "w1_scale": w1_scale_aiter,
        "w2_scale": w2_scale_aiter,
        "quant_type": qType,
        "activation": actType,
        "doweight_stage1": doweight_stage1,
        "intermediate_pad": intermediate_pad,
        "hidden_pad": hidden_pad,
        "bias1": exp_bias1_aiter,
        "bias2": exp_bias2_aiter,
        "swiglu_limit": swiglu_limit,
        "beta": beta,
        "linear_beta": linear_beta,
        "gate_mode": gateMode,
    }

    if kernel_bench:
        # Kernel-bench: time the stage1 / stage2 kernels in isolation. One eager
        # fused_moe call populates the per-stage launch callables (via the opt-in
        # kernel_bench_callable hook) and yields a correct output; then loop each
        # captured kernel launch alone (excludes input prep / quant / sorting).
        kernel_bench_callable = []
        aiter.fused_moe.kernel_bench_callable = kernel_bench_callable
        try:
            out2_ck = fused_moe(
                input,
                w1_qt_aiter,
                w2_qt_aiter,
                topk_weights,
                topk_ids,
                **_fused_moe_kwargs,
            )
        finally:
            aiter.fused_moe.kernel_bench_callable = None
        kernel_us = {}
        for _name, _call in kernel_bench_callable:
            _, _us = run_perftest(_call, num_iters=20, num_warmup=3)
            kernel_us[_name] = _us
        us1 = kernel_us.get("stage1")
        us2_stage = kernel_us.get("stage2")
        if not kernel_us:
            logger.warning(
                "kernel_bench: no kernels captured (non-2stage/1stage path?) (quant:%s)",
                AQDType,
            )
        else:
            logger.info(
                "kernel_bench: stage1=%s us, stage2=%s us (quant:%s)",
                "n/a" if us1 is None else f"{us1:.2f}",
                "n/a" if us2_stage is None else f"{us2_stage:.2f}",
                AQDType,
            )
        return {
            "us": (us1 or 0.0) + (us2_stage or 0.0),
            "us_stage1": us1,
            "us_stage2": us2_stage,
        }

    out2_ck, us2 = run_perftest(
        fused_moe,
        input,
        w1_qt_aiter,
        w2_qt_aiter,
        topk_weights,
        topk_ids,
        num_iters=5,
        num_warmup=2,
        **_fused_moe_kwargs,
    )
    # Regression guard for aiter #3117 (MXFP4 fused-MoE stage2 EP-prefill):
    # the unfixed K-padding tail-tile path leaves the padded lanes uninitialized,
    # producing NaN in the fused_moe output. checkAllclose's err/logits_diff can be
    # masked by atomic-reduction noise, so detect NaN explicitly and deterministically.
    has_nan = out2_ck.isnan().any().item()
    if has_nan:
        logger.error(
            "output contains NaN! (possible aiter #3117 stage2 K-pad regression)"
        )
    err = checkAllclose(
        out2_ref,
        out2_ck,
        msg=f"ck_moe_2stages:{us2:>8.2f} us, {token*model_dim*inter_dim*3*topk*2/us2/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    )

    def calc_diff(x: torch.Tensor, y: torch.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = 2 * (x * y).sum() / denominator
        return 1 - sim

    logits_diff = calc_diff(out2_ref, out2_ck)
    if logits_diff > 1e-3:
        logger.warning(
            f"logits_diff: {logits_diff} is too large, please check the implementation"
        )
    if strict_accuracy:
        assert not has_nan, "accuracy check failed: output contains NaN"
        assert not (
            err != 0 and logits_diff > 0.01
        ), f"accuracy check failed: checkAllclose err={err}, logits_diff={logits_diff}"
    elif has_nan:
        logger.warning("accuracy check failed (non-strict): output contains NaN")
    elif err != 0 and logits_diff > 0.01:
        logger.warning(
            f"accuracy check failed (non-strict): err={err}, logits_diff={logits_diff}"
        )

    return {"us": us2, "logits_diff": float(logits_diff)}


l_quant = [
    (aiter.QuantType.No, None, None),  # a16w16
    (aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, torch.int4),  # a8w4
    (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2),  # a4w4
    (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2),  # a16w4
    (aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2),  # a8w4
    (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.i4x2),  # a16wi4
    (aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp8),  # mxfp8 (a8w8)
]


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16, fp16}",
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="*",
    default=[(7168, 256)],
    help="""Model dimension.
    e.g.: -dim 6144,4096""",
)

parser.add_argument(
    "-t",
    "--tokenNum",
    type=int,
    nargs="*",
    default=[
        1,
        3,
        5,
        16,
        32,
        64,
        128,
        256,
        1024,
        4096,
        8192,
        163840,
    ],
    help="""Number of tokens.
    e.g.: -t 1024""",
)

parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(l_quant)),
    help="""select quantization type:
    0 : aiter.QuantType.No, None, None),  # a16w16
    1: aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8  # a8w8
    2: aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8
    3: aiter.QuantType.per_Token, dtypes.fp8, torch.int4  # a8w4
    4: aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2  # a4w4
    5: aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8,  # a8w8,
    6: aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2,  # a16w4,
    7: aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2,  # a8w4,
    8: aiter.QuantType.per_1x32, dtypes.bf16, dtypes.i4x2,  # a16wi4,
    9: aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp8,  # mxfp8 (a8w8),""",
)

parser.add_argument(
    "-a",
    "--act",
    type=dtypes.str2ActivationType,
    nargs="*",
    default=[aiter.ActivationType.Silu],
    help="""Select activation type. Default: [Silu].
    e.g.: -a gelu        # [Gelu]
          -a silu gelu    # [Silu, Gelu]""",
)

parser.add_argument(
    "-s",
    "--doweight_stage1",
    type=dtypes.str2bool,
    nargs="*",
    default=[False],
    help="""Whether to do weight in stage 1. Default is [False].
    -s f    # False.
    -s t    # True.""",
)

parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=257,
    help="""Number of experts.
    e.g.: -e 8""",
)

parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=9,
    help="""Number of top experts.
    e.g.: -k 2""",
)

parser.add_argument(
    "-p",
    "--preshuffle",
    type=dtypes.str2bool,
    nargs="*",
    default=[True],
    help="""Whether to use pre-shuffle weight mode. Default is [False, True].
    -p f    # False.
    -p t    # True.""",
)
parser.add_argument(
    "-hip",
    "--hidden_intermediate_pad",
    type=dtypes.str2tuple,
    nargs="*",
    default=[(192, 128)],
    help="""Hidden intermediate pad.
    e.g.: -hip 0,0""",
)
parser.add_argument(
    "--stage2-bias",
    type=dtypes.str2bool,
    nargs="*",
    default=[True],
    help="""Whether CLI-generated cases include a random per-expert stage-2
    bias. Default is [True] for backward compatibility. Use
    --stage2-bias f for bias-free models, or --stage2-bias t f to test both.""",
)
parser.add_argument(
    "--no-flydsl-csv",
    action="store_true",
    help="Skip validating FlyDSL/Opus shapes from tuned fmoe CSVs.",
)
parser.add_argument(
    "--csv-filter",
    nargs="*",
    default=None,
    help="Only run CSV rows whose kernelName1/2 contain any of these substrings "
    "(e.g. --csv-filter abf16_wbf16 to validate just bf16-dense flydsl rows).",
)
parser.add_argument(
    "--no-legacy",
    action="store_true",
    help="Skip the original hardcoded shape sweep and skinny tests.",
)
parser.add_argument(
    "--bm16-scale-boundary",
    action="store_true",
    help="Run only the deterministic BM16 tiled-scale boundary regression.",
)
parser.add_argument(
    "--swiglu-limit",
    "-sl",
    type=float,
    default=None,
    help="swiglu/silu clamp limit. Default None means the kernel default (7.0).",
)
parser.add_argument(
    "--beta",
    type=float,
    default=None,
    help="SiTUv2 gate scale param (beta). Default None uses the kernel default (4.0).",
)
parser.add_argument(
    "--linear-beta",
    type=float,
    default=None,
    help="SiTUv2 up (linear) scale param (linear_beta). "
    "Default None uses the kernel default (25.0).",
)
parser.add_argument(
    "--kernel",
    action="store_true",
    help="""Time the stage1 / stage2 kernels in isolation (loop each launch
    alone, excluding input prep) and report them as us_stage1 / us_stage2.
    Only the 2-stage path exposes per-kernel launches; the 1-stage path reports
    n/a.""",
)
parser.add_argument(
    "--ref-dtype",
    choices=["bf16", "fp32"],
    default="bf16",
    help="Precision of the torch reference stage-1 intermediate. 'fp32' keeps it "
    "in fp32 so the a2 mxfp4 amax is taken over fp32 (mirrors the fused asm "
    "kernel, which computes amax on the in-register fp32 silu result, instead "
    "of the bf16 round-trip the 2-stage flydsl/torch path does). Default: bf16.",
)

args = parser.parse_args()


l_quant = [l_quant[args.quant]] if args.quant is not None else l_quant


# ---------------------------------------------------------------------------
# Both modes (CLI sweep / model-csv) reduce to the same shape:
#   yield (test_fmoe_kwargs, extras_for_df)
# A single runner consumes the stream.
# ---------------------------------------------------------------------------
# Only kept for dtypes that may not exist as torch attributes in older builds;
# anything else falls through to getattr(torch, attr).
_DTYPE_STR_FALLBACK = {
    "torch.float4_e2m1fn_x2": dtypes.fp4x2,
    "torch.float8_e8m0fnu": dtypes.fp8_e8m0,
}


def _str2dtype(s):
    s = s.strip()
    if s in ("None", "none", ""):
        return None
    if s.startswith("torch."):
        attr = s.split(".", 1)[1]
        if hasattr(torch, attr):
            return getattr(torch, attr)
    if s in _DTYPE_STR_FALLBACK:
        return _DTYPE_STR_FALLBACK[s]
    raise ValueError(f"unsupported dtype string: {s!r}")


def _str2enum(s, enum_cls):
    return getattr(enum_cls, s.strip().split(".")[-1])


def _row_to_kwargs(row):
    q_type = _str2enum(row["q_type"], aiter.QuantType)
    aq_dtype = _str2dtype(row["q_dtype_a"])
    wq_dtype = _str2dtype(row["q_dtype_w"])
    act_type = _str2enum(row["act_type"], aiter.ActivationType)
    inter_dim = int(row["inter_dim"])
    # Tuned CSV rows do not carry gate mode explicitly. Infer the runtime mode
    # from the selected activation/weight dtype layout used by fused_moe.
    gate_mode = _effective_gate_mode(q_type, aq_dtype, wq_dtype)
    return {
        "dtype": _str2dtype(row["dtype"]),
        "token": int(row["token"]),
        "model_dim": int(row["model_dim"]),
        "inter_dim": inter_dim,
        "E": int(row["expert"]),
        "topk": int(row["topk"]),
        "actType": act_type,
        "gateMode": gate_mode,
        "qType": q_type,
        "AQDType": aq_dtype,
        "WQDType": wq_dtype,
        "use_g1u1": dtypes.str2bool(str(row["use_g1u1"])),
        "doweight_stage1": dtypes.str2bool(str(row["doweight_stage1"])),
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "preshuffle": True,
        "swiglu_limit": _effective_swiglu_limit(
            q_type, aq_dtype, wq_dtype, args.swiglu_limit
        ),
    }


def _iter_csv_cases():
    """Yield FlyDSL/Opus cases from every selected model CSV."""
    cu = get_cu_num()
    merged_csv = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
    df_csv = pd.read_csv(merged_csv)
    rows = df_csv[df_csv["cu_num"] == cu]
    for _, row in rows.iterrows():
        tag = row.get("_tag", "")
        if pd.notna(tag) and str(tag).strip():
            continue
        kernel_name1 = str(row.get("kernelName1", "") or "")
        kernel_name2 = str(row.get("kernelName2", "") or "")
        if (
            "flydsl_" not in kernel_name1
            and "flydsl_" not in kernel_name2
            and not kernel_name1.startswith("opus_moe1_")
        ):
            continue
        if args.csv_filter:
            _kn = kernel_name1 + " " + kernel_name2
            if not any(sub in _kn for sub in args.csv_filter):
                continue
        try:
            kwargs = _row_to_kwargs(row)
        except Exception as e:  # noqa: BLE001
            aiter.logger.warning(
                "skip row token=%s dim=(%s,%s): parse error %s",
                row.get("token"),
                row.get("model_dim"),
                row.get("inter_dim"),
                e,
            )
            continue
        if kwargs["actType"] == aiter.ActivationType.Situv2:
            kwargs["beta"] = 4.0 if args.beta is None else float(args.beta)
            kwargs["linear_beta"] = (
                25.0 if args.linear_beta is None else float(args.linear_beta)
            )
        # The reference path below uses the CSV q_dtype_a directly, while
        # fused_moe selects q_dtype_a from the current runtime mode. Skip CSV
        # rows tuned for a different mode (e.g. a4w4/a8w4 without the opt-in env).
        if kwargs["actType"] == aiter.ActivationType.Situv2:
            expected_aq_dtype = _runtime_situv2_mxfp4_q_dtype_a(
                kwargs["qType"], kwargs["WQDType"]
            )
            runtime_mode = "SiTUv2 MXFP4"
            # SiTUv2 a16w4 never ran before this ordering fix and every row
            # fails: _effective_gate_mode asks for INTERLEAVE while stage1 binds
            # gate_mode="separated" for non-fp8 activations.
            if kwargs["AQDType"] == dtypes.bf16 and kwargs["WQDType"] == dtypes.fp4x2:
                continue
        else:
            expected_aq_dtype = _runtime_swiglu_mxfp4_q_dtype_a(
                kwargs["token"],
                kwargs["actType"],
                kwargs["gateMode"],
                kwargs["qType"],
                kwargs["AQDType"],
                kwargs["WQDType"],
            )
            runtime_mode = "Swiglu MXFP4"
        if expected_aq_dtype is not None and kwargs["AQDType"] != expected_aq_dtype:
            aiter.logger.info(
                "skip row token=%s dim=(%s,%s): q_dtype_a=%s does not match "
                "current %s runtime mode (expected %s)",
                row.get("token"),
                row.get("model_dim"),
                row.get("inter_dim"),
                kwargs["AQDType"],
                runtime_mode,
                expected_aq_dtype,
            )
            continue
        kwargs["strict_accuracy"] = True
        # Targeted configs and env-selected SiTUv2 modes have no pre-registered
        # AOT cache entry, so let those cases compile on demand.
        kwargs["check_aot_cache"] = (
            args.csv_filter is None and kwargs["actType"] != aiter.ActivationType.Situv2
        )
        kwargs["disable_stage2_bias"] = kernel_name2.startswith("opus_")
        yield kwargs, {
            "kernelName1": kernel_name1,
            "kernelName2": kernel_name2,
        }


_PER1X32_BF16_FP4 = (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2)
_PER1X32_FP8_FP4 = (aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2)
_PER1X32_FP4_FP4 = (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2)
_PER1X32_BF16_I4 = (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.i4x2)

# SiTUv2 only routes to the FlyDSL MXFP4 kernel for per_1x32 + fp4/fp8 activation
# (a4w4 fp4 act, a8w4 fp8 act). Any other quant would silently fall off the
# FlyDSL path, so SiTUv2 is skipped for those combos.
_SITUV2_SUPPORTED_TRIPLES = (_PER1X32_FP8_FP4, _PER1X32_FP4_FP4)


def _situv2_beta_kwargs(act_type):
    """beta/linear_beta are only meaningful for SiTUv2; leave them unset (None)
    for every other activation so silu/swiglu/gelu behavior is unchanged."""
    if act_type == aiter.ActivationType.Situv2:
        return {"beta": args.beta, "linear_beta": args.linear_beta}
    return {}


def _effective_gate_mode(q_type, aq_dtype, wq_dtype):
    # a16w4 (bf16 A x mxfp4 W) SiTUv2 is served by the ported FlyDSL kernel via
    # fused_moe_'s SEPARATED dispatch; keep it in SEPARATED (bf16 activation) so
    # the abf16_wfp4 rows exercise that kernel instead of downgrading to a8w4/fp8.
    if aq_dtype == dtypes.bf16 and wq_dtype == dtypes.fp4x2:
        return GateMode.SEPARATED.value
    # a8w4 mxfp4 weights run the gate/up-interleaved (guinterleave) layout,
    # matching serving's ATOM_MOE_GU_ITLV=1. gate_mode is a runtime weight-layout
    # property (not a tuned-config key); request INTERLEAVE here for them.
    if aq_dtype == dtypes.fp8 and wq_dtype == dtypes.fp4x2:
        return GateMode.INTERLEAVE.value
    # mxfp8 (a8w8) uses the gate-up interleave stage1 path as well.
    if (
        q_type == aiter.QuantType.per_1x32
        and aq_dtype == dtypes.fp8
        and wq_dtype == dtypes.fp8
    ):
        return GateMode.INTERLEAVE.value
    return GateMode.SEPARATED.value


def _effective_swiglu_limit(quant_type, aq_dtype, wq_dtype, swiglu_limit):
    if (quant_type, aq_dtype, wq_dtype) in (
        _PER1X32_BF16_FP4,
        _PER1X32_FP8_FP4,
        _PER1X32_FP4_FP4,
    ):
        return swiglu_limit
    return None


def _runtime_situv2_mxfp4_q_dtype_a(q_type, wq_dtype):
    """Mirror fused_moe's SiTUv2 MXFP4 activation-dtype routing."""
    if q_type != aiter.QuantType.per_1x32 or wq_dtype != dtypes.fp4x2:
        return None

    if get_gfx() == "gfx1250":
        return (
            dtypes.fp8
            if os.environ.get("AITER_FORCE_A8W4", "0") == "1"
            else dtypes.fp4x2
        )

    # fused_moe tests SiTUv2 ahead of the Swiglu/INTERLEAVE branch, so gate mode
    # and token count do not enter into it -- mirror that order here, otherwise
    # a4w4/a8w4 rows are skipped as "mode mismatch" under gate_mode=INTERLEAVE
    # and the opt-in paths go untested.
    if os.environ.get("AITER_SITUV2_A8W4", "0") == "1":
        return dtypes.fp8
    if os.environ.get("AITER_SITUV2_A4W4", "0") == "1":
        return dtypes.fp4x2
    return dtypes.bf16


def _runtime_swiglu_mxfp4_q_dtype_a(
    token, act_type, gate_mode, q_type, aq_dtype, wq_dtype
):
    """Return the q_dtype_a that fused_moe will select for Swiglu MXFP4."""
    if act_type != aiter.ActivationType.Swiglu:
        return None
    if q_type != aiter.QuantType.per_1x32 or wq_dtype != dtypes.fp4x2:
        return None
    if aq_dtype not in [dtypes.bf16, dtypes.fp16, dtypes.fp8, dtypes.fp4x2]:
        return None

    gate_mode = GateMode(gate_mode)
    if gate_mode == GateMode.SEPARATED:
        bound = int(os.environ.get("GPTOSS_SWIGLU_MXFP4_BF16_BOUND", "256"))
        return dtypes.bf16 if token < bound else dtypes.fp4x2

    bound = int(os.environ.get("AITER_BF16_FP8_MOE_BOUND", "256"))
    return dtypes.bf16 if get_gfx() != "gfx950" or token < bound else dtypes.fp8


def _moe_2stage_reference_workspace_gib(kwargs):
    token = kwargs["token"]
    topk = kwargs["topk"]
    model_dim = kwargs["model_dim"]
    inter_dim = kwargs["inter_dim"]
    expert = kwargs["E"]
    stage1_dim = inter_dim * 2 if kwargs["use_g1u1"] else inter_dim

    # torch_moe_stage1/2 materialize routed fp32 activations and dequantized
    # fp32 weights. This is a preflight estimate; OOM is still caught per case.
    routed_activation_bytes = token * topk * (model_dim * 2 + stage1_dim) * 4
    dequant_weight_bytes = expert * model_dim * (stage1_dim + inter_dim) * 4
    input_score_bytes = token * (model_dim + expert) * 2
    return (routed_activation_bytes + dequant_weight_bytes + input_score_bytes) / (
        1024**3
    )


def _moe_2stage_route_workspace_skip_reason(kwargs):
    """Return a reason when effective route-reduce exceeds 32-bit addressing."""
    token = kwargs["token"]
    model_dim = kwargs["model_dim"]
    topk = kwargs["topk"]
    q_dtype_a = resolve_activation_dtype(
        kwargs["qType"],
        kwargs["WQDType"],
        activation=kwargs["actType"],
        gate_mode=kwargs["gateMode"],
        M=token,
        hidden_dtype=kwargs["dtype"],
        gfx=get_gfx(),
    )
    if q_dtype_a is None:
        return None
    is_shuffled = not (
        kwargs["WQDType"] == dtypes.fp4x2
        and kwargs["AQDType"] == dtypes.fp4x2
        and not kwargs.get("preshuffle", False)
    )
    supports_expert_bias = (
        kwargs["qType"] == aiter.QuantType.per_1x32
        and kwargs["WQDType"] == dtypes.fp4x2
        and kwargs["AQDType"] in (dtypes.bf16, dtypes.fp16, dtypes.fp8)
        and kwargs["actType"] != aiter.ActivationType.Situv2
    )
    try:
        metadata = get_2stage_cfgs(
            get_padded_M(token),
            model_dim,
            kwargs["inter_dim"],
            kwargs["E"],
            topk,
            kwargs["dtype"],
            q_dtype_a,
            kwargs["WQDType"],
            kwargs["qType"],
            kwargs["use_g1u1"],
            kwargs["actType"],
            kwargs["doweight_stage1"],
            kwargs.get("hidden_pad", 0),
            kwargs.get("intermediate_pad", 0),
            is_shuffled,
            kwargs["gateMode"],
            situ_beta=kwargs.get("beta", 1.0),
            situ_linear_beta=kwargs.get("linear_beta", 1.0),
            swiglu_limit=kwargs.get("swiglu_limit"),
            opus_weights_shuffled=is_shuffled,
            has_stage1_bias=supports_expert_bias,
            has_stage2_bias=(
                supports_expert_bias and not kwargs.get("disable_stage2_bias", False)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        aiter.logger.warning(
            "moe_2stage: failed to resolve stage2 mode during preflight; "
            "continuing to runtime validation: %s",
            exc,
        )
        return None
    if not _is_flydsl_stage2(metadata.stage2):
        return None
    if not _stage2_runtime_uses_route_reduce(metadata.stage2, token, model_dim):
        return None
    route_bytes = _flydsl_route_output_nbytes(
        metadata.stage2,
        token,
        topk,
        model_dim,
        torch.empty((), dtype=kwargs["dtype"]).element_size(),
    )
    if route_bytes <= FHMOE_BUFFER_MAX_BYTES:
        return None
    return (
        f"FlyDSL stage2 reduce workspace {route_bytes / (1024**3):.1f} GiB "
        "exceeds the current 32-bit buffer-addressing limit"
    )


def _format_moe_2stage_case(kwargs):
    return (
        f"token={kwargs['token']}, dim=({kwargs['model_dim']},{kwargs['inter_dim']}), "
        f"E={kwargs['E']}, topk={kwargs['topk']}, act={kwargs['actType']}, "
        f"q={kwargs['qType']}, aq={kwargs['AQDType']}, wq={kwargs['WQDType']}, "
        f"use_g1u1={kwargs['use_g1u1']}, doweight_stage1={kwargs['doweight_stage1']}"
    )


def _moe_2stage_skip_reason(kwargs):
    route_workspace_reason = _moe_2stage_route_workspace_skip_reason(kwargs)
    if route_workspace_reason is not None:
        return route_workspace_reason

    # Set AITER_MOE_2STAGE_MAX_FREE_MEMORY_FRACTION=0 to run manually.
    max_free_memory_fraction = float(
        os.environ.get("AITER_MOE_2STAGE_MAX_FREE_MEMORY_FRACTION", "0.8")
    )
    if max_free_memory_fraction <= 0:
        return None

    reference_gib = _moe_2stage_reference_workspace_gib(kwargs)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError as exc:
        aiter.logger.warning(
            "moe_2stage: failed to query GPU memory, continuing without "
            "preflight skip: %s",
            exc,
        )
        return None

    free_gib = free_bytes / (1024**3)
    total_gib = total_bytes / (1024**3)
    max_reference_gib = free_gib * max_free_memory_fraction
    if reference_gib <= max_reference_gib:
        return None

    return (
        f"estimated reference workspace {reference_gib:.1f} GiB exceeds "
        f"{max_free_memory_fraction:.0%} of current free GPU memory "
        f"({max_reference_gib:.1f} GiB of {free_gib:.1f} GiB free, "
        f"{total_gib:.1f} GiB total)"
    )


def _iter_legacy_cases():
    """Yield (kwargs, extras) for the original CLI-driven sweep."""
    extras = {"model": "legacy"}

    def _kw(
        dtype,
        m,
        model_dim,
        inter_dim,
        quant_type,
        aq_dtype,
        wq_dtype,
        doweight_stage1,
        act_type,
        use_g1u1=True,
        **over,
    ):
        return dict(
            dtype=dtype,
            token=m,
            model_dim=model_dim,
            inter_dim=inter_dim,
            E=args.expert,
            topk=args.topk,
            actType=act_type,
            gateMode=_effective_gate_mode(quant_type, aq_dtype, wq_dtype),
            qType=quant_type,
            AQDType=aq_dtype,
            WQDType=wq_dtype,
            use_g1u1=use_g1u1,
            doweight_stage1=doweight_stage1,
            strict_accuracy=False,
            check_aot_cache=False,
            swiglu_limit=_effective_swiglu_limit(
                quant_type, aq_dtype, wq_dtype, args.swiglu_limit
            ),
            **over,
        )

    for (
        dtype,
        (quant_type, aq_dtype, wq_dtype),
        (model_dim, inter_dim),
        doweight_stage1,
        stage2_bias,
    ) in itertools.product(
        args.dtype,
        l_quant,
        args.dim,
        args.doweight_stage1,
        args.stage2_bias,
    ):
        triple = (quant_type, aq_dtype, wq_dtype)

        if triple == _PER1X32_BF16_FP4:
            bf16_fp4_acts = [
                act_type
                for act_type in args.act
                if act_type in (aiter.ActivationType.Swiglu, aiter.ActivationType.Relu2)
            ] or [aiter.ActivationType.Swiglu]
            for hidden_pad, intermediate_pad in args.hidden_intermediate_pad:
                for act_type in bf16_fp4_acts:
                    for m in args.tokenNum:
                        yield _kw(
                            dtype,
                            m,
                            model_dim,
                            inter_dim,
                            quant_type,
                            aq_dtype,
                            wq_dtype,
                            doweight_stage1,
                            act_type,
                            use_g1u1=(act_type != aiter.ActivationType.Relu2),
                            hidden_pad=hidden_pad,
                            intermediate_pad=intermediate_pad,
                            disable_stage2_bias=not stage2_bias,
                        ), extras
        elif triple == _PER1X32_FP8_FP4:
            for hidden_pad, intermediate_pad in args.hidden_intermediate_pad:
                for act_type in args.act:
                    for m in args.tokenNum:
                        yield _kw(
                            dtype,
                            m,
                            model_dim,
                            inter_dim,
                            quant_type,
                            aq_dtype,
                            wq_dtype,
                            doweight_stage1,
                            act_type,
                            hidden_pad=hidden_pad,
                            intermediate_pad=intermediate_pad,
                            disable_stage2_bias=not stage2_bias,
                            **_situv2_beta_kwargs(act_type),
                        ), extras
        elif triple == _PER1X32_FP4_FP4:
            for preshuffle in args.preshuffle:
                for act_type in args.act:
                    if act_type == aiter.ActivationType.Relu2:
                        continue
                    for m in args.tokenNum:
                        yield (
                            _kw(
                                dtype,
                                m,
                                model_dim,
                                inter_dim,
                                quant_type,
                                aq_dtype,
                                wq_dtype,
                                doweight_stage1,
                                act_type,
                                use_g1u1=True,
                                preshuffle=preshuffle,
                                hidden_pad=0,
                                intermediate_pad=0,
                                disable_stage2_bias=not stage2_bias,
                                **_situv2_beta_kwargs(act_type),
                            ),
                            extras,
                        )
        elif triple == _PER1X32_BF16_I4:
            for m in args.tokenNum:
                yield _kw(
                    dtype,
                    m,
                    model_dim,
                    inter_dim,
                    quant_type,
                    aq_dtype,
                    wq_dtype,
                    doweight_stage1,
                    aiter.ActivationType.Silu,
                    disable_stage2_bias=not stage2_bias,
                ), extras
        else:
            for act_type in args.act:
                # SiTUv2 only routes to FlyDSL on per_1x32 + fp4/fp8; skip it for
                # every other quant so we never compare an unsupported combo.
                if (
                    act_type == aiter.ActivationType.Situv2
                    and triple not in _SITUV2_SUPPORTED_TRIPLES
                ):
                    continue
                for m in args.tokenNum:
                    yield (
                        _kw(
                            dtype,
                            m,
                            model_dim,
                            inter_dim,
                            quant_type,
                            aq_dtype,
                            wq_dtype,
                            doweight_stage1,
                            act_type,
                            use_g1u1=(act_type != aiter.ActivationType.Relu2),
                            disable_stage2_bias=not stage2_bias,
                            **_situv2_beta_kwargs(act_type),
                        ),
                        extras,
                    )


def test_route_workspace_token_capacity():
    cases = (
        (1, 1),
        (247, 256),
        (256, 256),
        (257, 512),
        (32768, 32768),
        (32769, 65536),
        (65536, 65536),
        (65537, 131072),
        (117626, 131072),
        (128332, 131072),
        (131072, 131072),
        (131073, 262144),
    )
    for token_num, expected_capacity in cases:
        assert _route_workspace_token_capacity(token_num) == expected_capacity

    for token_num in (0, -1):
        try:
            _route_workspace_token_capacity(token_num)
        except ValueError as error:
            assert "must be positive" in str(error)
        else:
            raise AssertionError(f"expected ValueError for token_num={token_num}")

    aiter.logger.info("moe_2stage: route workspace capacity passed")


def test_bm16_tiled_scale_boundary():
    """Validate tuned BM16 dispatch and the 33-row scale boundary."""
    if get_gfx() != "gfx950":
        aiter.logger.info("skip BM16 tiled-scale boundary test on %s", get_gfx())
        return

    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    expected_kernels = (
        "flydsl_moe1_afp8_wfp4_bf16_t16x128x256_w3_gui_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t16x256x256_atomic",
    )
    for token in (32, 64):
        metadata = get_2stage_cfgs(
            get_padded_M(token),
            7168,
            512,
            385,
            7,
            torch.bfloat16,
            dtypes.fp8,
            dtypes.fp4x2,
            aiter.QuantType.per_1x32,
            True,
            aiter.ActivationType.Silu,
            False,
            0,
            0,
            True,
            GateMode.INTERLEAVE.value,
        )
        assert metadata.block_m == 16
        stages = (metadata.stage1, metadata.stage2)
        kernel_names = tuple(stage.keywords["kernelName"] for stage in stages)
        assert kernel_names == expected_kernels
        stage1_params = get_flydsl_kernel_params(kernel_names[0])
        stage2_params = get_flydsl_kernel_params(kernel_names[1])
        assert stage1_params is not None
        assert stage2_params is not None
        assert stage1_params["tile_m"] == 16
        assert stage1_params["waves_per_eu"] == 3
        assert stage2_params["tile_m"] == 16
        assert stage2_params["tile_n"] == 256
        assert stage2_params.get("xcd_swizzle", 0) == 0

    old_moe_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    torch.manual_seed(0)
    try:
        test_fmoe(
            dtype=torch.bfloat16,
            token=33,
            model_dim=7168,
            inter_dim=512,
            E=385,
            topk=7,
            actType=aiter.ActivationType.Silu,
            gateMode=GateMode.INTERLEAVE.value,
            qType=aiter.QuantType.per_1x32,
            AQDType=dtypes.fp8,
            WQDType=dtypes.fp4x2,
            use_g1u1=True,
            strict_accuracy=True,
            check_aot_cache=False,
        )
    finally:
        if old_moe_bound is None:
            os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
        else:
            os.environ["AITER_BF16_FP8_MOE_BOUND"] = old_moe_bound


def test_output_buffer_contract():
    """Validate output identity, copy-back, validation, and compile contracts."""
    torch.manual_seed(0)
    token, model_dim, inter_dim, E, topk = 32, 512, 256, 8, 2
    dtype = dtypes.bf16
    hidden = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    gating = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(hidden, gating, topk, True)
    args = (hidden, w1, w2, topk_weights, topk_ids)

    ref = fused_moe(*args)
    # The comparisons below are exact, so check that is a kernel property.
    assert torch.equal(ref, fused_moe(*args)), "fused_moe is not run-to-run stable"

    def fresh_buffer():
        return torch.full((token, model_dim), -7.0, dtype=dtype)

    def call_and_read_back(buf):
        out = fused_moe(*args, output=buf)
        # Reading buf afterwards is the trap: compiled, it is answered from
        # whatever the fake said the op returned.
        return out.sum(), buf.sum()

    sort = aiter.fused_moe._moe_sorting_impl

    def sort_without_output(*a, **kw):
        kw["output"] = None
        return sort(*a, **kw)

    # in_place=False stands in for FLAT, adaptive-aux atomic, and grouped gfx1250
    # paths, which cannot take the caller's buffer and must copy their result into it.
    for in_place in (True, False):
        try:
            if not in_place:
                aiter.fused_moe._moe_sorting_impl = sort_without_output
            buf = fresh_buffer()
            assert fused_moe(*args, output=buf) is buf, f"{in_place=}: buf not returned"
            assert torch.equal(buf, ref), f"{in_place=}: buf does not hold the result"

            eager = call_and_read_back(fresh_buffer())
            compiled = torch.compile(call_and_read_back)(fresh_buffer())
            assert [float(x) for x in eager] == [
                float(x) for x in compiled
            ], f"{in_place=}: eager {eager} != compiled {compiled}"
            assert eager[0] == eager[1] == ref.sum()
        finally:
            aiter.fused_moe._moe_sorting_impl = sort

    def expect_raise(described, **kwargs):
        try:
            fused_moe(*args, **kwargs)
        except RuntimeError:
            return
        raise AssertionError(f"fused_moe accepted {described}")

    expect_raise("a mis-shaped output", output=torch.empty((token, model_dim + 1)))
    expect_raise(
        "an output of the wrong dtype",
        output=torch.empty((token, model_dim), dtype=dtypes.fp32),
    )
    expect_raise(
        "a non-contiguous output",
        output=torch.empty((token, model_dim * 2), dtype=dtype)[:, ::2],
    )
    # moe_sorting zeroes output before stage1 reads hidden_states.
    expect_raise("an output overlapping hidden_states", output=hidden)

    # Disjoint slices of one pool are the symmetric-buffer case, and stay legal.
    pool = torch.zeros((2 * token, model_dim), dtype=dtype)
    pool[:token] = hidden
    assert torch.equal(fused_moe(pool[:token], *args[1:], output=pool[token:]), ref)

    stage2_scatter = Stage2ScatterContext(
        arena_handle=0,
        combine_input_offset=0,
        slot_stride_bytes=1,
        max_tokens_per_rank=token,
        world_size=1,
        source_token_map=torch.zeros_like(topk_ids, dtype=dtypes.i32),
    )

    def call_with_stage2_scatter(buf):
        return fused_moe(
            *args,
            stage2_scatter=stage2_scatter,
            output=buf,
        )

    def expect_stage2_scatter_raise(described, call):
        try:
            call()
        except RuntimeError as error:
            assert "output= is incompatible with stage2_scatter" in str(error)
            return
        raise AssertionError(f"fused_moe accepted {described}")

    expect_stage2_scatter_raise(
        "output with stage2_scatter",
        lambda: call_with_stage2_scatter(fresh_buffer()),
    )
    expect_stage2_scatter_raise(
        "output with stage2_scatter under torch.compile",
        lambda: torch.compile(call_with_stage2_scatter)(fresh_buffer()),
    )

    aiter.logger.info("moe_2stage: output buffer contract passed")


def test_swiglu_mxfp4_nonsplit_nobias_dispatch():
    if get_gfx() != "gfx950":
        aiter.logger.info(
            "skip SwiGLU MXFP4 non-split no-bias dispatch test on %s", get_gfx()
        )
        return
    model_dim, inter_dim, E, topk = 6144, 1536, 129, 5
    for gate_up_layout in (True, False):
        for token in (128, 255, 256, 257, 512):
            runtime_aq_dtype = dtypes.fp8 if token >= 256 else dtypes.bf16
            metadata = get_2stage_cfgs(
                get_padded_M(token),
                model_dim,
                inter_dim,
                E,
                topk,
                torch.bfloat16,
                runtime_aq_dtype,
                dtypes.fp4x2,
                aiter.QuantType.per_1x32,
                True,
                aiter.ActivationType.Swiglu,
                False,
                0,
                0,
                True,
                GateMode.INTERLEAVE.value,
                has_stage1_bias=False,
                has_stage2_bias=False,
                gate_up_interleave=gate_up_layout,
            )
            stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
            stage2_func = getattr(metadata.stage2, "func", metadata.stage2)
            assert stage1_func is cktile_moe_stage1, (
                f"{gate_up_layout=} {token=}: expected CK-Tile stage1, got "
                f"{stage1_func!r}"
            )
            assert stage2_func is cktile_moe_stage2, (
                f"{gate_up_layout=} {token=}: expected CK-Tile stage2, got "
                f"{stage2_func!r}"
            )
            dispatched_split_k = metadata.stage1.keywords["split_k"]
            if gate_up_layout:
                assert dispatched_split_k == 1, (
                    f"{token=}: gate_up_layout=True (GUGU) must dispatch to the "
                    f"fused split_k=1 gate-up kernel, got split_k={dispatched_split_k}"
                )
            else:
                assert dispatched_split_k >= 2, (
                    f"{token=}: gate_up_layout=False (GGUU) must dispatch to "
                    f"split_k>=2 (NoBias_swiglu_SplitK path), got "
                    f"split_k={dispatched_split_k}; this is the bf16/fp16-activation "
                    "accuracy bug this PR originally fixed -- see PR discussion."
                )
            assert metadata.has_bias is False
            assert metadata.stage2_has_bias is False

            torch.manual_seed(0)
            test_fmoe(
                dtype=torch.bfloat16,
                token=token,
                model_dim=model_dim,
                inter_dim=inter_dim,
                E=E,
                topk=topk,
                actType=aiter.ActivationType.Swiglu,
                gateMode=GateMode.INTERLEAVE.value,
                qType=aiter.QuantType.per_1x32,
                AQDType=dtypes.fp8 if token >= 256 else dtypes.bf16,
                WQDType=dtypes.fp4x2,
                use_g1u1=True,
                preshuffle=True,
                strict_accuracy=True,
                check_aot_cache=False,
                disable_stage1_bias=True,
                disable_stage2_bias=True,
                gate_up_layout=gate_up_layout,
            )
    aiter.logger.info(
        "moe_2stage: SwiGLU MXFP4 non-split no-bias dispatch regression test "
        "(layout x M sweep) passed"
    )


def test_swiglu_mxfp4_untagged_layout_dispatch():
    """Callers that never tag w1.gate_up_interleave (None) -- ATOM's MiniMax-M3, for
    one -- must keep the CK-Tile split-k route with the standard post-activation
    layout for bf16 activations. MiniMax-M3-MXFP4 at TP8 (per-rank MoE call:
    model_dim 6144, inter_dim 512, intermediate_pad 0) fell through to the
    unshuffled-weights fallback instead and decoded garbage until max tokens."""
    if get_gfx() != "gfx950":
        aiter.logger.info(
            "skip SwiGLU MXFP4 untagged-layout dispatch test on %s", get_gfx()
        )
        return
    model_dim, inter_dim, intermediate_pad, E, topk = 6144, 512, 0, 129, 5
    for token in (1, 8, 32, 128):
        metadata = get_2stage_cfgs(
            get_padded_M(token),
            model_dim,
            inter_dim,
            E,
            topk,
            torch.bfloat16,
            dtypes.bf16,
            dtypes.fp4x2,
            aiter.QuantType.per_1x32,
            True,
            aiter.ActivationType.Swiglu,
            False,
            0,
            intermediate_pad,
            True,
            GateMode.SEPARATED.value,
            swiglu_limit=7.0,
            opus_weights_shuffled=True,
            gate_up_interleave=None,
        )
        stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
        assert (
            stage1_func is cktile_moe_stage1
        ), f"{token=}: expected CK-Tile stage1, got {stage1_func!r}"
        layout = metadata.stage1.keywords.get("post_activation_layout")
        assert layout == "standard", (
            f"{token=}: untagged gate/up layout with bf16 activations must keep "
            f"post_activation_layout='standard', got {layout!r}"
        )
        assert metadata.stage1.keywords["split_k"] >= 2, (
            f"{token=}: expected the CK-Tile split-k route, got "
            f"split_k={metadata.stage1.keywords['split_k']}"
        )
    aiter.logger.info(
        "moe_2stage: SwiGLU MXFP4 untagged-layout dispatch regression test passed"
    )


# The MXFP8 conversion in the fused sort+quant kernel is gfx950-only.
A8W4_SORT_QUANT_GFX = ["gfx950"]
# Kimi-K3 A8W4 MoE: E=896, topk=16, model_dim=3584 (inter_dim 384 at TP=8,
# 512 at TP=6).
_KIMI_A8W4_E, _KIMI_A8W4_TOPK, _KIMI_A8W4_MODEL_DIM = 896, 16, 3584


@benchmark()
def test_a8w4_sort_quant_kernel(
    token,
    row_stride,
    routing,
    E=_KIMI_A8W4_E,
    topk=_KIMI_A8W4_TOPK,
    model_dim=_KIMI_A8W4_MODEL_DIM,
):
    """The MXFP8 sort+quant kernel at the Kimi-K3 A8W4 shape.

    Its sort must give the MXFP4 sort_quant's expert blocks and be a valid
    routing (order within an expert block is not deterministic), its quant must
    match per_1x32_f8_scale_f8_quant, it must read a row-strided input
    (row_stride > model_dim), and it must zero moe_buf for the atomic stage2.
    routing="mixed" pads every other route with -1, which both sorts must skip.
    """
    mb = 32
    alloc = aiter.fused_moe._alloc_aux_sort_buffers

    def alloc_bufs(quant_dtype):
        _, *sort_bufs = alloc(token, E, topk, mb, torch.device("cuda"))
        if quant_dtype == dtypes.fp8:
            a_quant = torch.empty((token, model_dim), dtype=dtypes.fp8)
        else:
            a_quant = torch.empty((token, model_dim // 2), dtype=torch.uint8)
        a_scale = torch.empty((token, model_dim // 32), dtype=torch.uint8)
        moe_buf = torch.full((token, model_dim), float("nan"), dtype=dtypes.bf16)
        return (*sort_bufs, a_quant, a_scale, moe_buf)

    def launch(hidden, topk_ids, topk_weights, bufs):
        ids, experts, nv, weights, rev, m_idx, a_quant, a_scale, moe_buf = bufs
        aiter.mxfp4_moe_sort_quant(
            a_input=hidden,
            topk_ids=topk_ids,
            topk_weight=topk_weights,
            sorted_token_ids=ids,
            sorted_expert_ids=experts,
            cumsum_tensor=nv,
            reverse_sorted=rev,
            sorted_weights=weights,
            a_quant=a_quant,
            a_scale=a_scale,
            m_indices=m_idx,
            bf16_zero_out=moe_buf,
            NE=E,
            TOPK=topk,
            D_HIDDEN=model_dim,
            MB=mb,
        )
        return bufs

    torch.manual_seed(0)
    # A row_stride above model_dim makes the input a row-strided view.
    hidden = torch.randn((token, row_stride), dtype=dtypes.bf16)[:, :model_dim]
    gating = torch.randn((token, E), dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(hidden, gating, topk, True)
    if routing == "mixed":
        # moe_sorting accepts -1 padding routes (test_moe_sorting.py "mixed").
        invalid = torch.arange(token * topk).view(token, topk) % 2 == 1
        topk_ids = topk_ids.masked_fill(invalid, -1)
        topk_weights = topk_weights.masked_fill(invalid, 0)
    flat_ids = topk_ids.flatten().long()
    valid = flat_ids >= 0
    n_routes = int(valid.sum())
    counts = torch.bincount(flat_ids[valid], minlength=E)
    expected_n_valid = int(((counts + mb - 1) // mb * mb).sum())
    # The MXFP4 sort_quant needs dense rows. At token=1, .contiguous() keeps the
    # row-strided one-row view, which it must still accept: one row has no
    # row stride to honor.
    hidden_mxfp4 = hidden.contiguous()
    if token == 1:
        assert hidden_mxfp4.stride(0) == row_stride
    ref_q, ref_s = per_1x32_f8_scale_f8_quant(
        hidden_mxfp4, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    bufs4 = launch(hidden_mxfp4, topk_ids, topk_weights, alloc_bufs(dtypes.fp4x2))
    experts4, nv4 = bufs4[1], bufs4[2]

    def check(bufs):
        ids8, experts8, nv8, w8, rev, m_idx, q8, s8, buf8 = bufs
        n_valid = int(nv8[0].item())
        assert n_valid == expected_n_valid, f"M={token}: num_valid_ids {n_valid}"
        assert torch.equal(nv8, nv4), f"M={token}: num_valid_ids differ"
        n_blocks = n_valid // mb
        assert torch.equal(
            experts8[:n_blocks], experts4[:n_blocks]
        ), f"M={token}: expert blocks differ"
        rev_all = rev[: token * topk]
        assert torch.all(rev_all[~valid] == -1), f"M={token}: -1 route was placed"
        # Every valid route lands once, in a block of its expert, with its weight.
        route = rev_all[valid].long()
        tok = torch.arange(token).repeat_interleave(topk)[valid]
        slot = torch.arange(topk).repeat(token)[valid]
        assert torch.equal(ids8[route].long(), slot << 24 | tok), f"M={token}: ids"
        assert torch.equal(
            w8[route], topk_weights.flatten()[valid]
        ), f"M={token}: bad weights"
        assert torch.equal(
            experts8[route // mb].long(), flat_ids[valid]
        ), f"M={token}: route in the wrong expert block"
        assert torch.equal(m_idx[route].long(), tok), f"M={token}: bad m_indices"
        # The rest of the valid region is padding.
        assert (
            int((ids8[:n_valid] != token).sum()) == n_routes
        ), f"M={token}: bad padding"
        assert torch.equal(
            s8, ref_s.view(torch.uint8)
        ), f"M={token}: e8m0 scales differ"
        assert torch.equal(
            q8.view(torch.uint8), ref_q.view(torch.uint8)
        ), f"M={token}: fp8 activations differ"
        assert torch.count_nonzero(buf8) == 0, f"M={token}: moe_buf not zeroed"

    # First launch into a NaN moe_buf checks the zeroing; the timed launches
    # reuse the same buffers, and their final output is checked again.
    bufs8 = launch(hidden, topk_ids, topk_weights, alloc_bufs(dtypes.fp8))
    check(bufs8)
    bufs8, us = run_perftest(launch, hidden, topk_ids, topk_weights, bufs8)
    check(bufs8)
    q8 = bufs8[6]
    err = checkAllclose(
        ref_q.to(dtypes.fp32),
        q8.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg=f"a8w4 sort_quant M={token} stride={row_stride}: fp8 activations ",
    )
    # bf16 input + fp8 out + e8m0 scales + zeroed moe_buf + routing in/out.
    nbytes = (
        token * model_dim * (2 + 1 + 2)
        + token * model_dim // 32
        + topk_ids.nbytes
        + topk_weights.nbytes
        + sum(b.nbytes for b in bufs8[:6])
    )
    return {
        "gfx": get_gfx(),
        "sort_quant_fp8 us": us,
        "sort_quant_fp8 TB/s": nbytes / us / 1e6,
        "sort_quant_fp8 err": err,
    }


@functools.lru_cache(maxsize=1)
def _kimi_a8w4_weights(inter_dim, E, model_dim):
    """Per-1x32 fp4x2 weights: raw (torch reference) and shuffled (fused_moe)."""
    torch.manual_seed(0)
    dtype = dtypes.bf16
    torch_quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    del w1
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    del w2
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    raw = (w1_qt, w2_qt, w1_scale, w2_scale)
    # a8w4 runs gate/up-interleaved w1 (gate_mode INTERLEAVE).
    shuffled = (
        shuffle_weight_a16w4(w1_qt, 16, True),
        shuffle_weight_a16w4(w2_qt, 16, False),
        shuffle_scale_a16w4(w1_scale, E, True),
        shuffle_scale_a16w4(w2_scale, E, False),
    )
    return raw, shuffled


@benchmark()
def test_a8w4_fused_sort_quant_e2e(
    token,
    inter_dim,
    E=_KIMI_A8W4_E,
    topk=_KIMI_A8W4_TOPK,
    model_dim=_KIMI_A8W4_MODEL_DIM,
):
    """fused_moe with the fused sort+quant against the separate launches.

    Candidates: "fused" (bf16 input, default path), "unfused" (bf16 input with
    the fused sort+quant disabled) and "fp8_input" (MXFP8 dispatch: fp8 input
    with an unsorted e8m0 a1_scale, whose scales must still get sorted). Only
    "fused" may take the fused sort+quant, and only up to the token cap. Each is
    checked against the torch reference, and "fused" must also match the other
    two element-wise.
    """
    dtype = dtypes.bf16
    beta, linear_beta = DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
    raw, shuffled = _kimi_a8w4_weights(inter_dim, E, model_dim)
    w1_qt, w2_qt, w1_scale, w2_scale = raw
    w1_sh, w2_sh, w1_scale_sh, w2_scale_sh = shuffled
    kwargs = {
        "w1_scale": w1_scale_sh,
        "w2_scale": w2_scale_sh,
        "quant_type": aiter.QuantType.per_1x32,
        "activation": aiter.ActivationType.Situv2,
        "beta": beta,
        "linear_beta": linear_beta,
        "gate_mode": GateMode.INTERLEAVE.value,
    }

    torch.manual_seed(0)
    hidden = torch.randn((token, model_dim), dtype=dtype)
    gating = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(hidden, gating, topk, True)
    args = (w1_sh, w2_sh, topk_weights, topk_ids)
    a1, a1_scale = per_1x32_f8_scale_f8_quant(
        hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )

    # Torch reference (untimed), as in test_fmoe for a8w4.
    out1_ref = torch_moe_stage1(
        a1,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=aiter.ActivationType.Situv2,
        quant_type=aiter.QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )
    a2, a2_scale = per_1x32_f8_scale_f8_quant(
        out1_ref, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    ref = torch_moe_stage2(
        a2.view(token, topk, -1),
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=aiter.QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
    )

    def calc_diff(x, y):
        x, y = x.double(), y.double()
        return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())

    fm = aiter.fused_moe
    candidates = {
        "fused": lambda: fused_moe(hidden, *args, **kwargs),
        "unfused": lambda: fused_moe(hidden, *args, **kwargs),
        "fp8_input": lambda: fused_moe(
            a1, *args, a1_scale=a1_scale, dtype=dtype, **kwargs
        ),
    }
    # Set per candidate so AITER_MOE_A8W4_FUSED_SORT_QUANT=0 cannot skew them;
    # fp8_input keeps fusion on so only its dtype gate excludes it.
    fusion_enabled = {"fused": True, "unfused": False, "fp8_input": True}

    flops = 2 * token * topk * model_dim * inter_dim * 3
    n_experts = int(topk_ids.unique().numel())
    nbytes = (
        n_experts * (w1_sh.nbytes + w2_sh.nbytes + w1_scale_sh.nbytes) // E
        + n_experts * w2_scale_sh.nbytes // E
        + hidden.nbytes
        + token * model_dim * hidden.element_size()
    )

    real_fused = fm._a8w4_fused_sort_quant
    real_fusion_enabled = fm._MOE_A8W4_FUSED_SORT_QUANT
    calls = []

    def counting_fused(*a, **kw):
        calls.append(a[0].shape[0])
        return real_fused(*a, **kw)

    cap = fm._A8W4_SORT_QUANT_MAX_TOKENS
    tag = f"inter_dim={inter_dim} token={token}"
    ret = {"gfx": get_gfx()}
    outs = {}
    with override_env("AITER_SITUV2_A8W4", "1"):
        try:
            fm._a8w4_fused_sort_quant = counting_fused
            for name, fn in candidates.items():
                fm._MOE_A8W4_FUSED_SORT_QUANT = fusion_enabled[name]
                calls.clear()
                out, us = run_perftest(fn)
                torch.cuda.synchronize()
                want = {token} if name == "fused" and token <= cap else set()
                assert set(calls) == want, f"{tag} {name}: fused calls {set(calls)}"
                assert not out.isnan().any(), f"{tag} {name}: NaN"
                # The fp32 torch reference differs from every kernel path by
                # ~2% (fp8 intermediate, accumulation order), so err uses a
                # matching tolerance and accuracy is gated on logits_diff, as
                # in test_fmoe.
                err = checkAllclose(
                    ref.to(dtypes.fp32),
                    out.to(dtypes.fp32),
                    rtol=5e-2,
                    atol=0.5,
                    msg=f"{tag} {name} ",
                )
                diff = calc_diff(ref, out)
                assert diff < 1e-2, f"{tag} {name}: logits_diff {diff} vs torch"
                outs[name] = out
                ret[f"{name} us"] = us
                ret[f"{name} TFLOPS"] = flops / us / 1e6
                ret[f"{name} TB/s"] = nbytes / us / 1e6
                ret[f"{name} err"] = err
                ret[f"{name} logits_diff"] = diff
        finally:
            fm._a8w4_fused_sort_quant = real_fused
            fm._MOE_A8W4_FUSED_SORT_QUANT = real_fusion_enabled

    for name in ("unfused", "fp8_input"):
        other = outs[name]
        # Atomic bf16 accumulation reorders run to run: even two calls on the
        # same path differ by up to 0.375 at |out| ~ 32 (bf16 step 0.25). atol
        # is two such steps; every element must pass.
        mismatch = checkAllclose(
            outs["fused"],
            other,
            rtol=1e-2,
            atol=0.5,
            tol_err_ratio=0.0,
            msg=f"{tag}: fused vs {name} ",
        )
        assert mismatch == 0, f"{tag}: fused vs {name} mismatch {mismatch}"
        diff = calc_diff(outs["fused"], other)
        assert diff < 1e-3, f"{tag}: fused vs {name} diff {diff}"
    return ret


def _log_summary(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def run_a8w4_sort_quant_tests():
    if get_gfx() not in A8W4_SORT_QUANT_GFX:
        aiter.logger.warning("a8w4 fused sort_quant unsupported on %s; skip", get_gfx())
        return
    _log_summary(
        "a8w4 sort_quant kernel",
        [
            test_a8w4_sort_quant_kernel(token, row_stride, routing)
            for token, row_stride, routing in itertools.product(
                (1, 7, 32), (_KIMI_A8W4_MODEL_DIM, 4096), ("valid", "mixed")
            )
        ],
    )
    cap = aiter.fused_moe._A8W4_SORT_QUANT_MAX_TOKENS
    _log_summary(
        "a8w4 fused sort_quant e2e",
        [
            test_a8w4_fused_sort_quant_e2e(token, inter_dim)
            for inter_dim, token in itertools.product((384, 512), (1, cap, cap + 1))
        ],
    )
    _kimi_a8w4_weights.cache_clear()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _iter_with_env(case_iter, **env_overrides):
    """Keep temporary environment overrides active while cases are consumed."""
    with ExitStack() as stack:
        for name, value in env_overrides.items():
            stack.enter_context(override_env(name, value))
        yield from case_iter


_case_iters = []
test_route_workspace_token_capacity()
if args.bm16_scale_boundary:
    test_bm16_tiled_scale_boundary()
else:
    test_output_buffer_contract()
    test_swiglu_mxfp4_nonsplit_nobias_dispatch()
    test_swiglu_mxfp4_untagged_layout_dispatch()
    run_a8w4_sort_quant_tests()
    # Skip unrelated tuned-CSV validation for an explicit CLI quant sweep.
    if not args.no_flydsl_csv and args.quant is None:
        _case_iters.append(
            _iter_with_env(
                _iter_csv_cases(),
                AITER_SITUV2_A8W4="1",
                AITER_SITUV2_A4W4=None,
            )
        )
    if not args.no_legacy:
        _case_iters.append(_iter_legacy_cases())
case_iter = itertools.chain(*_case_iters)

_csv_out = os.environ.get("AITER_TUNED_OP_BENCH_CSV", "tuned_op_bench.csv")


def _write_bench_csv(rows):
    if not _csv_out or len(rows) == 0:
        return
    row = rows[-1]
    if row.get("model") == "legacy":
        return
    written = append_tuned_op_bench_rows(
        _csv_out,
        [row],
        op_name="moe_2stage",
        metric_cols=("us",),
        default_impl="fused_moe",
    )
    if written:
        aiter.logger.info(
            "moe_2stage: appended %d tuned op bench row(s) to %s", written, _csv_out
        )


df = []
seen = 0
for kwargs, extras in case_iter:
    seen += 1
    skip_reason = _moe_2stage_skip_reason(kwargs)
    if skip_reason is not None:
        aiter.logger.warning(
            "skip moe_2stage case: %s (%s)",
            _format_moe_2stage_case(kwargs),
            skip_reason,
        )
        continue
    _old_moe_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    _force_moe_bound_zero = (
        kwargs["qType"],
        kwargs["AQDType"],
        kwargs["WQDType"],
    ) in (_PER1X32_BF16_FP4, _PER1X32_FP8_FP4)
    if _force_moe_bound_zero:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    try:
        aot_guard = (
            run_only_env() if kwargs.get("check_aot_cache", False) else nullcontext()
        )
        with aot_guard:
            ret = test_fmoe(
                **kwargs, kernel_bench=args.kernel, ref_dtype=args.ref_dtype
            )
    except torch.cuda.OutOfMemoryError as exc:
        aiter.logger.warning(
            "skip moe_2stage case after OOM: %s (%s)",
            _format_moe_2stage_case(kwargs),
            exc,
        )
        ret = None
    finally:
        if _force_moe_bound_zero:
            if _old_moe_bound is None:
                os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
            else:
                os.environ["AITER_BF16_FP8_MOE_BOUND"] = _old_moe_bound
        # Reclaim per-case VRAM to avoid OOM under fragmentation.
        gc.collect()
        torch.cuda.empty_cache()
    if ret is None:
        continue
    ret.update(extras)
    df.append(ret)
    _write_bench_csv(df)

aiter.logger.info(
    "moe_2stage: scanned %d cases, recorded %d results (skipped %d)",
    seen,
    len(df),
    seen - len(df),
)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_2stage summary (markdown):\n%s", df_md)
