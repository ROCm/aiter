# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch references and compare functions used by the fused MoE tuner."""

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    torch_dynamic_mxfp8_quant,
    torch_moe,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.fused_moe_bf16_asm import torch_moe_tkw1
from aiter.ops.flydsl.moe_common import DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
from aiter.test_common import COS_DIFF_THRESHOLD, cosine_diff_compare


def _a16w_sorted_cos(ref, res, msg="", printLog=True):
    """compare_fn for a16w-mix stage1/stage2 in SORTED layout: ref is the sorted bf16
    reference ([max_sorted, inter], padding rows all-zero), res the kernel's sorted
    output. Cosine-diff over the valid (non-zero-ref) rows only. Same semantics as
    cosine_diff_compare (0 == exact); avoids the gather/re-sort that would otherwise
    pollute the kernel's tuned latency.
    """
    n = ref.shape[0]
    valid = (ref.abs().sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return 1.0
    x = ref[valid].double().flatten()
    y = res[:n][valid].double().flatten()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    if printLog:
        tag = "passed~" if cos_diff < COS_DIFF_THRESHOLD else "failed!"
        print(f"{msg}[cosine_diff={cos_diff:.6f} {tag}]")
    return cos_diff


def _token_major_to_sorted_routes(
    values, sorted_ids, num_valid_ids, token_num, *, mark_padding=False
):
    """Gather [token, topk, K] values into the padded sorted-route row ABI."""
    packed = sorted_ids.to(torch.int64)
    token = packed & 0x00FFFFFF
    slot = (packed >> 24) & 0xFF
    valid = (
        (torch.arange(sorted_ids.numel(), device=sorted_ids.device) < num_valid_ids[0])
        & (token < int(token_num))
        & (slot < int(values.shape[1]))
    )
    out = torch.empty(
        (sorted_ids.numel(), values.shape[-1]), dtype=values.dtype, device=values.device
    )
    if mark_padding:
        out.fill_(float("nan"))
    else:
        out.zero_()
    out[valid] = values[token[valid], slot[valid]]
    return out


def sorted_stage1_cosine_diff_compare(ref, res, msg="", printLog=True):
    # Padding rows are intentionally not written by Stage1. The reference keeps
    # them NaN, so compare only rows containing a real route. A real route that
    # quantizes to all zeros remains valid and is still checked.
    valid = ~torch.isnan(ref.float()).all(dim=-1)
    return cosine_diff_compare(ref[valid], res[valid], msg=msg, printLog=printLog)


def run_torch_moe_stage1(
    a1_qt,
    w1_qt,
    w2_qt,
    topk_weights,
    topk_ids,
    a1_scale,
    w1_scale,
    sorted_ids=None,
    num_valid_ids=None,
    w1_bias=None,
    dtype=dtypes.bf16,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    topk=1,
    blockM=32,
    fuse_fp4=False,
    fuse_fp8=False,
    situ_beta=DEFAULT_SITUV2_BETA,
    situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    swiglu_limit=None,
    output_sorted=False,
):
    # a16wi4: convert int8 weights to i4x2 so reference function detects the right path
    if (
        quant_type == QuantType.per_1x32
        and w1_qt.dtype == dtypes.i8
        and w1_scale is not None
        and w1_scale.dtype == dtypes.bf16
    ):
        w1_qt = w1_qt.view(dtypes.i4x2)
        w2_qt = w2_qt.view(dtypes.i4x2)
    ref1 = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        activation=activation,
        quant_type=quant_type,
        dtype=dtype,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        w1_bias=w1_bias,
        doweight=doweight_stage1,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
    )
    token_num = a1_qt.shape[0]
    if fuse_fp4:
        from aiter.ops.quant import per_1x32_f4_quant

        a2, _a2_scale = per_1x32_f4_quant(ref1, quant_dtype=dtypes.fp4x2)
        return a2.view(token_num, topk, -1)
    elif fuse_fp8:
        inter_dim = ref1.shape[-1]
        a2_fp8_bytes, _a2_scale_e8m0 = torch_dynamic_mxfp8_quant(
            ref1.reshape(-1, inter_dim)
        )
        a2 = a2_fp8_bytes.view(dtypes.fp8).view(token_num, topk, inter_dim)
        if output_sorted:
            a2 = _token_major_to_sorted_routes(
                a2, sorted_ids, num_valid_ids, token_num, mark_padding=True
            )
        return a2
    if quant_type == QuantType.per_1x128:
        ref1, _ref_scale = aiter.pertoken_quant(
            ref1.view(ref1.shape[0], -1, 128), quant_dtype=a1_qt.dtype
        )
        ref1 = ref1.view(ref1.shape[0], topk, -1)
    return ref1


def run_torch_moe_stage2(
    a2_qt,
    w1_qt,
    w2_qt,
    topk_weights,
    topk_ids,
    a2_scale,
    w2_scale,
    w2_bias=None,
    dtype=dtypes.bf16,
    quant_type=QuantType.No,
    doweight_stage1=False,
):
    # a16wi4: convert int8 weights to i4x2 so reference function detects the right path
    if (
        quant_type == QuantType.per_1x32
        and w2_qt.dtype == dtypes.i8
        and w2_scale is not None
        and w2_scale.dtype == dtypes.bf16
    ):
        w1_qt = w1_qt.view(dtypes.i4x2)
        w2_qt = w2_qt.view(dtypes.i4x2)
    return torch_moe_stage2(
        a2_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype,
        quant_type,
        a2_scale=a2_scale,
        w2_scale=w2_scale,
        w2_bias=w2_bias,
        doweight=not doweight_stage1,
    )


def run_torch_moe_stage1_ref(
    a1_qt,
    w1_qt,
    w2_qt,
    topk_weights,
    topk_ids,
    a1_scale,
    w1_scale,
    dtype,
    activation,
    quant_type,
    doweight_stage1,
    topk,
):
    ref1 = run_torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        activation=activation,
        quant_type=quant_type,
        dtype=dtype,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight_stage1=doweight_stage1,
        topk=topk,
    )
    token = a1_qt.shape[0]
    inter_dim = w2_qt.shape[-1]
    if quant_type == QuantType.per_1x128:
        ref1, ref_scale = aiter.pertoken_quant(
            ref1.view(ref1.shape[0], -1, 128), quant_dtype=a1_qt.dtype
        )
        ref1 = ref1.view(ref1.shape[0], topk, -1)
        ref_scale = ref_scale.view(token, -1)
        a2_qt = ref1
        a2_qt = a2_qt.view(token, topk, -1)
        a2_scale = ref_scale
        ratio = a1_scale.element_size() // a1_qt.element_size()
        out1 = torch.zeros(
            (token + (token * ratio + 127) // 128, topk, inter_dim),
            dtype=a1_qt.dtype,
            device=a1_qt.device,
        )
        ref1_asm = torch.zeros_like(out1)
        ref1_asm[:token] = a2_qt
        ref1_asm[token:, ...].view(-1)[: token * topk * inter_dim * ratio // 128] = (
            a2_scale.view(a1_qt.dtype).view(-1)
        )
        return ref1_asm

    else:
        out1 = torch.empty(
            (token, topk, inter_dim),
            dtype=dtype,
            device=a1_qt.device,
        )
        return ref1


## 1 stage ref
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    activation=ActivationType.Silu,
    doweight_stage1=False,
    q_type_a=dtypes.fp8,
):
    if doweight_stage1 & (q_type_a == dtypes.fp8):
        return torch_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            None,
            activation,
        )
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        None,
        activation,
    )
