# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference MoE used by the tests to check every backend.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module moved here from aiter/fused_moe.py, which re-exports it for the tests
(`from aiter.fused_moe import torch_moe`, ...). Changes must keep the results those
tests check against and should loop in the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.ops.moe_limits import RUNTIME_SILU_LIMIT_QUANT_TYPES, normalize_swiglu_limit
from aiter.utility import dtypes

from ..policy import quant_remap
from ..shapes import get_inter_dim


def _mxfp4_per_1x32_weight_expert_to_f32(weight, scale, expert_id, rows, cols):
    from aiter.utility import fp4_utils

    weight_e = fp4_utils.mxfp4_to_f32(weight[expert_id])
    scale_e = scale.reshape(weight.shape[0], rows, cols // 32)[expert_id]
    return (
        weight_e.view(rows, cols // 32, 32) * scale_e.view(rows, cols // 32, 1)
    ).view(rows, cols)


def _int4_per_1x32_weight_expert_to_f32(weight, scale, expert_id, rows, cols):
    scale_e = scale.reshape(weight.shape[0], cols // 32, rows)[expert_id]
    return (
        weight[expert_id].to(dtypes.fp32).reshape(rows, cols // 32, 32)
        * scale_e.permute(1, 0).unsqueeze(-1)
    ).reshape(rows, cols)


def torch_moe_act(act_input, torch_act, inter_dim, activation=None):
    # OAI SwiGLU is a gated (g1u1) activation with its own formula; it has no
    # single-input torch_act, so it must never reach torch_act below.
    if activation == ActivationType.Swiglu:
        if act_input.shape[-1] != 2 * inter_dim:
            raise NotImplementedError("ActivationType.Swiglu requires g1u1 weights")
        gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
        return swiglu(gate, up)
    if act_input.shape[-1] == inter_dim:
        return torch_act(act_input)
    else:
        gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
        return torch_act(gate) * up


def torch_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    torch_act = aiter.get_torch_act(activation)
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )

    inter_dim = w2.shape[2]

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
        w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            act_out = torch_moe_act(
                act_input, torch_act, inter_dim, activation=activation
            )
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


# temp workaround for swiglu
def swiglu(x_glu, x_linear, alpha: float = 1.702, limit: float | None = 7.0):
    if limit is None:
        limit = 7.0
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def situv2(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    situ_gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_scaled = linear_beta * torch.tanh(up / linear_beta)
    return situ_gate * up_scaled


def torch_moe_stage1(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weight,
    topk_ids,
    dtype=dtypes.fp16,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    # following for quant
    a1_scale=None,  # [token, 1]
    w1_scale=None,  # [expert, inter_dim, 1]
    w1_bias=None,  # [expert, inter_dim, 1]
    doweight=False,
    swiglu_limit=None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
):
    quant_type = quant_remap.get(quant_type, quant_type)
    if (
        quant_type in RUNTIME_SILU_LIMIT_QUANT_TYPES
        and activation == ActivationType.Silu
    ):
        swiglu_limit = normalize_swiglu_limit(swiglu_limit)
    ctype = dtypes.fp32  # compute type
    B, _D = hidden_states.shape
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    lazy_w1_int4_dequant = False
    lazy_w1_mxfp4_dequant = False
    if quant_type == QuantType.per_1x32 and w1.dtype == dtypes.i4x2:
        # a16wi4: int4 weights viewed as int8 for compute
        hidden_states = hidden_states.to(ctype)
        w1 = w1.view(dtypes.i8)
        lazy_w1_int4_dequant = True
    elif quant_type == QuantType.per_1x32:
        from aiter.utility import fp4_utils

        if w1.dtype == dtypes.fp8:  # mxfp8 weight
            w1 = w1.to(ctype)
        else:
            lazy_w1_mxfp4_dequant = True
        w1_scale = fp4_utils.e8m0_to_f32(w1_scale)
        if a1_scale is not None:  # skip a16w4 / mxfp8-bf16-activation ref
            if hidden_states.dtype == dtypes.fp8:  # a8w4 mxfp8 activation
                hidden_states = hidden_states.to(ctype)
            else:
                hidden_states = fp4_utils.mxfp4_to_f32(hidden_states)
            a1_scale = fp4_utils.e8m0_to_f32(a1_scale)
        else:  # a16w4 / mxfp8 (bf16 reference activation)
            hidden_states = hidden_states.to(ctype)
    else:
        hidden_states = hidden_states.to(ctype)
        w1 = w1.to(ctype)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        w1 = w1 * w1_scale.view(w1_scale.shape[0], -1, 1)
        hidden_states = hidden_states * a1_scale
    # per_128x128
    elif quant_type in [QuantType.per_128x128, QuantType.per_1x128]:
        w1_shape = w1.shape
        w1 = w1.view(
            w1.shape[0], w1.shape[1] // 128, 128, w1.shape[2] // 128, 128
        ) * w1_scale.view(
            w1_scale.shape[0], w1.shape[1] // 128, 1, w1.shape[2] // 128, 1
        )
        w1 = w1.view(w1_shape)

        a1_scale = a1_scale.view(hidden_states.shape[0], -1, 1)
        a1_scale = a1_scale.repeat(
            1, 1, hidden_states.shape[-1] // a1_scale.shape[1]
        ).view(hidden_states.shape[0], -1)
        hidden_states = hidden_states * a1_scale
    elif quant_type == QuantType.No:
        pass
    elif (
        quant_type == QuantType.per_1x32
        and w1_scale is not None
        and w1_scale.dtype == dtypes.bf16
    ):
        # a16wi4: groupwise dequant int4 weights with scale [E, K//32, N]
        group_size = 32
        num_groups = model_dim // group_size
        if not lazy_w1_int4_dequant:
            w1_shape = w1.shape
            # w1: [E, N, K] -> apply scale per group of K
            w1 = w1.reshape(E, N, num_groups, group_size)
            w1.mul_(w1_scale.reshape(E, num_groups, N).permute(0, 2, 1).unsqueeze(-1))
            w1 = w1.reshape(w1_shape)
        # activations are bf16, no scaling needed
    elif quant_type == QuantType.per_1x32:
        if not lazy_w1_mxfp4_dequant:
            w1_shape = w1.shape
            w1 = w1.view(E, N, model_dim // 32, 32) * w1_scale.view(
                E, N, model_dim // 32, 1
            )
            w1 = w1.view(w1_shape)

        a1_shape = hidden_states.shape
        hidden_states = hidden_states.view(a1_shape[0], a1_shape[1] // 32, 32)
        if a1_scale is not None:
            a1_scale = a1_scale[: a1_shape[0]]
            hidden_states = hidden_states * a1_scale.view(
                a1_shape[0], a1_shape[1] // 32, 1
            )
        hidden_states = hidden_states.view(a1_shape)
    else:
        assert False, f"Unsupported quant_type: {quant_type}"

    hidden_states = hidden_states.view(B, -1, model_dim).repeat(1, topk, 1)

    out = torch.zeros(
        (B, topk, N),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if lazy_w1_int4_dequant:
                w1_e = _int4_per_1x32_weight_expert_to_f32(
                    w1, w1_scale, E_id, N, model_dim
                )
            elif lazy_w1_mxfp4_dequant:
                w1_e = _mxfp4_per_1x32_weight_expert_to_f32(
                    w1, w1_scale, E_id, N, model_dim
                )
            else:
                w1_e = w1[E_id]
            act_input = sub_tokens @ (w1_e.transpose(0, 1))
            if doweight:
                act_input = act_input * topk_weight[mask].view(-1, 1)
            out[mask] = act_input
            if w1_bias is not None:
                out[mask] = out[mask] + w1_bias[E_id].view(1, -1)
    use_g1u1 = w1.shape[1] == (2 * inter_dim)
    use_swiglu = activation == aiter.ActivationType.Swiglu
    use_situv2 = activation == aiter.ActivationType.Situv2
    torch_act = aiter.get_torch_act(activation)
    if use_g1u1:
        gate, up = out.split([inter_dim, inter_dim], dim=-1)
        if use_swiglu:
            out = swiglu(gate, up, limit=swiglu_limit)
        elif use_situv2:
            out = situv2(gate, up, beta=situ_beta, linear_beta=situ_linear_beta)
        else:
            if swiglu_limit:
                gate, up = _clamp_swiglu_gate_up(gate, up, swiglu_limit)
            out = torch_act(gate) * up
    else:
        out = torch_act(out)
    return out.to(dtype)


def _clamp_swiglu_gate_up(gate: torch.Tensor, up: torch.Tensor, limit: float):
    limit = min(limit, torch.finfo(gate.dtype).max)
    return gate.clamp(max=limit), up.clamp(min=-limit, max=limit)


def torch_moe_stage2(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weights,
    topk_ids,
    dtype=dtypes.fp16,
    quant_type=QuantType.No,
    w2_scale=None,  # [1]
    a2_scale=None,  # [expert]]'
    w2_bias=None,
    doweight=True,
    post_gemm2_rms_weight=None,
    post_gemm2_rms_eps=1e-6,
    post_gemm2_rms_zero_centered=False,
):
    ctype = dtypes.fp32  # compute type
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    lazy_w2_int4_dequant = False
    lazy_w2_mxfp4_dequant = False
    if quant_type == QuantType.per_1x32 and w2.dtype == dtypes.i4x2:
        # a16wi4: int4 weights viewed as int8 for compute
        hidden_states = hidden_states.to(ctype)
        w2 = w2.view(dtypes.i8)
        lazy_w2_int4_dequant = True
    elif quant_type == QuantType.per_1x32:
        from aiter.utility import fp4_utils

        if w2.dtype == dtypes.fp8:  # mxfp8 weight
            w2 = w2.to(ctype)
        else:
            lazy_w2_mxfp4_dequant = True
        w2_scale = fp4_utils.e8m0_to_f32(w2_scale)
        if a2_scale is not None:
            if hidden_states.dtype == dtypes.fp8:  # a8w4 mxfp8 activation
                hidden_states = hidden_states.to(ctype)
            else:
                hidden_states = fp4_utils.mxfp4_to_f32(hidden_states)
            a2_scale = fp4_utils.e8m0_to_f32(a2_scale)
        else:  # a16w4 / mxfp8 (bf16 reference activation)
            hidden_states = hidden_states.to(ctype)
    else:
        hidden_states = hidden_states.to(ctype)
        w2 = w2.to(ctype)

    token_num, topk = topk_ids.shape
    hidden_states = hidden_states.view(token_num, topk, inter_dim)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        hidden_states = hidden_states * a2_scale.view(a2_scale.shape[0], -1, 1)
        w2 = w2 * w2_scale.view(w2_scale.shape[0], -1, 1)
    elif quant_type in [QuantType.per_128x128, QuantType.per_1x128]:
        a2_scale = a2_scale.view(hidden_states.shape[0], topk, -1, 1)
        a2_scale = a2_scale.repeat(1, 1, 1, 128).view(hidden_states.shape[0], topk, -1)
        hidden_states = hidden_states * a2_scale

        w2_shape = w2.shape
        w2 = w2.view(
            w2.shape[0], w2.shape[1] // 128, 128, w2.shape[2] // 128, 128
        ) * w2_scale.view(
            w2_scale.shape[0], w2.shape[1] // 128, 1, w2.shape[2] // 128, 1
        )
        w2 = w2.view(w2_shape)
    elif (
        quant_type == QuantType.per_1x32
        and w2_scale is not None
        and w2_scale.dtype == dtypes.bf16
    ):
        # a16wi4: groupwise dequant int4 weights with scale
        # w2: [E, model_dim, inter_dim], w2_scale is [E, inter_dim//32, model_dim]
        group_size = 32
        num_groups = inter_dim // group_size
        if not lazy_w2_int4_dequant:
            w2_shape = w2.shape
            # w2: [E, model_dim, inter_dim] -> apply scale per group of inter_dim
            w2 = w2.reshape(E, model_dim, num_groups, group_size)
            w2.mul_(
                w2_scale.reshape(E, num_groups, model_dim)
                .permute(0, 2, 1)
                .unsqueeze(-1)
            )
            w2 = w2.reshape(w2_shape)
        # activations are bf16, no scaling
    elif quant_type == QuantType.per_1x32:
        a2_shape = hidden_states.shape
        if a2_scale is not None:
            a2_scale = a2_scale[: a2_shape[0] * topk]
            a2_scale = a2_scale.view(token_num, topk, inter_dim // 32, 1)
            hidden_states = (
                hidden_states.view(token_num, topk, inter_dim // 32, 32) * a2_scale
            )
        hidden_states = hidden_states.view(a2_shape)

        if not lazy_w2_mxfp4_dequant:
            w2_shape = w2.shape
            w2 = w2.view(E, model_dim, inter_dim // 32, 32) * w2_scale.view(
                E, model_dim, inter_dim // 32, 1
            )
            w2 = w2.view(w2_shape)

    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if lazy_w2_int4_dequant:
                w2_e = _int4_per_1x32_weight_expert_to_f32(
                    w2, w2_scale, E_id, model_dim, inter_dim
                )
            elif lazy_w2_mxfp4_dequant:
                w2_e = _mxfp4_per_1x32_weight_expert_to_f32(
                    w2, w2_scale, E_id, model_dim, inter_dim
                )
            else:
                w2_e = w2[E_id]
            act_input = sub_tokens @ (w2_e.transpose(0, 1))
            out[mask] = act_input
            if w2_bias is not None:
                out[mask] = out[mask] + w2_bias[E_id].view(1, -1)
    if post_gemm2_rms_weight is not None:
        rms_weight = post_gemm2_rms_weight.view(1, 1, model_dim).to(ctype)
        if post_gemm2_rms_zero_centered:
            rms_weight = rms_weight + 1.0
        # The scaled input is rounded to bf16 once before the RMS.
        out = (out * rms_weight).to(dtypes.bf16).to(ctype)
        out = out * torch.rsqrt(
            out.square().mean(dim=-1, keepdim=True) + float(post_gemm2_rms_eps)
        )
    if doweight:
        # In-place: out and topk_weights are both fp32 (ctype), so this is
        # numerically identical to `out = out * ...` but avoids allocating a
        # second full (token_num, topk, model_dim) fp32 tensor. That transient
        # is the largest single allocation in the stage-2 reference (tens of GiB
        # for large-token FP4 shapes) and was the site of the CI OOM.
        out.mul_(topk_weights.view(token_num, -1, 1))
    return out.sum(1).to(dtype)
