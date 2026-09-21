# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SonicMoE: Pure-Triton grouped GEMM MoE with full autograd support.
# Ported from sonic-moe/sonicmoe/functional_rocm/.

import torch
import torch.nn.functional as F

from aiter.ops.triton._triton_kernels.moe.sonicmoe.activation_kernels import (
    activation_fwd,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.backward import (
    _down_projection_backward_act,
    _token_broadcast_backward,
    _up_projection_backward_act,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.enums import ActivationType, is_glu
from aiter.ops.triton._triton_kernels.moe.sonicmoe.forward import (
    _router_forward,
    _topk_softmax_bwd,
    _topk_softmax_fwd,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
    _local_tensor,
    grouped_gemm,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.routing import (
    TC_topk_router_metadata_triton,
    general_routing_router_metadata_triton,
)

SonicMoEActivationType = ActivationType
sonicmoe_is_glu = is_glu

__all__ = [
    "SonicMoEActivationType",
    "moe_TC_softmax_topk_layer",
    "moe_general_routing_inputs",
    "moe_pre_routed_inputs",
    "sonicmoe_is_glu",
]


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        router_logits: torch.Tensor,
        E: int,
        K: int,
        is_softmax_over_topk: bool,
        norm_topk_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        topk_router_score = torch.empty(
            T, K, dtype=torch.float32, device=router_logits.device
        )
        topk_router_indices = torch.empty(
            T, K, dtype=torch.int32, device=router_logits.device
        )

        _topk_softmax_fwd(
            router_logits,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=is_softmax_over_topk,
            norm_topk_probs=norm_topk_probs,
        )

        ctx.save_for_backward(topk_router_score, topk_router_indices, router_logits)
        ctx.E = E
        ctx.dtype = router_logits.dtype
        ctx.is_softmax_over_topk = is_softmax_over_topk
        ctx.norm_topk_probs = norm_topk_probs

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(ctx, dtopk_score: torch.Tensor, _: torch.Tensor):
        T, K = dtopk_score.size()
        E = ctx.E
        topk_router_score, topk_router_indices, router_logits = ctx.saved_tensors
        dlogits = torch.zeros(
            T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device
        )

        _topk_softmax_bwd(
            router_logits,
            dlogits,
            None,
            dtopk_score,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=ctx.is_softmax_over_topk,
            norm_topk_probs=ctx.norm_topk_probs,
        )

        return dlogits, None, None, None, None


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_frequency_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_each_token_has_variable_activated_experts: bool,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool = False,
        grouped_weight_layout: bool = False,
        inputs_are_pre_routed: bool = False,
    ) -> torch.Tensor:
        T, H = x.shape
        E = expert_frequency_offset.numel() - 1
        if grouped_weight_layout:
            E_w, H_w, I_full = w1.shape
            if E_w != E or H_w != H:
                raise ValueError(
                    f"Grouped w1 must be [E={E}, H={H}, I], got {tuple(w1.shape)}"
                )
            gemm_w1 = w1
        else:
            I_full, H_w, E_w = w1.shape
            if E_w != E or H_w != H:
                raise ValueError(
                    f"Legacy w1 must be [I, H={H}, E={E}], got {tuple(w1.shape)}"
                )
            gemm_w1 = w1.permute(2, 1, 0)
        is_glu_activation = is_glu(activation_type)
        I = I_full // 2 if is_glu_activation else I_full
        TK = total_expert_freq

        h = torch.empty(TK, I_full, dtype=x.dtype, device=x.device)
        grouped_gemm(
            x,
            gemm_w1,  # (E, H, I_full)
            expert_frequency_offset,
            out=h,
            bias=b1,
            A_idx=None if inputs_are_pre_routed else x_gather_idx,
        )

        a = activation_fwd(h, I, activation_type.value, concat_layout)

        h_save = h if not is_inference_mode_enabled else None

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.is_each_token_has_variable_activated_experts = (
            is_each_token_has_variable_activated_experts
        )
        ctx.is_glu_activation = is_glu_activation
        ctx.concat_layout = concat_layout and is_glu_activation
        ctx.grouped_weight_layout = grouped_weight_layout
        ctx.inputs_are_pre_routed = inputs_are_pre_routed

        ctx.save_for_backward(
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ctx.mark_non_differentiable(a)
        ctx.set_materialize_grads(False)

        return a, h_save

    @staticmethod
    def backward(ctx, _: None, dh: torch.Tensor):
        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_glu_activation = ctx.is_glu_activation
        is_each_token_has_variable_activated_experts = (
            ctx.is_each_token_has_variable_activated_experts
        )
        concat_layout = ctx.concat_layout

        (
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            _s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = ctx.saved_tensors

        dx_expanded = torch.empty(TK, H, dtype=dh.dtype, device=dh.device)
        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)

        _up_projection_backward_act(
            w1=_local_tensor(w1),
            dx_expanded=dx_expanded,
            dh=dh,
            db1=_local_tensor(db1),
            expert_frequency_offset=expert_frequency_offset,
            is_glu_activation=is_glu_activation,
            concat_layout=concat_layout,
            grouped_weight_layout=ctx.grouped_weight_layout,
        )

        grouped_gemm(
            x,
            dh,
            expert_frequency_offset,
            out=dw1 if ctx.grouped_weight_layout else dw1.permute(2, 1, 0),
            A_idx=None if ctx.inputs_are_pre_routed else x_gather_idx,
            A_is_transposed=True,
        )

        if ctx.inputs_are_pre_routed:
            dx_reduced = dx_expanded
        else:
            dx_reduced = torch.empty(T, H, dtype=dh.dtype, device=dh.device)
            _token_broadcast_backward(
                dx_reduced=dx_reduced,
                dx_expanded=dx_expanded,
                s_reverse_scatter_idx=s_reverse_scatter_idx,
                num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
                varlen_K_max=(E if is_each_token_has_variable_activated_experts else K),
                H=H,
                is_varlen_K=is_each_token_has_variable_activated_experts,
            )

        return dx_reduced, dw1, db1, *[None] * 14


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        h: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        T: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        grouped_weight_layout: bool,
        concat_layout: bool,
    ) -> torch.Tensor:
        TK = a.size(0)
        E = expert_frequency_offset.numel() - 1
        if grouped_weight_layout:
            E_w, I, H = w2.shape
            if E_w != E or I != a.size(1):
                raise ValueError(
                    f"Grouped w2 must be [E={E}, I={a.size(1)}, H], "
                    f"got {tuple(w2.shape)}"
                )
            gemm_w2 = w2
        else:
            H, I, E_w = w2.shape
            if E_w != E or I != a.size(1):
                raise ValueError(
                    f"Legacy w2 must be [H, I={a.size(1)}, E={E}], "
                    f"got {tuple(w2.shape)}"
                )
            gemm_w2 = w2.permute(2, 1, 0)

        y = torch.empty(TK, H, dtype=a.dtype, device=a.device)
        grouped_gemm(a, gemm_w2, expert_frequency_offset, out=y, bias=b2)

        o = torch.empty(T, H, device=a.device, dtype=a.dtype)
        topk_scores_flat = topk_scores.view(-1)

        _router_forward(
            y=y,
            o=o,
            topk_scores=topk_scores_flat,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.activation_type = activation_type
        ctx.grouped_weight_layout = grouped_weight_layout
        ctx.concat_layout = concat_layout

        ctx.save_for_backward(
            h,
            w2,
            b2,
            topk_scores_flat,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        )

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type

        (
            h,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        ) = ctx.saved_tensors

        dw2 = torch.empty_like(w2)
        db2 = None if b2 is None else torch.empty_like(b2)
        dh = torch.empty_like(h)

        I = w2.size(1)
        TK = x_gather_idx.size(0)

        a_prime = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        ds = torch.empty_like(topk_scores)

        _down_projection_backward_act(
            dout=dout,
            h=h,
            w2=_local_tensor(w2),
            dh=dh,
            ds=ds,
            b2=_local_tensor(b2),
            db2=_local_tensor(db2),
            a_prime=a_prime,
            topk_scores=topk_scores,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=x_gather_idx,
            s_scatter_idx=s_scatter_idx,
            activation_type=activation_type.value,
            grouped_weight_layout=ctx.grouped_weight_layout,
            concat_layout=ctx.concat_layout,
        )

        s = topk_scores[s_scatter_idx]
        dout_gathered = dout[x_gather_idx]
        dy = dout_gathered * s.unsqueeze(-1)

        grouped_gemm(
            a_prime,
            dy,
            expert_frequency_offset,
            out=dw2 if ctx.grouped_weight_layout else dw2.permute(2, 1, 0),
            A_is_transposed=True,
        )

        if not is_varlen_K:
            ds = ds.view(T, K)

        return None, dh, dw2, db2, ds, *[None] * 11


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    _stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
    concat_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or ((b1 is not None) and (b2 is not None))
    E = router_w.size(0)
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
        router_logits, E, K, is_softmax_over_topk, norm_topk_probs
    )

    T, K = topk_indices.size()
    TK = T * K
    device = topk_indices.device

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )

    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        False,
        False,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        topk_scores,
        expert_frequency_offset,
        T,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,
        activation_type,
        False,
        concat_layout,
    )

    return o, router_logits, expert_frequency


def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    _stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
    grouped_weight_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or ((b1 is not None) and (b2 is not None))

    T = x.size(0)
    TK = router_scores.size(0)
    device = router_scores.device

    if router_scores.dtype != torch.float32:
        router_scores = router_scores.float()

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    num_activated_expert_per_token_offset = torch.empty(
        T + 1, dtype=torch.int32, device=device
    )

    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        T,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        None,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        grouped_weight_layout,
        False,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        router_scores,
        expert_frequency_offset,
        T,
        None,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,
        activation_type,
        grouped_weight_layout,
        concat_layout,
    )

    return o, expert_frequency


def moe_pre_routed_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    expert_frequency: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    _stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
    grouped_weight_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SonicMoE on expert-major tokens from an all-to-all dispatcher."""
    T = x.size(0)
    if router_scores.numel() != T:
        raise ValueError(
            f"Expected one router score per pre-routed token ({T}), "
            f"got {router_scores.numel()}"
        )
    if router_scores.dtype != torch.float32:
        router_scores = router_scores.float()

    expert_frequency = expert_frequency.to(device=x.device, dtype=torch.int32)
    expert_frequency_offset = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=x.device),
            expert_frequency.cumsum(dim=0, dtype=torch.int32),
        )
    )
    identity = torch.arange(T, dtype=torch.int32, device=x.device)

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        T,
        1,
        identity,
        identity,
        identity,
        None,
        False,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        grouped_weight_layout,
        True,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        router_scores.reshape(T, 1),
        expert_frequency_offset,
        T,
        1,
        identity,
        identity,
        identity,
        None,
        False,
        activation_type,
        grouped_weight_layout,
        concat_layout,
    )
    return o, expert_frequency
