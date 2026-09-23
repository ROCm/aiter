# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.sonicmoe.backward import (
    db1_kernel,
    db2_and_ds_kernel,
)
from aiter.ops.triton.moe.sonicmoe_activations import activation_bwd, activation_fwd
from aiter.ops.triton.moe.sonicmoe_grouped_gemm import grouped_gemm
from aiter.ops.triton.moe.sonicmoe_token_gather import (
    token_gather_and_sum_varlen_K_triton,
)
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_sonicmoe_kernel_config,
    split_launch_config,
)

LIBRARY_NAME = "aiter_sonicmoe"


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_up_projection_backward_act_rocm",
    mutates_args={"dx_expanded", "db1"},
)
def _up_projection_backward_act(
    w1: torch.Tensor,
    dx_expanded: torch.Tensor,
    dh: torch.Tensor,
    db1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    is_glu_activation: bool,
    concat_layout: bool = False,
    grouped_weight_layout: bool = False,
) -> None:
    if grouped_weight_layout:
        E, _, I_full = w1.size()
        gemm_w1 = w1
    else:
        I_full, _, E = w1.size()
        gemm_w1 = w1.permute(2, 0, 1)
    I = I_full // 2 if is_glu_activation else I_full

    grouped_gemm(
        dh,
        gemm_w1,
        expert_frequency_offset,
        out=dx_expanded,
        B_is_transposed=grouped_weight_layout,
    )

    if db1 is not None:
        db1_cfg = get_sonicmoe_kernel_config("db1_kernel")
        constexprs, launch = split_launch_config(db1_cfg)
        db1_kernel[(E,)](
            dh,
            db1,
            expert_frequency_offset,
            (2 * I if is_glu_activation else I),
            E,
            CONCAT_LAYOUT=concat_layout and is_glu_activation,
            **constexprs,
            **launch,
        )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_down_projection_backward_act_rocm",
    mutates_args={"dh", "ds", "db2", "a_prime"},
)
def _down_projection_backward_act(
    dout: torch.Tensor,
    h: torch.Tensor,
    w2: torch.Tensor,
    dh: torch.Tensor,
    ds: torch.Tensor,
    b2: torch.Tensor | None,
    db2: torch.Tensor | None,
    a_prime: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    activation_type: str,
    grouped_weight_layout: bool = False,
    concat_layout: bool = False,
) -> None:
    if grouped_weight_layout:
        E, I, H = w2.size()
        gemm_w2 = w2
    else:
        H, I, E = w2.size()
        gemm_w2 = w2.permute(2, 0, 1)
    TK = x_gather_idx.size(0)
    s = topk_scores[s_scatter_idx]

    # Compute u = dout @ w2.T once. The router gradient reuses this GEMM:
    # dot(dout, a @ w2) == dot(a, dout @ w2.T), while da = score * u.
    dout_gathered = dout[x_gather_idx]
    dh_unscaled = torch.empty(TK, I, dtype=dh.dtype, device=dh.device)
    grouped_gemm(
        dout_gathered,
        gemm_w2,
        expert_frequency_offset,
        out=dh_unscaled,
        B_is_transposed=grouped_weight_layout,
    )

    a_prime_val = activation_fwd(h, I, activation_type, concat_layout)
    a_prime.copy_(a_prime_val)
    ds_scattered = (a_prime_val.float() * dh_unscaled.float()).sum(dim=-1)

    dh_raw = dh_unscaled * s.unsqueeze(-1)
    dh_act = activation_bwd(h, dh_raw, I, activation_type, concat_layout)
    dh.copy_(dh_act)

    if db2 is None:
        ds[s_scatter_idx] = ds_scattered
    else:
        old_ds_partial = torch.empty(
            TK, 1, device=ds_scattered.device, dtype=ds_scattered.dtype
        )
        old_ds_partial[s_scatter_idx, 0] = ds_scattered

        db2_cfg = get_sonicmoe_kernel_config("db2_and_ds_kernel")
        block_h_max = db2_cfg.pop("BLOCK_H_MAX")
        BLOCK_H = min(triton.next_power_of_2(H), block_h_max)
        NUM_H_BLOCKS = triton.cdiv(H, BLOCK_H)
        new_ds_partial = torch.empty(
            TK, NUM_H_BLOCKS, dtype=torch.float32, device=ds.device
        )

        constexprs, launch = split_launch_config(db2_cfg)
        db2_and_ds_kernel[(E, NUM_H_BLOCKS)](
            dout,
            topk_scores,
            new_ds_partial,
            old_ds_partial,
            b2,
            db2,
            x_gather_idx,
            s_scatter_idx,
            expert_frequency_offset,
            H,
            E,
            1,
            BLOCK_H=BLOCK_H,
            **constexprs,
            **launch,
        )

        if NUM_H_BLOCKS == 1:
            ds.copy_(new_ds_partial.view(-1).to(dtype=ds.dtype))
        else:
            ds.copy_(new_ds_partial.sum(dim=-1, dtype=ds.dtype))


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_token_broadcast_backward_rocm", mutates_args={"dx_reduced"}
)
def _token_broadcast_backward(
    dx_reduced: torch.Tensor,
    dx_expanded: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor | None,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    if num_activated_expert_per_token_offset is None:
        assert not is_varlen_K
    token_gather_and_sum_varlen_K_triton(
        dx_expanded,
        None,
        dx_reduced,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        dx_reduced.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )
