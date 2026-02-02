# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_mhc_layer"


@compile_ops("module_mhc_layer")
def mhc_layer_fwd(
    out: Tensor,
    x_expanded: Tensor,
    rmsnorm_weight: Tensor,
    phi_pre: Tensor,
    phi_post: Tensor,
    phi_res: Tensor,
    b_pre: Tensor,
    b_post: Tensor,
    b_res: Tensor,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    sinkhorn_iters: int,
    eps: float,
    use_pdl: bool = True,
) -> None:
    """
    MHC layer forward (CUDA/HIP).
    """
    ...


@compile_ops("module_mhc_layer", fc_name="mhc_layer_fwd_debug")
def mhc_layer_fwd_debug(
    out: Tensor,
    x_expanded: Tensor,
    rmsnorm_weight: Tensor,
    phi_pre: Tensor,
    phi_post: Tensor,
    phi_res: Tensor,
    b_pre: Tensor,
    b_post: Tensor,
    b_res: Tensor,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    sinkhorn_iters: int,
    eps: float,
    H_proj_raw: Tensor,
    H_pre: Tensor,
    H_post: Tensor,
    M: Tensor,
    x_agg_bf16: Tensor,
    layer_out_bf16: Tensor,
    rms_values: Tensor,
    use_pdl: bool = True,
) -> None:
    """
    MHC layer forward with debug intermediates (CUDA/HIP).
    """
    ...


def mhc_layer(
    x_expanded: Tensor,
    rmsnorm_weight: Tensor,
    phi_pre: Tensor,
    phi_post: Tensor,
    phi_res: Tensor,
    b_pre: Tensor,
    b_post: Tensor,
    b_res: Tensor,
    alpha_pre: float = 0.01,
    alpha_post: float = 0.01,
    alpha_res: float = 0.01,
    sinkhorn_iters: int = 20,
    eps: float = 1e-5,
    use_pdl: bool = True,
) -> Tensor:
    out = torch.empty(x_expanded.shape, device=x_expanded.device, dtype=torch.float32)
    mhc_layer_fwd(
        out,
        x_expanded,
        rmsnorm_weight,
        phi_pre,
        phi_post,
        phi_res,
        b_pre,
        b_post,
        b_res,
        alpha_pre,
        alpha_post,
        alpha_res,
        sinkhorn_iters,
        eps,
        use_pdl,
    )
    return out


def mhc_layer_debug(
    x_expanded: Tensor,
    rmsnorm_weight: Tensor,
    phi_pre: Tensor,
    phi_post: Tensor,
    phi_res: Tensor,
    b_pre: Tensor,
    b_post: Tensor,
    b_res: Tensor,
    alpha_pre: float = 0.01,
    alpha_post: float = 0.01,
    alpha_res: float = 0.01,
    sinkhorn_iters: int = 20,
    eps: float = 1e-5,
    use_pdl: bool = True,
):
    B, n, C = x_expanded.shape
    total_H_dim = n + n + n * n
    out = torch.empty_like(x_expanded, dtype=torch.float32, device=x_expanded.device)
    H_proj_raw = torch.empty((B, total_H_dim), device=x_expanded.device, dtype=torch.float32)
    H_pre = torch.empty((B, n), device=x_expanded.device, dtype=torch.float32)
    H_post = torch.empty((B, n), device=x_expanded.device, dtype=torch.float32)
    M = torch.empty((B, n, n), device=x_expanded.device, dtype=torch.float32)
    x_agg_bf16 = torch.empty((B, C), device=x_expanded.device, dtype=torch.bfloat16)
    layer_out_bf16 = torch.empty((B, C), device=x_expanded.device, dtype=torch.bfloat16)
    rms_values = torch.empty((B,), device=x_expanded.device, dtype=torch.float32)

    mhc_layer_fwd_debug(
        out,
        x_expanded,
        rmsnorm_weight,
        phi_pre,
        phi_post,
        phi_res,
        b_pre,
        b_post,
        b_res,
        alpha_pre,
        alpha_post,
        alpha_res,
        sinkhorn_iters,
        eps,
        H_proj_raw,
        H_pre,
        H_post,
        M,
        x_agg_bf16,
        layer_out_bf16,
        rms_values,
        use_pdl,
    )

    return {
        "out": out,
        "H_proj_raw": H_proj_raw,
        "H_pre": H_pre,
        "H_post": H_post,
        "M": M,
        "x_agg_bf16": x_agg_bf16,
        "layer_out_bf16": layer_out_bf16,
        "rms_values": rms_values,
    }
