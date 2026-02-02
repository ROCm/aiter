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
