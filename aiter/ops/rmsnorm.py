# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_rmsnorm"


@compile_ops("module_rmsnorm")
def rms_norm_cu(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
):
    """
    Cuda version of rmsnorm
    """
    ...


@compile_ops("module_rmsnorm")
def fused_add_rms_norm_cu(
    input: Tensor,  # input/out
    residual_in: Tensor,  # residual_in/out
    weight: Tensor,
    epsilon: float,
):
    """
    Cuda version of rmsnorm fused add
    """
    ...


@compile_ops("module_rmsnorm", fc_name="rmsnorm2d_fwd")
def rms_norm(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int,
):
    """
    CK version of rmsnorm
    """
    ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd(
    input: torch.Tensor, weight: torch.Tensor, epsilon: float, use_model_sensitive_rmsnorm: int,
) -> torch.Tensor: ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int,
): ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int,
): ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_add_smoothquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int,
): ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_dynamicquant(
    out: Tensor, input: Tensor, yscale: Tensor, weight: Tensor, epsilon: float, use_model_sensitive_rmsnorm: int,
): ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_add_dynamicquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int,
): ...
