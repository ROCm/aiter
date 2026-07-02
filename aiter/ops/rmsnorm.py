# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
from torch import Tensor
from ..jit.core import compile_ops
from typing import Optional

from .rmsnorm_opus import fused_add_rms_norm_opus as _fused_add_rms_norm_opus
from .rmsnorm_opus import rms_norm_opus as _rms_norm_opus

MD_NAME = "module_rmsnorm"


def get_rmsnorm_backend() -> str:
    """rmsnorm backend from AITER_RMSNORM_BACKEND: 'ck' (default) or 'opus' (#4055)."""
    return os.environ.get("AITER_RMSNORM_BACKEND", "ck").strip().lower()


def _use_opus(
    input, use_model_sensitive_rmsnorm: int = 0, gemma_norm: bool = False
) -> bool:
    """True when opus can serve this call (plain bf16/fp16, non-quant/T5/gemma)."""
    return (
        get_rmsnorm_backend() == "opus"
        and use_model_sensitive_rmsnorm == 0
        and not gemma_norm
        and input.dtype in (torch.float16, torch.bfloat16)
    )


@compile_ops("module_rmsnorm", fc_name="rms_norm_cu")
def rms_norm_cu_ck(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """
    Cuda version of rmsnorm (CK / module_rmsnorm)
    """
    ...


def rms_norm_cu(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """out = rmsnorm(input) * weight. Uses opus when AITER_RMSNORM_BACKEND=opus."""
    if _use_opus(input):
        _rms_norm_opus(out, input, weight, epsilon)
    else:
        rms_norm_cu_ck(out, input, weight, epsilon)


@compile_ops("module_rmsnorm", fc_name="fused_add_rms_norm_cu")
def fused_add_rms_norm_cu_ck(
    input: Tensor,  # input/out
    residual_in: Tensor,  # residual_in/out
    weight: Tensor,
    epsilon: float,
) -> None:
    """
    Cuda version of rmsnorm fused add (CK / module_rmsnorm)
    """
    ...


def fused_add_rms_norm_cu(
    input: Tensor,  # input/out
    residual_in: Tensor,  # residual_in/out
    weight: Tensor,
    epsilon: float,
) -> None:
    """In-place fused add + rmsnorm. Uses opus when AITER_RMSNORM_BACKEND=opus."""
    if _use_opus(input):
        _fused_add_rms_norm_opus(input, residual_in, weight, epsilon)
    else:
        fused_add_rms_norm_cu_ck(input, residual_in, weight, epsilon)


def gen_rms_norm_fake_tensor(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    return torch.empty_like(input, dtype=input.dtype, device=input.device)


def rms_norm(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    """
    rmsnorm; CK by default, opus when AITER_RMSNORM_BACKEND=opus (plain bf16/fp16).
    """
    if _use_opus(input, use_model_sensitive_rmsnorm):
        out = torch.empty_like(input)
        _rms_norm_opus(out, input, weight, epsilon)
        return out
    return rmsnorm2d_fwd_ck(input, weight, epsilon, use_model_sensitive_rmsnorm)


def rmsnorm2d_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    if _use_opus(input, use_model_sensitive_rmsnorm):
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
        _rms_norm_opus(out, input, weight, epsilon)
        return out
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        out = rmsnorm2d_fwd_ck(input, weight, epsilon, use_model_sensitive_rmsnorm)
    else:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
        rmsnorm(out, input, weight, epsilon)
    return out


def rmsnorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    gemma_norm: bool = False,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    if _use_opus(input, use_model_sensitive_rmsnorm, gemma_norm):
        # opus kernel is in-place; stage into out/residual_out, leave inputs intact
        out.copy_(input)
        residual_out.copy_(residual_in)
        _fused_add_rms_norm_opus(out, residual_out, weight, epsilon)
        return
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        rmsnorm2d_fwd_with_add_ck(
            out,
            input,
            residual_in,
            residual_out,
            weight,
            epsilon,
            use_model_sensitive_rmsnorm,
        )
    else:
        add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon, gemma_norm)


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


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
    out_before_quant: Optional[Tensor] = None,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


def rmsnorm2d_fwd_with_dynamicquant(
    out: Tensor,
    input: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        assert group_size == 0, "group_size is not supported for ck rmsnorm"
        assert not shuffle_scale, "shuffle_scale is not supported for ck rmsnorm"
        rmsnorm2d_fwd_with_dynamicquant_ck(
            out, input, yscale, weight, epsilon, use_model_sensitive_rmsnorm
        )
    else:
        rmsnorm_quant(out, input, yscale, weight, epsilon, group_size, shuffle_scale)


def rmsnorm2d_fwd_with_add_dynamicquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        assert group_size == 0, "group_size is not supported for ck rmsnorm"
        assert not shuffle_scale, "shuffle_scale is not supported for ck rmsnorm"
        rmsnorm2d_fwd_with_add_dynamicquant_ck(
            out,
            input,
            residual_in,
            residual_out,
            yscale,
            weight,
            epsilon,
            use_model_sensitive_rmsnorm,
        )
    else:
        add_rmsnorm_quant(
            out,
            input,
            residual_in,
            residual_out,
            yscale,
            weight,
            epsilon,
            group_size,
            shuffle_scale,
        )


@compile_ops(
    "module_rmsnorm", gen_fake=gen_rms_norm_fake_tensor, fc_name="rmsnorm2d_fwd"
)
def rmsnorm2d_fwd_ck(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor: ...


@compile_ops("module_rmsnorm", fc_name="rmsnorm2d_fwd_with_add")
def rmsnorm2d_fwd_with_add_ck(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


@compile_ops("module_rmsnorm", fc_name="rmsnorm2d_fwd_with_dynamicquant")
def rmsnorm2d_fwd_with_dynamicquant_ck(
    out: Tensor,
    input: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


@compile_ops("module_rmsnorm", fc_name="rmsnorm2d_fwd_with_add_dynamicquant")
def rmsnorm2d_fwd_with_add_dynamicquant_ck(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


@compile_ops("module_rmsnorm_quant")
def add_rmsnorm_quant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant")
def add_rmsnorm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant")
def rmsnorm_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant")
def rmsnorm(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    gemma_norm: bool = False,
) -> None: ...
