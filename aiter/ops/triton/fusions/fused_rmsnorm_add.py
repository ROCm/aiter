# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_rmsnorm_add import (
    _gluon_fused_rms_kernel,
    _gluon_fused_rms_tdm_kernel,
)
from aiter.ops.triton._triton_kernels.fusions.fused_rmsnorm_add import (
    _triton_fused_rms_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)

_CONFIG_NAME = "FUSED_RMSNORM_ADD"


def _get_config(M: int, N: int) -> dict:
    """The gfx1250 config for M rows of N columns: BLOCK_M rows a program,
    WARPS_N waves a row, and USE_TDM for the kernel that copies the rows
    through LDS with TDM instead of loading them into registers."""
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, axes=("N", "M"), N=N, M=M)


def _fused_rmsnorm_add_core(x, weight, epsilon, res1, gemma_norm):
    """The launcher of the two custom ops below: returns (out, out_res1), with
    out_res1 None without res1. Each op has one return type, because a custom op
    cannot return a Tensor or a tuple."""
    assert x.dim() == 2, "fused_rmsnorm_add expects a 2D tensor"
    M, N = x.shape
    assert (
        weight.shape == (N,) and weight.dtype == x.dtype
    ), f"weight must be ({N},) {x.dtype}, got {tuple(weight.shape)} {weight.dtype}"
    x, weight = x.contiguous(), weight.contiguous()
    out1 = torch.empty((M, N), dtype=x.dtype, device=x.device)
    out_res1 = None
    strides = (x.stride(0), 0, out1.stride(0), 0)
    if res1 is not None:
        assert (
            res1.shape == x.shape and res1.dtype == x.dtype
        ), f"res1 must be {tuple(x.shape)} {x.dtype}, got {tuple(res1.shape)} {res1.dtype}"
        res1 = res1.contiguous()
        out_res1 = torch.empty_like(out1)
        strides = (x.stride(0), res1.stride(0), out1.stride(0), out_res1.stride(0))
    BLOCK_N = max(triton.next_power_of_2(N), 32)

    if get_arch() == "gfx1250":
        config = _get_config(M, N)
        kernel = (
            _gluon_fused_rms_tdm_kernel
            if config["USE_TDM"]
            else _gluon_fused_rms_kernel
        )
        kernel[(triton.cdiv(M, config["BLOCK_M"]),)](
            x,
            res1,
            weight,
            out1,
            out_res1,
            M,
            *strides,
            epsilon,
            N=N,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=BLOCK_N,
            WARPS_N=config["WARPS_N"],
            HAS_RES=res1 is not None,
            GEMMA=gemma_norm,
            num_warps=config["BLOCK_M"] * config["WARPS_N"],
        )
    else:
        _triton_fused_rms_kernel[(M,)](
            x,
            weight,
            res1,
            out1,
            out_res1,
            epsilon,
            M,
            N,
            *strides,
            BLOCK_SIZE_N=BLOCK_N,
            FIRST_INPUT_RES=res1 is not None,
            GEMMA=gemma_norm,
        )
    return out1, out_res1


def _fused_rmsnorm_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, gemma_norm: bool
) -> torch.Tensor:
    return torch.empty_like(x)


@torch_compile_guard(gen_fake=_fused_rmsnorm_fake)
def _fused_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, gemma_norm: bool
) -> torch.Tensor:
    out1, _ = _fused_rmsnorm_add_core(x, weight, epsilon, None, gemma_norm)
    return out1


def _fused_rmsnorm_add_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res1: torch.Tensor,
    gemma_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@torch_compile_guard(gen_fake=_fused_rmsnorm_add_fake)
def _fused_rmsnorm_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res1: torch.Tensor,
    gemma_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out1, out_res1 = _fused_rmsnorm_add_core(x, weight, epsilon, res1, gemma_norm)
    return out1, out_res1


def fused_rmsnorm_add(x, weight, epsilon, res1=None, gemma_norm=False):
    """RMSNorm of each row of x (M, N), with an optional residual add.

    Without res1: returns out = x / rms(x) * weight.
    With res1 (M, N): returns (out, out_res1), out_res1 = x + res1 and
    out = out_res1 / rms(out_res1) * weight.
    gemma_norm: scale by 1 + weight (added in fp32) instead of weight.
    weight (N,) and res1 have the dtype of x (bf16 or fp16); the sum, the norm and
    the scale are in fp32. On gfx1250 a Gluon kernel runs, with the config of N
    and M; on other GPUs a Triton kernel.
    """
    if res1 is None:
        return _fused_rmsnorm(x, weight, epsilon, gemma_norm)
    return _fused_rmsnorm_add(x, weight, epsilon, res1, gemma_norm)
