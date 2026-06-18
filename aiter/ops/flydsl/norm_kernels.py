# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch-facing FlyDSL RMSNorm and LayerNorm APIs."""

from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor

from flydsl.expr.typing import Stream

from .kernels.layernorm_kernel import (
    build_fused_add_layernorm_dynamicquant_module,
    build_fused_add_layernorm_module,
    build_fused_add_layernorm_smoothquant_module,
    build_layernorm_dynamicquant_module,
    build_layernorm_module,
    build_layernorm_smoothquant_module,
)
from .kernels.rmsnorm_kernel import (
    build_fused_add_rmsnorm_dynamicquant_module,
    build_fused_add_rmsnorm_module,
    build_fused_add_rmsnorm_smoothquant_module,
    build_rmsnorm_dynamicquant_module,
    build_rmsnorm_module,
    build_rmsnorm_smoothquant_module,
)

__all__ = [
    "flydsl_rmsnorm",
    "flydsl_add_rmsnorm",
    "flydsl_rmsnorm_dynamicquant",
    "flydsl_rmsnorm_smoothquant",
    "flydsl_add_rmsnorm_dynamicquant",
    "flydsl_add_rmsnorm_smoothquant",
    "flydsl_layernorm",
    "flydsl_add_layernorm",
    "flydsl_layernorm_dynamicquant",
    "flydsl_layernorm_smoothquant",
    "flydsl_add_layernorm_dynamicquant",
    "flydsl_add_layernorm_smoothquant",
]

DEFAULT_EPS = 1e-5

_DTYPE_TO_KERNEL = {
    torch.float32: "f32",
    torch.float: "f32",
    torch.float16: "f16",
    torch.half: "f16",
    torch.bfloat16: "bf16",
}


def _kernel_dtype(dtype: torch.dtype) -> str:
    dtype_str = _DTYPE_TO_KERNEL.get(dtype)
    if dtype_str is None:
        raise TypeError(
            f"FlyDSL norm only supports fp32/fp16/bf16 inputs, got {dtype!r}"
        )
    return dtype_str


def _check_cuda_tensor(name: str, tensor: Tensor, device: torch.device) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _check_contiguous(name: str, tensor: Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")


def _check_vector(name: str, tensor: Tensor, n: int, dtype: torch.dtype, device: torch.device) -> None:
    _check_cuda_tensor(name, tensor, device)
    if tensor.shape != (n,):
        raise ValueError(f"{name} shape must be ({n},), got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
    _check_contiguous(name, tensor)


def _check_2d_like(
    name: str,
    tensor: Tensor,
    shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    _check_cuda_tensor(name, tensor, device)
    if tensor.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
    _check_contiguous(name, tensor)


def _check_norm_input(input: Tensor, weight: Tensor) -> Tuple[int, int, str, torch.device]:
    if input.dim() != 2:
        raise ValueError(f"input must be 2D [M, N], got shape={tuple(input.shape)}")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA/ROCm tensor")
    _check_contiguous("input", input)
    m, n = input.shape
    dtype_str = _kernel_dtype(input.dtype)
    _check_vector("weight", weight, n, input.dtype, input.device)
    return m, n, dtype_str, input.device


def _check_optional_out(
    name: str,
    out: Optional[Tensor],
    like: Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    out_dtype = like.dtype if dtype is None else dtype
    if out is None:
        return torch.empty_like(like, dtype=out_dtype, device=like.device)
    _check_2d_like(name, out, tuple(like.shape), out_dtype, like.device)
    return out


def _check_optional_yscale(yscale: Optional[Tensor], m: int, device: torch.device) -> Tensor:
    if yscale is None:
        return torch.empty((m,), dtype=torch.float32, device=device)
    _check_cuda_tensor("yscale", yscale, device)
    if yscale.shape != (m,):
        raise ValueError(f"yscale shape must be ({m},), got {tuple(yscale.shape)}")
    if yscale.dtype != torch.float32:
        raise TypeError(f"yscale dtype must be torch.float32, got {yscale.dtype}")
    _check_contiguous("yscale", yscale)
    return yscale


def _normalize_quant_dtype(quant_dtype: str) -> str:
    if quant_dtype in ("i8", "int8"):
        return quant_dtype
    raise ValueError(f"unsupported quant dtype: {quant_dtype!r} (expected 'i8')")


def _normalize_stream(
    device: torch.device,
    stream: Optional[torch.cuda.Stream],
) -> torch.cuda.Stream:
    launch_stream = torch.cuda.current_stream(device=device) if stream is None else stream
    if launch_stream.device != device:
        raise ValueError(f"stream must be on {device}, got {launch_stream.device}")
    return launch_stream


def _run_launcher(device: torch.device, launcher, *args, stream=None) -> None:
    launch_stream = _normalize_stream(device, stream)
    with torch.cuda.device(device):
        launcher(*args, stream=Stream(launch_stream))


@lru_cache(maxsize=64)
def _compile_rmsnorm(m: int, n: int, dtype_str: str, eps: float):
    return build_rmsnorm_module(m, n, dtype_str, eps=eps)


@lru_cache(maxsize=64)
def _compile_add_rmsnorm(m: int, n: int, dtype_str: str, eps: float):
    return build_fused_add_rmsnorm_module(m, n, dtype_str, eps=eps)


@lru_cache(maxsize=64)
def _compile_rmsnorm_dynamicquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_rmsnorm_dynamicquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


@lru_cache(maxsize=64)
def _compile_rmsnorm_smoothquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_rmsnorm_smoothquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


@lru_cache(maxsize=64)
def _compile_add_rmsnorm_dynamicquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_fused_add_rmsnorm_dynamicquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


@lru_cache(maxsize=64)
def _compile_add_rmsnorm_smoothquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_fused_add_rmsnorm_smoothquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


@lru_cache(maxsize=64)
def _compile_layernorm(m: int, n: int, dtype_str: str, eps: float):
    return build_layernorm_module(m, n, dtype_str, eps=eps)


@lru_cache(maxsize=64)
def _compile_add_layernorm(m: int, n: int, dtype_str: str, eps: float):
    return build_fused_add_layernorm_module(m, n, dtype_str, eps=eps)


@lru_cache(maxsize=64)
def _compile_layernorm_dynamicquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_layernorm_dynamicquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


@lru_cache(maxsize=64)
def _compile_layernorm_smoothquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_layernorm_smoothquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


@lru_cache(maxsize=64)
def _compile_add_layernorm_dynamicquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_fused_add_layernorm_dynamicquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


@lru_cache(maxsize=64)
def _compile_add_layernorm_smoothquant(
    m: int,
    n: int,
    dtype_str: str,
    quant_dtype: str,
    eps: float,
):
    return build_fused_add_layernorm_smoothquant_module(
        m,
        n,
        dtype_str,
        quant_dtype_str=quant_dtype,
        eps=eps,
    )


def flydsl_rmsnorm(
    input: Tensor,
    weight: Tensor,
    *,
    out: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tensor:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    out = _check_optional_out("out", out, input)
    launcher = _compile_rmsnorm(m, n, dtype_str, float(epsilon))
    _run_launcher(device, launcher, input, weight, out, m, stream=stream)
    return out


def flydsl_add_rmsnorm(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    *,
    out: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_2d_like("residual_in", residual_in, tuple(input.shape), input.dtype, device)
    out = _check_optional_out("out", out, input)
    residual_out = _check_optional_out("residual_out", residual_out, input)
    launcher = _compile_add_rmsnorm(m, n, dtype_str, float(epsilon))
    _run_launcher(
        device,
        launcher,
        input,
        residual_in,
        weight,
        out,
        residual_out,
        m,
        stream=stream,
    )
    return out, residual_out


def flydsl_rmsnorm_dynamicquant(
    input: Tensor,
    weight: Tensor,
    *,
    out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_rmsnorm_dynamicquant(m, n, dtype_str, quant_dtype, float(epsilon))
    _run_launcher(device, launcher, input, weight, out, yscale, m, stream=stream)
    return out, yscale


def flydsl_rmsnorm_smoothquant(
    input: Tensor,
    weight: Tensor,
    xscale: Tensor,
    *,
    out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_vector("xscale", xscale, n, input.dtype, device)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_rmsnorm_smoothquant(m, n, dtype_str, quant_dtype, float(epsilon))
    _run_launcher(device, launcher, input, weight, xscale, out, yscale, m, stream=stream)
    return out, yscale


def flydsl_add_rmsnorm_dynamicquant(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    *,
    out: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_2d_like("residual_in", residual_in, tuple(input.shape), input.dtype, device)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    residual_out = _check_optional_out("residual_out", residual_out, input)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_add_rmsnorm_dynamicquant(
        m,
        n,
        dtype_str,
        quant_dtype,
        float(epsilon),
    )
    _run_launcher(
        device,
        launcher,
        input,
        residual_in,
        weight,
        out,
        residual_out,
        yscale,
        m,
        stream=stream,
    )
    return out, residual_out, yscale


def flydsl_add_rmsnorm_smoothquant(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    xscale: Tensor,
    *,
    out: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_2d_like("residual_in", residual_in, tuple(input.shape), input.dtype, device)
    _check_vector("xscale", xscale, n, input.dtype, device)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    residual_out = _check_optional_out("residual_out", residual_out, input)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_add_rmsnorm_smoothquant(
        m,
        n,
        dtype_str,
        quant_dtype,
        float(epsilon),
    )
    _run_launcher(
        device,
        launcher,
        input,
        residual_in,
        weight,
        xscale,
        out,
        residual_out,
        yscale,
        m,
        stream=stream,
    )
    return out, residual_out, yscale


def flydsl_layernorm(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    out: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tensor:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_vector("bias", bias, n, input.dtype, device)
    out = _check_optional_out("out", out, input)
    launcher = _compile_layernorm(m, n, dtype_str, float(epsilon))
    _run_launcher(device, launcher, input, weight, bias, out, m, stream=stream)
    return out


def flydsl_add_layernorm(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    out: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_vector("bias", bias, n, input.dtype, device)
    _check_2d_like("residual_in", residual_in, tuple(input.shape), input.dtype, device)
    out = _check_optional_out("out", out, input)
    residual_out = _check_optional_out("residual_out", residual_out, input)
    launcher = _compile_add_layernorm(m, n, dtype_str, float(epsilon))
    _run_launcher(
        device,
        launcher,
        input,
        residual_in,
        weight,
        bias,
        out,
        residual_out,
        m,
        stream=stream,
    )
    return out, residual_out


def flydsl_layernorm_dynamicquant(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_vector("bias", bias, n, input.dtype, device)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_layernorm_dynamicquant(
        m,
        n,
        dtype_str,
        quant_dtype,
        float(epsilon),
    )
    _run_launcher(device, launcher, input, weight, bias, out, yscale, m, stream=stream)
    return out, yscale


def flydsl_layernorm_smoothquant(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    xscale: Tensor,
    *,
    out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_vector("bias", bias, n, input.dtype, device)
    _check_vector("xscale", xscale, n, input.dtype, device)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_layernorm_smoothquant(
        m,
        n,
        dtype_str,
        quant_dtype,
        float(epsilon),
    )
    _run_launcher(
        device,
        launcher,
        input,
        weight,
        bias,
        xscale,
        out,
        yscale,
        m,
        stream=stream,
    )
    return out, yscale


def flydsl_add_layernorm_dynamicquant(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    out: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_vector("bias", bias, n, input.dtype, device)
    _check_2d_like("residual_in", residual_in, tuple(input.shape), input.dtype, device)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    residual_out = _check_optional_out("residual_out", residual_out, input)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_add_layernorm_dynamicquant(
        m,
        n,
        dtype_str,
        quant_dtype,
        float(epsilon),
    )
    _run_launcher(
        device,
        launcher,
        input,
        residual_in,
        weight,
        bias,
        out,
        residual_out,
        yscale,
        m,
        stream=stream,
    )
    return out, residual_out, yscale


def flydsl_add_layernorm_smoothquant(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    bias: Tensor,
    xscale: Tensor,
    *,
    out: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    yscale: Optional[Tensor] = None,
    epsilon: float = DEFAULT_EPS,
    quant_dtype: str = "i8",
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    m, n, dtype_str, device = _check_norm_input(input, weight)
    _check_vector("bias", bias, n, input.dtype, device)
    _check_2d_like("residual_in", residual_in, tuple(input.shape), input.dtype, device)
    _check_vector("xscale", xscale, n, input.dtype, device)
    quant_dtype = _normalize_quant_dtype(quant_dtype)
    out = _check_optional_out("out", out, input, dtype=torch.int8)
    residual_out = _check_optional_out("residual_out", residual_out, input)
    yscale = _check_optional_yscale(yscale, m, device)
    launcher = _compile_add_layernorm_smoothquant(
        m,
        n,
        dtype_str,
        quant_dtype,
        float(epsilon),
    )
    _run_launcher(
        device,
        launcher,
        input,
        residual_in,
        weight,
        bias,
        xscale,
        out,
        residual_out,
        yscale,
        m,
        stream=stream,
    )
    return out, residual_out, yscale

