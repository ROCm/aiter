# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_activation"


@compile_ops("module_activation", develop=True)
def silu_and_mul(out: Tensor, input: Tensor, limit: float = 0.0) -> None: ...


@compile_ops("module_activation", develop=True)
def swiglu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def silu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def swiglu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def silu_and_mul_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    group_size: int,
    limit: float = 0.0,
    shuffle_scale: bool = False,
) -> None: ...


@compile_ops("module_activation", develop=True)
def situv2_and_mul_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    group_size: int,
    beta: float,
    linear_beta: float,
    shuffle_scale: bool = False,
) -> None:
    """Apply SiTUv2 and per-token FP8 quantization.

    All tensors must be contiguous ROCm tensors. ``input`` is BF16 with shape
    ``[..., 2 * d]`` where ``d`` is divisible by 8, ``out`` is FP8 with
    ``input.numel() / 2`` elements, and ``scale`` is FP32 with one element per
    token. Only ``group_size == d`` and ``shuffle_scale=False`` are supported.
    """


@compile_ops("module_activation", develop=True)
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_fast(out: Tensor, input: Tensor) -> None: ...
