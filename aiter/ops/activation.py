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
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_fast(out: Tensor, input: Tensor) -> None: ...
