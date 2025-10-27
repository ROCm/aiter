# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops

MD_NAME = "module_softmax"


def gen_softmax_fake_tensors(
    input: Tensor,
    dim: List[int],
) -> Tensor:
    return torch.empty_like(
        input,
        dtype=input.dtype,
        device=input.device,
    )


@compile_ops(
    "module_softmax", fc_name="softmax2d_fwd", gen_fake=gen_softmax_fake_tensors
)
def softmax(
    input: Tensor,
    dim: List[int],
) -> Tensor: ...


@compile_ops(
    "module_softmax", fc_name="softmax2d_fwd", gen_fake=gen_softmax_fake_tensors
)
def softmax2d_fwd(
    input: Tensor,
    dim: List[int],
) -> Tensor: ...


@compile_ops("module_softmax")
def softmax2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    dim: List[int],
) -> None: ...


@compile_ops("module_softmax")
def softmax2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    dim: List[int],
) -> None: ...


@compile_ops("module_softmax")
def softmax2d_fwd_with_add_smoothquant(
    out: Tensor,
    input: Tensor,
    dim: List[int],
) -> None: ...

@compile_ops("module_softmax")
def softmax2d_with_add_asm(
    out: Tensor,
    input: Tensor,
    dim: List[int],
) -> None: ...


@compile_ops("module_softmax")
def softmax2d_with_add_smoothquant_asm(
    out: Tensor,
    input: Tensor,
    dim: List[int],
) -> None: ...


@compile_ops("module_softmax")
def softmax2d_hip(
    input: Tensor,
    dim: List[int],
) -> Tensor: ...
