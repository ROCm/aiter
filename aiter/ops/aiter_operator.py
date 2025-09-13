# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops, AITER_CSRC_DIR
from functools import partial
from typing import Any
import torch

MD_NAME = "module_aiter_operator"


def cmdGenFunc(op_name: str, input: Tensor, other: Tensor) -> dict[str, Any]:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype {op_name} --dtypes {dtype_str}"
    ]
    return {
        "md_name": f"module_aiter_{op_name}_{dtype_str}",
        "blob_gen_cmd": blob_gen_cmd,
    }


def binary_fake_shape(input: Tensor, other: Tensor) -> Tensor:
    shape1 = list(input.shape)
    shape2 = list(other.shape)

    max_dim = max(len(shape1), len(shape2))
    shape1 = [1] * (max_dim - len(shape1)) + shape1
    shape2 = [1] * (max_dim - len(shape2)) + shape2

    result_shape = []
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == 1:
            result_shape.append(dim2)
        elif dim2 == 1:
            result_shape.append(dim1)
        elif dim1 == dim2:
            result_shape.append(dim1)
        else:
            raise RuntimeError(
                f"Incompatible shapes for binary operator: {input.shape} and {other.shape}"
            )

    return torch.empty(
        size=result_shape,
        dtype=input.dtype,
        device=input.device,
    )


def sigmoid_fake_shape(input: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        size=input.shape,
        dtype=input.dtype,
        device=input.device,
    )


binary_add_build_args = partial(cmdGenFunc, "add")
binary_sub_build_args = partial(cmdGenFunc, "sub")
binary_mul_build_args = partial(cmdGenFunc, "mul")
binary_div_build_args = partial(cmdGenFunc, "div")


@compile_ops(
    "module_aiter_operator", gen_func=binary_add_build_args, gen_fake=binary_fake_shape
)
def add(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(
    "module_aiter_operator", gen_func=binary_sub_build_args, gen_fake=binary_fake_shape
)
def sub(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(
    "module_aiter_operator", gen_func=binary_mul_build_args, gen_fake=binary_fake_shape
)
def mul(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(
    "module_aiter_operator", gen_func=binary_div_build_args, gen_fake=binary_fake_shape
)
def div(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(
    "module_aiter_operator", gen_func=binary_add_build_args, gen_fake=binary_fake_shape
)
def add_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(
    "module_aiter_operator", gen_func=binary_sub_build_args, gen_fake=binary_fake_shape
)
def sub_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(
    "module_aiter_operator", gen_func=binary_mul_build_args, gen_fake=binary_fake_shape
)
def mul_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(
    "module_aiter_operator", gen_func=binary_div_build_args, gen_fake=binary_fake_shape
)
def div_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_unary", gen_fake=sigmoid_fake_shape)
def sigmoid(input: Tensor) -> Tensor: ...


@compile_ops("module_aiter_unary", gen_fake=sigmoid_fake_shape)
def tanh(input: Tensor) -> Tensor: ...
