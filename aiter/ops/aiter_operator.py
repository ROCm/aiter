# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops, AITER_CSRC_DIR
from functools import partial
from typing import Any

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


binary_add_build_args = partial(cmdGenFunc, "add")
binary_sub_build_args = partial(cmdGenFunc, "sub")
binary_mul_build_args = partial(cmdGenFunc, "mul")
binary_div_build_args = partial(cmdGenFunc, "div")


@compile_ops("module_aiter_operator", gen_func=binary_add_build_args)
def add(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator", gen_func=binary_sub_build_args)
def sub(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator", gen_func=binary_mul_build_args)
def mul(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator", gen_func=binary_div_build_args)
def div(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator", gen_func=binary_add_build_args)
def add_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator", gen_func=binary_sub_build_args)
def sub_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator", gen_func=binary_mul_build_args)
def mul_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator", gen_func=binary_div_build_args)
def div_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_unary")
def sigmoid(input: Tensor) -> Tensor: ...


@compile_ops("module_aiter_unary")
def tanh(input: Tensor) -> Tensor: ...
