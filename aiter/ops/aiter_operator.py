# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops, AITER_CSRC_DIR

MD_NAME = "module_aiter_operator"


@compile_ops("module_aiter_operator")
def _add(input: Tensor, other: Tensor) -> Tensor: ...

# @compile_ops("module_aiter_operator")
def add(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype) + '_' + str(other.dtype)
    path_str = AITER_CSRC_DIR + "/../aiter/jit/build/module_aiter_operator/build/srcs"
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype add --dtypes {{}}".format(path_str, dtype_str)
    ]
    out = _add(input, other, custom_build_args={"md_name": "module_aiter_operator", "blob_gen_cmd": blob_gen_cmd})
    return out



# @compile_ops("module_aiter_operator")
# def sub(input: Tensor, other: Tensor) -> Tensor: ...


# @compile_ops("module_aiter_operator")
# def mul(input: Tensor, other: Tensor) -> Tensor: ...


# @compile_ops("module_aiter_operator")
# def div(input: Tensor, other: Tensor) -> Tensor: ...


# @compile_ops("module_aiter_operator")
# def add_(input: Tensor, other: Tensor) -> Tensor: ...


# @compile_ops("module_aiter_operator")
# def sub_(input: Tensor, other: Tensor) -> Tensor: ...


# @compile_ops("module_aiter_operator")
# def mul_(input: Tensor, other: Tensor) -> Tensor: ...


# @compile_ops("module_aiter_operator")
# def div_(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def sigmoid(input: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def tanh(input: Tensor) -> Tensor: ...
