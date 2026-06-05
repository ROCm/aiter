# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops, AITER_CSRC_DIR
from functools import partial
from typing import Any
import torch

MD_NAME = "module_aiter_operator"


def cmdGenFunc(op_name: str, input: Tensor, other: Tensor, *_args) -> dict[str, Any]:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype {op_name} --dtypes {dtype_str}"
    ]
    return {
        "md_name": f"module_aiter_{op_name}_{dtype_str}",
        "blob_gen_cmd": blob_gen_cmd,
    }


def binary_out_fake_shape(input: Tensor, other: Tensor, output: Tensor) -> Tensor:
    return output


def binary_inp_fake_shape(input: Tensor, other: Tensor) -> Tensor:
    return input


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


def _make_output(input: Tensor, other: Tensor) -> Tensor:
    out_shape = torch.broadcast_shapes(input.shape, other.shape)
    out_dtype = torch.promote_types(input.dtype, other.dtype)
    return torch.empty(out_shape, dtype=out_dtype, device=input.device)


@compile_ops(
    MD_NAME,
    fc_name="add",
    develop=True,
    gen_func=binary_add_build_args,
    gen_fake=binary_out_fake_shape,
)
def _add_kernel(input: Tensor, other: Tensor, output: Tensor) -> bool: ...


@compile_ops(
    MD_NAME,
    fc_name="sub",
    develop=True,
    gen_func=binary_sub_build_args,
    gen_fake=binary_out_fake_shape,
)
def _sub_kernel(input: Tensor, other: Tensor, output: Tensor) -> bool: ...


@compile_ops(
    MD_NAME,
    fc_name="mul",
    develop=True,
    gen_func=binary_mul_build_args,
    gen_fake=binary_out_fake_shape,
)
def _mul_kernel(input: Tensor, other: Tensor, output: Tensor) -> bool: ...


@compile_ops(
    MD_NAME,
    fc_name="div",
    develop=True,
    gen_func=binary_div_build_args,
    gen_fake=binary_out_fake_shape,
)
def _div_kernel(input: Tensor, other: Tensor, output: Tensor) -> bool: ...


@compile_ops(
    MD_NAME,
    fc_name="add_",
    develop=True,
    gen_func=binary_add_build_args,
    gen_fake=binary_inp_fake_shape,
)
def _add_kernel_(input: Tensor, other: Tensor) -> bool: ...


@compile_ops(
    MD_NAME,
    fc_name="sub_",
    develop=True,
    gen_func=binary_sub_build_args,
    gen_fake=binary_inp_fake_shape,
)
def _sub_kernel_(input: Tensor, other: Tensor) -> bool: ...


@compile_ops(
    MD_NAME,
    fc_name="mul_",
    develop=True,
    gen_func=binary_mul_build_args,
    gen_fake=binary_inp_fake_shape,
)
def _mul_kernel_(input: Tensor, other: Tensor) -> bool: ...


@compile_ops(
    MD_NAME,
    fc_name="div_",
    develop=True,
    gen_func=binary_div_build_args,
    gen_fake=binary_inp_fake_shape,
)
def _div_kernel_(input: Tensor, other: Tensor) -> bool: ...


def add(input: Tensor, other: Tensor) -> Tensor:
    output = _make_output(input, other)
    if not _add_kernel(input, other, output):
        output = torch.add(input, other)
    return output


def sub(input: Tensor, other: Tensor) -> Tensor:
    output = _make_output(input, other)
    if not _sub_kernel(input, other, output):
        output = torch.sub(input, other)
    return output


def mul(input: Tensor, other: Tensor) -> Tensor:
    output = _make_output(input, other)
    if not _mul_kernel(input, other, output):
        output = torch.mul(input, other)
    return output


def div(input: Tensor, other: Tensor) -> Tensor:
    output = _make_output(input, other)
    if not _div_kernel(input, other, output):
        output = torch.div(input, other)
    return output


def add_(input: Tensor, other: Tensor) -> Tensor:
    if not _add_kernel_(input, other):
        input.add_(other)
    return input


def sub_(input: Tensor, other: Tensor) -> Tensor:
    if not _sub_kernel_(input, other):
        input.sub_(other)
    return input


def mul_(input: Tensor, other: Tensor) -> Tensor:
    if not _mul_kernel_(input, other):
        input.mul_(other)
    return input


def div_(input: Tensor, other: Tensor) -> Tensor:
    if not _div_kernel_(input, other):
        input.div_(other)
    return input


@compile_ops("module_aiter_unary", gen_fake=sigmoid_fake_shape)
def sigmoid(input: Tensor) -> Tensor: ...


@compile_ops("module_aiter_unary", gen_fake=sigmoid_fake_shape)
def tanh(input: Tensor) -> Tensor: ...
