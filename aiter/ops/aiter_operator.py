# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import partial
from typing import Any

import torch
from torch import Tensor

from ..jit.core import AITER_CSRC_DIR, compile_ops

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


def unary_out_fake_shape(out: Tensor, input: Tensor) -> Tensor:
    return out


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


@compile_ops(
    "module_aiter_unary", fc_name="sigmoid", develop=True, gen_fake=unary_out_fake_shape
)
def _sigmoid(out: Tensor, input: Tensor) -> None: ...


@compile_ops(
    "module_aiter_unary", fc_name="tanh", develop=True, gen_fake=unary_out_fake_shape
)
def _tanh(out: Tensor, input: Tensor) -> None: ...


def _unary_tile_supported(input: Tensor) -> bool:
    # Mirror the C++ tile fast-path condition (unary_operator.cu): contiguous,
    # N % 8 == 0 and K % vec == 0, where vec is the number of elements spanning
    # 16 bytes for this dtype (fp16/bf16 -> 8, fp32 -> 4).
    if not input.is_contiguous():
        return False
    dim = input.dim()
    if dim == 2:
        n, k = input.size(0), input.size(1)
    else:
        n, k = input.size(1), input.size(2)
    vec = 16 // input.element_size()
    return n % 8 == 0 and k % vec == 0


def sigmoid(input: Tensor) -> Tensor:
    if not _unary_tile_supported(input):
        return torch.sigmoid(input)
    out = torch.empty_like(input)
    _sigmoid(out, input)
    return out


def tanh(input: Tensor) -> Tensor:
    if not _unary_tile_supported(input):
        return torch.tanh(input)
    out = torch.empty_like(input)
    _tanh(out, input)
    return out
