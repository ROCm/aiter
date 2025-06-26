# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops, AITER_CSRC_DIR

MD_NAME = "module_aiter_operator"


@compile_ops("module_aiter_operator")
def _add(input: Tensor, other: Tensor) -> Tensor: ...


def add(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype add --dtypes {dtype_str}"
    ]
    out = _add(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_add_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def _sub(input: Tensor, other: Tensor) -> Tensor: ...


def sub(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype sub --dtypes {dtype_str}"
    ]
    out = _sub(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_sub_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def _mul(input: Tensor, other: Tensor) -> Tensor: ...


def mul(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype mul --dtypes {dtype_str}"
    ]
    out = _mul(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_mul_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def _div(input: Tensor, other: Tensor) -> Tensor: ...


def div(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype div --dtypes {dtype_str}"
    ]
    out = _div(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_div_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def _add_(input: Tensor, other: Tensor) -> Tensor: ...


def add_(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype add --dtypes {dtype_str}"
    ]
    out = _add_(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_add_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def _sub_(input: Tensor, other: Tensor) -> Tensor: ...


def sub_(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype sub --dtypes {dtype_str}"
    ]
    out = _sub_(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_sub_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def _mul_(input: Tensor, other: Tensor) -> Tensor: ...


def mul_(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype mul --dtypes {dtype_str}"
    ]
    out = _mul_(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_mul_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def _div_(input: Tensor, other: Tensor) -> Tensor: ...


def div_(input: Tensor, other: Tensor) -> Tensor:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py --working_path {{}} --optype div --dtypes {dtype_str}"
    ]
    out = _div_(
        input,
        other,
        custom_build_args={
            "md_name": "module_aiter_div_"+dtype_str,
            "blob_gen_cmd": blob_gen_cmd,
        },
    )
    return out


@compile_ops("module_aiter_operator")
def sigmoid(input: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def tanh(input: Tensor) -> Tensor: ...
