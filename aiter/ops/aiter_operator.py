# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import math
from functools import cache, partial
from typing import Any

import torch
from torch import Tensor

from ..jit.core import AITER_CSRC_DIR, compile_ops
from ..jit.utils.torch_guard import is_torch_equal_or_newer

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


# binary_scalar's op values; BINARY_SCALAR_ADD / _SUB / _MUL in csrc/include/aiter_operator.h
# must match. div has none: it runs as mul by the reciprocal.
_BINARY_SCALAR_ADD = 0
_BINARY_SCALAR_SUB = 1
_BINARY_SCALAR_MUL = 2
# the op name of each value, which names its module
_BINARY_SCALAR_OPS = {
    _BINARY_SCALAR_ADD: "add",
    _BINARY_SCALAR_SUB: "sub",
    _BINARY_SCALAR_MUL: "mul",
}


def binary_scalar_build_args(input: Tensor, op: int, *_args) -> dict[str, Any]:
    # tensor (op) host scalar runs in the (input.dtype, input.dtype) module of op; compile_ops calls
    # this on every call, so the answer is cached per (op, dtype)
    return _binary_scalar_module_args(op, input.dtype)


@cache
def _binary_scalar_module_args(op: int, dtype: torch.dtype) -> dict[str, Any]:
    # an unknown value names no known op's module, so its build fails as an unknown op name's does
    probe = torch.empty(0, dtype=dtype)
    return cmdGenFunc(_BINARY_SCALAR_OPS.get(op, str(op)), probe, probe)


def binary_scalar_fake_shape(
    input: Tensor, op: int, other: float, output: Tensor
) -> Tensor:
    return output


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


@compile_ops(
    MD_NAME,
    fc_name="binary_scalar",
    develop=True,
    gen_func=binary_scalar_build_args,
    gen_fake=binary_scalar_fake_shape,
)
def _binary_scalar_kernel(
    input: Tensor, op: int, other: float, output: Tensor
) -> bool: ...


_FLAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_DIV_RECIPROCAL_IN_DOUBLE = is_torch_equal_or_newer("2.10.0")


def _is_host_scalar(other: Any) -> bool:
    # bool included: it takes the host-scalar branch and falls back to torch there
    return isinstance(other, (int, float))


def _flat_scalar_ok(input: Tensor, other: Any) -> bool:
    # the conditions binary_scalar_process puts on input (dtype, contiguity, 16-byte alignment, a
    # whole number of 16-byte vectors below 2 GiB, device arch), checked before an output is
    # allocated: an eager binding call costs more than torch's own call, so an input the C++ would
    # reject goes straight to torch. A strided view is turned away by the first three tests. Under
    # torch.compile the wrappers call torch, which keeps the graph whole (they branch on the
    # binding's result); that test follows the ones Dynamo can trace and precedes data_ptr(), which
    # it cannot. Zero elements pass (torch calls them contiguous, and binary_scalar_process returns
    # true for them); the C++ checks stay authoritative.
    if (
        isinstance(other, bool)
        or input.dtype not in _FLAT_DTYPES
        or not input.is_contiguous()
        or torch.compiler.is_compiling()
    ):
        return False
    nbytes = input.nbytes
    return (
        (not nbytes or (nbytes < 1 << 31 and (input.data_ptr() | nbytes) % 16 == 0))
        and input.is_cuda
        and (_all_devices_flat or _flat_device_ok(input))
    )


# must match binary_flat_arch_supported() in csrc/include/binary_operator.cuh, which stays the
# authoritative gate; this copy only keeps other archs from allocating an output torch then replaces
_FLAT_GFX = ("gfx950", "gfx942")
# None until the first eager host-scalar call, then whether every visible device is a _FLAT_GFX arch
_all_devices_flat = None


@cache
def _flat_arch_ok(device_index: int) -> bool:
    # per device, because one process may drive GPUs of different archs; imported here so that
    # importing this module queries no device
    from .asm.asm_utils import get_gfx_from_device

    return get_gfx_from_device(device_index) in _FLAT_GFX


def _flat_device_ok(input: Tensor) -> bool:
    global _all_devices_flat
    if _all_devices_flat is None:
        _all_devices_flat = all(
            _flat_arch_ok(i) for i in range(torch.cuda.device_count())
        )
    return _all_devices_flat or _flat_arch_ok(input.get_device())


def _host_reciprocal(other: Any) -> float:
    # torch divides by a CPU scalar as a multiply by its reciprocal, formed in double from torch
    # 2.10 on and in fp32 before; 1 / +-0 is +-inf
    other = float(other) if _DIV_RECIPROCAL_IN_DOUBLE else ctypes.c_float(other).value
    return math.copysign(math.inf, other) if other == 0.0 else 1.0 / other


def _scalar_op(
    op: int, torch_op, input: Tensor, other: Any, reciprocal=False
) -> Tensor:
    if _flat_scalar_ok(input, other):
        try:
            scalar = _host_reciprocal(other) if reciprocal else float(other)
        except OverflowError:
            return torch_op(input, other)
        output = torch.empty_like(input)
        if _binary_scalar_kernel(input, op, scalar, output):
            return output
    return torch_op(input, other)


def _scalar_op_(
    op: int, torch_op_, input: Tensor, other: Any, reciprocal=False
) -> Tensor:
    if _flat_scalar_ok(input, other):
        try:
            scalar = _host_reciprocal(other) if reciprocal else float(other)
        except OverflowError:
            scalar = None
        if scalar is not None and _binary_scalar_kernel(input, op, scalar, input):
            return input
    torch_op_(input, other)
    return input


def add(input: Tensor, other: Tensor | float) -> Tensor:
    if _is_host_scalar(other):
        return _scalar_op(_BINARY_SCALAR_ADD, torch.add, input, other)
    output = _make_output(input, other)
    if not _add_kernel(input, other, output):
        output = torch.add(input, other)
    return output


def sub(input: Tensor, other: Tensor | float) -> Tensor:
    if _is_host_scalar(other):
        return _scalar_op(_BINARY_SCALAR_SUB, torch.sub, input, other)
    output = _make_output(input, other)
    if not _sub_kernel(input, other, output):
        output = torch.sub(input, other)
    return output


def mul(input: Tensor, other: Tensor | float) -> Tensor:
    """input * other. Like torch, the result is always a new tensor, also for a host scalar
    other == 1 (the scalar kernel then writes an exact copy of input)."""
    if _is_host_scalar(other):
        return _scalar_op(_BINARY_SCALAR_MUL, torch.mul, input, other)
    output = _make_output(input, other)
    if not _mul_kernel(input, other, output):
        output = torch.mul(input, other)
    return output


def div(input: Tensor, other: Tensor | float) -> Tensor:
    """input / other. A host scalar other runs the multiply kernel with 1 / other, which is how
    torch divides by a CPU scalar, so the bits match torch's."""
    if _is_host_scalar(other):
        return _scalar_op(_BINARY_SCALAR_MUL, torch.div, input, other, reciprocal=True)
    output = _make_output(input, other)
    if not _div_kernel(input, other, output):
        output = torch.div(input, other)
    return output


def add_(input: Tensor, other: Tensor | float) -> Tensor:
    if _is_host_scalar(other):
        return _scalar_op_(_BINARY_SCALAR_ADD, Tensor.add_, input, other)
    if not _add_kernel_(input, other):
        input.add_(other)
    return input


def sub_(input: Tensor, other: Tensor | float) -> Tensor:
    if _is_host_scalar(other):
        return _scalar_op_(_BINARY_SCALAR_SUB, Tensor.sub_, input, other)
    if not _sub_kernel_(input, other):
        input.sub_(other)
    return input


def mul_(input: Tensor, other: Tensor | float) -> Tensor:
    """input *= other in place. A host scalar other == 1 on a floating-point input changes
    nothing, so it launches nothing and returns input."""
    if _is_host_scalar(other):
        if other == 1 and not isinstance(other, bool) and input.dtype.is_floating_point:
            return input
        return _scalar_op_(_BINARY_SCALAR_MUL, Tensor.mul_, input, other)
    if not _mul_kernel_(input, other):
        input.mul_(other)
    return input


def div_(input: Tensor, other: Tensor | float) -> Tensor:
    if _is_host_scalar(other):
        return _scalar_op_(
            _BINARY_SCALAR_MUL, Tensor.div_, input, other, reciprocal=True
        )
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
