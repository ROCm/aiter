# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
from ..jit.utils.chip_info import get_gfx
from ..jit.core import compile_ops
from ..ops.enum import QuantType, ActivationType
from .aiter_types import aiter_dtypes, aiter_tensor_t
import argparse

defaultDtypes = {
    "gfx942": {"fp8": torch.float8_e4m3fnuz},
    "gfx950": {"fp8": torch.float8_e4m3fn},
    "gfx1250": {"fp8": torch.float8_e4m3fn},
}

_8bit_fallback = torch.uint8


def get_dtype_fp8():
    return defaultDtypes.get(get_gfx(), {"fp8": _8bit_fallback})["fp8"]


i4x2 = getattr(torch, "int4", _8bit_fallback)
fp4x2 = getattr(torch, "float4_e2m1fn_x2", _8bit_fallback)
fp8 = get_dtype_fp8()
fp8_e8m0 = getattr(torch, "float8_e8m0fnu", _8bit_fallback)
fp16 = torch.float16
bf16 = torch.bfloat16
fp32 = torch.float32
u32 = torch.uint32
i32 = torch.int32
i16 = torch.int16
i8 = torch.int8
u8 = torch.uint8
i64 = torch.int64
u64 = torch.uint64

d_dtypes = {name: globals()[name] for name in aiter_dtypes}

globals().update({f"AITER_DTYPE_{name}": idx for name, idx in aiter_dtypes.items()})
_torch_to_aiter_dtype = {globals()[name]: idx for name, idx in aiter_dtypes.items()}


@compile_ops("module_aiter_tensor", "make_aiter_tensor")
def _make_aiter_tensor(
    data_ptr: int,
    numel: int,
    ndim: int,
    shape: list,
    strides: list,
    dtype: int,
    device_id: int,
): ...


def torch_to_aiter_pybind(tensor: torch.Tensor):
    """Convert torch.Tensor to pybind aiter_tensor_t for passing to C++ ops.

    Unlike torch_to_aiter() which returns a ctypes aiter_tensor_t struct,
    this function constructs a *pybind11* aiter_tensor_t via
    module_aiter_tensor.  The two types are not interchangeable.
    """
    return _make_aiter_tensor(
        tensor.data_ptr(),
        tensor.numel(),
        tensor.ndim,
        list(tensor.shape),
        list(tensor.stride()),
        _torch_to_aiter_dtype[tensor.dtype],
        tensor.device.index or 0,
    )


def torch_to_aiter(tensor: torch.Tensor) -> aiter_tensor_t:
    """This is for ctypes binding.
    torch.Tensor -> aiter_tensor_t, zero-copy, points to the same GPU memory."""
    assert tensor.is_cuda, "aiter_tensor_t only supports CUDA tensors"
    assert (
        tensor.ndim <= 8
    ), f"aiter_tensor_t supports at most 8 dims, got {tensor.ndim}"
    assert tensor.dtype in _torch_to_aiter_dtype, f"Unsupported dtype: {tensor.dtype}"

    at = aiter_tensor_t()
    at.ptr = tensor.data_ptr()
    at.numel_ = tensor.numel()
    at.ndim = tensor.ndim
    for i in range(tensor.ndim):
        at.shape[i] = tensor.shape[i]
        at.strides[i] = tensor.stride(i)
    at.dtype_ = _torch_to_aiter_dtype[tensor.dtype]
    at.device_id = tensor.device.index or 0
    return at


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2tuple(v):
    """
    Convert string to int or tuple of ints.
    - "512" -> 512 (single value without comma returns int)
    - "512," -> (512,) (trailing comma returns tuple)
    - "512,1024" -> (512, 1024) (multiple values return tuple)
    """
    try:
        parts = [int(p.strip()) for p in v.strip("()").split(",") if p.strip()]
        # Return single value if only one element and no comma; otherwise return tuple
        if "," not in v and len(parts) == 1:
            return parts[0]
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid format of input: {v}") from e


def str2Dtype(v):
    def _convert(s):
        if s.lower() == "none":
            return None
        elif s in d_dtypes:
            return d_dtypes[s]
        else:
            # Case-insensitive lookup for QuantType
            s_lower = s.lower()
            for name in dir(QuantType):
                if not name.startswith("_") and name.lower() == s_lower:
                    return getattr(QuantType, name)
            raise ValueError(f"'{s}' not in d_dtypes or QuantType")

    try:
        parts = [p.strip() for p in v.strip("()").split(",") if p.strip()]
        # Return single value if only one element and no comma; otherwise return tuple
        if len(parts) == 1 and "," not in v:
            return _convert(parts[0])
        return tuple(_convert(p) for p in parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid format of type: {v}") from e


def str2ActivationType(s):
    """Convert string to ActivationType."""
    return getattr(ActivationType, s.capitalize())
