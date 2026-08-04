# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse

import torch

from ..jit.utils.chip_info import get_gfx_runtime
from ..ops.enum import ActivationType, QuantType
from .aiter_types import aiter_dtypes, aiter_tensor_t

defaultDtypes = {
    "gfx942": {"fp8": torch.float8_e4m3fnuz},
    "gfx950": {"fp8": torch.float8_e4m3fn},
    "gfx1200": {"fp8": torch.float8_e4m3fn},
    "gfx1201": {"fp8": torch.float8_e4m3fn},
    "gfx1250": {"fp8": torch.float8_e4m3fn},
}

_8bit_fallback = torch.uint8


def get_dtype_fp8():
    return defaultDtypes.get(get_gfx_runtime(), {"fp8": _8bit_fallback})["fp8"]


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


def _aiter_dtype_id(dtype) -> int:
    """torch dtype -> AiterDtype enum id, or raise with the same message the
    former `assert dtype in _torch_to_aiter_dtype` produced."""
    try:
        return _torch_to_aiter_dtype[dtype]
    except KeyError:
        raise AssertionError(f"Unsupported dtype: {dtype}") from None


def torch_to_aiter_pybind(tensor: torch.Tensor):
    """Convert torch.Tensor to pybind aiter_tensor_t for passing to C++ ops.

    Unlike torch_to_aiter() which returns a ctypes aiter_tensor_t struct,
    this function constructs a *pybind11* aiter_tensor_t via
    module_aiter_core.  The two types are not interchangeable.
    """
    shape = tensor.shape
    ndim = len(shape)
    assert ndim <= 8, f"aiter_tensor_t supports at most 8 dims, got {ndim}"
    dtype_ = _aiter_dtype_id(tensor.dtype)
    index = tensor.device.index

    from ..jit.core import get_module

    aiter_tensor_cls = get_module("module_aiter_core").aiter_tensor_t
    return aiter_tensor_cls(
        tensor.data_ptr(),
        tensor.numel(),
        ndim,
        list(shape),
        list(tensor.stride()),
        dtype_,
        -1 if index is None else index,
    )


def torch_to_aiter(tensor: torch.Tensor) -> aiter_tensor_t:
    """This is for ctypes binding.
    torch.Tensor -> aiter_tensor_t, zero-copy, points to the same GPU memory.

    On the hot path of every ffi_type="ctypes" op, so each torch attribute is
    read exactly once and shape/strides go in as whole tuples: the per-tensor
    cost is O(1) in ndim rather than O(ndim). `tensor.stride()` (no arg) hands
    back the full tuple, and ctypes arrays accept slice assignment, so no
    per-dim Python loop is needed.
    """
    shape = tensor.shape
    strides = tensor.stride()
    ndim = len(shape)
    assert ndim <= 8, f"aiter_tensor_t supports at most 8 dims, got {ndim}"
    dtype_ = _aiter_dtype_id(tensor.dtype)
    # device.index is None for CPU tensors (and for an un-indexed device);
    # -1 is the C-side "not on a GPU" sentinel.
    index = tensor.device.index

    at = aiter_tensor_t()
    at.ptr = tensor.data_ptr()
    at.numel_ = tensor.numel()
    at.ndim = ndim
    at.shape[:ndim] = shape
    at.strides[:ndim] = strides
    at.dtype_ = dtype_
    at.device_id = -1 if index is None else index
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
