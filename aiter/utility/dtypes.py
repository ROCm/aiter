# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
from ..jit.utils.chip_info import get_gfx
from ..ops.enum import QuantType
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

d_dtypes = {
    "fp8": fp8,
    "fp8_e8m0": fp8_e8m0,
    "fp16": fp16,
    "bf16": bf16,
    "fp32": fp32,
    "i4x2": i4x2,
    "fp4x2": fp4x2,
    "u32": u32,
    "i32": i32,
    "i16": i16,
    "i8": i8,
}


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
