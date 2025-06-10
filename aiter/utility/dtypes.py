# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
from ..jit.utils.chip_info import get_gfx
import argparse

defaultDtypes = {
    "gfx942": {"fp8": torch.float8_e4m3fnuz},
    "gfx950": {"fp8": torch.float8_e4m3fn},
}


def get_dtype_fp8():
    return defaultDtypes[get_gfx()]["fp8"]


i4x2 = getattr(torch, "int4", torch.uint8)
fp4x2 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
fp8 = get_dtype_fp8()
fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
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
    try:
        parts = v.strip("()").split(",")

        return tuple(int(p.strip()) for p in parts)
    except:
        raise argparse.ArgumentTypeError(f"invalid format of input: {v}")
