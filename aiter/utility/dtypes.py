# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import functools
from ..jit.utils.chip_info import get_gfx

defaultDtypes = {
    "gfx942": {"fp8": torch.float8_e4m3fnuz},
    "gfx950": {"fp8": torch.float8_e4m3fn},
}


def get_dtype_fp8():
    return defaultDtypes[get_gfx()]["fp8"]


fp8 = get_dtype_fp8()
fp16 = torch.float16
bf16 = torch.bfloat16
fp32 = torch.float32
u32 = torch.uint32
i32 = torch.int32
i16 = torch.int16
i8 = torch.int8
