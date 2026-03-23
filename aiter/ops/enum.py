# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Mirror of csrc/include/aiter_enum.h -- update both when changing enum values
from enum import IntEnum

Enum = int


class ActivationType(IntEnum):
    No = -1
    Silu = 0
    Gelu = 1
    Swiglu = 2


class QuantType(IntEnum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x32 = 3
    per_1x128 = 4
    per_128x128 = 5
    per_256x128 = 6
    per_1024x128 = 7
