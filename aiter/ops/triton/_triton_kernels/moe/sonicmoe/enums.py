# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from enum import Enum

LIBRARY_NAME = "aiter_sonicmoe"


class ActivationType(Enum):
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"

    RELU_SQ = "relu_sq"
    RELU = "relu"
    GELU = "gelu_tanh_approx"
    SILU = "silu"


def is_glu(activation_type: ActivationType):
    return activation_type in [
        ActivationType.SWIGLU,
        ActivationType.REGLU,
        ActivationType.GEGLU,
    ]
