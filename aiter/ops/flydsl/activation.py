# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter import ActivationType

DEFAULT_SITUV2_BETA = 4.0
DEFAULT_SITUV2_LINEAR_BETA = 25.0

_FLYDSL_ACTIVATION_NAMES = {
    ActivationType.Silu: "silu",
    ActivationType.Swiglu: "swiglu",
    ActivationType.Situv2: "situv2",
}


def get_flydsl_activation_name(activation) -> str:
    try:
        return _FLYDSL_ACTIVATION_NAMES[activation]
    except KeyError as error:
        raise ValueError(
            f"Unsupported FlyDSL MoE activation: {activation!r}"
        ) from error
