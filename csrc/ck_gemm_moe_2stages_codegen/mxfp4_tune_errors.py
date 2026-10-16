# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only classification shared by coupled tuning and coverage checks."""


def is_fatal_gpu_error(error: object) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "illegal memory access",
            "illegal address",
            "memory access fault",
            "device-side assert",
            "device side assert",
            "unspecified launch failure",
            "launch failure",
            "hiperrorillegaladdress",
            "hiperrorassert",
            "hiperrorlaunchfailure",
            "hiperrorlaunchfailed",
            "hiperrorcontextisdestroyed",
            "context is destroyed",
            "context has been destroyed",
        )
    )
