# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small runtime helpers shared by Python ASM dispatch policy."""

import torch


def get_gfx_from_device(device=None) -> str:
    """Return the architecture of a live torch device."""
    props = torch.cuda.get_device_properties(device)
    return props.gcnArchName.split(":", 1)[0].lower()
