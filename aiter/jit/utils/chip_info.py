# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import functools


@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        import torch

        device = torch.cuda.current_device()
        gfx = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return gfx


@functools.lru_cache(maxsize=1)
def get_cu_num():
    import torch

    device = torch.cuda.current_device()
    cu_num = torch.cuda.get_device_properties(device).multi_processor_count
    return cu_num
