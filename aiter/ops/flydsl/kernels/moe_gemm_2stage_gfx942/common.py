# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.float64: fx.Float64,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
}


def down_device_config_from_properties(gcn_arch_name, cu_count):
    is_gfx942_80cu = gcn_arch_name.split(":", 1)[0] == "gfx942" and cu_count == 80
    return is_gfx942_80cu, 4 if is_gfx942_80cu else 8


def get_down_device_config():
    target_arch = os.environ.get("FLYDSL_GPU_ARCH")
    if target_arch is not None and "CU_NUM" in os.environ:
        try:
            target_cu_count = int(os.environ["CU_NUM"])
        except ValueError as error:
            raise ValueError(
                f"CU_NUM must be an integer, got {os.environ['CU_NUM']!r}"
            ) from error
        return down_device_config_from_properties(target_arch, target_cu_count)
    if not torch.cuda.is_available():
        return False, 8
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return down_device_config_from_properties(
        properties.gcnArchName,
        properties.multi_processor_count,
    )


@functools.cache
def _get_device_cache_key(device):
    properties = torch.cuda.get_device_properties(device)
    return (
        device,
        properties.name,
        properties.gcnArchName,
        properties.multi_processor_count,
    )


def get_device_cache_key():
    target_arch = os.environ.get("FLYDSL_GPU_ARCH")
    target_cu_count = os.environ.get("CU_NUM")
    if not torch.cuda.is_available():
        return None, target_arch, target_cu_count
    return (
        _get_device_cache_key(torch.cuda.current_device()),
        target_arch,
        target_cu_count,
    )


def torch_tensor_to_pointer(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())
