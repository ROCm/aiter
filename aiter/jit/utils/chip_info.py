# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess
import torch
import csrc.cpp_itfs.torch_utils
from torch.library import Library

aiter_lib = Library("aiter", "FRAGMENT")

from cpp_extension import executable_path


@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        try:
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            for line in output.split("\n"):
                if "gfx" in line.lower():
                    return line.split(":")[-1].strip()
        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rocminfo failed {str(e)}")
    elif ";" in gfx:
        gfx = gfx.split(";")[-1]
    return gfx


CU_NUM = 0


def get_cu_num_custom_op(dummy: torch.Tensor) -> None:
    global CU_NUM
    cu_num = int(os.getenv("CU_NUM", 0))
    if cu_num == 0:
        try:
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            devices = re.split(r"Agent\s*\d+", output)
            gpu_compute_units = []
            for device in devices:
                for line in device.split("\n"):
                    if "Device Type" in line and line.find("GPU") != -1:
                        match = re.search(r"Compute Unit\s*:\s*(\d+)", device)
                        if match:
                            gpu_compute_units.append(int(match.group(1)))
                        break
        except Exception as e:
            raise RuntimeError(f"Get GPU Compute Unit from rocminfo failed {str(e)}")
        assert len(set(gpu_compute_units)) == 1
        cu_num = gpu_compute_units[0]
    CU_NUM = cu_num


_CU_NUM_OP_REGISTERED = False


@functools.lru_cache(maxsize=1)
def get_cu_num():

    global _CU_NUM_OP_REGISTERED

    if not _CU_NUM_OP_REGISTERED:
        op_name = "aiter::get_cu_num_custom_op"
        schema_str = "(Tensor dummy) -> ()"
        torch.library.define(op_name, schema_str, lib=aiter_lib)
        torch.library.impl(op_name, "cuda", get_cu_num_custom_op, lib=aiter_lib)
        torch.library.register_fake(op_name, get_cu_num_custom_op, lib=aiter_lib)
        _CU_NUM_OP_REGISTERED = True

    x = torch.empty(1, device="cuda")
    torch.ops.aiter.get_cu_num_custom_op(x)
    return CU_NUM


def get_device_name():
    gfx = get_gfx()

    if gfx == "gfx942":
        cu = get_cu_num()
        if cu == 304:
            return "MI300"
        elif cu == 80 or cu == 64:
            return "MI308"
    elif gfx == "gfx950":
        return "MI350"
    else:
        raise RuntimeError("Unsupported gfx")
