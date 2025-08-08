# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess
import torch
from torch_guard import torch_compile_guard
from torch.library import Library

aiter_lib = Library("aiter", "FRAGMENT")

from cpp_extension import executable_path

# Since custom op return int will cause graph break in fullgraph
CU_NUM = 0


@torch_compile_guard()
def get_gfx_custom_op(dummy: torch.Tensor) -> torch.Tensor:
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
                    gfx = line.split(":")[-1].strip()
                    encoded = torch.tensor([ord(c) for c in gfx], dtype=torch.int32)
                    return encoded
        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rocminfo failed {str(e)}")
    elif ";" in gfx:
        gfx = gfx.split(";")[-1]
        encoded = torch.tensor([ord(c) for c in gfx], dtype=torch.int32)
        return encoded


@functools.lru_cache(maxsize=1)
def get_gfx():
    encoded_tensor = get_gfx_custom_op(torch.empty(1, device="cpu"))
    return "".join(chr(c) for c in encoded_tensor.cpu().tolist())


@torch_compile_guard()
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


# _CU_NUM_OP_REGISTERED = False


@functools.lru_cache(maxsize=1)
def get_cu_num():
    x = torch.empty(1, device="cuda")
    get_cu_num_custom_op(x)
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
