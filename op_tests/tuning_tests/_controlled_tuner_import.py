# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Load the real tuner toolchain under offline, synthetic discovery metadata.

Only import-time rocminfo/hipinfo discovery is simulated. Runtime device support
tests still call the real tuner identity/preflight against controlled properties.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import torch


def import_moe_tuner():
    run = subprocess.run
    set_default_device = torch.set_default_device

    def discover(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and Path(command[0]).name in (
            "rocminfo",
            "hipinfo",
        ):
            output = (
                "Agent 1\n  Name: gfx950\n  Device Type: GPU\n"
                "  Compute Unit: 256\n  ASIC Revision: 1\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        return run(command, *args, **kwargs)

    with (
        mock.patch.dict(
            os.environ,
            {
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
                "GPU_ARCHS": "gfx950",
                "CU_NUM": "256",
            },
        ),
        mock.patch.object(subprocess, "run", side_effect=discover),
        mock.patch.object(
            torch,
            "set_default_device",
            side_effect=lambda device: set_default_device("cpu"),
        ),
    ):
        tuner = importlib.import_module(
            "csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune"
        )
    # Synthetic import discovery must not become runtime evidence.
    for name in ("chip_info", "aiter.jit.utils.chip_info"):
        module = sys.modules.get(name)
        if module is not None:
            for function in (
                "_gpu_info_output",
                "_detect_native",
                "get_gfx_runtime",
                "get_gfx",
                "get_gfx_custom_op_core",
                "_rocminfo_gpu_agents",
                "get_gfx_list",
            ):
                clear = getattr(getattr(module, function), "cache_clear", None)
                if clear is not None:
                    clear()
    return tuner
