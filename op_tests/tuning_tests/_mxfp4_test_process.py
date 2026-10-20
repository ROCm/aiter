# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Process guard shared by the MXFP4 tuner test modules.

Importing the tuner with GPUs hidden pins state for the whole process: HIP
enumerates no device, ``aiter.dtypes.fp8`` degrades to ``uint8`` and FlyDSL
caches a fallback arch. The modules therefore run only in a pytest process
started with every GPU hidden, and skip elsewhere instead of polluting it.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

GPU_VISIBILITY = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
COMMAND = (
    "ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= "
    "python3 -m pytest op_tests/tuning_tests/test_mxfp4_tuner.py "
    "op_tests/tuning_tests/test_mxfp4_staged.py "
    "op_tests/tuning_tests/test_mxfp4_aot_configuration.py"
)


def require_gpu_hidden_process() -> "None":
    """Skip the calling module unless this process never saw a GPU."""
    visible = [name for name in GPU_VISIBILITY if os.environ.get(name) != ""]
    if visible:
        reason = f"{', '.join(visible)} not set to empty"
    # Checked only once hidden: the first call initializes HIP with this env.
    elif torch._C._cuda_getDeviceCount() > 0:
        reason = "HIP enumerated GPUs before they were hidden"
    else:
        return
    pytest.skip(
        f"MXFP4 tuner tests need their own GPU-hidden process ({reason}); "
        f"run: {COMMAND}",
        allow_module_level=True,
    )


def default_device() -> "torch.device | None":
    # torch.get_default_device() would initialize CUDA for a "cuda" default.
    context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
    return None if context is None else context.device


@contextmanager
def use_default_device(device: "str | None") -> "Iterator[None]":
    state = torch._GLOBAL_DEVICE_CONTEXT
    had_context = hasattr(state, "device_context")
    previous = default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous)
        # set_default_device(None) leaves device_context=None, which breaks
        # torch.get_default_device(); restore the original absent attribute.
        if not had_context:
            state.__dict__.pop("device_context", None)
