# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Scoped host environment for MXMOE tests that never execute GPU kernels."""

import contextlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _clear_chip_info_caches():
    for name in ("chip_info", "aiter.jit.utils.chip_info"):
        module = sys.modules.get(name)
        if module is not None:
            for value in vars(module).values():
                if callable(value) and hasattr(value, "cache_clear"):
                    value.cache_clear()


def get_configured_default_device():
    # The public query may allocate to resolve an unindexed CUDA default.
    # Prefer its raw setting when available, then guard the public fallback.
    global_context = getattr(torch, "_GLOBAL_DEVICE_CONTEXT", None)
    if global_context is not None and hasattr(global_context, "device_context"):
        device_context = global_context.device_context
        if device_context is None:
            return torch.device("cpu")
        if hasattr(device_context, "device"):
            return torch.device(device_context.device)
    getter = getattr(torch, "get_default_device", None)
    if not callable(getter):
        raise RuntimeError(  # noqa: TRY004 - unsupported test environment
            "MXMOE CPU tests cannot safely read the default device"
        )
    with patch.object(
        torch.cuda,
        "_lazy_init",
        side_effect=RuntimeError("default device query requires GPU initialization"),
    ):
        try:
            return torch.device(getter())
        except Exception as exc:
            raise RuntimeError(
                "MXMOE CPU tests cannot safely read the default device with this "
                "torch version; no device setting was changed"
            ) from exc


@contextlib.contextmanager
def cpu_tuner_environment():
    previous_device = get_configured_default_device()
    run, popen = subprocess.run, subprocess.Popen

    def host_run(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and Path(command[0]).name in (
            "rocminfo",
            "hipinfo",
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                "Agent 1\n  Name: gfx950\n  Device Type: GPU\n"
                "  Compute Unit: 256\n  ASIC Revision: 1\n",
                "",
            )
        return run(command, *args, **kwargs)

    def host_popen(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and Path(command[0]).name in {
            "hipcc",
            "clang",
            "clang++",
            "gcc",
            "g++",
            "c++",
            "ninja",
            "cmake",
            "make",
        }:
            raise AssertionError("MXMOE CPU tests must not compile extensions")
        return popen(command, *args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.callback(torch.set_default_device, previous_device)
        # Architecture probes cache their result. Do not leave synthetic host
        # metadata behind for real GPU tests later in the same process.
        _clear_chip_info_caches()
        stack.callback(_clear_chip_info_caches)
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "GPU_ARCHS": "gfx950",
                    "CU_NUM": "256",
                    "AITER_REBUILD": "0",
                    # Gluon kernels are never executed here. Keep this opt-in
                    # scoped to the CPU tests, including older system Triton.
                    "AITER_USE_SYSTEM_TRITON": "1",
                },
            )
        )
        stack.enter_context(patch.object(subprocess, "run", host_run))
        stack.enter_context(patch.object(subprocess, "Popen", host_popen))
        stack.enter_context(
            patch.object(
                torch.cuda,
                "_lazy_init",
                side_effect=AssertionError("MXMOE CPU tests must not initialize GPUs"),
            )
        )
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=0))
        stack.enter_context(patch.object(torch.cuda, "current_device", return_value=0))
        stack.enter_context(
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=256),
            )
        )
        torch.set_default_device("cpu")
        jit = importlib.import_module("aiter.jit.core")
        stack.enter_context(
            patch.object(
                jit,
                "build_module",
                side_effect=AssertionError("MXMOE CPU tests must not build extensions"),
            )
        )
        yield
