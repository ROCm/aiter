# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU unit tests for FlyDSL helpers (no GPU required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_UTILS_PATH = (
    Path(__file__).resolve().parents[2] / "aiter" / "ops" / "flydsl" / "utils.py"
)


def _load_utils():
    spec = importlib.util.spec_from_file_location("flydsl_utils_under_test", _UTILS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lds_limit_follows_aot_arch_not_live_device(monkeypatch):
    """LDS budget must come from the arch we compile for, not the GPU we build on.

    A multi-arch wheel (GPU_ARCHS='gfx942;gfx950') compiles gfx950 8-wave tiles
    even on a gfx942 host. Reading the host's 64 KiB limit rejected tiles that
    fit gfx950's 160 KiB, aborting FlyDSL AOT before any wheel was written.
    """

    class Props:
        shared_memory_per_block = 65536
        gcnArchName = "gfx942"

    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Props())

    flydsl_utils = _load_utils()
    flydsl_utils._default_cuda_device_index.cache_clear()
    flydsl_utils._get_shared_memory_per_block_cached.cache_clear()

    assert flydsl_utils.get_shared_memory_per_block(fallback_gfx="gfx950") == 163840
