# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Shared utilities for chunk_delta_attn kernels."""

import inspect
import math
import os

import torch
import triton

SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)
_FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"
autotune_cache_kwargs: dict = (
    {"cache_results": _FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
)

CHUNK_DELTA_ATTN_TRITON_AUTOTUNE: bool = os.getenv(
    "CHUNK_DELTA_ATTN_TRITON_AUTOTUNE", "0"
).lower() in ("1", "true", "yes", "on")


def chunk_delta_attn_autotune_configs(
    configs: list,
    default_config=None,
) -> list:
    """Return configs for @triton.autotune."""
    if CHUNK_DELTA_ATTN_TRITON_AUTOTUNE:
        return configs
    return [default_config if default_config is not None else configs[0]]


RCP_LN2: float = math.log2(math.e)  # 1/ln(2), for log2-space gate arithmetic


def _get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except (ImportError, RuntimeError):
        return "cpu"


_device_platform = _get_available_device()

IS_TF32_SUPPORTED: bool = (
    _device_platform == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
)
IS_GATHER_SUPPORTED: bool = hasattr(triton.language, "gather")


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """Return True if the device has enough shared memory for large tile configs."""
    try:
        props = torch.cuda.get_device_properties(tensor_idx)
        gc_arch = getattr(props, "gcnArchName", "").split(":")[0]
        _LARGE_SHMEM = {"gfx95", "gfx94", "gfx90"}
        if any(gc_arch.startswith(a) for a in _LARGE_SHMEM):
            return True
        if arch == "ampere":
            cap = torch.cuda.get_device_capability(tensor_idx)
            return cap[0] >= 8
        return False
    except (ImportError, RuntimeError):
        return False


import functools
import os
from collections.abc import Callable
from typing import Any

import triton.language as tl
import triton.language.extra.libdevice as tldevice

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
else:
    exp = tl.exp
    exp2 = tl.math.exp2


@triton.jit
def softplus(x):
    """Numerically-stable softplus: log(1 + exp(x))."""
    return tl.log(1.0 + tl.exp(x))


def input_guard(fn: Callable) -> Callable:
    """Ensure all tensor arguments are contiguous before kernel launch."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)

    return wrapper
