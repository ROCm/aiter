# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Shared utilities for chunk_delta_attn kernels."""

import functools
import inspect
import math
import os

import torch
import triton

from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

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


def chunk_delta_attn_tuned_config(
    kernel_name: str, fallback: triton.Config
) -> triton.Config:
    """This family's tile for the current device, from its published config."""
    return get_tuned_kernel_config(
        "attention", "CHUNK_DELTA_ATTN", kernel_name, fallback
    )


RCP_LN2: float = math.log2(math.e)  # 1/ln(2), for log2-space gate arithmetic


def _same_arg(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return a is b
    return type(a) is type(b) and a == b


def _same_call(prev_args, prev_kwargs, args, kwargs) -> bool:
    return (
        len(args) == len(prev_args)
        and kwargs.keys() == prev_kwargs.keys()
        and all(_same_arg(a, b) for a, b in zip(args, prev_args, strict=True))
        and all(_same_arg(v, prev_kwargs[k]) for k, v in kwargs.items())
    )


def tensor_cache(fn):
    """Cache the single most recent result of a function taking tensors.

    Tensor arguments match on identity, not contents: reading contents would
    need the device-to-host copy this exists to avoid. A caller that rebuilds
    an equal tensor therefore misses, which is fine for the hit that matters --
    one ``cu_seqlens`` shared by every layer of a forward pass. Mutating a
    cached tensor in place is not detected.
    """
    last: list = []

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if last and _same_call(last[0], last[1], args, kwargs):
            return last[2]
        result = fn(*args, **kwargs)
        last[:] = [args, kwargs, result]
        return result

    return wrapper


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


import os
from collections.abc import Callable
from typing import Any

import triton.language as tl
import triton.language.extra.libdevice as tldevice

# The fp32 cast lives inside the wrapper, as it does in fla. Every call site
# today already passes fp32, but a bf16 gate reaching a bare tl.math.exp2 would
# exponentiate at bf16 precision and diverge from fla with nothing to flag it.
if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":

    @triton.jit
    def exp(x):
        return tldevice.fast_expf(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tldevice.exp2(x.to(tl.float32))

else:

    @triton.jit
    def exp(x):
        return tl.exp(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tl.math.exp2(x.to(tl.float32))


@triton.jit
def softplus(x):
    """log(1 + exp(x)), falling back to the identity above x=20.

    The two agree to fp32 precision past x=20, and the switch keeps exp(x) from
    overflowing to inf around x=89. That matters more than it looks: the gate
    this feeds is cumulatively summed, so a single overflowing token turns every
    later cumsum into -inf and the gate differences into NaN, wiping out the rest
    of the sequence and the recurrent state carried out of it.
    """
    return tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x)


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
