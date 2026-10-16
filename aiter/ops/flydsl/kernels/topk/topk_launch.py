# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch of the FlyDSL TopK launchers."""

import copy

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

# A JitFunction's in-process caches. FlyDSL keys them -- its own and its
# disk-cache manager's -- without the device, and a build loaded on one GPU
# cannot launch on another.
_PER_DEVICE_STATE = (
    "_mem_cache",
    "_call_state_cache",
    "_last_compiled",
    "cache_manager",
)


def _launch_key(args):
    """What FlyDSL's build key reads from each tensor argument: its dtype, rank
    and first axis of stride 1. FlyDSL holds that stride static and every
    extent and other stride dynamic, so a (64, 1) tensor, strides (1, 1), builds
    a kernel that reads rows at a pitch of 1."""
    key = ()
    for a in args:
        if isinstance(a, torch.Tensor):
            s = a.stride()
            try:
                key = (key, a.dtype, len(s), s.index(1))
            except ValueError:
                key = (key, a.dtype, len(s), -1)
    return key


def _launcher_on(exe, device):
    """`exe` on the first GPU it runs on; on each further GPU, a copy of it whose
    in-process caches start empty and reload from disk."""
    by_device = exe.__dict__.get("_topk_by_device")
    if by_device is None:
        by_device = exe._topk_by_device = {}
    own = by_device.get(device)
    if own is None:
        if not by_device or not all(hasattr(exe, n) for n in _PER_DEVICE_STATE):
            own = exe
        else:
            own = copy.copy(exe)
            for n in ("_topk_by_device", "_cf", "_cf_by_specialization"):
                own.__dict__.pop(n, None)
            own._mem_cache, own._call_state_cache, own._last_compiled = {}, {}, None
            own.cache_manager, own.manager_key = None, None
            if hasattr(exe, "_extern_linkage_keys"):
                own._extern_linkage_keys = set(exe._extern_linkage_keys)
        by_device[device] = own
    return own


def run_topk_launcher(exe, *args):
    """`_run_compiled`, with one compiled kernel per GPU and per `_launch_key`.

    `_run_compiled` alone keeps a launcher's first build for every later call,
    which skips the parts of FlyDSL's key that change between TopK calls.
    """
    _run_compiled(
        _launcher_on(exe, torch._C._cuda_getDevice()),
        *args,
        specialization_key=_launch_key(args),
    )
