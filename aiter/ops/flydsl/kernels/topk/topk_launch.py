# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch of the FlyDSL TopK launchers."""

import copy

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

# A JitFunction's per-device state, each with the value it starts from: its
# in-process caches, its own and its disk-cache manager's, which FlyDSL keys
# without the device -- a build loaded on one GPU cannot launch on another --
# and the GPU target it resolves once, with the signature, on its first call.
_PER_DEVICE_STATE = {
    "_mem_cache": dict,
    "_call_state_cache": dict,
    "_last_compiled": lambda: None,
    "cache_manager": lambda: None,
    "manager_key": lambda: None,
    "_sig": lambda: None,
    "_backend_target": lambda: None,
}


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
    per-device state starts afresh, its builds reloading from disk.

    Raises rather than share `exe` with a further GPU when this FlyDSL lacks a
    piece of that state: the copy could not be told apart from the original.
    """
    by_device = exe.__dict__.get("_topk_by_device")
    if by_device is None:
        by_device = exe._topk_by_device = {}
    own = by_device.get(device)
    if own is None:
        if not by_device:
            own = exe
        else:
            missing = [n for n in _PER_DEVICE_STATE if not hasattr(exe, n)]
            if missing:
                raise RuntimeError(
                    f"TopK launcher {getattr(exe, '__name__', type(exe).__name__)} "
                    f"cannot get a launcher of its own on cuda:{device}: this "
                    f"FlyDSL has no {', '.join(missing)}, so the one built on "
                    "another GPU would be reused"
                )
            own = copy.copy(exe)
            for n in ("_topk_by_device", "_cf", "_cf_by_specialization"):
                own.__dict__.pop(n, None)
            for n, fresh in _PER_DEVICE_STATE.items():
                setattr(own, n, fresh())
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
