# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch of the FlyDSL TopK launchers."""

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _launcher_on, _run_compiled

__all__ = ["_launcher_on", "run_topk_launcher"]


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


def run_topk_launcher(exe, *args):
    """`_run_compiled`, with one compiled kernel per `_launch_key`.

    `_run_compiled` alone keeps a launcher's first build for every later call,
    which skips the parts of FlyDSL's key that change between TopK calls.
    """
    _run_compiled(exe, *args, specialization_key=_launch_key(args))
