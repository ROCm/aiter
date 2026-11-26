# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""When FP8 GEMM and per-token quant entry points may skip their custom ops.

These guards serve the FP8 GEMM and per-token quant fast paths. Tuned BF16
GEMM uses its ordinary registered-op path and does not use this helper.

Outside torch.compile some entry points make the C++ call their custom op
would make, without the torch.ops dispatch in between. `plain_eager` says
when that is the same thing: no dispatch mode, torch-function mode, functorch
transform or jit trace is active, no input intercepts ops, and autograd would
record nothing. Callers check ``torch.compiler.is_compiling()`` first, so
dynamo folds the fast path away before reaching here. An entry point that
stands in for a routed op also checks `unrouted`.
"""

import torch

from aiter.plugin import _config as _plugin_config

_tensor_types: dict[type, bool] = {torch.Tensor: True, torch.nn.Parameter: True}


def _plain_tensor_type(cls: type) -> bool:
    """Whether tensors of type ``cls`` may skip the dispatcher.

    A subclass that takes over ``__torch_dispatch__`` (FakeTensor, functional
    and wrapper tensors) has to see the op. One that only overrides
    ``__torch_function__`` is caught per call by ``torch._C._has_torch_function``.
    """
    ok = _tensor_types.get(cls)
    if ok is None:
        ok = issubclass(cls, torch.Tensor) and (
            cls.__torch_dispatch__ is torch.Tensor.__torch_dispatch__
        )
        _tensor_types[cls] = ok
    return ok


def unrouted() -> bool:
    """Whether no kernel plugin can serve any routed op: ``AITER_PLUGINS=0``,
    or nothing enabled by `aiter.plugin.configure`. That is when a routed op's
    dispatch runs its native body and records nothing, so a call may skip it."""
    config = _plugin_config()
    return config.kill_switch or not config.enabled


def plain_eager(tensors: tuple) -> bool:
    """Whether a call on ``tensors`` (None entries ignored) runs exactly as the
    custom op's own implementation would, so that implementation may be called
    directly."""
    if (
        torch._C._len_torch_dispatch_stack()
        or torch._C._are_functorch_transforms_active()
        or torch._C._is_tracing()
        or torch._C._has_torch_function(tensors)
    ):
        return False
    known = _tensor_types.get
    grad = torch.is_grad_enabled()
    for t in tensors:
        if t is None:
            continue
        if not (known(type(t)) or _plain_tensor_type(type(t))):
            return False
        if grad and t.requires_grad:
            return False
    return True
