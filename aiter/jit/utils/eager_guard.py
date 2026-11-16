# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""When an aiter entry point may skip its custom op.

Outside torch.compile some entry points make the C++ call their custom op
would make, without the torch.ops dispatch in between. `plain_eager` says
when that is the same thing: no dispatch mode, torch-function mode, functorch
transform or jit trace is active, no input intercepts ops, and autograd would
record nothing. Callers check ``torch.compiler.is_compiling()`` first, so
dynamo folds the fast path away before reaching here.
"""

import torch

_tensor_types: dict[type, bool] = {torch.Tensor: True, torch.nn.Parameter: True}

# Private probes; without any of them nothing is plain.
_dispatch_modes = getattr(torch._C, "_len_torch_dispatch_stack", None)
_functorch_active = getattr(torch._C, "_are_functorch_transforms_active", None)
_jit_tracing = getattr(torch._C, "_is_tracing", None)
probes_available = None not in (_dispatch_modes, _functorch_active, _jit_tracing)


def plain_tensor_type(cls: type) -> bool:
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


def plain_eager(tensors: tuple) -> bool:
    """Whether a call on ``tensors`` (None entries ignored) runs exactly as the
    custom op's own implementation would, so that implementation may be called
    directly."""
    if (
        not probes_available
        or _dispatch_modes()
        or _functorch_active()
        or _jit_tracing()
        or torch._C._has_torch_function(tensors)
    ):
        return False
    known = _tensor_types.get
    grad = torch.is_grad_enabled()
    for t in tensors:
        if t is None:
            continue
        if not (known(type(t)) or plain_tensor_type(type(t))):
            return False
        if grad and t.requires_grad:
            return False
    return True
