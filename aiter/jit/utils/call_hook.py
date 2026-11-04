# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Callbacks on every aiter op call, for tools that record which ops a run
makes and with what arguments.

A callback is ``fn(name, depth, args, kwargs)``: ``name`` is the op's name,
``depth`` the number of aiter ops the call is nested in (0 for a call from
outside aiter, i.e. a production entry point), ``args``/``kwargs`` the call's
arguments. It runs before the op, on the calling thread, and must neither
launch device work nor keep the arguments alive.

The hook sits in the body `torch_compile_guard` registers as the custom op's
implementation, so it fires on every eager call and on every call a compiled
graph makes at runtime; while Dynamo traces, it does not. Under a CUDA graph
capture it fires once, when the op is recorded, not on each replay. With no
callback registered an op pays one truthiness test.
"""

import functools
import threading
from collections.abc import Callable
from typing import Any

_hooks: list[Callable[[str, int, tuple, dict], None]] = []
_local = threading.local()


def register_call_hook(fn: Callable[[str, int, tuple, dict], None]) -> None:
    if fn not in _hooks:
        _hooks.append(fn)


def remove_call_hook(fn: Callable[[str, int, tuple, dict], None]) -> None:
    if fn in _hooks:
        _hooks.remove(fn)


def observe(fn: Callable | None = None, *, name: str | None = None):
    """Show the callbacks a plain Python entry point -- one no custom op
    registers, e.g. a Triton kernel's launcher -- as they see a registered op.
    Under ``name`` (default: the function's own), at the depth it is called.

    Usable as ``@observe`` or ``@observe(name=...)``. With no callback
    registered a call pays one truthiness test."""

    def wrap(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        def observed(*args, **kwargs):
            if _hooks:
                return call_with_hooks(op_name, func, args, kwargs)
            return func(*args, **kwargs)

        return observed

    return wrap(fn) if fn is not None else wrap


def call_with_hooks(name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
    import torch

    if torch.compiler.is_compiling():
        return func(*args, **kwargs)
    depth = getattr(_local, "depth", 0)
    for fn in tuple(_hooks):
        fn(name, depth, args, kwargs)
    _local.depth = depth + 1
    try:
        return func(*args, **kwargs)
    finally:
        _local.depth = depth
