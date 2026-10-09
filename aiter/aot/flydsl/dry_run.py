# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run aiter's real host code on CPU, compiling every FlyDSL launch it reaches.

Inside :func:`dry_run`, tensors are FakeTensors on a logical ``cuda:0``, aiter's
device custom ops dispatch to their fakes, and ``COMPILE_ONLY=1`` makes each
``@flyc.jit`` call compile and persist its artifact instead of launching. The
host code that picks kernels, sizes buffers and derives compile-time
parameters is the runtime's own, so the cache keys it produces are the ones
the runtime will look up. Any attempt to reach the GPU runtime raises.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock


class _Stream:
    """Stand-in for ``torch.cuda.current_stream()``; launches never happen."""

    cuda_stream = 0

    def __init__(self, device):
        self.device = device


def _forbid(*_args, **_kwargs):
    raise RuntimeError("FlyDSL AOT dry run reached the GPU runtime")


def _torch_guard_modules() -> list:
    # jit/core.py imports torch_guard through its own sys.path entry, so the
    # module is loaded twice and each copy decorates its own ops.
    modules = []
    for name in ("torch_guard", "aiter.jit.utils.torch_guard"):
        module = sys.modules.get(name)
        if module is not None and module not in modules:
            modules.append(module)
    return modules


def _register_stream_stub() -> None:
    from flydsl.compiler.jit_argument import JitArgumentRegistry
    from flydsl.expr.typing import Stream

    if _Stream not in JitArgumentRegistry.registry:
        JitArgumentRegistry.register(_Stream)(Stream)


@contextmanager
def dry_run(arch: str, cu_num: int) -> Iterator[None]:
    """Make the current process look like an ``arch`` GPU with ``cu_num`` CUs."""
    import torch
    from flydsl.compiler.jit_executor import CompiledArtifact
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from aiter.jit import core as jit_core
    from aiter.jit.utils import chip_info

    _register_stream_stub()
    stream = _Stream(torch.device("cuda", 0))
    properties = SimpleNamespace(
        name=arch, gcnArchName=arch, multi_processor_count=cu_num
    )
    real_data_ptr = torch.Tensor.data_ptr

    def data_ptr(tensor):
        if not isinstance(tensor, FakeTensor):
            return real_data_ptr(tensor)
        if tensor.numel() == 0:
            return 0
        # Storage identity keeps aliasing visible to host code comparing
        # pointers; the address is never dereferenced.
        return (
            tensor.untyped_storage()._cdata
            + tensor.storage_offset() * tensor.element_size()
        )

    convert, tensor_cls, _, _ = jit_core._pybind_develop_hooks()
    cached = (
        chip_info.get_gfx,
        chip_info.get_gfx_custom_op_core,
        chip_info.get_gfx_runtime,
        chip_info.get_cu_num,
    )
    env = {
        "COMPILE_ONLY": "1",
        "FLYDSL_GPU_ARCH": arch,
        "ARCH": arch,
        "GPU_ARCHS": arch,
        "CU_NUM": str(cu_num),
    }
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, env))
        for function in cached:
            function.cache_clear()
            stack.callback(function.cache_clear)
        for owner, name, value in (
            (torch.cuda, "_lazy_init", _forbid),
            (CompiledArtifact, "_ensure_engine", _forbid),
            (torch.cuda, "current_stream", lambda *_a, **_k: stream),
            (torch.cuda, "current_device", lambda: 0),
            (torch.cuda, "get_device_properties", lambda *_a, **_k: properties),
            (torch.cuda, "synchronize", lambda *_a, **_k: None),
            (torch.Tensor, "data_ptr", data_ptr),
            (chip_info, "_detect_native", lambda: [arch]),
            (
                jit_core,
                "_pybind_develop_hooks_cache",
                (convert, tensor_cls, lambda _index: 0, lambda: 0),
            ),
        ):
            stack.enter_context(mock.patch.object(owner, name, value))
        for module in _torch_guard_modules():
            previous = module.run_host_scalar_ops_for_real(True)
            stack.callback(module.run_host_scalar_ops_for_real, previous)
        stack.enter_context(FakeTensorMode(allow_non_fake_inputs=True))
        yield
