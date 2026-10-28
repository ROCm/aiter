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
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock


class _Stream:
    """Stand-in for a ``torch.cuda`` stream; launches never happen."""

    cuda_stream = 0

    def __init__(self, device=None, priority=0):
        self.device = device


class _Event:
    """Stand-in for ``torch.cuda.Event``; nothing is ever recorded."""

    def __init__(self, *_args, **_kwargs):
        pass


class _Device:
    """Stand-in for ``torch.cuda.device``: entering it selects nothing. A class,
    since torch tests ``isinstance(device, torch.cuda.device)``."""

    def __init__(self, device=None):
        self.idx = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class GpuRuntimeReached(RuntimeError):
    """Host code under :func:`dry_run` reached the GPU runtime."""


def _forbid(*_args, **_kwargs):
    raise GpuRuntimeReached("FlyDSL AOT dry run reached the GPU runtime")


# The index items of basic indexing, which answers a view of its operand.
_VIEW_INDEX = (int, slice, type(None), type(Ellipsis))


class _TritonDriver:
    """Stand-in for Triton's active driver: answers the target queries that
    modules make at import (e.g. picking autotune configs per backend). A
    Triton launch is another backend's, which the dry run does not carry out."""

    def __init__(self, arch: str):
        from triton.backends.compiler import GPUTarget

        self._target = GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64)

    def get_current_target(self):
        return self._target

    def get_current_device(self):
        return 0

    def __getattr__(self, name):
        raise RuntimeError(f"Triton {name} is outside the FlyDSL AOT dry run")


def _triton_driver_patches(arch: str) -> list:
    try:
        from triton.runtime.driver import driver
    except ImportError:
        return []
    return [(driver, "_active", _TritonDriver(arch))]


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
def dry_run(
    arch: str, cu_num: int, name: str | None = None, *, python_ops: bool = False
) -> Iterator[None]:
    """Make the current process look like an ``arch`` GPU with ``cu_num`` CUs.
    ``name`` is the device name host code reads, e.g. to pick a per-GPU-model
    tuning table; it defaults to ``arch``. With ``python_ops``, aiter's
    registered ops that have a Python body run it instead of their fake, so a
    replay can enter at a public op (e.g. ``gemm_a8w8_bpreshuffle``)."""
    import torch
    from flydsl.compiler.jit_executor import CompiledArtifact
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from aiter.jit import core as jit_core
    from aiter.jit.utils import chip_info
    from aiter.utility import graph_alloc

    _register_stream_stub()
    stream = _Stream(torch.device("cuda", 0))
    properties = SimpleNamespace(
        name=name or arch, gcnArchName=arch, multi_processor_count=cu_num
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

    real_getitem = torch.Tensor.__getitem__

    def getitem(tensor, index):
        """``tensor[index]``, which torch takes a device guard for before it
        dispatches: with the GPUs hidden that guard initializes HIP and
        raises, while every other op reaches the fake. Basic indexing is
        replayed on a meta tensor of the same layout and taken again of this
        one, so the view keeps the storage and offset ``data_ptr`` reads."""
        if not isinstance(tensor, FakeTensor) or tensor.device.type != "cuda":
            return real_getitem(tensor, index)
        items = index if isinstance(index, tuple) else (index,)
        if all(isinstance(item, _VIEW_INDEX) for item in items):
            layout = torch.empty_strided(
                tensor.shape, tensor.stride(), dtype=tensor.dtype, device="meta"
            )
            view = real_getitem(layout, index)
            return tensor.as_strided(
                view.shape,
                view.stride(),
                tensor.storage_offset() + view.storage_offset(),
            )
        if all(item is None or isinstance(item, torch.Tensor) for item in items):
            # Advanced indexing copies; it is this op, which dispatches
            # without the guard.
            return torch.ops.aten.index(tensor, list(items))
        return real_getitem(tensor, index)

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
        for owner, attribute, value in (
            (torch.cuda, "_lazy_init", _forbid),
            (CompiledArtifact, "_ensure_engine", _forbid),
            (torch.cuda, "current_stream", lambda *_a, **_k: stream),
            (torch.cuda, "Stream", _Stream),
            (torch.cuda, "Event", _Event),
            (torch.cuda, "current_device", lambda: 0),
            (torch.cuda, "device", _Device),
            (torch.cuda, "get_device_properties", lambda *_a, **_k: properties),
            (torch.cuda, "synchronize", lambda *_a, **_k: None),
            # Scratch buffers that outlive a graph capture come from a MemPool
            # of their own; a FakeTensor needs none.
            (torch.cuda, "use_mem_pool", lambda *_a, **_k: nullcontext()),
            (torch.cuda, "is_current_stream_capturing", lambda: False),
            (graph_alloc, "_persistent_pool", lambda _index: None),
            (torch.Tensor, "data_ptr", data_ptr),
            (torch.Tensor, "__getitem__", getitem),
            (torch.Tensor, "record_stream", lambda *_a, **_k: None),
            (chip_info, "_detect_native", lambda: [arch]),
            (
                jit_core,
                "_pybind_develop_hooks_cache",
                (convert, tensor_cls, lambda _index: 0, lambda: 0),
            ),
            # Only host code of a C++ module runs, which any arch's build has.
            # Rebuilding one for this arch would pull it from under replays of
            # other archs that share it.
            (jit_core, "_needs_arch_rebuild", lambda _md_name: False),
            *_triton_driver_patches(arch),
        ):
            stack.enter_context(mock.patch.object(owner, attribute, value))
        for module in _torch_guard_modules():
            previous = module.run_host_scalar_ops_for_real(True)
            stack.callback(module.run_host_scalar_ops_for_real, previous)
            if python_ops:
                previous = module.run_python_ops_for_real(True)
                stack.callback(module.run_python_ops_for_real, previous)
        stack.enter_context(FakeTensorMode(allow_non_fake_inputs=True))
        yield
