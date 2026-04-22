#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Callable, Iterator


def job_identity(job: dict[str, Any]) -> tuple:
    return tuple(sorted(job.items()))


def dedupe_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_jobs = []
    seen = set()
    for job in jobs:
        key = job_identity(job)
        if key in seen:
            continue
        seen.add(key)
        unique_jobs.append(job)
    return unique_jobs


def collect_aot_jobs(
    csv_paths: list[str],
    parse_csv: Callable[[str], list[dict[str, Any]]],
    on_missing_csv: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    jobs = []
    for csv_path in csv_paths:
        if os.path.isfile(csv_path):
            jobs.extend(parse_csv(csv_path))
        elif on_missing_csv is not None:
            on_missing_csv(csv_path)
    return dedupe_jobs(jobs)


@contextmanager
def compile_only_env() -> Iterator[None]:
    prev = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev


def aot_compile_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aot_compile_stream(device=None):
    import torch
    from flydsl.expr.typing import Stream

    if torch.cuda.is_available():
        return torch.cuda.current_stream(device=device)
    return Stream(0)


class CompileOnlyTensorArg:
    """CPU tensor wrapper that preserves FlyDSL compile-only metadata paths.

    FlyDSL's default TensorAdaptor always exports DLPack with ``stream=-1``,
    which is correct for GPU tensors but invalid for CPU tensors.  This wrapper
    switches CPU tensors to the CPU DLPack path so ``COMPILE_ONLY=1`` can build
    launcher signatures without requiring a visible HIP device.
    """

    def __init__(
        self,
        tensor,
        assumed_align: int | None = None,
        use_32bit_stride: bool = False,
    ):
        import torch

        float8_dtypes = tuple(
            dt
            for dt in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e4m3fnuz", None),
                getattr(torch, "float8_e5m2fnuz", None),
            )
            if dt is not None
        )

        dlpack_tensor = tensor.detach() if tensor.requires_grad else tensor
        if float8_dtypes and dlpack_tensor.dtype in float8_dtypes:
            dlpack_tensor = dlpack_tensor.view(torch.uint8)

        self._tensor = tensor
        self._dlpack_tensor = dlpack_tensor
        self._tensor_adaptor = None
        self.assumed_align = assumed_align
        self.use_32bit_stride = use_32bit_stride

    def _ensure_tensor_adaptor(self):
        if self._tensor_adaptor is not None:
            return self._tensor_adaptor

        from flydsl.compiler.jit_argument import DLTensorAdaptor

        if self._dlpack_tensor.device.type == "cpu":
            dlpack_capsule = self._dlpack_tensor.__dlpack__()
        else:
            dlpack_capsule = self._dlpack_tensor.__dlpack__(stream=-1)
        self._tensor_adaptor = DLTensorAdaptor(
            dlpack_capsule, self.assumed_align, self.use_32bit_stride
        )
        self._tensor_adaptor.build_memref_desc()
        return self._tensor_adaptor

    def __fly_types__(self):
        return [self._ensure_tensor_adaptor().get_memref_type()]

    def __fly_ptrs__(self):
        return self._ensure_tensor_adaptor().get_c_pointers()

    def __cache_signature__(self):
        return (
            self._tensor.dtype,
            self.assumed_align,
            self.use_32bit_stride,
        )


def _register_compile_only_tensor_arg() -> None:
    from flydsl.compiler.jit_argument import JitArgumentRegistry
    from flydsl.expr.typing import Tensor

    try:
        JitArgumentRegistry.register_jit_arg(CompileOnlyTensorArg, Tensor)
    except ValueError:
        pass


_register_compile_only_tensor_arg()


def prepare_compile_arg(arg):
    import torch

    if isinstance(arg, torch.Tensor) and arg.device.type == "cpu":
        return CompileOnlyTensorArg(arg)
    return arg


def prepare_compile_args(args):
    return tuple(prepare_compile_arg(arg) for arg in args)


def _jit_cache_hit(exe, *args, **kwargs) -> bool:
    ensure_sig = getattr(exe, "_ensure_sig", None)
    ensure_cache_manager = getattr(exe, "_ensure_cache_manager", None)
    make_cache_key = getattr(exe, "_make_cache_key", None)
    cache_key_to_str = getattr(exe, "_cache_key_to_str", None)
    sig = getattr(exe, "_sig", None)
    mem_cache = getattr(exe, "_mem_cache", None)
    cache_manager = getattr(exe, "cache_manager", None)

    if (
        ensure_sig is None
        or ensure_cache_manager is None
        or make_cache_key is None
        or cache_key_to_str is None
    ):
        return False

    ensure_sig()
    ensure_cache_manager()
    sig = getattr(exe, "_sig", sig)
    cache_manager = getattr(exe, "cache_manager", cache_manager)
    mem_cache = getattr(exe, "_mem_cache", mem_cache)
    if sig is None:
        return False

    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    cache_key = make_cache_key(bound.arguments)
    if mem_cache is not None and cache_key in mem_cache:
        return True

    str_key = cache_key_to_str(cache_key)
    return bool(cache_manager is not None and str_key in cache_manager)


def compile_executable_to_cache(exe, *args, **kwargs):
    prepared_args = prepare_compile_args(args)
    prepared_kwargs = {k: prepare_compile_arg(v) for k, v in kwargs.items()}

    if _jit_cache_hit(exe, *prepared_args, **prepared_kwargs):
        print("[flydsl] COMPILE_ONLY=1, cache hit")
        return None

    with compile_only_env():
        return exe(*prepared_args, **prepared_kwargs)
