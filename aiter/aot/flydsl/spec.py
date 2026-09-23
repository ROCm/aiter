# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Declarative FlyDSL AOT compile specifications.

This mirrors the useful part of FlashInfer's JIT/AOT design: the build driver
works with small, importable specs instead of knowing how every operator finds
and compiles its kernels.  FlyDSL remains responsible for the artifact format,
cache key, persistence, and runtime loading.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class AotSpec:
    """One FlyDSL operator family's AOT registration.

    Attribute names are stored instead of callable objects so specs remain
    lightweight and pickle cleanly for ``spawn`` workers.  Importing a spec
    module is deferred until jobs are collected or one job is compiled.
    """

    name: str
    module: str
    jobs_factory: str = "get_aot_jobs"
    compiler: str = "compile_one_config"

    def __post_init__(self) -> None:
        for field_name in ("name", "module", "jobs_factory", "compiler"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def _resolve(self, attribute: str):
        module = importlib.import_module(self.module)
        try:
            value = getattr(module, attribute)
        except AttributeError as error:
            raise AttributeError(
                f"FlyDSL AOT spec {self.name!r}: {self.module!r} does not "
                f"define {attribute!r}"
            ) from error
        if not callable(value):
            raise TypeError(
                f"FlyDSL AOT spec {self.name!r}: "
                f"{self.module}.{attribute} must be callable"
            )
        return value

    def collect_jobs(self) -> list[dict[str, Any]]:
        jobs = self._resolve(self.jobs_factory)()
        if not isinstance(jobs, list):
            jobs = list(jobs)

        normalized: list[dict[str, Any]] = []
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                raise TypeError(
                    f"FlyDSL AOT spec {self.name!r}: job {index} must be a dict, "
                    f"got {type(job).__name__}"
                )
            normalized.append(dict(job))
        return normalized

    def compile(self, job: dict[str, Any]) -> dict[str, Any]:
        result = self._resolve(self.compiler)(**dict(job))
        if not isinstance(result, dict):
            raise TypeError(
                f"FlyDSL AOT spec {self.name!r}: compiler must return a dict, "
                f"got {type(result).__name__}"
            )
        return result


class AotSpecRegistry:
    """Ordered registry of FlyDSL AOT operator families."""

    def __init__(self) -> None:
        self._specs: dict[str, AotSpec] = {}
        self._lock = RLock()

    def register(self, spec: AotSpec) -> AotSpec:
        if not isinstance(spec, AotSpec):
            raise TypeError(f"spec must be an AotSpec, got {type(spec).__name__}")
        with self._lock:
            existing = self._specs.get(spec.name)
            if existing is not None and existing != spec:
                raise ValueError(f"FlyDSL AOT spec {spec.name!r} is already registered")
            self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> AotSpec:
        with self._lock:
            try:
                return self._specs[name]
            except KeyError as error:
                raise KeyError(f"unknown FlyDSL AOT spec: {name!r}") from error

    def get_all_specs(self) -> tuple[AotSpec, ...]:
        with self._lock:
            return tuple(self._specs.values())

    def clear(self) -> None:
        with self._lock:
            self._specs.clear()


DEFAULT_AOT_SPEC_REGISTRY = AotSpecRegistry()

_DEFAULT_SPECS = (
    AotSpec("moe", "aiter.aot.flydsl.moe"),
    AotSpec("mxfp4_moe", "aiter.aot.flydsl.mxfp4_moe"),
    AotSpec("gemm", "aiter.aot.flydsl.gemm"),
    AotSpec("grouped_moe", "aiter.aot.flydsl.grouped_moe"),
    AotSpec("chunk_gdn_h", "aiter.aot.flydsl.chunk_gdn_h"),
    AotSpec("mega_moe", "aiter.aot.flydsl.mega_moe"),
)


def register_default_specs(
    registry: AotSpecRegistry = DEFAULT_AOT_SPEC_REGISTRY,
) -> int:
    """Register and return the default FlyDSL AOT operator-family count."""

    for spec in _DEFAULT_SPECS:
        registry.register(spec)
    return len(_DEFAULT_SPECS)


__all__ = [
    "DEFAULT_AOT_SPEC_REGISTRY",
    "AotSpec",
    "AotSpecRegistry",
    "register_default_specs",
]
