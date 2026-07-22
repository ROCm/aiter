# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Declarative, CPU-only FlyDSL compile plans.

The registry binds compile arguments against the registered callable with
``inspect.signature(...).bind(...)`` and applies its defaults. Resolution only
imports and inspects callables chosen by the caller; it never queries a GPU,
allocates tensors, opens a stream, compiles, or launches. ``compile()`` is the
only operation that invokes a registered callable, under the unit's explicit
ROCm target environment.

Artifact persistence, manifests, deduplication, and runtime lookup intentionally
belong to later AOT layers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import inspect
import os
import re
from threading import RLock
from typing import Any, Callable, Iterator

__all__ = [
    "ArgumentKind",
    "BoundCall",
    "CompileOpRegistry",
    "CompilePlan",
    "CompileSpec",
    "CompileUnit",
    "DEFAULT_COMPILE_OP_REGISTRY",
    "KernelSignature",
    "RocmTarget",
    "SignatureArg",
    "register_compile_op",
]

_ARCH_RE = re.compile(r"gfx[0-9a-f]+")
_OP_ID_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\.v[1-9][0-9]*")
_ARG_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED_BINDING_NAMES = frozenset(("signature", "target"))


def _validate_op_id(op_id: object) -> None:
    if not isinstance(op_id, str):
        raise TypeError(f"op_id must be a string, got {type(op_id).__name__}")
    if _OP_ID_RE.fullmatch(op_id) is None:
        raise ValueError(
            "op_id must be a lowercase dot-separated identifier ending in "
            f"'.vN', got {op_id!r}"
        )


@dataclass(frozen=True)
class RocmTarget:
    """An explicit ROCm compilation target."""

    arch: str
    cu_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.arch, str) or _ARCH_RE.fullmatch(self.arch) is None:
            raise ValueError(f"invalid canonical ROCm arch: {self.arch!r}")
        if isinstance(self.cu_count, bool) or not isinstance(self.cu_count, int):
            raise TypeError("cu_count must be an integer")
        if self.cu_count <= 0:
            raise ValueError("cu_count must be positive")


@dataclass(frozen=True)
class BoundCall:
    """Normalized arguments bound to one registered callable signature.

    Values must be hashable compile metadata. Mutable containers are rejected
    rather than silently changing the types passed to the callable.
    """

    arguments: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        try:
            arguments = tuple(tuple(item) for item in self.arguments)
        except TypeError as error:
            raise TypeError("arguments must contain (name, value) pairs") from error
        if any(len(item) != 2 for item in arguments):
            raise TypeError("arguments must contain (name, value) pairs")

        names = []
        for name, value in arguments:
            if not isinstance(name, str) or _ARG_NAME_RE.fullmatch(name) is None:
                raise ValueError(f"invalid bound argument name: {name!r}")
            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    f"compile argument {name!r} must be hashable, "
                    f"got {type(value).__name__}"
                ) from error
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("bound argument names must be unique")
        object.__setattr__(self, "arguments", arguments)

    def as_kwargs(self) -> dict[str, Any]:
        """Return a fresh mapping suitable for invoking the callable."""

        return dict(self.arguments)


@dataclass(frozen=True)
class CompileSpec:
    """Stable op identity, explicit target, and normalized callable arguments."""

    op_id: str
    target: RocmTarget
    call: BoundCall

    def __post_init__(self) -> None:
        _validate_op_id(self.op_id)
        if not isinstance(self.target, RocmTarget):
            raise TypeError("target must be a RocmTarget")
        if not isinstance(self.call, BoundCall):
            raise TypeError("call must be a BoundCall")


class ArgumentKind(str, Enum):
    """Kinds of arguments represented by a kernel ABI."""

    TENSOR = "tensor"
    POINTER = "pointer"
    SCALAR = "scalar"
    STREAM = "stream"


def _normalize_dimensions(values: object, field_name: str) -> tuple[int | None, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable")
    try:
        dimensions = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{field_name} must be an iterable") from error
    for index, value in enumerate(dimensions):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name}[{index}] must be an integer or None")
        if value < 0:
            raise ValueError(f"{field_name}[{index}] must be non-negative")
    return dimensions


@dataclass(frozen=True)
class SignatureArg:
    """One manually supplied kernel ABI argument."""

    name: str
    kind: ArgumentKind
    dtype: str | None = None
    shape: tuple[int | None, ...] = ()
    strides: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _ARG_NAME_RE.fullmatch(self.name) is None:
            raise ValueError(f"invalid ABI argument name: {self.name!r}")
        if not isinstance(self.kind, ArgumentKind):
            raise TypeError("kind must be an ArgumentKind")
        shape = _normalize_dimensions(self.shape, "shape")
        strides = _normalize_dimensions(self.strides, "strides")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)

        if self.dtype is not None and (
            not isinstance(self.dtype, str)
            or not self.dtype
            or self.dtype != self.dtype.strip()
        ):
            raise ValueError("dtype must be a canonical non-empty string or None")
        if self.kind is ArgumentKind.TENSOR:
            if self.dtype is None:
                raise ValueError("tensor arguments require a dtype")
            if len(shape) != len(strides):
                raise ValueError("tensor shape and strides must have the same rank")
        elif self.kind in (ArgumentKind.POINTER, ArgumentKind.SCALAR):
            if self.dtype is None:
                raise ValueError(f"{self.kind.value} arguments require a dtype")
            if shape or strides:
                raise ValueError(f"{self.kind.value} arguments cannot declare a shape")
        elif self.dtype is not None or shape or strides:
            raise ValueError("stream arguments cannot declare dtype, shape, or strides")


@dataclass(frozen=True)
class KernelSignature:
    """An ordered, explicit kernel launch ABI."""

    arguments: tuple[SignatureArg, ...]

    def __post_init__(self) -> None:
        try:
            arguments = tuple(self.arguments)
        except TypeError as error:
            raise TypeError("arguments must be an iterable of SignatureArg") from error
        if not all(isinstance(argument, SignatureArg) for argument in arguments):
            raise TypeError("arguments must contain only SignatureArg values")
        names = tuple(argument.name for argument in arguments)
        if len(names) != len(set(names)):
            raise ValueError("signature argument names must be unique")
        object.__setattr__(self, "arguments", arguments)


@dataclass(frozen=True)
class CompileUnit:
    """One normalized builder call paired with its launch ABI."""

    spec: CompileSpec
    signature: KernelSignature

    def __post_init__(self) -> None:
        if not isinstance(self.spec, CompileSpec):
            raise TypeError("spec must be a CompileSpec")
        if not isinstance(self.signature, KernelSignature):
            raise TypeError("signature must be a KernelSignature")


@dataclass(frozen=True)
class CompilePlan:
    """An ordered collection of compile units; duplicates are retained."""

    units: tuple[CompileUnit, ...]

    def __post_init__(self) -> None:
        try:
            units = tuple(self.units)
        except TypeError as error:
            raise TypeError("units must be an iterable of CompileUnit") from error
        if not all(isinstance(unit, CompileUnit) for unit in units):
            raise TypeError("units must contain only CompileUnit values")
        object.__setattr__(self, "units", units)


@dataclass(frozen=True)
class _RegisteredCompileOp:
    compiler: Callable[..., Any]
    signature: inspect.Signature


_TARGET_ENV_LOCK = RLock()


@contextmanager
def _target_environment(target: RocmTarget) -> Iterator[None]:
    values = {"FLYDSL_GPU_ARCH": target.arch, "CU_NUM": str(target.cu_count)}
    with _TARGET_ENV_LOCK:
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class CompileOpRegistry:
    """Callable registry keyed by stable versioned operator IDs."""

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredCompileOp] = {}
        self._lock = RLock()

    def register(
        self, op_id: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that introspects and registers one callable."""

        _validate_op_id(op_id)

        def decorator(compiler: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(compiler):
                raise TypeError("compiler must be callable")
            signature = inspect.signature(compiler)
            unsupported = {
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }
            reserved = _RESERVED_BINDING_NAMES.intersection(signature.parameters)
            if unsupported:
                raise TypeError(
                    "registered callables must have fixed keyword-bindable "
                    f"parameters; unsupported: {sorted(unsupported)}"
                )
            if reserved:
                raise TypeError(
                    f"registered callable uses reserved parameters: {sorted(reserved)}"
                )
            with self._lock:
                if op_id in self._entries:
                    raise ValueError(f"compile op {op_id!r} is already registered")
                self._entries[op_id] = _RegisteredCompileOp(compiler, signature)
            return compiler

        return decorator

    def _entry(self, op_id: str) -> _RegisteredCompileOp:
        _validate_op_id(op_id)
        with self._lock:
            entry = self._entries.get(op_id)
        if entry is None:
            raise KeyError(f"no compile op registered for {op_id!r}")
        return entry

    def lookup(self, op_id: str) -> Callable[..., Any]:
        """Return the registered callable."""

        return self._entry(op_id).compiler

    @staticmethod
    def _bind(
        op_id: str, entry: _RegisteredCompileOp, kwargs: dict[str, Any]
    ) -> BoundCall:
        try:
            bound = entry.signature.bind(**kwargs)
        except TypeError as error:
            raise TypeError(f"{op_id}: {error}") from error
        bound.apply_defaults()
        return BoundCall(
            tuple(
                (name, bound.arguments[name])
                for name in entry.signature.parameters
                if name in bound.arguments
            )
        )

    def make_unit(
        self,
        op_id: str,
        *,
        target: RocmTarget,
        signature: KernelSignature,
        **kwargs: Any,
    ) -> CompileUnit:
        """Bind ``kwargs`` to the registered callable and create one unit."""

        entry = self._entry(op_id)
        call = self._bind(op_id, entry, kwargs)
        return CompileUnit(CompileSpec(op_id, target, call), signature)

    def compile(self, unit: CompileUnit) -> Any:
        """Invoke a unit's registered callable with its normalized kwargs."""

        if not isinstance(unit, CompileUnit):
            raise TypeError("unit must be a CompileUnit")
        entry = self._entry(unit.spec.op_id)
        kwargs = unit.spec.call.as_kwargs()
        if self._bind(unit.spec.op_id, entry, kwargs) != unit.spec.call:
            raise ValueError(
                f"{unit.spec.op_id}: bound call does not match registered signature"
            )
        with _target_environment(unit.spec.target):
            return entry.compiler(**kwargs)

    def compile_plan(self, plan: CompilePlan) -> tuple[Any, ...]:
        """Compile units in declaration order."""

        if not isinstance(plan, CompilePlan):
            raise TypeError("plan must be a CompilePlan")
        return tuple(self.compile(unit) for unit in plan.units)


DEFAULT_COMPILE_OP_REGISTRY = CompileOpRegistry()


def register_compile_op(
    op_id: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a callable in the process-wide default registry."""

    return DEFAULT_COMPILE_OP_REGISTRY.register(op_id)
