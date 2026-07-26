# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only FlyDSL compile requests and callable registration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import inspect
import re
from threading import RLock
from typing import Any, Callable, Generic, Protocol, TypeVar

__all__ = [
    "ArgumentKind",
    "CompileBackend",
    "CompileContext",
    "CompileOpRegistry",
    "CompileRequest",
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


def _normalize_bound_kwargs(
    values: object,
) -> tuple[tuple[str, Any], ...]:
    try:
        items = tuple(tuple(item) for item in values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("bound_kwargs must contain (name, value) pairs") from error
    if any(len(item) != 2 for item in items):
        raise TypeError("bound_kwargs must contain (name, value) pairs")

    names = []
    for name, value in items:
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
    return items


@dataclass(frozen=True)
class CompileRequest:
    """One normalized builder call and the ABI needed to compile it."""

    op_id: str
    target: RocmTarget
    bound_kwargs: tuple[tuple[str, Any], ...]
    signature: KernelSignature

    def __post_init__(self) -> None:
        _validate_op_id(self.op_id)
        if not isinstance(self.target, RocmTarget):
            raise TypeError("target must be a RocmTarget")
        if not isinstance(self.signature, KernelSignature):
            raise TypeError("signature must be a KernelSignature")
        object.__setattr__(
            self,
            "bound_kwargs",
            _normalize_bound_kwargs(self.bound_kwargs),
        )

    def as_kwargs(self) -> dict[str, Any]:
        """Return a fresh mapping suitable for invoking the builder."""

        return dict(self.bound_kwargs)


# The old name is a zero-cost compatibility alias, not a second abstraction.
CompileUnit = CompileRequest

ArtifactT = TypeVar("ArtifactT")


class CompileBackend(Protocol[ArtifactT]):
    """Backend interface carried by :class:`CompileContext`."""

    def compile_aot(
        self,
        request: CompileRequest,
        *,
        context: "CompileContext[ArtifactT]",
    ) -> ArtifactT: ...

    def load_aot(
        self,
        request: CompileRequest,
        *,
        context: "CompileContext[ArtifactT]",
        strict: bool = True,
    ) -> ArtifactT: ...

    def resolve_aot(
        self,
        request: CompileRequest,
        *,
        context: "CompileContext[ArtifactT]",
    ) -> ArtifactT: ...


@dataclass
class _RegisteredCompiler:
    loader: Callable[[], Callable[..., Any]]
    compiler: Callable[..., Any] | None = None
    signature: inspect.Signature | None = None


def _inspect_compiler(compiler: Callable[..., Any]) -> inspect.Signature:
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
            "registered callables must have fixed keyword-bindable parameters; "
            f"unsupported: {sorted(unsupported)}"
        )
    if reserved:
        raise TypeError(
            f"registered callable uses reserved parameters: {sorted(reserved)}"
        )
    return signature


class CompileOpRegistry:
    """Lazy callable registry keyed by stable versioned operation IDs."""

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredCompiler] = {}
        self._lock = RLock()

    def register(
        self,
        op_id: str,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that registers an already imported builder."""

        _validate_op_id(op_id)

        def decorator(compiler: Callable[..., Any]) -> Callable[..., Any]:
            signature = _inspect_compiler(compiler)
            with self._lock:
                if op_id in self._entries:
                    raise ValueError(f"compile op {op_id!r} is already registered")
                self._entries[op_id] = _RegisteredCompiler(
                    loader=lambda: compiler,
                    compiler=compiler,
                    signature=signature,
                )
            return compiler

        return decorator

    def ensure_lazy(
        self,
        op_id: str,
        loader: Callable[[], Callable[..., Any]],
    ) -> None:
        """Register a builder loader unless the operation is already known."""

        _validate_op_id(op_id)
        if not callable(loader):
            raise TypeError("loader must be callable")
        with self._lock:
            self._entries.setdefault(op_id, _RegisteredCompiler(loader=loader))

    def is_registered(self, op_id: str) -> bool:
        _validate_op_id(op_id)
        with self._lock:
            return op_id in self._entries

    def _entry(self, op_id: str) -> _RegisteredCompiler:
        _validate_op_id(op_id)
        with self._lock:
            entry = self._entries.get(op_id)
            if entry is None:
                raise KeyError(f"no compile op registered for {op_id!r}")
            if entry.compiler is None:
                compiler = entry.loader()
                entry.compiler = compiler
                entry.signature = _inspect_compiler(compiler)
            return entry

    def lookup(self, op_id: str) -> Callable[..., Any]:
        """Resolve and return the registered builder."""

        compiler = self._entry(op_id).compiler
        assert compiler is not None
        return compiler

    def parameter_names(self, op_id: str) -> tuple[str, ...]:
        """Return builder parameter names in callable order."""

        signature = self._entry(op_id).signature
        assert signature is not None
        return tuple(signature.parameters)

    @staticmethod
    def _bind(
        op_id: str,
        entry: _RegisteredCompiler,
        kwargs: dict[str, Any],
    ) -> tuple[tuple[str, Any], ...]:
        assert entry.signature is not None
        assert entry.compiler is not None
        try:
            bound = entry.signature.bind(**kwargs)
        except TypeError as error:
            name = getattr(
                entry.compiler,
                "__qualname__",
                getattr(entry.compiler, "__name__", type(entry.compiler).__name__),
            )
            builder = f"{entry.compiler.__module__}.{name}"
            raise TypeError(f"{op_id} ({builder}): {error}") from error
        bound.apply_defaults()
        return _normalize_bound_kwargs(
            (name, bound.arguments[name])
            for name in entry.signature.parameters
            if name in bound.arguments
        )

    def make_request(
        self,
        op_id: str,
        *,
        target: RocmTarget,
        signature: KernelSignature,
        **kwargs: Any,
    ) -> CompileRequest:
        """Bind builder kwargs/defaults and create one immutable request."""

        entry = self._entry(op_id)
        return CompileRequest(
            op_id=op_id,
            target=target,
            bound_kwargs=self._bind(op_id, entry, kwargs),
            signature=signature,
        )

    def resolve(self, request: CompileRequest) -> Callable[..., Any]:
        """Resolve a request to its builder after validating its binding."""

        if not isinstance(request, CompileRequest):
            raise TypeError(
                f"request must be a CompileRequest, got {type(request).__name__}"
            )
        entry = self._entry(request.op_id)
        if (
            self._bind(request.op_id, entry, request.as_kwargs())
            != request.bound_kwargs
        ):
            raise ValueError(
                f"{request.op_id}: bound kwargs do not match registered signature"
            )
        assert entry.compiler is not None
        return entry.compiler

    def compile(self, request: CompileRequest) -> Any:
        """Invoke the request's builder with normalized kwargs."""

        return self.resolve(request)(**request.as_kwargs())


DEFAULT_COMPILE_OP_REGISTRY = CompileOpRegistry()


@dataclass(frozen=True)
class CompileContext(Generic[ArtifactT]):
    """Immutable target, registry, and backend for compilation or AOT loading."""

    target: RocmTarget
    registry: CompileOpRegistry
    backend: CompileBackend[ArtifactT]

    def __post_init__(self) -> None:
        if not isinstance(self.target, RocmTarget):
            raise TypeError("target must be a RocmTarget")
        if not isinstance(self.registry, CompileOpRegistry):
            raise TypeError("registry must be a CompileOpRegistry")
        for method_name in ("compile_aot", "load_aot", "resolve_aot"):
            if not callable(getattr(self.backend, method_name, None)):
                raise TypeError(
                    f"backend must implement {method_name}(request, *, context, ...)"
                )


def register_compile_op(
    op_id: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a builder in the process-wide default registry."""

    return DEFAULT_COMPILE_OP_REGISTRY.register(op_id)
