# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Aiter compatibility backend for ABI-driven FlyDSL AOT compilation.

FlyDSL 0.2.x exposes ``@jit`` launchers and an environment-controlled disk
cache, but not yet a public ``compile_aot(signature=..., target=...)`` API.  This
module is the single compatibility boundary for that gap.  Tensor arguments use
FlyDSL's ``TorchTensorJitArg`` with a metadata-only tensor object so their cache
signature is byte-for-byte compatible with normal torch runtime arguments,
without constructing a FakeTensor or allocating storage.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import functools
import inspect
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from .compile_plan import (
    ArgumentKind,
    CompileContext,
    CompileOpRegistry,
    CompileUnit,
    DEFAULT_COMPILE_OP_REGISTRY,
    KernelSignature,
    RocmTarget,
    SignatureArg,
)

__all__ = [
    "AotArtifact",
    "AotBackend",
    "AotBackendError",
    "AotCacheMissError",
    "compile_aot",
    "create_compile_context",
    "create_runtime_compile_context",
    "load_aot",
]


class AotBackendError(RuntimeError):
    """Base error carrying the unit and target that failed."""

    def __init__(
        self,
        operation: str,
        unit: CompileUnit,
        context: CompileContext[Any],
        detail: str,
    ) -> None:
        self.operation = operation
        self.op_id = unit.spec.op_id
        self.target = context.target
        self.signature = unit.signature
        self.cache_dir = _cache_dir()
        super().__init__(
            f"{operation} failed for op_id={self.op_id!r}, "
            f"target={self.target.arch}/{self.target.cu_count}, "
            f"signature={self.signature!r}, cache_dir={self.cache_dir}: {detail}"
        )


class AotCacheMissError(AotBackendError):
    """Strict AOT lookup could not find a usable FlyDSL cache artifact."""


@dataclass(frozen=True)
class AotArtifact:
    """Current FlyDSL launcher plus the ABI metadata used to compile/load it."""

    unit: CompileUnit
    launcher: Any
    compile_args: tuple[Any, ...]
    loaded: bool
    cache_dir: str

    def launch(self, *args: Any) -> Any:
        """Dispatch with real runtime arguments through the loaded launcher."""

        return self.launcher(*args)


@dataclass(frozen=True)
class _TensorAbiValue:
    """Metadata-only object accepted by FlyDSL's torch tensor JIT adapter."""

    dtype: Any
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_bytes: int

    def element_size(self) -> int:
        return self.element_bytes

    def stride(self) -> tuple[int, ...]:
        return self.strides

    @staticmethod
    def data_ptr() -> int:
        return 0


_ENV_LOCK = RLock()


def _cache_dir() -> str:
    return str(
        Path(
            os.path.expanduser(
                os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
            )
        )
    )


@contextmanager
def _flydsl_environment(
    target: RocmTarget,
    *,
    run_only: bool,
) -> Iterator[None]:
    # ARCH is the 0.2.x compiler override, while FLYDSL_GPU_ARCH is consumed
    # by target-aware kernel helpers.  Both are required for deterministic
    # cross-compilation with the current FlyDSL release.
    values = {
        "ARCH": target.arch,
        "FLYDSL_GPU_ARCH": target.arch,
        "CU_NUM": str(target.cu_count),
        "COMPILE_ONLY": "1",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
        "FLYDSL_RUNTIME_RUN_ONLY": "1" if run_only else "0",
    }
    with _ENV_LOCK:
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


def _validate_unit_context(
    unit: CompileUnit,
    context: CompileContext[Any],
) -> None:
    if not isinstance(unit, CompileUnit):
        raise TypeError(f"unit must be a CompileUnit, got {type(unit).__name__}")
    if not isinstance(context, CompileContext):
        raise TypeError(
            f"context must be a CompileContext, got {type(context).__name__}"
        )
    if unit.spec.target != context.target:
        raise ValueError(
            f"{unit.spec.op_id}: unit target {unit.spec.target!r} does not match "
            f"context target {context.target!r}"
        )


def _validate_signature(unit: CompileUnit) -> tuple[SignatureArg, ...]:
    signature = unit.signature
    if not isinstance(signature, KernelSignature):
        raise TypeError("unit signature must be a KernelSignature")
    try:
        arguments = tuple(signature.arguments)
    except TypeError as error:
        raise TypeError("signature arguments must be iterable") from error
    if not all(isinstance(argument, SignatureArg) for argument in arguments):
        raise TypeError("signature arguments must contain only SignatureArg values")
    names = tuple(argument.name for argument in arguments)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate ABI fields: {names!r}")
    return arguments


def _launcher_signature(launcher: Any) -> inspect.Signature:
    function = getattr(launcher, "func", launcher)
    try:
        return inspect.signature(function, eval_str=True)
    except (NameError, TypeError, ValueError):
        return inspect.signature(function)


def _expected_scalar_type(dtype: str) -> type:
    import flydsl.expr as fx

    names = {
        "bool": "Boolean",
        "i8": "Int8",
        "i16": "Int16",
        "i32": "Int32",
        "i64": "Int64",
        "u8": "Uint8",
        "u16": "Uint16",
        "u32": "Uint32",
        "u64": "Uint64",
        "f16": "Float16",
        "bf16": "BFloat16",
        "f32": "Float32",
        "f64": "Float64",
    }
    type_name = names.get(dtype)
    scalar_type = getattr(fx, type_name, None) if type_name is not None else None
    if scalar_type is None:
        raise TypeError(f"unsupported scalar dtype {dtype!r}")
    return scalar_type


def _expected_pointer_type(dtype: str) -> type:
    # Pointer element types share FlyDSL's Numeric classes with scalar values.
    return _expected_scalar_type(dtype)


def _torch_dtype(dtype: str) -> Any:
    import torch

    names = {
        "bool": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
        "f16": "float16",
        "fp16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "fp32": "float32",
        "f64": "float64",
        "fp8e4m3fn": "float8_e4m3fn",
        "fp8e4m3fnuz": "float8_e4m3fnuz",
        "fp8e5m2": "float8_e5m2",
        "fp8e5m2fnuz": "float8_e5m2fnuz",
    }
    name = names.get(dtype)
    value = getattr(torch, name, None) if name is not None else None
    if value is None:
        raise TypeError(f"unsupported tensor dtype {dtype!r}")
    return value


def _representative_layout(
    argument: SignatureArg,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not argument.shape:
        raise ValueError(f"tensor ABI field {argument.name!r} must have positive rank")
    declared_unit_strides = [
        index for index, stride in enumerate(argument.strides) if stride == 1
    ]
    if len(declared_unit_strides) != 1:
        raise ValueError(
            f"tensor ABI field {argument.name!r} must declare exactly one "
            "unit-stride dimension"
        )

    shape = tuple(2 if value is None else value for value in argument.shape)
    contiguous = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        contiguous[index] = contiguous[index + 1] * max(shape[index + 1], 1)
    strides = tuple(
        contiguous[index] if value is None else value
        for index, value in enumerate(argument.strides)
    )
    if strides.index(1) != declared_unit_strides[0]:
        raise ValueError(
            f"tensor ABI field {argument.name!r} has an ambiguous representative "
            f"layout: shape={argument.shape!r}, strides={argument.strides!r}"
        )
    return shape, strides


def _materialize_argument(argument: SignatureArg) -> Any:
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    if argument.kind is ArgumentKind.POINTER:
        return flyc.from_c_void_p(
            _expected_pointer_type(str(argument.dtype)),
            None,
        )
    if argument.kind is ArgumentKind.SCALAR:
        scalar_type = _expected_scalar_type(str(argument.dtype))
        from flydsl.expr.numeric import Float

        return scalar_type(0.0 if issubclass(scalar_type, Float) else 0)
    if argument.kind is ArgumentKind.STREAM:
        return fx.Stream(None)
    if argument.kind is ArgumentKind.TENSOR:
        from flydsl.compiler.jit_argument import TorchTensorJitArg

        dtype = _torch_dtype(str(argument.dtype))
        shape, strides = _representative_layout(argument)
        metadata = _TensorAbiValue(
            dtype=dtype,
            shape=shape,
            strides=strides,
            element_bytes=_torch_element_size(dtype),
        )
        # FlyDSL 0.2.x has no public metadata-only memref constructor.  Its
        # TorchTensorJitArg constructor only reads this small metadata protocol,
        # so using it directly preserves the exact runtime cache signature.
        return TorchTensorJitArg(metadata)
    raise TypeError(f"unsupported ABI kind {argument.kind!r}")


def _torch_element_size(dtype: Any) -> int:
    import torch

    # Constructing a tensor is intentionally forbidden in this adapter.
    sizes = {
        torch.bool: 1,
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
    }
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    ):
        value = getattr(torch, name, None)
        if value is not None:
            sizes[value] = 1
    try:
        return sizes[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported tensor dtype {dtype!r}") from error


def _annotation_contract(annotation: Any) -> tuple[ArgumentKind | None, str | None]:
    import flydsl.expr as fx

    if annotation is inspect.Parameter.empty:
        return None, None
    if annotation is fx.Pointer:
        return ArgumentKind.POINTER, None
    if annotation is fx.Tensor:
        return ArgumentKind.TENSOR, None
    if annotation is fx.Stream:
        return ArgumentKind.STREAM, None
    scalar_dtypes = {
        _expected_scalar_type(dtype): dtype
        for dtype in (
            "bool",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "f16",
            "bf16",
            "f32",
            "f64",
        )
    }
    dtype = scalar_dtypes.get(annotation)
    if dtype is not None:
        return ArgumentKind.SCALAR, dtype
    return None, None


def _materialize_compile_args(
    unit: CompileUnit,
    launcher: Any,
) -> tuple[Any, ...]:
    arguments = _validate_signature(unit)
    compiler_signature = _launcher_signature(launcher)
    parameters = tuple(compiler_signature.parameters.values())
    abi_names = tuple(argument.name for argument in arguments)
    compiler_names = tuple(parameter.name for parameter in parameters)
    if abi_names != compiler_names:
        missing = tuple(name for name in compiler_names if name not in abi_names)
        extra = tuple(name for name in abi_names if name not in compiler_names)
        raise ValueError(
            "ABI/compiler parameter mismatch: "
            f"abi={abi_names!r}, compiler={compiler_names!r}, "
            f"missing={missing!r}, extra={extra!r}"
        )

    materialized = []
    for argument, parameter in zip(arguments, parameters):
        compiler_kind, compiler_dtype = _annotation_contract(parameter.annotation)
        if compiler_kind is None:
            raise TypeError(
                f"compiler parameter {parameter.name!r} has unsupported or missing "
                f"annotation {parameter.annotation!r}"
            )
        if compiler_kind is not argument.kind:
            raise TypeError(
                f"ABI/compiler kind mismatch for {argument.name!r}: "
                f"{argument.kind.value} != {compiler_kind.value}"
            )
        if (
            compiler_dtype is not None
            and argument.dtype is not None
            and compiler_dtype != argument.dtype
        ):
            raise TypeError(
                f"ABI/compiler dtype mismatch for {argument.name!r}: "
                f"{argument.dtype!r} != {compiler_dtype!r}"
            )
        materialized.append(_materialize_argument(argument))
    return tuple(materialized)


class AotBackend:
    """Compile/load current FlyDSL launchers through the future-facing API."""

    def __init__(self, *, strict_runtime: bool = False) -> None:
        self.strict_runtime = bool(strict_runtime)
        self._resolved_artifacts: dict[CompileUnit, AotArtifact] = {}
        self._resolved_lock = RLock()

    def _prepare(
        self,
        unit: CompileUnit,
        *,
        context: CompileContext[AotArtifact],
        load: bool,
    ) -> AotArtifact:
        operation = "load_aot" if load else "compile_aot"
        try:
            _validate_unit_context(unit, context)
            with _flydsl_environment(context.target, run_only=load):
                launcher = context.registry.compile(unit)
                compile_args = _materialize_compile_args(unit, launcher)
                # Direct @jit invocation under COMPILE_ONLY compiles or checks
                # the disk cache without packing/launching runtime arguments.
                launcher(*compile_args)
        except AotBackendError:
            raise
        except Exception as error:
            if not isinstance(unit, CompileUnit) or not isinstance(
                context, CompileContext
            ):
                raise
            is_strict_miss = load and "FLYDSL_RUNTIME_RUN_ONLY=1" in str(error)
            error_type = AotCacheMissError if is_strict_miss else AotBackendError
            raise error_type(operation, unit, context, str(error)) from error
        return AotArtifact(
            unit=unit,
            launcher=launcher,
            compile_args=compile_args,
            loaded=load,
            cache_dir=_cache_dir(),
        )

    def compile_aot(
        self,
        unit: CompileUnit,
        *,
        context: CompileContext[AotArtifact],
    ) -> AotArtifact:
        return self._prepare(unit, context=context, load=False)

    def load_aot(
        self,
        unit: CompileUnit,
        *,
        context: CompileContext[AotArtifact],
        strict: bool = True,
    ) -> AotArtifact:
        if not strict:
            raise ValueError(
                "FlyDSL 0.2.x has no non-compiling load API; use compile_aot() "
                "explicitly instead of requesting a fallback"
            )
        return self._prepare(unit, context=context, load=True)

    def resolve_aot(
        self,
        unit: CompileUnit,
        *,
        context: CompileContext[AotArtifact],
    ) -> AotArtifact:
        with self._resolved_lock:
            artifact = self._resolved_artifacts.get(unit)
            if artifact is None:
                artifact = (
                    self.load_aot(unit, context=context, strict=True)
                    if self.strict_runtime
                    else self.compile_aot(unit, context=context)
                )
                self._resolved_artifacts[unit] = artifact
            return artifact


def create_compile_context(
    target: RocmTarget,
    *,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
    strict_runtime: bool = False,
) -> CompileContext[AotArtifact]:
    """Construct an explicit Aiter/FlyDSL compile context."""

    backend = AotBackend(strict_runtime=strict_runtime)
    return CompileContext(target=target, registry=registry, backend=backend)


@functools.lru_cache(maxsize=16)
def _cached_runtime_compile_context(
    target: RocmTarget,
    registry: CompileOpRegistry,
    strict_runtime: bool,
) -> CompileContext[AotArtifact]:
    return create_compile_context(
        target,
        registry=registry,
        strict_runtime=strict_runtime,
    )


def create_runtime_compile_context(
    device: Any = None,
    *,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileContext[AotArtifact]:
    """Resolve the live target once at the outer runtime boundary."""

    import torch

    properties = torch.cuda.get_device_properties(device)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not arch:
        raise RuntimeError("current ROCm device did not report gcnArchName")
    target = RocmTarget(arch=arch, cu_count=int(properties.multi_processor_count))
    strict = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY", "0") == "1"
    return _cached_runtime_compile_context(
        target,
        registry,
        strict,
    )


def compile_aot(
    unit: CompileUnit,
    *,
    context: CompileContext[AotArtifact],
) -> AotArtifact:
    """Future-facing AOT compile entry point."""

    return context.backend.compile_aot(unit, context=context)


def load_aot(
    unit: CompileUnit,
    *,
    context: CompileContext[AotArtifact],
    strict: bool = True,
) -> AotArtifact:
    """Future-facing AOT cache-load entry point."""

    return context.backend.load_aot(unit, context=context, strict=strict)
