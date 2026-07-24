# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only recorder for the current gfx950 MoE FlyDSL compile requests.

The recorder deliberately drives the production host entry points.  It replaces
only the compile/build/launch boundaries, so shape and option derivation remains
owned by the runtime/AOT code under test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import enum
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterator
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

_ARCH = "gfx950"
_CU_COUNT = 256
_SCHEMA_VERSION = 1
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_HOST_MODULES = {
    "aiter.aot.flydsl.moe",
    "aiter.ops.flydsl.moe_kernels",
    "aiter.ops.flydsl.moe_sorting",
    "aiter.ops.flydsl.kernels.moe_sorting_kernel",
}

_BUILDER_LABELS = {
    "compile_mixed_moe_gemm1": "gemm",
    "compile_mixed_moe_gemm2": "gemm",
    "compile_moe_gemm1": "gemm",
    "compile_moe_gemm2": "gemm",
    "compile_moe_reduction": "reduction",
    "build_silu_and_mul_fq_module": "post_activation",
    "build_swiglu_and_mul_module": "post_activation",
    "_compile_moe_sorting_oneshot": "oneshot_builder",
    "_compile_moe_sorting_multiphase": "multiphase_builder",
}

_EXPECTED_SCENARIOS = {
    "stage1.main.non_split.bias.route_weighted": (
        ("compile_mixed_moe_gemm1",),
        ("stage1.gemm",),
    ),
    "stage1.int4.splitk": (
        ("compile_moe_gemm1",),
        ("stage1.gemm",),
    ),
    "stage1.splitk.fp4.silu.separated": (
        ("compile_mixed_moe_gemm1", "build_silu_and_mul_fq_module"),
        ("stage1.gemm", "post_activation.silu_and_mul_fq"),
    ),
    "stage1.splitk.fp8.swiglu.interleaved.bias": (
        ("compile_mixed_moe_gemm1", "build_silu_and_mul_fq_module"),
        ("stage1.gemm", "post_activation.silu_and_mul_fq"),
    ),
    "stage1.splitk.none.silu.interleaved": (
        ("compile_mixed_moe_gemm1", "build_silu_and_mul_fq_module"),
        ("stage1.gemm", "post_activation.silu_and_mul_fq"),
    ),
    "cktile.epilogue.silu": (
        ("build_silu_and_mul_fq_module",),
        ("post_activation.silu_and_mul_fq",),
    ),
    "cktile.epilogue.swiglu": (
        ("build_swiglu_and_mul_module",),
        ("post_activation.swiglu_and_mul",),
    ),
    "stage2.atomic.bias": (
        ("compile_mixed_moe_gemm2",),
        ("stage2.gemm",),
    ),
    "stage2.int4.atomic": (
        ("compile_moe_gemm2",),
        ("stage2.gemm",),
    ),
    "stage2.reduce.plain": (
        ("compile_mixed_moe_gemm2", "compile_moe_reduction"),
        ("stage2.gemm", "stage2.reduction"),
    ),
    "stage2.reduce.plain.large_auto_persist": (
        ("compile_mixed_moe_gemm2", "compile_moe_reduction"),
        ("stage2.gemm", "stage2.reduction"),
    ),
    "stage2.reduce.masked_ep": (
        ("compile_mixed_moe_gemm2", "compile_moe_reduction"),
        ("stage2.gemm", "stage2.reduction"),
    ),
    "sorting.oneshot.unmasked": (
        ("_compile_moe_sorting_oneshot",),
        ("sorting.oneshot",),
    ),
    "sorting.oneshot.masked": (
        ("_compile_moe_sorting_oneshot",),
        ("sorting.oneshot",),
    ),
    "sorting.multiphase.p0v2.unmasked.e384": (
        ("_compile_moe_sorting_multiphase",),
        ("sorting.multiphase.p0v2_p23",),
    ),
    "sorting.multiphase.4k.masked": (
        ("_compile_moe_sorting_multiphase",),
        ("sorting.multiphase.4k_fused",),
    ),
}


class ForbiddenBoundaryError(AssertionError):
    """Raised if recording reaches a real compiler, device, or launcher."""


@dataclass(frozen=True)
class _FakeExecutable:
    request_index: int
    launcher: str

    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ForbiddenBoundaryError(
            f"launcher {self.launcher!r} was called outside the mocked boundary"
        )


class _RecorderBackend:
    """Resolve registered builders without entering the real FlyDSL compiler."""

    def __init__(self, recorder: "_RequestRecorder") -> None:
        self.recorder = recorder
        self.calls: list[tuple[str, str]] = []

    def _artifact(self, mode: str, unit: Any, context: Any, *, consume: bool):
        self.calls.append((mode, unit.spec.op_id))
        launcher = context.registry.compile(unit)
        if consume:
            self.recorder.run_moe_compiled(launcher, ())
        return types.SimpleNamespace(launcher=launcher, unit=unit)

    def compile_aot(self, unit: Any, *, context: Any):
        return self._artifact("compile", unit, context, consume=True)

    def load_aot(self, unit: Any, *, context: Any, strict: bool = True):
        if not strict:
            raise AssertionError("recorder only supports strict loads")
        return self._artifact("load", unit, context, consume=True)

    def resolve_aot(self, unit: Any, *, context: Any):
        return self._artifact("resolve", unit, context, consume=False)


@dataclass(frozen=True)
class _HostImports:
    flyc: types.ModuleType
    aot_backend: types.ModuleType
    moe: types.ModuleType
    mixed_gemm: types.ModuleType
    standard_gemm: types.ModuleType
    silu: types.ModuleType
    swiglu: types.ModuleType
    sorting_kernel: types.ModuleType
    sorting_wrapper: types.ModuleType
    aot_moe: types.ModuleType


def _normalize(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return _normalize(value.value)
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"compile request contains unsupported value {value!r}")


def canonical_json(value: Any) -> str:
    """Serialize a recording in the checked-in golden format."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _install_namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
    return module


@contextmanager
def _isolated_host_imports() -> Iterator[_HostImports]:
    """Import target files without executing the GPU-heavy ``aiter.__init__``."""

    # FlyDSL owns native MLIR registrations and cannot be safely unloaded and
    # re-imported in one process (pytest/plugins may already have imported it).
    # Isolate only project modules; leaving FlyDSL resident is both safe and
    # representative of normal test-suite execution.
    managed_prefixes = ("aiter",)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in managed_prefixes
        )
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)

    try:
        _install_namespace("aiter", _PACKAGE_ROOT)
        _install_namespace("aiter.ops", _PACKAGE_ROOT / "ops")
        _install_namespace("aiter.ops.flydsl", _PACKAGE_ROOT / "ops" / "flydsl")
        _install_namespace(
            "aiter.ops.flydsl.kernels",
            _PACKAGE_ROOT / "ops" / "flydsl" / "kernels",
        )
        _install_namespace("aiter.aot", _PACKAGE_ROOT / "aot")
        _install_namespace("aiter.aot.flydsl", _PACKAGE_ROOT / "aot" / "flydsl")
        utility = _install_namespace("aiter.utility", _PACKAGE_ROOT / "utility")
        jit = _install_namespace("aiter.jit", _PACKAGE_ROOT / "jit")

        # The host paths only need dtype tags.  Importing the production dtype
        # module would pull the top-level extension/JIT stack into this CPU test.
        dtypes = types.ModuleType("aiter.utility.dtypes")
        dtypes.fp4x2 = torch.uint8
        dtypes.fp8 = torch.uint8
        dtypes.fp8_e8m0 = torch.uint8
        dtypes.bf16 = torch.bfloat16
        dtypes.fp16 = torch.float16
        sys.modules[dtypes.__name__] = dtypes
        utility.dtypes = dtypes

        # ``aot.flydsl.moe`` reads one config path at import time, while
        # ``utility.mx_types`` only needs the lazy decorator definition.
        jit_core = types.ModuleType("aiter.jit.core")
        jit_core.AITER_CONFIGS = types.SimpleNamespace(
            AITER_CONFIG_FMOE_FILE="unused-by-recorder.csv"
        )
        jit_core.compile_ops = lambda *_args, **_kwargs: lambda function: function
        sys.modules[jit_core.__name__] = jit_core
        jit.core = jit_core

        imports = _HostImports(
            flyc=importlib.import_module("flydsl.compiler"),
            aot_backend=importlib.import_module("aiter.ops.flydsl.aot_backend"),
            moe=importlib.import_module("aiter.ops.flydsl.moe_kernels"),
            mixed_gemm=importlib.import_module(
                "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage"
            ),
            standard_gemm=importlib.import_module(
                "aiter.ops.flydsl.kernels.moe_gemm_2stage"
            ),
            silu=importlib.import_module("aiter.ops.flydsl.kernels.silu_and_mul_fq"),
            swiglu=importlib.import_module("aiter.ops.flydsl.kernels.swiglu_and_mul"),
            sorting_kernel=importlib.import_module(
                "aiter.ops.flydsl.kernels.moe_sorting_kernel"
            ),
            sorting_wrapper=importlib.import_module("aiter.ops.flydsl.moe_sorting"),
            aot_moe=importlib.import_module("aiter.aot.flydsl.moe"),
        )
        yield imports
    finally:
        for name in list(sys.modules):
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in managed_prefixes
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


@contextmanager
def _recording_environment() -> Iterator[None]:
    values = {
        "ARCH": _ARCH,
        "COMPILE_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "FLYDSL_GPU_ARCH": _ARCH,
        "GPU_ARCHS": _ARCH,
        "HIP_VISIBLE_DEVICES": "",
    }
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


class _RequestRecorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.backend: _RecorderBackend | None = None
        self.compile_context: Any = None
        self._scenario: str | None = None
        self._request_label_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._cuda_calls: defaultdict[str, list[str]] = defaultdict(list)

    @contextmanager
    def scenario(self, name: str) -> Iterator[None]:
        if self._scenario is not None:
            raise AssertionError(
                f"nested recorder scenario: {self._scenario} -> {name}"
            )
        self._scenario = name
        try:
            yield
        finally:
            self._scenario = None

    def _host_path(self, builder: str) -> list[str]:
        path: list[str] = []
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            while frame is not None:
                module_name = str(frame.f_globals.get("__name__", ""))
                if module_name in _HOST_MODULES:
                    item = f"{module_name}.{frame.f_code.co_name}"
                    if frame.f_code.co_name in {
                        "_resolve_plan_launchers",
                        "_resolve_stage1_plan_launchers",
                        "_compile_cktile_epilogue_plan",
                    }:
                        frame = frame.f_back
                        continue
                    if not path or path[-1] != item:
                        path.append(item)
                frame = frame.f_back
        finally:
            del frame
        path.reverse()
        path.append(builder)
        return path

    def builder_proxy(
        self,
        original: Any,
        builder: str,
        launchers: tuple[str, ...],
    ) -> Any:
        signature = inspect.signature(original)
        short_name = builder.rsplit(".", 1)[-1]

        def proxy(*args: Any, **kwargs: Any) -> Any:
            if self._scenario is None:
                raise ForbiddenBoundaryError(
                    f"builder {builder} was reached without a trigger scenario"
                )
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            label = _BUILDER_LABELS[short_name]
            count_key = (self._scenario, label)
            self._request_label_counts[count_key] += 1
            count = self._request_label_counts[count_key]
            request_id = f"{self._scenario}/{label}"
            if count > 1:
                request_id += f"#{count}"

            request_index = len(self.requests)
            self.requests.append(
                {
                    "id": request_id,
                    "builder": builder,
                    "kwargs": _normalize(dict(bound.arguments)),
                    "trigger": {
                        "scenario": self._scenario,
                        "host_path": self._host_path(builder),
                        "launchers": [],
                    },
                }
            )
            executables = tuple(
                _FakeExecutable(request_index, launcher) for launcher in launchers
            )
            return executables[0] if len(executables) == 1 else executables

        proxy.__name__ = getattr(original, "__name__", "recorded_builder")
        proxy.__signature__ = signature
        return proxy

    def _consume_launcher(self, executable: Any, args: Any) -> None:
        if not isinstance(executable, _FakeExecutable):
            raise ForbiddenBoundaryError(
                f"unexpected executable crossed mocked launch boundary: {executable!r}"
            )
        if self._scenario is None:
            raise ForbiddenBoundaryError(
                f"launcher {executable.launcher!r} had no trigger scenario"
            )
        request = self.requests[executable.request_index]
        if request["trigger"]["scenario"] != self._scenario:
            raise ForbiddenBoundaryError(
                f"launcher {executable.launcher!r} escaped scenario "
                f"{request['trigger']['scenario']!r} into {self._scenario!r}"
            )
        self._assert_fake_tensors(args)
        request["trigger"]["launchers"].append(executable.launcher)

    def run_moe_compiled(self, executable: Any, args: Any) -> None:
        self._consume_launcher(executable, args)

    def run_sorting_compiled(self, executable: Any, *args: Any) -> None:
        self._consume_launcher(executable, args)

    def _assert_fake_tensors(self, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if not isinstance(value, FakeTensor):
                raise ForbiddenBoundaryError(
                    f"real tensor reached launch boundary: {type(value).__name__}"
                )
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                self._assert_fake_tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                self._assert_fake_tensors(item)

    def forbidden_compile(self, *_args: Any, **_kwargs: Any) -> None:
        raise ForbiddenBoundaryError("real flydsl.compiler.compile was invoked")

    def forbidden_cuda(self, *args: Any, **kwargs: Any) -> None:
        raise ForbiddenBoundaryError(
            f"unexpected real CUDA boundary: args={args!r}, kwargs={kwargs!r}"
        )

    def cuda_properties(self, _device: Any = None) -> types.SimpleNamespace:
        self._note_sorting_cuda_call("get_device_properties")
        return types.SimpleNamespace(
            gcnArchName=_ARCH,
            multi_processor_count=_CU_COUNT,
            shared_memory_per_block=163840,
        )

    def cuda_stream(self, _device: Any = None) -> int:
        self._note_sorting_cuda_call("current_stream")
        return 0

    def _note_sorting_cuda_call(self, name: str) -> None:
        if self._scenario is None or not self._scenario.startswith("sorting."):
            raise ForbiddenBoundaryError(
                f"CUDA mock {name} crossed outside a sorting trigger"
            )
        self._cuda_calls[self._scenario].append(name)

    def validate(self) -> None:
        grouped_builders: defaultdict[str, list[str]] = defaultdict(list)
        grouped_launchers: defaultdict[str, list[str]] = defaultdict(list)
        for request in self.requests:
            scenario = request["trigger"]["scenario"]
            grouped_builders[scenario].append(request["builder"].rsplit(".", 1)[-1])
            grouped_launchers[scenario].extend(request["trigger"]["launchers"])

        if set(grouped_builders) != set(_EXPECTED_SCENARIOS):
            raise AssertionError(
                "scenario coverage changed: "
                f"expected={sorted(_EXPECTED_SCENARIOS)}, "
                f"actual={sorted(grouped_builders)}"
            )
        for scenario, (
            expected_builders,
            expected_launchers,
        ) in _EXPECTED_SCENARIOS.items():
            actual_builders = tuple(grouped_builders[scenario])
            actual_launchers = tuple(grouped_launchers[scenario])
            if actual_builders != expected_builders:
                raise AssertionError(
                    f"{scenario}: expected builders {expected_builders}, "
                    f"got {actual_builders}"
                )
            if actual_launchers != expected_launchers:
                raise AssertionError(
                    f"{scenario}: expected launchers {expected_launchers}, "
                    f"got {actual_launchers}"
                )
            cuda_calls = tuple(self._cuda_calls.get(scenario, ()))
            expected_cuda = (
                ("current_stream",) if scenario.startswith("sorting.") else ()
            )
            if cuda_calls != expected_cuda:
                raise AssertionError(
                    f"{scenario}: expected CUDA mocks {expected_cuda}, got {cuda_calls}"
                )


def _install_boundary_mocks(
    stack: ExitStack, imports: _HostImports, recorder: _RequestRecorder
) -> None:
    core = importlib.import_module("aiter.ops.flydsl.compile_plan")
    recorder.backend = _RecorderBackend(recorder)
    recorder.compile_context = core.CompileContext(
        target=core.RocmTarget(_ARCH, _CU_COUNT),
        registry=core.DEFAULT_COMPILE_OP_REGISTRY,
        backend=recorder.backend,
    )
    stack.enter_context(
        mock.patch.object(
            imports.aot_backend,
            "create_runtime_compile_context",
            lambda _device=None: recorder.compile_context,
        )
    )

    builder_specs = (
        (
            imports.mixed_gemm,
            "compile_mixed_moe_gemm1",
            "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage.compile_mixed_moe_gemm1",
            ("stage1.gemm",),
        ),
        (
            imports.mixed_gemm,
            "compile_mixed_moe_gemm2",
            "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage.compile_mixed_moe_gemm2",
            ("stage2.gemm",),
        ),
        (
            imports.standard_gemm,
            "compile_moe_gemm1",
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.compile_moe_gemm1",
            ("stage1.gemm",),
        ),
        (
            imports.standard_gemm,
            "compile_moe_gemm2",
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.compile_moe_gemm2",
            ("stage2.gemm",),
        ),
        (
            imports.standard_gemm,
            "compile_moe_reduction",
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.compile_moe_reduction",
            ("stage2.reduction",),
        ),
        (
            imports.silu,
            "build_silu_and_mul_fq_module",
            "aiter.ops.flydsl.kernels.silu_and_mul_fq.build_silu_and_mul_fq_module",
            ("post_activation.silu_and_mul_fq",),
        ),
        (
            imports.swiglu,
            "build_swiglu_and_mul_module",
            "aiter.ops.flydsl.kernels.swiglu_and_mul.build_swiglu_and_mul_module",
            ("post_activation.swiglu_and_mul",),
        ),
        (
            imports.sorting_kernel,
            "_compile_moe_sorting_oneshot",
            "aiter.ops.flydsl.kernels.moe_sorting_kernel._compile_moe_sorting_oneshot",
            ("sorting.oneshot",),
        ),
        (
            imports.sorting_kernel,
            "_compile_moe_sorting_multiphase",
            "aiter.ops.flydsl.kernels.moe_sorting_kernel."
            "_compile_moe_sorting_multiphase",
            (
                "sorting.multiphase.clear_workspace",
                "sorting.multiphase.p0",
                "sorting.multiphase.p1",
                "sorting.multiphase.p23",
                "sorting.multiphase.p0v2",
                "sorting.multiphase.p0v2_p23",
                "sorting.multiphase.4k_fused",
            ),
        ),
    )
    for module, attribute, qualified_name, launchers in builder_specs:
        original = getattr(module, attribute)
        stack.enter_context(
            mock.patch.object(
                module,
                attribute,
                recorder.builder_proxy(original, qualified_name, launchers),
            )
        )

    stack.enter_context(
        mock.patch.object(imports.moe, "_run_compiled", recorder.run_moe_compiled)
    )
    stack.enter_context(
        mock.patch.object(
            imports.sorting_kernel,
            "_run_compiled",
            recorder.run_sorting_compiled,
        )
    )
    stack.enter_context(
        mock.patch.object(imports.flyc, "compile", recorder.forbidden_compile)
    )
    stack.enter_context(
        mock.patch.object(imports.sorting_kernel, "get_hip_arch", lambda: _ARCH)
    )


def _install_cuda_boundary_mocks(stack: ExitStack, recorder: _RequestRecorder) -> None:
    """Install CUDA guards before importing any project host module."""

    stack.enter_context(
        mock.patch.object(torch.cuda, "get_device_properties", recorder.cuda_properties)
    )
    stack.enter_context(
        mock.patch.object(torch.cuda, "current_stream", recorder.cuda_stream)
    )
    for name in ("_lazy_init", "current_device", "init", "set_device", "synchronize"):
        if hasattr(torch.cuda, name):
            stack.enter_context(
                mock.patch.object(torch.cuda, name, recorder.forbidden_cuda)
            )


def _clear_scenario_caches(imports: _HostImports) -> None:
    imports.moe._get_compiled_silu_fused.cache_clear()
    imports.moe._get_compiled_swiglu.cache_clear()
    imports.moe.compile_flydsl_moe_reduction.cache_clear()
    imports.moe.compile_flydsl_moe_stage2.cache_clear()
    imports.sorting_kernel._compute_sub_tokens.cache_clear()
    imports.sorting_kernel._p23_block_size.cache_clear()
    imports.sorting_kernel.compile_moe_sorting_oneshot.cache_clear()
    imports.sorting_kernel.compile_moe_sorting_p0v2_p23.cache_clear()
    imports.sorting_kernel.compile_moe_sorting_4k_fused.cache_clear()
    imports.sorting_wrapper._workspace_cache.clear()


def _stage_shape(*, experts: int = 256) -> dict[str, Any]:
    return {
        "model_dim": 7168,
        "inter_dim": 2048,
        "experts": experts,
        "topk": 8,
        "token_num": 16,
    }


def _kernel_params(
    imports: _HostImports, kernel_name: str, expected_stage: int
) -> dict[str, Any]:
    params = imports.moe.get_flydsl_kernel_params(kernel_name)
    if params is None:
        raise AssertionError(f"unknown recorder kernel name: {kernel_name}")
    if params["stage"] != expected_stage:
        raise AssertionError(
            f"{kernel_name}: expected stage {expected_stage}, got {params['stage']}"
        )
    return params


def _storage_dtype(dtype: str):
    if dtype in ("fp4", "fp8", "int4"):
        return torch.uint8
    if dtype in ("fp16", "f16"):
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise AssertionError(f"unsupported recorder dtype: {dtype}")


def _runtime_kwargs(function: Any, config: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(function).parameters
    return {name: config[name] for name in parameters if name in config}


def _stage1_runtime_inputs(
    imports: _HostImports,
    config: dict[str, Any],
) -> dict[str, Any]:
    tokens = config["token_num"]
    topk = config["topk"]
    model_dim = config["model_dim"]
    inter_dim = config["inter_dim"]
    experts = config["experts"]
    a_dtype = config["a_dtype"]
    b_dtype = config["b_dtype"]
    tile_m = config["tile_m"]
    route_capacity = max(tokens * topk, tile_m)

    a_shape = (tokens, model_dim // 2) if a_dtype == "fp4" else (tokens, model_dim)
    w1_k = model_dim // 2 if b_dtype in ("fp4", "int4") else model_dim
    values = {
        "a": torch.empty(a_shape, dtype=_storage_dtype(a_dtype)),
        "w1": torch.empty(
            (experts, 2 * inter_dim, w1_k),
            dtype=_storage_dtype(b_dtype),
        ),
        "sorted_token_ids": torch.empty(route_capacity, dtype=torch.int32),
        "sorted_expert_ids": torch.empty(route_capacity, dtype=torch.int32),
        "num_valid_ids": torch.empty(2, dtype=torch.int32),
        "out": None,
        "topk": topk,
        "w1_scale": None,
        "a1_scale": None,
        "sorted_weights": (
            torch.empty(route_capacity, dtype=torch.float32)
            if config.get("doweight_stage1", False)
            else None
        ),
        "bias": (
            torch.empty((experts, 2 * inter_dim), dtype=torch.float32)
            if config.get("enable_bias", False)
            else None
        ),
        "topk_ids": (
            torch.empty((tokens, topk), dtype=torch.int32)
            if config.get("enable_bias", False)
            else None
        ),
    }
    values.update(
        _runtime_kwargs(
            imports.moe.flydsl_moe_stage1,
            {"use_async_copy": True, **config},
        )
    )
    return values


def _stage2_runtime_inputs(
    imports: _HostImports,
    config: dict[str, Any],
    *,
    masked: bool,
) -> dict[str, Any]:
    tokens = config["token_num"]
    topk = config["topk"]
    model_dim = config["model_dim"]
    inter_dim = config["inter_dim"]
    experts = config["experts"]
    a_dtype = config["a_dtype"]
    b_dtype = config["b_dtype"]
    tile_m = config["tile_m"]
    sort_block_m = config.get("sort_block_m", 0) or tile_m
    routing_blocks = max(1, tokens * topk)
    route_capacity = routing_blocks * sort_block_m

    a_k = inter_dim // 2 if a_dtype == "fp4" else inter_dim
    w2_k = inter_dim // 2 if b_dtype in ("fp4", "int4") else inter_dim
    values = {
        "inter_states": torch.empty(
            (tokens, topk, a_k),
            dtype=_storage_dtype(a_dtype),
        ),
        "w2": torch.empty(
            (experts, model_dim, w2_k),
            dtype=_storage_dtype(b_dtype),
        ),
        "sorted_token_ids": torch.empty(route_capacity, dtype=torch.int32),
        "sorted_expert_ids": torch.empty(routing_blocks, dtype=torch.int32),
        "num_valid_ids": torch.empty(2, dtype=torch.int32),
        "out": None,
        "topk": topk,
        "w2_scale": None,
        "a2_scale": None,
        "sorted_weights": (
            torch.empty(route_capacity, dtype=torch.float32)
            if config.get(
                "doweight_stage2",
                not config.get("doweight_stage1", False),
            )
            else None
        ),
        "bias": (
            torch.empty((experts, model_dim), dtype=torch.float32)
            if config.get("enable_bias", False)
            else None
        ),
    }
    values.update(_runtime_kwargs(imports.moe.flydsl_moe_stage2, config))
    if masked:
        values["expert_mask"] = torch.ones(256, dtype=torch.int32)
        values["topk_ids"] = torch.zeros((tokens, topk), dtype=torch.int32)
    return values


def _run_stage1(
    imports: _HostImports,
    kernel_name: str,
    **options: Any,
) -> None:
    config = {
        **_stage_shape(),
        **_kernel_params(imports, kernel_name, expected_stage=1),
        **options,
    }
    inputs = _stage1_runtime_inputs(imports, config)
    inputs["compile_context"] = imports.aot_backend.create_runtime_compile_context()
    inputs["launch_context"] = imports.moe.LaunchContext(0)
    imports.moe.flydsl_moe_stage1(**inputs)


def _run_stage2(
    imports: _HostImports,
    kernel_name: str,
    *,
    masked: bool = False,
    **options: Any,
) -> None:
    config = {
        **_stage_shape(experts=32 if masked else 256),
        **_kernel_params(imports, kernel_name, expected_stage=2),
        **options,
    }
    inputs = _stage2_runtime_inputs(imports, config, masked=masked)
    inputs["compile_context"] = imports.aot_backend.create_runtime_compile_context()
    inputs["launch_context"] = imports.moe.LaunchContext(0)
    imports.moe.flydsl_moe_stage2(**inputs)


def _run_sorting(
    imports: _HostImports,
    *,
    tokens: int,
    masked: bool,
    experts: int = 256,
) -> None:
    topk = 8
    unit_size = 32
    # Deliberately oversized test buffers: deriving production sorting capacity
    # here would duplicate the host formula this recorder is meant to observe.
    sorted_capacity = 65536
    expert_block_capacity = 4096

    topk_ids = torch.zeros((tokens, topk), dtype=torch.int32)
    topk_weights = torch.zeros((tokens, topk), dtype=torch.float32)
    sorted_ids = torch.empty(sorted_capacity, dtype=torch.int32)
    sorted_weights = torch.empty(sorted_capacity, dtype=torch.float32)
    sorted_expert_ids = torch.empty(expert_block_capacity, dtype=torch.int32)
    num_valid_ids = torch.empty(2, dtype=torch.int32)
    moe_buf = torch.empty((tokens, 64), dtype=torch.bfloat16)
    expert_mask = torch.ones(experts, dtype=torch.int32) if masked else None

    imports.sorting_wrapper.flydsl_moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        experts,
        unit_size,
        expert_mask,
    )


def _record_scenarios(imports: _HostImports, recorder: _RequestRecorder) -> None:
    scenarios = (
        (
            "stage1.main.non_split.bias.route_weighted",
            lambda: _run_stage1(
                imports,
                "flydsl_moe1_afp4_wfp4_bf16_t32x128x256",
                act="silu",
                doweight_stage1=True,
                enable_bias=True,
            ),
        ),
        (
            "stage1.int4.splitk",
            lambda: _run_stage1(
                imports,
                "flydsl_moe1_abf16_wint4_bf16_t16x64x128_kb4",
                model_dim=7168,
                inter_dim=256,
                experts=384,
                topk=8,
                token_num=16,
                act="silu",
            ),
        ),
        (
            "stage1.splitk.fp4.silu.separated",
            lambda: _run_stage1(
                imports,
                "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_kb4_fp4",
                act="silu",
            ),
        ),
        (
            "stage1.splitk.fp8.swiglu.interleaved.bias",
            lambda: _run_stage1(
                imports,
                "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8",
                act="swiglu",
                # The public stage entry supports split-K fp8 even though the
                # current registry has no named fp8 k_batch variant.
                k_batch=4,
                enable_bias=True,
                swiglu_limit=7.0,
            ),
        ),
        (
            "stage1.splitk.none.silu.interleaved",
            lambda: _run_stage1(
                imports,
                "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui",
                act="silu",
                # See the fp8 split-K scenario above.
                k_batch=4,
            ),
        ),
        (
            "cktile.epilogue.silu",
            lambda: imports.aot_moe._precompile_epilogue_to_cache(
                act="silu",
                inter_dim=2048,
                topk=8,
                context=recorder.compile_context,
            ),
        ),
        (
            "cktile.epilogue.swiglu",
            lambda: imports.aot_moe._precompile_epilogue_to_cache(
                act="swiglu",
                inter_dim=2048,
                topk=8,
                context=recorder.compile_context,
            ),
        ),
        (
            "stage2.atomic.bias",
            lambda: _run_stage2(
                imports,
                "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2",
                enable_bias=True,
            ),
        ),
        (
            "stage2.int4.atomic",
            lambda: _run_stage2(
                imports,
                "flydsl_moe2_abf16_wint4_bf16_t16x128x128_atomic",
                model_dim=7168,
                inter_dim=256,
                experts=384,
                topk=8,
                token_num=16,
            ),
        ),
        (
            "stage2.reduce.plain",
            lambda: _run_stage2(
                imports,
                "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
            ),
        ),
        (
            "stage2.reduce.plain.large_auto_persist",
            lambda: _run_stage2(
                imports,
                "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
                token_num=4096,
            ),
        ),
        (
            "stage2.reduce.masked_ep",
            lambda: _run_stage2(
                imports,
                "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
                masked=True,
            ),
        ),
        (
            "sorting.oneshot.unmasked",
            lambda: _run_sorting(imports, tokens=8, masked=False),
        ),
        (
            "sorting.oneshot.masked",
            lambda: _run_sorting(imports, tokens=8, masked=True),
        ),
        (
            "sorting.multiphase.p0v2.unmasked.e384",
            lambda: _run_sorting(
                imports,
                tokens=128,
                masked=False,
                experts=384,
            ),
        ),
        (
            "sorting.multiphase.4k.masked",
            lambda: _run_sorting(imports, tokens=4096, masked=True),
        ),
    )

    for name, run in scenarios:
        _clear_scenario_caches(imports)
        with recorder.scenario(name):
            run()


def record_compile_requests() -> dict[str, Any]:
    """Collect normalized requests while prohibiting real compile/GPU work."""

    # FakeTensorMode imports torch._dynamo while it is constructed.  Do that
    # before replacing CUDA callables so Dynamo never tries to classify the
    # boundary guards themselves as torch APIs.
    fake_mode = FakeTensorMode()
    recorder = _RequestRecorder()
    with _recording_environment(), ExitStack() as stack:
        _install_cuda_boundary_mocks(stack, recorder)
        with _isolated_host_imports() as imports:
            _install_boundary_mocks(stack, imports, recorder)
            with fake_mode:
                _record_scenarios(imports, recorder)
        recorder.validate()
        return {
            "schema_version": _SCHEMA_VERSION,
            "target": {"arch": _ARCH, "cu_count": _CU_COUNT},
            # IDs are unique and semantic. Sorting by them makes the golden
            # insensitive to declaration/import ordering while retaining
            # meaningful builder and launcher order inside each request.
            "requests": sorted(recorder.requests, key=lambda request: request["id"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        type=Path,
        help="write canonical JSON to this path instead of stdout",
    )
    args = parser.parse_args()
    output = canonical_json(record_compile_requests())
    if args.write is None:
        print(output, end="")
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(output)


if __name__ == "__main__":
    main()
