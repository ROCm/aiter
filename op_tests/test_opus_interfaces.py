# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Interface goldens for caller-resolved OPUS GEMM/BMM ``kid`` APIs."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
    DEFAULT_COMPILED_KIDS,
    DEFAULT_COMPILED_KIDS_BY_ARCH,
    OPUS_KERNEL_TAGS_BY_ARCH_FAMILY,
    OPUS_MANDATORY_A8_KIDS,
    a8w8_mxscale_bmm_kernels_list,
    a8w8_kernels_list,
    a8w8_scale_kernels_list,
    default_compiled_kids_for_arch,
    get_kernel_instance,
    gfx1250_clusterlaunch_kernels_list,
    gfx1250_kernels_list,
    gfx1250_splitk_fuse_kernels_list,
    gfx942_a8w8_kernels_list,
    gfx942_nosplit_kernels_list,
    gfx942_splitk_kernels_list,
    kernels_list,
)


_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_PARAMETERS = (
    "XQ",
    "WQ",
    "Y",
    "kid",
    "layout",
    "x_scale",
    "w_scale",
    "bias",
    "split_k",
    "workspace",
)
_A16_RAW_PARAMETERS = (
    "XQ",
    "WQ",
    "Y",
    "bias",
    "workspace",
    "kid",
    "split_k",
)
_BPRESHUFFLE_TAG = "a8w8_blockscale_bpreshuffle_singlebuf"


def _parameter_names(callable_) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_).parameters)


def _python_definition_parameter_names(
    source_path: Path, function_name: str
) -> tuple[str, ...]:
    tree = ast.parse(source_path.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    arguments = definition.args
    return tuple(
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    )


def _cpp_parameter_names(source: str, function_name: str) -> tuple[str, ...]:
    declaration = re.search(
        rf"\bvoid\s+{re.escape(function_name)}\s*\((.*?)\)\s*;",
        source,
        flags=re.DOTALL,
    )
    assert declaration is not None, f"missing C++ declaration for {function_name}"
    names = []
    for parameter in declaration.group(1).split(","):
        name = re.search(r"([A-Za-z_]\w*)\s*$", parameter.strip())
        assert name is not None
        names.append(name.group(1))
    return tuple(names)


def _pybind_parameter_names(source: str, macro_name: str) -> tuple[str, ...]:
    start = source.index(f"#define {macro_name}")
    end = source.find("\n#define ", start + 1)
    block = source[start:] if end < 0 else source[start:end]
    return tuple(re.findall(r'py::arg\("([^"]+)"\)', block))


def _instance_arch(instance) -> str:
    return (instance.arch_prefix or "gfx950").lower()


def test_public_gemm_and_bmm_entries_keep_one_exact_kid_contract():
    import aiter

    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")

    assert opus.__all__ == ["opus_gemm", "opus_bmm"]
    assert aiter.opus_gemm is opus.opus_gemm
    assert aiter.opus_bmm is opus.opus_bmm
    assert a16.__all__ == []
    assert a8.__all__ == []
    assert _parameter_names(opus.opus_gemm) == _PUBLIC_PARAMETERS
    assert _parameter_names(opus.opus_bmm) == _PUBLIC_PARAMETERS

    for public_entry in (opus.opus_gemm, opus.opus_bmm):
        parameters = inspect.signature(public_entry).parameters
        assert parameters["kid"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["kid"].default is inspect.Parameter.empty
        assert parameters["layout"].default == "plain"
        assert parameters["split_k"].default == 0
        assert parameters["workspace"].default is None

    removed = (
        "gemm_a16w16_opus",
        "opus_gemm_a16w16_launch",
        "opus_gemm_a16w16_tune",
        "opus_gemm_workspace_init",
        "opus_gemm_a8w8_launch",
        "opus_gemm_a8w8_blockscale_launch",
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
        "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
    )
    assert all(not hasattr(opus, name) for name in removed)


def test_family_modules_keep_only_private_exact_kid_adapters():
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    plan_module = importlib.import_module("aiter.ops.opus.launch_plan")

    assert callable(a16._execute_a16w16)
    assert callable(a16._launch_a16w16_backend)
    assert callable(a16._launch_a16w16_gemm)
    assert callable(a16._launch_a16w16_bmm)
    assert callable(plan_module._get_cached_a16w16_launch_plan)
    assert not hasattr(a16, "LaunchConfig")
    assert not hasattr(a16, "_resolve_exact_a16w16_config")
    assert not hasattr(a16, "_launch_a16w16_exact")
    assert not hasattr(a16, "_explicit_a16w16_launch")
    assert not hasattr(a16, "_opus_gemm_a16w16_launch_ctypes_raw")
    assert callable(a8._launch_a8w8_backend)
    assert callable(a8._launch_a8w8_gemm)
    assert callable(a8._launch_a8w8_blockscale_gemm)
    assert callable(a8._launch_a8w8_blockscale_bpreshuffle_gemm)
    assert callable(a8._launch_a8w8_mxscale_bmm)
    assert callable(plan_module._get_cached_a8w8_mxscale_bmm_plan)
    assert not hasattr(a8, "_launch_a8w8_exact")
    assert not hasattr(a8, "_launch_a8w8_blockscale_exact")
    assert not hasattr(a8, "_launch_a8w8_blockscale_bpreshuffle_exact")
    assert not hasattr(a8, "_launch_a8w8_mxscale_bmm_exact")
    assert not hasattr(a8, "_launch_a8w8_noscale_gemm")
    assert not hasattr(a8, "_launch_a8w8_noscale_bmm")
    assert not hasattr(a8, "_launch_a8w8_bmm")
    assert not hasattr(a8, "_launch_a8w8_blockscale_bmm")
    assert not hasattr(a8, "_launch_a8w8_blockscale_bpreshuffle_bmm")
    assert not hasattr(a8, "_check_same_device")
    assert not hasattr(a8, "_require_logical_rank")
    assert not hasattr(a8, "_mxscale_bmm_shape_for_plan")
    assert _parameter_names(a16._launch_a16w16_gemm) == (
        "XQ",
        "WQ",
        "Y",
        "bias",
        "kid",
        "split_k",
        "workspace",
        "route_arch",
        "instance",
    )
    assert _parameter_names(a16._launch_a16w16_bmm) == _parameter_names(
        a16._launch_a16w16_gemm
    )
    assert _parameter_names(a8._launch_a8w8_gemm) == (
        "XQ",
        "WQ",
        "Y",
        "kid",
        "route_arch",
        "instance",
    )


def test_a8w8_adapters_use_one_pybind_backend_boundary():
    source_path = _ROOT / "aiter/ops/opus/gemm_op_a8w8.py"
    tree = ast.parse(source_path.read_text())
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    raw_names = {
        "_opus_gemm_a8w8_launch_raw",
        "_opus_gemm_a8w8_blockscale_launch_raw",
        "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
        "_opus_gemm_a8w8_mxscale_bmm_launch_raw",
    }

    raw_call_owners = {name: set() for name in raw_names}
    for owner, definition in definitions.items():
        for call in (
            node for node in ast.walk(definition) if isinstance(node, ast.Call)
        ):
            if isinstance(call.func, ast.Name) and call.func.id in raw_names:
                raw_call_owners[call.func.id].add(owner)

    assert all(
        owners == {"_launch_a8w8_backend"}
        for owners in raw_call_owners.values()
    )
    for adapter in (
        "_launch_a8w8_gemm",
        "_launch_a8w8_blockscale_gemm",
        "_launch_a8w8_blockscale_bpreshuffle_gemm",
        "_launch_a8w8_mxscale_bmm",
    ):
        calls = {
            call.func.id
            for call in ast.walk(definitions[adapter])
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "_launch_a8w8_backend" in calls
        assert calls.isdisjoint(raw_names)

    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "ctypes" for alias in node.names)
        )
        for node in imports
    )


def test_a8w8_mxscale_caller_policy_is_isolated_from_hot_execution():
    policy = importlib.import_module("aiter.ops.opus.policy")
    caller = importlib.import_module("aiter.ops.batched_gemm_op_a8w8")
    policy_source = inspect.getsource(policy)
    caller_source = inspect.getsource(caller)

    assert callable(policy.lookup_mxscale_bmm_config)
    assert callable(policy.resolve_a8w8_mxscale_bmm_plan)
    assert not hasattr(policy.resolve_a8w8_mxscale_bmm_plan, "cache_info")
    assert caller.lookup_mxscale_bmm_config is policy.lookup_mxscale_bmm_config
    assert "def _load_mxscale_bmm_tuned(" in policy_source
    assert "def _heuristic_mxscale_bmm_kid(" in policy_source
    assert "def resolve_a8w8_mxscale_bmm_plan(" in policy_source
    assert "def _load_mxscale_bmm_tuned(" not in caller_source
    assert "def _heuristic_mxscale_bmm_kid(" not in caller_source
    assert "def _resolve_mxscale_bmm_launch(" not in caller_source
    assert "_MXSCALE_BMM_LAUNCH_PLANS" in caller_source
    assert "_resolve_a8w8_mxscale_bmm_plan" in caller_source


def test_a16w16_launch_plan_is_immutable_metadata():
    plan_module = importlib.import_module("aiter.ops.opus.launch_plan")
    spec = plan_module.WorkspaceSpec(shape=(2, 1, 64, 64), dtype=torch.float32)
    plan = plan_module.A16W16LaunchPlan(
        registry_arch="gfx950",
        resolved_kid=200,
        workspace_capacity_split_k=2,
        abi_split_k=2,
        workspace_spec=spec,
    )

    assert is_dataclass(spec) and spec.__dataclass_params__.frozen
    assert is_dataclass(plan) and plan.__dataclass_params__.frozen
    assert plan_module.__all__ == [
        "A16W16LaunchPlan",
        "A8W8MxscaleBMMPlan",
        "WorkspaceSpec",
    ]
    with pytest.raises(FrozenInstanceError):
        plan.abi_split_k = 3


def test_a8w8_launch_plan_is_immutable_metadata():
    plan_module = importlib.import_module("aiter.ops.opus.launch_plan")
    spec = plan_module.WorkspaceSpec(shape=(32768,), dtype=torch.float32)
    plan = plan_module.A8W8MxscaleBMMPlan(
        registry_arch="gfx950",
        resolved_kid=8000,
        abi_split_k=2,
        workspace_spec=spec,
    )

    assert is_dataclass(spec) and spec.__dataclass_params__.frozen
    assert is_dataclass(plan) and plan.__dataclass_params__.frozen
    assert plan_module.__all__ == [
        "A16W16LaunchPlan",
        "A8W8MxscaleBMMPlan",
        "WorkspaceSpec",
    ]
    with pytest.raises(FrozenInstanceError):
        plan.abi_split_k = 3


def test_cpp_pybind_family_abis_remain_exact_kid_and_policy_free():
    header = (_ROOT / "csrc/opus_gemm/include/opus_gemm.h").read_text()
    bmm_header = (_ROOT / "csrc/opus_gemm/include/opus_bmm.h").read_text()
    implementation = (_ROOT / "csrc/opus_gemm/opus_gemm.cu").read_text()
    pybind = (_ROOT / "csrc/include/rocm_ops.hpp").read_text()
    registration = (_ROOT / "csrc/pybind/opus_gemm_pybind.cu").read_text()

    expected = {
        "opus_gemm_a16w16_launch": _A16_RAW_PARAMETERS,
        "opus_gemm_a8w8_launch": ("XQ", "WQ", "Y", "kid"),
        "opus_gemm_a8w8_blockscale_launch": (
            "XQ",
            "WQ",
            "Y",
            "x_scale",
            "w_scale",
            "kid",
        ),
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch": (
            "XQ",
            "WQ",
            "x_scale",
            "w_scale",
            "Y",
            "kid",
        ),
        "opus_gemm_a8w8_mxscale_bmm_launch": (
            "XQ",
            "WQ",
            "Y",
            "x_scale",
            "w_scale",
            "workspace",
            "kid",
            "split_k",
        ),
    }
    macros = {
        "opus_gemm_a16w16_launch": "OPUS_GEMM_A16W16_LAUNCH_PYBIND",
        "opus_gemm_a8w8_launch": "OPUS_GEMM_A8W8_LAUNCH_PYBIND",
        "opus_gemm_a8w8_blockscale_launch": (
            "OPUS_GEMM_A8W8_BLOCKSCALE_LAUNCH_PYBIND"
        ),
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch": (
            "OPUS_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_LAUNCH_PYBIND"
        ),
        "opus_gemm_a8w8_mxscale_bmm_launch": (
            "OPUS_GEMM_A8W8_MXSCALE_BMM_LAUNCH_PYBIND"
        ),
    }
    for function_name, parameters in expected.items():
        declaration_source = (
            bmm_header
            if function_name == "opus_gemm_a8w8_mxscale_bmm_launch"
            else header
        )
        assert _cpp_parameter_names(declaration_source, function_name) == parameters
        assert _pybind_parameter_names(pybind, macros[function_name]) == parameters

    assert _python_definition_parameter_names(
        _ROOT / "aiter/ops/opus/gemm_op_a16w16.py",
        "_opus_gemm_a16w16_launch_raw",
    ) == _A16_RAW_PARAMETERS
    assert "OPUS_GEMM_A16W16_LAUNCH_PYBIND;" in registration
    assert re.search(r"\bvoid\s+opus_gemm\s*\(", header) is None
    assert re.search(r"\bvoid\s+opus_gemm\s*\(", implementation) is None
    assert "OPUS_GEMM_PYBIND" not in pybind
    assert "opus_gemm_a16w16_tune" not in header + implementation + pybind
    assert "opus_gemm_a8w8_blockscale_bpreshuffle_tune" not in (
        header + implementation + pybind
    )


def test_public_module_uses_lazy_family_imports():
    public_source = (_ROOT / "aiter/ops/opus/__init__.py").read_text()
    tree = ast.parse(public_source)
    top_level_family_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module in {"gemm_op_a16w16", "gemm_op_a8w8"}
    ]
    assert top_level_family_imports == []

    contract_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_contract"
    )
    contract_imports = {
        alias.name
        for node in ast.walk(contract_function)
        if isinstance(node, ast.ImportFrom) and node.module is None
        for alias in node.names
    }
    assert contract_imports == {
        "gemm_op_a16w16",
        "gemm_op_a8w8",
        "launch_plan",
    }

    public_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "opus_gemm"
    )
    launch_imports = {
        node.module
        for node in ast.walk(public_function)
        if isinstance(node, ast.ImportFrom)
    }
    assert launch_imports == set()

    assert "kernels_list.get(kid)" in public_source
    assert "get_kernel_route" not in public_source
    assert "_opus_gemm_a8w8_mxscale_bmm_launch_raw" not in public_source
    assert "_launch_a8w8_mxscale_bmm" in public_source
    assert "_FP8_DTYPES" not in public_source
    assert "_A8W8_FAMILY_LAYOUT" not in public_source
    assert "def _validate_a16w16_public_contract(" in public_source
    assert "def _validate_a16w16_public_contract(" not in (
        _ROOT / "aiter/ops/opus/gemm_op_a16w16.py"
    ).read_text()
    assert "def _validate_a8w8_public_contract(" not in (
        _ROOT / "aiter/ops/opus/gemm_op_a8w8.py"
    ).read_text()
    assert "def _validate_a8w8_public_contract(" in (
        _ROOT / "aiter/ops/opus/launch_plan.py"
    ).read_text()

    opus_dir = _ROOT / "aiter/ops/opus"
    assert not (opus_dir / "a16w16_plan.py").exists()
    assert not (opus_dir / "a8w8_plan.py").exists()


def test_production_callers_use_operation_specific_entry_not_family_wrappers():
    callers = {
        "aiter/tuned_gemm.py": "opus_gemm",
        "aiter/ops/gemm_op_a8w8.py": "opus_gemm",
        "aiter/ops/batched_gemm_op_a8w8.py": "opus_bmm",
        "csrc/opus_gemm/opus_gemm_tune.py": "opus_bmm",
        "csrc/opus_gemm/opus_bmm_mxscale_tune.py": "opus_bmm",
        "csrc/gemm_a16w16/gemm_a16w16_tune.py": "opus_gemm",
        "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py": "opus_gemm",
    }
    removed_names = {
        "gemm_a16w16_opus",
        "opus_gemm_a16w16_launch",
        "opus_gemm_a16w16_tune",
        "opus_gemm_a8w8_launch",
        "opus_gemm_a8w8_blockscale_launch",
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
    }
    for relative_path, public_name in callers.items():
        tree = ast.parse((_ROOT / relative_path).read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert any(
            node.module in {"aiter.ops.opus", "opus"}
            and any(alias.name == public_name for alias in node.names)
            for node in imports
        ), relative_path
        assert all(
            alias.name not in removed_names
            for node in imports
            for alias in node.names
        ), relative_path

    deepgemm = (_ROOT / "aiter/ops/deepgemm.py").read_text()
    assert "from .opus.gemm_op_a16w16 import" not in deepgemm


def test_deepgemm_legacy_a16_entry_forwards_to_current_exact_gemm(monkeypatch):
    import aiter

    deepgemm = importlib.import_module("aiter.ops.deepgemm")
    opus = importlib.import_module("aiter.ops.opus")
    calls = []
    sentinel = object()

    def fake_opus_gemm(XQ, WQ, Y, **kwargs):
        calls.append((XQ, WQ, Y, kwargs))
        return sentinel

    monkeypatch.setattr(opus, "opus_gemm", fake_opus_gemm)
    XQ, WQ, Y = object(), object(), object()
    with pytest.warns(DeprecationWarning, match="opus_gemm"):
        result = deepgemm.opus_gemm_a16w16_tune(XQ, WQ, Y, 200, 3)

    assert result is sentinel
    assert calls == [(XQ, WQ, Y, {"kid": 200, "split_k": 3})]
    assert aiter.opus_gemm_a16w16_tune is deepgemm.opus_gemm_a16w16_tune


def test_a8_raw_bindings_do_not_register_dummy_tensor_arguments():
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    raw_names = (
        "_opus_gemm_a8w8_launch_raw",
        "_opus_gemm_a8w8_blockscale_launch_raw",
        "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
        "_opus_gemm_a8w8_mxscale_bmm_launch_raw",
    )

    for name in raw_names:
        assert getattr(a8, name) is not None
        schema = str(getattr(torch.ops.aiter, name).default._schema)
        assert "Tensor dummy" not in schema


def test_device_info_cache_is_scoped_by_explicit_device(monkeypatch):
    arch_module = importlib.import_module("aiter.ops.opus._arch")
    monkeypatch.setattr(arch_module, "_DEVICE_INFO_CACHE", {})
    current_device = [0]
    property_reads = []

    class Properties:
        def __init__(self, arch, cu_num):
            self.gcnArchName = arch
            self.multi_processor_count = cu_num

    devices = {
        torch.device("cuda", 0): Properties("gfx942:sramecc+:xnack-", 304),
        torch.device("cuda", 1): Properties("gfx950:sramecc+:xnack-", 256),
    }

    def get_device_properties(device):
        explicit = torch.device(device)
        property_reads.append(explicit)
        return devices[explicit]

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device[0])

    assert arch_module._device_arch_and_cu("cuda:0") == ("gfx942", 304)
    assert arch_module._device_arch_and_cu(torch.device("cuda", 0)) == (
        "gfx942",
        304,
    )
    assert arch_module._device_arch(torch.device("cuda")) == "gfx942"
    current_device[0] = 1
    assert arch_module._device_arch(torch.device("cuda")) == "gfx950"
    assert arch_module._device_arch_and_cu(1) == ("gfx950", 256)
    assert property_reads == [torch.device("cuda", 0), torch.device("cuda", 1)]


def test_failed_a8_registry_lookup_is_not_cached(monkeypatch):
    plan_module = importlib.import_module("aiter.ops.opus.launch_plan")
    plan_module._require_registered_kid_cached.cache_clear()
    calls = []

    def lookup(arch, family, kid, output_dtype):
        calls.append((arch, family, kid, output_dtype))
        return None if len(calls) == 1 else object()

    monkeypatch.setattr(plan_module, "get_kernel_instance", lookup)
    kwargs = dict(
        arch="gfx950",
        family="a8w8",
        kid=2,
        output_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="no registered OPUS kernel"):
        plan_module._require_registered_kid(**kwargs)
    assert plan_module._require_registered_kid(**kwargs) == 2
    assert len(calls) == 2
    plan_module._require_registered_kid_cached.cache_clear()


def test_registry_counts_routes_and_a8_contracts_are_stable():
    assert len(kernels_list) == 706
    assert Counter(_instance_arch(instance) for instance in kernels_list.values()) == {
        "gfx950": 187,
        "gfx942": 23,
        "gfx1250": 496,
    }
    assert {
        "gfx950_plain_scale": len(a8w8_scale_kernels_list),
        "gfx950_no_scale": len(a8w8_kernels_list),
        "gfx942_non_workspace": len(gfx942_nosplit_kernels_list),
        "gfx942_workspace": len(gfx942_splitk_kernels_list),
        "gfx942_bpreshuffle": len(gfx942_a8w8_kernels_list),
        "gfx1250_plain": len(gfx1250_kernels_list),
        "gfx1250_cluster": len(gfx1250_clusterlaunch_kernels_list),
        "gfx1250_fused": len(gfx1250_splitk_fuse_kernels_list),
    } == {
        "gfx950_plain_scale": 1,
        "gfx950_no_scale": 1,
        "gfx942_non_workspace": 14,
        "gfx942_workspace": 8,
        "gfx942_bpreshuffle": 1,
        "gfx1250_plain": 28,
        "gfx1250_cluster": 468,
        "gfx1250_fused": 0,
    }
    assert kernels_list[1] is a8w8_scale_kernels_list[1]
    assert kernels_list[2] is a8w8_kernels_list[2]
    assert kernels_list[8000] is a8w8_mxscale_bmm_kernels_list[8000]
    assert kernels_list[11000] is gfx942_a8w8_kernels_list[11000]
    assert kernels_list[20000] is gfx1250_kernels_list[20000]
    assert _instance_arch(kernels_list[1]) == "gfx950"
    assert _instance_arch(kernels_list[11000]) == "gfx942"
    assert _instance_arch(kernels_list[20000]) == "gfx1250"
    assert get_kernel_instance("gfx950", "a8w8", 2, torch.float32) is (
        a8w8_kernels_list[2]
    )
    assert get_kernel_instance(
        "gfx942", "a8w8_blockscale_bpreshuffle", 11000, torch.bfloat16
    ) is gfx942_a8w8_kernels_list[11000]


def test_capability_slots_and_default_compile_floor_are_explicit():
    assert OPUS_MANDATORY_A8_KIDS == {
        "gfx950": frozenset({1, 2}),
        "gfx942": frozenset({11000}),
        "gfx1250": frozenset(),
    }
    assert {
        arch: OPUS_KERNEL_TAGS_BY_ARCH_FAMILY[arch][
            "a8w8_blockscale_bpreshuffle"
        ]
        for arch in ("gfx942", "gfx950", "gfx1250")
    } == {
        "gfx942": frozenset({_BPRESHUFFLE_TAG}),
        "gfx950": frozenset(),
        "gfx1250": frozenset(),
    }
    assert DEFAULT_COMPILED_KIDS == frozenset().union(
        *DEFAULT_COMPILED_KIDS_BY_ARCH.values()
    )
    for arch, kids in DEFAULT_COMPILED_KIDS_BY_ARCH.items():
        assert default_compiled_kids_for_arch({arch}) == kids
        assert all(_instance_arch(kernels_list[kid]) == arch for kid in kids)


def test_subset_compile_uses_exact_id_floor_and_arch_filter(tmp_path):
    csv_path = tmp_path / "tuned.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,libtype,solidx\n"
        "gfx950,256,16,16,128,opus,4\n"
        "gfx950,256,16,16,128,ck,5\n"
        "gfx942,304,16,16,128,opus,10000\n"
        "gfx950,256,16,16,128,opus,999999\n"
    )
    sidecar_path = tmp_path / "compiled_kids.json"
    sidecar_path.write_text(json.dumps([6, 10000, 999999]))
    working_path = tmp_path / "generated"
    working_path.mkdir()

    env = os.environ.copy()
    env["GPU_ARCHS"] = "gfx950"
    completed = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "csrc/opus_gemm/gen_instances.py"),
            "--working_path",
            str(working_path),
            "--tune_files",
            str(csv_path),
            "--compiled_kids_sidecar",
            str(sidecar_path),
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    expected = (
        {4, 10000, 999999}
        | {6, 10000, 999999}
        | set(DEFAULT_COMPILED_KIDS)
    ) & set(kernels_list)
    expected = {
        kid for kid in expected if _instance_arch(kernels_list[kid]) == "gfx950"
    }
    expected |= set(OPUS_MANDATORY_A8_KIDS["gfx950"])
    assert set(json.loads(sidecar_path.read_text())) == expected
    assert {1, 2, 4, 6} <= expected
    assert {5, 10000, 999999}.isdisjoint(expected)


def test_unified_docs_separate_caller_policy_from_exact_public_launch():
    paths = (
        _ROOT / "aiter/ops/opus/README.md",
        _ROOT / "csrc/opus_gemm/README.md",
    )
    combined = "\n".join(path.read_text() for path in paths)
    assert "opus_gemm" in combined
    assert "kid" in combined
    assert "caller" in combined.lower()
    assert "no valid row -> skinny (if eligible) -> PyTorch fallback" in combined
    assert "public/C++ path" in combined
    for stale in (
        "gemm_a16w16_opus",
        "opus_gemm_a16w16_tune",
        "opus_gemm_workspace_init",
    ):
        assert stale not in combined


def test_non_opus_arch_does_not_truncate_top_level_import():
    env = os.environ.copy()
    env["GPU_ARCHS"] = "gfx90a"
    env.pop("AITER_REBUILD", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import aiter; assert callable(aiter.opus_gemm); "
            "assert callable(aiter.rmsnorm2d_fwd_with_add)",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
