"""flydsl_moe_sorting_fwd must not share its multiphase workspace between calls.

A workspace cached per device is handed to every CUDA graph that captures a
FlyDSL MoE sort, so graphs replayed at the same time on different streams race
on it. The wrapper itself only needs torch, so buffer ownership is checked here
on CPU with the FlyDSL kernel module replaced by a recorder. The GPU regression
is test_moe_sorting_flydsl_concurrent_graph_replay in op_tests/test_moe_sorting.py.
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_WRAPPER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "aiter"
    / "ops"
    / "flydsl"
    / "moe_sorting.py"
)
_PKG = "_flydsl_moe_sorting_under_test"
_ONESHOT_MAX_T = 16  # the stub's oneshot limit; above it the sort needs a workspace


@pytest.fixture
def sorting(monkeypatch):
    """Load the wrapper against a stub kernel module that records each workspace."""
    launched = []

    def moe_sorting_get_workspace_size(M, num_experts, topk, unit_size):
        return 0 if M <= _ONESHOT_MAX_T else num_experts + 1 + M

    def moe_sorting_flydsl(*args):
        launched.append(args[-1])

    package = types.ModuleType(_PKG)
    package.__path__ = []
    kernels = types.ModuleType(f"{_PKG}.kernels")
    kernels.__path__ = []
    kernel = types.ModuleType(f"{_PKG}.kernels.moe_sorting_kernel")
    kernel.moe_sorting_get_workspace_size = moe_sorting_get_workspace_size
    kernel.moe_sorting_flydsl = moe_sorting_flydsl
    for module in (package, kernels, kernel):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    spec = importlib.util.spec_from_file_location(f"{_PKG}.moe_sorting", _WRAPPER)
    wrapper = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, wrapper)
    spec.loader.exec_module(wrapper)

    def sort(M, num_experts=384, topk=6, unit_size=32):
        topk_ids = torch.zeros((M, topk), dtype=torch.int32)
        wrapper.flydsl_moe_sorting_fwd(
            topk_ids, None, None, None, None, None, None, num_experts, unit_size
        )
        return launched[-1]

    return sort


def test_each_multiphase_sort_gets_its_own_workspace(sorting):
    first = sorting(_ONESHOT_MAX_T + 1)
    second = sorting(_ONESHOT_MAX_T + 1)
    for workspace in (first, second):
        assert workspace.dtype == torch.int32
        assert workspace.numel() == 384 + 1 + _ONESHOT_MAX_T + 1
    assert first.data_ptr() != second.data_ptr()


def test_oneshot_sort_gets_no_workspace(sorting):
    assert sorting(_ONESHOT_MAX_T) is None
