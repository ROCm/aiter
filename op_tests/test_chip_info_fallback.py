# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""rocminfo/hipinfo fallbacks in chip_info must read the current HIP device.

The tool output and torch device queries are faked so a mixed node (device 0
gfx950 / 256 CUs, device 1 gfx942 / 304 CUs, device 1 selected) is simulated
on any machine.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from aiter.jit.utils import chip_info

_DEVICES = {
    0: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-", multi_processor_count=256),
    1: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-", multi_processor_count=304),
}
# chip_info is also importable as a top-level module (aiter/jit/utils is put on
# sys.path), and the torch custom op behind get_cu_num_custom_op may be bound
# to that copy, so both copies get the same fake tool output.
_MODULES = {id(m): m for m in (chip_info, sys.modules.get("chip_info")) if m}.values()
_CACHED = [
    getattr(m, name)
    for m in _MODULES
    for name in ("_gpu_info_output", "_detect_native", "get_gfx_runtime", "get_cu_num")
]


def _clear_caches():
    for fn in _CACHED:
        fn.cache_clear()


def _fake_tool_output(monkeypatch, text):
    for m in _MODULES:
        monkeypatch.setattr(m, "_gpu_info_output", lambda: text)


@pytest.fixture
def device1_selected_empty_tool(monkeypatch):
    monkeypatch.delenv("GPU_ARCHS", raising=False)
    monkeypatch.delenv("CU_NUM", raising=False)
    _fake_tool_output(monkeypatch, "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _DEVICES[i])
    _clear_caches()
    yield
    monkeypatch.undo()
    _clear_caches()


def test_arch_fallback_reads_current_device(device1_selected_empty_tool):
    assert chip_info._torch_gcn_arch() == "gfx942"
    assert chip_info.get_gfx_runtime() == "gfx942"


def test_build_target_is_not_the_runtime_arch(device1_selected_empty_tool, monkeypatch):
    # No tool output and no torch device: an explicit GPU_ARCHS build target
    # must not be reported as the live GPU, or _needs_arch_rebuild would reject
    # a prebuilt extension on a GPU-less host instead of skipping the check.
    monkeypatch.setenv("GPU_ARCHS", "gfx950")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _clear_caches()
    with pytest.raises(RuntimeError, match="No gfx arch"):
        chip_info.get_gfx_runtime()


def test_cu_fallback_reads_current_device(device1_selected_empty_tool):
    assert chip_info._torch_cu_num() == 304
    assert chip_info.get_cu_num_custom_op() == 304


def test_windows_fallbacks_read_current_device(
    device1_selected_empty_tool, monkeypatch
):
    def missing_hipinfo():
        raise FileNotFoundError("hipinfo")

    for m in _MODULES:
        monkeypatch.setattr(m, "IS_WINDOWS", True)
        monkeypatch.setattr(m, "_gpu_info_output", missing_hipinfo)
    monkeypatch.setattr(torch.version, "hip", "7.2")
    _clear_caches()
    assert chip_info.get_gfx_runtime() == "gfx942"
    assert chip_info.get_cu_num_custom_op() == 304


@pytest.mark.parametrize("wgp_mode, cu_num", [("1", 40), ("0", 20)])
def test_cu_fallback_counts_rdna_cus(
    device1_selected_empty_tool, monkeypatch, wgp_mode, cu_num
):
    # In WGP mode HIP reports workgroup processors, two CUs each.
    rdna = SimpleNamespace(gcnArchName="gfx1151", multi_processor_count=20)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: rdna)
    monkeypatch.setenv("GPU_ENABLE_WGP_MODE", wgp_mode)
    assert chip_info._torch_cu_num() == cu_num
    assert chip_info.get_cu_num_custom_op() == cu_num


def test_mixed_cu_counts_fall_back_to_current_device(
    device1_selected_empty_tool, monkeypatch
):
    if chip_info.IS_WINDOWS:
        pytest.skip("rocminfo-format output")
    mixed = (
        "Agent 1\n  Name: gfx950\n  Compute Unit: 256\n  Device Type: GPU\n"
        "Agent 2\n  Name: gfx942\n  Compute Unit: 304\n  Device Type: GPU\n"
    )
    _fake_tool_output(monkeypatch, mixed)
    assert chip_info.get_cu_num_custom_op() == 304


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
