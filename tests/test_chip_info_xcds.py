# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for HIP topology queries and their caches."""

import ctypes
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# Import the utilities directly, without importing GPU-dependent aiter ops.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aiter" / "jit" / "utils"))
import chip_info
import hip_runtime


def _hip_result(value, error=0):
    def query(output, *args):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = value
        return error

    return query


@pytest.fixture
def hip(monkeypatch):
    lib = mock.Mock()
    lib.hipGetDevice.side_effect = _hip_result(0)
    lib.hipRuntimeGetVersion.side_effect = _hip_result(70_100_123)
    lib.hipDeviceGetAttribute.side_effect = _hip_result(4)
    monkeypatch.setattr(hip_runtime, "load_hip_runtime", lambda: lib)
    monkeypatch.setattr(chip_info, "load_hip_runtime", lambda: lib)
    archs = {0: "gfx950:sramecc+:xnack-", 1: "gfx950:sramecc+:xnack-"}
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda device: SimpleNamespace(gcnArchName=archs[device]),
    )
    lib.archs = archs
    chip_info._num_xcds_for_device.cache_clear()
    hip_runtime._query_hip_runtime_version.cache_clear()
    try:
        yield lib
    finally:
        chip_info._num_xcds_for_device.cache_clear()
        hip_runtime._query_hip_runtime_version.cache_clear()


@pytest.mark.parametrize("error", [1, 101, 801, 999])
def test_attribute_failure_is_retried_before_caching_topology(hip, error):
    hip.hipDeviceGetAttribute.side_effect = _hip_result(8, error)
    with pytest.raises(
        RuntimeError, match=rf"NumberOfXccs\) failed with error {error} for device 0"
    ):
        chip_info.get_device_name()
    assert chip_info._num_xcds_for_device.cache_info().currsize == 0

    hip.hipDeviceGetAttribute.side_effect = _hip_result(4)
    assert chip_info.get_device_name() == "MI350P"
    assert chip_info.get_num_xcds() == 4
    assert chip_info.get_num_xcds() == 4
    assert hip.hipDeviceGetAttribute.call_count == 2
    assert hip.hipDeviceGetAttribute.call_args.args[1:] == (10018, 0)
    assert hip.hipRuntimeGetVersion.call_count == 1


def test_unknown_version_does_not_query_or_cache_topology(hip):
    hip.hipRuntimeGetVersion.side_effect = _hip_result(0, 999)
    with pytest.raises(RuntimeError, match="Could not read the HIP runtime version"):
        chip_info.get_num_xcds(default=None)
    hip.hipDeviceGetAttribute.assert_not_called()
    assert chip_info._num_xcds_for_device.cache_info().currsize == 0

    hip.hipRuntimeGetVersion.side_effect = _hip_result(70_100_123)
    assert chip_info.get_device_name() == "MI350P"
    assert chip_info.get_num_xcds() == 4
    assert hip_runtime.get_hip_runtime_version() == (7, 1, 123)
    assert hip.hipRuntimeGetVersion.call_count == 2
    hip.hipDeviceGetAttribute.assert_called_once()


@pytest.mark.parametrize("failure", [999, OSError("runtime unavailable")])
def test_version_failure_returns_none_and_remains_retryable(hip, failure):
    hip.hipRuntimeGetVersion.side_effect = (
        failure if isinstance(failure, Exception) else _hip_result(0, failure)
    )
    assert hip_runtime.get_hip_runtime_version() is None

    hip.hipRuntimeGetVersion.side_effect = _hip_result(70_100_123)
    assert hip_runtime.get_hip_runtime_version() == (7, 1, 123)
    assert hip_runtime.get_hip_runtime_version() == (7, 1, 123)
    assert hip.hipRuntimeGetVersion.call_count == 2


def test_runtime_load_failure_remains_retryable(hip, monkeypatch):
    load = mock.Mock(side_effect=[OSError("runtime unavailable"), hip])
    monkeypatch.setattr(hip_runtime, "load_hip_runtime", load)
    assert hip_runtime.get_hip_runtime_version() is None
    assert hip_runtime.get_hip_runtime_version() == (7, 1, 123)
    assert hip_runtime.get_hip_runtime_version() == (7, 1, 123)
    assert load.call_count == 2


@pytest.mark.parametrize("version", [60_000_000, 60_400_000])
def test_legacy_runtime_falls_back_without_querying_attribute(hip, version):
    hip.hipRuntimeGetVersion.side_effect = _hip_result(version)
    with mock.patch.object(chip_info.logger, "warning") as warning:
        assert chip_info.get_num_xcds() == 8
        assert chip_info.get_num_xcds() == 8
        assert chip_info.get_num_xcds(default=7) == 7
    warning.assert_called_once()
    assert "predates hipDeviceAttributeNumberOfXccs" in warning.call_args.args[0]
    hip.hipDeviceGetAttribute.assert_not_called()
    hip.hipRuntimeGetVersion.assert_called_once()


@pytest.mark.parametrize("version", [70_000_000, 70_100_000, 71_526_333])
def test_supported_runtime_queries_attribute(hip, version):
    hip.hipRuntimeGetVersion.side_effect = _hip_result(version)
    assert chip_info.get_num_xcds() == 4
    hip.hipDeviceGetAttribute.assert_called_once()


@pytest.mark.parametrize("count", [0, -1, 65])
def test_invalid_count_is_not_cached(hip, count):
    hip.hipDeviceGetAttribute.side_effect = _hip_result(count)
    with pytest.raises(RuntimeError, match="not a usable XCD count"):
        chip_info.get_num_xcds(default=None)

    hip.hipDeviceGetAttribute.side_effect = _hip_result(4)
    assert chip_info.get_num_xcds() == 4
    assert hip.hipDeviceGetAttribute.call_count == 2


def test_counts_are_cached_per_current_device(hip):
    def query(output, attribute, device):
        assert attribute == 10018
        return _hip_result({0: 4, 1: 8}[device])(output)

    hip.hipDeviceGetAttribute.side_effect = query
    assert chip_info.get_device_name() == "MI350P"
    hip.hipGetDevice.side_effect = _hip_result(1)
    assert chip_info.get_device_name() == "MI350"
    assert chip_info.get_num_xcds() == 8
    hip.hipGetDevice.side_effect = _hip_result(0)
    assert chip_info.get_num_xcds() == 4
    assert hip.hipDeviceGetAttribute.call_count == 2


def test_current_device_failure_does_not_cache_topology(hip):
    hip.hipGetDevice.side_effect = _hip_result(0, 999)
    with pytest.raises(RuntimeError, match="hipGetDevice failed"):
        chip_info.get_num_xcds(default=None)
    hip.hipDeviceGetAttribute.assert_not_called()

    hip.hipGetDevice.side_effect = _hip_result(0)
    assert chip_info.get_num_xcds() == 4
    hip.hipDeviceGetAttribute.assert_called_once()


def test_legacy_runtime_raises_without_default(hip):
    hip.hipRuntimeGetVersion.side_effect = _hip_result(60_400_000)
    with pytest.raises(RuntimeError, match="predates hipDeviceAttributeNumberOfXccs"):
        chip_info.get_num_xcds(default=None)
    with pytest.raises(RuntimeError, match="predates hipDeviceAttributeNumberOfXccs"):
        chip_info.get_device_name()
    hip.hipDeviceGetAttribute.assert_not_called()


def test_default_is_returned_on_failure_without_caching(hip):
    chip_info._warn_xcd_fallback.cache_clear()
    hip.hipDeviceGetAttribute.side_effect = _hip_result(4, 1)
    with mock.patch.object(chip_info.logger, "warning") as warning:
        assert chip_info.get_num_xcds() == 8
        assert chip_info.get_num_xcds() == 8
    warning.assert_called_once()
    assert chip_info._num_xcds_for_device.cache_info().currsize == 0

    hip.hipDeviceGetAttribute.side_effect = _hip_result(4)
    assert chip_info.get_num_xcds() == 4
    assert chip_info.get_num_xcds(default=None) == 4
    assert hip.hipDeviceGetAttribute.call_count == 3


def test_name_follows_the_current_device_arch(hip):
    hip.archs[1] = "gfx942:sramecc+:xnack-"

    def attribute(output, attribute, device):
        value = {10018: 4, 10020: 0x74A2}[attribute]
        return _hip_result(value)(output)

    hip.hipDeviceGetAttribute.side_effect = attribute
    assert chip_info.get_device_name() == "MI350P"
    hip.hipGetDevice.side_effect = _hip_result(1)
    assert chip_info.get_device_name() == "MI308"
    assert hip.hipDeviceGetAttribute.call_args.args[1:] == (10020, 1)


@pytest.fixture
def fresh_loader():
    hip_runtime.load_hip_runtime.cache_clear()
    try:
        yield
    finally:
        hip_runtime.load_hip_runtime.cache_clear()


def _fake_cdll(loaded):
    """ctypes.CDLL stand-in: names in `loaded` resolve under RTLD_NOLOAD."""

    def cdll(name, mode=0):
        if mode & hip_runtime.os.RTLD_NOLOAD:
            if name not in loaded:
                raise OSError(f"{name}: not loaded")
            return f"loaded:{name}"
        return f"opened:{name}"

    return mock.Mock(side_effect=cdll)


def test_loader_binds_to_the_already_loaded_runtime(fresh_loader, monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    cdll = _fake_cdll({"libamdhip64.so.7"})
    monkeypatch.setattr(hip_runtime.ctypes, "CDLL", cdll)
    assert hip_runtime.load_hip_runtime() == "loaded:libamdhip64.so.7"
    assert all(
        c.kwargs["mode"] & hip_runtime.os.RTLD_NOLOAD for c in cdll.call_args_list
    )


def test_loader_binds_to_a_runtime_named_by_torch_hip_version(
    fresh_loader, monkeypatch
):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    torch = SimpleNamespace(version=SimpleNamespace(hip="123.45.67"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hip_runtime.ctypes, "CDLL", _fake_cdll({"libamdhip64.so.123"}))
    assert hip_runtime.load_hip_runtime() == "loaded:libamdhip64.so.123"


def test_loader_opens_by_name_when_nothing_is_loaded(fresh_loader, monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    monkeypatch.setattr(hip_runtime.ctypes, "CDLL", _fake_cdll(set()))
    assert hip_runtime.load_hip_runtime() == "opened:libamdhip64.so"


def test_loader_does_not_probe_loaded_runtimes_on_windows(fresh_loader, monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "win32")
    monkeypatch.setattr(hip_runtime, "_library_names", lambda: ("amdhip64_7.dll",))
    cdll = mock.Mock(return_value="dll")
    monkeypatch.setattr(hip_runtime.ctypes, "CDLL", cdll)
    assert hip_runtime.load_hip_runtime() == "dll"
    cdll.assert_called_once_with("amdhip64_7.dll")


@pytest.mark.parametrize(
    "version, attribute",
    [(70_000_000, 10019), (70_100_000, 10020), (71_526_333, 10020)],
)
def test_pci_chip_id_attribute_follows_runtime_version(hip, version, attribute):
    hip.archs[0] = "gfx942:sramecc+:xnack-"
    hip.hipRuntimeGetVersion.side_effect = _hip_result(version)
    hip.hipDeviceGetAttribute.side_effect = _hip_result(0x74A2)
    assert chip_info.get_device_name() == "MI308"
    assert hip.hipDeviceGetAttribute.call_args.args[1:] == (attribute, 0)


def test_pci_chip_id_raises_on_unknown_version(hip):
    hip.archs[0] = "gfx942:sramecc+:xnack-"
    hip.hipRuntimeGetVersion.side_effect = _hip_result(0, 999)
    with pytest.raises(RuntimeError, match="PciChipId attribute of device 0"):
        chip_info.get_device_name()
    hip.hipDeviceGetAttribute.assert_not_called()


def test_windows_library_names_search_the_rocm_home(monkeypatch, tmp_path):
    monkeypatch.setattr(hip_runtime.sys, "platform", "win32")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "amdhip64_7.dll").touch()
    rocm_home = SimpleNamespace(_find_rocm_home=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "cpp_extension", rocm_home)
    assert hip_runtime._library_names() == (
        str(tmp_path / "bin" / "amdhip64_7.dll"),
        "amdhip64.dll",
    )

    rocm_home._find_rocm_home = lambda: None
    assert hip_runtime._library_names() == ("amdhip64.dll",)
