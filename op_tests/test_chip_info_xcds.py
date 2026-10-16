# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for HIP topology queries and their caches."""

import ctypes
import sys
from pathlib import Path
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
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
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


@pytest.mark.parametrize("version", [70_000_000, 70_100_000])
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
