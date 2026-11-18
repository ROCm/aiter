# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the compute partition and CU count of the current GPU."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aiter" / "jit" / "utils"))
import chip_info


class FakeAmdSmiException(Exception):
    pass


def _gpu(gfx, cu, mode, num_partitions, nps="NPS1", bus=0, secondary_reports=False):
    """The partitions of one GPU.

    Without the amdsmi fix for secondary partitions, only partition 0 reports
    the partition fields.
    """
    return [
        {
            "gfx": gfx,
            "cu": cu,
            "bdf": f"0000:{bus:02x}:00.{i}",
            "mode": mode if i == 0 or secondary_reports else None,
            "n": num_partitions if i == 0 or secondary_reports else None,
            "nps": nps if i == 0 or secondary_reports else None,
        }
        for i in range(num_partitions or 1)
    ]


MI300X_SPX = _gpu("gfx942", 304, "SPX", 1, bus=0x0C) + _gpu("gfx942", 304, "SPX", 1)
MI300X_DPX = _gpu("gfx942", 152, "DPX", 2, "NPS2")
MI300X_CPX = _gpu("gfx942", 38, "CPX", 8)
MI355X_DPX = _gpu("gfx950", 128, "DPX", 2)
MI308X_SPX = _gpu("gfx942", 80, "SPX", 1)
MI210 = _gpu("gfx90a", 104, None, None, None)
MIXED = _gpu("gfx942", 304, "SPX", 1, bus=0x0C) + MI300X_DPX
# A container that is given only partition 1 of a GPU.
SECONDARY_ONLY = MI300X_DPX[1:]
SECONDARY_ONLY_FIXED = _gpu("gfx942", 152, "DPX", 2, "NPS2", secondary_reports=True)[1:]
SECONDARY_ONLY_COUNT = [dict(SECONDARY_ONLY[0], n=2)]


def fake_amdsmi(gpus, missing=()):
    """An amdsmi module whose handles are the given GPUs, in reverse handle order.

    The functions in missing do not exist, as in an older amdsmi.
    """
    mod = ModuleType("amdsmi")
    mod.AmdSmiException = FakeAmdSmiException
    mod.init_calls = 0

    def init():
        mod.init_calls += 1

    mod.amdsmi_init = init
    mod.amdsmi_shut_down = lambda: None
    mod.amdsmi_get_processor_handles = lambda: list(reversed(range(len(gpus))))
    mod.amdsmi_get_gpu_enumeration_info = lambda h: {
        "hip_id": h,
        "hip_uuid": f"GPU-{h:016x}",
    }
    mod.amdsmi_get_gpu_device_bdf = lambda h: gpus[h]["bdf"]
    mod.amdsmi_get_gpu_asic_info = lambda h: {
        "num_compute_units": gpus[h]["cu"],
        "target_graphics_version": gpus[h]["gfx"],
    }

    def profile(h):
        # Like the real binding, which returns N/A fields for NOT_SUPPORTED.
        return {
            "partition_id": [0],
            "partition_profile": {
                "profile_type": gpus[h]["mode"] or "N/A",
                "num_partitions": gpus[h]["n"] or "N/A",
            },
        }

    def reported(field):
        def get(h):
            if gpus[h][field] is None:
                raise FakeAmdSmiException("not supported")
            return gpus[h][field]

        return get

    mod.amdsmi_get_gpu_accelerator_partition_profile = profile
    mod.amdsmi_get_gpu_compute_partition = reported("mode")
    mod.amdsmi_get_gpu_memory_partition = reported("nps")
    for name in missing:
        delattr(mod, name)
    return mod


def _clear_caches():
    chip_info._node_gpus.cache_clear()
    chip_info._cu_num.cache_clear()
    chip_info._shared_cu_num.cache_clear()


@pytest.fixture
def node(monkeypatch):
    for var in (
        "CU_NUM",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 0)

    def install(gpus, missing=()):
        mod = fake_amdsmi(gpus, missing)
        monkeypatch.setitem(sys.modules, "amdsmi", mod)
        _clear_caches()
        return mod

    yield install
    _clear_caches()


@pytest.mark.parametrize(
    "gpus, expected",
    [
        (MI300X_SPX, ("gfx942", 304, 304, "SPX", "NPS1", 1)),
        (MI300X_DPX, ("gfx942", 152, 304, "DPX", "NPS2", 2)),
        (MI300X_CPX, ("gfx942", 38, 304, "CPX", "NPS1", 8)),
        (MI355X_DPX, ("gfx950", 128, 256, "DPX", "NPS1", 2)),
        (MI308X_SPX, ("gfx942", 80, 80, "SPX", "NPS1", 1)),
        (MI210, ("gfx90a", 104, None, None, None, None)),
        (SECONDARY_ONLY, ("gfx942", 152, None, None, None, None)),
        (SECONDARY_ONLY_FIXED, ("gfx942", 152, 304, "DPX", "NPS2", 2)),
        (SECONDARY_ONLY_COUNT, ("gfx942", 152, 304, None, None, 2)),
    ],
)
def test_partition_cu_num_is_the_hip_cu_num(node, gpus, expected):
    """cu_num is the partition's CU count, which HIP reports as multiProcessorCount."""
    node(gpus)
    assert tuple(chip_info.get_partition_info()) == expected
    assert chip_info.get_cu_num() == expected[1]


@pytest.mark.parametrize(
    "missing, expected",
    [
        (
            ("amdsmi_get_gpu_accelerator_partition_profile",),
            ("gfx942", 152, None, "DPX", "NPS2", None),
        ),
        (
            ("amdsmi_get_gpu_enumeration_info",),
            ("gfx942", 152, 304, "DPX", "NPS2", 2),
        ),
        (
            ("amdsmi_get_gpu_memory_partition",),
            ("gfx942", 152, 304, "DPX", None, 2),
        ),
    ],
)
def test_old_amdsmi_keeps_the_fields_it_has(node, missing, expected):
    """A function that an older amdsmi lacks drops only the fields it gives."""
    node(MI300X_DPX, missing)
    assert tuple(chip_info.get_partition_info(1)) == expected


def test_one_amdsmi_session_per_process(node, monkeypatch):
    """The amdsmi session runs once, and get_cu_num is cached per device."""
    amdsmi = node(MIXED)
    assert chip_info.get_cu_num() == 304
    monkeypatch.setattr(chip_info, "_current_device", lambda: 2)
    assert chip_info.get_cu_num() == 152
    for device in range(3):
        chip_info.get_partition_info(device)
    assert amdsmi.init_calls == 1
    assert chip_info._cu_num.cache_info().currsize == 2


def test_same_cu_num_on_every_gpu_needs_no_device_query(node, monkeypatch):
    """The hot path does not ask torch for the current device on a uniform node."""
    node(MI300X_SPX)

    def no_query():
        raise AssertionError("queried the current device")

    monkeypatch.setattr(chip_info, "_current_device", no_query)
    assert chip_info.get_cu_num() == 304


def test_failed_amdsmi_session_is_not_retried(node, monkeypatch):
    amdsmi = node(MI300X_SPX)

    def fail():
        amdsmi.init_calls += 1
        raise FakeAmdSmiException("init failed")

    amdsmi.amdsmi_init = fail
    monkeypatch.setattr(chip_info, "_gpu_info_output", lambda: ROCMINFO)
    for device in (0, 0, 0):
        assert chip_info.get_partition_info(device).cu_num == 128
    assert amdsmi.init_calls == 1


def test_mixed_modes_use_the_current_device(node, monkeypatch):
    node(MIXED)
    assert chip_info.get_partition_info(0).compute_partition == "SPX"
    assert chip_info.get_partition_info(2) == ("gfx942", 152, 304, "DPX", "NPS2", 2)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 1)
    assert chip_info.get_cu_num() == 152


@pytest.mark.parametrize(
    "env, device, physical",
    [
        ({"HIP_VISIBLE_DEVICES": "2"}, 0, 2),
        ({"CUDA_VISIBLE_DEVICES": "2,0"}, 1, 0),
        ({"HIP_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "0"}, 0, 1),
        ({"ROCR_VISIBLE_DEVICES": "2,1", "HIP_VISIBLE_DEVICES": "1"}, 0, 1),
        ({"ROCR_VISIBLE_DEVICES": "gpu-0000000000000002"}, 0, 2),
        ({"HIP_VISIBLE_DEVICES": "GPU-0000000000000002"}, 0, 2),
    ],
)
def test_visible_devices_select_the_handle(node, monkeypatch, env, device, physical):
    node(MIXED)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    gpus, _ = chip_info._node_gpus()
    assert chip_info.get_partition_info(device) == gpus[physical][1]
    assert chip_info.get_cu_num() == chip_info.get_partition_info(0).cu_num


def test_device_outside_visible_devices_raises(node, monkeypatch):
    node(MIXED)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="GPU 1 is not visible"):
        chip_info.get_partition_info(1)


def test_cu_num_env_overrides_only_the_cu_count(node, monkeypatch):
    node(MI300X_SPX)
    monkeypatch.setenv("CU_NUM", "152")
    assert chip_info.get_partition_info() == ("gfx942", 152, 304, "SPX", "NPS1", 1)


ROCMINFO = """\
*******
Agent 1
*******
  Name:                    AMD EPYC 9575F 64-Core Processor
  Device Type:             CPU
  Compute Unit:            96
*******
Agent 3
*******
  Name:                    gfx950
  Uuid:                    GPU-c84490c587b21daa
  Marketing Name:          AMD Instinct MI350X VF
  Device Type:             GPU
  Compute Unit:            128
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-
"""


def test_without_amdsmi_rocminfo_gives_the_cu_count(node, monkeypatch):
    """A DPX partition, as rocminfo prints it on a CI node, without amdsmi."""
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    monkeypatch.setattr(chip_info, "_gpu_info_output", lambda: ROCMINFO)
    _clear_caches()
    assert chip_info.get_partition_info() == ("gfx950", 128, None, None, None, None)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-c84490c587b21daa")
    assert chip_info.get_cu_num() == 128
