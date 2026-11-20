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


def _gpu(
    gfx, cu, mode, num_partitions, nps="NPS1", bus=0, secondary_reports=False, did=None
):
    """The partitions of one GPU.

    Partition 0 owns the render node of the PCI device, the other partitions
    own the render node of an amdgpu_xcp platform device. Without the amdsmi
    fix for secondary partitions, only partition 0 reports the partition fields.
    """
    return [
        {
            "gfx": gfx,
            "cu": cu,
            "bdf": f"0000:{bus:02x}:00.{i}",
            "dev": f"amdgpu_xcp_{bus * 8 + i}" if i else f"0000:{bus:02x}:00.0",
            "mode": mode if i == 0 or secondary_reports else None,
            "n": num_partitions if i == 0 or secondary_reports else None,
            "nps": nps if i == 0 or secondary_reports else None,
            "did": did,
        }
        for i in range(num_partitions or 1)
    ]


MI300X_SPX = _gpu("gfx942", 304, "SPX", 1, bus=0x0C) + _gpu("gfx942", 304, "SPX", 1)
MI300X_DPX = _gpu("gfx942", 152, "DPX", 2, "NPS2")
MI300X_DPX_ID = _gpu("gfx942", 152, "DPX", 2, "NPS2", did=0x74A1)
MI300X_CPX = _gpu("gfx942", 38, "CPX", 8, did=0x74A1)
MI355X_DPX = _gpu("gfx950", 128, "DPX", 2, did=0x75A3)
MI355X_SPX = _gpu("gfx950", 256, "SPX", 1, did=0x75A3)
MI308X_SPX = _gpu("gfx942", 80, "SPX", 1, did=0x74A2)
MI210 = _gpu("gfx90a", 104, None, None, None)
MIXED = _gpu("gfx942", 304, "SPX", 1, bus=0x0C) + MI300X_DPX
# A container that is given only partition 1 of a GPU.
SECONDARY_ONLY = MI300X_DPX[1:]
SECONDARY_ONLY_FIXED = _gpu("gfx942", 152, "DPX", 2, "NPS2", secondary_reports=True)[1:]
SECONDARY_ONLY_COUNT = [dict(SECONDARY_ONLY[0], n=2)]
SECONDARY_ONLY_ID = MI300X_DPX_ID[1:]
# SR-IOV VFs that are functions 0 and 1 of one PCI bus:device. Each VF owns the
# render node of its own PCI function. VF 1 does not report a mode.
TWO_VFS = [
    {
        "gfx": "gfx942",
        "cu": 304,
        "bdf": "0000:85:00.0",
        "dev": "0000:85:00.0",
        "mode": "SPX",
        "n": 1,
        "nps": "NPS1",
    },
    {
        "gfx": "gfx942",
        "cu": 304,
        "bdf": "0000:85:00.1",
        "dev": "0000:85:00.1",
        "mode": None,
        "n": None,
        "nps": None,
    },
]


def _vf(gfx, cu, did, rev=0x00, market="N/A"):
    """An SR-IOV VF guest that reports no partition fields."""
    bdf = "0000:8b:00.1"
    return {
        "gfx": gfx,
        "cu": cu,
        "bdf": bdf,
        "dev": bdf,
        "mode": None,
        "n": None,
        "nps": None,
        "did": did,
        "rev": rev,
        "market": market,
    }


# The GPU of vLLM CI queue amd_mi355_dpx: amd-smi shows Platform "Linux Guest",
# 0000:8b:00.1 "AMD Instinct MI350X VF", Partition-Mode N/A; rocminfo shows
# gfx950, Chip ID 0x75b0, 128 CUs.
MI355X_DPX_VF = _vf("gfx950", 128, 0x75B0, market="AMD Instinct MI350X VF")


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
        "drm_render": 128 + h,
    }
    mod.amdsmi_get_gpu_device_bdf = lambda h: gpus[h]["bdf"]
    mod.amdsmi_get_gpu_asic_info = lambda h: {
        "market_name": gpus[h].get("market", "N/A"),
        "device_id": f"0x{gpus[h]['did']:04x}" if gpus[h].get("did") else "N/A",
        "rev_id": f"0x{gpus[h].get('rev', 0):02x}",
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
    chip_info._query_node_gpus.cache_clear()
    chip_info._cu_num.cache_clear()
    chip_info._shared_cu_num.cache_clear()


@pytest.fixture
def node(monkeypatch, tmp_path):
    for var in (
        "CU_NUM",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 0)

    drm = tmp_path / "drm"
    monkeypatch.setattr(chip_info, "_SYSFS_DRM", str(drm))

    def install(gpus, missing=()):
        for h, gpu in enumerate(gpus):
            (tmp_path / "devices" / gpu["dev"]).mkdir(parents=True, exist_ok=True)
            (drm / f"renderD{128 + h}").mkdir(parents=True)
            (drm / f"renderD{128 + h}" / "device").symlink_to(
                tmp_path / "devices" / gpu["dev"]
            )
        mod = fake_amdsmi(gpus, missing)
        monkeypatch.setitem(sys.modules, "amdsmi", mod)
        _clear_caches()
        return mod

    yield install
    _clear_caches()


@pytest.mark.parametrize(
    "gpus, expected",
    [
        (MI300X_SPX, ("gfx942", 304, 304, "SPX", "NPS1", 1, False)),
        (MI300X_DPX, ("gfx942", 152, 304, "DPX", "NPS2", 2, False)),
        (MI300X_CPX, ("gfx942", 38, 304, "CPX", "NPS1", 8, False)),
        (MI355X_DPX, ("gfx950", 128, 256, "DPX", "NPS1", 2, False)),
        (MI308X_SPX, ("gfx942", 80, 80, "SPX", "NPS1", 1, False)),
        (MI210, ("gfx90a", 104, None, None, None, None, False)),
        (SECONDARY_ONLY, ("gfx942", 152, None, None, None, None, False)),
        (SECONDARY_ONLY_FIXED, ("gfx942", 152, 304, "DPX", "NPS2", 2, False)),
        (SECONDARY_ONLY_COUNT, ("gfx942", 152, 304, None, None, 2, False)),
        (MI300X_DPX_ID, ("gfx942", 152, 304, "DPX", "NPS2", 2, False)),
        (MI355X_SPX, ("gfx950", 256, 256, "SPX", "NPS1", 1, False)),
        (SECONDARY_ONLY_ID, ("gfx942", 152, 304, "DPX", None, 2, True)),
    ],
)
def test_partition_cu_num_is_the_hip_cu_num(node, gpus, expected):
    """cu_num is the partition's CU count, which HIP reports as multiProcessorCount."""
    node(gpus)
    assert tuple(chip_info.get_partition_info()) == expected
    assert chip_info.get_cu_num() == expected[1]


@pytest.mark.parametrize(
    "gpu, expected",
    [
        (MI355X_DPX_VF, ("gfx950", 128, 256, "DPX", None, 2, True)),
        (_vf("gfx950", 256, 0x75B3), ("gfx950", 256, 256, "SPX", None, 1, True)),
        (_vf("gfx950", 128, 0x75A8), ("gfx950", 128, 128, "SPX", None, 1, True)),
        (_vf("gfx950", 128, 0x75B8), ("gfx950", 128, 128, "SPX", None, 1, True)),
        (_vf("gfx942", 80, 0x74A2), ("gfx942", 80, 80, "SPX", None, 1, True)),
        (_vf("gfx942", 228, 0x74A0), ("gfx942", 228, 228, "SPX", None, 1, True)),
        (_vf("gfx942", 152, 0x74A2, 0xA1), ("gfx942", 152, 304, "DPX", None, 2, True)),
        (_vf("gfx942", 76, 0x74B5), ("gfx942", 76, 304, "QPX", None, 4, True)),
        (_vf("gfx942", 38, 0x74B9), ("gfx942", 38, 304, "CPX", None, 8, True)),
        (
            _vf("gfx942", 152, 0x74A1, 0x01),
            ("gfx942", 152, None, None, None, None, False),
        ),
        (_vf("gfx950", 128, 0x1234), ("gfx950", 128, None, None, None, None, False)),
    ],
)
def test_partition_inferred_from_full_chip_cu_num(node, gpu, expected):
    """Without a reported mode, 1/2, 1/4 or 1/8 of the CUs of a known chip is a partition.

    All of the CUs of a known chip is SPX. MI350P, MI308X and MI300A are full
    chips with their own device IDs, so their CU counts are SPX.
    """
    node([gpu])
    assert chip_info.get_partition_info() == expected


@pytest.mark.parametrize(
    "missing, expected",
    [
        (
            ("amdsmi_get_gpu_accelerator_partition_profile",),
            ("gfx942", 152, None, "DPX", "NPS2", None, False),
        ),
        (
            ("amdsmi_get_gpu_enumeration_info",),
            ("gfx942", 152, 304, "DPX", "NPS2", 2, False),
        ),
        (
            ("amdsmi_get_gpu_memory_partition",),
            ("gfx942", 152, 304, "DPX", None, 2, False),
        ),
    ],
)
def test_old_amdsmi_keeps_the_fields_it_has(node, missing, expected):
    """A function that an older amdsmi lacks drops only the fields it gives."""
    node(MI300X_DPX, missing)
    assert tuple(chip_info.get_partition_info(1)) == expected


def test_vf_does_not_take_the_mode_of_another_function(node):
    """Only a secondary partition takes fields from partition 0 of its GPU."""
    node(TWO_VFS)
    assert chip_info.get_partition_info(0) == (
        "gfx942",
        304,
        304,
        "SPX",
        "NPS1",
        1,
        False,
    )
    assert chip_info.get_partition_info(1) == (
        "gfx942",
        304,
        None,
        None,
        None,
        None,
        False,
    )


def test_partition_without_render_device_takes_no_fields(node, monkeypatch, tmp_path):
    """A partition that sysfs does not show as an amdgpu_xcp device keeps its fields."""
    node(MI300X_DPX)
    monkeypatch.setattr(chip_info, "_SYSFS_DRM", str(tmp_path / "missing"))
    assert chip_info.get_partition_info(1) == (
        "gfx942",
        152,
        None,
        None,
        None,
        None,
        False,
    )


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


def test_failed_rocminfo_fallback_is_not_retried(node, monkeypatch):
    """When amdsmi and rocminfo both fail, each later call raises without a new query."""
    amdsmi = node(MI300X_SPX)
    calls = {"amdsmi": 0, "rocminfo": 0}

    def fail_amdsmi():
        calls["amdsmi"] += 1
        raise FakeAmdSmiException("init failed")

    def fail_rocminfo():
        calls["rocminfo"] += 1
        raise FileNotFoundError("rocminfo")

    amdsmi.amdsmi_init = fail_amdsmi
    monkeypatch.setattr(chip_info, "_gpu_info_output", fail_rocminfo)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="rocminfo failed"):
            chip_info.get_partition_info()
    assert calls == {"amdsmi": 1, "rocminfo": 1}


def test_mixed_modes_use_the_current_device(node, monkeypatch):
    node(MIXED)
    assert chip_info.get_partition_info(0).compute_partition == "SPX"
    assert chip_info.get_partition_info(2) == (
        "gfx942",
        152,
        304,
        "DPX",
        "NPS2",
        2,
        False,
    )
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
    assert chip_info.get_partition_info() == (
        "gfx942",
        152,
        304,
        "SPX",
        "NPS1",
        1,
        False,
    )


def test_cu_num_env_does_not_change_the_inferred_partition(node, monkeypatch):
    node([MI355X_DPX_VF])
    monkeypatch.setenv("CU_NUM", "256")
    assert chip_info.get_partition_info() == ("gfx950", 256, 256, "DPX", None, 2, True)


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
    assert chip_info.get_partition_info() == (
        "gfx950",
        128,
        None,
        None,
        None,
        None,
        False,
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-c84490c587b21daa")
    assert chip_info.get_cu_num() == 128


MI300A_SPX = _gpu("gfx942", 228, "SPX", 1)
MI350P_SPX = _gpu("gfx950", 128, "SPX", 1)


@pytest.fixture
def tuned_node(node, monkeypatch):
    monkeypatch.delenv("GPU_ARCHS", raising=False)

    def install(gpus):
        node(gpus)
        chip_info._tuned_cu_nums_of_device.cache_clear()
        chip_info._shared_tuned_cu_nums.cache_clear()

    yield install
    chip_info._tuned_cu_nums_of_device.cache_clear()
    chip_info._shared_tuned_cu_nums.cache_clear()


@pytest.mark.parametrize(
    "gpus, cu_nums",
    [
        (MI300X_DPX, (152, 304)),
        (MI300X_CPX, (38, 304)),
        (MI355X_DPX, (128, 256)),
        ([MI355X_DPX_VF], (128, 256)),
        (SECONDARY_ONLY_ID, (152, 304)),
        (MI300X_SPX, (304,)),
        (MI308X_SPX, (80,)),
        (MI300A_SPX, (228,)),
        (MI350P_SPX, (128,)),
        (MI210, (104,)),
        (SECONDARY_ONLY, (152,)),
    ],
)
def test_only_partitions_use_full_chip_rows(tuned_node, gpus, cu_nums):
    """A partition falls back to the rows of its full chip; other GPUs match exactly."""
    tuned_node(gpus)
    assert chip_info.get_tuned_cu_nums() == cu_nums
    assert chip_info.get_tuned_cu_nums(cu_nums[0]) == cu_nums
    assert chip_info.get_tuned_cu_nums(cu_nums[0] + 1) == (cu_nums[0] + 1,)


def test_each_device_uses_its_own_partition(tuned_node, monkeypatch):
    tuned_node(MIXED)
    assert chip_info.get_tuned_cu_nums(device=0) == (304,)
    assert chip_info.get_tuned_cu_nums(152, device=2) == (152, 304)
    assert chip_info.get_tuned_cu_nums(152, device=0) == (152,)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 2)
    assert chip_info.get_tuned_cu_nums(152) == (152, 304)


def test_same_gpus_need_no_device_query(tuned_node, monkeypatch):
    """Tuned lookups run per call; on a node of one GPU kind they query no device."""
    tuned_node(MI300X_DPX)
    calls = []
    monkeypatch.setattr(chip_info, "_current_device", lambda: calls.append(0) or 0)
    for _ in range(3):
        assert chip_info.get_tuned_cu_nums(152) == (152, 304)
    assert calls == []


def test_cu_mask_emulation_is_not_a_partition(tuned_node, monkeypatch):
    tuned_node(MI300X_SPX)
    monkeypatch.setenv("CU_NUM", "152")
    assert chip_info.get_tuned_cu_nums() == (152,)


def test_partition_builds_full_chip_rows(tuned_node, monkeypatch):
    tuned_node(MI300X_DPX)
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx942")
    assert chip_info.get_build_targets() == [("gfx942", 152), ("gfx942", 304)]


def test_find_tuned_cu_row_prefers_the_own_cu_count(tuned_node):
    tuned_node(MI300X_DPX)
    table = {
        ("gfx942", 152, 1): "own",
        ("gfx942", 304, 1): "full",
        ("gfx942", 304, 2): "full",
    }
    key = lambda shape: lambda cu: ("gfx942", cu, shape)
    assert chip_info.find_tuned_cu_row(table, key(1), "gfx942", 152, "t") == "own"
    assert chip_info.find_tuned_cu_row(table, key(2), "gfx942", 152, "t") == "full"
    assert chip_info.find_tuned_cu_row(table, key(3), "gfx942", 152, "t") is None
    assert chip_info.find_tuned_cu_row(table, key(2), "gfx942", 80, "t") is None
