# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for custom all-reduce XGMI topology detection (no GPUs needed).

amdsmi, the PCI address lookup and the CPU-group gather are faked, so the test
exercises the pair checks, the PCI-address mapping and the fallbacks.

    python -m pytest op_tests/multigpu_tests/test_custom_allreduce_topology.py
"""

import importlib
import sys
import types

import pytest

car = importlib.import_module("aiter.dist.device_communicators.custom_all_reduce")

XGMI, PCIE = 2, 1


def fake_amdsmi(addrs, link):
    """amdsmi stand-in. `addrs` lists BDFs in amdsmi enumeration order;
    link(i, j) -> (type, hops) for enumeration indices i, j."""
    m = types.ModuleType("amdsmi")
    m.amdsmi_init = lambda: None
    m.amdsmi_shut_down = lambda: None
    m.amdsmi_get_processor_handles = lambda: list(range(len(addrs)))
    m.amdsmi_get_gpu_device_bdf = lambda h: addrs[h] + ".0"
    m.amdsmi_topo_get_link_type = lambda a, b: dict(zip(("type", "hops"), link(a, b)))
    return m


@pytest.fixture
def env(monkeypatch):
    def setup(rank_addrs, smi):
        monkeypatch.setattr(
            car.dist,
            "all_gather_object",
            lambda out, obj, group=None: out.__setitem__(slice(None), rank_addrs),
        )
        monkeypatch.setattr(car, "_local_pci_address", lambda dev: rank_addrs[0])
        if smi is None:
            monkeypatch.setitem(sys.modules, "amdsmi", None)  # import fails
        else:
            monkeypatch.setitem(sys.modules, "amdsmi", smi)

    return setup


ADDRS = [f"0000:{b:02x}:00" for b in (0x05, 0x44, 0x8E, 0xC9, 0x06, 0x46, 0x90, 0xCA)]


def test_all_xgmi(env):
    env(ADDRS, fake_amdsmi(ADDRS, lambda a, b: (XGMI, 1)))
    assert car._is_full_xgmi(None, None, 8) is True


def test_pcie_only(env):
    env(ADDRS, fake_amdsmi(ADDRS, lambda a, b: (PCIE, 2)))
    assert car._is_full_xgmi(None, None, 8) is False


def test_multi_hop_xgmi_is_not_full(env):
    env(
        ADDRS[:4],
        fake_amdsmi(ADDRS[:4], lambda a, b: (XGMI, 1 if {a, b} != {0, 3} else 2)),
    )
    assert car._is_full_xgmi(None, None, 4) is False


def test_subgroup_mapped_by_pci_address(env):
    # amdsmi enumerates GPUs in a different order than the ranks; XGMI only
    # inside {0x05, 0x44, 0x8e, 0xc9}. A TP4 group on those must pass.
    smi_order = ADDRS[::-1]
    xgmi_set = set(ADDRS[:4])

    def link(a, b):
        both = smi_order[a] in xgmi_set and smi_order[b] in xgmi_set
        return (XGMI, 1) if both else (PCIE, 3)

    env(ADDRS[:4], fake_amdsmi(smi_order, link))
    assert car._is_full_xgmi(None, None, 4) is True
    env(ADDRS[2:6], fake_amdsmi(smi_order, link))
    assert car._is_full_xgmi(None, None, 4) is False


def test_missing_amdsmi_keeps_historical_assumption(env):
    env(ADDRS, None)
    assert car._is_full_xgmi(None, None, 8) is True


def test_unknown_address_keeps_historical_assumption(env):
    env(ADDRS[:3] + [None], fake_amdsmi(ADDRS, lambda a, b: (PCIE, 2)))
    assert car._is_full_xgmi(None, None, 4) is True


def test_pcie_custom_ar_archs():
    assert "gfx950" in car._PCIE_CUSTOM_AR_ARCHS
    assert "gfx1201" not in car._PCIE_CUSTOM_AR_ARCHS
    assert car._PCIE_CAR_MIN_SIZE >= 4 * 8192 * 2  # covers the TP8 LL window
