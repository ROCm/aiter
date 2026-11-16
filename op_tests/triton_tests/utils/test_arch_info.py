# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest
import triton

from aiter.ops.triton.utils._triton import arch_info


def _fake_kfd(monkeypatch, tmp_path, versions):
    for i, ver in enumerate(versions):
        node = tmp_path / str(i)
        node.mkdir()
        (node / "properties").write_text(
            f"cpu_cores_count 0\ngfx_target_version {ver}\nsimd_count 4\n"
        )
    monkeypatch.setattr(arch_info, "_KFD_NODES", str(tmp_path))
    monkeypatch.delenv("TRITON_OVERRIDE_ARCH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)


@pytest.mark.parametrize(
    "ver, arch",
    [
        (90010, "gfx90a"),
        (90402, "gfx942"),
        (90500, "gfx950"),
        (110000, "gfx1100"),
        (110501, "gfx1151"),
        (120001, "gfx1201"),
    ],
)
def test_kfd_arch_decode(monkeypatch, tmp_path, ver, arch):
    # Node 0 is a CPU node: version 0, so the probe skips it.
    _fake_kfd(monkeypatch, tmp_path, [0, ver, ver])
    assert arch_info._probe_kfd_arch() == arch


@pytest.mark.parametrize(
    "versions",
    [
        [0, 90402, 90500],  # mixed archs
        [0, 120500],  # gfx1250 can also be gfx1250-strict
        [0],  # no GPU node
    ],
)
def test_kfd_arch_falls_back(monkeypatch, tmp_path, versions):
    _fake_kfd(monkeypatch, tmp_path, versions)
    assert arch_info._probe_kfd_arch() is None


def test_kfd_arch_missing_sysfs(monkeypatch, tmp_path):
    _fake_kfd(monkeypatch, tmp_path / "missing", [])
    assert arch_info._probe_kfd_arch() is None


@pytest.mark.parametrize(
    "env, value",
    [("TRITON_OVERRIDE_ARCH", "gfx950"), ("HSA_OVERRIDE_GFX_VERSION", "9.5.0")],
)
def test_kfd_arch_override_env(monkeypatch, tmp_path, env, value):
    _fake_kfd(monkeypatch, tmp_path, [0, 90402])
    monkeypatch.setenv(env, value)
    assert arch_info._probe_kfd_arch() is None


@pytest.mark.skipif(
    bool(
        os.environ.get("TRITON_OVERRIDE_ARCH")
        or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    ),
    reason="arch override is set",
)
def test_kfd_arch_matches_driver():
    # On a real host, the sysfs arch must match the arch of the Triton driver.
    arch = arch_info._probe_kfd_arch()
    if arch is None:
        pytest.skip(f"{arch_info._KFD_NODES} has no single usable arch")
    assert arch == triton.runtime.driver.active.get_current_target().arch
