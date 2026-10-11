"""CPU bandwidth limits must constrain fan-out even without a cpuset."""

import os
from fractions import Fraction
from unittest.mock import patch

import pytest

import aiter_worker_limits as limits


@pytest.fixture
def hierarchy(tmp_path, monkeypatch):
    mount = tmp_path / "cgroup"
    child = mount / "job"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    mountinfo = tmp_path / "mountinfo"
    membership.write_text("0::/job\n")
    mountinfo.write_text(f"36 25 0:32 / {mount} rw - cgroup2 none rw\n")
    monkeypatch.setattr(limits, "_PROC_SELF_CGROUP_PATH", str(membership))
    monkeypatch.setattr(limits, "_PROC_SELF_MOUNTINFO_PATH", str(mountinfo))
    monkeypatch.setattr(limits, "_process_cpu_count", lambda: 192)
    monkeypatch.setattr(limits, "_host_available_memory_bytes", lambda: 1024**4)
    monkeypatch.delenv("AITER_MAX_JOBS", raising=False)
    monkeypatch.delenv("MAX_JOBS", raising=False)
    return mount, child, membership, mountinfo


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("800000 100000", 6),
        ("250000 100000", 2),
        ("50000 100000", 1),
        ("100000 100000", 1),
        ("max 100000", 153),
        ("0 100000", 153),
        ("100000 0", 153),
        ("-1 100000", 153),
        ("garbage", 153),
        ("100000 -1", 153),
        ("100000 x", 153),
        ("", 153),
        ("800000 100000 extra", 153),
    ],
)
def test_v2_quota_and_invalid_fallback(hierarchy, raw, expected):
    _, child, _, _ = hierarchy
    (child / "cpu.max").write_text(raw)
    assert limits.get_cpu_worker_budget() == expected
    assert limits.get_worker_count() == expected


@pytest.mark.parametrize(
    "quota,period,expected",
    [
        ("800000", "100000", 6),
        ("250000", "100000", 2),
        ("-1", "100000", 153),
        ("100000", "0", 153),
        ("bad", "100000", 153),
    ],
)
def test_v1_cpu_controller_combined_mount(hierarchy, quota, period, expected):
    mount, child, membership, mountinfo = hierarchy
    membership.write_text("4:cpu,cpuacct:/container/job\n5:memory:/different\n")
    mountinfo.write_text(
        f"29 23 0:26 /container {mount} rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    (child / "cpu.cfs_quota_us").write_text(quota)
    (child / "cpu.cfs_period_us").write_text(period)
    assert limits.get_cpu_worker_budget() == expected


def test_ancestor_limit_survives_unlimited_or_bad_child(hierarchy):
    mount, child, _, _ = hierarchy
    (mount / "cpu.max").write_text("400000 100000")
    for value in ("max 100000", "bad", "800000 100000"):
        (child / "cpu.max").write_text(value)
        assert limits.get_cpu_worker_budget() == 3
    (child / "cpu.max").unlink()
    assert limits.get_cpu_worker_budget() == 3
    # A tighter child still wins, even with a different period.
    (child / "cpu.max").write_text("125000 50000")
    assert limits._cgroup_cpu_quota() == Fraction(5, 2)
    assert limits.get_cpu_worker_budget() == 2


def test_hybrid_scans_both_mounts_and_decodes_spaces(hierarchy, tmp_path):
    mount, child, membership, mountinfo = hierarchy
    v1 = tmp_path / "cpu mount"
    v1.mkdir()
    membership.write_text("0::/job\n4:cpu,cpuacct:/\n")
    escaped = str(v1).replace(" ", r"\040")
    mountinfo.write_text(
        f"36 25 0:32 / {mount} rw - cgroup2 none rw\n29 23 0:26 / {escaped} rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    (child / "cpu.max").write_text("max 100000")
    (v1 / "cpu.cfs_quota_us").write_text("800000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert limits.get_cpu_worker_budget() == 6


def test_affinity_memory_and_user_ceilings_still_apply(hierarchy, monkeypatch):
    _, child, _, _ = hierarchy
    (child / "cpu.max").write_text("800000 100000")
    monkeypatch.setattr(limits, "_process_cpu_count", lambda: 4)
    assert limits.get_worker_count() == 3
    monkeypatch.setattr(limits, "_process_cpu_count", lambda: 192)
    monkeypatch.setenv("AITER_MAX_JOBS", "999")
    assert limits.get_worker_count() == 6
    monkeypatch.setenv("AITER_MAX_JOBS", "2")
    assert limits.get_worker_count() == 2
    monkeypatch.delenv("AITER_MAX_JOBS")
    monkeypatch.setenv("MAX_JOBS", "2")
    before = dict(os.environ)
    assert limits.get_compile_worker_count() == 2
    assert limits.get_worker_count() == 6
    assert dict(os.environ) == before
    monkeypatch.setattr(
        limits, "_host_available_memory_bytes", lambda: limits.EST_WORKER_RSS_BYTES
    )
    assert limits.get_worker_count() == 1


def test_quota_is_reread_and_not_exported(hierarchy):
    _, child, _, _ = hierarchy
    before = dict(os.environ)
    for raw, expected in [
        ("800000 100000", 6),
        ("250000 100000", 2),
        ("max 100000", 153),
    ]:
        (child / "cpu.max").write_text(raw)
        assert limits.get_worker_count() == expected
    assert dict(os.environ) == before


def test_unreadable_child_does_not_hide_parent(hierarchy):
    mount, child, _, _ = hierarchy
    (mount / "cpu.max").write_text("800000 100000")
    original = open

    def read(path, *args, **kwargs):
        if str(path) == str(child / "cpu.max"):
            raise PermissionError("masked")
        return original(path, *args, **kwargs)

    with patch("builtins.open", side_effect=read):
        assert limits.get_cpu_worker_budget() == 6


def test_no_proc_metadata_falls_back_to_affinity(hierarchy):
    _, _, membership, mountinfo = hierarchy
    membership.unlink()
    mountinfo.unlink()
    assert limits.get_cpu_worker_budget() == 153
