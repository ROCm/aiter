"""Behavioral coverage for diagnostic thresholds and concurrent reporting."""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

import aiter_worker_limits as limits


@pytest.fixture(autouse=True)
def isolated_diagnostics():
    with patch.dict(os.environ, {}, clear=True), patch.object(
        limits, "_memory_diagnostic_last_time", None
    ), patch.object(limits, "_memory_diagnostic_last_budget", None):
        yield


@pytest.fixture
def cgroup(tmp_path):
    (tmp_path / "memory.max").write_text(str(64 * 1024**3))
    (tmp_path / "memory.current").write_text(str(52 * 1024**3))
    (tmp_path / "memory.stat").write_text("anon 1073741824\ninactive_file 2147483648\n")
    with patch.object(
        limits, "_cgroup_memory_directories", return_value=[("v2", str(tmp_path))]
    ), patch.object(
        limits, "_host_available_memory_bytes", return_value=1024**4
    ), patch.object(
        limits, "get_cpu_worker_budget", return_value=204
    ):
        yield tmp_path


def test_real_cgroup_reports_eight_workers_and_cache(cgroup, caplog):
    assert limits.get_worker_count() == 8
    assert limits.get_worker_count() == 8
    assert len(caplog.records) == 1
    message = caplog.text
    for expected in (
        "requested=204",
        "selected=8",
        "limit=64.00",
        "current=52.00",
        "inactive_file",
        "configured_ceiling=unset",
    ):
        assert expected in message


@pytest.mark.parametrize(
    "cpu,memory,ceiling,reported",
    [
        (204, 8, None, True),
        (32, 8, None, True),
        (31, 8, None, False),
        (204, 8, "4", False),
        (4, 2, None, True),
        (1, 1, None, False),
        (204, 1, "1", False),
        (204, 1, "0", False),
        (204, 1, "-3", False),
        (204, 8, "auto", True),
    ],
)
def test_threshold_and_ceiling(cgroup, caplog, cpu, memory, ceiling, reported):
    if ceiling is not None:
        os.environ["AITER_MAX_JOBS"] = ceiling
    _, observation = limits._cgroup_memory_bound()
    with patch.object(
        limits, "_automatic_worker_snapshot", return_value=(cpu, memory, observation)
    ):
        selected = limits.get_worker_count()
    normalized = max(1, int(ceiling)) if ceiling and ceiling != "auto" else cpu
    assert selected == max(1, min(cpu, memory, normalized))
    assert bool(caplog.records) == reported


def test_runtime_legacy_ceiling_and_precedence(cgroup, caplog):
    os.environ["MAX_JOBS"] = "4"
    before = dict(os.environ)
    assert limits.get_compile_worker_count() == 4
    assert not caplog.records
    assert dict(os.environ) == before
    # Generic policy ignores legacy; explicit AITER setting also supersedes it.
    assert limits.get_worker_count() == 8
    assert "requested=204" in caplog.text
    os.environ["AITER_MAX_JOBS"] = "6"
    assert limits.get_compile_worker_count() == 6


def test_host_limited_not_attributed_to_cgroup(cgroup, caplog):
    with patch.object(
        limits, "_host_available_memory_bytes", return_value=limits.EST_WORKER_RSS_BYTES
    ):
        assert limits.get_worker_count() == 1
    assert not caplog.records


def test_unreadable_usage_remains_fail_closed(cgroup, caplog):
    (cgroup / "memory.current").unlink()
    assert limits.get_worker_count() == 1
    assert "usage was unreadable" in caplog.text
    assert "current=unavailable" in caplog.text


def test_worsening_cooldown_and_recovery(cgroup, caplog):
    _, observation = limits._cgroup_memory_bound()
    with patch.object(limits.time, "monotonic", return_value=0) as clock, patch.object(
        limits, "_automatic_worker_snapshot", return_value=(204, 8, observation)
    ) as snapshot:
        assert limits.get_worker_count() == 8
        snapshot.return_value = (204, 4, observation)
        clock.return_value = 59
        assert limits.get_worker_count() == 4
        assert len(caplog.records) == 1
        clock.return_value = 60
        assert limits.get_worker_count() == 4
        assert len(caplog.records) == 2
        clock.return_value = 120
        assert limits.get_worker_count() == 4
        assert len(caplog.records) == 2
        snapshot.return_value = (204, 1, observation)
        assert limits.get_worker_count() == 1
        assert len(caplog.records) == 3
        # Recovery clears the baseline, but preserves the cooldown.
        snapshot.return_value = (204, 300, observation)
        assert limits.get_worker_count() == 204
        snapshot.return_value = (204, 8, observation)
        assert limits.get_worker_count() == 8
        assert len(caplog.records) == 3
        clock.return_value = 180
        assert limits.get_worker_count() == 8
        assert len(caplog.records) == 4


def test_reaching_one_from_three_reports(cgroup, caplog):
    _, observation = limits._cgroup_memory_bound()
    with patch.object(limits.time, "monotonic", return_value=0) as clock, patch.object(
        limits, "_automatic_worker_snapshot", return_value=(4, 3, observation)
    ) as snapshot:
        limits.get_worker_count()
        snapshot.return_value = (4, 1, observation)
        clock.return_value = 60
        limits.get_worker_count()
    assert len(caplog.records) == 2


def test_concurrent_calls_keep_observations_local(tmp_path, caplog):
    barrier = Barrier(8)
    for index in range(8):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "memory.max").write_text(str((index + 10) * 1024**3))
        (folder / "memory.current").write_text(str((index + 9) * 1024**3))
    from threading import local

    thread = local()

    def directories():
        return [("v2", str(tmp_path / str(thread.index)))]

    original = limits._cgroup_memory_bound

    def synchronized_bound():
        result = original()
        barrier.wait(timeout=10)
        return result

    def run(index):
        thread.index = index
        return limits.get_worker_count()

    with patch.object(
        limits, "_cgroup_memory_directories", side_effect=directories
    ), patch.object(
        limits, "_cgroup_memory_bound", side_effect=synchronized_bound
    ), patch.object(
        limits, "_host_available_memory_bytes", return_value=1024**4
    ), patch.object(
        limits, "get_cpu_worker_budget", return_value=204
    ), patch.object(
        limits,
        "_read_cgroup_memory_stat",
        side_effect=lambda observation: {
            "anon": int(str(observation["directory"]).rsplit("/", 1)[1]) * 1024**3
        },
    ), ThreadPoolExecutor(
        max_workers=8
    ) as pool:
        assert list(pool.map(run, range(8))) == [1] * 8
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    import re

    limit = float(re.search(r"limit=([0-9.]+)", message)[1])
    anon = float(re.search(r"anon=([0-9.]+)", message)[1])
    assert limit == anon + 10


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_child_does_not_inherit_locked_diagnostic_state():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import aiter_worker_limits as w
w._memory_diagnostic_lock.acquire()
pid = os.fork()
if pid == 0:
    acquired = w._memory_diagnostic_lock.acquire(blocking=False)
    os._exit(0 if acquired else 1)
w._memory_diagnostic_lock.release()
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
