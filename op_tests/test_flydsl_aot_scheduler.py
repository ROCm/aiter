# SPDX-License-Identifier: MIT

from __future__ import annotations

import multiprocessing
import os
import time

import pytest

pytest.importorskip("flydsl")

from aiter.aot.flydsl.common import (
    OpKind,
    _make_sequential_aot_run,
    wait_aot,
)


def _record_worker_arch(kind, job):
    with job["lock"]:
        job["active"].value += 1
        job["peak"].value = max(job["peak"].value, job["active"].value)
        job["events"].append(
            ("start", job["job_id"], job.get("arch"), os.getpid(), time.monotonic())
        )
    time.sleep(0.05)
    with job["lock"]:
        job["events"].append(
            ("end", job["job_id"], job.get("arch"), os.getpid(), time.monotonic())
        )
        job["active"].value -= 1
    return kind, {
        "kernel_name": job["kernel_name"],
        "compile_time": 0.01,
        "pid": os.getpid(),
        "observed_arch": job.get("arch"),
        "job_id": job["job_id"],
    }


def _record_or_fail(kind, job):
    job["events"].append((job["job_id"], job.get("arch"), os.getpid()))
    if job.get("fail"):
        raise RuntimeError("intentional worker failure")
    time.sleep(0.1)
    return kind, {"kernel_name": job["kernel_name"], "compile_time": 0.01}


@pytest.mark.parametrize("configured_cap", (1, 2, 8))
def test_package_aot_scheduler_enforces_global_sequential_budget(
    monkeypatch, configured_cap
):
    monkeypatch.setenv("AITER_FLYDSL_AOT_WORKERS", str(configured_cap))
    with multiprocessing.Manager() as manager:
        events = manager.list()
        active = manager.Value("i", 0)
        peak = manager.Value("i", 0)
        lock = manager.Lock()
        jobs = []
        specs = (("gfx942", 3), ("gfx950", 2), (None, 1))
        for architecture, count in specs:
            for index in range(count):
                job_id = f"{architecture}-{index}"
                jobs.append(
                    (
                        OpKind.GEMM,
                        {
                            "kernel_name": job_id,
                            "job_id": job_id,
                            "arch": architecture,
                            "events": events,
                            "active": active,
                            "peak": peak,
                            "lock": lock,
                        },
                    )
                )

        run = _make_sequential_aot_run(jobs, _record_worker_arch)
        assert run is not None
        wait_aot(run, {})
        observations = [result for _, result, _ in run.results]

        expected_cap = min(configured_cap, len(jobs))
        assert run.global_worker_cap == expected_cap
        assert run.peak_live_workers <= expected_cap
        assert peak.value <= expected_cap
        assert run.partition_order == [
            "gfx942",
            "gfx950",
            "architecture-neutral",
        ]
        assert run.partition_job_counts == {
            "gfx942": 3,
            "gfx950": 2,
            "architecture-neutral": 1,
        }
        assert len(observations) == len(jobs)
        assert {result["job_id"] for result in observations} == {
            job["job_id"] for _, job in jobs
        }

        arches_by_pid = {}
        for result in observations:
            arch = result["observed_arch"] or "architecture-neutral"
            arches_by_pid.setdefault(result["pid"], set()).add(arch)
        assert all(len(arches) == 1 for arches in arches_by_pid.values())

        intervals = {}
        for event, job_id, architecture, pid, timestamp in list(events):
            intervals.setdefault(architecture, []).append((event, timestamp))
        ordered_arches = ("gfx942", "gfx950", None)
        for previous, following in zip(ordered_arches, ordered_arches[1:]):
            previous_end = max(
                timestamp
                for event, timestamp in intervals[previous]
                if event == "end"
            )
            following_start = min(
                timestamp
                for event, timestamp in intervals[following]
                if event == "start"
            )
            assert previous_end <= following_start


def test_package_aot_scheduler_failure_stops_later_partitions(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_AOT_WORKERS", "1")
    with multiprocessing.Manager() as manager:
        events = manager.list()
        before = {child.pid for child in multiprocessing.active_children()}
        jobs = [
            (
                OpKind.GEMM,
                {
                    "kernel_name": "fail-first",
                    "job_id": "fail-first",
                    "arch": "gfx942",
                    "events": events,
                    "fail": True,
                },
            ),
            (
                OpKind.GEMM,
                {
                    "kernel_name": "must-not-run",
                    "job_id": "must-not-run",
                    "arch": "gfx950",
                    "events": events,
                },
            ),
        ]
        run = _make_sequential_aot_run(jobs, _record_or_fail)
        assert run is not None
        with pytest.raises(RuntimeError, match="intentional worker failure"):
            wait_aot(run, {})
        assert all(architecture != "gfx950" for _, architecture, _ in events)
        assert run.partition_order == ["gfx942"]
        after = {child.pid for child in multiprocessing.active_children()}
        assert after == before
