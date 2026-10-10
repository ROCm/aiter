# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Offline behavioral checks for the acceptance task controller."""

import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path


def controller():
    path = Path(__file__).with_name("run_gpu_task.py")
    spec = importlib.util.spec_from_file_location("run_gpu_task", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(*, use=0, memory=297766912, processes=()):
    return {
        "status": {
            "Unique ID": "0x5d8a72a332aa124c",
            "PCI Bus": "0000:F5:00.0",
            "GPU use (%)": str(use),
            "VRAM Total Used Memory (B)": str(memory),
        },
        "processes": list(processes),
    }


def test_idle_gate_distinguishes_discovery_from_real_gpu_work():
    task = controller()
    assert task.is_idle(sample())
    assert task.is_idle(
        sample(processes=[{"mem_usage": {"value": 0}, "cu_occupancy": 0}])
    )
    assert not task.is_idle(sample(use=1))
    assert not task.is_idle(sample(memory=1024**3))
    assert not task.is_idle(sample(processes=[{"mem_usage": {"value": 1}}]))
    assert not task.is_idle(sample(processes=[{"usage": {"gfx": {"value": 1}}}]))
    assert not task.is_idle(sample(processes=[{"cu_occupancy": 1}]))


def test_physical_identity_requires_uuid_and_pci():
    task = controller()
    assert task.identity_matches(sample())
    wrong = sample()
    wrong["status"]["Unique ID"] = "0x1234"
    assert not task.identity_matches(wrong)
    wrong = sample()
    wrong["status"]["PCI Bus"] = "0000:65:00.0"
    assert not task.identity_matches(wrong)


def test_cancellation_stops_the_owned_process_group_before_waiting(monkeypatch):
    task = controller()
    observed = []

    class Process:
        pid = 123

        def wait(self):
            observed.append("wait")

    monkeypatch.setattr(task.os, "killpg", lambda pid, sig: observed.append((pid, sig)))
    task.stop_owned_processes(Process())
    assert observed == [(123, task.signal.SIGTERM), "wait", (123, task.signal.SIGKILL)]


def test_unavailable_occupancy_uses_remaining_idle_evidence():
    task = controller()
    discovery = {
        "mem_usage": {"value": 0},
        "usage": {"gfx": {"value": 0}},
        "cu_occupancy": "N/A",
    }
    assert task.is_idle(sample(processes=[discovery]))
    assert not task.is_idle(sample(use=1, processes=[discovery]))
    assert not task.is_idle(sample(processes=[dict(discovery, mem_usage={"value": 1})]))


def test_cancellation_cleans_a_cpu_descendant(tmp_path):
    task = controller()
    pid_file = tmp_path / "descendant.pid"
    child_code = (
        "import os, signal; "
        "os.write(1, str(os.getpid()).encode()+b'\\n'); "
        "signal.pause()"
    )
    parent_code = (
        "import subprocess, sys, signal; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1]]); "
        "signal.pause()"
    )
    process = None
    descendant = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code, child_code],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        descendant = int(process.stdout.readline())
        pid_file.write_text(str(descendant))
        task.stop_owned_processes(process)
        assert process.returncode == -signal.SIGTERM
        # A killed orphan may briefly remain a zombie; it cannot own GPU work.
        status_path = Path(f"/proc/{descendant}/status")
        if status_path.exists():
            assert "State:\tZ" in status_path.read_text()
    finally:
        if process is not None and process.poll() is None:
            task.stop_owned_processes(process)
        if process is not None:
            process.stdout.close()
        if descendant is not None:
            try:
                os.kill(descendant, signal.SIGKILL)
            except ProcessLookupError:
                pass
