# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys


def run_in_subprocess(child_code: str, timeout: int = 180):
    """Run a test body that can fault the GPU in a fresh Python subprocess.

    A GPU memory fault aborts the whole process, so no try/except in pytest
    can catch it. HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 prevents multi-GB GPU
    coredump files when the host has the coredump tool installed (verified
    env var name on ROCm 7.2.0 via
    `strings libhsa-runtime64.so | grep coredump`)."""
    env = dict(os.environ)
    env["HSA_DISABLE_COREDUMP_ON_EXCEPTION"] = "1"
    return subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def is_gpu_memory_fault(rc: int, stderr: str) -> bool:
    """Detect HSA GPU memory-fault death.

    Empirically on ROCm 7.2.0 / MI300X the signal is SIGABRT (-6); on
    ROCm 7.x with different coredump-tool config it can be SIGPIPE (-13).
    Match on (process killed by ANY signal) AND (HSA fault keyword in
    stderr) -- both conditions must hold to avoid false positives from
    unrelated SIGPIPE / SIGABRT.
    """
    killed_by_signal = (rc < 0) or (128 <= rc < 256)
    if not killed_by_signal:
        return False
    msg = stderr.lower()
    return "memory access fault" in msg or "hsa_status_error_memory_fault" in msg


def assert_no_gpu_memory_fault(result, fault_message: str):
    """Assert that a child from `run_in_subprocess` exited cleanly.

    `fault_message` describes the bug that a GPU memory fault indicates."""
    stderr_tail = result.stderr[-2048:]
    assert not is_gpu_memory_fault(
        result.returncode, result.stderr
    ), f"{fault_message} (rc={result.returncode}):\n{stderr_tail}"
    assert (
        result.returncode == 0
    ), f"child failed (rc={result.returncode}):\n{stderr_tail}"
