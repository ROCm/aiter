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
