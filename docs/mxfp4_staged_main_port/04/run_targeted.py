# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Retained root-coordinator execution of the ten targeted winner replays."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/root/workspace/tmp_dev/aiter-mxfp4-04")
BASE = ROOT / "docs/mxfp4_staged_main_port/04"
RUNTIME = Path("/root/workspace/tmp_dev/mxfp4-port-runtime")
PYTHON = "/root/workspace/tmp_dev/mxfp4-port-env/bin/python"
for job in json.loads((BASE / "targeted-replay-plan.json").read_text()):
    directory = BASE / "gpu/targeted" / job["id"]
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(Path(PYTHON).parent) + os.pathsep + env.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "GPU_ARCHS": "gfx950",
            "FLYDSL_GPU_ARCH": "gfx950",
            "AITER_USE_SYSTEM_TRITON": "1",
            "MAX_JOBS": "8",
            "CK_DIR": "/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel",
            "AITER_JIT_DIR": str(RUNTIME / "jit-default"),
            "AITER_MXFP4_TUNE_SWIGLU_LIMIT": (
                str(job["limit"]) if job["limit"] is not None else ""
            ),
            "AITER_FLYDSL_STAGE2_FP8": "1" if job["fp8"] else "0",
            "AITER_MXFP4_INTERMEDIATE": "1" if job["fp4"] else "0",
            "TUNE_MOE_EXPERT_BALANCE": "False",
            "TUNE_MOE_ROUTING_SEED": "0",
            "FLYDSL_RUNTIME_CACHE_DIR": str(RUNTIME / ("targeted-04-" + job["id"])),
        }
    )
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    command = [
        PYTHON,
        str(ROOT / "docs/mxfp4_staged_main_port/scripts/run_gpu_task.py"),
        "--artifact",
        str(directory / "task"),
        "--cwd",
        str(ROOT),
        "--",
        "docs/mxfp4_staged_main_port/04/alternating_replay.py",
        "--left",
        str(BASE / "gpu" / job["left"] / "winner.csv"),
        "--right",
        str(BASE / "gpu" / job["right"] / "winner.csv"),
        "--result",
        str(directory / "result.json"),
        "--reason",
        job["reason"],
        "--repeats",
        str(job["repeats_per_pair"]),
    ]
    print("TARGET", job["id"], flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
print("All ten targeted alternating replay tasks complete", flush=True)
