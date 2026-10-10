#!/usr/bin/env python3
"""Root coordinator: run prescribed ticket03 pair/deployment tasks sequentially."""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument(
    "--runtime", type=Path, default=Path("/root/workspace/tmp_dev/mxfp4-port-runtime")
)
parser.add_argument("--python", default=sys.executable)
args = parser.parse_args()
ROOT = args.root.resolve()
CONTROLLER = ROOT / "docs/mxfp4_staged_main_port/scripts/run_gpu_task.py"
PYTHON = args.python
RUNTIME = args.runtime.resolve()
GPU = ROOT / "docs/mxfp4_staged_main_port/03/gpu"
JOBS = json.loads((ROOT / "docs/mxfp4_staged_main_port/03/fixtures.json").read_text())
MANIFEST = GPU / "controller-outcomes.json"
outcomes = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []

for fixture in JOBS:
    config = f"fp8{int(fixture['fp8'])}-fp4{int(fixture['fp4'])}-limit{fixture['limit'] or 'default'}"
    for action in ("pair", "replay", "aot", "precompiled"):
        artifact = GPU / fixture["id"] / (action + "-task")
        result_path = GPU / fixture["id"] / (action + ".json")
        if (artifact / "task.json").exists():
            prior = json.loads((artifact / "task.json").read_text())
            if prior.get("exit_code") == 0 and result_path.exists():
                print(f"EXISTING PASS {fixture['id']} {action}", flush=True)
                continue
            raise RuntimeError(
                f"Existing incomplete/failed task needs diagnosis: {artifact}"
            )
        env = os.environ.copy()
        env.update(
            {
                "PATH": str(Path(PYTHON).parent) + os.pathsep + env.get("PATH", ""),
                "PYTHONPATH": str(ROOT),
                "GPU_ARCHS": "gfx950",
                "FLYDSL_GPU_ARCH": "gfx950",
                "AITER_USE_SYSTEM_TRITON": "1",
                "MAX_JOBS": "8",
                "CK_DIR": os.environ.get(
                    "CK_DIR", str(ROOT / "3rdparty/composable_kernel")
                ),
                "AITER_JIT_DIR": str(RUNTIME / "jit-default"),
                "AITER_FLYDSL_STAGE2_FP8": "1" if fixture["fp8"] else "0",
                "AITER_MXFP4_INTERMEDIATE": "1" if fixture["fp4"] else "0",
                "AITER_MXFP4_TUNE_SWIGLU_LIMIT": (
                    str(fixture["limit"]) if fixture["limit"] is not None else ""
                ),
                "FLYDSL_RUNTIME_CACHE_DIR": str(
                    RUNTIME
                    / (
                        ("aot-03-" if action in ("aot", "precompiled") else "pair-03-")
                        + config
                    )
                ),
            }
        )
        env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
        if action == "precompiled":
            env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
        worker = [
            "docs/mxfp4_staged_main_port/scripts/configured_replay.py",
            "--csv",
            fixture["csv"],
            "--result",
            str(result_path),
            "--action",
            "replay" if action == "precompiled" else action,
        ]
        if fixture["limit"] is not None:
            worker += ["--swiglu-limit", str(fixture["limit"])]
        if fixture["clamp_fixture"]:
            worker += ["--clamp-fixture"]
        print(f"RUN {fixture['id']} {action} {config}", flush=True)
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        completed = subprocess.run(
            [
                PYTHON,
                str(CONTROLLER),
                "--artifact",
                str(artifact),
                "--cwd",
                str(ROOT),
                "--",
                *worker,
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        record = {
            "fixture": fixture["id"],
            "action": action,
            "config": config,
            "start_utc": start,
            "exit_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "exit_code": completed.returncode,
            "artifact": str(artifact),
        }
        outcomes.append(record)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(outcomes, indent=2) + "\n")
        if completed.returncode:
            raise SystemExit(
                f"Task returned failure; diagnose before continuing: {record}"
            )
print("All prescribed ticket03 task records completed", flush=True)
