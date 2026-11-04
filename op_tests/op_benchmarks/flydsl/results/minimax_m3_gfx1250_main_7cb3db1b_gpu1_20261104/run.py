# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run alternating, checked MiniMax-M3 matrices on physical GPU 1."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def fingerprint(source):
    paths = [source / "aiter/ops/flydsl/pa_decode.py"]
    paths += sorted((source / "aiter/ops/flydsl/kernels").glob("pa_decode*.py"))
    paths += sorted((source / "aiter/ops/flydsl/kernels/pa_decode").glob("*.py"))
    return {
        "commit": git(source, "rev-parse", "HEAD"),
        "dirty": bool(git(source, "status", "--porcelain")),
        "files": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--flydsl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        HIP_VISIBLE_DEVICES="1", FLYDSL_RUNTIME_CACHE_DIR=str(root / "flydsl-cache")
    )
    sources = {"previous": args.previous.resolve(), "main": args.main.resolve()}
    before = {k: fingerprint(p) for k, p in sources.items()}
    assert all(not item["dirty"] for item in before.values())
    flydsl_before = {
        "commit": git(args.flydsl, "rev-parse", "HEAD"),
        "dirty": bool(git(args.flydsl, "status", "--porcelain")),
    }
    harness_sha = hashlib.sha256(args.harness.read_bytes()).hexdigest()
    commands = []
    stop = threading.Event()

    def snapshot(path):
        with path.open("w") as stream:
            subprocess.run(
                [
                    "rocm-smi",
                    "-d",
                    "1",
                    "--showuse",
                    "--showmeminfo",
                    "vram",
                    "--showclocks",
                    "--showtemp",
                    "--showpower",
                    "--showuniqueid",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def monitor():
        with (root / "gpu1-monitor.jsonl").open("w") as stream:
            while not stop.is_set():
                result = subprocess.run(
                    [
                        "rocm-smi",
                        "-d",
                        "1",
                        "--showuse",
                        "--showmeminfo",
                        "vram",
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                stream.write(
                    json.dumps(
                        {
                            "timestamp_utc": utc(),
                            "exit_code": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                stop.wait(5)

    snapshot(root / "gpu-start.txt")
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for label, repeat in [
            ("previous", 1),
            ("main", 1),
            ("main", 2),
            ("previous", 2),
            ("previous", 3),
            ("main", 3),
        ]:
            name = f"{label}-{repeat}"
            assert hashlib.sha256(args.harness.read_bytes()).hexdigest() == harness_sha
            snapshot(root / f"{name}-gpu-start.txt")
            argv = [
                sys.executable,
                "-u",
                str(args.harness.resolve()),
                "--source",
                str(sources[label]),
                "--profile",
                "both",
                "--suite",
                "dense-decode",
                "dense-verify",
                "sparse-decode",
                "sparse-verify",
                "sparse-prefill",
                "cluster-boundary",
                "--rounds",
                "5",
                "--label",
                name,
                "--output",
                str(root / f"{name}.json"),
            ]
            if label == "main":
                argv += ["--compare-to", str(root / "previous-1.json")]
            record = {
                "label": name,
                "argv": argv,
                "environment": {
                    k: env.get(k)
                    for k in (
                        "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                        "CUDA_VISIBLE_DEVICES",
                        "AITER_RUNTIME_GPU_ARCH",
                        "FLYDSL_GPU_ARCH",
                        "AITER_USE_SYSTEM_TRITON",
                        "AITER_FLYDSL_KERNARG_PRELOAD",
                        "FLYDSL_AUTOTUNE",
                        "FLYDSL_RUNTIME_CACHE_DIR",
                    )
                },
                "started_utc": utc(),
            }
            print(f"Starting {name}: {record['started_utc']}", flush=True)
            with (root / f"{name}.log").open("w") as stream:
                result = subprocess.run(
                    argv,
                    env=env,
                    cwd=sources[label],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record.update(exit_code=result.returncode, ended_utc=utc())
            commands.append(record)
            (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            snapshot(root / f"{name}-gpu-end.txt")
            print(
                f"Finished {name}: exit={result.returncode}, {record['ended_utc']}",
                flush=True,
            )
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")
    finally:
        stop.set()
        thread.join(timeout=10)
        snapshot(root / "gpu-end.txt")
        after = {k: fingerprint(p) for k, p in sources.items()}
        flydsl_after = {
            "commit": git(args.flydsl, "rev-parse", "HEAD"),
            "dirty": bool(git(args.flydsl, "status", "--porcelain")),
        }
        info = {
            "physical_gpu": 1,
            "started_sources": before,
            "ended_sources": after,
            "flydsl_before": flydsl_before,
            "flydsl_after": flydsl_after,
            "harness_sha256": harness_sha,
            "commands": "commands.json",
        }
        (root / "run-info.json").write_text(json.dumps(info, indent=2) + "\n")
        assert before == after, "source changed during measurement"
        assert flydsl_before == flydsl_after, "FlyDSL changed during measurement"


if __name__ == "__main__":
    main()
