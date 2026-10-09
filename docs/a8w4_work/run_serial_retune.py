#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prepare model shape unions and run one A8W4 retune at a time."""

from __future__ import annotations

import argparse
import csv
import ctypes
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHAPE_FIELDS = (
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
MODELS = (
    ("dsv4_a8w4", "dsv4_fp8fp4", 187),
    ("kimik3_a8w4", "kimik3_a8w4", 35),
    ("gptoss_a8w4", "gptoss_fp8fp4", 16),
)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as source:
        return [
            {str(key).strip(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(source)
        ]


def csv_fields(path: str | Path) -> tuple[str, ...]:
    with Path(path).open(newline="") as source:
        return tuple(field.strip() for field in next(csv.reader(source)))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_key(
    row: dict[str, Any], fields: Sequence[str] = SHAPE_FIELDS
) -> tuple[str, ...]:
    values = []
    for field in fields:
        value = str(row[field]).strip()
        if field in ("token", "model_dim", "inter_dim", "expert", "topk"):
            number = float(value)
            if not number.is_integer() or number <= 0:
                raise ValueError(f"invalid {field}={value!r}")
            value = str(int(number))
        elif field in ("use_g1u1", "doweight_stage1"):
            value = {"true": "1", "false": "0", "1": "1", "0": "0"}[value.lower()]
        values.append(value)
    return tuple(values)


def lookup_token(token: int | str) -> int:
    # Public fused_moe get_padded_M uses these lookup tiers. Input M is kept.
    token = int(token)
    if token < 32768:
        return 1 << (token - 1).bit_length()
    return 131072 if token >= 131072 else 32768


def prepare_model(
    name: str, untuned: str | Path, tuned: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Snapshot source bytes and save each unique execution shape once."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = csv_fields(untuned)
    if not set(SHAPE_FIELDS).issubset(fields):
        raise ValueError(f"{untuned}: incomplete MoE execution header")
    sources, shapes = [], {}
    for kind, source in (("untuned", Path(untuned)), ("tuned", Path(tuned))):
        snapshot = output_dir / f"{name}_source_{kind}.csv"
        source_bytes = source.read_bytes()
        snapshot.write_bytes(source_bytes)
        rows = read_csv(snapshot)
        sources.append(
            {
                "path": str(source.resolve()),
                "snapshot": str(snapshot.resolve()),
                "sha256": file_hash(snapshot),
                "rows": len(rows),
            }
        )
        for line, row in enumerate(rows, 2):
            key = shape_key(row, fields)
            if key not in shapes:
                shapes[key] = {
                    "shape": dict(zip(fields, key)),
                    "lookup_token": lookup_token(row["token"]),
                    "sources": [],
                }
            shapes[key]["sources"].append(
                {
                    "source": kind,
                    "line": line,
                    "tag": row.get("_tag", ""),
                    "previous_us": row.get("us", ""),
                }
            )
    token_index = fields.index("token")
    ordered = sorted(shapes, key=lambda key: (int(key[token_index]), key))
    input_path = output_dir / f"{name}_input.csv"
    with input_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shapes[key]["shape"] for key in ordered)
    record = {
        "model": name,
        "input": str(input_path.resolve()),
        "input_sha256": file_hash(input_path),
        "shape_count": len(shapes),
        "shape_fields": fields,
        "sources": sources,
        "shapes": [shapes[key] for key in ordered],
    }
    write_json(output_dir / f"{name}_input_sources.json", record)
    return record


def prepare_inputs(repo: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    repo, output_dir = Path(repo), Path(output_dir)
    source_dir = repo / "aiter/configs/model_configs"
    records = []
    for name, prefix, expected in MODELS:
        record = prepare_model(
            name,
            source_dir / f"{prefix}_untuned_fmoe.csv",
            source_dir / f"{prefix}_tuned_fmoe.csv",
            output_dir,
        )
        if record["shape_count"] != expected:
            raise ValueError(
                f"{name}: expected {expected} shapes, found {record['shape_count']}"
            )
        records.append(record)
    write_json(output_dir / "retune_inputs.json", {"models": records})
    return records


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def python_entry(argv: list[str]) -> tuple[str, str] | None:
    """Return the executed Python entry, ignoring tool input file arguments."""
    if not argv or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name):
        return None
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in ("-m", "-c"):
            return (argv[index + 1] if index + 1 < len(argv) else "", arg)
        if arg in ("-W", "-X"):
            index += 2
        elif arg == "--":
            return (argv[index + 1] if index + 1 < len(argv) else "", "script")
        elif arg == "-" or not arg.startswith("-"):
            return (arg, "script")
        else:
            index += 1
    return None


def processes() -> list[dict[str, Any]]:
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            argv = (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            rows.append(
                {
                    "pid": int(path.name),
                    "ppid": int(stat[1]),
                    "pgid": int(stat[2]),
                    "starttime": int(stat[19]),
                    "state": stat[0],
                    "argv": argv,
                }
            )
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            continue
    return rows


class LaunchSystem:
    """The scheduler's OS boundary: locks, process ownership and idle samples."""

    @contextmanager
    def locks(self, gpus: list[int]) -> Iterator[None]:
        descriptors = []
        try:
            paths = ["/tmp/aiter-independent-tune.lock"] + [
                f"/tmp/gpu-{gpu}.lock" for gpu in sorted(gpus)
            ]
            for path in paths:
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                descriptors.append(fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(
                        f"tune/GPU lock held: {path}; no task launched"
                    ) from exc
            self.descriptors = tuple(descriptors)
            yield
        finally:
            for fd in descriptors:
                os.close(fd)

    def tune_processes(self) -> list[dict[str, Any]]:
        found = []
        for row in processes():
            if row["pid"] == os.getpid() or row["state"] == "Z":
                continue
            entry = python_entry(row["argv"])
            if entry is None:
                continue
            name, mode = entry
            leaf = name.rsplit(".", 1)[-1] if mode == "-m" else Path(name).stem
            if (
                mode != "-c"
                and (leaf.endswith("_tune") or leaf.startswith("tune_"))
                and "--run_config" not in row["argv"]
            ):
                found.append(dict(row, entry=name))
        return found

    def source_state(self, repo: str | Path) -> dict[str, Any]:
        return source_state(repo)

    def gpu_snapshot(self, gpus: list[int]) -> list[dict[str, Any]]:
        # Direct SMI/PCI probes avoid importing aiter, allocating tensors or
        # changing clocks. Match handles by BDF, never by SMI enumeration index.
        binding_dir = "/opt/rocm/share/amd_smi"
        inserted = binding_dir not in sys.path
        if inserted:
            sys.path.insert(0, binding_dir)
        try:
            smi = importlib.import_module("amdsmi")
        finally:
            if inserted:
                sys.path.remove(binding_dir)
        hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
        smi.amdsmi_init()
        try:
            handles = {
                smi.amdsmi_get_gpu_device_bdf(handle).lower(): handle
                for handle in smi.amdsmi_get_processor_handles()
            }
            samples = []
            for sample in range(3):
                for gpu in gpus:
                    bdf = ctypes.create_string_buffer(64)
                    code = hip.hipDeviceGetPCIBusId(bdf, 64, gpu)
                    if code != 0:
                        raise RuntimeError(
                            f"physical HIP{gpu} BDF probe failed: {code}"
                        )
                    address = bdf.value.decode().lower()
                    handle = handles[address]
                    device = smi.amdsmi_get_gpu_asic_info(handle)
                    activity = smi.amdsmi_get_gpu_activity(handle)
                    jobs = smi.amdsmi_get_gpu_process_list(handle)
                    used = smi.amdsmi_get_gpu_memory_usage(
                        handle, smi.AmdSmiMemoryType.VRAM
                    )
                    total = smi.amdsmi_get_gpu_memory_total(
                        handle, smi.AmdSmiMemoryType.VRAM
                    )
                    idle_contexts, busy_jobs = [], []
                    for job in jobs:
                        engine = job.get("engine_usage") or {}
                        memory = job.get("memory_usage") or {}
                        resources = [
                            job.get("mem"),
                            job.get("cu_occupancy"),
                            engine.get("gfx"),
                            engine.get("enc"),
                            memory.get("gtt_mem"),
                            memory.get("cpu_mem"),
                            memory.get("vram_mem"),
                        ]
                        if all(
                            isinstance(value, (int, float))
                            and math.isfinite(value)
                            and value == 0
                            for value in resources
                        ):
                            idle_contexts.append(job["pid"])
                        else:
                            busy_jobs.append(job)
                    gfx_activity = activity.get("gfx_activity")
                    quiet = (
                        isinstance(gfx_activity, (int, float))
                        and math.isfinite(gfx_activity)
                        and 0 <= gfx_activity <= 2
                    )
                    idle = (
                        quiet
                        and not busy_jobs
                        and isinstance(used, int)
                        and 0 <= used < 1024**3
                        and isinstance(total, int)
                        and total > used
                    )
                    reasons = []
                    if not quiet:
                        reasons.append(
                            f"gfx activity exceeds idle range or is unknown: {gfx_activity}"
                        )
                    if busy_jobs:
                        reasons.append(
                            f"GPU processes with nonzero/unknown resources {[job['pid'] for job in busy_jobs]}"
                        )
                    if not isinstance(used, int) or not 0 <= used < 1024**3:
                        reasons.append(f"VRAM exceeds idle ceiling: {used} bytes")
                    if not isinstance(total, int) or total <= used:
                        reasons.append(f"VRAM capacity is unknown/invalid: {total}")
                    if device["target_graphics_version"] != "gfx950":
                        idle = False
                        reasons.append(
                            f"unsupported gfx {device['target_graphics_version']}"
                        )
                    samples.append(
                        {
                            "utc": utc_now(),
                            "sample": sample,
                            "hip_id": gpu,
                            "bdf": address,
                            "device": device,
                            "activity": activity,
                            "processes": jobs,
                            "idle_context_pids": idle_contexts,
                            "busy_processes": busy_jobs,
                            "vram_used_bytes": used,
                            "vram_total_bytes": total,
                            "idle": idle,
                            "idle_reason": "; ".join(reasons)
                            or "only zero-resource HIP contexts, gfx activity at most 2%, VRAM below 1 GiB",
                        }
                    )
                if sample < 2:
                    time.sleep(1)
            return samples
        finally:
            smi.amdsmi_shut_down()

    def launch(self, command: list[str], **kwargs: Any) -> subprocess.Popen:
        return subprocess.Popen(
            command, pass_fds=self.descriptors, start_new_session=True, **kwargs
        )

    def drain_group(self, process: subprocess.Popen) -> None:
        # No deadline, reset or kill. A live worker below Python must actually
        # finish before another model can start on any device.
        while True:
            remaining = [
                row
                for row in processes()
                if row["pgid"] == process.pid and row["state"] != "Z"
            ]
            if not remaining:
                return
            print(
                f"Waiting for model process group {process.pid}: {[r['pid'] for r in remaining]}",
                flush=True,
            )
            time.sleep(5)


def source_state(repo: str | Path) -> dict[str, Any]:
    repo = Path(repo)

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=repo)

    files = [
        repo / "aiter/fused_moe.py",
        repo / "aiter/jit/core.py",
        repo / "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
        repo / "aiter/ops/moe_mxfp4_aux.py",
        repo / "csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py",
    ]
    files.extend(sorted((repo / "aiter/ops/flydsl").rglob("*.py")))
    flydsl = importlib.metadata.distribution("flydsl")
    compiler = Path(flydsl.locate_file("flydsl/compiler/jit_function.py"))
    return {
        "revision": git("rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
        "sources": {str(path.relative_to(repo)): file_hash(path) for path in files},
        "flydsl_key_source": str(compiler),
        "flydsl_key_sha256": file_hash(compiler),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "flydsl")
        },
        "working_tree": git("status", "--porcelain").decode(),
    }


def check_coverage(
    input_path: str | Path,
    tuned_path: str | Path,
    failure_path: str | Path,
    profile_path: str | Path,
) -> dict[str, Any]:
    fields = csv_fields(input_path)
    expected = {shape_key(row, fields) for row in read_csv(input_path)}
    winners = read_csv(tuned_path) if Path(tuned_path).is_file() else []
    failures = read_csv(failure_path) if Path(failure_path).is_file() else []
    actual = [shape_key(row, fields) for row in winners]
    failed = {shape_key(row, fields) for row in failures}
    profiles = read_csv(profile_path) if Path(profile_path).is_file() else []
    observations = {}
    for row in profiles:
        key = (shape_key(row, fields), row["kernelName1"], row["kernelName2"])
        observations.setdefault(key, []).append(row)
    # This parser module has only a stdlib re import. Load its exact shared
    # code without importing aiter's HIP/JIT package during CPU preparation.
    parser_path = (
        Path(__file__).resolve().parents[2] / "aiter/ops/flydsl/mxfp4_kname.py"
    )
    spec = importlib.util.spec_from_file_location("a8w4_shared_kname", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    invalid = []
    for row in winners:
        try:
            g1 = parser._parse_mxfp4_g1_kname(row["kernelName1"])
            g2 = parser.parse_flydsl_v2_gemm2_kernel(row["kernelName2"])
            valid = g2 is not None and (g1["a_dtype"], g1["out_dtype"]) == (
                "fp8",
                "fp8",
            )
            valid &= (g2["a_dtype"], g2["b_dtype"], g2["out_dtype"]) == (
                "fp8",
                "fp4",
                "bf16",
            )
            valid &= (
                int(row["block_m"])
                == g1["BM"]
                == g2["tile_m"]
                == (g2["sort_block_m"] or g2["tile_m"])
            )
            valid &= math.isfinite(float(row["us"])) and float(row["us"]) > 0
            completed = observations.get(
                (shape_key(row, fields), row["kernelName1"], row["kernelName2"]), []
            )
            valid &= any(
                item["status"] == "ok"
                and item["precision"] == "A8W4"
                and item["search_mode"] == "full"
                and math.isfinite(float(item["error"]))
                and 0 <= float(item["error"]) <= 0.1
                and math.isfinite(float(item["pipeline_us"]))
                and float(item["pipeline_us"]) > 0
                and math.isclose(
                    float(item["pipeline_us"]),
                    float(row["us"]),
                    rel_tol=1e-6,
                    abs_tol=1e-4,
                )
                for item in completed
            )
            if not valid:
                invalid.append(dict(zip(fields, shape_key(row, fields))))
        except (ValueError, KeyError, TypeError):
            invalid.append(row)
    missing = expected - set(actual) - failed
    return {
        "expected": len(expected),
        "winners": len(winners),
        "failed": len(failures),
        "missing": [dict(zip(fields, key)) for key in sorted(missing)],
        "duplicate_winners": len(actual) - len(set(actual)),
        "invalid_winners": invalid,
        "unexpected_winners": len(set(actual) - expected),
        "profile_exists": Path(profile_path).is_file(),
        "failure_file_exists": Path(failure_path).is_file(),
        "status": (
            "passed"
            if (
                set(actual) == expected
                and len(actual) == len(expected)
                and not failures
                and not invalid
                and Path(profile_path).is_file()
                and Path(failure_path).is_file()
            )
            else "failed"
        ),
    }


def run_models(
    records: list[dict[str, Any]],
    repo: str | Path,
    output_dir: str | Path,
    gpus: list[int],
    system: LaunchSystem | None = None,
    batch: int = 4,
) -> dict[str, Any]:
    """Run the fixed model sequence, with one leader and drained workers."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    system = system or LaunchSystem()
    repo, output_dir = Path(repo).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": utc_now(), "status": "running", "models": []}
    with system.locks(gpus):
        others = system.tune_processes()
        if others:
            raise RuntimeError(f"another tune is running; no task launched: {others}")
        report["source"] = system.source_state(repo)
        report["source"] = {
            key: value
            for key, value in report["source"].items()
            if key != "working_tree"
        }
        write_json(output_dir / "retune_run.json", report)
        (output_dir / "retune_source_status.txt").write_text(
            system.source_state(repo).get("working_tree", "")
        )
        ordered = {row["model"]: row for row in records}
        for name, _prefix, _expected in MODELS:
            row = ordered[name]
            current_source = system.source_state(repo)
            frozen_source = {
                key: value
                for key, value in current_source.items()
                if key != "working_tree"
            }
            if frozen_source != report["source"]:
                raise RuntimeError(
                    "measured source changed since scheduler startup; no task launched"
                )
            samples = system.gpu_snapshot(gpus)
            others = system.tune_processes()
            if others or not samples or any(not item["idle"] for item in samples):
                raise RuntimeError(
                    f"devices or another tune became busy; no task launched: {others}, {samples}"
                )
            if file_hash(row["input"]) != row["input_sha256"]:
                raise RuntimeError(f"input snapshot changed: {row['input']}")
            for source in row.get("sources", []):
                if file_hash(source["snapshot"]) != source["sha256"]:
                    raise RuntimeError(f"source snapshot changed: {source['snapshot']}")
            tuned = output_dir / f"{name}_tuned_fmoe.csv"
            profile = output_dir / f"{name}_profile.csv"
            failure = tuned.with_suffix(".failed_shapes.csv")
            if any(path.exists() for path in (tuned, profile, failure)):
                raise FileExistsError(
                    f"retune artifacts already exist for {name}; use a fresh output directory"
                )
            command = [
                sys.executable,
                "-u",
                str(repo / "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"),
                "--mxfp4-flydsl",
                "--mxfp4-search-mode",
                "full",
                "--timeout",
                "300",
                "--errRatio",
                "0.1",
                "--batch",
                str(batch),
                "--mp",
                str(len(gpus)),
                "--all",
                "-i",
                row["input"],
                "-o",
                str(tuned),
                "--profile_file",
                str(profile),
            ]
            environment = os.environ.copy()
            environment.update(
                HIP_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONUNBUFFERED="1"
            )
            environment.setdefault("AITER_USE_SYSTEM_TRITON", "1")
            environment.setdefault("GPU_ARCHS", "gfx950")
            environment["AITER_REBUILD"] = "0"
            environment.setdefault(
                "FLYDSL_EXTRA_SOURCE_DIRS", str(repo / "aiter/ops/flydsl/kernels")
            )
            environment.setdefault(
                "FLYDSL_RUNTIME_CACHE_DIR", str(output_dir / "flydsl_cache")
            )
            record = {
                "model": name,
                "started_utc": utc_now(),
                "input_sha256": row["input_sha256"],
                "source": current_source,
                "status": "starting",
                "command": command,
                "gpus": gpus,
                "mp": len(gpus),
                "batch": batch,
                "idle_samples": samples,
                "warmup": 5,
                "iters": 101,
                "environment": {
                    key: value
                    for key, value in environment.items()
                    if key.startswith(("AITER_", "FLYDSL_", "TUNE_MOE_", "MXFP4_"))
                    or key in ("HIP_VISIBLE_DEVICES", "GPU_ARCHS", "CU_NUM")
                },
            }
            with (output_dir / f"{name}_retune.log").open("w") as log:
                report["models"].append(record)
                write_json(output_dir / f"{name}_run.json", record)
                write_json(output_dir / "retune_run.json", report)
                process = system.launch(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                record["pid"] = record["pgid"] = process.pid
                owned = next(
                    (row for row in processes() if row["pid"] == process.pid), None
                )
                if owned is not None:
                    record["process_starttime"] = owned["starttime"]
                record["status"] = "running"
                write_json(output_dir / f"{name}_run.json", record)
                write_json(output_dir / "retune_run.json", report)
                record["exit_code"] = process.wait()
                system.drain_group(process)
            record["workers_ended"] = True
            record["finished_utc"] = utc_now()
            record["coverage"] = check_coverage(row["input"], tuned, failure, profile)
            record["status"] = (
                "passed"
                if record["exit_code"] == 0 and record["coverage"]["status"] == "passed"
                else "failed"
            )
            write_json(output_dir / f"{name}_run.json", record)
            write_json(output_dir / f"{name}_coverage.json", record["coverage"])
            write_json(output_dir / "retune_run.json", report)
        report["status"] = (
            "passed"
            if all(
                item["exit_code"] == 0 and item["coverage"]["status"] == "passed"
                for item in report["models"]
            )
            else "failed"
        )
        report["finished_utc"] = utc_now()
        write_json(output_dir / "retune_run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="prepare input snapshots without GPU work",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the three prepared models in fixed order",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        help="physical HIP GPU IDs, rechecked before each launch",
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="shapes per tuning batch (default: 4)"
    )
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("--batch must be positive")
    output_dir = args.output_dir or args.repo / "docs/a8w4_work"
    if not args.prepare and not args.run:
        parser.error("specify --prepare and/or --run")
    if args.prepare:
        records = prepare_inputs(args.repo, output_dir)
        print(
            json.dumps(
                {
                    "prepared": [
                        {"model": r["model"], "shapes": r["shape_count"]}
                        for r in records
                    ]
                }
            )
        )
    else:
        records = json.loads((output_dir / "retune_inputs.json").read_text())["models"]
    if args.run:
        if not args.gpus or len(args.gpus) != len(set(args.gpus)) or min(args.gpus) < 0:
            parser.error("--run requires unique nonnegative --gpus physical IDs")
        if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
            "ROCR_VISIBLE_DEVICES"
        ):
            parser.error(
                "unset inherited HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES before selecting physical GPUs"
            )
        report = run_models(records, args.repo, output_dir, args.gpus, batch=args.batch)
        raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
