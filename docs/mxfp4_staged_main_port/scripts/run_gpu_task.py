#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run one acceptance task on the prescribed physical GPU after an idle gate.

The coordinator invokes tasks sequentially and waits for this process to exit.
This is an acceptance script, not a tuner-wide GPU lock or hard deadline.
"""

import argparse
import ctypes
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MONITOR_GPU = 7
UUID = "GPU-5d8a72a332aa124c"
PCI = "0000:f5:00.0"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def command_json(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    positions = [
        p for p in (result.stdout.find("{"), result.stdout.find("[")) if p >= 0
    ]
    return json.loads(result.stdout[min(positions) :])


def snapshot():
    status = command_json(
        [
            "rocm-smi",
            "-d",
            str(MONITOR_GPU),
            "--showuse",
            "--showmeminfo",
            "vram",
            "--showuniqueid",
            "--showbus",
            "--json",
        ]
    )[f"card{MONITOR_GPU}"]
    devices = command_json(["amd-smi", "process", "-g", str(MONITOR_GPU), "--json"])
    processes = [
        entry["process_info"]
        for device in devices
        for entry in device.get("process_list", [])
        if isinstance(entry.get("process_info"), dict)
    ]
    return {"utc": utc_now(), "status": status, "processes": processes}


def identity_matches(sample):
    status = sample["status"]
    hardware_id = status["Unique ID"].lower().removeprefix("0x")
    return "GPU-" + hardware_id == UUID and status["PCI Bus"].lower() == PCI


def is_idle(sample):
    active = any(
        entry.get("mem_usage", {}).get("value", 0) > 0
        or entry.get("usage", {}).get("gfx", {}).get("value", 0) > 0
        or (
            entry.get("cu_occupancy", 0) not in (None, "N/A")
            and float(entry.get("cu_occupancy", 0)) > 0
        )
        for entry in sample["processes"]
    )
    status = sample["status"]
    return (
        not active
        and int(status["GPU use (%)"]) == 0
        and int(status["VRAM Total Used Memory (B)"]) < 512 * 1024**2
    )


def await_idle(artifact):
    consecutive = 0
    while consecutive < 3:
        sample = snapshot()
        with (artifact / "idle.jsonl").open("a") as stream:
            stream.write(json.dumps(sample) + "\n")
        if not identity_matches(sample):
            raise RuntimeError(f"GPU{MONITOR_GPU} identity changed: {sample['status']}")
        consecutive = consecutive + 1 if is_idle(sample) else 0
        if not consecutive:
            print(f"GPU{MONITOR_GPU} busy; waiting on prescribed card", flush=True)
        if consecutive < 3:
            time.sleep(2)
    return sample


def verify_child(artifact):
    import torch

    if os.environ.get("ROCR_VISIBLE_DEVICES") != UUID:
        raise RuntimeError("Child did not inherit the prescribed physical UUID")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Child did not inherit logical device mapping")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Task must have exactly one visible GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    arch = properties.gcnArchName.split(":", 1)[0]
    if arch != "gfx950" or properties.multi_processor_count != 256:
        raise RuntimeError(f"Unexpected runtime target: {properties}")
    hip = ctypes.CDLL("libamdhip64.so")
    buffer = ctypes.create_string_buffer(64)
    if hip.hipDeviceGetPCIBusId(buffer, 64, 0) != 0:
        raise RuntimeError("HIP PCI identity query failed")
    pci = buffer.value.decode().lower()
    if pci != PCI:
        raise RuntimeError(f"Task mapped to {pci}, expected {PCI}")
    write_json(
        artifact / "runtime-device.json",
        {
            "utc": utc_now(),
            "pid": os.getpid(),
            "logical_gpu": 0,
            "physical_uuid_selector": UUID,
            "monitor_gpu": MONITOR_GPU,
            "pci": pci,
            "gfx": arch,
            "cu_num": properties.multi_processor_count,
            "name": properties.name,
            "total_memory": properties.total_memory,
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
    )


def stop_owned_processes(process):
    # The worker owns a new session; cancellation also stops its spawned tuner
    # workers. The final kill removes any descendant that ignored termination.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--set-env", action="append", default=[])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Python script or -m command is required after --")
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    cwd = args.cwd.resolve()
    if args.worker:
        verify_child(artifact)
        os.chdir(cwd)
        os.execvpe(sys.executable, [sys.executable, *command], os.environ)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({"ROCR_VISIBLE_DEVICES": UUID, "HIP_VISIBLE_DEVICES": "0"})
    for setting in args.set_env:
        name, separator, value = setting.partition("=")
        if not separator:
            parser.error("--set-env must be NAME=VALUE")
        env[name] = value
    env["ROCR_VISIBLE_DEVICES"] = UUID
    env["HIP_VISIBLE_DEVICES"] = "0"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    recorded_names = [
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "GPU_ARCHS",
        "CU_NUM",
        "FLYDSL_GPU_ARCH",
        "AITER_USE_SYSTEM_TRITON",
        "AITER_JIT_DIR",
        "CK_DIR",
        "FLYDSL_RUNTIME_CACHE_DIR",
        "FLYDSL_RUNTIME_RUN_ONLY",
        "AITER_CONFIG_FMOE",
        "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
        "AITER_FLYDSL_STAGE2_FP8",
        "AITER_MXFP4_INTERMEDIATE",
        "TUNE_MOE_EXPERT_BALANCE",
        "TUNE_MOE_ROUTING_SEED",
        "AITER_LOG_MORE",
        "AITER_MOE_EXPERT_BALANCE",
        "MAX_JOBS",
    ]
    record = {
        "coordinator_start_utc": utc_now(),
        "cwd": str(cwd),
        "python": sys.executable,
        "command": [sys.executable, *command],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
        ).strip(),
        "environment": {name: env.get(name) for name in recorded_names},
        "timeout_contract": "No task or shape hard deadline; wait for child exit",
    }
    write_json(artifact / "task.json", record)
    record["idle_before_start"] = await_idle(artifact)
    child_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--artifact",
        str(artifact),
        "--cwd",
        str(cwd),
        "--",
        *command,
    ]
    process = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"Task controller received signal {signum}")

    signal.signal(signal.SIGTERM, cancelled)
    try:
        with (artifact / "console.log.txt").open("w") as console:
            process = subprocess.Popen(
                child_command,
                env=env,
                cwd=cwd,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(pid=process.pid, process_start_utc=utc_now())
            write_json(artifact / "task.json", record)
            print(f"Started GPU7 task pid={process.pid}: {command}", flush=True)
            record["exit_code"] = process.wait()
    except BaseException:
        if process is not None:
            stop_owned_processes(process)
        record["coordinator_interrupted"] = True
        record["exit_code"] = process.returncode if process is not None else None
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        record["process_exit_utc"] = utc_now()
        write_json(artifact / "task.json", record)
    print(f"GPU7 task exited {record['exit_code']}: {artifact}", flush=True)
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
