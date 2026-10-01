#!/usr/bin/env python3
"""Replay on a CI runner or an equivalent manually allocated runner pod."""

import argparse
import json
import math
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text())
MODES = ("audit", "baseline", "affinity", "prebuild", "ipc-copy")


def call(command, *, timeout=120, check=True, capture=False):
    return subprocess.run(
        command, text=True, check=check, timeout=timeout,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def collect(command, *, capture=False):
    try:
        return call(command, check=False, capture=capture)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Diagnostic/cleanup command failed: {command[0]}: {exc}", flush=True)
        return subprocess.CompletedProcess(command, 1, stdout="")


def device_flags(path, expected):
    flags = shlex.split(Path(path).read_text())
    devices = []
    cpus = None
    for index in range(0, len(flags), 2):
        option, value = flags[index:index + 2]
        if option == "--device":
            if not value.startswith("/dev/dri/renderD") or not value[16:].isdigit():
                raise ValueError(f"Unexpected device mapping: {value}")
            devices.append(value)
        elif option == "--cpuset-cpus":
            cpus = value
        elif option != "--group-add":
            raise ValueError(f"Unexpected runner option: {option}")
    if len(set(devices)) != expected or not cpus:
        raise ValueError(f"Expected {expected} allocated GPUs and an explicit CPU set: {flags}")
    return flags


def accuracy_result(output, threshold):
    files = list((output / "accuracy_test_results").glob("*.json"))
    if not files:
        raise RuntimeError("Accuracy evaluation produced no result")
    result = json.loads(max(files, key=lambda p: p.stat().st_mtime_ns).read_text())
    value = result["results"]["gsm8k"]["exact_match,flexible-extract"]
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < threshold:
        raise RuntimeError(f"Accuracy {value!r} is below {threshold}")
    samples = result["n-samples"]["gsm8k"]
    if samples["effective"] != 1319:
        raise RuntimeError(f"Expected all 1319 GSM8K samples, got {samples}")
    return {"accuracy": value, "threshold": threshold, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=PINS["models"], required=True)
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--output", default="atom_diagnostics")
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=False)
    relative_output = output.relative_to(workspace)
    remote_output = Path("/workspace") / relative_output
    remote_tools = Path("/workspace") / HERE.relative_to(workspace)
    sha = call(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    assert sha == PINS["atom"], f"ATOM is {sha}, expected {PINS['atom']}"
    models = json.loads(Path(".github/benchmark/models_accuracy.json").read_text())
    model = next(item for item in models if item["model_name"] == args.model)
    model_pin = PINS["models"][args.model]
    flags = device_flags("/etc/podinfo/gha-render-devices", model_pin["gpus"])
    model_root = Path("/models") / model["model_path"]
    if (model_root / ".hf-revision").read_text().strip() != model_pin["revision"]:
        raise RuntimeError("Shared model revision differs from the failed run; no download attempted")
    name = f"atom-pr5989-{uuid.uuid4().hex[:12]}"
    image = f"aiter-pr5989-triage:{PINS['aiter'][:12]}"
    summary = {"model": args.model, "mode": args.mode, "pins": PINS, "start": time.time(),
               "runner_flags": flags, "container": name, "result": "incomplete"}
    (output / "container-name.txt").write_text(name + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    monitor = None

    def docker_exec(command, **kwargs):
        return call(["docker", "exec", name, *map(str, command)], **kwargs)

    def logged_exec(label, command, timeout):
        start = time.time()
        with (output / f"{label}.log").open("w") as stream:
            result = subprocess.run(
                ["docker", "exec", name, *map(str, command)],
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
            )
        summary[label] = {"seconds": time.time() - start, "exit_code": result.returncode}
        print(f"{label}: exit={result.returncode}, seconds={time.time() - start:.1f}", flush=True)
        result.check_returncode()

    try:
        with (output / "image-build.log").open("w") as stream:
            subprocess.run(
                ["docker", "build", "--network=host", "--build-arg", f"BASE_IMAGE={PINS['image']}",
                 "--build-arg", f"AITER_SHA={PINS['aiter']}", "-t", image, "-f",
                 str(HERE / "Dockerfile"), str(HERE)],
                stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800,
            )
        environment = {"ATOM_DISABLE_MMAP": "true", "AITER_JIT_VERBOSE": "1"}
        for line in model.get("env_vars", "").splitlines():
            if line.strip():
                key, value = line.split("=", 1)
                environment[key] = value
        if args.mode == "affinity":
            environment["MAX_JOBS"] = "20"
        env_flags = [part for key, value in environment.items() for part in ("-e", f"{key}={value}")]
        if os.environ.get("HF_TOKEN"):
            env_flags.extend(["-e", "HF_TOKEN"])
        call([
            "docker", "run", "-dt", "--name", name, "--device=/dev/kfd", *flags,
            "--ipc=host", "--group-add", "video", "--shm-size=16G", "--privileged",
            "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined",
            "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", *env_flags,
            "-v", "/models:/models:ro", "-v", f"{workspace}:/workspace", "-w", "/workspace", image,
        ])
        # No model execution until the nested container sees only its assigned GPUs.
        logged_exec("preflight", ["python3", remote_tools / "inspect_runtime.py", "--model",
                    args.model, "--expected-devices", model_pin["gpus"], "--output",
                    remote_output / "runtime.json"], 120)
        if args.mode == "audit":
            summary["result"] = "preflight passed; model not run"
            return
        if args.mode == "ipc-copy":
            logged_exec("ipc-copy", ["git", "-C", "/app/aiter-test", "apply",
                        remote_tools / "copy-capture.patch"], 30)
        monitor_stream = (output / "monitor.log").open("w")
        monitor = subprocess.Popen(
            ["docker", "exec", name, "python3", str(remote_tools.parent / "monitor.py"), str(remote_output)],
            stdout=monitor_stream, stderr=subprocess.STDOUT,
        )
        monitor_stream.close()
        if args.mode == "prebuild":
            logged_exec("prebuild", ["python3", remote_tools.parent / "prebuild.py",
                        "/app/aiter-test", "module_gemm_a8w8_bpreshuffle_cktile"], 1800)
        deadline = time.monotonic() + 5400
        launch = [".github/scripts/atom_test.sh", "launch", str(model_root), *shlex.split(model["extraArgs"])]
        logged_exec("launch", ["bash", "-lc", shlex.join(launch)], 5400)
        accuracy = [".github/scripts/atom_test.sh", "accuracy", str(model_root)]
        logged_exec("accuracy", ["bash", "-lc", shlex.join(accuracy)], max(1, deadline - time.monotonic()))
        docker_exec(["cp", "-r", "/workspace/accuracy_test_results", remote_output])
        summary.update(accuracy_result(output, float(model["accuracy_threshold"])))
        if (model_root / ".hf-revision").read_text().strip() != model_pin["revision"]:
            raise RuntimeError("Shared model revision changed during the experiment")
        summary["result"] = "passed"
    except BaseException as exc:
        summary["result"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        (output / "stop-monitor").touch()
        for filename in ("atom_server.log", "atom_client.log"):
            collect(["docker", "cp", f"{name}:/tmp/{filename}", str(output / filename)])
        collect(["docker", "cp", f"{name}:/workspace/accuracy_test_results", str(output)])
        state = collect(["docker", "inspect", "--format", "{{json .State}}", name], capture=True)
        (output / "container-state.json").write_text(state.stdout)
        collect(["docker", "exec", name, "bash", "-lc",
                     "cd /app/aiter-test/aiter/jit/build && "
                     "find . -type f \\( -name .ninja_log -o -name build.ninja \\) -print0 | "
                     f"tar --null -T - -czf {shlex.quote(str(remote_output / 'jit-build-logs.tar.gz'))}"])
        try:
            summary["model_revision_after"] = (model_root / ".hf-revision").read_text().strip()
        except OSError as exc:
            summary["model_revision_after"] = str(exc)
        collect(["docker", "stop", "--time", "20", name])
        collect(["docker", "rm", "-f", name])
        if monitor:
            try:
                monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        summary["end"] = time.time()
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
