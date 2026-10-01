#!/usr/bin/env python3
"""Read-only preflight: refuse to run a model outside the assigned GPU set."""

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    catalog = json.loads(Path(".github/benchmark/models_accuracy.json").read_text())
    model = next(item for item in catalog if item["model_name"] == args.model)
    model_root = Path("/models") / model["model_path"]
    import torch

    probe = {
        "packages": {name: importlib.metadata.version(name) for name in pins["packages"]},
        "hip": torch.version.hip,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "device_count": torch.cuda.device_count(),
        "render_devices": sorted(p.name for p in Path("/dev/dri").glob("renderD*")),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
                "AITER_JIT_VERBOSE", "MAX_JOBS", "ATOM_DISABLE_MMAP",
            )
        },
        "cgroup": {
            name: read_optional("/sys/fs/cgroup/" + name)
            for name in ("cpu.max", "cpuset.cpus.effective", "memory.max", "memory.events")
        },
        "model_revision": read_optional(model_root / ".hf-revision"),
        "aiter": subprocess.check_output(
            ["git", "-C", "/app/aiter-test", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    Path(args.output).write_text(json.dumps(probe, indent=2) + "\n")
    print(json.dumps(probe, indent=2), flush=True)
    assert probe["packages"] == pins["packages"], "Runtime packages drifted from the failed job"
    assert probe["aiter"] == pins["aiter"], "Wrong AITER checkout"
    assert probe["device_count"] == args.expected_devices, (
        "GPU visibility does not match the runner allocation; refusing model launch"
    )
    assert len(probe["cpu_affinity"]) == 20, "CPU allocation differs from the CI runner"
    assert probe["model_revision"] == pins["models"][args.model]["revision"], (
        "The shared model cache is missing or has a different revision; leaving it untouched"
    )


if __name__ == "__main__":
    main()
