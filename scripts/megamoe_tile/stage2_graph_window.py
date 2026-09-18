"""CPU-only resource checks before a coordinated EP16 Graph run."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


def assess_snapshot(cards, expected_gpus=8, max_idle_vram_bytes=2 << 30):
    """Zero utilization alone does not exclude an idle inference server."""
    if not isinstance(cards, dict):
        return ["GPU inventory is not a JSON object"]
    failures = []
    devices = {name: value for name, value in cards.items() if name.startswith("card")}
    if len(devices) != expected_gpus:
        failures.append(f"expected {expected_gpus} GPUs, observed {len(devices)}")
    for name, value in devices.items():
        try:
            used = int(value["VRAM Total Used Memory (B)"])
            busy = int(value["GPU use (%)"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{name}: incomplete utilization/VRAM evidence")
            continue
        if used > max_idle_vram_bytes or busy != 0:
            failures.append(f"{name}: VRAM={used} bytes, GPU use={busy}%")
    return failures


def observe_idle_window(read_snapshot, *, attempts=3, sleep=time.sleep):
    """Give transient startup activity up to two seconds to become idle.

    Every accepted observation must still satisfy the original zero-utilization
    and VRAM limits. Keep failed observations so a retry cannot erase evidence
    of contention. No rank allocates test tensors until both nodes agree.
    """
    observations = []
    for attempt in range(attempts):
        observed = {"cards": {}, "failures": []}
        try:
            observed["cards"] = read_snapshot()
            observed["failures"] = assess_snapshot(observed["cards"])
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            observed["failures"] = [str(error)]
        observations.append(observed)
        if not observed["failures"]:
            break
        if attempt + 1 < attempts:
            sleep(1)
    return observations


def guard_distributed_window(output_dir):
    """All CPU Gloo ranks agree before any rank starts GPU allocation."""
    import os
    import torch.distributed as dist

    local = None
    if int(os.environ["LOCAL_RANK"]) == 0:
        def read_snapshot():
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, check=True, timeout=20,
            )
            return json.loads(result.stdout)
        observations = observe_idle_window(read_snapshot)
        local = {"rank": dist.get_rank(), **observations[-1],
                 "observations": observations}
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    nodes = [row for row in gathered if row is not None]
    failed = len(nodes) != 2 or any(row["failures"] for row in nodes)
    if local is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "occupancy_preflight.json").write_text(
            json.dumps({"passed": not failed, "nodes": nodes}, indent=2) + "\n"
        )
    if failed:
        raise RuntimeError(
            "EP16 test window is occupied or unverified; no GPU job started. "
            f"See occupancy_preflight.json: {nodes}"
        )


if __name__ == "__main__":
    import argparse
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    try:
        guard_distributed_window(args.output_dir)
    finally:
        dist.destroy_process_group()
