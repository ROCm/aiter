#!/usr/bin/env python3
"""Summarize Aiter prebuild timing from setup.py build logs."""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

FINISH_RE = re.compile(r"finish build \[([^\]]+)\], cost ([0-9.]+)s")


def parse_module_costs(log_path: Path) -> list[tuple[str, float]]:
    module_costs: list[tuple[str, float]] = []
    try:
        with log_path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                match = FINISH_RE.search(line)
                if match:
                    module_costs.append((match.group(1), float(match.group(2))))
    except OSError as exc:
        print(f"::warning::Unable to read prebuild log {log_path}: {exc}")
    return module_costs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", required=True, type=Path, help="Path to the tee'd prebuild log"
    )
    parser.add_argument(
        "--build-status", required=True, type=int, help="setup.py exit code"
    )
    parser.add_argument(
        "--start", required=True, type=int, help="Prebuild start timestamp in seconds"
    )
    parser.add_argument(
        "--end", required=True, type=int, help="Prebuild end timestamp in seconds"
    )
    parser.add_argument(
        "--kernel-glob",
        default="aiter/jit/*.so",
        help="Glob used to count prebuilt kernel shared objects",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module_costs = parse_module_costs(args.log)
    kernels = sorted(glob.glob(args.kernel_glob))
    wall_seconds = max(0, args.end - args.start)
    total_module_seconds = sum(cost for _, cost in module_costs)

    print("=== Aiter prebuild summary ===")
    print(f"Runner: {os.environ.get('AITER_RUNNER_NAME', 'unknown')}")
    print(f"GPU_ARCHS: {os.environ.get('GPU_ARCHS', 'unknown')}")
    print(f"PREBUILD_KERNELS: {os.environ.get('PREBUILD_KERNELS', 'unknown')}")
    print(f"MAX_JOBS: {os.environ.get('MAX_JOBS', 'unknown')}")
    print(f"Build status: {args.build_status}")
    print(f"Prebuild wall time: {wall_seconds}s ({wall_seconds / 60:.1f} min)")
    print(f"Kernel count: {len(kernels)}")
    print(f"Module builds observed: {len(module_costs)}")
    print(
        f"Total module compile cost: {total_module_seconds:.1f}s ({total_module_seconds / 60:.1f} min)"
    )

    if module_costs:
        print("Top slowest module builds:")
        for name, cost in sorted(module_costs, key=lambda item: item[1], reverse=True)[
            :20
        ]:
            print(f"  {name}: {cost:.1f}s")

        print("All module build costs (seconds):")
        for name, cost in sorted(module_costs):
            print(f"  {name}: {cost:.1f}")
    else:
        print("::warning::No module build cost lines were found in the prebuild log")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
