# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune a GEMM with rocprofv3 and save a faster or missing config.

python tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024 --backend both
python tune_gemm.py --list
"""

import argparse
import csv
import inspect
import itertools
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from gemm_cases import CASES, backend_kwarg


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", nargs="?", choices=CASES)
    parser.add_argument("dims", nargs="*", metavar="DIM=INT")
    parser.add_argument("--list", action="store_true", help="list cases without a GPU")
    parser.add_argument(
        "--backend",
        choices=("triton", "gluon", "both"),
        help="default: wrapper's choice; both: tune each backend separately",
    )
    parser.add_argument("--space", nargs="+", default=[], metavar="KEY=V1,V2")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="seconds per config, including compilation",
    )
    parser.add_argument(
        "--runs", type=int, default=250, help="profiled invocations per config"
    )
    parser.add_argument(
        "--kernel-names",
        nargs="+",
        help="kernel-name substrings to include in GPU timing",
    )
    args = parser.parse_args()
    if not args.list and not args.op:
        parser.error("choose a GEMM, or use --list")
    if args.timeout <= 0 or args.runs <= 0:
        parser.error("--timeout and --runs must be positive")
    return args


def parse_dims(tokens):
    dims = {}
    for token in tokens:
        name, _, value = token.partition("=")
        if not name or not value.isdigit() or int(value) <= 0 or name in dims:
            raise ValueError(f"Expected a positive DIM=INT, got {token!r}")
        dims[name] = int(value)
    return dims


def parse_space(tokens):
    space = {}
    for token in tokens:
        key, separator, values = token.partition("=")
        if not key or not separator or not values:
            raise ValueError(f"Expected KEY=V1,V2, got {token!r}")
        space[key] = []
        for value in values.split(","):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # Strings such as .cg do not need quotes.
            space[key].append(value)
    return space


def candidate_configs(record, overrides):
    """Only search keys declared by this kernel's architecture/backend default."""
    keys = dict.fromkeys(
        key
        for bucket in record["defaults"].values()
        if isinstance(bucket, dict)
        for key in bucket
    )
    unknown = overrides.keys() - keys.keys()
    if unknown:
        print(
            f"Skipping search keys absent from this backend's DEFAULT.json: {sorted(unknown)}"
        )
    lookup = record["lookup"]
    space = {}
    for key in keys:
        if key == "kpack" and record["arch"] != "gfx942":
            continue  # Follow the config rules for retired kpack on newer architectures.
        space[key] = overrides.get(
            key,
            candidate_values(
                key, lookup["M"], lookup["N"], lookup["K"], record["seen"]
            ),
        )
    print("Search space:", json.dumps(space), flush=True)
    for values in itertools.product(*space.values()):
        yield dict(zip(space, values))


def candidate_values(key, M, N, K, seen):
    """Values to try for one config key."""
    fixed = {
        "GROUP_SIZE_M": [1, 4, 8],
        "num_warps": [1, 2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [1, 2, 4, 6, 8],
        "matrix_instr_nonkdim": [16, 32],
        "cache_modifier": [".cg", None],
        "kpack": [1, 2],
    }
    if key in ("BLOCK_SIZE_M", "BLOCK_M"):
        values = [4, 8, 16] + [v for v in (32, 64, 128, 256, 512) if v <= M]
    elif key in ("BLOCK_SIZE_N", "BLOCK_N"):
        values = [16] + [v for v in (32, 64, 128, 256) if N is None or v <= N]
    elif key in ("BLOCK_SIZE_K", "BLOCK_K"):
        values = [16, 32, 64, 128] + [
            v for v in (256, 512, 1024) if K is None or v <= K
        ]
    elif key == "NUM_KSPLIT":
        values = [1] + [v for v in (2, 3, 4, 7, 8, 14, 16, 28) if K is None or v <= K]
    else:
        values = list(fixed.get(key, []))
    # Keep published values even when a generic heuristic would omit them.
    # Unknown keys (buffers, variants, ...) need no change to this script:
    # the author's JSON values or --space define their search.
    for value in seen.get(key, []):
        if value not in values:
            values.append(value)
    return values


def kernel_time(directory, names, runs):
    """Median sum of selected GPU kernel durations per invocation, in microseconds."""
    rows = []
    for path in directory.rglob("*_kernel_trace.csv"):
        with path.open(newline="") as trace:
            rows.extend(csv.DictReader(trace))
    samples, duration = [], None
    for row in sorted(rows, key=lambda row: int(row["Start_Timestamp"])):
        name = row["Kernel_Name"].lower()
        if "tuning_run_marker" in name:
            if duration is not None:
                samples.append(duration)
            duration = 0
        elif duration is not None and any(part.lower() in name for part in names):
            elapsed = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            if elapsed <= 0:
                raise ValueError(f"Invalid kernel duration for {name}: {elapsed}")
            duration += elapsed
    if len(samples) != runs or any(value <= 0 for value in samples):
        raise ValueError(
            f"Incomplete kernel trace: expected {runs} nonempty runs, got {len(samples)}; check kernel names {names}"
        )
    return statistics.median(samples) / 1000


def profile_config(spec, directory, timeout, names, errors, results):
    """A fresh process contains compilation errors, GPU crashes and timeouts."""
    directory.mkdir()
    spec_path = directory / "spec.json"
    spec_path.write_text(json.dumps(spec))
    command = [
        "rocprofv3",
        "--kernel-trace",
        "-f",
        "csv",
        "-d",
        str(directory),
        "-o",
        "profile",
        "--",
        sys.executable,
        str(Path(__file__).with_name("profile_gemm.py")),
        str(spec_path),
    ]
    config = spec["config"]
    elapsed, error = None, None
    try:
        # A file avoids pipe deadlocks and retains the full compiler/driver error.
        with (directory / "output.log").open("w") as output:
            process = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                process.wait(timeout=timeout)
            finally:
                # Kill the whole group, including a hung GPU child of rocprofv3.
                if process.poll() is None or process.returncode != 0:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
        if process.returncode:
            raise RuntimeError(f"rocprofv3 exited with code {process.returncode}")
        elapsed = kernel_time(directory, names, spec["runs"])
    except Exception:  # noqa: BLE001 -- log this config's failure and keep tuning
        error = traceback.format_exc()
        print(f"Failed config; continuing. See {errors.name}", flush=True)
    # The worker records the baseline config before executing it, even if it crashes.
    if config is None and Path(spec["record"]).exists():
        config = json.loads(Path(spec["record"]).read_text()).get("config")
    if error is not None:
        errors.write(
            json.dumps({"backend": spec["backend"], "config": config}) + "\n" + error
        )
        output_path = directory / "output.log"
        if output_path.exists():
            errors.write(output_path.read_text(errors="replace"))
        errors.write("\n")
        errors.flush()
    results.write(
        json.dumps(
            {"config": config, "time_us": elapsed, "status": "error" if error else "ok"}
        )
        + "\n"
    )
    results.flush()
    return elapsed if elapsed is not None else math.inf


def tune(args, dims, backend, errors, results):
    case = CASES[args.op]
    overrides = {**case.space, **parse_space(args.space)}
    names = args.kernel_names or case.kernels
    with tempfile.TemporaryDirectory(prefix="aiter-tune-") as temporary:
        directory = Path(temporary)
        spec = {
            "op": args.op,
            "dims": dims,
            "backend": backend,
            "config": None,
            "runs": args.runs,
            "record": str(directory / "lookup.json"),
            "reference": str(directory / "reference.pt"),
            "lookup": None,
        }

        # 1. Profile the config used today. Its lookup tells us where to save.
        current_us = profile_config(
            spec, directory / "current", args.timeout, names, errors, results
        )
        if not Path(spec["record"]).exists():
            print(
                "No config lookup was reached; check backend support and the error log."
            )
            return False
        record = json.loads(Path(spec["record"]).read_text())
        spec["lookup"] = record["lookup"]
        selected = record["lookup"]["backend"]
        if backend is not None and selected != backend:
            errors.write(
                f"Requested {backend}, but wrapper selected {selected}; no configs written.\n"
            )
            print(f"Requested {backend}, but wrapper selected {selected}; skipping.")
            return False
        print(
            f"{record['arch']}/{selected}: current {current_us:.3f} us ({record['source']})",
            flush=True,
        )
        if not Path(spec["reference"]).exists():
            print(
                "Current config failed; candidates will only be checked for finite outputs."
            )
        best_config = record["config"] if math.isfinite(current_us) else None
        best_us = current_us

        # 2. Each config gets its own process. A failure cannot stop the next one.
        for index, config in enumerate(candidate_configs(record, overrides)):
            spec["config"] = config
            elapsed = profile_config(
                spec, directory / str(index), args.timeout, names, errors, results
            )
            if elapsed < best_us:
                best_config, best_us = config, elapsed
                print(f"Best: {best_us:.3f} us {json.dumps(config)}", flush=True)

        # 3. Save only a faster config, or fill a missing tuned M bucket.
        if best_config is None:
            print("No config worked; no config file changed.")
            return False
        if not record["is_tuned"] or best_us < current_us:
            table = record["table"]
            table[record["bucket"]] = best_config
            Path(record["target"]).write_text(json.dumps(table, indent=4) + "\n")
            print(f"Saved {record['bucket']} in {record['target']}")
        else:
            print("No faster config found; kept the existing config.")
        return True


def main():
    args = parse_args()
    if args.list:
        for name, case in CASES.items():
            print(f"{name}{inspect.signature(case)}")
        return
    case = CASES[args.op]
    dims = parse_dims(args.dims)
    backends = ["gluon", "triton"] if args.backend == "both" else [args.backend]
    for backend in backends:
        inspect.signature(case).bind(**dims, **backend_kwarg(backend))
    if os.name != "posix" or shutil.which("rocprofv3") is None:
        raise SystemExit("Run tuning on Linux with ROCm and rocprofv3 on PATH")
    shape = "-".join(f"{key}={value}" for key, value in dims.items())
    succeeded = True
    for backend in backends:
        tag = f"{args.op}-{shape}-{backend or 'auto'}"
        with open(f"errors-{tag}.txt", "w") as errors, open(
            f"results-{tag}.jsonl", "w"
        ) as results:
            print(f"Logs: {errors.name}, {results.name}", flush=True)
            succeeded = tune(args, dims, backend, errors, results) and succeeded
    if not succeeded:
        raise SystemExit("One or more backends could not be tuned; see the error logs")


if __name__ == "__main__":
    main()
