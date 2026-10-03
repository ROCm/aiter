# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Driver: sweep the full search space for one kernel and shape, then install the winners.

Flow, per M:   plan       harness.py --plan: load defaults, generate combinations, prune
               benchmark  harness.py --configs ...: the candidates are dealt round-robin to the
                          GPUs, one serial worker per GPU, with crash and hang recovery; with
                          several GPUs a final round re-times the baseline and the ten fastest
                          on the first GPU, so the winner is picked from same-device numbers
               summary
Afterwards:    install    write_best_configs.py: select winners, merge buckets, validate, write

--all-buckets sweeps every M bucket of the family up to TOP_BUCKET_M (4096); "any" is left as
installed. The driver never imports torch or aiter: every step is a subprocess that sees
HIP_VISIBLE_DEVICES=<gpu>. It resumes from the records in --runs-dir.
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from _utils import (
    FINAL_STATUSES,
    TOP_BUCKET_M,
    add_shape_args,
    append_record,
    config_key,
    read_records,
    results_path,
    shape_from_args,
    shape_tag,
)
from kernels import get_spec

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_ROUND_SIZE = 10


def worker_command(spec, shape, backend, extra):
    command = [sys.executable, os.path.join(HERE, "harness.py"), spec.name]
    for d in spec.dims:
        command += [f"--{d}", str(shape[d])]
    if backend:
        command += ["--backend", backend]
    return command + extra


def plan(spec, shape, backend, gpu, runs_dir):
    space_file = os.path.join(runs_dir, f"plan-{spec.name}-{shape_tag(shape)}.json")
    command = worker_command(
        spec, shape, backend, ["--plan", "--space-out", space_file]
    )
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or "planning failed"
        sys.exit(message.splitlines()[-1])
    with open(space_file) as f:
        return json.load(f)


def print_plan(plan_, shape):
    print(
        f"{plan_['kernel']} {plan_['backend']} {plan_['arch']} {shape}: keys from {plan_['default_path']}"
    )
    for key, values in plan_["candidate_values"].items():
        print(f"  {key}: {values}" + ("  (pinned)" if key in plan_["pinned"] else ""))
    skipped = ", ".join(f"{n} by {why}" for why, n in plan_["skipped"].items())
    print(
        f"  {plan_['raw']} combinations, skipped {skipped} -> {plan_['final']} configs, "
        f"~{plan_['final'] * 5 / 3600:.1f} GPU-hours at 5 s/config",
        flush=True,
    )


def run_worker_with_watchdog(command, env, log_path, stall, setup_timeout):
    """Run the worker; kill its process group when its log stops growing.

    The worker prints "ready" once it is set up and one line per finished candidate. Until
    "ready" the limit is setup_timeout (imports, inputs, first compile), afterwards stall seconds
    per candidate. Returns (returncode, ready, killed).
    """

    def size():
        return os.path.getsize(log_path) if os.path.exists(log_path) else 0

    def became_ready():
        with open(log_path) as f:
            f.seek(start)
            return any(line.startswith("ready") for line in f)

    start = last = size()
    t_last = time.time()
    ready = False
    with open(log_path, "a") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while proc.poll() is None:
            time.sleep(2)
            if size() != last:
                last, t_last = size(), time.time()
                ready = ready or became_ready()
            if time.time() - t_last > (stall if ready else setup_timeout):
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return proc.returncode, ready, True
    return proc.returncode, ready or became_ready(), False


def run_batch(spec, shape, backend, batch, out, gpu, args):
    """One worker process on one GPU for these candidates; marks the candidate it died on."""
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    todo_file, log_path = f"{out}.gpu{gpu}.todo.json", f"{out}.gpu{gpu}.log"
    with open(todo_file, "w") as f:
        json.dump(batch, f)
    command = worker_command(
        spec,
        shape,
        backend,
        [
            "--replays",
            str(args.replays),
            "--calls",
            str(args.calls),
            "--cold-mb",
            str(args.cold_mb),
            "--configs",
            todo_file,
            "--out",
            out,
        ],
    )
    rc, ready, killed = run_worker_with_watchdog(
        command, env, log_path, args.stall, args.setup_timeout
    )
    if rc == 0:
        return
    if not ready:
        sys.exit(
            f"worker on gpu {gpu} failed before it was ready (rc={rc}); see {log_path}"
        )
    records, _ = read_records(out)
    for (
        candidate
    ) in batch:  # a start marker without a final record: the one the worker died on
        if records.get(config_key(candidate), {}).get("status") == "start":
            status = "hung" if killed else "crashed"
            append_record(
                out,
                {
                    "status": status,
                    "baseline": candidate is None,
                    "config": candidate,
                    "error": f"worker rc={rc}",
                },
            )
            print(f"  gpu {gpu}: {status}: {json.dumps(candidate)}", flush=True)


def benchmark_shard(spec, shape, backend, candidates, out, gpu, args):
    """Run every candidate of this shard that has no final record yet, in batches, on one GPU."""
    while True:
        records, _ = read_records(out)
        todo = [
            c
            for c in candidates
            if records.get(config_key(c), {}).get("status") not in FINAL_STATUSES
        ]
        if not todo:
            return
        print(
            f"  gpu {gpu}: {len(candidates) - len(todo)}/{len(candidates)} done, "
            f"running {min(len(todo), args.batch)}",
            flush=True,
        )
        run_batch(spec, shape, backend, todo[: args.batch], out, gpu, args)


def benchmark(spec, shape, backend, candidates, out, gpus, args):
    """Deal the candidates to the GPUs; with several GPUs, re-time the fastest on the first one."""
    shards = [candidates[i :: len(gpus)] for i in range(len(gpus))]
    with ThreadPoolExecutor(len(gpus)) as pool:
        futures = [
            pool.submit(benchmark_shard, spec, shape, backend, shard, out, gpu, args)
            for shard, gpu in zip(shards, gpus)
        ]
        for future in futures:
            future.result()
    if len(gpus) > 1:
        records, _ = read_records(out)
        ok = sorted(
            (
                r
                for r in records.values()
                if r["status"] == "ok" and not r.get("baseline")
            ),
            key=lambda r: r["us"],
        )
        fastest = [r["config"] for r in ok[:FINAL_ROUND_SIZE]]
        print(
            f"  final round on gpu {gpus[0]}: re-timing the baseline and the {len(fastest)} fastest",
            flush=True,
        )
        run_batch(spec, shape, backend, [None] + fastest, out, gpus[0], args)


def summarize(out, candidates):
    records, _ = read_records(out)
    baseline = records.get("baseline", {})
    if baseline.get("status") == "ok":
        print(
            f"  baseline (installed, is_tuned={baseline.get('is_tuned')}): "
            f"{baseline['us']:.3f} us  {json.dumps(baseline['config'])}"
        )
    ok = sorted(
        (r for r in records.values() if r["status"] == "ok" and not r.get("baseline")),
        key=lambda r: r["us"],
    )
    for record in ok[:10]:
        print(
            f"  {record['us']:10.3f} us  {record['tflops']:8.1f} TFLOPS  {json.dumps(record['config'])}"
        )
    counts = {}
    for candidate in candidates:
        status = records.get(config_key(candidate), {}).get("status", "missing")
        counts[status] = counts.get(status, 0) + 1
    print(f"  {counts}", flush=True)


def install(spec, args, backend, gpu):
    command = [
        sys.executable,
        os.path.join(HERE, "write_best_configs.py"),
        spec.name,
        "--N",
        str(args.N),
        "--K",
        str(args.K),
    ]
    if "B" in spec.dims:
        command += ["--B", str(args.B)]
    command += ["--backend", backend, "--runs-dir", args.runs_dir]
    print("\ninstalling:", " ".join(command[1:]), flush=True)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    return subprocess.run(command, env=env, check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kernel", type=get_spec)
    add_shape_args(parser, with_m=False)
    parser.add_argument("--M", type=int, nargs="+", help="M values to sweep")
    parser.add_argument(
        "--all-buckets",
        action="store_true",
        help=f"sweep every M bucket of the family up to {TOP_BUCKET_M}",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        required=True,
        help="GPUs to use; candidates are dealt round-robin, one serial worker per GPU",
    )
    parser.add_argument(
        "--backend",
        choices=("triton", "gluon"),
        help="default: what the wrapper picks on this arch",
    )
    parser.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    parser.add_argument(
        "--batch", type=int, default=100, help="configs per worker process"
    )
    parser.add_argument(
        "--stall",
        type=int,
        default=300,
        help="seconds without a finished candidate before a worker is killed",
    )
    parser.add_argument(
        "--setup-timeout",
        type=int,
        default=600,
        help="seconds allowed before a worker is ready",
    )
    parser.add_argument("--replays", type=int, default=25)
    parser.add_argument("--calls", type=int, default=24)
    parser.add_argument("--cold-mb", type=int, default=1024)
    parser.add_argument(
        "--fresh", action="store_true", help="discard existing records for these shapes"
    )
    args = parser.parse_args(argv)
    if bool(args.M) == args.all_buckets:
        parser.error("give --M values or --all-buckets, not both")

    spec = args.kernel
    gpus = args.gpu
    os.makedirs(args.runs_dir, exist_ok=True)

    Ms = args.M
    if (
        args.all_buckets
    ):  # every bucket of the family up to TOP_BUCKET_M; "any" stays as installed
        bounds = plan(
            spec,
            shape_from_args(args, spec.dims, 1),
            args.backend,
            gpus[0],
            args.runs_dir,
        )["bounds"]
        Ms = [b for b in bounds if b <= TOP_BUCKET_M]
        print(f"all buckets: M = {Ms}", flush=True)

    for M in Ms:
        shape = shape_from_args(args, spec.dims, M)
        plan_ = plan(spec, shape, args.backend, gpus[0], args.runs_dir)
        backend = plan_["backend"]
        print_plan(plan_, shape)

        out = results_path(args.runs_dir, plan_["arch"], backend, spec.name, shape)
        if args.fresh:
            for path in [out] + glob.glob(out + ".*"):
                os.remove(path)
        candidates = [None] + plan_["configs"]  # None is the installed baseline
        print(f"  results: {out}", flush=True)
        benchmark(spec, shape, backend, candidates, out, gpus, args)
        summarize(out, candidates)

    return install(spec, args, backend, gpus[0])


if __name__ == "__main__":
    sys.exit(main())
