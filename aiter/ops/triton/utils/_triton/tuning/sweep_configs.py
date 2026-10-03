# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Driver: sweep the full search space for one kernel and shape, then install the winners.

Per M:   plan            harness.py --plan: load defaults, generate combinations, prune
         check_settings  records can only be resumed when they were timed the same way
         sweep           harness.py --configs ...: the candidates are dealt round-robin to the
                         GPUs, one serial worker per GPU, with crash and hang recovery
         final_round     with several GPUs: re-time the baseline and the ten fastest on the first
                         GPU into <results>.final.jsonl, the only file the installer uses for
                         that M, so the winner comes from same-device numbers
         summarize
Then:    install         write_best_configs.py: select winners, merge buckets, validate, write

--all-buckets sweeps every M bucket of the family up to TOP_BUCKET_M (8192); the installer
sets "any" to a copy of the highest tuned bucket. The driver never imports torch or aiter: every step is a subprocess that sees
HIP_VISIBLE_DEVICES=<gpu>. It resumes from the records in --runs-dir.
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from _utils import (
    FINAL_STATUSES,
    TOP_BUCKET_M,
    add_shape_args,
    append_record,
    config_key,
    final_results_path,
    iter_records,
    plan_path,
    read_records,
    results_path,
    shape_from_args,
)
from kernels import get_spec

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_ROUND_SIZE = 10
STOP = (
    threading.Event()
)  # set on Ctrl-C or a fatal shard error: every running worker is terminated


def parse_args(argv=None):
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
        help="seconds allowed for planning and for a worker to become ready",
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
    return args


def worker_command(spec, shape, backend, extra):
    command = [sys.executable, os.path.join(HERE, "harness.py"), spec.name]
    for d in spec.dims:
        command += [f"--{d}", str(shape[d])]
    if backend:
        command += ["--backend", backend]
    return command + extra


def plan(spec, shape, backend, gpu, runs_dir, timeout):
    """Ask the worker for the space; keep the plan under a name unique to arch, backend, kernel and shape."""
    fd, space_file = tempfile.mkstemp(prefix="plan-", suffix=".json", dir=runs_dir)
    os.close(fd)
    command = worker_command(
        spec, shape, backend, ["--plan", "--space-out", space_file]
    )
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    try:
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        os.remove(space_file)
        sys.exit(
            f"planning {shape} on gpu {gpu} took more than {timeout} s (--setup-timeout)"
        )
    if result.returncode:
        os.remove(space_file)
        message = result.stderr.strip() or result.stdout.strip() or "planning failed"
        sys.exit(message.splitlines()[-1])
    with open(space_file) as f:
        plan_ = json.load(f)
    os.replace(
        space_file,
        plan_path(runs_dir, plan_["arch"], plan_["backend"], spec.name, shape),
    )
    return plan_


def all_bucket_ms(spec, args, gpu):
    """Every bucket of the family up to TOP_BUCKET_M; the installer copies the highest into "any"."""
    first = plan(
        spec,
        shape_from_args(args, spec.dims, 1),
        args.backend,
        gpu,
        args.runs_dir,
        args.setup_timeout,
    )
    ms = [b for b in first["bounds"] if b <= TOP_BUCKET_M]
    print(f"all buckets: M = {ms}", flush=True)
    return ms


def print_plan(plan_, shape):
    print(
        f"{plan_['kernel']} {plan_['backend']} {plan_['arch']} {shape}: keys from {plan_['default_path']}"
    )
    for key, values in plan_["candidate_values"].items():
        print(f"  {key}: {values}" + ("  (pinned)" if key in plan_["pinned"] else ""))
    if plan_["oversized_tiles_allowed"]:
        print(
            "  the kernel rejects every tile that fits the shape: block sizes above the shape are allowed"
        )
    skipped = ", ".join(f"{n} by {why}" for why, n in plan_["skipped"].items())
    print(
        f"  {plan_['raw']} combinations, skipped {skipped} -> {plan_['final']} configs, "
        f"~{plan_['final'] * 5 / 3600:.1f} GPU-hours at 5 s/config",
        flush=True,
    )


def check_settings(out, args):
    """Records can only be resumed when they were timed the same way."""
    ready = next((r for r in iter_records(out) if r["status"] == "ready"), None)
    if ready is None:
        return
    recorded = (ready["calls"], ready["replays"], ready.get("cold_mb"))
    if recorded != (args.calls, args.replays, args.cold_mb):
        sys.exit(
            f"{out} was timed with calls={recorded[0]} replays={recorded[1]} cold_mb={recorded[2]}; "
            "rerun with the same settings or --fresh"
        )


def discard(out):
    for path in glob.glob(
        out[: -len(".jsonl")] + "*"
    ):  # results, final round, logs, todo files
        os.remove(path)


def terminate(proc):
    """The one place a worker is killed and reaped."""
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def run_worker(command, env, log_path, stall, setup_timeout):
    """Run one worker to its end, or kill it.

    The worker prints "ready" once it is set up and one line per finished candidate. Until
    "ready" the limit is setup_timeout (imports, inputs, first compile), afterwards stall seconds
    per candidate. Returns "finished", "failed" (died after it was ready), "setup_failed",
    "timed_out" or "cancelled" (STOP was set).
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
        try:
            while proc.poll() is None:
                time.sleep(2)
                if size() != last:
                    last, t_last = size(), time.time()
                    ready = ready or became_ready()
                if STOP.is_set():
                    terminate(proc)
                    return "cancelled"
                if time.time() - t_last > (stall if ready else setup_timeout):
                    terminate(proc)
                    return "timed_out" if ready else "setup_failed"
        except KeyboardInterrupt:
            terminate(proc)
            raise
    if proc.returncode == 0:
        return "finished"
    return "failed" if ready or became_ready() else "setup_failed"


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
    outcome = run_worker(command, env, log_path, args.stall, args.setup_timeout)
    if outcome in ("finished", "cancelled"):
        return  # a cancelled worker's candidate keeps its start marker and is resumed next time
    if outcome == "setup_failed":
        sys.exit(f"worker on gpu {gpu} could not set up; see {log_path}")
    status = "hung" if outcome == "timed_out" else "crashed"
    records = read_records(out)
    for (
        candidate
    ) in batch:  # the start marker without a final record is the one it died on
        if records.get(config_key(candidate), {}).get("status") == "start":
            append_record(
                out,
                {
                    "status": status,
                    "baseline": candidate is None,
                    "config": candidate,
                    "error": f"worker {outcome}",
                },
            )
            print(f"  gpu {gpu}: {status}: {json.dumps(candidate)}", flush=True)


def sweep_shard(spec, shape, backend, candidates, out, gpu, args):
    """Run every candidate of this shard that has no final record yet, in batches, on one GPU."""
    while not STOP.is_set():
        records = read_records(out)
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


def sweep(spec, shape, backend, candidates, out, gpus, args):
    """Deal the candidates to the GPUs, one serial worker each; a failing shard stops the others."""
    shards = [candidates[i :: len(gpus)] for i in range(len(gpus))]
    with ThreadPoolExecutor(len(gpus)) as pool:
        futures = [
            pool.submit(sweep_shard, spec, shape, backend, shard, out, gpu, args)
            for shard, gpu in zip(shards, gpus)
        ]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            STOP.set()
            raise


def final_round(spec, shape, backend, out, gpu, args):
    """Re-time the baseline and the fastest candidates on one GPU into the final-round file."""
    records = read_records(out)
    ok = sorted(
        (r for r in records.values() if r["status"] == "ok" and not r.get("baseline")),
        key=lambda r: r["us"],
    )
    fastest = [r["config"] for r in ok[:FINAL_ROUND_SIZE]]
    final_out = final_results_path(out)
    if os.path.exists(final_out):
        os.remove(final_out)
    print(
        f"  final round on gpu {gpu}: re-timing the baseline and the {len(fastest)} fastest",
        flush=True,
    )
    run_batch(spec, shape, backend, [None] + fastest, final_out, gpu, args)


def summarize(out, candidates):
    records = read_records(out)
    counts = {}
    for candidate in candidates:
        status = records.get(config_key(candidate), {}).get("status", "missing")
        counts[status] = counts.get(status, 0) + 1
    if os.path.exists(final_results_path(out)):
        records = read_records(final_results_path(out))
        print("  final round (same-device numbers the installer uses):")
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
    args = parse_args(argv)
    spec, gpus = args.kernel, args.gpu
    os.makedirs(args.runs_dir, exist_ok=True)

    for M in args.M or all_bucket_ms(spec, args, gpus[0]):
        shape = shape_from_args(args, spec.dims, M)
        plan_ = plan(
            spec, shape, args.backend, gpus[0], args.runs_dir, args.setup_timeout
        )
        backend = plan_["backend"]
        print_plan(plan_, shape)

        out = results_path(args.runs_dir, plan_["arch"], backend, spec.name, shape)
        if args.fresh:
            discard(out)
        check_settings(out, args)
        print(f"  results: {out}", flush=True)
        candidates = [None] + plan_["configs"]  # None is the installed baseline
        sweep(spec, shape, backend, candidates, out, gpus, args)
        if len(gpus) > 1:
            final_round(spec, shape, backend, out, gpus[0], args)
        summarize(out, candidates)

    return install(spec, args, backend, gpus[0])


if __name__ == "__main__":
    sys.exit(main())
