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

With --compile-procs N, every M is planned first and N compile-only workers (harness.py
--precompile) on the --compile-gpus fill the Triton cache in sweep order while the timing
workers run, so the timing GPUs spend their time timing. The timing workers still compile
anything the pool has not reached; the results are the same with or without the pool.

--all-buckets sweeps every M bucket of the family up to TOP_BUCKET_M (8192); the installer
sets "any" to a copy of DEFAULT.json's "any". The driver never imports torch or aiter: every
step is a subprocess that sees HIP_VISIBLE_DEVICES=<gpu>. It resumes from the records in --runs-dir.
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
    inputs_hash,
    iter_records,
    plan_path,
    read_records,
    results_path,
    shape_from_args,
    shape_tag,
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
    parser.add_argument(
        "--plan-timeout",
        type=int,
        default=3600,
        help="seconds allowed for planning one M (a Triton space walks millions of combinations)",
    )
    parser.add_argument(
        "--compile-procs",
        type=int,
        default=0,
        help="compile-only workers that fill the Triton cache ahead of the sweep (0: off)",
    )
    parser.add_argument(
        "--compile-gpus",
        type=int,
        nargs="+",
        default=[],
        help="GPUs for the compile-only workers, not in --gpu (their input set-up runs kernels)",
    )
    parser.add_argument(
        "--screen",
        type=float,
        default=0,
        help="stop a candidate after one replay when it is slower than this x the fastest "
        "record of its M (0: off); the final round always times fully",
    )
    parser.add_argument("--replays", type=int, default=25)
    parser.add_argument("--calls", type=int, default=24)
    parser.add_argument("--cold-mb", type=int, default=1024)
    parser.add_argument(
        "--fresh", action="store_true", help="discard existing records for these shapes"
    )
    args = parser.parse_args(argv)
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if len(set(args.gpu)) != len(args.gpu):
        parser.error("--gpu values must be distinct: one worker per GPU")
    if bool(args.M) == args.all_buckets:
        parser.error("give --M values or --all-buckets, not both")
    if args.compile_procs and not args.compile_gpus:
        parser.error("--compile-procs needs --compile-gpus")
    if set(args.compile_gpus) & set(args.gpu):
        parser.error("--compile-gpus must not be timing GPUs (--gpu)")
    return args


def worker_command(spec, shape, backend, extra):
    command = [sys.executable, os.path.join(HERE, "harness.py"), spec.name]
    for d in spec.dims:
        command += [f"--{d}", str(shape[d])]
    if backend:
        command += ["--backend", backend]
    return command + extra


def kept_plan(spec, shape, backend, runs_dir):
    """The kept plan of this shape when the files it was built from have not changed, else None.

    Planning a Triton space walks millions of combinations, so a resumed sweep reuses it.
    """
    pattern = f"plan-*-{backend or '*'}-{spec.name}-{shape_tag(shape)}.json"
    for path in glob.glob(os.path.join(runs_dir, pattern)):
        with open(path) as f:
            plan_ = json.load(f)
        if plan_.get("inputs") and plan_["inputs_hash"] == inputs_hash(plan_["inputs"]):
            return plan_
    return None


def plan(spec, shape, backend, gpu, runs_dir, timeout):
    """Ask the worker for the space; keep the plan under a name unique to arch, backend, kernel and shape."""
    kept = kept_plan(spec, shape, backend, runs_dir)
    if kept is not None:
        return kept
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
            f"planning {shape} on gpu {gpu} took more than {timeout} s (--plan-timeout)"
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
    """Every bucket of the family up to TOP_BUCKET_M; "any" is DEFAULT.json's, never swept."""
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
    """Remove this shape's results, final round, logs and todo files (and no other shape's)."""
    final_out = final_results_path(out)
    paths = [out, final_out] + glob.glob(out + ".*") + glob.glob(final_out + ".*")
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)


def terminate(proc):
    """The one place a worker is stopped and reaped: SIGTERM lets it finish the candidate in
    hand (killing a worker while its kernels run can fault the GPU on gfx1250); a worker that
    does not stop within 30 s (a hung kernel or compile) is killed."""
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_worker(command, env, log_path, stall, setup_timeout, stop=STOP):
    """Run one worker to its end, or kill it.

    The worker prints "ready" once it is set up and one line per finished candidate. Until
    "ready" the limit is setup_timeout (imports, inputs, first compile), afterwards stall seconds
    per candidate. Returns "finished", "failed" (died after it was ready), "setup_failed",
    "timed_out" or "cancelled" (stop was set).
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
                if stop.is_set():
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


def run_batch(spec, shape, backend, batch, out, gpu, args, screen):
    """One worker process on one GPU for these candidates; marks the candidate it died on."""
    # the M's compile-ahead list (a final round has none, so its workers compile on their own)
    pool = ["--pool", f"{out}.pre.todo.json"] if args.compile_procs else []
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
            "--screen",
            str(screen),
            *pool,
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


def sweep_shard(spec, shape, backend, candidates, out, gpu, args, screen):
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
        run_batch(spec, shape, backend, todo[: args.batch], out, gpu, args, screen)


def sweep(spec, shape, backend, candidates, out, gpus, args):
    """Deal the candidates to the GPUs, one serial worker each; a failing shard stops the others."""
    shards = [candidates[i :: len(gpus)] for i in range(len(gpus))]
    with ThreadPoolExecutor(len(gpus)) as pool:
        futures = [
            pool.submit(
                sweep_shard, spec, shape, backend, shard, out, gpu, args, args.screen
            )
            for shard, gpu in zip(shards, gpus)
        ]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            STOP.set()
            raise


def final_round(spec, shape, backend, candidates, out, gpu, args):
    """Re-time the baseline and the fastest current candidates on one GPU into the final-round file."""
    records = read_records(out)
    current = {config_key(c) for c in candidates}  # records of an older plan stay out
    ok = sorted(
        (
            r
            for key, r in records.items()
            if key in current and r["status"] == "ok" and not r.get("baseline")
        ),
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
    sweep_shard(spec, shape, backend, [None] + fastest, final_out, gpu, args, 0)


def unclaimed(todo_file, size):
    with open(f"{todo_file}.next") as f:
        return int(f.read() or 0) < size


def precompile_worker(
    spec, shape, backend, todo_file, size, out, slot, gpu, args, stop
):
    """One compile-only slot: a worker claims the next candidate of todo_file until none is
    left; a worker killed on a huge compile is replaced, so the pool keeps its size."""
    log_path = f"{out}.pre{slot}.log"
    command = worker_command(
        spec,
        shape,
        backend,
        ["--precompile", "--configs", todo_file, "--claim", f"{todo_file}.next"],
    )
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    while not stop.is_set():
        outcome = run_worker(
            command + ["--out", out],
            env,
            log_path,
            4 * args.stall,  # a compile-only worker spends its time in one compile
            args.setup_timeout,
            stop,
        )
        if outcome == "finished":  # keep the log only when something went wrong
            os.remove(log_path)
            return
        if outcome == "setup_failed" or not unclaimed(todo_file, size):
            return


def pool_files(spec, args, plans):
    """Write every M's compile-ahead list up front, so the timing workers of an M the pool has
    not reached yet help it instead of compiling on their own. Returns {M: (out, todo, files)}.
    """
    pools = {}
    for M, plan_ in plans.items():
        shape = shape_from_args(args, spec.dims, M)
        out = results_path(
            args.runs_dir, plan_["arch"], plan_["backend"], spec.name, shape
        )
        records = read_records(out)
        todo = [
            c
            for c in [None] + plan_["configs"]
            if records.get(config_key(c), {}).get("status") not in FINAL_STATUSES
        ]
        # one shared list, claimed in order: a slow compile holds up only its own candidate
        todo_file = f"{out}.pre.todo.json"
        with open(todo_file, "w") as f:
            json.dump(todo, f)
        for suffix, content in ((".next", "0"), (".done", "")):
            with open(todo_file + suffix, "w") as f:
                f.write(content)
        pools[M] = (out, todo, todo_file)
    return pools


def remove_pool_files(todo_file):
    for path in (todo_file + ".next", todo_file + ".done", todo_file):
        if os.path.exists(path):
            os.remove(path)


def precompile_ahead(spec, args, plans, pools, stop):
    """Compile the unfinished candidates of every M, in sweep order, on the compile GPUs."""
    for M, plan_ in plans.items():
        out, todo, todo_file = pools[M]
        if stop.is_set() or STOP.is_set():
            remove_pool_files(todo_file)
            continue
        shape = shape_from_args(args, spec.dims, M)
        slots = min(args.compile_procs, len(todo))
        gpus = args.compile_gpus
        with ThreadPoolExecutor(max(slots, 1)) as pool:
            for slot in range(slots):
                pool.submit(
                    precompile_worker,
                    spec,
                    shape,
                    plan_["backend"],
                    todo_file,
                    len(todo),
                    out,
                    slot,
                    gpus[slot % len(gpus)],
                    args,
                    stop,
                )
        # the timing workers of this M now compile on their own
        remove_pool_files(todo_file)
        print(f"  compile-ahead: M={M} done", flush=True)


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
    signal.signal(
        signal.SIGTERM, lambda signum, frame: STOP.set()
    )  # stop the workers cleanly
    spec, gpus = args.kernel, args.gpu
    os.makedirs(args.runs_dir, exist_ok=True)

    ms = args.M or all_bucket_ms(spec, args, gpus[0])
    with ThreadPoolExecutor(len(ms)) as pool:  # a Triton space takes minutes per M
        planned = {
            M: pool.submit(
                plan,
                spec,
                shape_from_args(args, spec.dims, M),
                args.backend,
                gpus[0],
                args.runs_dir,
                args.plan_timeout,
            )
            for M in ms
        }
    plans = {}
    for M in ms:
        shape = shape_from_args(args, spec.dims, M)
        plans[M] = planned[M].result()
        out = results_path(
            args.runs_dir, plans[M]["arch"], plans[M]["backend"], spec.name, shape
        )
        if args.fresh:
            discard(out)
        check_settings(out, args)

    stop_precompile = threading.Event()
    pools = pool_files(spec, args, plans) if args.compile_procs else {}
    precompiler = threading.Thread(
        target=precompile_ahead,
        args=(spec, args, plans, pools, stop_precompile),
        daemon=True,
    )
    if args.compile_procs:
        precompiler.start()
    try:
        for M, plan_ in plans.items():
            shape = shape_from_args(args, spec.dims, M)
            backend = plan_["backend"]
            print_plan(plan_, shape)
            out = results_path(args.runs_dir, plan_["arch"], backend, spec.name, shape)
            print(f"  results: {out}", flush=True)
            candidates = [None] + plan_["configs"]  # None is the installed baseline
            sweep(spec, shape, backend, candidates, out, gpus, args)
            if STOP.is_set():
                sys.exit("stopped by SIGTERM; rerun the same command to resume")
            # the installer prefers a final-round file, so an existing one is always refreshed
            if len(gpus) > 1 or os.path.exists(final_results_path(out)):
                final_round(spec, shape, backend, candidates, out, gpus[0], args)
            summarize(out, candidates)
    finally:
        stop_precompile.set()
        if precompiler.is_alive():
            precompiler.join()

    return install(spec, args, backend, gpus[0])


if __name__ == "__main__":
    sys.exit(main())
