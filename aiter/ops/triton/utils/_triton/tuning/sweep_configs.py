"""Profile every candidate config of one GEMM at one shape and log the runtimes.

    python3 sweep_configs.py <op> <M> <DIM>=<int>... [--gpu G] [--backend B]

The op runs under rocprofv3 with each candidate answering its own
get_gemm_config() lookup. The keys swept are the keys of the config family's
DEFAULT.json for this GPU's arch and the backend the op runs, so one command
tunes any arch and either backend. Each candidate's output is checked against
the output of the config the library resolves. Results go to
sweeps/<op>/<arch>-<backend>/<dims>/M=<M>.jsonl for write_best_configs.py.
"""

import argparse
import sys

from _utils import (
    append_jsonl,
    block_keys,
    build_space,
    candidates,
    describe_lookup,
    get_case,
    parse_dims,
    parse_space,
    read_jsonl,
    resolve_family,
    run_worker,
    shape_tag,
    sweep_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("op", help="tuning case from gemm_cases.py, e.g. gemm_a8w8")
    parser.add_argument("M", type=int, help="M dim")
    parser.add_argument(
        "dims", nargs="+", metavar="DIM=INT", help="the op's other dims: N=7168 K=2048"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU to run on")
    parser.add_argument(
        "--backend",
        choices=("triton", "gluon"),
        help="backend to tune, for ops that have both (default: the wrapper's pick)",
    )
    parser.add_argument(
        "--space",
        nargs="+",
        default=[],
        metavar="KEY=V1,V2",
        help="values to try for a key instead of the defaults, e.g. BLOCK_SIZE_K=128",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="configs per profiled process"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds per profiled process"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="do not compare each config's output with the resolved config's",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing log"
    )
    parser.add_argument("--verbose", action="store_true", help="print every skip")
    return parser.parse_args()


def sweep(args, spec, configs, blocks, kernels, log):
    """Profile ``configs`` in batches, appending one result line each to ``log``.

    A process that crashes or hangs costs only the config it was running: the
    configs it had not reached run again in a new process. A config that runs
    out of resources rules out every later config with the same tile sizes."""
    pending = list(range(len(configs)))
    attempts = [0] * len(configs)
    failed_blocks = set()

    def block(i):
        return tuple(configs[i].get(key) for key in blocks)

    while pending:
        batch, pending = pending[: args.batch_size], pending[args.batch_size :]
        runnable = []
        for i in batch:
            if block(i) in failed_blocks:
                error = "skipped: these tile sizes ran out of resources"
                append_jsonl(log, {"config": configs[i], "error": error})
            else:
                runnable.append(i)
        if not runnable:
            continue
        print(
            f"Running cases {runnable[0]} ~ {runnable[-1]} of {len(configs)}",
            flush=True,
        )
        batch_spec = dict(spec, candidates=[configs[i] for i in runnable])
        run = run_worker(batch_spec, args.gpu, args.timeout, kernels)
        if run.record is None:
            tail = "\n".join(run.stderr.strip().splitlines()[-20:])
            sys.exit(f"the worker failed before running the op:\n{tail}")

        statuses = {s["index"]: s for s in run.statuses}
        attempted = [s["index"] for s in run.statuses if s["status"] != "skipped"]
        segments = (run.segments or [])[1:]  # segment 0 is setup
        timings = dict(zip(attempted, (us for _, us in segments)))
        # Exit code 3: the worker stopped after a GPU fault it reported itself.
        # Any other failure happened while running the first config with no
        # status, which is the one to blame.
        blame = run.returncode not in (0, 3)
        rerun = []
        for local, i in enumerate(runnable):
            status = statuses.get(local)
            if status is None:
                if blame:
                    blame = False
                    exit_reason = (
                        "timed out"
                        if run.returncode is None
                        else f"exited with {run.returncode}"
                    )
                    error = f"worker {exit_reason} while running this config"
                    append_jsonl(log, {"config": configs[i], "error": error})
                    if args.verbose:
                        print(run.stderr.strip()[-2000:], flush=True)
                else:
                    rerun.append(i)
            elif status["status"] == "ok":
                if local not in timings:
                    rerun.append(i)  # the trace was lost with the process
                elif timings[local] is None:
                    error = f"no kernel name contains any of {list(kernels)}"
                    append_jsonl(log, {"config": configs[i], "error": error})
                else:
                    append_jsonl(log, {"config": configs[i], "us": timings[local]})
            else:
                append_jsonl(log, {"config": configs[i], "error": status["error"]})
                if status["resource"]:
                    failed_blocks.add(block(i))
                if args.verbose:
                    print(f"\t{configs[i]}: {status['error']}", flush=True)

        for i in rerun:
            attempts[i] += 1
            if attempts[i] >= 2:
                error = "no runtime after two attempts"
                append_jsonl(log, {"config": configs[i], "error": error})
        pending = [i for i in rerun if attempts[i] < 2] + pending


def main():
    args = parse_args()
    dims = parse_dims(args.dims)
    case = get_case(args.op, dims, args.backend)
    spec = {"op": args.op, "M": args.M, "dims": dims, "backend": args.backend}

    # The op with the config the library resolves: which family, backend and
    # keys it uses, and the runtime to beat.
    run = run_worker(dict(spec, candidates=None), args.gpu, args.timeout, case.kernels)
    lookup, family = resolve_family(run, args.op)
    baseline = run.segments[1][1] if run.segments and len(run.segments) > 1 else None
    arch, backend = family["arch"], lookup["backend"]
    current_config = run.record.get("config") or {}
    print(f"{args.op} M={args.M} {shape_tag(args.op, dims)}: {describe_lookup(lookup)}")
    print(f"Current config on {arch}: {current_config}")
    if run.record["error"] is not None:
        print(f"Warning: the current config fails ({run.record['error']});")
        print("sweeping without checking outputs against it")
    if baseline is not None:
        print(f"Current runtime: {baseline:.3f} (us)")
    untunable = [k for k in current_config if k not in family["keys"]]
    if untunable:
        print(
            f"Note: the current config also sets {', '.join(untunable)}, which "
            "DEFAULT.json does not; add them there to tune them"
        )

    N = lookup["N"] if lookup["N"] is not None else dims.get("N")
    K = lookup["K"] if lookup["K"] is not None else dims.get("K")
    cli_space = parse_space(args.space)
    space, unknown = build_space(family, args.M, N, K, case.space, cli_space)
    for key in unknown:
        if key in cli_space:
            print(f"Warning: {key} is not a key of this config family; ignored")
    print("Raw tuning space:", flush=True)
    for key, values in space.items():
        print(f"\t{key} = {values}", flush=True)
    configs, pruned = candidates(space, K, args.verbose)
    print(f"{pruned} cases are removed during pre-pruning", flush=True)
    # Always include the config production uses today, so a winner has to beat
    # it under the same conditions.
    current = {k: current_config[k] for k in space if k in current_config}
    if (
        run.record["error"] is None
        and len(current) == len(space)
        and current not in configs
    ):
        configs.append(current)
        print("Added the current config as a candidate", flush=True)
    print(f"Total number of cases to run: {len(configs)}", flush=True)

    log_dir = sweep_dir(args.op, arch, backend, dims)
    log = log_dir / f"M={args.M}.jsonl"
    if log.exists() and not args.overwrite:
        sys.exit(f"{log} exists; move it or pass --overwrite")
    log_dir.mkdir(parents=True, exist_ok=True)
    log.unlink(missing_ok=True)
    header = {
        "op": args.op,
        "M": args.M,
        "dims": dims,
        "arch": arch,
        "backend": backend,
        "lookup": lookup,
        "family": {k: v for k, v in family.items() if k != "observed"},
        "config": current_config,
        "baseline_us": baseline,
        "space": space,
    }
    append_jsonl(log, header)
    print(f"Results go to {log}", flush=True)

    blocks = block_keys(space)
    spec.update(block_keys=blocks, check=not args.no_check)
    sweep(args, spec, configs, blocks, case.kernels, log)

    results = [r for r in read_jsonl(log)[1:] if "us" in r]
    print(f"{len(results)} of {len(configs)} configs ran")
    if results:
        best = min(results, key=lambda r: r["us"])
        speedup = f" ({baseline / best['us']:.2f}x)" if baseline else ""
        print(f"Best: {best['us']:.3f} (us){speedup} {best['config']}")


if __name__ == "__main__":
    main()
