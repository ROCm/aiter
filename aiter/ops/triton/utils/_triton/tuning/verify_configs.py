"""Profile one GEMM with the config the library resolves, as production runs it.

    python3 verify_configs.py <op> <M> <DIM>=<int>... [--gpu G] [--backend B]

Prints the config family and backend the op looked up, the file the config
came from (the tuned file for the shape, or DEFAULT.json when none matches),
the config itself, and the kernels' median runtime. Each run is a fresh
process, so it sees tuned files installed a moment ago.
"""

import argparse
import os
import sys

from _utils import describe_lookup, get_case, parse_dims, resolve_family, run_worker


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
        help="backend to run, for ops that have both (default: the wrapper's pick)",
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds for the profiled process"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dims = parse_dims(args.dims)
    case = get_case(args.op, dims, args.backend)
    spec = {
        "op": args.op,
        "M": args.M,
        "dims": dims,
        "backend": args.backend,
        "candidates": None,
    }
    run = run_worker(spec, args.gpu, args.timeout, case.kernels)
    lookup, family = resolve_family(run, args.op)

    tuned_file = family["tuned_file"]
    source = tuned_file or os.path.join(family["config_dir"], "DEFAULT.json")
    note = "tuned for this shape" if tuned_file else "no tuned file for this shape"
    print(f"{args.op} M={args.M} on {family['arch']}: {describe_lookup(lookup)}")
    print(f"File:    {os.path.relpath(source, family['configs_root'])} ({note})")
    print(f"Config:  {run.record.get('config')}")
    if run.record["error"] is not None:
        sys.exit(f"Failed:  {run.record['error']}")
    if run.segments and len(run.segments) > 1:
        kernels, runtime = run.segments[1]
        print(f"Kernels: {', '.join(kernels) or '(none matched)'}")
        if runtime is not None:
            print(f"Runtime: {runtime:.3f} (us)")


if __name__ == "__main__":
    main()
