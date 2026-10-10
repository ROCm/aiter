"""Compile selected MXFP4 CSVs through the shared AOT worker pool."""

import argparse

from aiter.aot.flydsl.common import collect_aot_jobs, run_jobs_parallel
from aiter.aot.flydsl.mxfp4_moe import compile_one_config, parse_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--swiglu-limit", type=float, default=None)
    args = parser.parse_args()
    jobs = collect_aot_jobs(
        args.csv, lambda path: parse_csv(path, swiglu_limit=args.swiglu_limit)
    )
    if not jobs:
        parser.error("no MXFP4 jobs found in the selected CSVs")
    results = run_jobs_parallel(compile_one_config, jobs)
    failures = sum(result["compile_time"] is None for result in results)
    print(f"Compiled: {len(results) - failures} ok, {failures} failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
