#!/usr/bin/env python3
"""Compare current production rows with B=1 compute-kernel candidates."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
    is_compute_kernel,
    kernel_fits_shape,
    kernels_list,
)
from real_backend_screen import error_ratio, make_data, runner


def config_rows(path):
    with open(path, newline="") as handle:
        return {
            (int(row["M"]), int(row["N"]), int(row["K"])): row
            for row in csv.DictReader(handle)
            if row["gfx"] == "gfx1250" and row["w_scale_block"] == "1x32"
        }


def samples(run, warmup, reps, inner):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    values = []
    for _ in range(reps):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(inner):
            run()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000 / inner)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--exclude-fused-splitk", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = config_rows(args.config)
    families = [tuple(map(int, item.split(","))) for item in args.families.split(";")]
    ms = list(map(int, args.ms.split(",")))
    output = []
    for n, k in families:
        for m in ms:
            data = make_data(m, n, k)
            baseline_name = rows[(m, n, k)]["kernelName"]
            names = [baseline_name]
            names.extend(
                instance.name.replace("mxfp8_128", "mxfp8_32")
                for instance in kernels_list.values()
                if is_compute_kernel(instance)
                and kernel_fits_shape(instance, m, n, k)
                and (not args.exclude_fused_splitk or not instance.fused_splitk)
            )
            names = list(dict.fromkeys(names))
            coarse = []
            for name in names:
                try:
                    run = runner(data, name)
                    err = error_ratio(run(), data["ref"])
                    timing = samples(run, 3, 3, 20)
                except Exception:
                    continue
                if err <= 0.05:
                    coarse.append((sorted(timing)[1], name, err))
            coarse.sort()
            finalists = list(dict.fromkeys([baseline_name, *[x[1] for x in coarse[:8]]]))
            measured = {name: [] for name in finalists}
            runs = {name: runner(data, name) for name in finalists}
            for run in runs.values():
                for _ in range(10):
                    run()
            torch.cuda.synchronize()
            for round_index in range(15):
                order = finalists[round_index % len(finalists) :] + finalists[
                    : round_index % len(finalists)
                ]
                if round_index % 2:
                    order.reverse()
                for name in order:
                    measured[name].extend(samples(runs[name], 0, 1, 50))
            stable = []
            for name, values in measured.items():
                values.sort()
                stable.append(
                    {
                        "kernelName": name,
                        "median_us": values[len(values) // 2],
                        "min_us": values[0],
                        "p90_us": values[math.ceil(0.9 * len(values)) - 1],
                        "samples_us": values,
                    }
                )
            stable.sort(key=lambda row: row["median_us"])
            baseline = next(row for row in stable if row["kernelName"] == baseline_name)
            winner = stable[0]
            result = {
                "M": m,
                "N": n,
                "K": k,
                "baseline": baseline,
                "winner": winner,
                "speedup": baseline["median_us"] / winner["median_us"],
                "winner_err_ratio": next(x[2] for x in coarse if x[1] == winner["kernelName"])
                if winner["kernelName"] != baseline_name
                else error_ratio(runs[baseline_name](), data["ref"]),
                "candidate_count": len(names),
                "finalists": stable,
            }
            output.append(result)
            print(json.dumps({key: result[key] for key in ("M", "N", "K", "speedup", "winner", "candidate_count")}), flush=True)
            del data
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
