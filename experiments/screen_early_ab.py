#!/usr/bin/env python3
"""Interleaved A/B screen for compute-kernel A/B TDM early merge."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from real_backend_screen import error_ratio, make_data, runner


def config_rows(path):
    with open(path, newline="") as handle:
        return {
            (int(row["M"]), int(row["N"]), int(row["K"])): row
            for row in csv.DictReader(handle)
            if row["gfx"] == "gfx1250" and row["w_scale_block"] == "1x32"
        }


def sample(run, inner=50):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(inner):
        run()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = config_rows(args.config)
    families = [tuple(map(int, item.split(","))) for item in args.families.split(";")]
    ms = list(map(int, args.ms.split(",")))
    output = []
    for n, k in families:
        for m in ms:
            baseline_name = rows[(m, n, k)]["kernelName"]
            if "compute_wmma" not in baseline_name:
                continue
            data = make_data(m, n, k)
            names = [baseline_name, f"{baseline_name}_eab"]
            runs = {name: runner(data, name) for name in names}
            errors = {name: error_ratio(run(), data["ref"]) for name, run in runs.items()}
            if max(errors.values()) > 0.05:
                raise RuntimeError(f"correctness failure for {(m, n, k)}: {errors}")
            for run in runs.values():
                for _ in range(10):
                    run()
            torch.cuda.synchronize()
            measured = {name: [] for name in names}
            for round_index in range(21):
                order = names if round_index % 2 == 0 else names[::-1]
                for name in order:
                    measured[name].append(sample(runs[name]))
            candidates = []
            for name, values in measured.items():
                values.sort()
                candidates.append(
                    {
                        "kernelName": name,
                        "median_us": values[len(values) // 2],
                        "min_us": values[0],
                        "p90_us": values[math.ceil(0.9 * len(values)) - 1],
                        "samples_us": values,
                        "err_ratio": errors[name],
                    }
                )
            candidates.sort(key=lambda row: row["median_us"])
            baseline = next(row for row in candidates if row["kernelName"] == baseline_name)
            early = next(row for row in candidates if row["kernelName"].endswith("_eab"))
            result = {
                "M": m,
                "N": n,
                "K": k,
                "baseline": baseline,
                "early_ab": early,
                "speedup": baseline["median_us"] / early["median_us"],
                "winner": candidates[0],
            }
            output.append(result)
            print(
                json.dumps(
                    {key: result[key] for key in ("M", "N", "K", "speedup", "winner")}
                ),
                flush=True,
            )
            del data
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
