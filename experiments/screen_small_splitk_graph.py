#!/usr/bin/env python3
"""Screen the native gfx1250 small-M split-K schedule with graph replay."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
    is_small_kernel,
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


def capture(run):
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def replay_us(graph, replays):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / replays


def check_changed_input(graph, data):
    saved_x = data["x"].clone()
    data["x"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    zero_max = float(data["out"].abs().max())
    data["x"].copy_(saved_x)
    graph.replay()
    torch.cuda.synchronize()
    restored_err = error_ratio(data["out"], data["ref"])
    return zero_max, restored_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = config_rows(args.config)
    families = [tuple(map(int, item.split(","))) for item in args.families.split(";")]
    ms = list(map(int, args.ms.split(",")))
    output = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for n, k in families:
        for m in ms:
            data = make_data(m, n, k)
            baseline_name = rows[(m, n, k)]["kernelName"]
            names = [baseline_name]
            names.extend(
                instance.name
                for instance in kernels_list.values()
                if is_small_kernel(instance) and kernel_fits_shape(instance, m, n, k)
            )
            names = list(dict.fromkeys(names))
            candidates = {}
            failures = {}
            for name in names:
                try:
                    run = runner(data, name)
                    graph = capture(run)
                    eager_err = error_ratio(run(), data["ref"])
                    zero_max, restored_err = check_changed_input(graph, data)
                    if eager_err > 0.05 or restored_err > 0.05 or zero_max != 0.0:
                        raise RuntimeError(
                            f"correctness eager={eager_err} restored={restored_err} "
                            f"changed_zero_max={zero_max}"
                        )
                    candidates[name] = {
                        "graph": graph,
                        "eager_err_ratio": eager_err,
                        "changed_zero_max": zero_max,
                        "restored_err_ratio": restored_err,
                        "samples_us": [],
                    }
                except Exception as exc:
                    failures[name] = f"{type(exc).__name__}: {exc}"

            finalists = list(candidates)
            if baseline_name not in candidates:
                raise RuntimeError(
                    f"baseline failed for {(m, n, k)}: {failures.get(baseline_name)}"
                )
            for round_index in range(args.rounds):
                order = (
                    finalists[round_index % len(finalists) :]
                    + finalists[: round_index % len(finalists)]
                )
                if round_index % 2:
                    order.reverse()
                for name in order:
                    candidates[name]["samples_us"].append(
                        replay_us(candidates[name]["graph"], args.replays)
                    )

            measured = []
            for name, candidate in candidates.items():
                values = sorted(candidate.pop("samples_us"))
                measured.append(
                    {
                        "kernelName": name,
                        "median_us": values[len(values) // 2],
                        "min_us": values[0],
                        "p90_us": values[math.ceil(0.9 * len(values)) - 1],
                        "samples_us": values,
                        **{
                            key: value
                            for key, value in candidate.items()
                            if key != "graph"
                        },
                    }
                )
            measured.sort(key=lambda row: row["median_us"])
            baseline = next(
                row for row in measured if row["kernelName"] == baseline_name
            )
            winner = measured[0]
            result = {
                "M": m,
                "N": n,
                "K": k,
                "baseline": baseline,
                "winner": winner,
                "speedup": baseline["median_us"] / winner["median_us"],
                "candidate_count": len(names),
                "failures": failures,
                "candidates": measured,
            }
            output.append(result)
            out.write_text(json.dumps(output, indent=2) + "\n")
            print(
                json.dumps(
                    {key: result[key] for key in ("M", "N", "K", "speedup", "winner")}
                ),
                flush=True,
            )
            del data, candidates
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
