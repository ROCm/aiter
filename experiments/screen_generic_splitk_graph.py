#!/usr/bin/env python3
"""Screen generic small-tile fused split-K with CUDA Graph replay."""

import argparse
import csv
import json
import statistics
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


def candidate_names(k):
    names = []
    for tile_k in (128, 256, 512):
        for num_buffers in (2, 3, 4):
            for split_k in (1, 2, 4, 8):
                if k % (split_k * tile_k):
                    continue
                if (k // split_k) // tile_k < num_buffers:
                    continue
                base = (
                    "flydsl_mxfp8_32_bpreshuffle_wmma_"
                    f"t16x32x{tile_k}_mw1_nw2_nb{num_buffers}_sk{split_k}_"
                    "cm1_cn1"
                )
                if split_k == 1:
                    names.append(base)
                else:
                    names.append(base + "_fsk")
    return names


def graph_measure(run, data, replays, repeats):
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000 / replays)
    graph.replay()
    torch.cuda.synchronize()
    err = error_ratio(data["out"], data["ref"])
    # Flip the FP8 sign bit in place: torch has no neg_cuda implementation for
    # float8, while graph replay must keep the captured input addresses stable.
    data["x"].view(torch.uint8).bitwise_xor_(0x80)
    graph.replay()
    torch.cuda.synchronize()
    changed_err = error_ratio(data["out"], -data["ref"])
    data["x"].view(torch.uint8).bitwise_xor_(0x80)
    torch.cuda.synchronize()
    return statistics.median(values), err, changed_err, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = config_rows(args.config)
    families = [tuple(map(int, item.split(","))) for item in args.families.split(";")]
    ms = list(map(int, args.ms.split(",")))
    results = []
    for n, k in families:
        for m in ms:
            data = make_data(m, n, k)
            baseline_name = rows[(m, n, k)]["kernelName"]
            names = list(dict.fromkeys((baseline_name, *candidate_names(k))))
            candidates = []
            for name in names:
                row = {"kernelName": name}
                try:
                    run = runner(data, name)
                    us, err, changed_err, samples = graph_measure(
                        run, data, args.replays, args.repeats
                    )
                    row.update(
                        median_us=us,
                        samples_us=samples,
                        err_ratio=err,
                        changed_err_ratio=changed_err,
                        status="ok" if max(err, changed_err) <= 0.05 else "mismatch",
                    )
                except Exception as exc:
                    row.update(status=f"error:{type(exc).__name__}:{exc}")
                candidates.append(row)
            valid = [row for row in candidates if row["status"] == "ok"]
            valid.sort(key=lambda row: row["median_us"])
            baseline = next(row for row in valid if row["kernelName"] == baseline_name)
            winner = valid[0]
            result = {
                "M": m,
                "N": n,
                "K": k,
                "baseline": baseline,
                "winner": winner,
                "speedup": baseline["median_us"] / winner["median_us"],
                "candidates": candidates,
            }
            results.append(result)
            print(
                json.dumps(
                    {
                        "M": m,
                        "N": n,
                        "K": k,
                        "speedup": result["speedup"],
                        "winner": winner,
                    }
                ),
                flush=True,
            )
            del data
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
