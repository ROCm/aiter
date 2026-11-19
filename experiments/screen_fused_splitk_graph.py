#!/usr/bin/env python3
"""Tune cluster-local MXFP8 split-K with CUDA Graph replay timings."""

import argparse
import csv
import json
from pathlib import Path

import torch

from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
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


def graph_measure(run, ref, replays):
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    us = begin.elapsed_time(end) * 1000 / replays
    err = error_ratio(run(), ref)
    return us, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--replays", type=int, default=200)
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
                if instance.fused_splitk
                and instance.tile_m == 128
                and instance.tile_n in (128, 256)
                and instance.cluster_m == instance.cluster_n == 1
                and instance.num_buffers in (3, 4)
                and kernel_fits_shape(instance, m, n, k)
            )
            names = list(dict.fromkeys(names))
            candidates = []
            for name in names:
                try:
                    run = runner(data, name)
                    us, err = graph_measure(run, data["ref"], args.replays)
                    if err <= 0.05:
                        candidates.append(
                            {"kernelName": name, "median_us": us, "err_ratio": err}
                        )
                except Exception:
                    continue
            candidates.sort(key=lambda row: row["median_us"])
            baseline = next(
                row for row in candidates if row["kernelName"] == baseline_name
            )
            winner = candidates[0]
            result = {
                "M": m,
                "N": n,
                "K": k,
                "baseline": baseline,
                "winner": winner,
                "speedup": baseline["median_us"] / winner["median_us"],
                "candidate_count": len(names),
                "candidates": candidates,
            }
            output.append(result)
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in ("M", "N", "K", "speedup", "winner")
                    }
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
