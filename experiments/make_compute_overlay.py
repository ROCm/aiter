#!/usr/bin/env python3
"""Overlay robust compute-route winners without touching repository configs."""

import argparse
import csv
import glob
import json


KEYS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")


def key(row):
    return tuple(row[name] for name in KEYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--results")
    parser.add_argument("--result-file", action="append", default=[])
    parser.add_argument(
        "--exclude-shape",
        action="append",
        default=[],
        help="M,N,K triple to leave unchanged; may be repeated",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    args = parser.parse_args()
    with open(args.base, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames
    winners = {}
    paths = list(args.result_file)
    if args.results:
        paths.extend(glob.glob(f"{args.results}/g*.json"))
    if not paths:
        parser.error("one of --results or --result-file is required")
    excluded = {tuple(map(int, value.split(","))) for value in args.exclude_shape}
    for path in paths:
        payload = json.load(open(path))
        result_rows = payload["results"] if isinstance(payload, dict) else payload
        for result in result_rows:
            if (result["M"], result["N"], result["K"]) in excluded:
                continue
            if result["speedup"] < args.min_speedup:
                continue
            winner = result["winner"]
            row_key = (
                "gfx1250",
                "256",
                str(result["M"]),
                str(result["N"]),
                str(result["K"]),
                "1x32",
            )
            winners[row_key] = winner
    replaced = []
    for row in rows:
        winner = winners.get(key(row))
        if winner is None:
            continue
        row["kernelName"] = winner["kernelName"]
        row["us"] = str(winner["median_us"])
        m, n, k = map(int, (row["M"], row["N"], row["K"]))
        row["tflops"] = str(2 * m * n * k / winner["median_us"] / 1e6)
        row["bw"] = "0.0"
        replaced.append((m, n, k))
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"replaced": len(replaced), "shapes": sorted(replaced)}))


if __name__ == "__main__":
    main()
