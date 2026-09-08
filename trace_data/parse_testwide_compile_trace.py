#!/usr/bin/env python3
"""Summarize numbered GPU profiler iterations from TestWideEpMoe traces."""

import argparse
import glob
import json
import os
import re


def classify(name):
    if name.startswith("EpDispatch"):
        return "dispatch"
    if "moe_sorting" in name or "mxfp4_moe_sort" in name:
        return "sorting"
    if name.startswith("mfma_moe1") or "moe1_" in name:
        return "gemm1"
    if name.startswith("gemm2_") or "moe2_" in name:
        return "gemm2"
    if name.startswith("EpCombine"):
        return "combine"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()
    result = []
    pattern = re.compile(r"/tpr([0-9]+)/rank([0-9]+)_bs([0-9]+)[.]json")
    for path in sorted(glob.glob(os.path.join(args.root, "tpr*", "rank*_bs*.json"))):
        match = pattern.search(path)
        if not match:
            continue
        tpr, rank, bs = map(int, match.groups())
        with open(path) as handle:
            events = json.load(handle)["traceEvents"]
        kernels = [
            event
            for event in events
            if event.get("ph") == "X" and event.get("cat") == "kernel"
        ]
        annotations = {}
        iter_pattern = re.compile(r"_bs" + str(bs) + r"_iter([0-9]+)")
        for event in events:
            iteration = iter_pattern.search(event.get("name", ""))
            if (
                iteration
                and event.get("ph") == "X"
                and event.get("cat") == "gpu_user_annotation"
            ):
                annotations[int(iteration.group(1))] = event
        selected = sorted(annotations)[-args.tail :]
        samples = []
        for iteration in selected:
            annotation = annotations[iteration]
            begin = float(annotation["ts"])
            end = begin + float(annotation["dur"])
            values = {
                key: 0.0
                for key in ("dispatch", "sorting", "gemm1", "gemm2", "combine")
            }
            names = {"gemm1": [], "gemm2": []}
            for event in kernels:
                timestamp = float(event.get("ts", 0))
                if not begin <= timestamp < end:
                    continue
                kind = classify(event.get("name", ""))
                if kind is None:
                    continue
                values[kind] += float(event.get("dur", 0))
                if kind in names:
                    names[kind].append(event.get("name", ""))
            values["pipeline"] = float(annotation["dur"])
            values["complete"] = all(
                values[key] > 0
                for key in ("dispatch", "sorting", "gemm1", "gemm2", "combine")
            )
            values["iteration"] = iteration
            values["gemm1_names"] = names["gemm1"]
            values["gemm2_names"] = names["gemm2"]
            samples.append(values)
        result.append({"tpr": tpr, "rank": rank, "samples": samples})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
