#!/usr/bin/env python3
"""Strict, CPU-only analysis of numbered Stage2 CUDA/HIP Graph traces.

Use actual GEMM2-to-combine device spans for A/B comparisons. The legacy
GEMM2-min + combine-mean statistic is reported separately, never as latency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics


ITERATION = re.compile(r"^stage2_graph_(candidate|mori)_tpr(\d+)_iter(\d+)$")


def classify(event):
    name = event.get("name", "")
    if event.get("cat") == "gpu_memcpy":
        return "d2d" if "DtoD" in name or "Device -> Device" in name else "memcpy_other"
    if "copyBuffer" in name:
        return "d2d"
    if "megamoe_tile_ep16_stage1" in name:
        return "fused_stage1"
    if "megamoe_tile_ep16_stage2" in name:
        return "fused_stage2"
    # 两 kernel 路径:kernel1(GEMM2+push) 与 kernel2(node 归约+rail+combine)
    # 合起来才等价于原来的 fused_stage2,归到同一组以保持对比口径不变。
    if name.startswith("megamoe_stage2_") or name.startswith("megamoe_k2_"):
        return "fused_stage2"
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
    return "other"


def parse_trace(trace, *, path, tokens, iterations=40, tail=20):
    if not 1 <= tail <= iterations:
        raise ValueError("require 1 <= tail <= iterations")
    events = trace["traceEvents"]
    annotations = {}
    for event in events:
        match = ITERATION.fullmatch(event.get("name", ""))
        if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X" or not match:
            continue
        kind, tpr, iteration = match.groups()
        if kind != path or int(tpr) != tokens:
            continue
        index = int(iteration)
        if index in annotations:
            raise ValueError(f"duplicate GPU annotation for iteration {index}")
        annotations[index] = event
    # Select the requested numerical range, not 'last 20 visible'. Missing
    # graph children/annotations must not silently change the sample window.
    selected = list(range(iterations - tail, iterations))
    missing = sorted(set(selected) - annotations.keys())
    if missing:
        raise ValueError(f"missing requested GPU iterations: {missing}")
    activities = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    required = ("fused_stage1", "fused_stage2") if path == "candidate" else ("dispatch", "sorting", "gemm1", "gemm2", "combine")
    samples = []
    for iteration in selected:
        annotation = annotations[iteration]
        begin = float(annotation["ts"])
        end = begin + float(annotation["dur"])
        active = sorted((e for e in activities if begin <= float(e["ts"]) < end), key=lambda e: float(e["ts"]))
        groups = {}
        for event in active:
            groups.setdefault(classify(event), []).append(event)
        absent = [name for name in required if not groups.get(name)]
        if absent:
            raise ValueError(f"iteration {iteration}: incomplete graph children, missing {absent}")
        import os as _os
        _two_kernel = _os.environ.get("MEGAMOE_TWO_KERNEL") == "1"
        _want = {"fused_stage1": 1, "fused_stage2": 2 if _two_kernel else 1}
        _total = 3 if _two_kernel else 2
        if path == "candidate" and (
            any(len(groups[name]) != _want[name] for name in required)
            or sum(e.get("cat") == "kernel" for e in active) != _total
        ):
            got = {name: len(groups.get(name, ())) for name in required}
            got["kernels"] = sum(e.get("cat") == "kernel" for e in active)
            raise ValueError(
                f"iteration {iteration}: candidate kernel shape {got}, "
                f"expected {_want} with {_total} kernels")
        streams = {e.get("args", {}).get("stream") for e in active if e.get("cat") == "kernel"}
        if len(streams - {None}) > 1:
            raise ValueError(f"iteration {iteration}: expected one stream, found {streams}")
        # User's comparison boundary is GEMM2 start -> combine end from one
        # full-forward replay. This includes the intervening copy/gaps and
        # excludes Stage1. Never reconstruct it from independently minimized
        # component durations or accept a trace with reversed stage ordering.
        first = groups["fused_stage2"] if path == "candidate" else groups["gemm2"]
        last = groups["fused_stage2"] if path == "candidate" else groups["combine"]
        pair_begin = min(float(e["ts"]) for e in first)
        pair_end = max(float(e["ts"]) + float(e["dur"]) for e in last)
        if path == "mori" and min(float(e["ts"]) for e in last) < max(
            float(e["ts"]) + float(e["dur"]) for e in first
        ):
            raise ValueError(f"iteration {iteration}: combine precedes GEMM2 completion")
        pair_activities = [e for e in active if pair_begin <= float(e["ts"]) < pair_end]
        if any(classify(e) in ("dispatch", "sorting", "gemm1", "fused_stage1") for e in pair_activities):
            raise ValueError(f"iteration {iteration}: invalid Stage2 ordering")
        durations = {name + "_us": sum(float(e["dur"]) for e in group) for name, group in groups.items()}
        for name in ("fused_stage1", "fused_stage2", "dispatch", "sorting", "gemm1", "gemm2", "combine", "d2d", "other", "memcpy_other"):
            durations.setdefault(name + "_us", 0.0)
        samples.append({
            "iteration": iteration,
            **durations,
            "stage2_span_us": pair_end - pair_begin,
            "stage2_activity_sum_us": sum(float(e["dur"]) for e in pair_activities),
            "pipeline_span_us": max(float(e["ts"]) + float(e["dur"]) for e in active) - float(active[0]["ts"]),
            "pipeline_activity_sum_us": sum(float(e["dur"]) for e in active),
            "annotation_us": float(annotation["dur"]),
            "d2d_count": len(groups.get("d2d", [])),
            "stage2_d2d_count": sum(classify(e) == "d2d" for e in pair_activities),
            "kernel_names": {name: sorted({e["name"] for e in group}) for name, group in groups.items()},
        })
    return samples


def aggregate(rank_samples, *, expected_world=16):
    ranks = [row["rank"] for row in rank_samples]
    if sorted(ranks) != list(range(expected_world)):
        raise ValueError(f"expected each rank exactly once, got {sorted(ranks)}")
    windows = [tuple(s["iteration"] for s in row["samples"]) for row in rank_samples]
    if not windows[0] or any(w != windows[0] for w in windows):
        raise ValueError("rank sample windows do not match")
    metrics = sorted(k for k in rank_samples[0]["samples"][0] if k.endswith("_us"))
    result = {}
    for metric in metrics:
        rank_means, rank_mins, rank_cvs = [], [], []
        for row in rank_samples:
            values = [s.get(metric, 0.0) for s in row["samples"]]
            mean = statistics.mean(values)
            rank_means.append(mean)
            rank_mins.append(min(values))
            rank_cvs.append(statistics.pstdev(values) / mean if mean else 0.0)
        result[metric] = {
            "rank_mean": statistics.mean(rank_means),
            "rank_mean_of_min": statistics.mean(rank_mins),
            # EPLB comparison can use the shortest observed complete replay
            # span, avoiding rank-arrival waits in the mean. In particular,
            # Stage2's minimum remains one measured GEMM2-through-combine
            # interval, not a sum of minima taken from different replays.
            "pooled_min": min(rank_mins),
            "rank_mean_cv": statistics.mean(rank_cvs),
            "worst_rank_mean": max(rank_means),
        }
    return {
        "world_size": expected_world,
        "samples": sum(len(r["samples"]) for r in rank_samples),
        "iterations": list(windows[0]),
        # Preserve the actual rank/replay and its component durations. These
        # components belong to the minimum complete interval, not independently
        # selected GEMM2/combine minima from different cards or iterations.
        "stage2_min_sample": min(
            ({"rank": row["rank"], **sample} for row in rank_samples for sample in row["samples"]),
            key=lambda sample: (sample["stage2_span_us"], sample["rank"], sample["iteration"]),
        ),
        "metrics": result,
        "legacy_gemm2_min_plus_combine_mean_us": result["gemm2_us"]["rank_mean_of_min"] + result["combine_us"]["rank_mean"],
        "d2d_count_per_iteration": sorted({s["d2d_count"] for r in rank_samples for s in r["samples"]}),
        "stage2_d2d_count_per_iteration": sorted({s["stage2_d2d_count"] for r in rank_samples for s in r["samples"]}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--path", choices=("candidate", "mori"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--tail-iters", type=int, default=20)
    args = parser.parse_args()
    rows = []
    for file in args.traces:
        match = re.fullmatch(r"rank(\d+)[.]json", file.name)
        if match is None:
            parser.error(f"expected rankN.json, got {file}")
        samples = parse_trace(json.loads(file.read_text()), path=args.path, tokens=args.tokens, iterations=args.iters, tail=args.tail_iters)
        rows.append({"rank": int(match.group(1)), "samples": samples})
    print(json.dumps(aggregate(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
