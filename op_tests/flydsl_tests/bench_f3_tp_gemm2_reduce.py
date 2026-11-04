#!/usr/bin/env python3
"""Reproducible A/B benchmark for the F3 TP GEMM2 route reduction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
DEFAULT_CONFIG = (
    ROOT
    / "aiter/configs/model_configs/minimax_m3_gfx1250_tuned_grouped_fmoe.csv"
)
DEFAULT_TOKENS = (4, 8, 16, 32, 48, 64, 2517, 4536)


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location("_f3_grouped_fixture", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fixture from {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "median_us": statistics.median(values),
        "p90_us": _percentile(values, 0.90),
        "slowest_us": max(values),
        "fastest_us": min(values),
    }


def _make_fixed_call(fixture, tokens: int):
    public = fixture.fused_moe
    captured: dict[str, object] = {}

    def capture(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return public(*args, **kwargs)

    original_ref = fixture._torch_moe_ref
    fixture._torch_moe_ref = lambda hidden, *_args, **_kwargs: torch.zeros_like(hidden)
    fixture.fused_moe = capture
    os.environ["AITER_FLYDSL_TP_GEMM2_REDUCE"] = "0"
    try:
        baseline, _, _, _ = fixture._run_grouped_via_fused_moe(
            experts=129,
            tokens=tokens,
            topk=5,
            model_dim=6144,
            inter_dim=768,
            data_format="a4w4",
            activation=fixture.ActivationType.Swiglu,
            use_bias=False,
            seed=393,
        )
    finally:
        fixture.fused_moe = public
        fixture._torch_moe_ref = original_ref
    return public, captured["args"], captured["kwargs"], baseline


def _call_with_arm(public, args, kwargs, fused: bool):
    os.environ["AITER_FLYDSL_TP_GEMM2_REDUCE"] = "1" if fused else "0"
    return public(*args, **kwargs)


def _make_graph(public, args, kwargs, fused: bool):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(10):
            output = _call_with_arm(public, args, kwargs, fused)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    # Warm and capture on the same stream: the fused counter arena performs its
    # one-time zeroing on first use of a stream, which must not enter the graph.
    with torch.cuda.graph(graph, stream=stream):
        output = _call_with_arm(public, args, kwargs, fused)
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _time_replays(graph, warmup: int, replays: int) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _profile_graph(graph) -> list[dict[str, object]]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        graph.replay()
        torch.cuda.synchronize()
    kernels = []
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels.append(
                {
                    "name": event.name,
                    "duration_us": float(event.device_time_total),
                }
            )
    return kernels


def _find_kernel(kernels: list[dict[str, object]], needle: str) -> float | None:
    matches = [
        float(kernel["duration_us"])
        for kernel in kernels
        if needle.lower() in str(kernel["name"]).lower()
    ]
    return sum(matches) if matches else None


def _run_shape(fixture, tokens: int, warmup: int, replays: int):
    public, call_args, call_kwargs, baseline_eager = _make_fixed_call(fixture, tokens)
    fused_eager = _call_with_arm(public, call_args, call_kwargs, True)
    repeats = [
        _call_with_arm(public, call_args, call_kwargs, True).clone() for _ in range(3)
    ]
    torch.cuda.synchronize()
    correctness = {
        "baseline_fused_bitwise": bool(torch.equal(baseline_eager, fused_eager)),
        "repeat_bitwise": all(torch.equal(fused_eager, value) for value in repeats),
        "max_abs": float(
            (baseline_eager.float() - fused_eager.float()).abs().max().item()
        ),
    }
    if not correctness["baseline_fused_bitwise"] or not correctness["repeat_bitwise"]:
        raise AssertionError(f"tokens={tokens}: fused output is not deterministic/bitwise")

    baseline_graph, baseline_output = _make_graph(
        public, call_args, call_kwargs, False
    )
    fused_graph, fused_output = _make_graph(public, call_args, call_kwargs, True)
    if not torch.equal(baseline_output, fused_output):
        raise AssertionError(f"tokens={tokens}: graph outputs differ")

    samples = {"baseline": [], "fused": []}
    rounds = []
    # B-A-A-B controls both thermal drift and launch-order bias.
    for arm in ("fused", "baseline", "baseline", "fused"):
        graph = fused_graph if arm == "fused" else baseline_graph
        values = _time_replays(graph, warmup, replays)
        samples[arm].extend(values)
        rounds.append({"arm": arm, **_summary(values)})

    baseline_kernels = _profile_graph(baseline_graph)
    fused_kernels = _profile_graph(fused_graph)
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    token_counters = [
        buffer
        for key, buffer in grouped._TP_GEMM2_REDUCE_COUNTER_CACHE.items()
        if key[0] == tokens
    ]
    counter_nonzero = sum(
        int(torch.count_nonzero(buffer).item()) for buffer in token_counters
    )
    correctness["counter_buffers"] = len(token_counters)
    correctness["counter_nonzero"] = counter_nonzero
    if counter_nonzero:
        raise AssertionError(f"tokens={tokens}: {counter_nonzero} counters not reset")
    baseline = _summary(samples["baseline"])
    fused = _summary(samples["fused"])
    net_us = float(baseline["median_us"]) - float(fused["median_us"])
    result = {
        "tokens": tokens,
        "correctness": correctness,
        "rounds": rounds,
        "baseline": baseline,
        "fused": fused,
        "net_median_us": net_us,
        "speedup_percent": net_us / float(baseline["median_us"]) * 100.0,
        "layer57_baseline_us": float(baseline["median_us"]) * 57,
        "layer57_fused_us": float(fused["median_us"]) * 57,
        "layer57_net_us": net_us * 57,
        "profile": {
            "baseline_kernel_count": len(baseline_kernels),
            "fused_kernel_count": len(fused_kernels),
            "baseline_gemm2_us": _find_kernel(baseline_kernels, "_K768_"),
            "baseline_gather_us": _find_kernel(
                baseline_kernels, "moe_gather_reduce"
            ),
            "fused_gemm2_us": _find_kernel(fused_kernels, "_K768_"),
            "baseline_kernels": baseline_kernels,
            "fused_kernels": fused_kernels,
        },
    }
    del baseline_graph, fused_graph, baseline_output, fused_output
    del baseline_eager, fused_eager, repeats, call_args, call_kwargs
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
    os.environ.setdefault("AITER_CONFIG_GROUPED_FMOE", str(DEFAULT_CONFIG))
    fixture = _load_fixture_module()
    payload = {
        "metadata": {
            "commit": os.environ.get("F3_COMMIT", "unknown"),
            "device": torch.cuda.get_device_name(),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "config": os.environ["AITER_CONFIG_GROUPED_FMOE"],
            "shape": {
                "experts": 129,
                "topk": 5,
                "model_dim": 6144,
                "inter_dim": 768,
                "path": "MXFP4 activations x MXFP4 weights",
            },
            "warmup_per_round": args.warmup,
            "replays_per_round": args.replays,
            "order": ["fused", "baseline", "baseline", "fused"],
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        result = _run_shape(fixture, tokens, args.warmup, args.replays)
        payload["results"].append(result)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
