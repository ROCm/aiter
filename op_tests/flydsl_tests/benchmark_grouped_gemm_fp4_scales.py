#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare scale32, real scale16, and NVFP4 using the same CSV tiles."""

import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path
from unittest.mock import patch


def graph_time_us(fn):
    import torch

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 16)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("fp4_scaling_results.json"))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    os.environ["AITER_MOE_EXPERT_BALANCE"] = "true"
    test_path = (
        Path(__file__).resolve().parents[1] / "test_flydsl_grouped_gemm_gfx1250.py"
    )
    spec = importlib.util.spec_from_file_location("grouped_test", test_path)
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    test._require_gfx1250()
    test.set_data_format("a4w4")

    import torch

    from aiter.test_common import run_perftest

    prepared = {}
    common_configs = None
    config_keys = (
        "tile_m",
        "tile_n",
        "tile_k",
        "m_warp",
        "n_warp",
        "num_buffers",
        "cluster_n",
        "waves_per_tensor_tdm",
        "next_stage_prefetch",
    )
    for name, block, fmt in (
        ("mxfp4_32", 32, 0),
        ("mxfp4_16", 16, 0),
        ("nvfp4", 16, 2),
    ):
        hooks = []

        def collect(fn, *, hooks=hooks, **_kwargs):
            hooks.append(fn)
            return fn(), 0.0

        with patch("aiter.test_common.run_perftest", collect):
            out, ref, _, _ = test._run_grouped_via_fused_moe(
                experts=96,
                tokens=args.tokens,
                topk=6,
                model_dim=7168,
                inter_dim=3072,
                data_format="a4w4",
                activation=test.ActivationType.Silu,
                use_bias=False,
                kernel_bench=True,
                scale_block_size=block,
                scale_format_a=fmt,
                scale_format_b=fmt,
                global_scale_a1=0.7 if fmt else 1.0,
                global_scale_a2=1.3 if fmt else 1.0,
                global_scale_w1=0.8 if fmt else 1.0,
                global_scale_w2=1.1 if fmt else 1.0,
            )
        assert len(hooks) == 2, "Expected GEMM1 and GEMM2 benchmark hooks"
        configs = [{key: fn.keywords[key] for key in config_keys} for fn in hooks]
        if common_configs is None:
            common_configs = configs
            print("Shared CSV launch settings: " + json.dumps(configs), flush=True)
        assert (
            configs == common_configs
        ), "Scale modes must use identical launch settings"
        diff = test._logits_diff(out, ref)
        assert diff < test.LOGITS_DIFF_TOL, (name, diff)
        prepared[name] = {
            "hooks": hooks,
            "diff": diff,
            "prof": [[], []],
            "graph": [[], []],
        }
        print(f"Prepared {name}: logits_diff={diff:.6g}", flush=True)
        del out, ref
        torch.cuda.empty_cache()

    for repeat in range(args.rounds):
        for name, data in prepared.items():
            for stage, fn in enumerate(data["hooks"]):
                _, us = run_perftest(fn, num_warmup=5, num_iters=101, testGraph=False)
                data["prof"][stage].append(us)
                data["graph"][stage].append(graph_time_us(fn))
            print(f"Round {repeat + 1}: {name}", flush=True)

    results = {
        name: {
            "tokens": args.tokens,
            "launch_configs": common_configs,
            "logits_diff": data["diff"],
            "profiler_median_us": [statistics.median(v) for v in data["prof"]],
            "graph_median_us": [statistics.median(v) for v in data["graph"]],
            "profiler_samples_us": data["prof"],
            "graph_samples_us": data["graph"],
        }
        for name, data in prepared.items()
    }
    serialized = json.dumps(results, indent=2) + "\n"
    args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
