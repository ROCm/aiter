# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reproduce selected MiniMax-M3 shapes; see minimax_m3.md for provenance."""

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import statistics
import subprocess
from pathlib import Path

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.utils import minimax_m3 as m3


def cases():
    result = {}
    for tokens in (16, 128, 2048, 3072):
        result[f"router-{tokens}"] = (
            functools.partial(m3.router_inputs, tokens),
            functools.partial(gemm_a16w16, backend="gluon"),
        )
        result[f"swiglu-{tokens}"] = (
            functools.partial(m3.swiglu_inputs, tokens),
            m3.swiglu,
        )
        result[f"norm-rope-{tokens}"] = (
            functools.partial(m3.norm_rope_inputs, tokens),
            m3.norm_rope,
        )
    result["context-score"] = (
        functools.partial(m3.index_inputs, 32, 4, 194048, 4),
        m3.context_score,
    )
    result["local-topk"] = (
        functools.partial(m3.candidate_inputs, 128, 194048),
        m3.local_topk,
    )
    result["merge-topk"] = (
        functools.partial(m3.merge_inputs, 128, 194048),
        m3.merge_topk,
    )
    result["index-prefill"] = (
        functools.partial(m3.prefill_inputs, 2763, 160075),
        m3.index_prefill,
    )
    result["cache-gather"] = (
        functools.partial(m3.gather_inputs, 160075),
        m3.cache_gather,
    )
    return result


def source_metadata():
    modules = ["aiter.ops.triton.gemm.basic.gemm_a16w16"]
    import os

    if os.getenv("AITER_TEST_MINIMAX_M3_ATOM") == "1":
        modules += [
            "atom.model_ops." + name
            for name in (
                "minimax_m3.index_topk",
                "minimax_m3.indexer_context_parallel",
                "minimax_m3.indexer_candidate_exchange",
                "swiglu_oai",
                "triton_fused_qkv_norm_rope_cache",
                "base_attention",
            )
        ]
    result = {}
    for name in modules:
        filename = Path(inspect.getfile(importlib.import_module(name)))
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(filename.parent), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            revision = None
        result[name] = {
            "file": str(filename),
            "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
            "git_revision": revision,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=list(cases()), default=list(cases())
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(42)
    report = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": {
            "name": torch.cuda.get_device_properties(0).name,
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "compute_units": torch.cuda.get_device_properties(0).multi_processor_count,
        },
        "arch": triton.runtime.driver.active.get_current_target().arch,
        "sources": source_metadata(),
        "results": {},
        "method": "RotatingBufferBench defaults, 3 independent graph means unless --repeats overrides",
    }
    for name in args.cases:
        generate, fn = cases()[name]
        sample = generate()
        fields = list(sample)
        description = {
            key: (
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "stride": list(value.stride()),
                }
                if isinstance(value, torch.Tensor)
                else str(value)
            )
            for key, value in sample.items()
        }
        del sample
        bencher = RotatingBufferBench()
        bencher.build_pool(
            [lambda generate=generate: tuple(generate().values())], [fields]
        )
        times = [bencher.bench(fn) for _ in range(args.repeats)]
        entry = {
            "median_us": statistics.median(times) * 1000,
            "samples_us": [x * 1000 for x in times],
            "pool_len": len(bencher.rot_bufs),
            "inputs": description,
        }
        if args.profile_dir:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                fn(**bencher.rot_bufs[0])
                torch.cuda.synchronize()
            trace = args.profile_dir / (name + ".json")
            prof.export_chrome_trace(str(trace))
            entry["kernels"] = sorted(
                {
                    e["name"]
                    for e in json.loads(trace.read_text())["traceEvents"]
                    if e.get("cat") == "kernel"
                }
            )
        report["results"][name] = entry
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del bencher
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
