# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reproduce selected MiniMax-M3 shapes; see minimax_m3.md for provenance."""

import functools
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import triton

from aiter.ops.triton.attention.minimax_m3 import (
    minimax_m3_packed_topk,
    minimax_m3_prefill_score,
)
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
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
    result["prefill-score"] = (
        functools.partial(m3.prefill_score_inputs, 2763, 160075),
        minimax_m3_prefill_score,
    )
    result["packed-topk"] = (
        functools.partial(m3.packed_topk_inputs, 2763, 160075),
        minimax_m3_packed_topk,
    )
    result["cache-gather"] = (
        functools.partial(m3.gather_inputs, 160075),
        m3.cache_gather,
    )
    return result


def source_metadata():
    modules = [
        "aiter.ops.triton.gemm.basic.gemm_a16w16",
        "aiter.ops.triton.attention.minimax_m3",
    ]
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


def atom_dependency(sources):
    if os.getenv("AITER_TEST_MINIMAX_M3_ATOM") != "1":
        return None
    dependency = json.loads(
        Path(__file__).with_name("minimax_m3_atom.json").read_text()
    )
    for name, expected in dependency["modules"].items():
        if sources[name]["sha256"] != expected:
            raise RuntimeError(
                f"ATOM source mismatch: {name}; use revision {dependency['revision']}"
            )
        actual_revision = sources[name]["git_revision"]
        if actual_revision is not None and actual_revision != dependency["revision"]:
            raise RuntimeError(
                f"ATOM revision mismatch: {actual_revision}; expected {dependency['revision']}"
            )
    return {
        **dependency,
        "verification": "Selected module SHA256 values; Git HEAD also checked when available",
    }


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
        }
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: describe(item) for key, item in vars(value).items()}
    return value


def main():
    parser = get_parser(kernel_name="MiniMax-M3 Triton/Gluon")
    override_parser_action(parser, "--metric", choices=["time"], default="time")
    override_parser_action(
        parser,
        "--backend",
        choices=["gluon"],
        default="gluon",
        help="Router backend; all non-router cases use Triton",
    )
    parser.add_argument(
        "--cases", nargs="+", choices=list(cases()), default=list(cases())
    )
    parser.add_argument(
        "--output", type=Path, help="Optional source/profiler metadata JSON"
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("-o", action="store_true", help="Save standard perf_report CSV")
    args = parser.parse_args()
    if args.model:
        parser.error("MiniMax-M3 uses fixed trace shapes; --model is not supported")
    if (
        any(not name.startswith("router-") for name in args.cases)
        and os.getenv("AITER_TEST_MINIMAX_M3_ATOM") != "1"
    ):
        parser.error(
            "Install the pinned ATOM checkout and set AITER_TEST_MINIMAX_M3_ATOM=1"
        )
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
        "method": "One RotatingBufferBench graph mean per case; outer runner owns repetition/aggregation",
        "benchmark_config": {
            **get_rot_bufs_kwargs(args),
            **get_do_bench_graph_kwargs(args),
        },
        "schema_version": 2,
    }
    report["atom_dependency"] = atom_dependency(report["sources"])
    config = triton.testing.Benchmark(
        x_names=["case"],
        x_vals=args.cases,
        line_arg="provider",
        line_vals=["native"],
        line_names=["Time_ms"],
        styles=[("blue", "-")],
        ylabel="ms",
        plot_name="bench_minimax_m3",
        args={},
    )

    @triton.testing.perf_report([config])
    def run(case, provider):
        name = case
        generate, fn = cases()[name]
        sample = generate()
        fields = list(sample)
        description = {key: describe(value) for key, value in sample.items()}
        del sample
        bencher = RotatingBufferBench.from_dict(
            {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
        )
        bencher.build_pool(
            [lambda generate=generate: tuple(generate().values())], [fields]
        )
        ms = bencher.bench(fn)
        entry = {
            "time_ms": ms,
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
            launches = [
                e["name"]
                for e in json.loads(trace.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ]
            entry["kernels"] = launches
            entry["kernel_launch_count"] = len(launches)
            if len(launches) != 1:
                raise RuntimeError(
                    f"{name}: expected one GPU kernel launch, observed {launches}"
                )
        report["results"][name] = entry
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        del bencher
        torch.cuda.empty_cache()
        return ms

    run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
