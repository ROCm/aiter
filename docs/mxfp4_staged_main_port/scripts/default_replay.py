#!/usr/bin/env python3
"""Coordinator-launched exact CSV replay, AOT build and precompiled replay."""

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from unittest import mock

from docs.mxfp4_staged_main_port.scripts.replay_common import (
    fingerprint,
    trace_kernel_metrics,
)

ENV_KEYS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
    "AITER_FLYDSL_STAGE2_FP8",
    "AITER_MXFP4_INTERMEDIATE",
    "AITER_JIT_DIR",
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_RUNTIME_RUN_ONLY",
    "GPU_ARCHS",
    "CK_DIR",
)


def write_json(path: "Path", value: "object") -> "None":
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> "None":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--action", choices=("replay", "aot"), default="replay")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    args = parser.parse_args()
    started = time.perf_counter()
    implementation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    os.environ["AITER_CONFIG_FMOE"] = str(args.csv.resolve())
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1, rows
    row = rows[0]
    if args.action == "aot":
        from aiter.aot.flydsl.mxfp4_moe import compile_one_config, parse_csv

        jobs = parse_csv(str(args.csv))
        results = [compile_one_config(**job) for job in jobs]
        assert all(result["compile_time"] is not None for result in results), results
        write_json(
            args.result,
            {
                "implementation_commit": implementation_commit,
                "csv": str(args.csv.resolve()),
                "jobs": jobs,
                "results": results,
                "wall_seconds": time.perf_counter() - started,
                "environment": {key: os.environ.get(key) for key in ENV_KEYS},
            },
        )
        return

    import torch

    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.mxfp4_kname import (
        _parse_mxfp4_g1_kname,
        parse_flydsl_v2_gemm2_kernel,
    )
    from aiter.test_common import get_trace_perf
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
        Mxfp4FlydslTuner,
        cosine_diff_compare,
    )

    fm = importlib.import_module("aiter.fused_moe")
    torch.cuda.set_device(0)
    shape = {
        name: int(row[name])
        for name in ("token", "model_dim", "inter_dim", "expert", "topk")
    }
    activation = getattr(ActivationType, row["act_type"].rsplit(".", 1)[-1])
    tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
    setup_started = time.perf_counter()
    data = tuner._prepare_case(**shape, dtype=dtypes.bf16)
    reference = tuner._torch_ref(data, shape["topk"], dtypes.bf16, activation)
    setup_seconds = time.perf_counter() - setup_started
    operations = []
    kernel_module = importlib.import_module("aiter.ops.flydsl.mxfp4_gemm1_kernels")
    g2_module = importlib.import_module("aiter.ops.flydsl.kernels.mxmoe_dispatcher")
    parsed_g1 = _parse_mxfp4_g1_kname(row["kernelName1"])
    parsed_g2 = parse_flydsl_v2_gemm2_kernel(row["kernelName2"])
    stage1 = kernel_module.flydsl_mxfp4_gemm1
    stage2 = g2_module.mxfp4_moe_gemm2

    def capture_g1(**kwargs: "object") -> "tuple[torch.Tensor, torch.Tensor]":
        for key in (
            "BM",
            "BN",
            "BK",
            "use_nt",
            "inline_quant",
            "prefetch_hidden",
            "xcd_swizzle",
            "num_waves",
            "k_wave",
            "a_dtype",
            "out_dtype",
            "act",
        ):
            assert kwargs[key] == parsed_g1[key], (key, kwargs[key], parsed_g1[key])
        assert kwargs["interleave"] is False
        assert kwargs["native_scale_layout"] == (parsed_g1["BM"] == 16)
        operations.append(
            {
                "op": "g1",
                "BM": kwargs["BM"],
                "a_dtype": kwargs["a_dtype"],
                "out_dtype": kwargs["out_dtype"],
                "inline_quant": kwargs["inline_quant"],
                "native_scale_layout": kwargs["native_scale_layout"],
                "interleave": kwargs["interleave"],
                "hidden_dtype": str(kwargs["hidden_states"].dtype),
                "swiglu_limit": str(kwargs["swiglu_limit"]),
            }
        )
        return stage1(**kwargs)

    def capture_g2(*positional: "object", **kwargs: "object") -> "torch.Tensor":
        for key, parsed_key in (
            ("BM", "tile_m"),
            ("BN", "tile_n"),
            ("BK", "tile_k"),
            ("use_nt", "use_nt"),
            ("epilog", "epilog"),
            ("a_dtype", "a_dtype"),
            ("b_dtype", "b_dtype"),
            ("out_dtype", "out_dtype"),
        ):
            assert kwargs[key] == parsed_g2[parsed_key], (
                key,
                kwargs[key],
                parsed_g2[parsed_key],
            )
        operations.append(
            {
                "op": "g2",
                "kwargs": {
                    key: str(value)
                    for key, value in kwargs.items()
                    if key
                    in (
                        "BM",
                        "BN",
                        "BK",
                        "epilog",
                        "a_dtype",
                        "b_dtype",
                        "out_dtype",
                        "persist",
                        "SBM",
                    )
                },
            }
        )
        return stage2(*positional, **kwargs)

    metadata = fm.get_2stage_cfgs(
        shape["token"],
        shape["model_dim"],
        shape["inter_dim"],
        shape["expert"],
        shape["topk"],
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        activation,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode="separated",
        opus_weights_shuffled=True,
    )
    assert metadata.stage1.keywords["kernelName1"] == row["kernelName1"], metadata
    assert metadata.stage2.keywords["kernelName2"] == row["kernelName2"], metadata
    assert (
        metadata.block_m == int(row["block_m"]) and metadata.fuse_quant == "fp4"
    ), metadata

    def production() -> "torch.Tensor":
        return fm.fused_moe(
            data["input"],
            data["w1_a16"],
            data["w2_a16"],
            data["topk_weights"],
            data["topk_ids"],
            activation=activation,
            quant_type=QuantType.per_1x32,
            dtype=dtypes.bf16,
            w1_scale=data["w1s_a16"],
            w2_scale=data["w2s_a16"],
            gate_mode="separated",
            quant_dtype_a=dtypes.fp4x2,
        )

    with mock.patch.object(
        kernel_module, "flydsl_mxfp4_gemm1", capture_g1
    ), mock.patch.object(g2_module, "mxfp4_moe_gemm2", capture_g2):
        output = production()
    torch.cuda.synchronize()
    error = float(cosine_diff_compare(reference, output, printLog=False))
    assert (
        math.isfinite(error) and error <= 0.1 and torch.isfinite(output).all().item()
    ), error
    assert (
        operations
        and operations[0]["op"] == "g1"
        and any(item["op"] == "g2" for item in operations)
    ), operations
    for _ in range(args.warmup):
        production()
    torch.cuda.synchronize()
    elapsed = []
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(args.iters):
            begin = time.perf_counter()
            production()
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - begin) * 1e6)
    trace = args.result.with_suffix(".trace.json")
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels, kernel_sum, span = trace_kernel_metrics(events, args.iters)
    metric = float(get_trace_perf(prof, args.iters))
    assert math.isfinite(metric) and metric > 0
    write_json(
        args.result,
        {
            "implementation_commit": implementation_commit,
            "csv": str(args.csv.resolve()),
            "shape": shape,
            "activation": row["act_type"],
            "kernelName1": row["kernelName1"],
            "kernelName2": row["kernelName2"],
            "block_m": metadata.block_m,
            "fuse_quant": metadata.fuse_quant,
            "explicit_q_dtype_a": str(dtypes.fp4x2),
            "quant_type": str(QuantType.per_1x32),
            "operations": operations,
            "finite": True,
            "cosine_diff": error,
            "tuner_kernel_sum_us": float(row["us"]),
            "synchronized_diagnostic_kernel_sum_with_main_aggregation_us": metric,
            "raw_real_kernel_duration_sum_us": kernel_sum,
            "matched_synchronized_real_kernel_trace_span_us": span,
            "matched_synchronized_elapsed_median_us": statistics.median(elapsed),
            "matched_synchronized_elapsed_min_us": min(elapsed),
            "matched_synchronized_elapsed_max_us": max(elapsed),
            "trace_real_kernel_count": len(kernels),
            "measurement": {
                "warmup": args.warmup,
                "iters": args.iters,
                "timing_scope": "synchronized per-call diagnostic, using main kernel-duration aggregation; span and elapsed from same synchronized calls; CSV tuner latency uses normal continuous run_perftest",
            },
            "setup_seconds": setup_seconds,
            "wall_seconds": time.perf_counter() - started,
            "torch_seed": 0,
            "fingerprints": {
                key: fingerprint(data[key])
                for key in ("input", "topk_ids", "topk_weights")
            },
            "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        },
    )


if __name__ == "__main__":
    main()
