# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 gfx1250 PTPC split-K baseline/experiment guard.

The production tuner times ``run_preshuffle_gemm_a8_gfx1250``, including its
standalone split-K reduction. This guard covers the three TP4 projection shapes
that select split-K at M=4..64. It checks:

* production output against torch and the explicit unchanged kernel path;
* repeated bitwise determinism;
* the configured two-kernel sequence and a same-tile split-K=1 control;
* optional HIP-graph timing (informational on a shared GPU).

Run:
  HIP_VISIBLE_DEVICES=0 python3 op_tests/test_gemm_a8w8_ptpc_splitk_f2.py
  HIP_VISIBLE_DEVICES=0 python3 op_tests/test_gemm_a8w8_ptpc_splitk_f2.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
    parse_wmma_kernel_name,
    run_preshuffle_gemm_a8_gfx1250,
)
from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
from aiter.ops.shuffle import shuffle_weight

M_VALUES = (4, 8, 16, 32, 48, 64)
PROJECTIONS = {
    "sparse_qkv": (2560, 6144),
    "dense_qkv": (2304, 6144),
    "dense_gate_up": (6144, 6144),
}
_TORCH_RTOL = 1e-2
_TORCH_ATOL = 1e-2
_MAX_MISMATCH_RATIO = 1e-2
_MAX_RELATIVE_L2 = 1e-2


def _require_gfx1250() -> None:
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250":
        pytest.skip("requires a gfx1250 GPU")


def _weight(n: int, k: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    source = torch.randn(
        (n, k), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight, scale = aiter.pertoken_quant(source, quant_dtype=dtypes.fp8)
    del source
    return weight, shuffle_weight(weight, layout=(16, 16)), scale


def _activation(m: int, k: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    source = torch.randn(
        (m, k), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    quant, scale = aiter.pertoken_quant(source, quant_dtype=dtypes.fp8)
    del source
    return quant, scale


def _config(m: int, n: int, k: int):
    config = get_GEMM_config_with_quant_type(
        m,
        n,
        k,
        dtypes.fp8,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    )
    assert config is not None, f"missing tuned config for {(m, n, k)}"
    parsed = parse_wmma_kernel_name(str(config["kernelName"]))
    assert parsed is not None, f"not a gfx1250 PTPC config: {config}"
    assert parsed["split_k"] > 1, f"expected a split-K selection: {config}"
    return config, parsed


def _run_explicit(x, wq, xs, ws, parsed, *, split_k=None):
    out = torch.empty((x.shape[0], wq.shape[0]), device=x.device, dtype=torch.bfloat16)
    run_preshuffle_gemm_a8_gfx1250(
        x,
        wq,
        xs,
        ws,
        out,
        parsed["tile_m"],
        parsed["tile_n"],
        parsed["tile_k"],
        num_buffers=parsed["num_buffers"],
        m_warp=parsed["m_warp"],
        n_warp=parsed["n_warp"],
        cluster_m=parsed["cluster_m"],
        cluster_n=parsed["cluster_n"],
        split_k=parsed["split_k"] if split_k is None else split_k,
    )
    return out


def _production(x, wq, xs, ws):
    return aiter.gemm_a8w8_bpreshuffle(x, wq, xs, ws)


def _torch_ref(x, weight, xs, ws):
    lhs = x.float() * xs.reshape(-1, 1)
    rhs = weight.float() * ws.reshape(-1, 1)
    return (lhs @ rhs.T).to(torch.bfloat16)


def _torch_metrics(actual, reference):
    actual_f32, reference_f32 = actual.float(), reference.float()
    delta = (actual_f32 - reference_f32).abs()
    mismatch_ratio = (
        (~torch.isclose(actual_f32, reference_f32, rtol=_TORCH_RTOL, atol=_TORCH_ATOL))
        .float()
        .mean()
    )
    relative_l2 = torch.linalg.vector_norm(
        actual_f32 - reference_f32, dtype=torch.float64
    ) / torch.linalg.vector_norm(reference_f32, dtype=torch.float64)
    return delta, mismatch_ratio.item(), relative_l2.item()


def _gpu_kernel_names(fn) -> list[str]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        event.name
        for event in prof.events()
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    ]


def _graph_median_us(fn, iterations: int) -> float:
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    graph.replay()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)
    )


def validate(
    *,
    m_values=M_VALUES,
    determinism_runs: int = 5,
    benchmark: bool = False,
    benchmark_iters: int = 101,
) -> list[dict]:
    _require_gfx1250()
    rows = []
    for projection_index, (projection, (n, k)) in enumerate(PROJECTIONS.items()):
        weight, weight_shuffled, weight_scale = _weight(n, k, 100 + projection_index)
        for m in m_values:
            x, x_scale = _activation(m, k, 1000 + projection_index * 100 + m)
            config, parsed = _config(m, n, k)
            production_call = partial(
                _production, x, weight_shuffled, x_scale, weight_scale
            )
            explicit_call = partial(
                _run_explicit, x, weight_shuffled, x_scale, weight_scale, parsed
            )
            split1_call = partial(
                _run_explicit,
                x,
                weight_shuffled,
                x_scale,
                weight_scale,
                parsed,
                split_k=1,
            )

            explicit = explicit_call()
            production = production_call()
            split1_call()  # warm the control before profiling/capture
            torch.cuda.synchronize()
            assert torch.equal(production, explicit)

            reference = _torch_ref(x, weight, x_scale, weight_scale)
            delta, mismatch_ratio, relative_l2 = _torch_metrics(production, reference)
            assert mismatch_ratio <= _MAX_MISMATCH_RATIO
            assert relative_l2 <= _MAX_RELATIVE_L2

            for repeat in range(determinism_runs):
                actual = production_call()
                assert torch.equal(actual, production), (
                    f"{projection} M={m}: nondeterministic output at repeat {repeat}"
                )

            production_kernels = _gpu_kernel_names(production_call)
            split1_kernels = _gpu_kernel_names(split1_call)
            assert len(production_kernels) == 2, production_kernels
            assert any("splitk_reduce" in name for name in production_kernels)
            assert len(split1_kernels) == 1, split1_kernels
            assert all("splitk_reduce" not in name for name in split1_kernels)

            denom = reference.float().abs().clamp_min(1e-12)
            row = {
                "projection": projection,
                "M": m,
                "N": n,
                "K": k,
                "config_kernel": config["kernelName"],
                "split_k": parsed["split_k"],
                "max_abs_error": delta.max().item(),
                "max_rel_error": (delta / denom).max().item(),
                "mismatch_ratio": mismatch_ratio,
                "relative_l2": relative_l2,
                "production_vs_explicit_bitwise": True,
                "determinism_runs": determinism_runs,
                "production_kernel_count": len(production_kernels),
                "production_kernels": production_kernels,
                "split1_kernel_count": len(split1_kernels),
                "split1_kernels": split1_kernels,
            }
            if benchmark:
                row["production_graph_us"] = _graph_median_us(
                    production_call, benchmark_iters
                )
                row["split1_graph_us"] = _graph_median_us(split1_call, benchmark_iters)
            rows.append(row)
        del weight, weight_shuffled, weight_scale
        torch.cuda.empty_cache()
    return rows


def test_minimax_m3_real_shapes():
    validate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-iters", type=int, default=101)
    parser.add_argument("--determinism-runs", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = validate(
        determinism_runs=args.determinism_runs,
        benchmark=args.benchmark,
        benchmark_iters=args.benchmark_iters,
    )
    payload = json.dumps(rows, indent=2)
    print(payload)
    if args.json:
        args.json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
