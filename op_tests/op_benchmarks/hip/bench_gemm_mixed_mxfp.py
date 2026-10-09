# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark A4W4, A4W6, A6W4, and A6W6 on diffusion GEMM shapes."""

import argparse
import csv
import gc
import resource
import statistics
from collections.abc import Callable

import torch

import aiter
from aiter.ops.gemm_op_a4w4 import (
    gemm_a4w4_asm,
    gemm_a4w4_blockscale,
    get_GEMM_config,
)
from aiter.ops.gemm_op_a4w6 import (
    _select_gemm_a4w6_kernel,
    gemm_a4w6_asm,
)
from aiter.ops.gemm_op_a6w4 import (
    _select_gemm_a6w4_kernel,
    gemm_a6w4_asm,
    quant_mxfp4_gemm,
)
from aiter.ops.gemm_op_a6w6 import (
    _ceil,
    _select_gemm_a6w6_kernel,
    gemm_a6w6_asm,
    quant_mxfp6_gemm,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import dtypes

_NAMES = ("A4W4", "A4W6", "A6W4", "A6W6")


def _event_time_us(launch: Callable[[], torch.Tensor], iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters)


def _setup_a4w4(x, w, M, N, K):
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    static_a, static_a_scale = quantize(x, quant_dtype=dtypes.fp4x2, shuffle=True)
    weight, weight_scale = quantize(w, quant_dtype=dtypes.fp4x2, shuffle=True)
    weight = shuffle_weight(weight, layout=(16, 16))

    config = get_GEMM_config(M, N, K)
    kernel = str(config.get("kernelName", "")) if config is not None else ""
    split_k = int(config.get("splitK", 0) or 0) if config is not None else 0
    use_ck = bool(config is not None and "_ZN" not in kernel)
    out = torch.empty((_ceil(M, 32), N), dtype=torch.bfloat16, device=x.device)

    def run(a, a_scale):
        if use_ck:
            gemm_a4w4_blockscale(
                a,
                weight,
                a_scale,
                weight_scale,
                out,
                splitK=split_k,
                kernelName=kernel,
            )
        else:
            gemm_a4w4_asm(
                a,
                weight,
                a_scale,
                weight_scale,
                out,
                kernelName=kernel,
                bpreshuffle=True,
                log2_k_split=split_k,
            )
        return out

    return (
        lambda: run(static_a, static_a_scale),
        lambda: run(*quantize(x, quant_dtype=dtypes.fp4x2, shuffle=True)),
        out,
        f"{'ck' if use_ck else 'asm'}:{kernel or 'heuristic'}",
    )


@torch.no_grad()
def _cosines(outputs, x, w, chunk_rows=256):
    dots = {
        name: torch.zeros((), dtype=torch.float64, device=x.device) for name in _NAMES
    }
    norms = {
        name: torch.zeros((), dtype=torch.float64, device=x.device) for name in _NAMES
    }
    ref_norm = torch.zeros((), dtype=torch.float64, device=x.device)
    for begin in range(0, x.shape[0], chunk_rows):
        ref = (x[begin : begin + chunk_rows] @ w.T).double()
        ref_norm += (ref * ref).sum()
        for name in _NAMES:
            value = outputs[name][begin : begin + chunk_rows].double()
            dots[name] += (value * ref).sum()
            norms[name] += (value * value).sum()
    return {
        name: float((dots[name] / (norms[name] * ref_norm).sqrt()).item())
        for name in _NAMES
    }


@torch.no_grad()
def benchmark_shape(args, shape_index, M, N, K):
    torch.manual_seed(args.seed + shape_index)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    a4, scale_a4 = quant_mxfp4_gemm(x, round_mode=args.round_mode)
    a6, scale_a6 = quant_mxfp6_gemm(x)
    w4, scale_w4 = quant_mxfp4_gemm(w, round_mode=args.round_mode)
    w6, scale_w6 = quant_mxfp6_gemm(w)
    padM, padN, padK = _ceil(M, 256), _ceil(N, 256), _ceil(K, 128)
    outputs = {
        name: torch.empty((padM, padN), dtype=torch.bfloat16, device=x.device)
        for name in _NAMES[1:]
    }
    kernels = {
        "A4W6": _select_gemm_a4w6_kernel(M, N, K, None, device=x.device),
        "A6W4": _select_gemm_a6w4_kernel(M, N, K, None, device=x.device),
        "A6W6": _select_gemm_a6w6_kernel(M, N, K, None),
    }

    def run_a4w6(a, scale):
        return gemm_a4w6_asm(
            a, w6, scale, scale_w6, outputs["A4W6"], padK, kernels["A4W6"]
        )

    def run_a6w4(a, scale):
        return gemm_a6w4_asm(
            a, w4, scale, scale_w4, outputs["A6W4"], padK, kernels["A6W4"]
        )

    def run_a6w6(a, scale):
        return gemm_a6w6_asm(
            a, w6, scale, scale_w6, outputs["A6W6"], padK, kernels["A6W6"]
        )

    gemm_launches = {
        "A4W6": lambda: run_a4w6(a4, scale_a4),
        "A6W4": lambda: run_a6w4(a6, scale_a6),
        "A6W6": lambda: run_a6w6(a6, scale_a6),
    }
    full_launches = {
        "A4W6": lambda: run_a4w6(*quant_mxfp4_gemm(x, round_mode=args.round_mode)),
        "A6W4": lambda: run_a6w4(*quant_mxfp6_gemm(x)),
        "A6W6": lambda: run_a6w6(*quant_mxfp6_gemm(x)),
    }
    a4_gemm, a4_full, a4_out, a4_dispatch = _setup_a4w4(x, w, M, N, K)
    outputs["A4W4"] = a4_out
    gemm_launches["A4W4"] = a4_gemm
    full_launches["A4W4"] = a4_full
    kernels["A4W4"] = a4_dispatch

    scopes = {"gemm": gemm_launches, "full": full_launches}
    for launches in scopes.values():
        for name in _NAMES:
            for _ in range(args.warmup):
                launches[name]()
    torch.cuda.synchronize()

    samples = {scope: {name: [] for name in _NAMES} for scope in scopes}
    for repeat in range(args.repeats):
        offset = repeat % len(_NAMES)
        names = _NAMES[offset:] + _NAMES[:offset]
        if repeat % 2:
            names = tuple(reversed(names))
        scope_names = ("gemm", "full") if repeat % 2 == 0 else ("full", "gemm")
        for scope in scope_names:
            for name in names:
                samples[scope][name].append(
                    _event_time_us(scopes[scope][name], args.iters)
                )

    for launch in gemm_launches.values():
        launch()
    torch.cuda.synchronize()
    cosines = _cosines({name: outputs[name][:M, :N] for name in _NAMES}, x, w)
    flop = 2.0 * M * N * K
    rows = []
    for name in _NAMES:
        gemm_us = float(statistics.median(samples["gemm"][name]))
        full_us = float(statistics.median(samples["full"][name]))
        row = {
            "M": M,
            "N": N,
            "K": K,
            "format": name,
            "gemm_us": gemm_us,
            "gemm_tflops": flop / gemm_us / 1e6,
            "quant_gemm_us": full_us,
            "quant_gemm_tflops": flop / full_us / 1e6,
            "cosine_similarity": cosines[name],
            "kernel": kernels[name],
        }
        rows.append(row)
        print(
            f"[result] {M}x{N}x{K} {name} "
            f"gemm={row['gemm_tflops']:.2f}TF "
            f"quant+gemm={row['quant_gemm_tflops']:.2f}TF "
            f"cos={row['cosine_similarity']:.8f}",
            flush=True,
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (512, 5120, 5120),
            (9450, 5120, 5120),
            (9450, 13824, 5120),
            (9450, 5120, 13824),
        ],
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--round-mode", type=int, choices=range(4), default=1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"bench_gemm_mixed_mxfp requires gfx950, got {arch}")
    args = parse_args()
    rows = []
    for index, shape in enumerate(args.shape):
        rows.extend(benchmark_shape(args, index, *shape))
        gc.collect()
        torch.cuda.empty_cache()
    with open(args.output, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
