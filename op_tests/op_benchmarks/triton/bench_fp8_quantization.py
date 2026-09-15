#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark FP8 quantization: AITER ``fused_flatten_fp8_group_quant`` vs optional PyTorch reference.

Uses ``utils/model_configs.json`` for ``--model`` (hidden_size as K) or explicit ``-K`` sweep.

Examples:
  python bench_fp8_quantization.py --kernel aiter_flatten --dtype bf16 -M 1 -K 7168
  python bench_fp8_quantization.py --kernel torch_row --model deepseek-V3 --dtype bf16 -o
"""
import argparse
import sys

import aiter
import torch
import triton
from aiter.ops.triton.quant.fused_fp8_quant import fused_flatten_fp8_group_quant
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    print_vgpr,
    get_caller_name_no_ext,
)

DEFAULT_M_SIZES = (1, 4, 16, 64, 256, 1024, 4096)
DEFAULT_K_SIZES = (7168,)
FP8_DTYPE = aiter.dtypes.fp8
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def parse_int_list(s: str | None, default: tuple[int, ...]) -> list[int]:
    if s is None or not str(s).strip():
        return list(default)
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        return list(default)
    return [int(x) for x in parts]


def quant_benchmark_configs(args) -> list[tuple[str, int, int, str]]:
    """Rows: (model_label, M, K, kernel_name)."""
    m_list = args.M_list
    kernel = args.kernel
    out: list[tuple[str, int, int, str]] = []

    if args.model is not None:
        configs = get_model_configs(
            config_path=args.model_configs, models=args.model
        )
        for model_name, cfg in configs.items():
            k = int(cfg["hidden_size"])
            for m in m_list:
                out.append((model_name, m, k, kernel))
        return out

    for k in args.K_list:
        for m in m_list:
            out.append((f"synthetic-K{k}", m, k, kernel))
    return out


def make_bench_fn(
    M: int,
    K: int,
    kernel: str,
    act_dtype: torch.dtype,
    group_size: int,
):
    x = torch.randn(M, K, dtype=act_dtype, device="cuda")
    x3 = x.view(M, 1, K).contiguous()

    if kernel == "aiter_flatten":

        def fn():
            fused_flatten_fp8_group_quant(
                x3,
                group_size=group_size,
                dtype_quant=FP8_DTYPE,
            )

    elif kernel == "torch_row":

        def fn():
            amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
            scale = amax / FP8_MAX
            _ = (x / scale).to(FP8_DTYPE)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return fn


def bytes_per_iter(M: int, K: int, act_dtype: torch.dtype, kernel: str, group_size: int) -> float:
    """Rough traffic: read activations + write fp8 output + block scales (AITER path)."""
    in_b = torch.tensor([], dtype=act_dtype).element_size() * M * K
    out_fp8 = M * K
    if kernel == "aiter_flatten":
        n_blk = triton.cdiv(K, group_size)
        scales = M * n_blk * 4
        return float(in_b + out_fp8 + scales)
    # torch_row: read x, write y fp8 + amax scales (M floats) kept small
    return float(in_b + out_fp8 + M * 4)


def run_benchmark(args):
    act_dtype = str_to_torch_dtype[args.dtype]
    print_time = args.print_time

    x_vals_list = quant_benchmark_configs(args)
    x_names = ["model", "M", "K", "kernel"]

    if print_time:
        line_names = ["Time_(ms)"]
        line_vals = ["time"]
    else:
        line_names = ["Time_(ms)", "TFLOPS", "Bandwidth_(GB/s)"]
        line_vals = ["time", "tflops", "bandwidth"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms / TFLOPS / GB/s",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fp8_quant(model, M, K, kernel, metric):
        fn = make_bench_fn(
            M, K, kernel, act_dtype, group_size=args.group_size
        )
        mem = bytes_per_iter(M, K, act_dtype, kernel, args.group_size)
        flops = 6.0 * M * K

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        bandwidth = mem / (ms * 1e-3) * 1e-9
        tflops = flops / ms * 1e-9

        if metric == "time":
            return ms
        if metric == "tflops":
            return tflops
        if metric == "bandwidth":
            return bandwidth
        raise ValueError("Unknown metric: " + metric)

    bench_fp8_quant.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark FP8 quantization (AITER + optional PyTorch reference)",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-model_configs",
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        dest="model_configs",
    )
    available = get_available_models()
    parser.add_argument(
        "--model",
        "-model",
        type=str,
        default=None,
        help=(
            "Model id(s) from model_configs.json; K = hidden_size. "
            "If omitted, use synthetic shapes with -K. Known: "
            + ", ".join(available[:20])
            + ("..." if len(available) > 20 else "")
        ),
    )
    parser.add_argument(
        "-M",
        type=str,
        default=",".join(str(x) for x in DEFAULT_M_SIZES),
        help="Row dimension(s) M (comma-separated).",
    )
    parser.add_argument(
        "-K",
        type=str,
        default=",".join(str(x) for x in DEFAULT_K_SIZES),
        help="Hidden size K when --model is not set (comma-separated). Ignored if --model is set.",
    )
    parser.add_argument(
        "--kernel",
        "-kernel",
        choices=("aiter_flatten", "torch_row"),
        default="aiter_flatten",
        help="aiter_flatten: fused_flatten_fp8_group_quant; torch_row: PyTorch amax/scale/cast.",
    )
    parser.add_argument(
        "--group-size",
        "-group-size",
        type=int,
        default=128,
        dest="group_size",
        help="Group size for aiter_flatten (ignored for torch_row).",
    )
    parser.add_argument(
        "--dtype",
        "-dtype",
        default="bf16",
        help="Activation dtype (str_to_torch_dtype), e.g. bf16, fp16.",
    )
    parser.add_argument("-print_time", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("-o", action="store_true", help="Write CSV via Triton perf_report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.M_list = parse_int_list(args.M, DEFAULT_M_SIZES)
    args.K_list = parse_int_list(args.K, DEFAULT_K_SIZES)

    if args.print_vgpr:

        def fun():
            run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
