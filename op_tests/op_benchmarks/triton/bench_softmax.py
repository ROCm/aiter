#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark softmax: AITER Triton ``softmax`` vs PyTorch ``F.softmax``.

Shape matches the common microbench: ``x`` is ``(M, N)``, softmax over the last dim.

Examples:
  python bench_softmax.py --kernel aiter --dtype bf16 -M 1 -N 7168
  python bench_softmax.py --kernel torch -M 1,4 -N 7168
  python bench_softmax.py --kernel aiter --model deepseek-V3 --dtype bf16 -o
  python bench_softmax.py --model llama3-405B --config-n hidden -o
"""
import argparse
import sys

import torch
import torch.nn.functional as F
import triton
from aiter.ops.triton.softmax import softmax as aiter_softmax
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    print_vgpr,
    get_caller_name_no_ext,
)

DEFAULT_M_SIZES = (1, 4, 16, 64, 256, 1024, 4096)
DEFAULT_N_SIZES = (7168,)


def parse_int_list(s: str | None, default: tuple[int, ...]) -> list[int]:
    if s is None or not str(s).strip():
        return list(default)
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        return list(default)
    return [int(x) for x in parts]


def _n_values_from_model_cfg(cfg: dict, config_n: str) -> list[int]:
    """Pick last-dim N from model_configs.json (hidden / intermediate / both)."""
    hidden = int(cfg["hidden_size"])
    inter = cfg.get("intermediate_size")
    inter_i = int(inter) if inter is not None else None

    if config_n == "hidden":
        return [hidden]
    if config_n == "intermediate":
        if inter_i is None:
            return [hidden]
        return [inter_i]
    # both
    if inter_i is None or inter_i == hidden:
        return [hidden]
    return [hidden, inter_i]


def softmax_benchmark_configs(args) -> list[tuple[str, int, int, str]]:
    """Rows: (model_label, M, N, kernel_name)."""
    m_list = args.M_list
    kernel = args.kernel
    out: list[tuple[str, int, int, str]] = []

    if args.model is not None:
        configs = get_model_configs(
            config_path=args.model_configs, models=args.model
        )
        for model_name, cfg in configs.items():
            for n in _n_values_from_model_cfg(cfg, args.config_n):
                for m in m_list:
                    out.append((model_name, m, n, kernel))
        return out

    for n in args.N_list:
        for m in m_list:
            out.append((f"synthetic-N{n}", m, n, kernel))
    return out


def make_bench_fn(
    M: int,
    N: int,
    kernel: str,
    act_dtype: torch.dtype,
):
    x = torch.randn(M, N, dtype=act_dtype, device="cuda")

    if kernel == "aiter":

        def fn():
            aiter_softmax(x)

    elif kernel == "torch":

        def fn():
            F.softmax(x, dim=-1)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return fn


def bytes_per_iter(M: int, N: int, act_dtype: torch.dtype) -> float:
    """Read x + write y (same dtype), same convention as M*N*esize*2*2 style refs."""
    es = torch.tensor([], dtype=act_dtype).element_size()
    return float(M * N * es * 2)


def run_benchmark(args):
    act_dtype = str_to_torch_dtype[args.dtype]
    print_time = args.print_time

    x_vals_list = softmax_benchmark_configs(args)
    x_names = ["model", "M", "N", "kernel"]

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

    # Softmax: rough op count (exp, sub, sum, div) ~ 5 eltwise passes over row
    flops_scale = 5.0

    @triton.testing.perf_report([benchmark])
    def bench_softmax(model, M, N, kernel, metric):
        fn = make_bench_fn(M, N, kernel, act_dtype)
        mem = bytes_per_iter(M, N, act_dtype)
        flops = flops_scale * M * N

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

    bench_softmax.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark softmax (AITER Triton vs PyTorch)",
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
            "Model id(s) from model_configs.json; N from --config-n (hidden/intermediate/both). "
            "If omitted, use synthetic shapes with -N. Examples: "
            + ", ".join(available[:12])
            + ("..." if len(available) > 12 else "")
        ),
    )
    parser.add_argument(
        "--config-n",
        choices=("hidden", "intermediate", "both"),
        default="both",
        dest="config_n",
        help=(
            "When --model is set: which dims from JSON to use as softmax N "
            "(default both: hidden_size and intermediate_size if present and distinct)."
        ),
    )
    parser.add_argument(
        "-M",
        type=str,
        default=",".join(str(x) for x in DEFAULT_M_SIZES),
        help="Row count M (comma-separated), i.e. batch of softmax rows.",
    )
    parser.add_argument(
        "-N",
        type=str,
        default=",".join(str(x) for x in DEFAULT_N_SIZES),
        help="Last dim N when --model is not set (comma-separated). Ignored if --model is set.",
    )
    parser.add_argument(
        "--kernel",
        "-kernel",
        choices=("aiter", "torch"),
        default="aiter",
        help="aiter: aiter.ops.triton.softmax.softmax; torch: torch.nn.functional.softmax.",
    )
    parser.add_argument(
        "--dtype",
        "-dtype",
        default="bf16",
        help="Input dtype (str_to_torch_dtype), e.g. bf16, fp16.",
    )
    parser.add_argument("-print_time", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write CSV via Triton perf_report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.M_list = parse_int_list(args.M, DEFAULT_M_SIZES)
    args.N_list = parse_int_list(args.N, DEFAULT_N_SIZES)

    if args.print_vgpr:

        def fun():
            run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
