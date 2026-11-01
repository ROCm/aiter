# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused SiTU-and-mul kernel against the eager and
torch.compile forms of the same activation.
"""

import sys

import torch
import triton

from aiter.ops.triton.fusions.fused_situ_mul import fused_situ_mul
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_situ_mul import (
    generate_fused_situ_mul_inputs,
    torch_situ_mul_ref,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_PROVIDERS = ("fused", "eager", "compiled")
_BETA, _LINEAR_BETA, _K3_D = 4.0, 25.0, 768
_DECODE_TOKENS = (1, 5, 10, 15, 16, 32, 64, 128, 320)
_PREFILL_TOKENS = (1112, 2136, 4096, 7690)


def get_benchmark_shapes(args):
    """Return [(N, D), ...] for the current CLI args; D is the output width."""
    if args.N and args.D:
        return [(args.N, args.D)]
    tokens = ()
    if args.sweep in ("decode", "all"):
        tokens += _DECODE_TOKENS
    if args.sweep in ("prefill", "all"):
        tokens += _PREFILL_TOKENS
    return [(n, _K3_D) for n in tokens]


# Dynamic in N, as vLLM compiles the model.
_compiled_ref = torch.compile(torch_situ_mul_ref, dynamic=True)


def bench_fused_situ_mul_fn(N, D, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    x = generate_fused_situ_mul_inputs((N, 2 * D), dtype)
    linear_beta = args.linear_beta if args.linear_beta > 0 else None
    # Read [gate | up], write the product.
    mem = 3 * N * D * x.element_size()

    if provider == "fused":

        def fn():
            return fused_situ_mul(x, args.beta, linear_beta)

    elif provider == "eager":

        def fn():
            return torch_situ_mul_ref(x, args.beta, linear_beta)

    else:

        def fn():
            return _compiled_ref(x, args.beta, linear_beta)

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["N", "D"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("red", "--"),
            ("blue", "--"),
            ("green", "--"),
        ][: len(line_vals)],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(N, D, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fused_situ_mul_fn(N, D, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="fused SiTU-and-mul")
    parser.add_argument("-N", type=int, default=None, help="Number of tokens")
    parser.add_argument(
        "-D", type=int, default=None, help="Output width (the input is 2 * D wide)"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["prefill", "decode", "all"],
        help="Token counts to sweep: prefill sizes, the decode range (1-320), or both",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused kernel, eager torch, torch.compile of the eager form, or all",
    )
    parser.add_argument("--beta", type=float, default=_BETA, help="SiTU beta")
    parser.add_argument(
        "--linear_beta",
        type=float,
        default=_LINEAR_BETA,
        help="SiTU linear beta for the up clip, 0 for no clip",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )
    override_parser_action(
        parser, action="--metric", choices=["all", "time", "bandwidth"], default="all"
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for fused_situ_mul Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
