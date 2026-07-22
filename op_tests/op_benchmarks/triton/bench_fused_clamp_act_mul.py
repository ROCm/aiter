# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf benchmark for ``fused_clamp_act_mul`` (SwiGLU clamp + act*up + optional
weights + optional FP8 group quant).

Structure mirrors the other op_benchmarks/triton bench scripts (a
``triton.testing.Benchmark`` + ``perf_report`` shape sweep, ``--metric`` /
``--shape`` / ``-o`` / ``-print_vgpr``). Timing uses ``do_bench_cudagraph`` (CUDA
graph replay) rather than ``do_bench`` to remove host launch overhead — this op
is a short, memory-bound elementwise pass where per-launch overhead would
otherwise dominate. ``--backend {auto,triton,gluon}`` selects the wrapper's
dispatch so the Triton and gfx1250 Gluon paths can be compared head-to-head.

Correctness is not checked here — see op_tests/test_fused_clamp_act_mul.py for
the value/scale reference test.
"""

import argparse

import torch
import triton
from aiter import dtypes
from aiter.ops.triton.fusions.fused_clamp_act_mul import (
    fused_clamp_act_mul,
    _is_gluon_available,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)

# quant mode -> (dtype_quant, quant_block_size, scale_dtype_fmt)
_QUANT_MODES = {
    "none": (None, 128, "fp32"),
    "fp8": (dtypes.fp8, 128, "fp32"),
    "ue8m0": (dtypes.fp8, 32, "ue8m0"),
}


def get_x_vals():
    """Default (M, N) sweep; N is n_half (the gate/up width, a multiple of 128)."""
    return [
        (1024, 2048),
        (8192, 2048),
        (16384, 4096),
        (2048, 1280),
        (4096, 5120),
        (8192, 8192),
    ]


def run_benchmark(args):
    # Resolve the backend the same way the wrapper does, so the plot name and the
    # actual dispatch agree; fail fast if gluon is explicitly asked for but absent.
    backend = args.backend
    if backend == "auto":
        backend = "gluon" if _is_gluon_available() else "triton"
    elif backend == "gluon" and not _is_gluon_available():
        raise ValueError(
            "backend='gluon' requested but this arch has no Gluon port "
            "(only gfx1250)."
        )

    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    elif args.metric == "throughput":
        ylabel = "Throughput_(TFLOPS)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    x_vals_list = [tuple(args.shape)] if args.shape is not None else get_x_vals()

    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=[ylabel],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=f"{get_caller_name_no_ext()}_{backend}_{args.quant}",
        args={"metric": args.metric},
    )

    dtype = args.dtype
    dtype_quant, quant_block_size, scale_dtype_fmt = _QUANT_MODES[args.quant]
    HAS_QUANT = dtype_quant is not None
    activation = args.activation
    swiglu_limit = args.swiglu_limit
    weighted = bool(args.weighted)

    @triton.testing.perf_report([benchmark])
    def bench_fused_clamp_act_mul(M, N, metric, **kwargs):
        n_half = N
        inp = torch.randn((M, 2 * n_half), dtype=dtype, device="cuda") * 3.0
        weights = (
            torch.randn((M, 1), dtype=dtype, device="cuda") if weighted else None
        )

        # Preallocate outputs so the timed fn does no allocation (clean CUDA-graph
        # capture): the wrapper writes into the provided out/scale buffers.
        out_dtype = dtype_quant if HAS_QUANT else dtype
        out = torch.empty((M, n_half), dtype=out_dtype, device="cuda")
        num_blocks = (n_half + quant_block_size - 1) // quant_block_size
        if HAS_QUANT:
            scale_dt = torch.uint8 if scale_dtype_fmt == "ue8m0" else torch.float32
            scale = torch.empty((M, num_blocks), dtype=scale_dt, device="cuda")
        else:
            scale = None

        fn = lambda: fused_clamp_act_mul(  # noqa: E731
            inp,
            out=out,
            scale=scale,
            swiglu_limit=swiglu_limit,
            activation=activation,
            weights=weights,
            dtype_quant=dtype_quant,
            quant_block_size=quant_block_size,
            scale_dtype_fmt=scale_dtype_fmt,
            backend=backend,
        )

        ms = triton.testing.do_bench_cudagraph(fn, rep=100, return_mode="mean")

        # Memory-bound: read gate+up (M x 2N), write out (M x N) + small scale
        # buffer; broadcast [M,1] weights are negligible but counted.
        mem_read = M * (2 * n_half) * inp.element_size()
        if weighted:
            mem_read += M * inp.element_size()
        mem_write = M * n_half * out.element_size()
        if HAS_QUANT:
            mem_write += M * num_blocks * scale.element_size()
        mem = mem_read + mem_write
        flops = 10 * M * n_half  # ~clamp+act+mul+weights+quant per element (approx)

        if metric == "time":
            return ms
        elif metric == "bandwidth":
            return mem / (ms * 1e-3) * 1e-9  # GB/s
        elif metric == "throughput":
            return flops / ms * 1e-9  # TFLOP/s
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_fused_clamp_act_mul.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark fused_clamp_act_mul",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dtype",
        type=dtypes.str2Dtype,
        default=dtypes.bf16,
        help="input/output dtype for the non-quant path (e.g. bf16)",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined (M, N=n_half) shape; N must be a multiple of 128",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth", "throughput"],
        default="bandwidth",
        help="metric to plot",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "triton", "gluon"],
        default="auto",
        help="dispatch backend; 'auto' picks gluon on gfx1250, triton elsewhere",
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "fp8", "ue8m0"],
        default="none",
        help="quant mode: none | fp8 (fp32 scales) | ue8m0 (MXFP8)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["silu", "gelu", "gelu_tanh"],
        default="silu",
        help="activation applied to gate",
    )
    parser.add_argument(
        "--swiglu-limit",
        type=float,
        default=7.0,
        help="SwiGLU clamp limit (<=0 disables the clamp)",
    )
    parser.add_argument(
        "--weighted",
        type=int,
        default=1,
        help="1 to multiply broadcast [M,1] row weights, 0 to skip",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
