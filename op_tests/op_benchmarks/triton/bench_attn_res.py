# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the attention-residual (residual candidate gating) kernel.

Sweep (N tokens, D hidden, L candidates). Report GPU time in ms and logical
bandwidth in GB/s from one rotating-buffer measurement per shape and layout.

Usage (from the repository root):
  python op_tests/op_benchmarks/triton/bench_attn_res.py
  python op_tests/op_benchmarks/triton/bench_attn_res.py --layout packed
  python op_tests/op_benchmarks/triton/bench_attn_res.py -N 8192 -D 7168 -L 8
  python op_tests/op_benchmarks/triton/bench_attn_res.py --metric bandwidth --onorm
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --add-hidden
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --add-hidden2
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --close-block
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --sweep decode
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --backend triton
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --backend gluon --sweep tdm --add-hidden2 --onorm
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --backend gluon --sweep tdm --onorm --out-quant fp8

The TDM preset uses synthetic shapes with model width D=7168, not exact trace
invocations. L includes the prefix; the gate block has B=L-1 candidates.

Logical bandwidth counts each input once and each required output once. It is
not measured HBM traffic; it excludes repeated reads, temporary buffers and
cache effects. Use the shared --graph-* and --rot-bufs-* flags for timing.
"""

import sys

import torch
import triton

from aiter import logger
from aiter.ops.triton.fusions.attn_res import attn_res_fwd, attn_res_gate
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
    positive_int,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.fusions.test_attn_res import (
    generate_attn_res_gate_inputs,
    generate_attn_res_inputs,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_LAYOUTS = ("sequence", "packed")


# Prefill-sized token counts, plus the decode range where the grid is small
# enough that launch overhead and occupancy, not bandwidth, set the runtime.
_PREFILL_TOKENS = (1166, 8192, 16384)
_DECODE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)
_TDM_TOKENS = (1, 128, 256, 257, 512, 2048, 16384)


def get_benchmark_shapes(args):
    """Return [(N, D, L), ...] for the current CLI args."""
    if args.N is not None:
        return [(args.N, args.D, args.L)]
    if args.sweep == "tdm":
        return [(N, 7168, L) for N in _TDM_TOKENS for L in (2, 3, 4, 9)]
    tokens = ()
    if args.sweep in ("decode", "all"):
        tokens += _DECODE_TOKENS
    if args.sweep in ("prefill", "all"):
        tokens += _PREFILL_TOKENS
    return [(N, 7168, L) for N in tokens for L in (2, 4, 8)]


def get_traffic_bytes(N, D, L, elem_size, args):
    """Count logical input and output bytes, not measured HBM traffic."""
    mem = N * L * D * elem_size
    if args.op == "gate":
        mem += D * elem_size  # score weight
        if args.add_hidden or args.add_hidden2:
            mem += 2 * N * D * elem_size  # first addend and new prefix
        if args.add_hidden2:
            mem += N * D * elem_size
        if args.close_block:
            mem += N * L * D * elem_size
    else:
        mem += 2 * D * elem_size  # query and RMS weight
    if args.onorm:
        mem += D * elem_size
    if args.out_quant == "fp8":
        mem += N * D + N * 4  # FP8 output and one FP32 scale per token
    else:
        mem += N * D * elem_size
    return mem


def bench_attn_res_fn(N, D, L, layout, args):
    dtype = arg_to_torch_dtype[args.dtype]
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    if args.op == "gate":
        out_quant_dtype = get_fp8_e4m3_dtype() if args.out_quant == "fp8" else None

        def generate_inputs():
            inputs = generate_attn_res_gate_inputs(
                N,
                D,
                L - 1,
                dtype,
                with_add=args.add_hidden or args.add_hidden2,
                with_add2=args.add_hidden2,
            )
            output_rms_weight = (
                torch.randn(D, dtype=dtype, device="cuda") if args.onorm else None
            )
            return (*inputs, output_rms_weight)

        keys = [
            "prefix",
            "block_residual",
            "score_weight",
            "add_hidden",
            "add_hidden2",
            "output_rms_weight",
        ]
        pool = bencher.build_pool([generate_inputs], [keys])
        elem_size = pool[0]["prefix"].element_size()

        def fn(**inputs):
            return attn_res_gate(
                **inputs,
                eps=args.eps,
                output_rms_eps=args.out_eps,
                scale=args.scale,
                close_block=args.close_block,
                out_quant_dtype=out_quant_dtype,
                backend=args.backend,
            )

    else:

        def generate_inputs():
            query, residuals, rms_weight, output_rms_weight = generate_attn_res_inputs(
                N, D, L, dtype, with_onorm=args.onorm
            )
            if layout == "packed":
                residuals = torch.stack(residuals, dim=-2).contiguous()
            return query, residuals, rms_weight, output_rms_weight

        keys = ["query", "residuals", "rms_weight", "output_rms_weight"]
        pool = bencher.build_pool([generate_inputs], [keys])
        elem_size = pool[0]["query"].element_size()

        def fn(**inputs):
            return attn_res_fwd(
                **inputs, rms_eps=args.eps, scale=args.scale, layout=layout
            )

    ms = bencher.bench(fn)
    mem = get_traffic_bytes(N, D, L, elem_size, args)
    input_layouts = {}
    for name, value in pool[0].items():
        if value is not None:
            tensors = value if isinstance(value, (list, tuple)) else [value]
            input_layouts[name] = [
                {
                    "shape": tuple(t.shape),
                    "stride": tuple(t.stride()),
                    "dtype": str(t.dtype),
                }
                for t in tensors
            ]
    logger.info("Input tensors: %s; output_mode=%s", input_layouts, args.out_quant)
    triton_kernel = (
        "aiter/ops/triton/_triton_kernels/fusions/attn_res.py:attnres_fwd_kernel"
    )
    gluon_kernel = (
        "aiter/ops/triton/_gluon_kernels/gfx1250/fusions/attn_res.py:"
        "attn_res_gate_tdm_kernel_gfx1250"
    )
    if args.op == "fwd" or args.backend == "triton":
        kernel = triton_kernel
    elif args.backend == "gluon":
        kernel = gluon_kernel
    else:
        kernel = "auto dispatch: " + triton_kernel + " or " + gluon_kernel
    logger.info(
        "op=%s N=%s D=%s L=%s B=%s layout=%s backend=%s source_kernel=%s "
        "time_ms=%s logical_bytes=%s logical_bandwidth_GBps=%s",
        args.op,
        N,
        D,
        L,
        L - 1 if args.op == "gate" else None,
        layout,
        "triton" if args.op == "fwd" else args.backend or "auto",
        kernel,
        ms,
        mem,
        mem / (ms * 1e-3) * 1e-9,
    )
    return ms, mem


def run_benchmark(args):
    logger.info("Benchmark flags: %s", vars(args))
    logger.info(
        "Timing: RotatingBufferBench GPU graph replay of the public wrapper; "
        "source names are not profiler observations"
    )
    if args.sweep == "tdm":
        logger.info(
            "TDM sweep: synthetic cases at model width D=7168, not exact trace invocations"
        )
    if args.op == "gate":
        layouts = ("packed",)  # the gate is packed-only
    else:
        layouts = _LAYOUTS if args.layout == "all" else (args.layout,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{layout}_{metric}" for metric in metrics for layout in layouts]
    units = {"time": "ms", "bandwidth": "GB/s (logical)"}
    line_names = [
        f"{layout}_{metric} [{units[metric]}]"
        for metric in metrics
        for layout in layouts
    ]

    benchmark = triton.testing.Benchmark(
        x_names=["N", "D", "L"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=(
            get_caller_name_no_ext()
            + f"_{args.op}_{args.dtype}_{args.backend or 'auto'}_{args.out_quant}"
            + f"_onorm{int(args.onorm)}_add{2 if args.add_hidden2 else int(args.add_hidden)}"
            + f"_close{int(args.close_block)}"
        ),
        args={},
    )

    measurements = {}

    @triton.testing.perf_report([benchmark])
    def bench_fn(N, D, L, provider):
        layout, metric = provider.rsplit("_", 1)
        key = (N, D, L, layout)
        if key not in measurements:
            measurements[key] = bench_attn_res_fn(N, D, L, layout, args)
        ms, mem = measurements[key]
        return ms if metric == "time" else mem / (ms * 1e-3) * 1e-9

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args(argv=None):
    parser = get_parser(kernel_name="attention-residual")
    parser.add_argument("-N", type=positive_int, default=None, help="Number of tokens")
    parser.add_argument("-D", type=positive_int, default=None, help="Hidden size")
    parser.add_argument(
        "-L",
        type=positive_int,
        default=None,
        help="Residual candidates, including the gate prefix",
    )
    parser.add_argument(
        "--op",
        type=str,
        default="fwd",
        choices=["fwd", "gate"],
        help="attn_res_fwd (both layouts) or attn_res_gate (packed, inference)",
    )
    parser.add_argument(
        "--add-hidden",
        dest="add_hidden",
        action="store_true",
        default=False,
        help="For --op gate: fold the prefix += hidden add into the kernel",
    )
    parser.add_argument(
        "--add-hidden2",
        dest="add_hidden2",
        action="store_true",
        default=False,
        help="For --op gate: also fold a second addend (implies --add-hidden)",
    )
    parser.add_argument(
        "--close-block",
        dest="close_block",
        action="store_true",
        default=False,
        help=(
            "For --op gate: also fuse cat([block_residual, prefix_out], -2) into "
            "the kernel, instead of a separate torch.cat"
        ),
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="prefill",
        choices=["prefill", "decode", "all", "tdm"],
        help="Token sweep; tdm uses synthetic D=7168 cases and requires --op gate --backend gluon",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="all",
        choices=[*_LAYOUTS, "all"],
        help="Residual layout: sequence (2-pass), packed (1-pass), or all",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )

    override_parser_action(
        parser, action="--metric", choices=["all", "time", "bandwidth"], default="all"
    )

    parser.add_argument(
        "--onorm", action="store_true", default=False, help="Enable the output RMSNorm"
    )
    parser.add_argument(
        "--out-quant",
        choices=["none", "fp8"],
        default="none",
        help="FP8 output with one FP32 scale per token; requires --op gate --onorm",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument(
        "--out-eps",
        type=float,
        default=1e-6,
        help="For --op gate: output RMSNorm epsilon (independent of --eps)",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Logit scale")
    parser.add_argument(
        "--warmup",
        type=float,
        help="Removed; use --graph-warmup-iters and --graph-warmup-replays",
    )
    parser.add_argument(
        "--rep",
        type=float,
        help="Removed; use --graph-num-replays and --graph-kernels-per-graph",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    args = parser.parse_args(argv)
    if args.warmup is not None or args.rep is not None:
        parser.error(
            "--warmup/--rep were removed; use --graph-warmup-iters, --graph-warmup-replays, --graph-num-replays and --graph-kernels-per-graph"
        )
    dimensions = (args.N, args.D, args.L)
    if any(v is not None for v in dimensions) and not all(
        v is not None for v in dimensions
    ):
        parser.error("-N, -D and -L must be specified together")
    if args.sweep == "tdm":
        if args.N is not None:
            parser.error("--sweep tdm cannot be combined with -N/-D/-L")
        if args.op != "gate" or args.backend != "gluon":
            parser.error("--sweep tdm requires --op gate --backend gluon")
    if args.out_quant == "fp8" and (args.op != "gate" or not args.onorm):
        parser.error("--out-quant fp8 requires --op gate --onorm")
    if args.op == "fwd" and (
        args.add_hidden
        or args.add_hidden2
        or args.close_block
        or args.backend == "gluon"
    ):
        parser.error(
            "--add-hidden, --add-hidden2, --close-block and --backend gluon require --op gate"
        )
    if args.op == "gate" and args.layout == "sequence":
        parser.error("--op gate requires the packed layout")
    if args.backend == "gluon":
        if args.dtype == "fp32" or args.close_block:
            parser.error(
                "--backend gluon requires fp16/bf16 and does not support --close-block"
            )
        if args.N is not None and (args.L < 2 or args.D > 8192 or args.D % 8):
            parser.error(
                "--backend gluon requires L >= 2, D <= 8192 and D divisible by 8"
            )
    return args


def main():
    args = parse_args()
    if args.print_vgpr:
        logger.info("Retrieving VGPR usage for attn_res Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
