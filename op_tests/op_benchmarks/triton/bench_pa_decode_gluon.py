import argparse
import matplotlib
matplotlib.rcParams["figure.max_open_warning"] = 0
import os
import sys
import torch
import triton
import aiter
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.test_pa_decode_gluon import run_pa_gluon_test
import op_tests.triton_tests.test_pa_decode_gluon as _test_module
from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot_prebuild import (
    prebuild_normal_performance_cases_aot_so,
    get_so_files_size_and_count,
)

_test_module.USE_TORCH_FLASH_REF = False


arg_to_compute_type = {
    "fp8": aiter.dtypes.fp8,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def get_quant_flags(compute_type_str):
    if compute_type_str == "fp8":
        return True, True
    return False, False


def bench_pa_decode_gluon_fn(
    batch_size,
    context_length,
    num_heads,
    head_size,
    block_size,
    compute_type,
    query_length,
    quant_mode,
    quant_q,
    quant_kv,
    use_aot_impl,
    use_sinks,
    sliding_window,
    ps,
    metric,
    kv_varlen=False,
):
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)
    try:
        result = run_pa_gluon_test(
            context_length=context_length,
            batch_size=batch_size,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            compute_type=compute_type,
            query_length=query_length,
            quant_mode=quant_mode,
            context_partition_size=256,
            trans_v=False,
            kv_varlen=kv_varlen,
            use_aot_impl=use_aot_impl,
            quant_q=quant_q,
            quant_kv=quant_kv,
            use_sinks=use_sinks,
            sliding_window=sliding_window,
            ps=ps,
        )
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
        sys.stdout = os.fdopen(1, "w", closefd=False)
        sys.stderr = os.fdopen(2, "w", closefd=False)
    if metric == "time":
        return result["us_gluon"] / 1000.0
    elif metric == "bandwidth":
        return result["gluon_bandwith(TB/s)"]
    else:
        raise ValueError("Unknown metric: " + metric)


def get_x_vals_normal():
    vals = []
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        vals.append((bs, 8192))
    return vals


def get_x_vals_sliding_window():
    vals = []
    for bs in [4, 128]:
        vals.append((bs, 1024))
    return vals


COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [
    ("fp8", aiter.dtypes.fp8, True, True),
    ("bf16", torch.bfloat16, False, False),
]


def run_normal_benchmark(args, use_aot_impl=False):
    head_dim = args.head_dim if args.head_dim is not None else 128
    num_heads = tuple(args.num_heads) if args.num_heads is not None else (64, 4)

    if args.batch_size and args.context_length:
        x_vals = [(args.batch_size, args.context_length)]
    else:
        x_vals = get_x_vals_normal()

    mode_name = "normal_aot" if use_aot_impl else "normal"
    plot_name = f"{get_caller_name_no_ext()}_{mode_name}"

    query_lengths = [args.query_length] if args.query_length else [1, 4]

    if args.compute_type:
        ct_options = [(args.compute_type, arg_to_compute_type[args.compute_type],
                       *get_quant_flags(args.compute_type))]
    else:
        ct_options = COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS

    for ct_name, compute_type, quant_q, quant_kv in ct_options:
        for block_size in ([args.block_size] if args.block_size else [16, 64]):
            for ql in query_lengths:
                benchmark = triton.testing.Benchmark(
                    x_names=["batch_size", "context_length"],
                    x_vals=x_vals,
                    line_arg="metric",
                    line_vals=["time", "bandwidth"],
                    line_names=["Time_(ms)", "Bandwidth_(TB/s)"],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="ms / TB/s",
                    plot_name=f"{plot_name}_{ct_name}_blk{block_size}_ql{ql}",
                    args={},
                )

                @triton.testing.perf_report([benchmark])
                def bench(batch_size, context_length, metric,
                          _ql=ql, _compute_type=compute_type,
                          _quant_q=quant_q, _quant_kv=quant_kv, **kwargs):
                    return bench_pa_decode_gluon_fn(
                        batch_size=batch_size,
                        context_length=context_length,
                        num_heads=num_heads,
                        head_size=head_dim,
                        block_size=block_size,
                        compute_type=_compute_type,
                        query_length=_ql,
                        quant_mode=args.quant_mode,
                        quant_q=_quant_q,
                        quant_kv=_quant_kv,
                        use_aot_impl=use_aot_impl,
                        use_sinks=False,
                        sliding_window=0,
                        ps=False,
                        metric=metric,
                    )

                bench.run(save_path="." if args.o else None, print_data=True)


def run_sliding_window_benchmark(args):
    head_dim = args.head_dim if args.head_dim is not None else 64
    num_heads = tuple(args.num_heads) if args.num_heads is not None else (64, 8)

    if args.batch_size and args.context_length:
        x_vals = [(args.batch_size, args.context_length)]
    else:
        x_vals = get_x_vals_sliding_window()

    plot_name = f"{get_caller_name_no_ext()}_sliding_window"

    if args.compute_type:
        ct_options = [(args.compute_type, arg_to_compute_type[args.compute_type],
                       *get_quant_flags(args.compute_type))]
    else:
        ct_options = COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS

    for ct_name, compute_type, quant_q, quant_kv in ct_options:
        for use_sinks in [False, True]:
            for sliding_window in [0, 128]:
                benchmark = triton.testing.Benchmark(
                    x_names=["batch_size", "context_length"],
                    x_vals=x_vals,
                    line_arg="metric",
                    line_vals=["time", "bandwidth"],
                    line_names=["Time_(ms)", "Bandwidth_(TB/s)"],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="ms / TB/s",
                    plot_name=f"{plot_name}_{ct_name}_sinks{use_sinks}_sw{sliding_window}",
                    args={},
                )

                @triton.testing.perf_report([benchmark])
                def bench(
                    batch_size,
                    context_length,
                    metric,
                    _use_sinks=use_sinks,
                    _sliding_window=sliding_window,
                    _compute_type=compute_type,
                    _quant_q=quant_q,
                    _quant_kv=quant_kv,
                    **kwargs,
                ):
                    return bench_pa_decode_gluon_fn(
                        batch_size=batch_size,
                        context_length=context_length,
                        num_heads=num_heads,
                        head_size=head_dim,
                        block_size=args.block_size or 16,
                        compute_type=_compute_type,
                        query_length=args.query_length or 1,
                        quant_mode=args.quant_mode,
                        quant_q=_quant_q,
                        quant_kv=_quant_kv,
                        use_aot_impl=False,
                        use_sinks=_use_sinks,
                        sliding_window=_sliding_window,
                        ps=True,
                        metric=metric,
                        kv_varlen=True,
                    )

                bench.run(save_path="." if args.o else None, print_data=True)


def _suppress_and_prebuild():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)
    try:
        prebuild_normal_performance_cases_aot_so()
        get_so_files_size_and_count()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def run_benchmark(args):
    if args.mode == "normal":
        run_normal_benchmark(args, use_aot_impl=False)
    elif args.mode == "normal_aot":
        _suppress_and_prebuild()
        run_normal_benchmark(args, use_aot_impl=True)
    elif args.mode == "sliding_window":
        run_sliding_window_benchmark(args)
    elif args.mode == "all":
        run_normal_benchmark(args, use_aot_impl=False)
        _suppress_and_prebuild()
        run_normal_benchmark(args, use_aot_impl=True)
        run_sliding_window_benchmark(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark PA Decode Gluon",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["normal", "normal_aot", "sliding_window", "all"],
        help="Benchmark mode: normal, normal_aot, sliding_window, or all.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        choices=["fp8", "bf16", "fp16"],
        help="Compute type. Default: sweep [fp8, bf16].",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="KV block size. Default: sweep [16, 64] for normal modes, 16 for sliding_window.",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=None,
        help="Head dimension. Default: 128 for normal modes, 64 for sliding_window.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        nargs=2,
        default=None,
        help="Number of query and kv heads, e.g. --num_heads 64 4.",
    )
    parser.add_argument(
        "--query_length",
        type=int,
        default=None,
        help="Query sequence length. Default: sweep [1, 4] for normal modes, 1 for sliding_window.",
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        default="per_tensor",
        choices=["per_token", "per_tensor"],
        help="Quantization mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Fixed batch size (overrides default sweep).",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="Fixed context length (overrides default sweep).",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file.",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
