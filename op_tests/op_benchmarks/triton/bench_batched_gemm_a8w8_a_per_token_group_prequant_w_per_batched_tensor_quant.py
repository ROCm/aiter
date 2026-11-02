import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    batched_model_benchmark_shapes,
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    generate_batched_gemm_a16w8_inputs as generate_batched_gemm_a8w8_per_token_group_inputs,
)


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    batch: int,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    group_size: int,
    has_bias: bool,
    transpose_bm: bool,
    backend: str | None = None,
):
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(
        generate_batched_gemm_a8w8_per_token_group_inputs,
        batch,
        M,
        N,
        K,
        c_dtype,
        has_bias=has_bias,
        output=True,
        layout=layout,
        transpose_bm=transpose_bm,
    )
    # kernel names its activation/weight/output kwargs X/WQ/YQ
    _out_vals_keys = ["X", "WQ", "w_scale", "bias", "YQ"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    kernel_fn = functools.partial(
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
        group_size=group_size,
        dtype=c_dtype,
        transpose_bm=transpose_bm,
        backend=backend,
    )

    # flops
    flops = 2.0 * batch * M * N * K
    # memory transfer. w_scale/bias keep their own numel (per-batched-tensor
    # scale and a (N,)-shaped bias, neither of which reduce to a clean
    # M/N/K product).
    bias_buf = rot_bufs[0]["bias"]
    mem_read = (batch * M * K) * rot_bufs[0]["X"].element_size() + (
        batch * N * K
    ) * rot_bufs[0]["WQ"].element_size()
    mem_read += rot_bufs[0]["w_scale"].numel() * rot_bufs[0][
        "w_scale"
    ].element_size() + (
        bias_buf.numel() * bias_buf.element_size() if bias_buf is not None else 0
    )
    mem_write = (batch * M * N) * rot_bufs[0]["YQ"].element_size()
    mem = mem_read + mem_write

    ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

    # Return exactly one scalar depending on which metric is active
    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / ms * 1e-6
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def run_model_benchmark(args):
    plot_name = get_caller_name_no_ext()
    x_names = ["M", "hidden_dim", "intermediate_dim", "batch", "model_name"]
    benchmark = get_model_benchmark_object(
        plot_name,
        args,
        x_names=x_names,
        model_benchmark_shapes_fn=batched_model_benchmark_shapes,
    )

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant(
        M, hidden_dim, intermediate_dim, batch, metric, layer, **kwargs
    ):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            batch,
            M,
            N,
            K,
            metric,
            args.layout,
            args.group_size,
            not args.no_bias,
            args.transpose_bm,
            args.backend,
        )

    bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant.run(
        save_path="." if args.o else None, print_data=True
    )


def run_shape_benchmark(args):
    plot_name = get_caller_name_no_ext()
    x_names = ["batch", "M", "N", "K"]
    benchmark = get_shape_benchmark_object(plot_name, args, x_names=x_names)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant(
        batch, M, N, K, metric, **kwargs
    ):
        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            batch,
            M,
            N,
            K,
            metric,
            args.layout,
            args.group_size,
            not args.no_bias,
            args.transpose_bm,
            args.backend,
        )

    bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant.run(
        save_path="." if args.o else None, print_data=True
    )


def run_benchmark(args, defaults):
    if args.model:
        run_model_benchmark(args)
    else:
        run_shape_benchmark(args)


def parse_args(args: list[str] | None = None):
    parser = get_parser(
        "Batched A8W8 GEMM (A per-token-group pre-quant, W per-batched-tensor quant)"
    )
    parser = add_argparse_ff(parser)

    # Overriding to reduce the runtime of the benchmark
    override_parser_action(parser, action="--graph-kernels-per-graph", default=200)

    parser.add_argument("-B", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        dest="group_size",
        help="Per-token group size for X quantization (default: 128).",
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        default=False,
        help="Disable bias (required for --backend gluon).",
    )
    parser.add_argument(
        "--transpose-bm",
        action="store_true",
        default=False,
        dest="transpose_bm",
        help="Transpose batch and M dimensions in the output tensor.",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.backend == "gluon" and arch_info.get_arch() != "gfx1250":
        print(
            f"--backend gluon is gfx1250-only (running on {arch_info.get_arch()}); "
            "nothing to benchmark."
        )
        return
    if parsed_args.print_vgpr:
        print_vgpr(lambda: run_benchmark(parsed_args, defaults))
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
