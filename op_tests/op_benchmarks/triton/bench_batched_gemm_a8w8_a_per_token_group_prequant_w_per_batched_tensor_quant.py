import math
import torch
import triton
from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    generate_batched_gemm_a16w8_inputs as generate_batched_gemm_a8w8_per_token_group_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    batched_model_benchmark_shapes,
    print_vgpr,
    get_caller_name_no_ext,
)


def bench_gemm_fn(
    batch: int,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    group_size: int,
    has_bias: bool,
    transpose_bm: bool,
):
    c_dtype = torch.bfloat16
    x, weight, w_scale, bias, y = generate_batched_gemm_a8w8_per_token_group_inputs(
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
    # flops
    flops = 2.0 * batch * M * N * K
    # memory transfer
    mem_read = (
        x.numel() * x.element_size()
        + weight.numel() * weight.element_size()
        + w_scale.numel() * w_scale.element_size()
        + (bias.numel() * bias.element_size() if bias is not None else 0)
    )
    mem_write = y.numel() * y.element_size()
    mem = mem_read + mem_write

    def fn():
        return batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x,
            weight,
            w_scale,
            group_size=group_size,
            bias=bias,
            dtype=c_dtype,
            YQ=y,
            transpose_bm=transpose_bm,
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)

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
            batch,
            M,
            N,
            K,
            metric,
            args.layout,
            args.group_size,
            not args.no_bias,
            args.transpose_bm,
        )

    bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant.run(
        save_path="." if args.o else None, print_data=True
    )


def run_shape_benchmark(args):
    plot_name = get_caller_name_no_ext()
    x_names = ["batch", "M", "N", "K"]
    benchmark = get_shape_benchmark_object(plot_name, args, x_names=x_names)

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant(
        batch, M, N, K, metric, **kwargs
    ):
        return bench_gemm_fn(
            batch,
            M,
            N,
            K,
            metric,
            args.layout,
            args.group_size,
            not args.no_bias,
            args.transpose_bm,
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
        help="Disable bias.",
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
    if parsed_args.print_vgpr:
        print_vgpr(lambda: run_benchmark(parsed_args, defaults))
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
