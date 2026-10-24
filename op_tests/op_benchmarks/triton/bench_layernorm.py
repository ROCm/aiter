import functools

import torch
import triton

from aiter.ops.triton.normalization.norm import (
    layer_norm,
    layernorm2d_fwd_with_add,
    layernorm2d_fwd_with_add_dynamicquant,
    layernorm2d_fwd_with_add_smoothquant,
    layernorm2d_fwd_with_dynamicquant,
    layernorm2d_fwd_with_smoothquant,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    print_vgpr,
)


def generate_layernorm_inputs(M, N, dtype, add_residual):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.rand(N, device="cuda", dtype=dtype)
    bias = torch.rand(N, device="cuda", dtype=dtype)
    res_in = torch.randn(M, N, device="cuda", dtype=dtype) if add_residual else None
    return x, weight, bias, res_in


def _generate_layernorm_backward_inputs(M, N, dtype, add_residual):
    eps = 1e-5
    x, weight, bias, res_in = generate_layernorm_inputs(M, N, dtype, add_residual)
    x.requires_grad_(True)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    if add_residual:
        out = torch.empty_like(x)
        res_out = torch.empty_like(x)
        y = layernorm2d_fwd_with_add(out, x, res_in, res_out, weight, bias, eps)
    else:
        y = layer_norm(x, weight, bias, eps)
    dy = torch.randn_like(y)
    return x, weight, bias, dy, y


def _generate_layernorm_forward_inputs(M, N, dtype, add_residual, quant):
    x, weight, bias, res_in = generate_layernorm_inputs(M, N, dtype, add_residual)
    res_out = torch.empty_like(x) if add_residual else None
    out = out_i8 = yscale = xscale = None
    if quant == "none":
        if add_residual:
            out = torch.empty_like(x)
    else:
        out_i8 = torch.empty((M, N), dtype=torch.int8, device="cuda")
        yscale = torch.empty((M, 1), dtype=torch.float32, device="cuda")
        xscale = torch.rand(N, device="cuda", dtype=torch.float32)
    return x, weight, bias, res_in, res_out, out, out_i8, yscale, xscale


def get_x_vals():
    x_vals = [
        # M scaling at the most common hidden size
        (1, 4096),
        (128, 4096),
        (1024, 4096),
        (4096, 4096),
        (16384, 4096),
        # model hidden sizes
        (8192, 512),
        (8192, 1536),
        (8192, 2880),
        (8192, 5120),
        (8192, 7168),
        (8192, 8192),
        # edge case in tester, odd widths, wide row
        (1, 4),
        (128, 2),
        (359, 1),
        (1, 131072),
        # BLOCK_SIZE caps at 65536//element_size; past it the kernel loops over N
        (256, 32768),
        (256, 65536),
    ]
    return x_vals


def run_benchmark(args):
    if args.backward and args.quant != "none":
        raise ValueError("--backward cannot be combined with --quant")
    if args.shape is not None and min(args.shape) <= 0:
        raise ValueError("--shape M N requires positive M and N")

    x_names = ["M", "N"]
    if args.shape is not None:
        x_vals_list = [list(args.shape)]
    else:
        x_vals_list = [list(shape) for shape in get_x_vals()]

    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    elif args.metric == "throughput":
        ylabel = "Throughput_(TFLOPS)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = [""]  # prevents double bandwidth text
    line_vals = [ylabel]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    quant = args.quant
    add_residual = args.add_residual
    do_backward = args.backward
    c_dtype = str_to_torch_dtype[args.dtype]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_layernorm(M, N, metric, **kwargs):
        eps = 1e-5

        if do_backward:
            # backward() builds its autograd graph on the default stream; let CUDA
            # graph capture (on its own side stream) adopt that stale reference.
            torch.autograd.graph.set_override_stale_capture_stream(True)
            _p_gen = functools.partial(
                _generate_layernorm_backward_inputs, M, N, c_dtype, add_residual
            )
            _out_val_keys = ["x", "weight", "bias", "dy", "y"]
            rot_bufs = generate_rotating_buffers_pool(
                [_p_gen], [_out_val_keys], **kwargs_rot_bufs
            )
            es = rot_bufs[0]["x"].element_size()

            def _run(x, weight, bias, dy, y):
                x.grad, weight.grad, bias.grad = None, None, None
                y.backward(dy, retain_graph=True)

            # x, dy in; dx, dw, db out
            mem_read = 2 * M * N * es + 2 * M * 4 + 2 * N * es
            mem_write = M * N * es + 2 * N * es
            mem = mem_read + mem_write
            flops = 12 * M * N

            ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
        else:
            _p_gen = functools.partial(
                _generate_layernorm_forward_inputs, M, N, c_dtype, add_residual, quant
            )
            _out_val_keys = [
                "x",
                "weight",
                "bias",
                "res_in",
                "res_out",
                "out",
                "out_i8",
                "yscale",
                "xscale",
            ]
            rot_bufs = generate_rotating_buffers_pool(
                [_p_gen], [_out_val_keys], **kwargs_rot_bufs
            )
            es = rot_bufs[0]["x"].element_size()

            if quant == "none":
                if add_residual:

                    def _run(x, weight, bias, res_in, res_out, out, **_):
                        layernorm2d_fwd_with_add(
                            out, x, res_in, res_out, weight, bias, eps
                        )

                else:

                    def _run(x, weight, bias, **_):
                        layer_norm(x, weight, bias, eps)

                out_bytes = M * N * es + 2 * M * 4  # y + mean/rstd
            else:
                if quant == "smooth" and add_residual:

                    def _run(
                        x, weight, bias, res_in, res_out, out_i8, yscale, xscale, **_
                    ):
                        layernorm2d_fwd_with_add_smoothquant(
                            out_i8,
                            x,
                            res_in,
                            res_out,
                            xscale,
                            yscale,
                            weight,
                            bias,
                            eps,
                        )

                elif quant == "smooth":

                    def _run(x, weight, bias, out_i8, yscale, xscale, **_):
                        layernorm2d_fwd_with_smoothquant(
                            out_i8, x, xscale, yscale, weight, bias, eps
                        )

                elif add_residual:

                    def _run(x, weight, bias, res_in, res_out, out_i8, yscale, **_):
                        layernorm2d_fwd_with_add_dynamicquant(
                            out_i8, x, res_in, res_out, yscale, weight, bias, eps
                        )

                else:

                    def _run(x, weight, bias, out_i8, yscale, **_):
                        layernorm2d_fwd_with_dynamicquant(
                            out_i8, x, yscale, weight, bias, eps
                        )

                out_bytes = M * N + M * 4  # int8 out + per-row scale

            mem_read = M * N * es + 2 * N * es
            mem_write = out_bytes
            if add_residual:
                mem_read += M * N * es
                mem_write += M * N * es
            if quant == "smooth":
                mem_read += N * 4
            mem = mem_read + mem_write

            flops = 8 * M * N
            if add_residual:
                flops += M * N
            if quant != "none":
                flops += 2 * M * N

            ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)

        if metric == "time":
            return ms
        elif metric == "bandwidth":
            return mem / (ms * 1e-3) * 1e-9  # GB/s
        elif metric == "throughput":
            return flops / ms * 1e-9  # TFLOP/s
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_layernorm.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="LayerNorm")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "dynamic", "smooth"],
        default="none",
        help="Fuse a quantization epilogue (int8 output, IS_SMOOTH selects the variant).",
    )
    parser.add_argument(
        "--add-residual",
        action="store_true",
        default=False,
        help="Fuse a residual add ahead of the norm.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        default=False,
        help="Benchmark the backward pass instead of forward. Cannot be used with --quant.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Input dtype.",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    override_parser_action(
        parser,
        action="--metric",
        default="bandwidth",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    args = parser.parse_args(args=args)
    return args


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
