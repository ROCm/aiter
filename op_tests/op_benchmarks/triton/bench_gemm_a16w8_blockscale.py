import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
    gemm_a16w8_blockscale,
    gemm_a16w8_blockscale_preshuffle,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    config_launch,
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    get_shape_benchmark_object,
    metric_value,
    output_check,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
    generate_gemm_a16w8_blockscale_inputs,
    run_torch,
)

block_shape = (128, 128)


def make_bench_case(
    kwargs_rot_bufs: dict,
    M: int,
    N: int,
    K: int,
    layout: str,
    preshuffle: bool = False,
):
    """(launch, rot_bufs, flops, mem, check) of one shape.

    ``launch(config, **bufs)`` and ``check(config)`` take a config; None is the installed one.
    The tuning scripts in aiter/ops/triton/utils/_triton/tuning use this function.
    """
    block_shape_n, block_shape_k = block_shape
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(
        generate_gemm_a16w8_blockscale_inputs,
        M,
        N,
        K,
        block_shape_n,
        block_shape_k,
        dtype=c_dtype,
        layout=layout,
        output=True,
        shuffle=preshuffle,
    )
    # (x, weight, weight_shuffled, w_scale, y); keep only the weight variant in use
    if preshuffle:
        _out_vals_keys = ["x", "_", "w", "w_scale", "y"]
        kernel_fn = functools.partial(
            gemm_a16w8_blockscale_preshuffle, dtype=c_dtype, prequant=False
        )
    else:
        _out_vals_keys = ["x", "w", "_", "w_scale", "y"]
        kernel_fn = functools.partial(
            gemm_a16w8_blockscale, dtype=c_dtype, prequant=False
        )
    launch = config_launch(kernel_fn)

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    flops = 2.0 * M * N * K
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    def reference():
        x, weight, weight_shuffled, w_scale, y = _p_gen()
        bufs = {
            "x": x,
            "w": weight_shuffled if preshuffle else weight,
            "w_scale": w_scale,
            "y": y,
        }
        return bufs, run_torch(x, weight, w_scale, torch.float32)

    check = output_check(launch, reference, "y", atol=0.1, rtol=0.1)
    return launch, rot_bufs, flops, mem, check


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    preshuffle: bool = False,
):
    launch, rot_bufs, flops, mem, _ = make_bench_case(
        kwargs_rot_bufs, M, N, K, layout, preshuffle=preshuffle
    )
    ms = do_bench_aiter_triton(
        functools.partial(launch, None), rot_bufs, **kwargs_do_bench_graph
    )
    return metric_value(metric, ms, flops, mem)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, K, metric, model_name=None, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            M,
            N,
            K,
            metric,
            args.layout,
            preshuffle=args.preshuffle,
        )

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    run_shape_benchmark(args)


def parse_args(args=None):
    parser = get_parser(kernel_name="A16W8 Blockscale GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-preshuffle",
        action="store_true",
        help="Use preshuffle implementation",
    )
    return get_ff_args(parser, args=args)


def main(args=None):
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
