import functools
import math

import torch
import triton

from aiter import logger
from aiter.ops.triton.gemm.basic.gemm_a16w4 import gemm_a16w4
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
    get_model_benchmark_object,
    get_shape_benchmark_object,
    metric_value,
    output_check,
    print_vgpr,
    run_input_generator,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w4 import (
    TOLERANCES,
    generate_gemm_a16w4_inputs,
    run_torch,
)


def make_bench_case(
    kwargs_rot_bufs: dict,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    group_size: int,
    has_zeros: bool = False,
    act_order: bool = False,
):
    """(launch, rot_bufs, flops, mem, check) of one shape.

    A K that is not a multiple of group_size gets per-channel scales.
    ``launch(config, **bufs)`` and ``check(config)`` take a config; None is the installed one.
    The tuning scripts in aiter/ops/triton/utils/_triton/tuning use this function.
    """
    if group_size != -1 and K % group_size != 0:
        logger.warning(
            "K=%d is not a multiple of group size %d: using per-channel scales",
            K,
            group_size,
        )
        group_size = -1
    _p_gen = functools.partial(
        generate_gemm_a16w4_inputs,
        M,
        N,
        K,
        group_size=group_size,
        dtype=dtype,
        has_zeros=has_zeros,
        act_order=act_order,
        output=True,
    )
    # (x, w, w_scales, w_zeros, x_perm, bias, y, q)
    _out_vals_keys = ["x", "w", "w_scales", "w_zeros", "x_perm", "_", "y", "_"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    launch = config_launch(gemm_a16w4)

    bufs = rot_bufs[0]
    flops = 2.0 * M * N * K
    w_bytes = (
        bufs["w"].numel() + bufs["w_scales"].numel() * bufs["w_scales"].element_size()
    )
    if bufs["w_zeros"] is not None:
        w_bytes += bufs["w_zeros"].numel()
    mem = (M * K + M * N) * bufs["x"].element_size() + w_bytes

    def reference():
        values = _p_gen()
        x, _, w_scales, w_zeros, x_perm, _, _, q = values
        expected = run_torch(x, q, w_scales, w_zeros, x_perm, dtype=torch.float32)
        return run_input_generator(lambda: values, _out_vals_keys), expected

    atol, rtol = TOLERANCES[dtype]
    check = output_check(launch, reference, "y", atol=atol, rtol=rtol)
    return launch, rot_bufs, flops, mem, check


def bench_gemm_fn(M: int, N: int, K: int, metric: str, args):
    launch, rot_bufs, flops, mem, _ = make_bench_case(
        get_rot_bufs_kwargs(args),
        M,
        N,
        K,
        torch.float16 if args.dtype == "fp16" else torch.bfloat16,
        args.group_size,
        has_zeros=args.has_zeros,
        act_order=args.act_order,
    )
    ms = do_bench_aiter_triton(
        functools.partial(launch, None), rot_bufs, **get_do_bench_graph_kwargs(args)
    )
    return metric_value(metric, ms, flops, mem)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w4(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        """
        fc1: (M, hidden_dim) x (hidden_dim, 2 * intermediate_dim) with GLU.
        fc2: (M, intermediate_dim) x (intermediate_dim, hidden_dim).
        Tensor parallelism divides intermediate_dim (N of fc1, K of fc2).
        """
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(M, N, K, metric, args)

    bench_gemm_a16w4.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w4(M, N, K, metric, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args)

    bench_gemm_a16w4.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        run_model_benchmark(args)
    else:
        unsupported_args = ["fc1", "fc2", "no_glu"]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A16W4 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Activation and output datatype.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size along K. Use -1 for per-channel scales.",
    )
    parser.add_argument(
        "--has-zeros",
        action="store_true",
        help="Use zero points (asymmetric quantization, as in AWQ).",
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Use an act-order permutation of K (GPTQ desc_act).",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
