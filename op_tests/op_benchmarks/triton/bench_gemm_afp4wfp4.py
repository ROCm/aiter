import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4 as triton_gemm_afp4wfp4,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    preshuffle: bool,
    gluon: bool,
):
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(
        generate_gemm_afp4wfp4_inputs,
        M,
        N,
        K,
        c_dtype,
        layout=layout,
        output=True,
        shuffle_scales_fg=preshuffle,
        shuffle_weight_fg=preshuffle,
    )
    # (x, w, w_shuffed, x_scales, w_scales, x_scales_shuffled, w_scales_shuffled, out_dtype, y)
    # gemm_afp4wfp4_preshuffle names its activation/weight kwargs x_fp4/w_preshuf;
    # the non-preshuffle kernel uses x/w. w_shuffed/*_shuffled already collapse to the
    # unshuffled tensors when preshuffle is False, so one mapping covers both kernels.
    x_key = "x_fp4" if preshuffle else "x"
    w_key = "w_preshuf" if preshuffle else "w"
    _out_vals_keys = [x_key, "_", w_key, "_", "_", "x_scales", "w_scales", "_", "y"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    if preshuffle:
        kernel_fn = functools.partial(gemm_afp4wfp4_preshuffle, dtype=c_dtype)
    elif gluon:
        kernel_fn = functools.partial(
            triton_gemm_afp4wfp4, dtype=c_dtype, backend="gluon"
        )
    else:
        kernel_fn = functools.partial(triton_gemm_afp4wfp4, dtype=c_dtype)

    # flops
    flops = 2.0 * M * N * K
    # memory transfer. x/w are e2m1, packed two values per byte, so the byte
    # count is K // 2 wide; the mx scale tensors keep their own numel (one
    # scale per 32 K-elements, which doesn't reduce to a clean M/N/K product).
    mem_read = (M * K // 2) * rot_bufs[0][x_key].element_size() + (
        N * K // 2
    ) * rot_bufs[0][w_key].element_size()
    mem_read += (
        rot_bufs[0]["x_scales"].numel() * rot_bufs[0]["x_scales"].element_size()
        + rot_bufs[0]["w_scales"].numel() * rot_bufs[0]["w_scales"].element_size()
    )
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
    # Return exactly one scalar depending on which metric is active
    if metric == "time":
        return ms
    elif metric == "throughput":
        tflops = flops / ms * 1e-9
        return tflops
    elif metric == "bandwidth":
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        return bandwidth
    else:
        raise ValueError("Unknown metric: " + metric)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object("GEMM MXFP4 x MXFP4 Benchmark", args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4(
        M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
    ):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            # Divide N by tensor parallel
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            # Divide K by tensor parallel
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            M,
            N,
            K,
            metric,
            args.layout,
            args.preshuffle,
            args.gluon,
        )

    bench_gemm_afp4wfp4.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object("GEMM MXFP4 x MXFP4 Benchmark", args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4(M, N, K, metric, model_name=None, **kwargs):
        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            M,
            N,
            K,
            metric,
            args.layout,
            args.preshuffle,
            args.gluon,
        )

    bench_gemm_afp4wfp4.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = get_parser("MXFP4 x MXFP4 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--shuffle",
        "--preshuffle",
        action="store_true",
        dest="preshuffle",
        help="Preshuffle weight and scales (gfx1250: Gluon gemm_mxfp4_preshuffle_gfx1250; else Triton).",
    )
    parser.add_argument(
        "--gluon",
        action="store_true",
        help="Use Gluon non-preshuffle kernel (ignored if --shuffle/--preshuffle is set).",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    assert arch_info.is_fp4_avail(), "MXFP4 is not available on this architecture"

    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, "GEMM")
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
