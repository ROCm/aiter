import functools
import math

import triton

from aiter.ops.triton.gemm.basic.gemm_a16w32 import gemm_a16w32
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    get_model_configs,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w32 import (
    generate_gemm_a16w32_inputs,
)


def _gen_inputs(M, N, K, keys):
    names = ("x", "w", "w_bf16x3", "y")
    inputs = dict(zip(names, generate_gemm_a16w32_inputs(M, N, K, output=True)))
    return tuple(inputs[key] for key in keys)


def bench_gemm_fn(
    bencher: RotatingBufferBench, M: int, N: int, K: int, metric: str, w_split: bool
):
    # The fp32 FMA kernel reads w. The bf16x3 kernel reads w and splits it, or
    # it reads the bf16 terms when the call gives them (w_split). The weight
    # that the kernel reads rotates. The other weight is fixed.
    config, _ = get_gemm_config("GEMM-A16W32", M, N, K)
    read = "w"
    if w_split and config["kernel_type"] == "compute_bound":
        read = "w_bf16x3"
    rot_keys = ["x", read, "y"]
    fixed_keys = [name for name in ("w", "w_bf16x3") if name != read]
    if not w_split:
        fixed_keys.remove("w_bf16x3")
    fixed = {}
    if fixed_keys:
        fixed = {
            "inp_gen_fixed_fns": [functools.partial(_gen_inputs, M, N, K, fixed_keys)],
            "inp_fixed_keys": [fixed_keys],
        }
    rot_bufs = bencher.build_pool(
        [functools.partial(_gen_inputs, M, N, K, rot_keys)], [rot_keys], **fixed
    )

    flops = 2.0 * M * N * K
    # The bytes of x and of the weight that the kernel reads, and of y.
    weight = rot_bufs[0][read]
    mem_read = (M * K) * rot_bufs[0]["x"].element_size()
    mem_read += weight.numel() * weight.element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    ms = bencher.bench(gemm_a16w32)

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    else:
        raise ValueError("Unknown metric: " + metric)


def model_benchmark_shapes(args):
    """(model_name, M, N, K) of the MoE router GEMM: N is num_expert, K is hidden_size.

    The router weight is not split over tensor-parallel ranks, so -tp does not apply.
    """
    configs = get_model_configs(config_path=args.model_configs, models=args.model)
    configs = {name: c for name, c in configs.items() if "num_expert" in c}
    if not configs:
        raise ValueError("no MoE model (with num_expert) in --model " + args.model)
    M_list = [args.M] if args.M is not None else [2**i for i in range(14)]
    return [
        (name, M, config["num_expert"], config["hidden_size"])
        for M in M_list
        for name, config in configs.items()
    ]


def run_benchmark(args):
    assert not (args.shape and args.model), "Specify --shape or --model, not both"
    if args.model:
        benchmark = triton.testing.Benchmark(
            x_names=["model_name", "M", "N", "K"],
            x_vals=model_benchmark_shapes(args),
            line_arg="unit",
            line_vals=[args.metric],
            line_names=[args.metric],
            styles=[("green", "-")],
            ylabel=args.metric,
            plot_name=get_caller_name_no_ext(),
            args={"metric": args.metric},
        )
    else:
        benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, K, metric, model_name=None, **kwargs):
        if model_name is None:
            N = math.ceil(N / args.tp)
        return bench_gemm_fn(bencher, M, N, K, metric, args.w_split)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args(args=None):
    parser = get_parser(kernel_name="A16W32 GEMM (bf16 activations, fp32 weights)")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--w-split",
        action="store_true",
        help="Also give the three bf16 terms of the weight, from split_bf16x3",
    )
    return get_ff_args(parser, args=args)


def main(args=None):
    parsed_args, _ = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
