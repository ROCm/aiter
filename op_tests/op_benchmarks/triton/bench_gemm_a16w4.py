import math

import torch
import triton

from aiter import logger
from aiter.ops.triton.gemm.basic.gemm_a16w4 import gemm_a16w4
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w4 import (
    generate_gemm_a16w4_inputs,
)


def bench_gemm_fn(M: int, N: int, K: int, metric: str, args):
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    group_size = args.group_size
    if group_size != -1 and K % group_size != 0:
        logger.warning(
            "K=%d is not a multiple of group size %d: using per-channel scales",
            K,
            group_size,
        )
        group_size = -1
    x, w, w_scales, w_zeros, x_perm, _, y, _ = generate_gemm_a16w4_inputs(
        M,
        N,
        K,
        group_size=group_size,
        dtype=dtype,
        has_zeros=args.has_zeros,
        act_order=args.act_order,
        output=True,
    )

    flops = 2.0 * M * N * K
    w_bytes = w.numel() + w_scales.numel() * w_scales.element_size()
    if w_zeros is not None:
        w_bytes += w_zeros.numel()
    mem = (M * K + M * N) * x.element_size() + w_bytes

    # Copies of the weights, so that each call reads its weights from memory.
    num_copies = max(1, int(args.rotate_mb * 2**20) // w_bytes)
    weights = [(w, w_scales, w_zeros)]
    weights += [
        (w.clone(), w_scales.clone(), None if w_zeros is None else w_zeros.clone())
        for _ in range(num_copies - 1)
    ]
    calls = [0]

    def fn():
        wi, si, zi = weights[calls[0] % len(weights)]
        calls[0] += 1
        return gemm_a16w4(x, wi, si, w_zeros=zi, x_perm=x_perm, y=y)

    ms = triton.testing.do_bench_cudagraph(fn)

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    else:
        raise ValueError("Unknown metric: " + metric)


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
    parser.add_argument(
        "--rotate-mb",
        type=float,
        default=0,
        help="Rotate over weight copies of this total size in MB, so that the "
        "weights do not stay in the caches. Use it for small M.",
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
