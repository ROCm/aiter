import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w16 import _is_gluon_available, gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
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
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    backend: str,
    atomic: bool = False,
    activation: str | None = None,
    persistent: bool = False,
    kernel_type: str = "bandwidth_bound",
    config: dict | None = None,
    **kwargs,
):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = torch.bfloat16
    x, w, bias, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, c_dtype, layout=layout, output=True, bias=True
    )
    # flops
    flops = 2.0 * M * N * K
    if activation is not None:
        flops += M * N  # elementwise ops on the GEMM output
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
    mem_write = (M * N) * x.element_size()
    mem = mem_read + mem_write

    bench_fn = triton.testing.do_bench_cudagraph

    if atomic:
        # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
        assert backend != "gluon", "Atomic kernel is triton-only"
        assert (
            activation is None
        ), "Atomic kernel does not currently support fused activation"
        y = y.to(torch.float32).zero_()
        ms = bench_fn(
            lambda: gemm_a16w16_atomic(x, w, torch.float32, y),
        )
    elif persistent:
        assert not (backend == "gluon" and layout not in ("TN", "TT")), (
            f"--persistent with --backend gluon requires --layout TN or TT, got "
            f"'{layout}' (the gluon persistent kernel has no 'N' path for A)"
        )
        ms = bench_fn(
            lambda: gemm_a16w16(
                x,
                w,
                bias,
                c_dtype,
                y,
                activation=activation,
                kernel_type=kernel_type,
                backend=backend,
                persistent=True,
            ),
        )
    else:
        ms = bench_fn(
            lambda: gemm_a16w16(
                x,
                w,
                bias,
                c_dtype,
                y,
                activation=activation,
                kernel_type=kernel_type,
                backend=backend,
                config=config,
            ),
        )

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


def run_model_benchmark(args, backend):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        """
        Fc1:
             M      K                  K           N          M       N
        A = (B, hidden_dim) @ W = (hidden_dim, 2*int_dim) -> (B, 2*int_dim) -> gating -> (B, int_dim)

        Fc2:
             M     K               K          N          M       N
        A = (B, int_dim) @ W = (int_dim, hidden_dim) -> (B, hidden_dim)

        Tensor parallel splits across int_dim (N for fc1, K for fc2)
        """
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
        # print(f"Layer: {layer}, M: {M}, N: {N}, K: {K}, hidden_dim: {hidden_dim}, intermediate_dim: {intermediate_dim}")

        return bench_gemm_fn(
            M,
            N,
            K,
            metric,
            args.layout,
            backend,
            atomic=args.atomic,
            activation=args.activation,
            persistent=args.persistent,
            kernel_type=args.kernel_type,
            config=streamk_config(args),
        )

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, backend):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, N, K, metric, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            M,
            N,
            K,
            metric,
            args.layout,
            backend,
            atomic=args.atomic,
            persistent=args.persistent,
            kernel_type=args.kernel_type,
            config=streamk_config(args),
        )

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def streamk_config(args) -> dict | None:
    """The config --streamk-config names: a Stream-K kernelName
    (gluon_streamk_...) or a tile BLOCK_MxBLOCK_NxBLOCK_K. None runs the kernel's
    default tile and plan."""
    if args.streamk_config is None:
        return None
    assert (
        args.kernel_type == "streamk"
    ), "--streamk-config applies to --kernel-type streamk only"
    if args.streamk_config.startswith("gluon_streamk_"):
        from aiter.tuned_gemm import parse_gluon_kernel_name

        return parse_gluon_kernel_name(args.streamk_config)[1]
    bm, bn, bk = (int(side) for side in args.streamk_config.split("x"))
    return {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    backend = args.backend or ("gluon" if _is_gluon_available() else "triton")
    print(f"Using backend: {backend}")

    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args, backend)
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
        run_shape_benchmark(args, backend)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A16W16 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="Use the atomic kernel (split-k with atomic_add) instead of the standard a16w16 kernel.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function to apply to the output. One of ('gelu', 'gelu_tanh', 'silu', 'silu_exp2', 'relu').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "gluon"],
        default=None,
        help="Backend to use. Default: auto-detect (gluon on gfx1250, triton elsewhere).",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        default=False,
        help="Use the persistent kernel (gemm_a16w16(..., persistent=True)) instead of "
        "the standard a16w16 kernel",
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=["bandwidth_bound", "compute_bound", "streamk"],
        default="bandwidth_bound",
        help="Kernel variant to use (gluon only; streamk on gfx950). Default: bandwidth_bound.",
    )
    parser.add_argument(
        "--streamk-config",
        type=str,
        default=None,
        help="With --kernel-type streamk: a tuned kernelName "
        "(e.g. gluon_streamk_256x256x64_g4_one_tile_rs_k8_wt) or a tile such as "
        "128x256x64 (the default plan). Default: 256x256x64 with the default plan.",
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
