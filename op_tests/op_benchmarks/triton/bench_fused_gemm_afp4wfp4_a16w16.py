import sys
import torch
import triton
from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 import (
    fused_gemm_afp4wfp4_a16w16,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)


def bench_fn(M: int, N4: int, N16: int, K: int, metric: str, **kwargs):
    """
    Single-shape timing of the fused AFP4WFP4 + A16W16 kernel via its public wrapper.

    Scope (intentional): this benchmark exercises the **non-preshuffled, no-bias**
    FP4 path (``is_fp4_preshuffled=False``, ``bias_fp4=None``, ``bias_bf16=None``),
    which is the path that maps to the shape-scoped config
    ``gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json``. It deliberately
    does NOT cover the wrapper's default preshuffled path
    (``is_fp4_preshuffled=True`` -> ``FUSED-GEMM-AFP4WFP4_PRESHUFFLED-A16W16``) nor
    the biased BF16 gating variant.

    Note on K: ``--shape`` K is the *logical* K (e.g. 7168). The FP4 generator packs
    two e2m1 values per byte, so ``x_fp4`` has K//2 columns; the wrapper's
    ``_get_config`` rebuilds the config filename with ``2*K`` (= logical K), so
    passing 7168 here is what selects the shape-scoped config file. Passing the
    packed value (3584) would fall back to the generic ``any`` config.

    Output buffers are pre-allocated and passed in to keep allocation out of the
    timed ``do_bench`` window.
    """
    c_dtype = torch.bfloat16

    # FP4 branch (non-preshuffled). ``output=True`` pre-allocates ``y_fp4``. With
    # shuffles off the ``_triton`` returns equal the plain ones; we pass the
    # ``_triton`` set to mirror the test's wrapper call exactly.
    (
        x_fp4,
        w_fp4,
        w_fp4_triton,
        x_fp4_scale,
        w_fp4_scale,
        x_fp4_scale_triton,
        w_fp4_scale_triton,
        _out_dtype,
        y_fp4,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N4,
        K,
        c_dtype,
        layout="TN",
        output=True,
        shuffle_scales_fg=False,
        shuffle_weight_fg=False,
    )
    # BF16 branch inputs (N16 = N_bf16). Same M, same logical K, and no bias.
    x_bf16, w_bf16, _, _, y_bf16 = generate_gemm_a16w16_inputs(
        M,
        N16,
        K,
        c_dtype,
        output=True,
        bias=False,
    )
    # flops
    flops = 2.0 * M * (N4 + N16) * K  # summed across the two fused outputs
    # memory transfer: numel * element_size() per tensor keeps the formula
    # correct if dtypes change later (e.g. fp16 outputs).
    mem = (
        x_fp4.numel() * x_fp4.element_size()
        + w_fp4_triton.numel() * w_fp4_triton.element_size()
        + x_fp4_scale_triton.numel() * x_fp4_scale_triton.element_size()
        + w_fp4_scale_triton.numel() * w_fp4_scale_triton.element_size()
        + x_bf16.numel() * x_bf16.element_size()
        + w_bf16.numel() * w_bf16.element_size()
        + y_fp4.numel() * y_fp4.element_size()
        + y_bf16.numel() * y_bf16.element_size()
    )

    ms = triton.testing.do_bench(
        lambda: fused_gemm_afp4wfp4_a16w16(  # noqa: E731
            x_fp4,
            w_fp4_triton,
            x_fp4_scale_triton,
            w_fp4_scale_triton,
            x_bf16,
            w_bf16,
            is_fp4_preshuffled=False,
            dtype=c_dtype,
            y_fp4=y_fp4,
            y_bf16=y_bf16,
        ),
        warmup=25,
        rep=100,
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


def get_x_vals(args=None):
    """Default (M, N4, N16, K) benchmarking shapes for the fused kernel.

    Specialized to the single shape family this fused op is tuned for: the
    ``gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168`` config. N4, N16
    and K are fixed and only M is swept. As in the shared helper, ``-M`` selects
    a single M.

    The default M sweep hits every explicit bucket of that config
    (M_LEQ_{8,16,32,64,128,256}) plus the ``any`` bucket sampled at
    M in {1024, 4096, 8192}. (This config has no M_LEQ_1024/2048 buckets, so all
    M > 256 resolve to ``any``.)
    """
    n4, n16, k = 512, 256, 7168
    if args is not None and getattr(args, "M", None) is not None:
        m_vals = [args.M]
    else:
        m_vals = [1, 8, 16, 32, 64, 128, 256, 1024, 4096, 8192]
    return [(m, n4, n16, k) for m in m_vals]


def get_shape_benchmark_object(plot_name, args, x_names=None):
    """Build the Benchmark object for the (M, N4, N16, K) shape sweep.

    There are two independent output-N dims (N4 for the FP4 branch, N16 for the
    BF16 branch) sharing M and K, so a 4-element ``--shape`` is (M, N4, N16, K)
    here, NOT the batched (B, M, N, K) that the shared helper / ``get_ff_args``
    assume.
    """
    if x_names is None:
        x_names = ["M", "N4", "N16", "K"]

    if args.shape:
        # enforce the fused 4-tuple here.
        if len(args.shape) != 4:
            raise ValueError(
                f"--shape expects 4 ints (M N4 N16 K); "
                f"got {len(args.shape)}: {args.shape}"
            )
        x_vals_list = [args.shape]
    else:
        x_vals_list = get_x_vals(args=args)

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    evaluation_metric_to_unit = {
        "throughput": "TFLOPS",
        "time": "Time_(ms)",
        "bandwidth": "Bandwidth_(GB/s)",
    }
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="unit",
        line_vals=[evaluation_metric_to_unit[args.metric]],
        line_names=[evaluation_metric_to_unit[args.metric]],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=plot_name,
        args={"metric": args.metric},
    )
    return benchmark


def run_shape_benchmark(args):
    """Runs a benchmark with given tensor shapes."""
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_fused_gemm_afp4wfp4_a16w16(M, N4, N16, K, metric, **kwargs):
        return bench_fn(M, N4, N16, K, metric)

    bench_fused_gemm_afp4wfp4_a16w16.run(
        save_path="." if args.o else None, print_data=True
    )


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    # --model has no meaning here: the op is tuned for one fixed shape family
    # (N4=512, N16=256, K=7168), not model-derived hidden/intermediate dims.
    if args.model:  # TODO: add in --model argument (via run_model_benchmark())
        raise NotImplementedError(
            "--model is not supported for fused_gemm_afp4wfp4_a16w16; "
            "it targets a single fixed shape family (N4=512, N16=256, K=7168). "
            "Use --shape M N4 N16 K or -M."
        )
    else:
        unsupported_args = ["fc1", "fc2", "no_glu", "tp", "layout"]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for "
                    f"fused_gemm_afp4wfp4_a16w16."
                )
        run_shape_benchmark(args)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="Fused AFP4WFP4 + A16W16 GEMM")
    parser = add_argparse_ff(parser)
    # get_ff_args destructures a 4-element --shape as (B, M, N, K); for this op
    # --shape is (M, N4, N16, K), so the shape path reads args.shape directly
    # and ignores that (B, M, N, K) mapping.
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    sys.exit(main())
