import torch
import triton
from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_mul_add import (
    fused_gemm_afp4wfp4_mul_add,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.bench_fused_gemm_commons import (
    metric_to_scalar,
    parse_fused_args,
    run_fused_benchmark,
    make_mul_add_ab,
)

dimension = ["M", "N", "K"]
kernel_name = "Fused AFP4WFP4 GEMM + mul_add"
kernel_label = "fused_gemm_afp4wfp4_mul_add"
shape = "(N=7168, K=256)"


def bench_fn(M: int, N: int, K: int, metric: str, **kwargs):
    """
    Single-shape timing of the fused AFP4WFP4 GEMM + mul_add via its
    public wrapper.

    This is a SINGLE-output (M, N) MXFP4 GEMM: ``out = a * (X @ W^T) + b`` with
    ``fuse_type=0``. The default measured case uses full (M, N) tensor operands
    for both ``a`` (mul) and ``b`` (add), so the timed path exercises the two
    extra (M, N) epilogue reads.

    Scope (intentional): the **non-preshuffled** FP4 path (``shuffle_*_fg=False``
    -> ``fused_gemm_afp4wfp4_mul_add``), which routes to the base
    ``GEMM-AFP4WFP4`` config family.

    Note on K: ``--shape`` K is the *logical* K (e.g. 256). The FP4 generator
    packs two e2m1 values per byte, so ``x_fp4`` has K//2 columns.

    Output buffer is pre-allocated and passed in to keep allocation out of the
    timed ``do_bench`` window.
    """
    c_dtype = torch.bfloat16

    # FP4 branch (non-preshuffled). ``output=True`` pre-allocates ``y_fp4``. With
    # shuffles off the ``_triton`` returns equal the plain ones; we pass the
    # ``_triton`` set to mirror the test's wrapper call exactly, so the plain
    # ``w`` / scale returns are unused here.
    (
        x_fp4,
        _,
        w_fp4_triton,
        _,
        _,
        x_fp4_scale_triton,
        w_fp4_scale_triton,
        _,
        y_fp4,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        c_dtype,
        layout="TN",
        output=True,
        shuffle_scales_fg=False,
        shuffle_weight_fg=False,
    )
    # mul/add operands: full (M, N) tensors (default operand case).
    a, b = make_mul_add_ab(M, N, c_dtype, a_kind="tensor", b_kind="tensor")

    flops = 2.0 * M * N * K  # single fused output
    # bytes moved, summed as numel() * element_size() per tensor; the (M, N)
    # mul/add tensors and the output are all counted.
    mem = (
        x_fp4.numel() * x_fp4.element_size()
        + w_fp4_triton.numel() * w_fp4_triton.element_size()
        + x_fp4_scale_triton.numel() * x_fp4_scale_triton.element_size()
        + w_fp4_scale_triton.numel() * w_fp4_scale_triton.element_size()
        + (a.numel() * a.element_size() if isinstance(a, torch.Tensor) else 0)
        + (b.numel() * b.element_size() if isinstance(b, torch.Tensor) else 0)
        + y_fp4.numel() * y_fp4.element_size()
    )

    ms = triton.testing.do_bench(
        lambda: fused_gemm_afp4wfp4_mul_add(  # noqa: E731
            x_fp4,
            w_fp4_triton,
            x_fp4_scale_triton,
            w_fp4_scale_triton,
            a,
            b,
            dtype=c_dtype,
            y=y_fp4,
            fuse_type=0,
        ),
        warmup=25,
        rep=100,
    )

    return metric_to_scalar(metric, ms, flops, mem)


def get_x_vals(args=None):
    """Default (M, N, K) benchmarking shapes for the fused mul_add kernel.

    This kernel routes through ``_get_config`` to the base ``GEMM-AFP4WFP4``
    config family. The fixed family ``N=7168, K=256`` matches the kernel's unit
    test and the ``gfx950-GEMM-AFP4WFP4-N=7168-K=256.json`` specialized config.
    ``-M`` selects a single M.

    The default M sweep covers small-M buckets plus the ``any`` bucket. M=1 is
    included here (the afp4wfp4 mul_add unit test exercises M=1 cleanly).
    """
    n, k = 7168, 256
    if args is not None and getattr(args, "M", None) is not None:
        m_vals = [args.M]
    else:
        m_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
    return [(m, n, k) for m in m_vals]


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_fused_args(kernel_name, args=args)
    plot_name = get_caller_name_no_ext()
    run = lambda: run_fused_benchmark(  # noqa: E731
        parsed_args,
        defaults,
        kernel_label,
        shape,
        dimension,
        bench_fn,
        get_x_vals,
        plot_name,
    )
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        print_vgpr(run, plot_name)
        return
    run()


if __name__ == "__main__":
    main()
