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
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.bench_fused_gemm_commons import (
    metric_to_scalar,
    parse_fused_args,
    run_fused_benchmark,
)

dimension = ["M", "N4", "N16", "K"]
kernel_name = "Fused AFP4WFP4 + A16W16 GEMM"
kernel_label = "fused_gemm_afp4wfp4_a16w16"
shape = "(N4=512, N16=256, K=7168)"


def bench_fn(M: int, N4: int, N16: int, K: int, metric: str, **kwargs):
    """
    Single-shape timing of the fused AFP4WFP4 + A16W16 kernel via its public wrapper.

    Scope (intentional for tuning): exercises the
    **non-preshuffled, no-bias** FP4 path, which maps to the shape-scoped config
    ``gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json``.

    Note on K: ``--shape`` K is the logical K (e.g. 7168). The FP4 generator
    packs two e2m1 values per byte, so ``x_fp4`` has K//2 columns.

    Output buffers are pre-allocated and passed in to keep allocation out of the
    timed ``do_bench`` window.
    """
    c_dtype = torch.bfloat16

    # FP4 branch (non-preshuffled). ``output=True`` pre-allocates ``y_fp4``. With
    # shuffles off the ``_triton`` returns equal the plain ones; we pass the
    # ``_triton`` set to mirror the test's wrapper call exactly.
    # Therefore ``w_fp4`` / scale returns are unused here.
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
    flops = 2.0 * M * (N4 + N16) * K  # summed across the two fused outputs
    # bytes moved, summed as numel() * element_size() per tensor.
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

    return metric_to_scalar(metric, ms, flops, mem)


def get_x_vals(args=None):
    """Default (M, N4, N16, K) benchmarking shapes for the fused kernel.

    Specialized to ``gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168``.
    N4, N16 and K are fixed and only M is swept, where ``-M`` selects a single M.

    The default M sweep hits every explicit bucket of that config
    (M_LEQ_{8,16,32,64,128,256}) plus the ``any`` bucket sampled at
    M in {1024, 4096, 8192}.
    """
    n4, n16, k = 512, 256, 7168
    if args is not None and getattr(args, "M", None) is not None:
        m_vals = [args.M]
    else:
        m_vals = [1, 8, 16, 32, 64, 128, 256, 1024, 4096, 8192]
    return [(m, n4, n16, k) for m in m_vals]


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
