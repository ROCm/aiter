import torch
import triton
from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_a16w16 import (
    fused_gemm_a8w8_blockscale_a16w16,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
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

block_shape = (128, 128)  # matches module-level `block_shape` in similar kernels

dimension = ["M", "N8", "N16", "K"]
kernel_name = "Fused A8W8 Blockscale + A16W16 GEMM"
kernel_label = "fused_gemm_a8w8_blockscale_a16w16"
shape = "(N8=512, N16=256, K=7168)"


def bench_fn(M: int, N8: int, N16: int, K: int, metric: str, **kwargs):
    """
    Single-shape timing of the fused kernel via its public wrapper.
    Output buffers are pre-allocated and passed in to keep allocation out of the
    timed ``do_bench`` window.
    """
    block_shape_n, block_shape_k = block_shape
    c_dtype = torch.bfloat16

    # FP8 branch inputs (N8 = N_fp8). ``output=True`` pre-allocates ``y_fp8``
    x_fp8, w_fp8, _, x_fp8_scale, _, w_fp8_scale, y_fp8 = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N8,
            K,
            block_shape_n,
            block_shape_k,
            dtype=c_dtype,
            output=True,
        )
    )
    # BF16 branch inputs (N16 = N_bf16). Same M, same K, and no bias
    x_bf16, w_bf16, _, _, y_bf16 = generate_gemm_a16w16_inputs(
        M,
        N16,
        K,
        c_dtype,
        output=True,
        bias=False,
    )
    flops = 2.0 * M * (N8 + N16) * K  # summed across the two fused outputs
    # bytes moved, summed as numel() * element_size() per tensor
    mem = (
        x_fp8.numel() * x_fp8.element_size()
        + w_fp8.numel() * w_fp8.element_size()
        + x_fp8_scale.numel() * x_fp8_scale.element_size()
        + w_fp8_scale.numel() * w_fp8_scale.element_size()
        + x_bf16.numel() * x_bf16.element_size()
        + w_bf16.numel() * w_bf16.element_size()
        + y_fp8.numel() * y_fp8.element_size()
        + y_bf16.numel() * y_bf16.element_size()
    )

    ms = triton.testing.do_bench(
        lambda: fused_gemm_a8w8_blockscale_a16w16(  # noqa: E731
            x_fp8,
            w_fp8,
            x_fp8_scale,
            w_fp8_scale,
            x_bf16,
            w_bf16,
            dtype=c_dtype,
            y_fp8=y_fp8,
            y_bf16=y_bf16,
        ),
        warmup=25,
        rep=100,
    )

    return metric_to_scalar(metric, ms, flops, mem)


def get_x_vals(args=None):
    """Default (M, N8, N16, K) benchmarking shapes for the fused kernel.

    Specialized to the single shape family that this fused op is tuned for: the
    ``gfx950-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168`` config.
    N8, N16 and K are fixed and only M is swept, where ``-M`` selects a single M.

    The default M sweep hits every bucket of that dedicated config file -
    (M_LEQ_{8,16,32,64,128,256,1024,2048}) plus the ``any`` bucket (M=4096).
    """
    n8, n16, k = 512, 256, 7168
    if args is not None and getattr(args, "M", None) is not None:
        m_vals = [args.M]
    else:
        m_vals = [1, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096]
    return [(m, n8, n16, k) for m in m_vals]


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
