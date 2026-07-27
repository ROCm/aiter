import torch
import triton
from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_mul_add import (
    fused_gemm_a8w8_blockscale_mul_add,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
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

block_shape = (128, 128)  # matches module-level `block_shape` in similar kernels

dimension = ["M", "N", "K"]
kernel_name = "Fused A8W8 Blockscale GEMM + mul_add"
kernel_label = "fused_gemm_a8w8_blockscale_mul_add"
shape = "(N=7168, K=256)"


def bench_fn(M: int, N: int, K: int, metric: str, **kwargs):
    """
    Single-shape timing of the fused A8W8 blockscale GEMM + mul_add epilogue via
    its public wrapper.

    This is a SINGLE-output (M, N) GEMM:
    ``out = a * (A@B-dequant) + b`` with ``fuse_type=0``. The default measured
    case uses full (M, N) tensor operands for both ``a`` (mul) and ``b`` (add),
    so the timed path exercises the two extra (M, N) epilogue reads.

    Output buffer is pre-allocated and passed in to keep allocation out of the
    timed ``do_bench`` window.
    """
    block_shape_n, block_shape_k = block_shape
    c_dtype = torch.bfloat16

    # GEMM inputs. ``output=True`` pre-allocates ``y`` of shape (M, N).
    x, w, _, x_scales, _, w_scales, y = generate_gemm_a8w8_blockscale_inputs(
        M,
        N,
        K,
        block_shape_n,
        block_shape_k,
        dtype=c_dtype,
        output=True,
    )
    # mul/add operands: full (M, N) tensors (default operand case).
    a, b = make_mul_add_ab(M, N, c_dtype, a_kind="tensor", b_kind="tensor")

    flops = 2.0 * M * N * K  # single fused output
    # bytes moved, summed as numel() * element_size() per tensor; the (M, N)
    # mul/add tensors and the output are all counted.
    mem = (
        x.numel() * x.element_size()
        + w.numel() * w.element_size()
        + x_scales.numel() * x_scales.element_size()
        + w_scales.numel() * w_scales.element_size()
        + (a.numel() * a.element_size() if isinstance(a, torch.Tensor) else 0)
        + (b.numel() * b.element_size() if isinstance(b, torch.Tensor) else 0)
        + y.numel() * y.element_size()
    )

    ms = triton.testing.do_bench(
        lambda: fused_gemm_a8w8_blockscale_mul_add(  # noqa: E731
            x,
            w,
            x_scales,
            w_scales,
            a,
            b,
            dtype=c_dtype,
            y=y,
            fuse_type=0,
        ),
        warmup=25,
        rep=100,
    )

    return metric_to_scalar(metric, ms, flops, mem)


def get_x_vals(args=None):
    """Default (M, N, K) benchmarking shapes for the fused mul_add kernel.

    This kernel routes through ``_get_config`` to the BASE
    ``GEMM-A8W8_BLOCKSCALE`` config family (not a dedicated fused config). The
    fixed family ``N=7168, K=256`` matches the kernel's unit test and the
    ``gfx950-GEMM-A8W8_BLOCKSCALE-N=7168-K=256.json`` specialized config, where ``-M``
    selects a single M.

    The default M sweep covers small-M buckets plus the ``any`` bucket
    """
    n, k = 7168, 256
    if args is not None and getattr(args, "M", None) is not None:
        m_vals = [args.M]
    else:
        m_vals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
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
