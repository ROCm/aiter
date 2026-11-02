import functools

import torch

from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_mul_add import (
    fused_gemm_a8w8_blockscale_mul_add,
)
from op_tests.op_benchmarks.triton.bench_fused_gemm_commons import (
    make_mul_add_ab,
    metric_to_scalar,
    parse_fused_args,
    run_fused_benchmark,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
)

block_shape = (128, 128)  # matches module-level `block_shape` in similar kernels

dimension = ["M", "N", "K"]
kernel_name = "Fused A8W8 Blockscale GEMM + mul_add"
kernel_label = "fused_gemm_a8w8_blockscale_mul_add"
shape = "(N=7168, K=256)"


def bench_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    **kwargs,
):
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
    _p_gen_gemm = functools.partial(
        generate_gemm_a8w8_blockscale_inputs,
        M,
        N,
        K,
        block_shape_n,
        block_shape_k,
        dtype=c_dtype,
        output=True,
    )
    # (x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y)
    _out_vals_keys_gemm = ["x", "w", "_", "x_scales", "_", "w_scales", "y"]

    # mul/add operands: full (M, N) tensors (default operand case).
    _p_gen_mul_add = functools.partial(
        make_mul_add_ab, M, N, c_dtype, a_kind="tensor", b_kind="tensor"
    )
    _out_vals_keys_mul_add = ["a", "b"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen_gemm, _p_gen_mul_add],
        [_out_vals_keys_gemm, _out_vals_keys_mul_add],
        **kwargs_rot_bufs,
    )

    kernel_fn = functools.partial(
        fused_gemm_a8w8_blockscale_mul_add, dtype=c_dtype, fuse_type=0
    )

    flops = 2.0 * M * N * K  # single fused output
    # bytes moved, derived from M/N/K; the blockscale scale tensors keep their
    # own numel (one scale per block, which doesn't reduce to a clean M/N/K
    # product). The (M, N) mul/add tensors and the output are all counted.
    buf0 = rot_bufs[0]
    a, b = buf0["a"], buf0["b"]
    mem = (
        (M * K) * buf0["x"].element_size()
        + (N * K) * buf0["w"].element_size()
        + buf0["x_scales"].numel() * buf0["x_scales"].element_size()
        + buf0["w_scales"].numel() * buf0["w_scales"].element_size()
        + ((M * N) * a.element_size() if isinstance(a, torch.Tensor) else 0)
        + ((M * N) * b.element_size() if isinstance(b, torch.Tensor) else 0)
        + (M * N) * buf0["y"].element_size()
    )

    ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

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
    run = lambda: run_fused_benchmark(
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
