import functools

import torch

from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_mul_add import (
    fused_gemm_afp4wfp4_mul_add,
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
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)

dimension = ["M", "N", "K"]
kernel_name = "Fused AFP4WFP4 GEMM + mul_add"
kernel_label = "fused_gemm_afp4wfp4_mul_add"
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

    # FP4 branch (non-preshuffled). ``output=True`` pre-allocates ``y``. With
    # shuffles off the shuffled returns equal the plain ones, so the plain
    # ``w``/unshuffled-scale returns are dropped ("_") here.
    _p_gen_fp4 = functools.partial(
        generate_gemm_afp4wfp4_inputs,
        M,
        N,
        K,
        c_dtype,
        layout="TN",
        output=True,
        shuffle_scales_fg=False,
        shuffle_weight_fg=False,
    )
    # (x, w, w_shuffed, x_scales, w_scales, x_scales_shuffled, w_scales_shuffled, out_dtype, y)
    _out_vals_keys_fp4 = ["x", "_", "w", "_", "_", "x_scales", "w_scales", "_", "y"]

    # mul/add operands: full (M, N) tensors (default operand case).
    _p_gen_mul_add = functools.partial(
        make_mul_add_ab, M, N, c_dtype, a_kind="tensor", b_kind="tensor"
    )
    _out_vals_keys_mul_add = ["a", "b"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen_fp4, _p_gen_mul_add],
        [_out_vals_keys_fp4, _out_vals_keys_mul_add],
        **kwargs_rot_bufs,
    )

    kernel_fn = functools.partial(
        fused_gemm_afp4wfp4_mul_add, dtype=c_dtype, fuse_type=0
    )

    flops = 2.0 * M * N * K  # single fused output
    # bytes moved, derived from M/N/K. x/w are e2m1, packed two values per
    # byte (K // 2 wide); the mx scale tensors keep their own numel (one scale
    # per 32 K-elements, which doesn't reduce to a clean M/N/K product). The
    # (M, N) mul/add tensors and the output are all counted.
    buf0 = rot_bufs[0]
    a, b = buf0["a"], buf0["b"]
    mem = (
        (M * K // 2) * buf0["x"].element_size()
        + (N * K // 2) * buf0["w"].element_size()
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
