import functools

import torch

from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 import (
    fused_gemm_afp4wfp4_a16w16,
)
from op_tests.op_benchmarks.triton.bench_fused_gemm_commons import (
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
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)

dimension = ["M", "N4", "N16", "K"]
kernel_name = "Fused AFP4WFP4 + A16W16 GEMM"
kernel_label = "fused_gemm_afp4wfp4_a16w16"
shape = "(N4=512, N16=256, K=7168)"


def bench_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N4: int,
    N16: int,
    K: int,
    metric: str,
    **kwargs,
):
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
    # shuffles off the shuffled returns equal the plain ones, so the plain
    # ``w_fp4``/unshuffled-scale returns are dropped ("_") here.
    _p_gen_fp4 = functools.partial(
        generate_gemm_afp4wfp4_inputs,
        M,
        N4,
        K,
        c_dtype,
        layout="TN",
        output=True,
        shuffle_scales_fg=False,
        shuffle_weight_fg=False,
    )
    # (x, w, w_shuffed, x_scales, w_scales, x_scales_shuffled, w_scales_shuffled, out_dtype, y)
    _out_vals_keys_fp4 = [
        "x_fp4",
        "_",
        "w_fp4",
        "_",
        "_",
        "x_fp4_scale",
        "w_fp4_scale",
        "_",
        "y_fp4",
    ]

    # BF16 branch inputs (N16 = N_bf16). Same M, same logical K, and no bias.
    _p_gen_bf16 = functools.partial(
        generate_gemm_a16w16_inputs,
        M,
        N16,
        K,
        c_dtype,
        output=True,
        bias=False,
    )
    # (x, weight, bias, out_dtype, y)
    _out_vals_keys_bf16 = ["x_bf16", "w_bf16", "_", "_", "y_bf16"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen_fp4, _p_gen_bf16],
        [_out_vals_keys_fp4, _out_vals_keys_bf16],
        **kwargs_rot_bufs,
    )

    kernel_fn = functools.partial(
        fused_gemm_afp4wfp4_a16w16, is_fp4_preshuffled=False, dtype=c_dtype
    )

    flops = 2.0 * M * (N4 + N16) * K  # summed across the two fused outputs
    # bytes moved, derived from M/N4/N16/K. x_fp4/w_fp4 are e2m1, packed two
    # values per byte (K // 2 wide); the mx scale tensors keep their own numel
    # (one scale per 32 K-elements, which doesn't reduce to a clean M/N/K
    # product).
    buf0 = rot_bufs[0]
    mem = (
        (M * K // 2) * buf0["x_fp4"].element_size()
        + (N4 * K // 2) * buf0["w_fp4"].element_size()
        + buf0["x_fp4_scale"].numel() * buf0["x_fp4_scale"].element_size()
        + buf0["w_fp4_scale"].numel() * buf0["w_fp4_scale"].element_size()
        + (M * K) * buf0["x_bf16"].element_size()
        + (N16 * K) * buf0["w_bf16"].element_size()
        + (M * N4) * buf0["y_fp4"].element_size()
        + (M * N16) * buf0["y_bf16"].element_size()
    )

    ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

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
