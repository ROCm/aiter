import functools

import torch

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_dual_splitk import (
    fused_gemm_a16w16_dual_splitk,
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

dimension = ["M", "N1", "N2", "K"]
kernel_name = "Fused A16W16 dual split-K GEMM"
kernel_label = "fused_gemm_a16w16_dual_splitk"
shape = "(N1=16384, N2=96, K=5120)"


def bench_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N1: int,
    N2: int,
    K: int,
    metric: str,
    **kwargs,
):
    """Single-shape timing of the dual split-K GEMM (both parts reduced) through its
    public wrapper, W = [W1 (N1 rows); W2 (N2 rows)]. The output buffer is
    pre-allocated and passed in to keep allocation out of the timed ``do_bench``
    window."""
    c_dtype = torch.bfloat16
    _p_gen = functools.partial(
        generate_gemm_a16w16_inputs, M, N1 + N2, K, c_dtype, output=True, bias=False
    )
    # (x, weight, bias, out_dtype, y)
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [["x", "w", "_", "_", "y"]], **kwargs_rot_bufs
    )
    kernel_fn = functools.partial(fused_gemm_a16w16_dual_splitk, n2=N2, dtype=c_dtype)

    flops = 2.0 * M * (N1 + N2) * K
    elem = rot_bufs[0]["x"].element_size()
    mem = (M * K + (N1 + N2) * K + M * (N1 + N2)) * elem

    ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

    return metric_to_scalar(metric, ms, flops, mem)


def get_x_vals(args=None):
    """Default (M, N1, N2, K) shapes: the tuned N1=16384, N2=96, K=5120 config, every
    M bucket of its file plus the ``any`` bucket; ``-M`` selects a single M."""
    n1, n2, k = 16384, 96, 5120
    if args is not None and getattr(args, "M", None) is not None:
        m_vals = [args.M]
    else:
        m_vals = [1, 4, 8, 16, 32, 64, 128, 256]
    return [(m, n1, n2, k) for m in m_vals]


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
