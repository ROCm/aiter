import functools

import torch
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_mul_add import (
    fused_gemm_a16w16_mul_add,
)
from op_tests.op_benchmarks.triton.bench_fused_gemm_commons import (
    metric_to_scalar,
    parse_fused_args,
    run_fused_benchmark,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.gemm.fused.test_fused_gemm_a16w16_mul_add import (
    generate_fused_gemm_a16w16_mul_add_inputs,
)

dimension = ["B", "S", "N", "K"]
kernel_name = "Fused A16W16 GEMM + mul_add"
kernel_label = "fused_gemm_a16w16_mul_add"
shape = "(B, S, N, K)"

_OUT_VAL_KEYS = ["x", "w", "a", "b", "bias", "y"]


def _generate_pool_inputs(B, S, N, K, dtype):
    x, w, a, b, bias = generate_fused_gemm_a16w16_mul_add_inputs(
        B, S, N, K, dtype, "gate", "full", bias=True
    )
    y = torch.empty((B, S, N), dtype=dtype, device="cuda")
    return x, w, a, b, bias, y


def bench_fn(
    bencher: RotatingBufferBench, B: int, S: int, N: int, K: int, metric: str, **kwargs
):
    """
    Single-shape timing of the gated residual
    ``out = residual + gate * (x @ w^T + bias)`` through the public wrapper.
    The gate has shape (B, 1, N) and the residual has shape (B, S, N).
    """
    dtype = torch.bfloat16
    _p_gen = functools.partial(_generate_pool_inputs, B, S, N, K, dtype)
    rot_bufs = bencher.build_pool([_p_gen], [_OUT_VAL_KEYS])

    sample = rot_bufs[0]
    flops = 2.0 * B * S * N * K
    mem = sum(sample[k].numel() * sample[k].element_size() for k in _OUT_VAL_KEYS)

    kernel_fn = functools.partial(fused_gemm_a16w16_mul_add, dtype=dtype)
    ms = bencher.bench(kernel_fn)

    return metric_to_scalar(metric, ms, flops, mem)


def get_x_vals(args=None):
    """Default (B, S, N, K) shapes: attention output projections of diffusion
    transformers, where the gated residual follows the GEMM."""
    return [
        (2, 4096, 1152, 1152),  # PixArt-XL at 1024px
        (2, 1024, 2432, 2432),  # SD3.5-Large image stream
        (2, 333, 2432, 2432),  # SD3.5-Large context stream
        (1, 32760, 1536, 1536),  # Wan2.1-1.3B
        (1, 4096, 3072, 3072),  # FLUX.1 image stream
    ]


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
