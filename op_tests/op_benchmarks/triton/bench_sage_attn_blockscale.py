import sys
import os
import importlib.util
import torch
import triton
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_shape_benchmark_object,
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)

# Import forward function from the .mi300x.py file
# Get the path relative to workspace root
_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_kernel_path = os.path.join(
    _workspace_root,
    "aiter/ops/triton/_triton_kernels/attn_qk_int8_per_block.mi300x.py"
)
_spec = importlib.util.spec_from_file_location(
    "attn_qk_int8_per_block_mi300x",
    _kernel_path
)
_attn_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attn_module)
forward = _attn_module.forward
_get_config = _attn_module._get_config


def generate_attn_qk_int8_per_block_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    """
    Generate inputs for the attention kernel with QK int8 per-block quantization.
    
    Args:
        batch_size: Batch size
        num_heads: Number of heads
        seq_len: Sequence length (same for Q, K, V in self-attention)
        head_dim: Head dimension
    
    Returns:
        q, k, v, q_scale, k_scale
    """
    # Get BLOCK_SIZE_M and BLOCK_SIZE_N from config
    # Config is loaded based on (M=seq_len, N=seq_len, K=head_dim)
    try:
        config = _get_config(seq_len, seq_len, head_dim)
        BLOCK_M = config.get("BLOCK_SIZE_M", 64)
        BLOCK_N = config.get("BLOCK_SIZE_N", 16)
    except Exception:
        # Fallback to defaults if config loading fails
        BLOCK_M = 64
        BLOCK_N = 16
    
    # Q, K are int8, V is fp16
    # Shape: (batch_size, num_heads, seq_len, head_dim)
    q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    
    # Scales are per-block
    # q_scale: (batch_size, num_heads, num_blocks_m, 1)
    q_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_M), 1, dtype=torch.float16, device='cuda')
    
    # k_scale: (batch_size, num_heads, num_blocks_n, 1)
    k_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_N), 1, dtype=torch.float16, device='cuda')
    
    return q, k, v, q_scale, k_scale


def bench_attn_fn(batch_size: int, num_heads: int, seq_len: int, head_dim: int, metric: str, impl: callable):
    """
    Benchmark the attention kernel.
    
    Args:
        batch_size: Batch size
        num_heads: Number of heads
        seq_len: Sequence length
        head_dim: Head dimension
        metric: Metric to measure ('time', 'throughput', 'bandwidth')
        impl: Implementation function to benchmark
    
    Returns:
        Scalar value for the requested metric
    """
    q, k, v, q_scale, k_scale = generate_attn_qk_int8_per_block_inputs(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )
    
    # FLOPS: 4 * batch_size * num_heads * head_dim * seq_len * seq_len
    # (QK^T: seq_len * seq_len * head_dim, softmax: seq_len * seq_len, PV: seq_len * head_dim * seq_len)
    flops = 4.0 * batch_size * num_heads * head_dim * seq_len * seq_len
    
    # Memory transfer
    # Read: Q (int8), K (int8), V (fp16), q_scale (fp16), k_scale (fp16)
    # Write: O (bf16)
    # Get scale sizes based on actual BLOCK_SIZE_M/BLOCK_SIZE_N from config
    try:
        config = _get_config(seq_len, seq_len, head_dim)
        BLOCK_M = config.get("BLOCK_SIZE_M", 64)
        BLOCK_N = config.get("BLOCK_SIZE_N", 16)
    except Exception:
        BLOCK_M = 64
        BLOCK_N = 16
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    
    mem_read = (
        batch_size * num_heads * seq_len * head_dim * 1  # Q int8
        + batch_size * num_heads * seq_len * head_dim * 1  # K int8
        + batch_size * num_heads * seq_len * head_dim * 2  # V fp16
        + batch_size * num_heads * num_blocks_m * 2  # q_scale fp16
        + batch_size * num_heads * num_blocks_n * 2  # k_scale fp16
    )
    mem_write = batch_size * num_heads * seq_len * head_dim * 2  # O bf16
    mem = mem_read + mem_write
    
    ms = triton.testing.do_bench(
        lambda: impl(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16),  # noqa: E731
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


def run_shape_benchmark(args, impl):
    """
    Runs benchmark given shape arguments.
    """
    # Map x_names to attention-specific names: B, M, N, K -> batch_size, num_heads, seq_len, head_dim
    x_names = ["B", "M", "N", "K"]
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args, x_names=x_names)

    @triton.testing.perf_report([benchmark])
    def bench_attn_qk_int8_per_block(B, M, N, K, metric, _model_name=None, **_kwargs):
        # B=batch_size, M=num_heads, N=seq_len, K=head_dim
        return bench_attn_fn(B, M, N, K, metric, impl)

    bench_attn_qk_int8_per_block.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    """
    Main benchmark runner.
    """
    impl = forward
    
    if args.model:
        raise NotImplementedError("Model benchmarking not yet implemented for attention kernels")
    else:
        # Check for unsupported feed-forward arguments
        unsupported_args = ["fc1", "fc2", "no_glu", "tp"]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise ValueError(
                    f"Argument '{arg}' is not supported for attention benchmarking."
                )
        run_shape_benchmark(args, impl)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = get_parser(kernel_name="QK Int8 Per-Block Attention")
    parser = add_argparse_ff(parser)
    
    return get_ff_args(parser)


def main():
    """
    Main entry point.
    """
    args, defaults = parse_args()
    
    # Handle shape argument mapping: B, M, N, K -> batch_size, num_heads, seq_len, head_dim
    if args.shape is not None:
        if len(args.shape) == 3:
            # 3D: M, N, K -> num_heads, seq_len, head_dim (with default B=1)
            args.B = 1
            args.M, args.N, args.K = args.shape
        elif len(args.shape) == 4:
            # 4D: B, M, N, K -> batch_size, num_heads, seq_len, head_dim
            args.B, args.M, args.N, args.K = args.shape
        else:
            raise ValueError(
                f"Expected 3D (M, N, K) or 4D (B, M, N, K) shape, got {len(args.shape)}D. "
                f"Shape should be: (batch_size, num_heads, seq_len, head_dim) or "
                f"(num_heads, seq_len, head_dim) with default batch_size=1"
            )
    
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
