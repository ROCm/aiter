# Inspired by https://github.com/thu-ml/SageAttention/blob/main/bench/bench_qk_int8_pv_fp16_triton.py
#

import sys
import torch
import triton
from aiter.ops.triton.attn_qk_int8_per_block import (
    attn_qk_int8_per_block,
    _get_config,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
)
from typing import Tuple


def get_tensors(batch_size: int, num_heads: int, seq_len: int, head_dim: int, config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]

    # Triton has seq_len after head_dim (HND layout)
    q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    q_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_SIZE_M), 1, dtype=torch.float16, device='cuda')
    k_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_SIZE_N), 1, dtype=torch.float16, device='cuda')
    
    return q, k, v, q_scale, k_scale


def bench_attn_fn(batch_size: int, num_heads: int, seq_len: int, head_dim: int, metric: str):
    """
    Benchmark function for attention kernel.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        metric: Metric to report ('time', 'throughput', or 'bandwidth')
    
    Returns:
        The requested metric value
    """
    config = _get_config()
    q, k, v, q_scale, k_scale = get_tensors(batch_size, num_heads, seq_len, head_dim, config)
    
    # Calculate FLOPs: 4 * num_heads * batch_size * head_dim * seq_len * seq_len
    flops = 4.0 * num_heads * batch_size * head_dim * seq_len * seq_len
    
    # Calculate memory transfer
    # Read: q, k, v, q_scale, k_scale
    mem_read = (
        q.element_size() * q.numel() +
        k.element_size() * k.numel() +
        v.element_size() * v.numel() +
        q_scale.element_size() * q_scale.numel() +
        k_scale.element_size() * k_scale.numel()
    )
    # Write: output (batch_size, num_heads, seq_len, head_dim) in bfloat16
    mem_write = batch_size * num_heads * seq_len * head_dim * 2  # bfloat16 = 2 bytes
    mem = mem_read + mem_write
    
    # Benchmark
    forward_fn = lambda: attn_qk_int8_per_block(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, config=config)
    ms = triton.testing.do_bench(forward_fn, warmup=25, rep=100)
    
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


def run_shape_benchmark(args):
    """
    Runs benchmark for attention kernel with specified shape.
    """
    # Determine shape parameters
    if args.shape is not None:
        if len(args.shape) != 4:
            raise ValueError(f"--shape must have 4 dimensions (batch_size, num_heads, seq_len, head_dim), got {len(args.shape)}")
        batch_size, num_heads, seq_len, head_dim = args.shape
    else:
        batch_size = args.batch_size
        num_heads = args.num_heads
        seq_len = args.seq_len
        head_dim = args.head_dim
    
    # Set up benchmark configuration
    x_names = ["batch_size", "num_heads", "seq_len", "head_dim"]
    x_vals_list = [[batch_size, num_heads, seq_len, head_dim]]
    
    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput_(TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")
    
    evaluation_metric_to_unit = {
        "throughput": "TFLOPS",
        "time": "Time_(ms)",
        "bandwidth": "Bandwidth_(GB/s)",
    }
    
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=False,
        y_log=False,
        line_arg="unit",
        line_vals=[evaluation_metric_to_unit[args.metric]],
        line_names=[evaluation_metric_to_unit[args.metric]],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )
    
    @triton.testing.perf_report([benchmark])
    def bench_attn_qk_int8_per_block(batch_size, num_heads, seq_len, head_dim, metric, **kwargs):
        return bench_attn_fn(batch_size, num_heads, seq_len, head_dim, metric)
    
    bench_attn_qk_int8_per_block.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    """
    Parse command-line arguments for attention benchmark.
    """
    parser = get_parser(kernel_name="QK Int8 Per Block Attention")
    parser = add_argparse_ff(parser)  # Adds -print_vgpr, -o, --shape, and other common flags
    
    # Add attention-specific arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=5,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=75600,
        help="Sequence length",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
        help="Head dimension",
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_shape_benchmark(args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_shape_benchmark(args)


if __name__ == '__main__':
    sys.exit(main())