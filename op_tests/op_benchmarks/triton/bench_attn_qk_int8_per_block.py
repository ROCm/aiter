# Inspired by https://github.com/thu-ml/SageAttention/blob/main/bench/bench_qk_int8_pv_fp16_triton.py
#

import torch
from aiter.ops.triton.attn_qk_int8_per_block import (
    attn_qk_int8_per_block,
    _get_config,
)
from triton.testing import do_bench

import argparse
from typing import Tuple
import json


def get_tensors(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = args.batch_size
    num_heads = args.num_heads
    seq_len = args.seq_len
    head_dim = args.head_dim

    # Load actual config to get BLOCK_SIZE_M and BLOCK_SIZE_N for scale tensor generation
    config = _get_config(batch_size, num_heads, seq_len, head_dim)
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]

    # Triton has seq_len after head_dim (HND layout)
    q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    q_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_SIZE_M), 1, dtype=torch.float16, device='cuda')
    k_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_SIZE_N), 1, dtype=torch.float16, device='cuda')
    
    return q, k, v, q_scale, k_scale


def get_parser():
    parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16 Attention on MI300X')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=5, help='Number of heads')
    parser.add_argument('--seq_len', type=int, default=75600, help='Sequence length')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--repeats', type=int, default=100, help='Number of repeats')
    parser.add_argument("--output_json", type=str, default=None, help='Output JSON file')
    return parser.parse_args()

def get_flops(args: argparse.Namespace) -> int:
    return 4 * args.num_heads * args.batch_size * args.head_dim * args.seq_len * args.seq_len

def benchmark(args: argparse.Namespace) -> float:
    q, k, v, q_scale, k_scale = get_tensors(args)

    forward_fn = lambda: attn_qk_int8_per_block(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
    time = do_bench(forward_fn, warmup=25, rep=args.repeats)
    
    return time

def output_json(args: argparse.Namespace, flops: float, time: float):
    with open(args.output_json, 'a') as f:
        f.write(json.dumps({
            'method': 'sage_v1_triton',
            'hw': 'MI300x',
            'batch_size': args.batch_size,
            'num_heads': args.num_heads,
            'seq_len': args.seq_len,
            'head_dim': args.head_dim,
            'flops': flops,
            'time': time,
        }) + '\n')

def main():
    args = get_parser()
    time = benchmark(args)
    flops = get_flops(args)/time*1e-9
    if args.output_json is not None:
        output_json(args, flops, time)
    else:
        print('MI300X QK Int8 PV FP16 Attention Benchmark')
        print(f'batch_size: {args.batch_size}, num_heads: {args.num_heads}, seq_len: {args.seq_len}, head_dim: {args.head_dim}')
        print(f'time: {time:.3f} ms')
        print(f'TFLOPS: {flops:.2f}')

if __name__ == '__main__':
    main()