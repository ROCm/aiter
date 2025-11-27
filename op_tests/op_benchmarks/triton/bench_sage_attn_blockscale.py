# Inspired by https://github.com/thu-ml/SageAttention/blob/main/bench/bench_qk_int8_pv_fp16_triton.py
#

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from flash_attn.utils.benchmark import benchmark_forward
from sageattention.triton.attn_qk_int8_per_block import forward

import argparse
from typing import Tuple


# TODO these should still match with the values hardcoded in the kernel
# in SageAttention/triton/attn_qk_int8_per_block.py
# TODO make this configurable
kernel_configs = {
    'sage_v1': {
        "MI300x": {
            'BLOCK_M': 64,
            'BLOCK_N': 16,
        },
        "H100": {
            'BLOCK_M': 128,
            'BLOCK_N': 64,
        },
    },
}

def get_hw() -> str:
    if torch.version.hip is not None:
        return 'MI300x'
    elif torch.cuda.is_available():
        return 'H100'
    else:
        raise ValueError("No GPU detected")

def get_tensors(hw: str, method: str, batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if method == 'sage_v1':
        kernel_config = kernel_configs[method][hw]
        BLOCK_M = kernel_config['BLOCK_M']
        BLOCK_N = kernel_config['BLOCK_N']
        
        q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
        k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        q_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_M), 1, dtype=torch.float16, device='cuda')
        k_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_N), 1, dtype=torch.float16, device='cuda')
    elif method == 'fa2':
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        q_scale = None
        k_scale = None
    return q, k, v, q_scale, k_scale

parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16 Triton')
parser.add_argument('--method', type=str, default='sage_v1', choices=['sage_v1', 'fa2'])
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_heads', type=int, default=5, help='Number of heads')
parser.add_argument('--seq_len', type=int, default=75600, help='Sequence length')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--repeats', type=int, default=100, help='Number of repeats')
parser.add_argument('--hw', type=str, default=get_hw(), choices=['MI300x', 'H100'])
args = parser.parse_args()

batch_size = args.batch_size
num_heads = args.num_heads
head_dim = args.head_dim
seq_len = args.seq_len
method = args.method
hw = args.hw

print(f"{method} {hw} QK Int8 PV FP16 Benchmark")
print(f"batch_size: {batch_size}, num_heads: {num_heads}, seq_len:{seq_len}, head_dim: {head_dim}")

flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len

if method == 'sage_v1':
    q, k, v, q_scale, k_scale = get_tensors(hw, method, batch_size, num_heads, seq_len, head_dim)
    for i in range(5): 
        forward(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
    # torch.cuda.synchronize() is needed here to ensure that all queued CUDA operations 
    # (specifically the warm-up forward passes above) are finished before 
    # the benchmarking begins. CUDA operations are asynchronous by default, 
    # so without this, the benchmark may include work from warm-up or not 
    # actually measure only the benchmarked region. 
    torch.cuda.synchronize()

    _, time = benchmark_forward(forward, q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, repeats=args.repeats, verbose=False, desc='Triton')
elif method == 'fa2':
    torch.backends.cuda.enable_flash_sdp(args.method == 'fa2')   # use FA2
    q, k, v, _, _ = get_tensors(hw, method, batch_size, num_heads, seq_len, head_dim)
    for i in range(5): sdpa(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    _, time = benchmark_forward(sdpa, q, k, v, is_causal=False, repeats=args.repeats, verbose=False, desc='Triton')
print(f'flops:{flops/time.mean*1e-12}')
