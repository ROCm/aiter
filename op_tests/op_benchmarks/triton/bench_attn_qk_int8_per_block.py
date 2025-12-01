# Inspired by https://github.com/thu-ml/SageAttention/blob/main/bench/bench_qk_int8_pv_fp16_triton.py
#

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

# requires aiter v0.1.7.post2
# e.g. git clone -b v0.1.7.post2 https://github.com/rocm/aiter.git; cd aiter; git submodule sync && git submodule update --init --recursive; python3 setup.py develop
try:
    from aiter.ops.triton.mha_v3 import (
        flash_attn_func as fa3_fwd
    )
except ImportError:
    fa3_fwd = None
from triton.testing import do_bench
from sageattention.triton.attn_qk_int8_per_block import forward

import argparse
from typing import Tuple
import json


# TODO these should still match with the values hardcoded in the kernel
# in SageAttention/triton/attn_qk_int8_per_block.py
# TODO make this configurable
kernel_configs = {
    'sage_v1_triton': {
        "MI300x": {
            'BLOCK_M': 128,
            'BLOCK_N': 32,
        },
        "H100": {
            'BLOCK_M': 128,
            'BLOCK_N': 64,
        },
    },
    "sage_v1_cuda": {
        "H100": {
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'WARP_Q': 32,
            'WARP_K': 64,
        },
    },
    # not used for benchmarking
    "fa2_sdpa": None,
    "fa3": None,
}

def get_hw() -> str:
    if torch.version.hip is not None:
        return 'MI300x'
    elif torch.cuda.is_available():
        return 'H100'
    else:
        raise ValueError("No GPU detected")

def get_tensors(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hw = args.hw
    method = args.method
    batch_size = args.batch_size
    num_heads = args.num_heads
    seq_len = args.seq_len
    head_dim = args.head_dim

    kernel_config = kernel_configs[method][hw] if kernel_configs[method] is not None else None
    BLOCK_M = kernel_config['BLOCK_M'] if kernel_config is not None else None
    BLOCK_N = kernel_config['BLOCK_N'] if kernel_config is not None else None

    if method == 'sage_v1_triton':
        # Triton has seq_len after head_dim
        q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
        k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        q_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_M), 1, dtype=torch.float16, device='cuda')
        k_scale = torch.randn(batch_size, num_heads, (seq_len // BLOCK_N), 1, dtype=torch.float16, device='cuda')
    elif method == 'sage_v1_cuda':
        # TODO this is a hack to make the seq_len divisible by 1024 as hardcoded in the cuda kernel
        # TODO remove this once the cuda kernel is updated to support non-divisible seq_lens
        new_seq_len = closest_divisible(seq_len, 1024)
        if new_seq_len != seq_len:
            print(f"seq_len is not divisible by 1024 as hardcoded in the cuda kernel, using {new_seq_len} instead")
            seq_len = new_seq_len
        # CUDA has seq_len before num_heads
        q = torch.randint(-95, 95, (batch_size, seq_len, num_heads, head_dim), dtype=torch.int8, device="cuda")
        k = torch.randint(-95, 95, (batch_size, seq_len, num_heads, head_dim), dtype=torch.int8, device="cuda")
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
        WARP_Q = kernel_config['WARP_Q']
        WARP_K = kernel_config['WARP_K']
        if args.quant_gran == 'per_warp':
            q_scale = torch.randn(batch_size, num_heads, seq_len // WARP_Q, dtype=torch.float, device="cuda")
            k_scale = torch.randn(batch_size, num_heads, seq_len // WARP_K, dtype=torch.float, device="cuda")
        elif args.quant_gran == 'per_thread':
            q_scale = torch.randn(batch_size, num_heads, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
            k_scale = torch.randn(batch_size, num_heads, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")
    elif method == 'fa2_sdpa':
        # Torch SDPA FA2 has seq_len after num_heads
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        # FA2 baseline without quantization
        q_scale = None
        k_scale = None
    elif method == 'fa3':
        # AITER Triton FA3 has seq_len before num_heads
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
        # FA3 baseline without quantization
        q_scale = None
        k_scale = None
    else:
        raise ValueError(f"Unsupported method: {method}")
    return q, k, v, q_scale, k_scale


def closest_divisible(n, k):
    return int(round(n / k) * k)

def get_parser():
    parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16 Triton')
    parser.add_argument('--method', type=str, default='sage_v1_triton', choices=['sage_v1_triton', 'sage_v1_cuda', 'fa2_sdpa', 'fa3'])
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=5, help='Number of heads')
    parser.add_argument('--seq_len', type=int, default=75600, help='Sequence length')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--repeats', type=int, default=100, help='Number of repeats')
    parser.add_argument('--hw', type=str, default=get_hw(), choices=['MI300x', 'H100'])
    parser.add_argument('--pv_accum_dtype', type=str, default='fp16', choices=['fp16', 'fp16+fp32', 'fp32'], help='PV accumulation dtype, relevant for CUDA backend')
    parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity, relevant for CUDA backend')
    parser.add_argument("--output_json", type=str, default=None, help='Output JSON file')
    return parser.parse_args()

def get_flops(args: argparse.Namespace) -> int:
    return 4 * args.num_heads * args.batch_size * args.head_dim * args.seq_len * args.seq_len

def benchmark(args: argparse.Namespace) -> float:
    if args.method == 'sage_v1_triton':
        q, k, v, q_scale, k_scale = get_tensors(args)

        forward_fn = lambda: forward(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
        time = do_bench(forward_fn, warmup=25, rep=args.repeats)
    elif args.method == 'sage_v1_cuda':
        if args.hw == 'MI300x':
            print("Skipping CUDA benchmark on MI300x as it is not supported")
            return -1

        try:
            import sageattention._qattn_sm80 as qattn
        except ImportError:
            print("sageattention._qattn_sm80 not found, skipping")
            return -1

        if args.pv_accum_dtype == 'fp32':
            kernel = qattn.qk_int8_sv_f16_accum_f32_attn # the kernel with fully fp32 accumulator
        elif args.pv_accum_dtype == 'fp16+fp32':
            kernel = qattn.qk_int8_sv_f16_accum_f16_attn_inst_buf # the kernel with fp32 longterm buffer and fp16 shortterm accumulator
        elif args.pv_accum_dtype == 'fp16':
            kernel = qattn.qk_int8_sv_f16_accum_f16_attn # the kernel with fully fp16 accumulator
        else:
            raise ValueError(f"Unsupported pv_accum_dtype: {args.pv_accum_dtype}")

        q, k, v, q_scale, k_scale = get_tensors(args)
        o = torch.empty(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")

        sm_scale = 1 / (head_dim ** 0.5)
        _qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2
        _is_causal = 0
        forward_fn = lambda: kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
        time = do_bench(forward_fn, warmup=25, rep=args.repeats)
    elif args.method == 'fa2_sdpa':
        torch.backends.cuda.enable_flash_sdp(True)
        q, k, v, _, _ = get_tensors(args)
        forward_fn = lambda: sdpa(q, k, v, is_causal=False)
        time = do_bench(forward_fn, warmup=25, rep=args.repeats)
    elif args.method == 'fa3':
        if fa3_fwd is None and args.hw == 'MI300x':
            raise ValueError("aiter v0.1.7.post2 not found, using fa3 from aiter v0.1.7")
        elif fa3_fwd is None and args.hw == 'H100':
            print("Skipping FA3 benchmark on H100 as aiter does not exist")
            return -1

        q, k, v, _, _ = get_tensors(args)

        forward_fn = lambda: fa3_fwd(
            q=q,
            k=k,
            v=v,
            softmax_scale=1.0,
            causal=False,
        )
        time = do_bench(forward_fn, warmup=25, rep=args.repeats)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    return time

def output_json(args: argparse.Namespace, flops: float, time: float):
    with open(args.output_json, 'a') as f:
        f.write(json.dumps({
            'method': args.method,
            'hw': args.hw,
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
    if time == -1:
        print(f"Skipping {args.method} {args.hw} QK Int8 PV FP16 Benchmark as it is not supported")
        return
    flops = get_flops(args)/time*1e-9
    if args.output_json is not None:
        output_json(args, flops, time)
    else:
        print(f'{args.method} {args.hw} QK Int8 PV FP16 Benchmark')
        print(f'batch_size: {args.batch_size}, num_heads: {args.num_heads}, seq_len:{args.seq_len}, head_dim: {args.head_dim}')
        print(f'flops:{flops}')

if __name__ == '__main__':
    main()