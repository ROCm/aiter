import torch
from utils import (
    per_custom_dims_cast_to_fp8,
    generate_cp_test_data,
)
import argparse
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

parser = argparse.ArgumentParser(description="")
parser.add_argument('--num_heads_q', type=int, default=16, help='')
parser.add_argument('--head_dim', type=int, default=512, help='')
parser.add_argument('--seq_q_l', type=int, default=1, help='')
parser.add_argument('--seq_kv_l', type=int, default=1024, help='')
parser.add_argument('--repeat', type=int, default=1000, help='')
parser.add_argument('--cache_size', type=str, default="512*1024*1024", help='')
args = parser.parse_args()
print(args)

cache_size = eval(args.cache_size)
cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
clear_cache = lambda: cache.zero_()

seq_len = args.seq_q_l
seq_len_kv = args.seq_kv_l
num_heads, head_dim = args.num_heads_q, args.head_dim
repeat = args.repeat

disable_cp = True
q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)

if disable_cp:
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)
else:
    ks, ke = generate_cp_test_data(seq_len, seq_len_kv)

q_fp8 = q.to(e4m3_dtype)
kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0, ), False)

func = lambda: fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke)

if repeat > 1:
    #warm-up
    warmup_cnt = 10 if repeat < 200 else 100
    for i in range(warmup_cnt):
        clear_cache()
        torch.cuda.synchronize()
        func()

    for i in range(repeat):
        clear_cache()
        torch.cuda.synchronize()
        func()
else:
    # used for ATT
    func()



