

from typing import Optional
import functools
import sys
import json
import os
import torch
import triton
import triton.language as tl
import numpy as np
import argparse
from itertools import product
#from gluon_unified_attention_w_shared import unified_attention
from unified_attention_2d_final import unified_attention
#from gluon_unified_attention_w_shared_stage import unified_attention
#from aiter.ops.triton.attention.unified_attention import unified_attention
#from unified_attention_new_layout import unified_attention
def generate_data(seq_lens, num_blocks=32768, block_size=32, head_size=64, num_heads=(16, 2), sliding_window=None,
                  q_dtype=torch.float8_e4m3fn, kv_dtype=torch.bfloat16):
    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:           
        window_size = (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=torch.float32,
                        device="cuda").to(q_dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=torch.float32,
                            device="cuda")
    value_cache = torch.randn_like(key_cache).to(kv_dtype)
    key_cache = key_cache.to(kv_dtype)
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32, device="cuda").cumsum(dim=0,
                                                           dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    total_ind_count = num_seqs * max_num_blocks_per_seq
    values = torch.arange(0, total_ind_count, dtype=torch.int)
    values = values[torch.randperm(total_ind_count)]
    block_tables = values.view(num_seqs, max_num_blocks_per_seq).contiguous().to("cuda")

    sinks = torch.randn(num_query_heads,
                        dtype=torch.float32,
                        device="cuda")
    
    output = torch.empty_like(query)

    q_descale = None
    k_descale = None
    v_descale = None

    if kv_dtype == torch.float8_e4m3fn:
        k_descale = torch.tensor(1.0).to("cuda")
        v_descale = torch.tensor(1.0).to("cuda")

    return (query, key_cache, value_cache,
            sinks,
            output, 
            cu_query_lens,
            kv_lens,
            max_query_len,
            max_kv_len,
            scale,
            window_size,
            block_tables,
            q_descale, k_descale, v_descale)




parser = argparse.ArgumentParser(description="")
parser.add_argument('--num_heads_q', type=int, default=16, help='')
parser.add_argument('--num_heads_k', type=int, default=2, help='')
parser.add_argument('--head_size', type=int, default=64, help='')
parser.add_argument('--seq_q_l', type=int, default=1, help='')
parser.add_argument('--seq_kv_l', type=int, default=1024, help='')
parser.add_argument('--bs', type=int, default=1, help='')
parser.add_argument('--prefill_cnt', type=int, default=-1, help='')
parser.add_argument('--decode_cnt', type=int, default=-1, help='')
parser.add_argument('--window_size', type=int, default=128, help='')
parser.add_argument('--block_size', type=int, default=16, help='')
parser.add_argument('--repeat', type=int, default=1000, help='')
parser.add_argument('--cache_size', type=str, default="512*1024*1024", help='')
parser.add_argument('--test', type=int, default=0, help='')
parser.add_argument('--path', type=str, default="res", help='')
args = parser.parse_args()
print(args)

cache_size = eval(args.cache_size)
cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
clear_cache = lambda: cache.zero_()

if not os.path.exists(args.path):
    os.makedirs(args.path)

repeat = args.repeat
block_size = args.block_size
soft_cap = None
if args.prefill_cnt == -1 and args.decode_cnt == -1:
    seq_lens = [(args.seq_q_l, args.seq_kv_l)] * args.bs
else:
    seq_lens = []
    for i in range(args.decode_cnt):
        seq_lens.append((1, args.seq_kv_l))  
    for i in range(args.prefill_cnt):
        seq_lens.append((args.seq_q_l, args.seq_q_l))
    import random
    random.shuffle(seq_lens)

(maybe_quantized_query, maybe_quantized_key_cache, maybe_quantized_value_cache, 
            sinks, output, 
            cu_query_lens,
            kv_lens,
            max_query_len,
            max_kv_len,
            scale,
            window_size,
            block_tables,
            q_descale, k_descale, v_descale) = generate_data(seq_lens, num_blocks=32768 // 2, block_size=block_size, head_size=args.head_size, 
                                                             num_heads=(args.num_heads_q, args.num_heads_k), sliding_window=args.window_size,
                                                             q_dtype=torch.bfloat16, kv_dtype=torch.bfloat16)

func = lambda:  unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
    )

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


