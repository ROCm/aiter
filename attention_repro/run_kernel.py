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
from unified_attention_2d import (
    unified_attention as gluon_unified_attention_2d,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from test_attention import generate_data, shuffle_kv_cache
DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)






parser = argparse.ArgumentParser(description="")
parser.add_argument('--num_heads_q', type=int, default=16, help='')
parser.add_argument('--num_heads_k', type=int, default=2, help='')
parser.add_argument('--head_size', type=int, default=64, help='')
parser.add_argument('--seq_q_l', type=int, default=1, help='')
parser.add_argument('--seq_kv_l', type=int, default=1024, help='')
parser.add_argument('--bs', type=int, default=1, help='')
parser.add_argument('--block_size', type=int, default=16, help='')
parser.add_argument('--waves_per_eu', type=int, default=1, help='')
parser.add_argument('--num_warps', type=int, default=4, help='')
parser.add_argument('--block_m', type=int, default=128, help='')
parser.add_argument('--remove_indirect_access', type=int, default=1, help='')
args = parser.parse_args()
print(args)

args.window_size = 0
args.num_kv_blocks = 1
args.use_tdm = 1
args.shuffled_kv_cache = 0

block_size = args.block_size
soft_cap = None
seq_lens = [(args.seq_q_l, args.seq_kv_l)] * args.bs

q_dtype = torch.bfloat16
kv_dtype = torch.bfloat16

(maybe_quantized_query, maybe_quantized_key_cache, maybe_quantized_value_cache, 
            sinks, output, 
            cu_query_lens,
            kv_lens,
            max_query_len,
            max_kv_len,
            scale,
            window_size,
            block_tables,
            q_descale, k_descale, v_descale, output_scale) = generate_data(seq_lens, num_blocks=128, block_size=block_size, head_size=args.head_size, 
                                                             num_heads=(args.num_heads_q, args.num_heads_k), sliding_window=args.window_size,
                                                             q_dtype=q_dtype, kv_dtype=kv_dtype, shuffled_kv_cache=args.shuffled_kv_cache, remove_indirect_access=args.remove_indirect_access,)
output_scale = None
print("After generate data")
print("--------------------------------")

use_tdm = args.use_tdm == 1
num_kv_blocks = args.num_kv_blocks
new_kv_layout = num_kv_blocks > 1

if output_scale is not None:
    output_scale = output_scale.cuda()
if q_descale is not None:
    q_descale = q_descale.cuda()
if k_descale is not None:
    k_descale = k_descale.cuda()
if v_descale is not None:
    v_descale = v_descale.cuda()
if new_kv_layout:
    maybe_quantized_key_cache = maybe_quantized_key_cache.permute(0, 2, 1, 3)    
    maybe_quantized_value_cache = maybe_quantized_value_cache.permute(0, 2, 1, 3)

func = lambda:  gluon_unified_attention_2d(
        q=maybe_quantized_query.cuda(),
        k=maybe_quantized_key_cache.cuda(),
        v=maybe_quantized_value_cache.cuda(),
        out=output.cuda(),
        cu_seqlens_q=cu_query_lens.cuda(),
        seqused_k=kv_lens.cuda(),
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables.cuda(),
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks.cuda(),
        output_scale=output_scale,
        use_tdm=use_tdm,
        num_kv_blocks=num_kv_blocks,
        new_kv_layout=new_kv_layout,
        waves_per_eu=args.waves_per_eu,
        shuffled_kv_cache=args.shuffled_kv_cache,
        num_warps=args.num_warps,
        block_m=args.block_m,
    )


print("Before calling func")
print("--------------------------------")
func()
print("After calling func")
print("--------------------------------")