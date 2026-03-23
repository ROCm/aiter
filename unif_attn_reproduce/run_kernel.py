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
from aiter.ops.triton.gluon.unified_attention_2d import (
    unified_attention as gluon_unified_attention_2d,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)

def shuffle_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    """
    Shuffle key and value cache layout for optimized memory access.

        layout: (num_lanes, num_elements_per_thread)
            gfx1250: (16, 8) for BF16 and FP8.
            gfx950: (16, 8) for BF16 and (16, 16) for FP8.

        WMMA/MFMA instruction shape:
            BF16: 16x16x32
            FP8: 16x16x64
    """
    dtype = key_cache.dtype
    assert value_cache.dtype == dtype
    assert dtype in (torch.bfloat16, e4m3_dtype)

    if IS_DEVICE_ARCH_GFX12:
        if dtype == torch.bfloat16:
            layout = (16, 8)
        else:
            # Caution: in gfx1250, the 16-bit and 8-bit layout should both be (16, 8), however, in order to enable ds_load_b128 for 8-bit WMMA,
            # we use (16, 16) here, noted that you must set k_width to 16 in the corresponding DotOperandLayout, the math will be equivalent.
            layout = (16, 16)
    else:
        if dtype == torch.bfloat16:
            layout = (16, 8)
        else:
            layout = (16, 16)

    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_blocks_v, block_size_v, num_kv_heads_v, head_size_v = value_cache.shape
    assert block_size >= 16
    assert num_blocks == num_blocks_v
    assert num_kv_heads == num_kv_heads_v
    assert head_size == head_size_v
    assert block_size == block_size_v

    num_lanes, num_elements_per_thread = layout
    key_cache_shuffled = key_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    key_cache_shuffled = key_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // num_lanes,
        num_lanes,
        head_size // (2 * num_elements_per_thread),
        2,  # there are 2 groups of threads, t0 ~ t15 and t16 ~ t31
        num_elements_per_thread,
    )
    key_cache_shuffled = key_cache_shuffled.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
    key_cache_shuffled = key_cache_shuffled.view(
        -1, num_kv_heads, block_size // 16, head_size * 16
    )
    
    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // (2 * num_elements_per_thread),
        2,
        num_elements_per_thread,
        head_size // num_lanes,
        num_lanes,
    )
    value_cache_shuffled = value_cache_shuffled.permute(
        0, 1, 5, 2, 3, 6, 4
    ).contiguous()
    value_cache_shuffled = value_cache_shuffled.view(
        -1, num_kv_heads, head_size // 16, block_size * 16
    )

    return key_cache_shuffled, value_cache_shuffled

def generate_data(seq_lens, num_blocks=32768, block_size=32, head_size=64, num_heads=(16, 2), sliding_window=None,
                  q_dtype=torch.float8_e4m3fn, kv_dtype=torch.bfloat16, shuffled_kv_cache=False):
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
                        device="cpu").to(q_dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=torch.float32,
                            device="cpu")
    value_cache = torch.randn_like(key_cache).to(kv_dtype)
    key_cache = key_cache.to(kv_dtype)
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32, device="cpu").cumsum(dim=0,
                                                           dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    max_num_blocks_per_seq = min(max_num_blocks_per_seq * num_seqs, num_blocks) // num_seqs
    total_ind_count = num_seqs * max_num_blocks_per_seq
    values = torch.arange(0, total_ind_count, dtype=torch.int)
    values = values[torch.randperm(total_ind_count)]
    block_tables = values.view(num_seqs, max_num_blocks_per_seq).contiguous().to("cpu")

    sinks = torch.randn(num_query_heads,
                        dtype=torch.float32,
                        device="cpu")
    
    output = torch.empty_like(query)

    q_descale = None
    k_descale = None
    v_descale = None
    output_scale = None
    if q_dtype == e4m3_dtype:
        q_descale = torch.tensor(1.0).to("cpu")
        output_scale = torch.tensor(1.0).to("cpu")
    if kv_dtype == e4m3_dtype:
        k_descale = torch.tensor(1.0).to("cpu")
        v_descale = torch.tensor(1.0).to("cpu")
        

    if shuffled_kv_cache:
        key_cache, value_cache = shuffle_kv_cache(key_cache, value_cache)

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
            q_descale, k_descale, v_descale, output_scale)




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
parser.add_argument('--use_tdm', type=int, default=1, help='')
parser.add_argument('--num_kv_blocks', type=int, default=1, help='')
parser.add_argument('--waves_per_eu', type=int, default=1, help='')
parser.add_argument('--shuffled_kv_cache', type=int, default=0, help='')
parser.add_argument('--num_warps', type=int, default=4, help='')
parser.add_argument('--q_fp8', type=int, default=0, help='')
parser.add_argument('--kv_fp8', type=int, default=0, help='')
parser.add_argument('--block_m', type=int, default=128, help='')

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
q_dtype = torch.bfloat16
kv_dtype = torch.bfloat16
if args.q_fp8 == 1:
    q_dtype = e4m3_dtype
if args.kv_fp8 == 1:
    kv_dtype = e4m3_dtype
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
                                                             q_dtype=q_dtype, kv_dtype=kv_dtype, shuffled_kv_cache=args.shuffled_kv_cache)
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

# #warm-up
# warmup_cnt = 10 if repeat < 200 else 100
# for i in range(warmup_cnt):
#     clear_cache()
#     torch.cuda.synchronize()
#     func()

# for i in range(repeat):
#     clear_cache()
#     torch.cuda.synchronize()
#     func()
print("Before calling func")
print("--------------------------------")
func()
print("After calling func")
print("--------------------------------")
