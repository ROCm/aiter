# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from typing import List, Optional, Tuple, Union
from math import ceil

MAX_TOKEN_SUPPORTED = 16384


@perftest(num_iters=3)
def run_torch(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, block_size, x, asm_layout, quantCfg={}):
    num_batch, num_tokens, num_heads, head_size = key.shape
    num_blocks = k_cache.shape[0]
    dtype = k_cache.dtype
    device = k_cache.device
    k_cache_shape = k_cache.shape
    v_cache_shape = v_cache.shape

    key = key.to(torch.float).contiguous()
    value = value.to(torch.float).contiguous()

    if asm_layout:
        k_cache = k_cache.view(num_blocks, num_heads, -1).to(torch.float) * k_scale.view(num_blocks, num_heads, 1)
        v_cache = v_cache.view(num_blocks, num_heads, -1).to(torch.float) * v_scale.view(num_blocks, num_heads, 1)
    else:
        k_cache = k_cache.view(num_blocks, num_heads, -1).to(torch.float) *  k_scale.t().view(num_blocks, num_heads, 1)
        v_cache = v_cache.view(num_blocks, num_heads, -1).to(torch.float) *  v_scale.t().view(num_blocks, num_heads, 1)

    # [num_blocks, num_heads, head_size//x, block_size, x]
    k_cache = k_cache.view(k_cache_shape).permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    
    if asm_layout:
        # [num_blocks, num_heads, block_size//x, head_size, x]
        v_cache = v_cache.view(v_cache_shape).permute(0, 2, 4, 1, 3).contiguous().view(-1, num_heads, head_size)
    else:
        # [num_blocks, num_heads, head_size, block_size]
        v_cache = v_cache.view(v_cache_shape).permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)

    k_cache[slot_mapping] = key.view(-1, num_heads, head_size)
    k_cache = k_cache.view(num_blocks, block_size, num_heads, head_size).permute(0, 2, 1, 3).view(num_blocks, num_heads, -1)
    k_cache, k_scale = aiter.pertoken_quant(k_cache, y_scale_dtype=quantCfg['y_scale_dtype'], quant_dtype=quantCfg['quant_dtype'])
    k_cache = k_cache.view(num_blocks, num_heads, block_size, head_size//x, x).permute(0, 1, 3, 2, 4).contiguous()
    k_scale = k_scale.view(num_blocks, num_heads)

    v_cache[slot_mapping] = value.view(-1, num_heads, head_size)
    v_cache = v_cache.view(num_blocks, block_size, num_heads, head_size).permute(0, 2, 1, 3).view(num_blocks, num_heads, -1)
    v_cache, v_scale = aiter.pertoken_quant(v_cache, y_scale_dtype=quantCfg['y_scale_dtype'], quant_dtype=quantCfg['quant_dtype'])
    v_scale = v_scale.view(num_blocks, num_heads)

    if asm_layout:
        v_cache = v_cache.view(num_blocks, num_heads, block_size//x, x, head_size).permute(0, 1, 2, 4, 3).contiguous()
    else:
        k_scale = k_scale.t().contiguous()
        v_scale = v_scale.t().contiguous()
        v_cache = v_cache.view(num_blocks, num_heads, block_size, head_size).permute(0, 1, 3, 2).contiguous()

    return k_cache, v_cache, k_scale, v_scale


@perftest()
def run_aiter(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, block_size, x, asm_layout, quantCfg={}):
    aiter.reshape_and_cache_with_block_quant(
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, asm_layout)
    return k_cache, v_cache, k_scale, v_scale


def test_reshape_and_cache(ctx_lens: int,
                           bs: int,
                           num_heads: Tuple[int, int],
                           head_size: int,
                           block_size: int,
                           DTyoe_KV: torch.dtype,
                           DTyoe_KVCache: torch.dtype,
                           quantCfg: dict = {}
                           ):
    asm_layout = True
    qhead, kvhead = num_heads
    num_blocks = (MAX_TOKEN_SUPPORTED+block_size-1)//block_size
    # num_blocks = (ctx_lens+1+block_size-1)//block_size
    max_token_num_support = num_blocks*block_size
    x = 16 // DTyoe_KVCache.itemsize
    if asm_layout:
        k_cache_shape = (bs*num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (bs*num_blocks, kvhead, block_size//x, head_size, x)
        kv_scale_shape = (bs*num_blocks, kvhead)
    else:
        k_cache_shape = (bs*num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (bs*num_blocks, kvhead, head_size, block_size)
        kv_scale_shape = (kvhead, bs*num_blocks)

    # ##################################################### prefill part
    qkv = torch.randn(
        bs*ctx_lens, qhead+2*kvhead, head_size, dtype=DTyoe_KV, device='cuda')
    _, key, value = torch.split(qkv, [qhead, kvhead, kvhead], dim=1)
    device = key.device
    k_cache = torch.zeros(k_cache_shape, dtype=DTyoe_KVCache, device=device)
    v_cache = torch.zeros(v_cache_shape, dtype=DTyoe_KVCache, device=device)
    k_scale = torch.ones(kv_scale_shape,
                            dtype=quantCfg['y_scale_dtype'], device=key.device)
    v_scale = torch.ones_like(k_scale)
    slot_mapping = torch.tensor([bsID*max_token_num_support+i for bsID in range(bs)
                                for i in range(ctx_lens)]).cuda()

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_scale_ref = k_scale.clone()
    v_scale_ref = v_scale.clone()
    (k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref), us_ref = run_torch(key.view(bs, ctx_lens, kvhead, head_size),
                                                                            value.view(bs, ctx_lens, kvhead, head_size),
                                                                            k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref,
                                                                            slot_mapping, block_size, x, asm_layout, quantCfg)

    k_cache_a = k_cache.clone()
    v_cache_a = v_cache.clone()
    k_scale_a = k_scale.clone()
    v_scale_a = v_scale.clone()
    (k_cache_a, v_cache_a, k_scale_a, v_scale_a), us_a = run_aiter(key.view(bs, ctx_lens, kvhead, head_size), 
                                                                    value.view(bs, ctx_lens, kvhead, head_size), 
                                                                    k_cache_a, v_cache_a, k_scale_a, v_scale_a,
                                                                    slot_mapping, block_size, x, asm_layout, quantCfg)

    print(f'prefill part: ref vs aiter {us_ref:.2f}us vs {us_a:.2f}us')
    slots_edit = torch.unique(slot_mapping // block_size)
    checkAllclose(k_cache_ref[slots_edit].to(torch.float32), k_cache_a[slots_edit].to(torch.float32), msg=f'k_cache {k_cache_ref.shape}')
    checkAllclose(v_cache_ref[slots_edit].to(torch.float32), v_cache_a[slots_edit].to(torch.float32), msg=f'v_cache {v_cache_ref.shape}')
    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()

    checkAllclose(k_scale_ref[slots_edit], k_scale_a[slots_edit], atol=0.001, msg=f'k_scale {k_scale_ref.shape}')
    checkAllclose(v_scale_ref[slots_edit], v_scale_a[slots_edit], atol=0.001, msg=f'v_scale {v_scale_ref.shape}')

    ##################################################### chunk-prefill part
    chunk_left_ctx_lens = 10
    
    qkv = torch.randn(
        bs*chunk_left_ctx_lens, qhead+2*kvhead, head_size, dtype=DTyoe_KV, device='cuda')*2
    _, key, value = torch.split(qkv, [qhead, kvhead, kvhead], dim=1)

    slot_mapping = torch.tensor(
        [bsID*max_token_num_support+ctx_lens+i for bsID in range(bs) for i in range(chunk_left_ctx_lens)]).cuda()

    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()
    
    (k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref), us_ref = run_torch(key.view(bs, chunk_left_ctx_lens, kvhead, head_size),
                                                                            value.view(bs, chunk_left_ctx_lens, kvhead, head_size),
                                                                            k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref,
                                                                            slot_mapping, block_size, x, asm_layout, quantCfg)

    (k_cache_a, v_cache_a, k_scale_a, v_scale_a), us_a = run_aiter(key.view(bs, chunk_left_ctx_lens, kvhead, head_size), 
                                                                    value.view(bs, chunk_left_ctx_lens, kvhead, head_size), 
                                                                    k_cache_a, v_cache_a, k_scale_a, v_scale_a,
                                                                    slot_mapping, block_size, x, asm_layout, quantCfg)

    print(f'chunk-prefill part: ref vs aiter {us_ref:.2f}us vs {us_a:.2f}us')
    slots_edit = torch.unique(slot_mapping // block_size)    
    checkAllclose(k_cache_ref[slots_edit].to(torch.float32), k_cache_a[slots_edit].to(torch.float32), msg=f'k_cache {k_cache_ref.shape}')
    checkAllclose(v_cache_ref[slots_edit].to(torch.float32), v_cache_a[slots_edit].to(torch.float32), msg=f'v_cache {v_cache_ref.shape}')
    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()
    checkAllclose(k_scale_ref[slots_edit], k_scale_a[slots_edit], atol=0.001, msg=f'k_scale {k_scale_ref.shape}')
    checkAllclose(v_scale_ref[slots_edit], v_scale_a[slots_edit], atol=0.001, msg=f'v_scale {v_scale_ref.shape}')

    ##################################################### decode part
    qkv = torch.randn(
        bs, qhead+2*kvhead, head_size, dtype=DTyoe_KV, device='cuda')*2
    _, key, value = torch.split(qkv, [qhead, kvhead, kvhead], dim=1)

    slot_mapping = torch.tensor(
        [bsID*max_token_num_support+ctx_lens+chunk_left_ctx_lens for bsID in range(bs)]).cuda()

    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()
    
    (k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref), us_ref = run_torch(key.view(bs, 1, kvhead, head_size),
                                                                            value.view(bs, 1, kvhead, head_size),
                                                                            k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref,
                                                                            slot_mapping, block_size, x, asm_layout, quantCfg)

    (k_cache_a, v_cache_a, k_scale_a, v_scale_a), us_a = run_aiter(key.view(bs, 1, kvhead, head_size), 
                                                                    value.view(bs, 1, kvhead, head_size), 
                                                                    k_cache_a, v_cache_a, k_scale_a, v_scale_a,
                                                                    slot_mapping, block_size, x, asm_layout, quantCfg)

    print(f'decode part: ref vs aiter {us_ref:.2f}us vs {us_a:.2f}us')
    slots_edit = torch.unique(slot_mapping // block_size)    
    checkAllclose(k_cache_ref[slots_edit].to(torch.float32), k_cache_a[slots_edit].to(torch.float32), msg=f'k_cache {k_cache_ref.shape}')
    checkAllclose(v_cache_ref[slots_edit].to(torch.float32), v_cache_a[slots_edit].to(torch.float32), msg=f'v_cache {v_cache_ref.shape}')
    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()
    checkAllclose(k_scale_ref[slots_edit], k_scale_a[slots_edit], atol=0.001, msg=f'k_scale {k_scale_ref.shape}')
    checkAllclose(v_scale_ref[slots_edit], v_scale_a[slots_edit], atol=0.001, msg=f'v_scale {v_scale_ref.shape}')

    print(
        f'finish test {ctx_lens=} {bs=} {num_heads=} {head_size=} {block_size=} {DTyoe_KV=} {DTyoe_KVCache=}')



print('\nstart quant fp16->fp8')
test_reshape_and_cache(4097, 128, (8, 1), 128, 16,
                       torch.float16, torch.float8_e4m3fnuz, quantCfg={'y_scale_dtype': torch.float,
                                                                       'quant_dtype': torch.float8_e4m3fnuz})
print('\nstart quant fp16->i8')
test_reshape_and_cache(4097, 128, (8, 1), 128, 16,
                       torch.float16, torch.int8, quantCfg={'y_scale_dtype': torch.float,
                                                            'quant_dtype': torch.int8})
print('\nstart quant bf16->i8')
test_reshape_and_cache(4097, 128, (8, 1), 128, 16,
                       torch.bfloat16, torch.int8, quantCfg={'y_scale_dtype': torch.float,
                                                             'quant_dtype': torch.int8})

print('\nstart quant bf16->fp8')
test_reshape_and_cache(4097, 128, (8, 1), 128, 128,
                       torch.bfloat16, torch.float8_e4m3fnuz, quantCfg={'y_scale_dtype': torch.float,
                                                             'quant_dtype': torch.float8_e4m3fnuz})
