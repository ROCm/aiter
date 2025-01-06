import torch
import torch.nn.functional as F
import ater
from ater.test_common import checkAllclose, perftest
from typing import List, Optional, Tuple, Union

MAX_TOKEN_SUPPORTED = 16384


@perftest()
def run_torch(key, value, k_cache, v_cache, slot_mapping, block_size, x, asm_layout, quantCfg={}):
    num_batch, num_tokens, num_heads, head_size = key.shape
    num_blocks = k_cache.shape[0]
    dtype = k_cache.dtype
    device = k_cache.device

    k_scale = None
    v_scale = None

    if quantCfg:
        key, k_scale = ater.pertoken_quant(key.contiguous(), **quantCfg)
        k_scale = k_scale.permute(2, 0, 1, 3).view(
            num_heads, num_tokens).contiguous()

    k_cache = k_cache.permute(0, 3, 1, 2, 4).contiguous().view(
        -1, num_heads, head_size)
    for i, batch in enumerate(key):
        # for K ..............................
        # [num_tokens, num_heads, head_size]
        # to
        # [num_blocks, num_heads, head_size/x, block_size, x]
        num_tokens = batch.shape[0]
        slotID = slot_mapping[i]
        k_cache[slotID:slotID+num_tokens] = batch.contiguous()
    k_cache = k_cache.view(num_blocks, block_size,
                           num_heads,
                           head_size//x, x).permute(0, 2, 3, 1, 4)

    if quantCfg:
        value, v_scale = ater.pertoken_quant(value.contiguous(), **quantCfg)
        v_scale = v_scale.permute(2, 0, 1, 3).view(
            num_heads, num_tokens).contiguous()
    if asm_layout:
        v_cache = v_cache.permute(0, 2, 4, 1, 3).contiguous().view(
            -1, num_heads, head_size)
    else:
        v_cache = v_cache.permute(0, 3, 1, 2).contiguous().view(
            -1, num_heads, head_size)
    for i, batch in enumerate(value):
        # for V ..............................
        # [num_tokens, num_heads, head_size]
        # to
        # [num_blocks, num_heads, block_size/X, head_size, X]
        num_tokens = batch.shape[0]
        slotID = slot_mapping[i]
        v_cache[slotID:slotID+num_tokens] = batch.contiguous()
    if asm_layout:
        v_cache = v_cache.view(num_blocks, block_size//x, x,
                               num_heads,
                               head_size).permute(0, 3, 1, 4, 2)
    else:
        # [num_blocks, num_heads, head_size, block_size]
        v_cache = v_cache.view(num_blocks, block_size,
                               num_heads,
                               head_size).permute(0, 2, 3, 1)

    return k_cache, v_cache, k_scale, v_scale


@perftest()
def run_ater(key, value, k_cache, v_cache, slot_mapping, block_size, x, asm_layout, quantCfg={}):
    if quantCfg:
        num_tokens, num_heads, head_size = key.shape
        k_scale = torch.empty(num_heads, num_tokens,
                              dtype=quantCfg['y_scale_dtype'], device=key.device)
        v_scale = torch.empty_like(k_scale)
        ater.reshape_and_cache_with_pertoken_quant(
            key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, asm_layout)
    else:
        k_scale = None
        v_scale = None
        ater.reshape_and_cache(
            key, value, k_cache, v_cache, slot_mapping, 'auto', 1.0, 1.0, asm_layout)
    return k_cache, v_cache, k_scale, v_scale


def test_reshape_and_cache(ctx_lens: int,
                           num_seqs: int,
                           num_heads: Tuple[int, int],
                           head_size: int,
                           block_size: int,
                           DTyoe_KV: torch.dtype,
                           DTyoe_KVCache: torch.dtype,
                           quantCfg: dict = {}
                           ):
    asm_layout = True
    qhead, kvhead = num_heads
    # num_blocks = (MAX_TOKEN_SUPPORTED+block_size-1)//block_size
    num_blocks = (ctx_lens+block_size-1)//block_size
    x = 16 // DTyoe_KVCache.itemsize
    if asm_layout:
        k_cache_shape = (num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (num_blocks, kvhead, block_size//x, head_size, x)
    else:
        k_cache_shape = (num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (num_blocks, kvhead, head_size, block_size)

    # prefill part
    qkv = torch.randn(
        ctx_lens, qhead+2*kvhead, head_size, dtype=DTyoe_KV, device='cuda')
    _, key, value = torch.split(qkv, [qhead, kvhead, kvhead], dim=1)
    dtype = key.dtype
    device = key.device
    k_cache = torch.empty(k_cache_shape, dtype=DTyoe_KVCache, device=device)
    v_cache = torch.empty(v_cache_shape, dtype=DTyoe_KVCache, device=device)
    slot_mapping = torch.tensor([i for i in range(ctx_lens)]).cuda()

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    out_ref, us_ref = run_torch(key.view(1, ctx_lens, kvhead, head_size),
                                value.view(1, ctx_lens, kvhead, head_size),
                                k_cache_ref, v_cache_ref,
                                slot_mapping, block_size, x, asm_layout, quantCfg)

    k_cache_a = k_cache.clone()
    v_cache_a = v_cache.clone()
    out_a, us_a = run_ater(key, value, k_cache_a, v_cache_a,
                           slot_mapping, block_size, x, asm_layout, quantCfg)

    names = ['k_cache', 'v_cache', 'k_scale', 'v_scale']
    for i, el in enumerate(out_ref):
        if el is None:
            continue
        print(names[i],el.shape, out_a[i].shape)
        checkAllclose(el, out_a[i],
                      msg=f'ref vs ater {us_ref}us vs {us_a}us ')


test_reshape_and_cache(4097, 128, (8, 1), 128, 16,
                       torch.bfloat16, torch.bfloat16)
print('start quant')
# test_reshape_and_cache(4097, 128, (8, 1), 128, 16,
#                        torch.bfloat16, torch.float8_e4m3fnuz, quantCfg={'y_scale_dtype': torch.float,
#                                                                         'quant_dtype': torch.int8})
test_reshape_and_cache(4097, 128, (8, 1), 128, 16,
                       torch.bfloat16, torch.int8, quantCfg={'y_scale_dtype': torch.float,
                                                             'quant_dtype': torch.int8})
