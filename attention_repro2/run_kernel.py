# import hip
# hip.hip.hipInit(0)
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
from unified_attention_w_3d import unified_attention
#from gluon_unified_attention_w_shared_stage import unified_attention
#from aiter.ops.triton.attention.unified_attention import unified_attention
#from unified_attention_new_layout import unified_attention
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

    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_blocks_v, block_size_v, num_kv_heads_v, head_size_v = value_cache.shape
    assert block_size >= 16
    assert num_blocks == num_blocks_v
    assert num_kv_heads == num_kv_heads_v
    assert head_size == head_size_v
    assert block_size == block_size_v

    k_width = 16 // key_cache.element_size()
    key_cache_shuffled = key_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 3, 1)
    key_cache_shuffled = key_cache_shuffled.view(
        -1,
        num_kv_heads,
        head_size // k_width,
        k_width,
        block_size,
    )
    key_cache_shuffled = key_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // k_width,
        k_width,
        head_size,
    )
    value_cache_shuffled = value_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    return key_cache_shuffled, value_cache_shuffled

def generate_data(seq_lens, num_blocks=32768, block_size=32, head_size=64, num_heads=(16, 2), sliding_window=None,
                  q_dtype=torch.float8_e4m3fn, kv_dtype=torch.bfloat16, shuffled_kv_cache=False, remove_indirect_access=False):
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

    if num_blocks == None:
        total_kv =max_kv_len * num_seqs + block_size  
        num_blocks = (total_kv // block_size)

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
    # NOTE: assume single batch
    if remove_indirect_access:
        inds = torch.arange(max_num_blocks_per_seq)
        block_tables[:] = inds    
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
parser.add_argument('--window_size', type=int, default=0, help='')
parser.add_argument('--block_size', type=int, default=16, help='')
parser.add_argument('--repeat', type=int, default=1000, help='0 => single run (no warmup); >0 => warmup+repeat, profiled')
parser.add_argument('--warmup', type=int, default=10, help='warmup iterations before the profiled run (repeat>0 only)')
parser.add_argument('--cache_size', type=str, default="512*1024*1024", help='')
parser.add_argument('--use_tdm', type=int, default=1, help='')
parser.add_argument('--waves_per_eu', type=int, default=-1, help='<=0 => kernel auto-selects')
parser.add_argument('--shuffled_kv_cache', type=int, default=0, help='')
parser.add_argument('--num_warps', type=int, default=-1, help='<=0 => kernel auto-selects')
parser.add_argument('--tile_size', type=int, default=-1, help='<=0 => kernel auto-selects')
parser.add_argument('--q_fp8', type=int, default=0, help='')
parser.add_argument('--kv_fp8', type=int, default=0, help='')
parser.add_argument('--block_m', type=int, default=128, help='')
parser.add_argument('--causal', type=int, default=0, help='')
parser.add_argument('--num_buffers', type=int, default=3, help='')
parser.add_argument('--remove_indirect_access', type=int, default=0, help='')
parser.add_argument('--loop_variant', type=int, default=-1, help='<0 => kernel auto-selects (0/1/2 force a variant)')
parser.add_argument('--num_splits', type=int, default=1, help='Split-KV factor (1 = original 2d behavior)')


args = parser.parse_args()
print(args)

cache_size = eval(args.cache_size)
cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
clear_cache = lambda: cache.zero_()

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
            q_descale, k_descale, v_descale, output_scale) = generate_data(seq_lens, num_blocks=None, block_size=block_size, head_size=args.head_size, 
                                                             num_heads=(args.num_heads_q, args.num_heads_k), sliding_window=args.window_size,
                                                             q_dtype=q_dtype, kv_dtype=kv_dtype, shuffled_kv_cache=args.shuffled_kv_cache, remove_indirect_access=args.remove_indirect_access)
output_scale = None
print("After generate data")
print("--------------------------------")

if output_scale is not None:
    output_scale = output_scale.cuda()
if q_descale is not None:
    q_descale = q_descale.cuda()
if k_descale is not None:
    k_descale = k_descale.cuda()
if v_descale is not None:
    v_descale = v_descale.cuda()

# Move inputs to the GPU once so the benchmark loop measures only the kernel.
q_gpu = maybe_quantized_query.cuda()
k_gpu = maybe_quantized_key_cache.cuda()
v_gpu = maybe_quantized_value_cache.cuda()
out_gpu = output.cuda()
cu_query_lens_gpu = cu_query_lens.cuda()
kv_lens_gpu = kv_lens.cuda()
block_tables_gpu = block_tables.cuda()
sinks_gpu = sinks.cuda()

func = lambda: unified_attention(
        q=q_gpu,
        k=k_gpu,
        v=v_gpu,
        out=out_gpu,
        cu_seqlens_q=cu_query_lens_gpu,
        seqused_k=kv_lens_gpu,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=args.causal,
        window_size=window_size,
        block_table=block_tables_gpu,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks_gpu,
        output_scale=output_scale,
        waves_per_eu=args.waves_per_eu if args.waves_per_eu > 0 else None,
        shuffled_kv_cache=args.shuffled_kv_cache,
        num_warps=args.num_warps if args.num_warps > 0 else None,
        tile_size=args.tile_size if args.tile_size > 0 else None,
        block_m=args.block_m,
        remove_indirect_access=args.remove_indirect_access,
        loop_variant=args.loop_variant if args.loop_variant >= 0 else None,
        num_splits=args.num_splits,
        num_buffers=args.num_buffers if args.num_buffers >= 0 else None,
    )


KERNEL_NAME = "_unified_attention_gluon_kernel_2d"


def compute_flops_bytes():
    """FLOPs and bytes moved for one full kernel invocation.

    Mirrors collect_results.py: sliding window caps KV at window_size, causal
    halves the matmul FLOPs (div=2), output is counted as bf16 (2 bytes). The
    sum over seq_lens replaces the homogeneous batch_size factor.
    """
    head_size = args.head_size
    nq, nk = args.num_heads_q, args.num_heads_k
    q_bytes = 1 if args.q_fp8 else 2
    kv_bytes = 1 if args.kv_fp8 else 2
    div = 2 if args.causal else 1

    total_flops = 0.0
    total_bytes = 0
    for q_len, kv_len in seq_lens:
        eff_kv = min(kv_len, args.window_size) if args.window_size > 0 else kv_len
        # QK^T + A@V, 2 flops/MAC each; causal ~ half the triangle.
        total_flops += (4.0 * q_len * eff_kv * nq * head_size) / div
        Q = q_len * nq * head_size * q_bytes
        K = V = eff_kv * nk * head_size * kv_bytes
        out = q_len * nq * head_size * 2  # output is always bf16
        total_bytes += Q + K + V + out
    return total_flops, total_bytes


def kernel_device_time_ms():
    """Per-call device time (ms) of our kernel, summed from the profiler trace."""
    from torch.profiler import profile, ProfilerActivity

    def dev_time(evt):
        return getattr(evt, "self_device_time_total",
                       getattr(evt, "self_cuda_time_total", 0.0))

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            clear_cache()
            func()
        torch.cuda.synchronize()

    total_us, count = 0.0, 0
    for evt in prof.key_averages():
        if KERNEL_NAME in evt.key:
            total_us += dev_time(evt)
            count += evt.count
    if count == 0:
        seen = sorted({e.key for e in prof.key_averages() if dev_time(e) > 0})
        raise RuntimeError(
            f"Kernel '{KERNEL_NAME}' not found in profile. Device kernels seen:\n"
            + "\n".join(seen))
    return (total_us / count) / 1000.0


def run_kernel():
    if repeat == 0:
        # Mode 1: single run -- no warmup, no profiling.
        func()
        torch.cuda.synchronize()
        print("Single run complete.")
        return

    # Mode 2: warmup, then profiled repeat with cache flushing between calls.
    for _ in range(args.warmup):
        clear_cache()
        func()
    torch.cuda.synchronize()

    ms = kernel_device_time_ms()
    flops, nbytes = compute_flops_bytes()
    sec = ms * 1e-3
    print("--------------------------------")
    print(f"Time:    {ms:.4f} ms")
    print(f"TFLOP/s: {flops / sec / 1e12:.2f}")
    print(f"BW:      {nbytes / sec / 1e9:.2f} GB/s")


run_kernel()
