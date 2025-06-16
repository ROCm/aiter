# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, perftest, run_perftest
import itertools
import argparse
import random
import math
import triton

Block_M = 64
Block_N = 16 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

def get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k):
    batch_size = cache_seqlens.size(0)
    cu_count = torch.cuda.get_device_properties(
        device='cuda').multi_processor_count
    cu_parts = int(cu_count / num_heads_k / ((num_heads_per_head_k + Block_M - 1) // Block_M))

    tile_scheduler_metadata = torch.zeros([cu_parts, 5]).to(torch.int)
    num_splits = torch.zeros([batch_size + 1]).to(torch.int)

    fixed_overhead_num_blocks = 5;

    num_blocks = cache_seqlens // Block_N
   
    sum_blocks = num_blocks.sum() + fixed_overhead_num_blocks * batch_size
    payload = (sum_blocks + cu_parts - 1) // cu_parts + fixed_overhead_num_blocks

    now_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0

    num_splits[0] = 0
    for i in range(cu_parts):
        tile_scheduler_metadata[i, 0] = now_idx
        tile_scheduler_metadata[i, 1] = now_block * Block_N 
        tile_scheduler_metadata[i, 4] = now_n_split_idx
        remain_payload = payload
        while now_idx < batch_size:
            num_block = num_blocks[now_idx]
            now_remain_blocks = num_block - now_block

            if remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
                cum_num_splits = cum_num_splits + now_n_split_idx + 1
                num_splits[now_idx + 1] = cum_num_splits
                remain_payload = remain_payload - (now_remain_blocks + fixed_overhead_num_blocks)
                now_idx = now_idx + 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - fixed_overhead_num_blocks > 0:
                    now_block += remain_payload - fixed_overhead_num_blocks
                    now_n_split_idx = now_n_split_idx + 1
                    remain_payload = 0
                break
        if now_block > 0:
            tile_scheduler_metadata[i, 2] = now_idx
            tile_scheduler_metadata[i, 3] = now_block * Block_N 
        else:
            tile_scheduler_metadata[i, 2] = now_idx - 1
            tile_scheduler_metadata[i, 3] = cache_seqlens[now_idx - 1]

    return tile_scheduler_metadata, num_splits


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1)
    attn_weight = attn_weight / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device="cuda")
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device="cuda").tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < 1e-3


@torch.inference_mode()
def test_flash_mla(dtype, b, s_q, mean_sk, h_q, h_kv, d, dv, page_block_size, causal, varlen, test_quality, test_perf):
    if not test_quality and not test_perf:
        return

    print(
        f"{dtype=}, {b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {page_block_size=}, {causal=}, {varlen=}"
    )

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32, device="cuda")
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    # max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    max_seqlen_pad = int(math.ceil(max_seqlen / 256) * 256)
    print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d, device="cuda", dtype=dtype)

    block_table = torch.arange(
        b * max_seqlen_pad // page_block_size, dtype=torch.int32, device="cuda"
    ).view(b, max_seqlen_pad // page_block_size)
    blocked_k = torch.randn(block_table.numel(), page_block_size, h_kv, d, device="cuda", dtype=dtype)
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            float("nan")
        )
    blocked_v = blocked_k[..., :dv]

    # tile_scheduler_metadata, num_splits = aiter.get_mla_metadata(
    #     cache_seqlens, s_q * h_q // h_kv, h_kv
    # )

    # tile_scheduler_metadata_torch, num_splits_torch = get_mla_metadata(
    #     cache_seqlens, s_q * h_q // h_kv, h_kv
    # )

    def flash_mla():
        # return aiter.flash_mla_fwd_with_kvcache(
        #     q,
        #     blocked_k,
        #     block_table,
        #     cache_seqlens,
        #     dv,
        #     tile_scheduler_metadata,
        #     num_splits,
        #     causal=causal,
        # )
        return aiter.flash_mla_fwd_prefill_with_kvcache(q, blocked_k, block_table, cache_seqlens, dv, causal=causal)

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device="cuda")
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device="cuda")
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = torch.nan_to_num(O.transpose(0, 1))
            lse[i] = LSE
        return out, lse

    if test_quality:
        out_torch, lse_torch = ref_mla()
        out_flash, lse_flash = flash_mla()
        checkAllclose(lse_flash, lse_torch, msg="lse")
        checkAllclose(out_flash, out_torch.to(dtype=dtype), msg="out")

    if test_perf:
        _, t = run_perftest(flash_mla,
                            num_iters=2,
                            num_warmup=0)
        FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
        bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
            torch.finfo(q.dtype).bits // 8
        )
        print(
            f"{t:.4f} us, {FLOPS / 10 ** 6 / t:.4f} TFLOPS, {bytes / 10 ** 3 / t:.4f} GB/s"
        )

    print("====================================")


if __name__ == "__main__":
    h_kv = 1
    d, dv = 576, 512

    for (dtype, b, s, h_q, s_q, page_block_size, varlen, causal) in itertools.product(
        (torch.float16, torch.bfloat16)[1:],
        [1, 3, 5, 16, 32, 64, 128, 256][3:4],
        [21, 64, 256, 512, 1200, 3200, 5200, 8192][:],
        (16, 64, 128)[:],
        # (1, 2), # s_q for decode
        (64,),  # s_q for prefill
        (1, 16, 64)[:],
        (False, True)[:],
        (False, True)[:]
    ):
        test_flash_mla(dtype, b, s_q, s, h_q, h_kv, d, dv, page_block_size, causal, varlen, True, False)

    # for (dtype, b, s, h_q, page_block_size, varlen, causal) in itertools.product(
    #     (torch.float16, torch.bfloat16)[1:],
    #     [1, 3, 5, 16, 32, 64, 128, 256][3:4],
    #     [21, 64, 256, 512, 1200, 3200, 5200, 8192][:],
    #     (16, 64, 128)[:1],
    #     (1, 16, 64)[:1],
    #     (False, True)[:1],
    #     (False, True)[1:]
    # ):
    #     test_flash_mla(dtype, b, s, s, h_q, h_kv, d, dv, page_block_size, causal, varlen, False, True)
