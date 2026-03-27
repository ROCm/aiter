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

# from aiter.ops.triton.attention.unified_attention import unified_attention
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.gluon.unified_attention_2d import (
    unified_attention as gluon_unified_attention_2d,
)

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


def generate_data(
    seq_lens,
    num_blocks=32768,
    block_size=32,
    head_size=64,
    num_heads=(16, 2),
    sliding_window=None,
    q_dtype=torch.float8_e4m3fn,
    kv_dtype=torch.bfloat16,
    shuffled_kv_cache=False,
):
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

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.float32, device="cpu"
    ).to(q_dtype)
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
        device="cpu",
    )
    value_cache = torch.randn_like(key_cache).to(kv_dtype)
    key_cache = key_cache.to(kv_dtype)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cpu"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    max_num_blocks_per_seq = (
        min(max_num_blocks_per_seq * num_seqs, num_blocks) // num_seqs
    )
    total_ind_count = num_seqs * max_num_blocks_per_seq
    values = torch.arange(0, total_ind_count, dtype=torch.int)
    values = values[torch.randperm(total_ind_count)]
    block_tables = values.view(num_seqs, max_num_blocks_per_seq).contiguous().to("cpu")

    sinks = torch.randn(num_query_heads, dtype=torch.float32, device="cpu")

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

    return (
        query,
        key_cache,
        value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    )


def calculate_mem_bw(
    batch_size,
    seq_q_l,
    seq_kv_l,
    num_heads_q,
    num_heads_k,
    head_size,
    block_size,
    window_size,
    time_us,
    q_fp8,
    kv_fp8,
):
    if window_size > 0:
        seq_kv_l = min(seq_kv_l, window_size)
    q_bytes = 1 if q_fp8 else 2
    kv_bytes = 1 if kv_fp8 else 2
    Q = seq_q_l * num_heads_q * head_size * q_bytes
    K = V = seq_kv_l * num_heads_k * head_size * kv_bytes
    mem = (Q + K + V + Q) * batch_size
    return (mem / 1e9) / (time_us * 1e-6)


def calculate_tflops(
    batch_size,
    seq_q_l,
    seq_kv_l,
    num_heads_q,
    num_heads_k,
    head_size,
    window_size,
    time_us,
):
    if window_size > 0:
        seq_kv_l = min(seq_kv_l, window_size)
    # FLOPs for QK^T (multiply + add)
    flops_qk = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // 2

    # FLOPs for A x V (multiply + add)
    flops_av = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // 2
    flops_softmax = (5.0 * batch_size * num_heads_q * seq_q_l * seq_kv_l) // 2
    # Total FLOPs
    total_flops = flops_qk + flops_av  # + flops_softmax

    time_s = time_us * 1e-6

    # TFLOPs = total FLOPs / (time in seconds * 1e12)
    tflops = total_flops / (time_s * 1e12)
    return tflops


def create_configs():
    x_names = [
        "num_heads_q",
        "num_heads_k",
        "head_size",
        "seq_q_l",
        "seq_kv_l",
        "bs",
        "window_size",
        "block_size",
        "shuffled_kv_cache",
        "q_fp8",
        "kv_fp8",
        "use_tdm",
        "num_kv_blocks",
    ]
    x_vals = []
    for q_heads in [64, 8]:
        for seq_l in [1024, 2048, 4096, 8192]:
            x_vals.append(
                [q_heads, q_heads // 8, 64, seq_l, seq_l, 1, 0, 64, 0, 0, 0, 1, 1]
            )
    sub_config = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["Gluon"],
        line_names=["Gluon"],
        ylabel="TFLOPS",
        plot_name="Unified Attention",
        args={},
    )

    return [
        sub_config,
    ]


def run_benchmark(configs):
    @triton.testing.perf_report(configs)
    def benchmark(
        num_heads_q,
        num_heads_k,
        head_size,
        seq_q_l,
        seq_kv_l,
        bs,
        window_size,
        block_size,
        shuffled_kv_cache,
        q_fp8,
        kv_fp8,
        use_tdm,
        num_kv_blocks,
        provider,
    ):
        seq_lens = [(seq_q_l, seq_kv_l)] * bs
        q_dtype = torch.bfloat16
        kv_dtype = torch.bfloat16
        if q_fp8 == 1:
            q_dtype = e4m3_dtype
        if kv_fp8 == 1:
            kv_dtype = e4m3_dtype
        (
            maybe_quantized_query,
            maybe_quantized_key_cache,
            maybe_quantized_value_cache,
            sinks,
            output,
            cu_query_lens,
            kv_lens,
            max_query_len,
            max_kv_len,
            scale,
            window_size_tuple,
            block_tables,
            q_descale,
            k_descale,
            v_descale,
            output_scale,
        ) = generate_data(
            seq_lens,
            num_blocks=1024,
            block_size=block_size,
            head_size=head_size,
            num_heads=(num_heads_q, num_heads_k),
            sliding_window=window_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            shuffled_kv_cache=shuffled_kv_cache,
        )
        output_scale = None
        soft_cap = None
        new_kv_layout = num_kv_blocks > 1
        waves_per_eu = 4
        num_warps = 4
        block_m = 128
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
            maybe_quantized_value_cache = maybe_quantized_value_cache.permute(
                0, 2, 1, 3
            )
        maybe_quantized_query = maybe_quantized_query.cuda()
        maybe_quantized_key_cache = maybe_quantized_key_cache.cuda()
        maybe_quantized_value_cache = maybe_quantized_value_cache.cuda()
        cu_query_lens = cu_query_lens.cuda()
        block_tables = block_tables.cuda()
        sinks = sinks.cuda()
        output = output.cuda()
        kv_lens = kv_lens.cuda()
        func = lambda: gluon_unified_attention_2d(
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
            window_size=window_size_tuple,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=sinks,
            output_scale=output_scale,
            use_tdm=use_tdm,
            num_kv_blocks=num_kv_blocks,
            new_kv_layout=new_kv_layout,
            waves_per_eu=waves_per_eu,
            shuffled_kv_cache=shuffled_kv_cache,
            num_warps=num_warps,
            block_m=block_m,
        )
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=quantiles)
        perf = lambda ms: calculate_tflops(
            bs,
            seq_q_l,
            seq_kv_l,
            num_heads_q,
            num_heads_k,
            head_size,
            window_size,
            ms * 1000.0,
        )
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(print_data=True)


def main():
    configs = create_configs()
    run_benchmark(configs)


if __name__ == "__main__":
    sys.exit(main())
