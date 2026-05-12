# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for ``rope_norm_store_kv_fp8``.

Measures the fused kernel under both prefill-like (mixed-length new tokens
per request) and decode-like (1 new token per request) regimes. Reports
time / bandwidth / throughput.

Examples:
    # default prefill sweep
    python bench_fused_rope_norm_store_kv_fp8.py --mode prefill --metric bandwidth

    # custom decode shape
    python bench_fused_rope_norm_store_kv_fp8.py --mode decode --num-req 64 \\
        --num-q-heads 16 --num-kv-heads 4 --head-dim 128 --block-size 64

    # one-off shape
    python bench_fused_rope_norm_store_kv_fp8.py --shape 8 1 128 64 64 1024
"""

import argparse

import torch
import triton

from aiter.ops.triton.fusions.fused_rope_norm_store_kv_fp8 import (
    rope_norm_store_kv_fp8,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)


def _make_inputs(
    num_req: int,
    new_tokens_per_req: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    fp8_dtype: torch.dtype,
    device: str = "cuda",
):
    n_new_list = [new_tokens_per_req] * num_req
    seq_lens = [seq_len] * num_req
    num_rows = sum(n_new_list)

    max_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = max_blocks * num_req

    hidden = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
    qkv = torch.randn(num_rows, hidden, dtype=torch.bfloat16, device=device)
    cos_sin = torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)
    num_seqlen_per_req = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    q_index = torch.zeros(num_req + 1, dtype=torch.int32, device=device)
    q_index[1:] = torch.tensor(
        n_new_list, dtype=torch.int32, device=device
    ).cumsum(0)

    kvcache_indices = torch.zeros(
        num_req, max_blocks, dtype=torch.int32, device=device
    )
    cursor = 0
    for r in range(num_req):
        kvcache_indices[r] = torch.arange(cursor, cursor + max_blocks, device=device)
        cursor += max_blocks

    key_cache = torch.zeros(
        total_blocks, block_size, num_kv_heads, head_dim,
        dtype=fp8_dtype, device=device,
    )
    value_cache = torch.zeros_like(key_cache)
    k_scale = torch.zeros(
        total_blocks, block_size // 32, num_kv_heads, 32,
        dtype=torch.float32, device=device,
    )
    v_scale = (torch.rand(num_kv_heads, dtype=torch.float32, device=device) + 0.5) * 0.05
    q_norm_w = torch.ones(head_dim, dtype=torch.float32, device=device)
    k_norm_w = torch.ones(head_dim, dtype=torch.float32, device=device)

    return dict(
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index,
        kvcache_indices=kvcache_indices,
        key_cache=key_cache,
        value_cache=value_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        num_rows=num_rows,
    )


def _bench_one(args, num_req, n_new, seq_len, fp8_dtype):
    inp = _make_inputs(
        num_req,
        n_new,
        seq_len,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        args.block_size,
        fp8_dtype,
    )
    is_prefill = n_new > 1

    fn = lambda: rope_norm_store_kv_fp8(  # noqa: E731
        qkv=inp["qkv"],
        cos_sin=inp["cos_sin"],
        num_seqlen_per_req=inp["num_seqlen_per_req"],
        q_index=inp["q_index"],
        kvcache_indices=inp["kvcache_indices"],
        key_cache=inp["key_cache"],
        value_cache=inp["value_cache"],
        k_scale=inp["k_scale"],
        v_scale=inp["v_scale"],
        is_prefill=is_prefill,
        max_seqlens=n_new,
        q_norm_weight=inp["q_norm_w"],
        k_norm_weight=inp["k_norm_w"],
        qk_norm_policy=args.qk_norm_policy,
        is_neox=True,
        rms_eps=1e-6,
    )

    # warm-up + timing
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)

    # Memory accounting:
    # reads:  qkv (bf16 = 2B) over num_rows*hidden
    #         cos/sin: num_rows * head_dim * 4B
    #         norm weights: 2 * head_dim * 4B (constant)
    #         v_scale: KH * 4B
    # writes: out_q (fp8 = 1B) over num_rows * QH * D
    #         q_scale_flat (fp32) over num_rows * QH
    #         key/value cache (fp8) over num_rows * KH * D each
    #         k_scale slab (fp32) over num_rows * KH
    num_rows = inp["num_rows"]
    qh, kh, d = args.num_q_heads, args.num_kv_heads, args.head_dim
    hidden = qh * d + 2 * kh * d
    bytes_read = (
        num_rows * hidden * 2          # qkv bf16
        + num_rows * d * 4              # cos+sin row
        + 2 * d * 4                     # norm weights
        + kh * 4                        # v_scale
    )
    bytes_write = (
        num_rows * qh * d * 1           # out_q fp8
        + num_rows * qh * 4             # q_scale flat
        + 2 * num_rows * kh * d * 1     # K + V cache fp8
        + num_rows * kh * 4             # k_scale slab
    )
    mem = bytes_read + bytes_write

    # FLOP estimate: RoPE ~4 ops/elem, RMSNorm ~3 ops/elem, quant ~2 ops/elem
    elems_q = num_rows * qh * d
    elems_k = num_rows * kh * d
    elems_v = num_rows * kh * d
    rope_flops = (elems_q + elems_k) * 4
    rms_flops = (elems_q + elems_k) * 3 if args.qk_norm_policy != 0 else 0
    quant_flops = (elems_q + elems_k + elems_v) * 2
    total_flops = rope_flops + rms_flops + quant_flops

    return ms, mem, total_flops


def get_x_vals_prefill():
    return [
        ("prefill_b1_t128_s512",  1, 128,  512),
        ("prefill_b1_t512_s2048", 1, 512,  2048),
        ("prefill_b4_t256_s1024", 4, 256,  1024),
        ("prefill_b8_t128_s512",  8, 128,  512),
        ("prefill_b8_t512_s2048", 8, 512,  2048),
    ]


def get_x_vals_decode():
    return [
        ("decode_b1_s512",    1,   1, 512),
        ("decode_b8_s512",    8,   1, 512),
        ("decode_b32_s512",  32,   1, 512),
        ("decode_b32_s2048", 32,   1, 2048),
        ("decode_b128_s1024", 128, 1, 1024),
    ]


def run_benchmark(args):
    fp8_dtype = get_fp8_e4m3_dtype()

    if args.shape is not None:
        qh, kh, d, bs, n_new, sl = args.shape
        args.num_q_heads = qh
        args.num_kv_heads = kh
        args.head_dim = d
        args.block_size = bs
        x_vals = [("custom", 1, n_new, sl)]
    elif args.mode == "prefill":
        x_vals = get_x_vals_prefill()
    else:
        x_vals = get_x_vals_decode()

    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    elif args.metric == "throughput":
        ylabel = "Throughput_(TFLOPS)"
    else:
        raise NotImplementedError(args.metric)

    bench = triton.testing.Benchmark(
        x_names=["case", "num_req", "n_new", "seq_len"],
        x_vals=x_vals,
        line_arg="unit",
        line_vals=[ylabel],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([bench])
    def _bench(case, num_req, n_new, seq_len, **_):
        ms, mem, flops = _bench_one(args, num_req, n_new, seq_len, fp8_dtype)
        if args.metric == "time":
            return ms
        elif args.metric == "bandwidth":
            return mem / (ms * 1e-3) * 1e-9
        else:
            return flops / ms * 1e-9

    _bench.run(save_path="." if args.o else None, print_data=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="bench_fused_rope_norm_store_kv_fp8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    p.add_argument("--num-q-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--qk-norm-policy", type=int, choices=[0, 1, 2], default=1)
    p.add_argument(
        "--shape", type=int, nargs=6,
        metavar=("QH", "KH", "D", "BLOCK", "N_NEW", "SEQLEN"),
        help="Custom shape (overrides --mode sweep): QH KH D BLOCK n_new seq_len.",
    )
    p.add_argument(
        "--metric", choices=["time", "bandwidth", "throughput"],
        default="bandwidth",
    )
    p.add_argument("-print_vgpr", action="store_true", default=False)
    p.add_argument("-o", action="store_true", help="Save CSV of results")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.print_vgpr:
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
