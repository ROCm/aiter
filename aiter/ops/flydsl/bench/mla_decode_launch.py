#!/usr/bin/env python3
"""Launcher for the FlyDSL MLA decode kernel — used by perf_mla_decode.sh for
roccap capture. Reuses the test's input generator + shuffle helper; the cache is
always pre-shuffled before dispatch (the kernel only consumes the shuffled
layout). Emits machine-parseable METRIC lines so the perf shell can compute
bandwidth/TFLOPS.

MLA dims are fixed by the model (d_c=512, d_rope=64, num_kv_heads=1); the tunables
(num_segs, num_warps, kv_compute_block_size) and the decode shape (num_seqs,
seq_len, num_q_heads, dtype, varlen) are CLI flags.

Examples:
    python mla_decode_launch.py                                       # defaults
    python mla_decode_launch.py --num-seqs 8 --seq-len 4096 --num-warps 2
"""

import os
import sys

_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

import flydsl  # noqa: E402,F401 — preload comgr before torch/HIP
import math
import torch

from aiter.ops.flydsl.mla_decode import flydsl_mla_decode
from aiter.ops.flydsl.tests.test_flydsl_mla_decode import (
    _generate_inputs,
    shuffle_kv_buffer,
)


def _compute_metrics(
    seq_lens_list,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_segs: int,
    kv_compute_block_size: int,
    elem_bytes: int,
):
    """Byte/FLOP metrics for one MLA decode launch (main + reduce)."""
    num_seqs = len(seq_lens_list)
    d_c = kv_lora_rank
    d_rope = qk_rope_head_dim
    qk_head_dim = d_c + d_rope

    sum_seq_lens = sum(seq_lens_list)
    aligned_tokens = sum(
        ((s + block_size - 1) // block_size) * block_size for s in seq_lens_list
    )

    # gluon-style segmentation: NUM_SEGS fixed, tiles_per_seg derived per seq.
    def _num_segs_actual(s):
        num_tiles = (s + kv_compute_block_size - 1) // kv_compute_block_size
        tps = max(1, (num_tiles + num_segs - 1) // num_segs)
        return (num_tiles + tps - 1) // tps
    live_segs = [_num_segs_actual(s) for s in seq_lens_list]
    total_live_segs = sum(live_segs)
    num_segs_max = max(live_segs) if live_segs else 0

    total_live_blocks = sum((s + block_size - 1) // block_size for s in seq_lens_list)

    # ---- per-tensor bytes ----
    # Q: [num_seqs, num_q_heads, d_c + d_rope]
    bytes_q = num_seqs * num_q_heads * qk_head_dim * elem_bytes
    # ONE shared KV cache of width (d_c + d_rope) (no separate K/V).
    bytes_kv_useful = num_kv_heads * qk_head_dim * elem_bytes * sum_seq_lens
    bytes_kv_executed = num_kv_heads * qk_head_dim * elem_bytes * aligned_tokens
    # Per-segment partial output is f32, in d_c space.
    bytes_tmp_out = total_live_segs * num_q_heads * d_c * 4
    # max_logits + exp_sums, both f32.
    bytes_lse = 2 * total_live_segs * num_q_heads * 4
    bytes_block_tables = total_live_blocks * 4
    bytes_seq_lens = num_seqs * 4
    # Final output: [num_seqs, num_q_heads, d_c], bf16/f16.
    bytes_out_final = num_seqs * num_q_heads * d_c * elem_bytes

    bytes_main_useful = (
        bytes_q + bytes_kv_useful + bytes_tmp_out + bytes_lse
        + bytes_block_tables + bytes_seq_lens
    )
    bytes_main_executed = (
        bytes_q + bytes_kv_executed + bytes_tmp_out + bytes_lse
        + bytes_block_tables + bytes_seq_lens
    )

    bytes_reduce_in = bytes_tmp_out + bytes_lse + bytes_seq_lens
    bytes_reduce_total = bytes_reduce_in + bytes_out_final

    bytes_combined_useful = bytes_main_useful + bytes_out_final
    bytes_combined_executed = bytes_main_executed + bytes_out_final

    # FLOPs: QK over (d_c + d_rope) + PV over d_c, per (seq, q-head).
    total_flops = 2 * num_q_heads * sum_seq_lens * (qk_head_dim + d_c)

    return {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "qk_head_dim": qk_head_dim,
        "sum_seq_lens": sum_seq_lens,
        "aligned_tokens": aligned_tokens,
        "total_live_segs": total_live_segs,
        "num_segs_max": num_segs_max,
        "total_live_blocks": total_live_blocks,
        "bytes_q": bytes_q,
        "bytes_kv_useful": bytes_kv_useful,
        "bytes_kv_executed": bytes_kv_executed,
        "bytes_tmp_out": bytes_tmp_out,
        "bytes_lse": bytes_lse,
        "bytes_block_tables": bytes_block_tables,
        "bytes_seq_lens": bytes_seq_lens,
        "bytes_out_final": bytes_out_final,
        "bytes_main_useful": bytes_main_useful,
        "bytes_main_executed": bytes_main_executed,
        "bytes_reduce_in": bytes_reduce_in,
        "bytes_reduce_total": bytes_reduce_total,
        "bytes_combined_useful": bytes_combined_useful,
        "bytes_combined_executed": bytes_combined_executed,
        "total_flops": total_flops,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch FlyDSL MLA decode kernels")
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--kv-block-size", type=int, default=32)
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-segs", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=2)
    parser.add_argument("--kv-compute-block-size", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "f16"])
    parser.add_argument("--varlen", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    elem_bytes = 2
    num_kv_heads = 1

    # Generate inputs on the HOST. torch.randn on the FFM-simulated device runs
    # the RNG element-by-element (~66s for the 4x8192 cache, ~3.3s/M elements);
    # CPU RNG is instant. The finished tensors are copied to the GPU below in one
    # H2D transfer each. Values are irrelevant for a benchmark, so CPU-vs-CUDA
    # RNG divergence does not matter here.
    query, kv_cache, block_tables, seq_lens = _generate_inputs(
        num_seqs=args.num_seqs,
        num_query_heads=args.num_q_heads,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        block_size=args.kv_block_size,
        ctx_len=args.seq_len,
        dtype=dtype,
        varlen=args.varlen,
        seed=args.seed,
        device="cpu",
    )

    seq_lens_list = seq_lens.tolist()
    metrics = _compute_metrics(
        seq_lens_list,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        block_size=args.kv_block_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=num_kv_heads,
        num_segs=args.num_segs,
        kv_compute_block_size=args.kv_compute_block_size,
        elem_bytes=elem_bytes,
    )

    tag = (
        f"dc{args.kv_lora_rank}_dr{args.qk_rope_head_dim}_nqh{args.num_q_heads}"
        f"_kvb{args.kv_block_size}_ns{args.num_seqs}_sl{args.seq_len}"
        f"_seg{args.num_segs}_nw{args.num_warps}_kc{args.kv_compute_block_size}"
    )

    # Machine-parseable METRIC lines (consumed by perf_mla_decode.sh).
    print(f"METRIC tag={tag}")
    print(f"METRIC kv_lora_rank={args.kv_lora_rank}")
    print(f"METRIC qk_rope_head_dim={args.qk_rope_head_dim}")
    print(f"METRIC kv_block_size={args.kv_block_size}")
    print(f"METRIC num_seqs={args.num_seqs}")
    print(f"METRIC seq_len={args.seq_len}")
    print(f"METRIC num_segs={args.num_segs}")
    print(f"METRIC num_warps={args.num_warps}")
    print(f"METRIC kv_compute_block_size={args.kv_compute_block_size}")
    print(f"METRIC dtype={args.dtype}")
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")

    print(
        f"Compiling MLA decode: d_c={args.kv_lora_rank}, "
        f"d_rope={args.qk_rope_head_dim}, nqh={args.num_q_heads}, "
        f"kvb={args.kv_block_size}, nseqs={args.num_seqs}, seq_len={args.seq_len}, "
        f"num_segs={args.num_segs}, num_warps={args.num_warps}, "
        f"kv_compute={args.kv_compute_block_size}, dtype={args.dtype}, "
        f"varlen={args.varlen}"
    )

    # Pre-shuffle on the host (a gather over the cache on the sim is as slow as
    # the RNG); then copy the finished kernel inputs to the GPU (one H2D each).
    kernel_kv_cache = shuffle_kv_buffer(kv_cache, args.kv_lora_rank)
    query = query.cuda()
    kernel_kv_cache = kernel_kv_cache.cuda()
    block_tables = block_tables.cuda()
    seq_lens = seq_lens.cuda()

    attn_scale = 1.0 / math.sqrt(args.kv_lora_rank + args.qk_rope_head_dim)
    output = torch.zeros(
        (args.num_seqs, args.num_q_heads, args.kv_lora_rank),
        dtype=dtype,
        device=query.device,
    )

    print("Launching kernel...")
    flydsl_mla_decode(
        output,
        query,
        kernel_kv_cache,
        block_tables,
        seq_lens,
        attn_scale,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        num_segs=args.num_segs,
        num_warps=args.num_warps,
        kv_compute_block_size=args.kv_compute_block_size,
    )
    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    main()
