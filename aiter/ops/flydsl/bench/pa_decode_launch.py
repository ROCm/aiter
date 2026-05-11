#!/usr/bin/env python3
"""Launcher for FlyDSL pa_decode_main kernel — used by perf_pa_decode.sh for roccap capture.

Mirrors test_flydsl_pa_decode._generate_inputs. Defaults exercise a GQA decode
shape; all params overridable via CLI flags. Emits machine-parseable METRIC
lines so the perf shell can compute bandwidth/TFLOPS without re-deriving them.

Examples:
    # Defaults (head=64 kvb=16 qg=8 nkv=8 ns=1 sl=8192 kc=128 ps=2048):
    python pa_decode_launch.py

    # Override:
    python pa_decode_launch.py --num-kv-heads 16 --seq-len 4096
"""

import os
import sys

_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

import flydsl  # noqa: E402,F401 — preload comgr before torch/HIP
import math
import random
import torch

from aiter.ops.flydsl.pa_decode import flydsl_paged_attention_decode


def _generate_inputs(
    num_seqs: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
    kv_block_size: int,
    seq_len: int,
    dtype: torch.dtype,
    seed: int = 42,
    random_seq_lens: bool = False,
):
    """Mirror of tests/test_flydsl_pa_decode._generate_inputs."""
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda"

    num_q_heads = num_kv_heads * query_group_size
    num_blocks_per_seq = (seq_len + kv_block_size - 1) // kv_block_size
    total_blocks = num_seqs * num_blocks_per_seq + 2

    query = torch.randn((num_seqs, num_q_heads, head_size), dtype=dtype, device=device) * 0.5
    key_cache = torch.randn(
        (total_blocks, num_kv_heads, kv_block_size, head_size), dtype=dtype, device=device
    ) * 0.5
    value_cache = torch.randn(
        (total_blocks, num_kv_heads, kv_block_size, head_size), dtype=dtype, device=device
    ) * 0.5

    all_blocks = torch.randperm(total_blocks, dtype=torch.int32, device=device)
    block_tables = torch.zeros(
        (num_seqs, num_blocks_per_seq), dtype=torch.int32, device=device
    )
    idx = 0
    for s in range(num_seqs):
        block_tables[s] = all_blocks[idx : idx + num_blocks_per_seq]
        idx += num_blocks_per_seq

    if random_seq_lens and num_seqs > 1:
        seq_lens_list = [random.randint(1, seq_len) for _ in range(num_seqs)]
        seq_lens_list[0] = seq_len
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    else:
        seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    return query, key_cache, value_cache, block_tables, seq_lens


def _compute_metrics(
    seq_lens_list,
    head_size: int,
    kv_block_size: int,
    query_group_size: int,
    num_kv_heads: int,
    partition_size: int,
    elem_bytes: int,
):
    """Compute byte/FLOP metrics for a single pa_decode_main launch."""
    num_seqs = len(seq_lens_list)
    num_q_heads = num_kv_heads * query_group_size

    sum_seq_lens = sum(seq_lens_list)
    aligned_tokens = sum(
        ((s + kv_block_size - 1) // kv_block_size) * kv_block_size
        for s in seq_lens_list
    )
    total_live_parts = sum(
        (s + partition_size - 1) // partition_size
        for s in seq_lens_list
    )
    max_seq_len = max(seq_lens_list) if seq_lens_list else 0
    num_partitions = (max_seq_len + partition_size - 1) // partition_size
    blocks_per_partition = partition_size // kv_block_size

    # Per-tensor byte breakdown (unique DRAM bytes; L2 absorbs cross-WG repeats).
    bytes_q = num_seqs * num_q_heads * head_size * elem_bytes
    bytes_kv_useful = 2 * num_kv_heads * head_size * elem_bytes * sum_seq_lens
    bytes_kv_executed = 2 * num_kv_heads * head_size * elem_bytes * aligned_tokens
    # Per-partition output is f32 (kernel writes pv_accs directly via buffer_store).
    bytes_tmp_out = total_live_parts * num_kv_heads * query_group_size * head_size * 4
    # max_logits + exp_sums, both f32.
    bytes_lse = 2 * total_live_parts * num_kv_heads * query_group_size * 4
    bytes_block_tables = num_seqs * num_partitions * blocks_per_partition * 4
    bytes_seq_lens = num_seqs * 4

    # ---- Main-kernel traffic ----
    # Main reads Q + K + V (+ block_tables + seq_lens) and writes tmp_out + LSE.
    bytes_main_useful = (
        bytes_q + bytes_kv_useful + bytes_tmp_out + bytes_lse
        + bytes_block_tables + bytes_seq_lens
    )
    bytes_main_executed = (
        bytes_q + bytes_kv_executed + bytes_tmp_out + bytes_lse
        + bytes_block_tables + bytes_seq_lens
    )

    # ---- Reduce-kernel traffic ----
    # Reduce reads tmp_out + max_logits + exp_sums (+ seq_lens) and writes the
    # final bf16/f16 output of shape [num_seqs, num_q_heads, head_size].
    bytes_out_final = num_seqs * num_q_heads * head_size * elem_bytes
    bytes_reduce_in = bytes_tmp_out + bytes_lse + bytes_seq_lens
    bytes_reduce_total = bytes_reduce_in + bytes_out_final

    # ---- Combined useful traffic (canonical "effective KV bandwidth" view) ----
    # Inputs + final output only — the tmp_out/LSE intermediates are produced
    # by main and immediately consumed by reduce, so a perfectly L2-cached
    # pipeline wouldn't hit DRAM for them. For the conventional pa_decode BW
    # number we count them once (so this matches bytes_main_useful + final out).
    bytes_combined_useful = (
        bytes_q + bytes_kv_useful + bytes_tmp_out + bytes_lse
        + bytes_block_tables + bytes_seq_lens + bytes_out_final
    )
    bytes_combined_executed = (
        bytes_q + bytes_kv_executed + bytes_tmp_out + bytes_lse
        + bytes_block_tables + bytes_seq_lens + bytes_out_final
    )

    # Back-compat aliases (the old perf shell read these names).
    bytes_total_useful = bytes_main_useful
    bytes_total_executed = bytes_main_executed

    # FLOPs: QK (2*QGS*D*S) + PV (2*QGS*D*S) per (seq, kv_head).
    total_flops = 4 * num_q_heads * head_size * sum_seq_lens

    return {
        "num_q_heads": num_q_heads,
        "num_partitions": num_partitions,
        "sum_seq_lens": sum_seq_lens,
        "total_live_parts": total_live_parts,
        "aligned_tokens": aligned_tokens,
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
        "bytes_total_useful": bytes_total_useful,
        "bytes_total_executed": bytes_total_executed,
        "total_flops": total_flops,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch FlyDSL pa_decode_main kernel")
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--kv-block-size", type=int, default=128)
    parser.add_argument("--query-group-size", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--kv-compute-block-size", type=int, default=128)
    parser.add_argument("--partition-size", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "f16"])
    parser.add_argument("--random-seq-lens", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    elem_bytes = 2

    query, key_cache, value_cache, block_tables, seq_lens = _generate_inputs(
        num_seqs=args.num_seqs,
        num_kv_heads=args.num_kv_heads,
        query_group_size=args.query_group_size,
        head_size=args.head_size,
        kv_block_size=args.kv_block_size,
        seq_len=args.seq_len,
        dtype=dtype,
        seed=args.seed,
        random_seq_lens=args.random_seq_lens,
    )

    seq_lens_list = seq_lens.tolist()
    metrics = _compute_metrics(
        seq_lens_list,
        head_size=args.head_size,
        kv_block_size=args.kv_block_size,
        query_group_size=args.query_group_size,
        num_kv_heads=args.num_kv_heads,
        partition_size=args.partition_size,
        elem_bytes=elem_bytes,
    )

    tag = (
        f"h{args.head_size}_kvb{args.kv_block_size}_qg{args.query_group_size}"
        f"_nkv{args.num_kv_heads}_ns{args.num_seqs}_sl{args.seq_len}"
        f"_kc{args.kv_compute_block_size}_ps{args.partition_size}"
    )

    # Machine-parseable METRIC lines (consumed by perf_pa_decode.sh).
    print(f"METRIC tag={tag}")
    print(f"METRIC head_size={args.head_size}")
    print(f"METRIC kv_block_size={args.kv_block_size}")
    print(f"METRIC query_group_size={args.query_group_size}")
    print(f"METRIC num_kv_heads={args.num_kv_heads}")
    print(f"METRIC num_seqs={args.num_seqs}")
    print(f"METRIC seq_len={args.seq_len}")
    print(f"METRIC kv_compute_block_size={args.kv_compute_block_size}")
    print(f"METRIC partition_size={args.partition_size}")
    print(f"METRIC dtype={args.dtype}")
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")

    print(
        f"Compiling pa_decode: head={args.head_size}, kvb={args.kv_block_size}, "
        f"qg={args.query_group_size}, nkv={args.num_kv_heads}, "
        f"nseqs={args.num_seqs}, seq_len={args.seq_len}, "
        f"kv_compute={args.kv_compute_block_size}, partition={args.partition_size}, "
        f"dtype={args.dtype}, random_seq_lens={args.random_seq_lens}"
    )

    attn_scale = 1.0 / math.sqrt(args.head_size)
    output = torch.zeros_like(query)

    print("Launching kernel...")
    flydsl_paged_attention_decode(
        output, query, key_cache, value_cache, block_tables, seq_lens, attn_scale,
        partition_size=args.partition_size,
        kv_compute_block_size=args.kv_compute_block_size,
    )
    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    main()
