# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
import torch
import triton
import pandas as pd
from aiter.ops.triton.gluon.mla_decode_gluon import mla_decode_gluon
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)

torch.set_default_device("cuda")


def bench_mla_decode_gluon_fn(
    batch_size,
    ctx_lens,
    nhead,
    kv_lora_rank,
    qk_rope_head_dim,
):
    decode_qlen = 1
    nhead_kv = 1
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    kv_max_sz = 65536 * 32
    num_page = kv_max_sz
    total_kv = batch_size * ctx_lens
    total_q = batch_size * decode_qlen

    kv_buffer = torch.randn(
        (num_page, 1, qk_head_dim), dtype=torch.bfloat16
    )
    q = torch.randn(
        (total_q, nhead, qk_head_dim), dtype=torch.bfloat16
    )

    q_nope = q[:, :, :v_head_dim].view(batch_size, nhead, v_head_dim)
    q_pe = q[:, :, v_head_dim:].view(batch_size, nhead, qk_head_dim - v_head_dim)
    kv_c = kv_buffer.view(-1, qk_head_dim)
    out_gluon = torch.empty(
        (batch_size, nhead, v_head_dim), dtype=torch.bfloat16
    )

    kv_indices = torch.randint(0, num_page, (total_kv,), dtype=torch.int)
    page_table = kv_indices[:total_kv].view(batch_size, ctx_lens)
    seq_lens = torch.full((batch_size,), ctx_lens, dtype=torch.int)

    fn = lambda: mla_decode_gluon(  # noqa: E731
        q_nope,
        q_pe,
        kv_c,
        out_gluon,
        page_table,
        seq_lens,
        sm_scale,
        use_2d_view=True,
        min_kv_seq_len=ctx_lens,
    )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)

    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    mem_bytes = (
        total_kv * nhead_kv * qk_head_dim * 2
        + total_q * nhead * qk_head_dim * 2
        + total_q * nhead * v_head_dim * 2
    )

    tflops = flops / ms * 1e-9
    tb_s = mem_bytes / (ms * 1e-3) * 1e-12

    return ms, tflops, tb_s


def run_benchmark(args):
    plot_name = get_caller_name_no_ext()
    rows = []
    for bs in args.batch_size:
        for cl in args.ctx_lens:
            for nh in args.nhead:
                ms, tflops, tb_s = bench_mla_decode_gluon_fn(
                    batch_size=bs,
                    ctx_lens=cl,
                    nhead=nh,
                    kv_lora_rank=args.kv_lora_rank,
                    qk_rope_head_dim=args.qk_rope_head_dim,
                )
                rows.append({
                    "batch_size": bs,
                    "ctx_lens": cl,
                    "nhead": nh,
                    "Time_(ms)": ms,
                    "TFLOPS": tflops,
                    "Bandwidth_(TB/s)": tb_s,
                })

    df = pd.DataFrame(rows)
    print(f"{plot_name}:")
    print(df.to_string(index=True))

    if args.o:
        save_path = "."
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, f"{plot_name}.csv")
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"\nSaved to {csv_path}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA Decode Gluon",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c", "--ctx_lens",
        type=int, nargs="+",
        default=[16384],
        help="Context lengths to sweep.",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, nargs="+",
        default=[64, 128],
        help="Batch sizes to sweep.",
    )
    parser.add_argument(
        "-n", "--nhead",
        type=int, nargs="+",
        default=[64, 128],
        help="Number of query heads.",
    )
    parser.add_argument(
        "-k", "--kv_lora_rank",
        type=int, default=512,
        help="KV LoRA rank (also v_head_dim in absorb mode).",
    )
    parser.add_argument(
        "-qr", "--qk_rope_head_dim",
        type=int, default=64,
        help="QK rope head dim.",
    )
    parser.add_argument(
        "-o", action="store_true", default=False,
        help="Write performance results to CSV file.",
    )
    parser.add_argument(
        "-print_vgpr", action="store_true", default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    return parser.parse_args(args=args)


def main(args=None):
    parsed_args = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
