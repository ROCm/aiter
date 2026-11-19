# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for mla_gluon decode (gfx950): today's KV split and the split rebalance.

The KV length of each sequence is drawn uniformly from [kv_min, kv_max]; with
kv_min == kv_max all sequences have the same length. The rebalance gets its
kv_split_info from mla_gluon_split_metadata, built once for each pool entry (a
caller builds it once for each decode step). The time is the mla_gluon call,
from CUDA-graph replay over rotating KV caches. --q-dtype fp8 gives an fp8 query
(fp8 QK and PV; fp8 KV only).
"""

import functools
import random
import sys

import torch
import triton

from aiter.ops.triton.gluon.mla_gluon import mla_gluon, mla_gluon_split_metadata
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
    positive_int,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_mla import (
    _fp8_query,
    _paged_seq_info,
    _split_rebalance_inputs,
)

KV_LORA_RANK = 512
HEAD_DIM = KV_LORA_RANK + 64
# (batch, heads, kv_min, kv_max): 8 and 16 heads per GPU (Kimi-K2.6 and
# DeepSeek-R1 at TP8), equal and mixed KV lengths around 1k and 8k tokens
DEFAULT_SHAPES = [
    (64, heads, kv_min, kv_max)
    for heads in (8, 16)
    for kv_min, kv_max in ((1024, 1024), (204, 1843), (8192, 8192), (1638, 14745))
]
KV_DTYPES = {"fp8": torch.float8_e4m3fn, "bf16": torch.bfloat16}
_POOL_FIELDS = ["q_nope", "q_pe", "kv_c", "o", "page_table", "seq_info", "info"]


def _kv_lens(batch, kv_min, kv_max, seed):
    rng = random.Random(seed)
    return [rng.randint(kv_min, kv_max) for _ in range(batch)]


def _inputs(lens, heads, kv_dtype, fp8_query):
    """One pool entry: q, its own paged KV cache (1-D view), o and kv_split_info."""
    kv_c, block_table, q_nope, q_pe = _split_rebalance_inputs(
        len(lens), 1, heads, kv_dtype, max(lens), KV_LORA_RANK
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    page_table, seq_info = _paged_seq_info(block_table, seq_lens, False)
    info = mla_gluon_split_metadata(seq_info, heads, kv_dtype, use_2d_view=False)
    q_nope, q_pe = q_nope[:, 0].contiguous(), q_pe[:, 0].contiguous()
    if fp8_query:
        (q_nope, _), (q_pe, _) = _fp8_query(q_nope, 1.0), _fp8_query(q_pe, 1.0)
    o = torch.empty(q_nope.shape, dtype=torch.bfloat16, device="cuda")
    return q_nope, q_pe, kv_c, o, page_table, seq_info, info


def benchmark(args):
    kv_dtype = KV_DTYPES[args.kv_dtype]
    unit = "us" if args.metric == "time" else "GB/s"
    config = triton.testing.Benchmark(
        x_names=["batch", "heads", "kv_min", "kv_max"],
        x_vals=[tuple(args.shape)] if args.shape else DEFAULT_SHAPES,
        line_arg="split",
        line_vals=["default", "rebalance"],
        line_names=["default split", "split rebalance"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([config])
    def bench_mla_gluon(batch, heads, kv_min, kv_max, split):
        lens = _kv_lens(batch, kv_min, kv_max, args.seed)
        gen = functools.partial(_inputs, lens, heads, kv_dtype, args.q_dtype == "fp8")
        bencher.build_pool([gen], [_POOL_FIELDS])
        if split == "rebalance" and bencher.rot_bufs[0]["info"] is None:
            return float("nan")

        def decode(q_nope, q_pe, kv_c, o, page_table, seq_info, info):
            mla_gluon(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c=kv_c,
                o=o,
                page_table=page_table,
                seq_info=seq_info,
                sm_scale=HEAD_DIM**-0.5,
                kv_pe_offset=KV_LORA_RANK,
                use_2d_view=False,
                kv_split_info=info if split == "rebalance" else None,
            )

        ms = bencher.bench(decode)
        if args.metric == "time":
            return ms * 1e3
        kv_bytes = sum(lens) * HEAD_DIM * kv_dtype.itemsize
        return kv_bytes * 1e-9 / (ms * 1e-3)

    bench_mla_gluon.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args():
    parser = get_parser(kernel_name="MLA Gluon decode")
    parser.add_argument(
        "--shape",
        type=positive_int,
        nargs=4,
        metavar=("BATCH", "HEADS", "KV_MIN", "KV_MAX"),
        help="One shape to benchmark instead of the default shapes; each KV "
        "length is drawn uniformly from [KV_MIN, KV_MAX]",
    )
    parser.add_argument("--kv-dtype", type=str, default="fp8", choices=list(KV_DTYPES))
    parser.add_argument(
        "--q-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8"],
        help="fp8: fp8 query (fp8 QK and PV), needs --kv-dtype fp8",
    )
    parser.add_argument(
        "--seed", type=positive_int, default=1, help="Seed of the KV lengths"
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to CSV"
    )
    args = parser.parse_args()
    if args.model:
        parser.error("--model is not supported; use --shape")
    if args.shape and args.shape[2] > args.shape[3]:
        parser.error("--shape needs KV_MIN <= KV_MAX")
    if args.q_dtype == "fp8" and args.kv_dtype != "fp8":
        parser.error("--q-dtype fp8 needs --kv-dtype fp8")
    return args


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
