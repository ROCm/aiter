# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark the paged MXFP4 MQA-logits kernel.
Examples:
    python3 bench_pa_mqa_logits_mxfp4.py --batch 32 --q-len 1 --kv-len 32768
    python3 bench_pa_mqa_logits_mxfp4.py --batch 8,32,128 --q-len 1,6 --kv-len 32768
    python3 bench_pa_mqa_logits_mxfp4.py --batch 1 --q-len 512 --kv-len 65536 \
        --no-preshuffle
    python3 bench_pa_mqa_logits_mxfp4.py --batch 32 --q-len 6 --kv-len 32768 \
        --heads 32,64 --json

q_len is the number of query tokens per sequence: 1 + speculative tokens on a
decode step, or the chunk width on a chunked-prefill step.

"""
import argparse
import itertools


import torch
from aiter.benchmark_reporting import print_json_table
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import run_perftest

from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (IDEAL_PAGE_SIZE, cache_format,
                                            pack_cache,
                                            paged_mxfp4_mqa_logits,
                                            plan_block_m, preshuffle_cache,
)

SCALE_GROUP = 32


def int_list(text):
    out = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{part!r} is not an integer")
        if v < 1:
            raise argparse.ArgumentTypeError(f"{part!r} must be positive")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("empty list")
    return out


def build(batch, next_n, num_heads, head_size, ctx, page_size, preshuffle,
          seed, clean=True):
    dev = "cuda"
    torch.manual_seed(seed)
    hb, ns = head_size // 2, head_size // SCALE_GROUP
    per_seq = (ctx + page_size - 1) // page_size
    total_pages = batch * per_seq
    g = torch.Generator().manual_seed(seed)
    block_table = (torch.randperm(total_pages, generator=g)
                   .reshape(batch, per_seq).to(dev).to(torch.int32))
    values = torch.randint(0, 256, (total_pages, page_size, hb),
                           dtype=torch.uint8, device=dev)
    scales = torch.randint(120, 135, (total_pages, page_size, ns),
                           dtype=torch.uint8, device=dev)
    if preshuffle:
        values, scales = preshuffle_cache(values, scales, num_heads, head_size)
    kv_cache = pack_cache(values, scales)
    del values, scales

    q = torch.randint(0, 256, (batch, next_n, num_heads, hb), dtype=torch.uint8,
                      device=dev)
    qs = torch.randint(120, 135, (batch, next_n, num_heads, ns),
                       dtype=torch.uint8, device=dev)
    weights = torch.randn(batch * next_n, num_heads, dtype=torch.float32,
                          device=dev)
    ctx_lens = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
    max_model_len = per_seq * page_size
    shape = (batch * next_n, max_model_len)
    out = (torch.full(shape, float("-inf"), dtype=torch.float32, device=dev)
           if clean else torch.empty(shape, dtype=torch.float32, device=dev))
    return dict(q=q, qs=qs, kv_cache=kv_cache, weights=weights,
                ctx_lens=ctx_lens, block_table=block_table, out=out,
                max_model_len=max_model_len)


def run_benchmark(args):
    assert arch_info.get_arch() == "gfx950", "gfx950 only"
    rows = []
    for heads in args.heads:
        for batch, next_n, ctx in itertools.product(args.batch, args.q_len,
                                                    args.kv_len):
            name = f"b{batch} q{next_n} kv{ctx}"
            try:
                d = build(batch, next_n, heads, args.head_size, ctx, args.page,
                          args.preshuffle, args.seed,
                          args.clean_logits)
            except torch.OutOfMemoryError:
                rows.append(dict(shape=name, heads=heads, err_msg="OOM"))
                torch.cuda.empty_cache()
                continue

            _, us = run_perftest(
                paged_mxfp4_mqa_logits,
                d["q"], d["qs"], d["kv_cache"], d["weights"], d["ctx_lens"],
                d["block_table"], d["max_model_len"], out_logits=d["out"],
                preshuffle=args.preshuffle, clean_logits=args.clean_logits,
                dynamic=int(args.dynamic),
                num_iters=args.iters, num_warmup=args.warmup)

            # Causal: row n of a sequence walks ctx - next_n + n + 1 keys, which
            # on a wide chunk is well short of next_n * ctx.
            walked = batch * sum(max(0, ctx - next_n + 1 + i)
                                 for i in range(next_n))
            tokens = batch * ctx
            per_token = args.head_size // 2 + args.head_size // SCALE_GROUP
            flop = 2.0 * walked * heads * args.head_size
            store = walked * 4.0
            block_m = min(_block_m(heads, next_n, args.preshuffle), next_n)
            passes = (next_n + block_m - 1) // block_m

            rows.append(dict(
                heads=heads, shape=name, preshuffle=int(args.preshuffle),
                clean=int(args.clean_logits),
                us=round(us, 1), tflops=round(flop / us / 1e6, 1),
                uniq_tbs=round((tokens * per_token + store) / us * 1e-6, 2),
                issued_tbs=round((passes * tokens * per_token + store)
                                 / us * 1e-6, 2),
                kv_passes=passes))
            del d
            torch.cuda.empty_cache()

    if args.json:
        print_json_table("pa_mqa_logits_mxfp4", rows)
    else:
        print(f"{'heads':>6}{'shape':>22}{'shuf':>6}{'us':>11}{'TF/s':>9}"
              f"{'uniq TB/s':>11}{'iss TB/s':>10}{'rd':>4}")
        for r in rows:
            if r.get("err_msg"):
                print(f"{r['heads']:>6}{r['shape']:>22}{r['err_msg']:>22}")
                continue
            print(f"{r['heads']:>6}{r['shape']:>22}{r['preshuffle']:>6}"
                  f"{r['us']:>11.1f}{r['tflops']:>9.0f}{r['uniq_tbs']:>11.2f}"
                  f"{r['issued_tbs']:>10.2f}{r['kv_passes']:>4}")
    return rows


def _block_m(heads, next_n, preshuffle):
    return plan_block_m(heads, next_n) if preshuffle else (
        2 if (heads <= 32 and next_n >= 2) else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int_list, required=True,
                   help="sequences per launch; comma separated to sweep")
    p.add_argument("--q-len", "--next-n", dest="q_len", type=int_list,
                   required=True,
                   help="query tokens per sequence (the ABI's next_n): 1 + "
                        "speculative tokens on decode, the chunk width on "
                        "prefill; comma separated to sweep")
    p.add_argument("--kv-len", dest="kv_len", type=int_list, required=True,
                   help="context length per sequence; comma separated to sweep")
    p.add_argument("--heads", type=int_list, default=[32, 64])
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--page", type=int, default=IDEAL_PAGE_SIZE,
                   help="KV page size")
    p.add_argument("--no-preshuffle", dest="preshuffle", action="store_false",
                   help="token-major cache; stages the KV tile through LDS")
    p.add_argument("--no-clean-logits", dest="clean_logits",
                   action="store_false",
                   help="leave the positions a row does not attend to "
                        "unspecified, which drops the causal predicate from the "
                        "store")
    p.add_argument("--dynamic", action="store_true",
                   help="build a work schedule on the device, for batches whose "
                        "sequences differ in length")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(preshuffle=True, clean_logits=True)
    a = p.parse_args()
    run_benchmark(a)
