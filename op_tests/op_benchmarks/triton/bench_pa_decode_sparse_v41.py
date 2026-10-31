# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Time gfx950 sparse MLA on V4 and V4.1 records with CUDA graph replay.

Run with the repository root in PYTHONPATH. All formats use the same index
streams. Timing uses repeated graph replay without a cache flush or weights.
Run each head count in a separate process.
"""

import argparse
import json
import statistics
from functools import partial

import torch
from triton.testing import do_bench_cudagraph

from aiter import logger
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.bench_pa_decode_sparse import (
    build_slot_indices,
    build_topk_indices,
    v4_pack_2buff,
)


def packed_cache(tokens, page, record):
    """Make valid records for timing; packing is outside the measured region."""
    blocks = tokens // page
    cache = torch.empty(blocks, page, record, dtype=torch.uint8, device="cuda")
    if record == 584:
        values = torch.randn(tokens, 512, dtype=torch.bfloat16, device="cuda")
        packed, rope = v4_pack_2buff(values)
        flat = cache.view(blocks, -1)
        data = flat[:, : page * 576].view(blocks, page, 576)
        packed_bytes = packed.view(torch.uint8).view(blocks, page, 512)
        data[:, :, :448] = packed_bytes[:, :, :448]
        data[:, :, 448:] = rope.view(torch.uint8).view(blocks, page, 128)
        scales = flat[:, page * 576 :].view(blocks, page, 8)
        scales[:, :, :7] = packed_bytes[:, :, 448:462:2]
        scales[:, :, 7] = 0
    elif record == 528:
        values = torch.randn(blocks, page, 512, dtype=torch.float32, device="cuda")
        cache[:, :, :512] = values.to(torch.float8_e4m3fn).view(torch.uint8)
        cache[:, :, 512:] = 127
    else:
        cache[:, :, :256] = torch.randint(
            0, 256, (blocks, page, 256), dtype=torch.uint8, device="cuda"
        )
        scales = torch.full(
            (blocks, page, 32), 0.125, dtype=torch.float32, device="cuda"
        )
        cache[:, :, 256:] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", type=int, default=[1, 8, 32, 128, 512])
    parser.add_argument("--heads", nargs="+", type=int, default=[32])
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["584/584", "528/528", "528/288"],
        default=["584/584", "528/528", "528/288"],
    )
    parser.add_argument("--extra-len", type=int, default=512)
    parser.add_argument("--page", type=int, default=128)
    parser.add_argument("--extra-page", type=int, default=64)
    parser.add_argument("--pool-tokens", type=int, default=131072)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if arch_info.get_arch() != "gfx950":
        parser.error("this benchmark requires gfx950")
    if len(args.heads) != 1:
        parser.error("run each head count in a separate process")
    if (
        min(
            args.heads
            + args.queries
            + [args.page, args.extra_page, args.pool_tokens, args.extra_len]
        )
        < 1
    ):
        parser.error(
            "heads, queries, page sizes, pool-tokens and extra-len must be positive"
        )
    if args.pool_tokens % args.page or args.pool_tokens % args.extra_page:
        parser.error("pool-tokens must contain complete pages")
    if args.repeats < 1 or args.rep_ms < 1:
        parser.error("repeats and rep-ms must be positive")
    torch.manual_seed(0)
    caches = {
        fmt: (
            packed_cache(args.pool_tokens, args.page, int(fmt.split("/")[0])),
            packed_cache(args.pool_tokens, args.extra_page, int(fmt.split("/")[1])),
        )
        for fmt in args.formats
    }
    results = []
    for heads in args.heads:
        for queries in args.queries:
            torch.manual_seed(1234 + heads * 1024 + queries)
            gen = torch.Generator(device="cuda").manual_seed(1234)
            q = torch.randn(queries, heads, 512, dtype=torch.bfloat16, device="cuda")
            sink = torch.zeros(heads, dtype=torch.float32, device="cuda")
            indices = build_slot_indices(
                queries,
                [128] * queries,
                args.pool_tokens // args.page,
                args.page,
                "cuda",
                gen,
            )
            extra_indices = build_topk_indices(
                queries,
                args.extra_len,
                args.pool_tokens // args.extra_page,
                args.extra_page,
                "cuda",
                gen,
            )
            indptr = torch.arange(queries + 1, dtype=torch.int32, device="cuda") * 128
            extra_indptr = (
                torch.arange(queries + 1, dtype=torch.int32, device="cuda")
                * args.extra_len
            )
            timings = {fmt: [] for fmt in args.formats}
            for repeat in range(args.repeats):
                order = (
                    args.formats[repeat % len(args.formats) :]
                    + args.formats[: repeat % len(args.formats)]
                )
                for fmt in order:
                    main_cache, extra_cache = caches[fmt]

                    run = partial(
                        pa_decode_sparse,
                        q,
                        main_cache,
                        indices,
                        indptr,
                        sink,
                        512**-0.5,
                        extra_cache=extra_cache,
                        extra_indices=extra_indices,
                        extra_indptr=extra_indptr,
                        has_invalid=True,
                    )

                    timings[fmt].append(do_bench_cudagraph(run, rep=args.rep_ms) * 1000)
            for fmt, samples in timings.items():
                row = {
                    "heads": heads,
                    "queries": queries,
                    "format": fmt,
                    "us": samples,
                    "median_us": statistics.median(samples),
                    "min_us": min(samples),
                    "max_us": max(samples),
                }
                results.append(row)
                logger.info(
                    "H=%s T=%s format=%s median_us=%s",
                    heads,
                    queries,
                    fmt,
                    row["median_us"],
                )
                with open(args.output, "w") as output:
                    json.dump(
                        {
                            "device": torch.cuda.get_device_name(),
                            "args": vars(args),
                            "results": results,
                        },
                        output,
                        indent=2,
                    )


if __name__ == "__main__":
    main()
