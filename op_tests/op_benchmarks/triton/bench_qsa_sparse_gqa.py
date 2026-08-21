# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the validated gfx950 QSA sparse paged-GQA geometry."""

import argparse

import torch
import triton

from aiter.ops.triton.attention.qsa import qsa_sparse_paged_gqa


def _inputs(index_order: str):
    torch.manual_seed(17)
    rows, query_heads, head_dim = 16, 10, 128
    page_size, pages, kv_heads = 16, 129, 2
    q = torch.randn(rows, query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        pages, page_size, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    block_table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(
        rows, 1
    )
    token_to_request = torch.arange(rows, device="cuda", dtype=torch.int32)
    valid = torch.arange(2048, device="cuda", dtype=torch.int32)
    if index_order == "ordered":
        indices = valid.repeat(rows, 1)
    else:
        indices = torch.stack(
            [valid[torch.randperm(valid.numel(), device="cuda")] for _ in range(rows)]
        )
    indices = torch.cat(
        (
            indices,
            torch.full((rows, 3), -1, device="cuda", dtype=torch.int32),
        ),
        dim=1,
    )
    return q, k, v, indices, block_table, token_to_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("ordered", "random"), default="ordered")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    args = parser.parse_args()
    inputs = _inputs(args.order)
    outputs = {}
    for backend in ("triton", "gluon"):
        fn = lambda backend=backend: qsa_sparse_paged_gqa(
            *inputs, backend=backend
        )
        outputs[backend] = fn()
        p50, p20, p80 = triton.testing.do_bench(
            fn, warmup=args.warmup, rep=args.rep, quantiles=[0.5, 0.2, 0.8]
        )
        print(
            f"{args.order} {backend}: p20={p20:.6f} ms "
            f"p50={p50:.6f} ms p80={p80:.6f} ms"
        )
    torch.testing.assert_close(
        outputs["gluon"], outputs["triton"], rtol=2e-2, atol=2e-2
    )


if __name__ == "__main__":
    main()
