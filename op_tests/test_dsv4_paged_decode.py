# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for the DeepSeek-V4 sparse paged-decode kernel (aiter port of ATOM's
atom/model_ops/v4_kernels/paged_decode.py)."""

import argparse
import sys

import torch

from aiter import dtypes
from aiter.ops.triton.dsv4.paged_decode import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_decode_reference,
)
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")
torch.manual_seed(0)


def run(num_tokens, heads, head_dim, total_pages, max_span, atol, rtol):
    q = (torch.randn(num_tokens, heads, head_dim) * 0.2).to(dtypes.bf16)
    unified_kv = (torch.randn(total_pages, head_dim) * 0.2).to(dtypes.bf16)
    attn_sink = torch.randn(heads, dtype=torch.float32)
    scale = head_dim**-0.5

    spans = torch.randint(1, max_span + 1, (num_tokens,))
    kv_indptr = torch.zeros(num_tokens + 1, dtype=torch.int32)
    kv_indptr[1:] = spans.cumsum(0).to(torch.int32)
    total = int(kv_indptr[-1].item())
    kv_indices = torch.randint(0, total_pages, (total,), dtype=torch.int32)

    ref = sparse_attn_v4_paged_decode_reference(
        q, unified_kv, kv_indices, kv_indptr, attn_sink, scale
    )
    out = sparse_attn_v4_paged_decode(
        q, unified_kv, kv_indices, kv_indptr, attn_sink, scale
    )
    return checkAllclose(
        ref,
        out,
        rtol=rtol,
        atol=atol,
        msg=f"paged_decode T{num_tokens} H{heads} D{head_dim} span<={max_span}",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--total-pages", type=int, default=4096)
    p.add_argument("--max-span", type=int, default=256)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)
    args = p.parse_args()
    fail = 0
    for t in args.tokens:
        err = run(
            t,
            args.heads,
            args.head_dim,
            args.total_pages,
            args.max_span,
            args.atol,
            args.rtol,
        )
        fail += 0 if (err == 0 or (isinstance(err, float) and err < 0.02)) else 1
    if fail:
        print(f"{fail} case(s) FAILED")
        sys.exit(1)
    print("All cases passed.")


if __name__ == "__main__":
    main()
