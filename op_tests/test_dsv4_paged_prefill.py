# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for the DeepSeek-V4 sparse paged-prefill kernel (aiter port of ATOM's
atom/model_ops/v4_kernels/paged_prefill.py). Two KV sources: paged unified_kv
prefix + per-fwd flat kv extend."""

import argparse
import sys

import torch

from aiter import dtypes
from aiter.ops.triton.dsv4.paged_prefill import (
    sparse_attn_v4_paged_prefill,
    sparse_attn_v4_paged_prefill_reference,
)
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")
torch.manual_seed(0)


def _ragged_indptr(spans):
    ip = torch.zeros(len(spans) + 1, dtype=torch.int32)
    ip[1:] = torch.tensor(spans, dtype=torch.int32).cumsum(0)
    return ip


def run(prefix_spans, extend_spans, heads, head_dim, pages, tokens, atol, rtol):
    T = len(prefix_spans)
    q = (torch.randn(T, heads, head_dim) * 0.2).to(dtypes.bf16)
    unified_kv = (torch.randn(pages, head_dim) * 0.2).to(dtypes.bf16)
    kv = (torch.randn(tokens, head_dim) * 0.2).to(dtypes.bf16)
    attn_sink = torch.randn(heads, dtype=torch.float32)
    scale = head_dim**-0.5

    kvip_p = _ragged_indptr(prefix_spans)
    kvip_e = _ragged_indptr(extend_spans)
    idx_p = torch.randint(0, pages, (int(kvip_p[-1].item()),), dtype=torch.int32)
    idx_e = torch.randint(0, tokens, (int(kvip_e[-1].item()),), dtype=torch.int32)

    ref = sparse_attn_v4_paged_prefill_reference(
        q, unified_kv, idx_p, kvip_p, kv, idx_e, kvip_e, attn_sink, scale
    )
    out = sparse_attn_v4_paged_prefill(
        q, unified_kv, idx_p, kvip_p, kv, idx_e, kvip_e, attn_sink, scale
    )
    return checkAllclose(
        ref,
        out,
        rtol=rtol,
        atol=atol,
        msg=f"paged_prefill T{T} H{heads} D{head_dim}",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)
    args = p.parse_args()
    errs = [
        run(
            [4, 0, 30],
            [1, 1, 1],
            args.heads,
            args.head_dim,
            512,
            64,
            args.atol,
            args.rtol,
        ),
        run(
            [64, 128],
            [8, 16],
            args.heads,
            args.head_dim,
            1024,
            128,
            args.atol,
            args.rtol,
        ),
    ]
    fail = sum(1 for e in errs if not (e == 0 or (isinstance(e, float) and e < 0.02)))
    if fail:
        print(f"{fail} case(s) FAILED")
        sys.exit(1)
    print("All cases passed.")


if __name__ == "__main__":
    main()
