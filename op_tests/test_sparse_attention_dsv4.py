# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.ops.triton.attention.sparse_attention_dsv4 import (
    sparse_attn_prefill,
)

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# DSV4 sparse-MLA layout: head_dim = NoPE(448) + RoPE(64) = 512.
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM


def build_dense_indices(num_q, max_slots, topk, device):
    """Dense `[num_q, topk]` index matrix (`-1`-padded) plus per-row valid
    lengths, each row keeps a random number of unique slots in [topk // 4, topk].
    This is the format the public `sparse_attn_prefill` interface consumes."""
    lens = torch.randint(
        max(1, topk // 4), topk + 1, (num_q,), dtype=torch.int32, device=device
    )
    dense = torch.full((num_q, topk), -1, dtype=torch.int32, device=device)
    for i in range(num_q):
        L = int(lens[i].item())
        dense[i, :L] = torch.randperm(max_slots, device=device, dtype=torch.int32)[:L]
    return dense, lens


def ragged_from_dense(dense, lens):
    """CSR `(flat, indptr)` from a `-1`-padded dense matrix. The torch-side
    mirror of the wrapper's dense -> ragged conversion, used to build the
    reference output."""
    rows = [dense[i, : int(lens[i].item())] for i in range(dense.shape[0])]
    flat = (
        torch.cat(rows)
        if rows
        else torch.empty(0, dtype=torch.int32, device=dense.device)
    )
    indptr = torch.zeros(dense.shape[0] + 1, dtype=torch.int32, device=dense.device)
    torch.cumsum(lens.to(torch.int32), dim=0, out=indptr[1:])
    return flat, indptr


def ref_sparse_prefill(q, kv, indices, indptr, scale, attn_sink):
    """torch reference: per-query masked softmax attention over gathered KV."""
    kv = kv.squeeze(1)
    out = torch.zeros_like(q, dtype=torch.float32)
    num_q = q.shape[0]
    for t in range(num_q):
        s, e = int(indptr[t]), int(indptr[t + 1])
        if e <= s:
            continue
        slots = indices[s:e].long()
        slots = slots[(slots >= 0) & (slots < kv.shape[0])]
        if slots.numel() == 0:
            continue
        K = kv[slots].float()  # [L, D]
        sc = (q[t].float() @ K.t()) * scale  # [H, L]
        if attn_sink is not None:
            m = torch.maximum(sc.max(-1).values, attn_sink.float())
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1) + torch.exp(attn_sink.float() - m)
        else:
            m = sc.max(-1).values
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1)
        out[t] = (p / denom[:, None]) @ K
    return out.to(q.dtype)


def run_prefill(q, kv, dense_indices, lens, attn_sink, scale, has_sink):
    out = torch.empty_like(q)
    sparse_attn_prefill(
        q,
        kv,
        dense_indices,
        lens,
        scale,
        HEAD_DIM,
        NOPE_DIM,
        ROPE_DIM,
        attn_sink if has_sink else None,
        out,
    )
    return out


@benchmark()
def test_sparse_prefill(num_queries, num_heads, num_kv, topk, has_sink):
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(
        num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(num_kv, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dense_indices, lens = build_dense_indices(num_queries, num_kv, topk, device)
    indices, indptr = ragged_from_dense(dense_indices, lens)  # reference CSR
    nnz = int(indptr[-1].item())
    scale = 1.0 / (HEAD_DIM**0.5)
    if has_sink:
        attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    else:
        attn_sink = torch.empty(1, dtype=torch.float32, device=device)

    out_triton, us = run_perftest(
        run_prefill, q, kv, dense_indices, lens, attn_sink, scale, has_sink
    )
    out_ref = ref_sparse_prefill(
        q, kv, indices, indptr, scale, attn_sink if has_sink else None
    )

    # FLOPs: QK + PV = 4 * H * D per (query, kv-pair). Bytes: Q + gathered KV + Out.
    flops = 4.0 * num_heads * HEAD_DIM * nnz
    nbytes = q.numel() * 2 + nnz * HEAD_DIM * 2 + out_triton.numel() * 2

    err = checkAllclose(
        out_ref.float(),
        out_triton.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"prefill [torch vs triton]: {us:>8.2f} us, "
        f"{flops / us / 1e6:>7.1f} TFLOPS, {nbytes / us / 1e3:>7.1f} GB/s",
    )

    return {
        "nnz": nnz,
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "GB/s": nbytes / us / 1e3,
        "err": err,
    }


def _parse_cfg(s):
    parts = tuple(int(x) for x in s.split(","))
    assert (
        len(parts) == 4
    ), f"--cfgs expects 'num_queries,num_heads,num_kv,topk', got '{s}'"
    return parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="DSV4 Gluon sparse-MLA prefill test",
    )
    parser.add_argument(
        "--cfgs",
        type=_parse_cfg,
        nargs="*",
        default=[(4096, 128, 4096, 512)],
        metavar="Q,H,KV,TOPK",
        help="""configs as 'num_queries,num_heads,num_kv,topk'.
    e.g.: --cfgs 4096,128,4096,512""",
    )
    parser.add_argument(
        "--sink",
        action="store_true",
        help="enable the optional per-head attention sink. Default: False.",
    )
    args = parser.parse_args()

    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "Gluon sparse_attention_dsv4 prefill requires gfx950 (CDNA4); "
            f"current arch is {get_gfx()}. Skipping."
        )
        raise SystemExit(0)

    rows = []
    for num_queries, num_heads, num_kv, topk in args.cfgs:
        rows.append(
            test_sparse_prefill(num_queries, num_heads, num_kv, topk, args.sink)
        )
    df = pd.DataFrame(rows)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        # `tabulate` (used by to_markdown) is optional; fall back to plain text.
        summary = df.to_string(index=False)
    aiter.logger.info("sparse_attention_dsv4 prefill summary:\n%s", summary)
