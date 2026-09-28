# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL BF16 paged decode vs the CK varlen kernel it replaces, at QSA shapes.

Reproduces Qwen3.8's post-compaction sparse attention as SGLang invokes it
(``qwen_sparse_attn_backend._forward_paged_attention``): the indexer has already
selected up to ``budget`` KV tokens per query, they have been gathered into
scratch, and one attention call follows. Every query row is its own length-one
sequence with its own selected KV -- speculative verify rows included, which
arrive as extra rows, not as a longer query.

The two candidates therefore read the same logical KV through different ABIs:

  ck              ``flash_attn_varlen_func`` over back-to-back packed rows.
                  On ROCm this is CK's group-mode ``FmhaFwdKernel``, whose tile
                  shape puts only ``seqlen_q`` on the MFMA M axis -- a decode row
                  fills 1 of 128.
  flydsl          ``flydsl_pa_decode`` over the page-aligned vectorized-5D cache.
                  Flattens ``query_length * query_group_size`` onto M instead:
                  12 of 16 rows at Hq=24 / Hkv=2.

``relayout`` times the permute+copy that turns the gathered scratch into that
cache. SGLang's Stage-1 wiring pays it per call; folding the layout into the
gather's destination index removes it, so it is reported separately rather than
hidden inside either candidate.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.pa_decode import (
    flydsl_pa_decode,
    flydsl_pa_decode_partials,
    flydsl_pa_decode_recommended_splits,
    flydsl_pa_decode_supported,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
# 16 bytes per vector lane; the paged cache layout is defined in these units.
KV_VECTOR_BYTES = 16


def run_torch(q, k, v, lengths, num_kv_heads, dtype=dtypes.bf16):
    """Reference: fp32 softmax attention, one compact sequence at a time."""
    group = q.size(1) // num_kv_heads
    out = torch.empty_like(q, dtype=dtypes.fp32)
    scale = q.size(-1) ** -0.5
    offset = 0
    for row, length in enumerate(lengths):
        kr = k[offset : offset + length].to(dtypes.fp32).repeat_interleave(group, dim=1)
        vr = v[offset : offset + length].to(dtypes.fp32).repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].to(dtypes.fp32), kr) * scale
        out[row] = torch.einsum("hk,khd->hd", scores.softmax(-1), vr)
        offset += length
    return out.to(dtype)


def relayout_paged_kv(packed_k, packed_v, num_blocks, page, num_kv_heads, head_dim):
    """Page-aligned ``[token, head, dim]`` scratch -> the kernel's 5D paged cache.

    Mirrors ``sglang.srt.layers.attention.qsa.pa_decode_flydsl.relayout_paged_kv``.
    Source is (block, slot, head, dim); split the axis each target vectorizes over
    -- dim for K, slot for V -- and permute straight to the destination, so each
    cache costs exactly one copy.
    """
    vec = KV_VECTOR_BYTES // packed_k.element_size()
    key_cache = (
        packed_k.view(num_blocks, page, num_kv_heads, head_dim // vec, vec)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    value_cache = (
        packed_v.view(num_blocks, page // vec, vec, num_kv_heads, head_dim)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return key_cache, value_cache


@benchmark()
def test_pa_decode(rows, budget, num_q_heads, num_kv_heads, head_dim, page, dtype):
    group = num_q_heads // num_kv_heads
    scale = head_dim**-0.5
    # Selected length per row: the budget saturates once the context is long
    # enough, leaving only the incomplete compression group as a tail residue.
    lengths = [min(budget, budget - 3 + (i % 4)) for i in range(rows)]
    pages_per_row = (budget + page - 1) // page
    stride = pages_per_row * page
    num_blocks = rows * pages_per_row

    q = torch.randn(rows, num_q_heads, head_dim, dtype=dtype)
    # Packed (CK) scratch: rows back to back, capacity-sized as in production.
    packed_k = torch.randn(rows * budget, num_kv_heads, head_dim, dtype=dtype)
    packed_v = torch.randn_like(packed_k)
    cu_q = torch.arange(rows + 1, dtype=dtypes.i32)
    cu_k = torch.tensor([0] + lengths, dtype=dtypes.i32).cumsum(0, dtype=dtypes.i32)
    ref = run_torch(q, packed_k, packed_v, lengths, num_kv_heads, dtype)

    # Page-aligned (FlyDSL) scratch: the same rows at page-aligned starts, with
    # every slot past the valid count zeroed -- a paged kernel reads whole pages.
    strided_k = torch.zeros(rows * stride, num_kv_heads, head_dim, dtype=dtype)
    strided_v = torch.zeros_like(strided_k)
    offset = 0
    for row, length in enumerate(lengths):
        strided_k[row * stride : row * stride + length] = packed_k[
            offset : offset + length
        ]
        strided_v[row * stride : row * stride + length] = packed_v[
            offset : offset + length
        ]
        offset += length
    block_tables = torch.arange(num_blocks, dtype=dtypes.i32).view(rows, pages_per_row)
    seq_lens = torch.tensor(lengths, dtype=dtypes.i32)

    candidates = {
        "ck": lambda: aiter.flash_attn_varlen_func(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=budget,
            softmax_scale=scale,
            causal=True,
        )
    }

    key_cache, value_cache = relayout_paged_kv(
        strided_k, strided_v, num_blocks, page, num_kv_heads, head_dim
    )
    # Skip the FlyDSL candidate where the kernel's own asserts would reject the
    # shape (arch, dtype, head_dim % 64, block_size) rather than run it anyway.
    if flydsl_pa_decode_supported(q, key_cache, block_size=page):
        splits = flydsl_pa_decode_recommended_splits(rows, num_kv_heads, page)
        pmax, psum, pout = flydsl_pa_decode_partials(
            rows, num_kv_heads, splits, 1, group, head_dim, dtype=dtype, device=q.device
        )
        # Preallocated, as the backend passes it: the ABI has no allocating form.
        out_fly = torch.empty(rows, num_q_heads, head_dim, dtype=dtype)
        candidates["flydsl"] = lambda: flydsl_pa_decode(
            out_fly,
            q,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            num_partitions=splits,
            pmax=pmax,
            psum=psum,
            pout=pout,
        )
    else:
        aiter.logger.warning(
            "%s: flydsl pa_decode does not serve rows=%d D=%d page=%d; skipping",
            get_gfx(),
            rows,
            head_dim,
            page,
        )

    total_kv = sum(lengths)
    # QK^T over the selected KV plus PV, both 2*M*N*K with M = num_q_heads.
    flops = 2 * 2 * num_q_heads * total_kv * head_dim
    # KV dominates: Q and O are rows*heads*dim, K and V are total_kv*kv_heads*dim.
    nbytes = (
        2 * total_kv * num_kv_heads * head_dim + 2 * rows * num_q_heads * head_dim
    ) * q.element_size()

    ret = {"gfx": get_gfx(), "kv_tokens": total_kv}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        out = out[0] if isinstance(out, tuple) else out
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: qsa paged decode",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6

    if "flydsl" in candidates:
        # Stage-1 wiring cost, reported beside the kernels rather than inside one:
        # SGLang pays this per call today, and folding the layout into the gather's
        # destination index removes it without changing either kernel.
        _, relayout_us = run_perftest(
            lambda: relayout_paged_kv(
                strided_k, strided_v, num_blocks, page, num_kv_heads, head_dim
            )
        )
        ret["relayout us"] = relayout_us
        ret["flydsl+relayout us"] = ret["flydsl us"] + relayout_us
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("qsa paged decode unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        # A string default is re-parsed through `type`; the trailing comma makes
        # str2Dtype return a tuple rather than a bare dtype.
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        nargs="*",
        default=[1, 4, 8, 16, 32, 64],
        help="query rows: CONC for decode, 4*CONC for four-token verify",
    )
    parser.add_argument(
        "-k",
        "--budget",
        type=int,
        nargs="*",
        default=[2051],
        help="selected KV capacity: indexer budget 2048 plus the tail group",
    )
    parser.add_argument(
        "-p",
        "--page",
        type=int,
        nargs="*",
        default=[64],
        help="paged block size (16 or 64)",
    )
    parser.add_argument(
        "-n",
        "--heads",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(24, 2, 256)],
        help="(num_q_heads, num_kv_heads, head_dim); Qwen3.8 TP=1 is 24,2,256",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for (hq, hkv, d), budget, page, rows in itertools.product(
            args.heads, args.budget, args.page, args.rows
        ):
            df.append(test_pa_decode(rows, budget, hq, hkv, d, page, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "qsa paged decode summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
