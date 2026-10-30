# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-KV FMHA prefill on gfx1250: the FlyDSL m32x8 kernel against CK.

The page table is a random permutation, so a kernel that assumes pages are laid
out sequentially fails loudly rather than quietly.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_paged_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]


def build_paged_inputs(batch, sq, sk, nhead_q, nhead_kv, hdim, page_size, dtype):
    """Ragged q/kv of a fixed length each, with a PERMUTED page table."""
    pages_per_seq = (sk + page_size - 1) // page_size
    num_blocks = pages_per_seq * batch + 8  # slack so some pages stay unreferenced
    pool_shape = (num_blocks, page_size, nhead_kv, hdim)
    k_pool = torch.randn(pool_shape, dtype=dtype)
    v_pool = torch.randn(pool_shape, dtype=dtype)
    page_ids = torch.randperm(num_blocks).int()[: pages_per_seq * batch].contiguous()
    q = torch.randn((batch * sq, nhead_q, hdim), dtype=dtype)
    qo_indptr = torch.arange(0, batch * sq + 1, sq, dtype=dtypes.i32)
    kv_indptr = torch.arange(0, pages_per_seq * batch + 1, pages_per_seq, dtype=dtypes.i32)
    last_page_lens = torch.full(
        (batch,), sk - (pages_per_seq - 1) * page_size, dtype=dtypes.i32
    )
    return q, k_pool, v_pool, qo_indptr, kv_indptr, page_ids, last_page_lens


def gather_dense(pool, page_ids, batch, sk, page_size, pages_per_seq):
    """The dense KV the page table spells out, per request."""
    flat = pool.view(-1, pool.shape[-2], pool.shape[-1])
    out = []
    for b in range(batch):
        ids = page_ids[b * pages_per_seq : (b + 1) * pages_per_seq].tolist()
        rows = torch.cat([flat[i * page_size : (i + 1) * page_size] for i in ids], 0)
        out.append(rows[:sk])
    return out


def run_torch(q, k_dense, v_dense, batch, sq, causal, nhead_q, nhead_kv):
    """Reference only: fp32 math per request. Not timed, not in the table."""
    gqa = nhead_q // nhead_kv
    outs = []
    for b in range(batch):
        qb = q[b * sq : (b + 1) * sq].to(dtypes.fp32)            # [sq, hq, d]
        kb = k_dense[b].to(dtypes.fp32).repeat_interleave(gqa, dim=1)
        vb = v_dense[b].to(dtypes.fp32).repeat_interleave(gqa, dim=1)
        s = torch.einsum("qhd,khd->hqk", qb, kb) / (qb.shape[-1] ** 0.5)
        if causal:
            sk_len = kb.shape[0]
            qpos = torch.arange(sq, device=s.device) + (sk_len - sq)
            kpos = torch.arange(sk_len, device=s.device)
            s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vb))
    return torch.cat(outs, 0).to(q.dtype)


@benchmark()
def test_paged_prefill(batch, sq, sk, nhead_q, nhead_kv, hdim, page_size, causal, dtype):
    q, k_pool, v_pool, qo_indptr, kv_indptr, page_ids, last_lens = build_paged_inputs(
        batch, sq, sk, nhead_q, nhead_kv, hdim, page_size, dtype
    )
    pages_per_seq = (sk + page_size - 1) // page_size
    k_dense = gather_dense(k_pool, page_ids, batch, sk, page_size, pages_per_seq)
    v_dense = gather_dense(v_pool, page_ids, batch, sk, page_size, pages_per_seq)
    ref = run_torch(q, k_dense, v_dense, batch, sq, causal, nhead_q, nhead_kv)

    # No CK candidate: mha_batch_prefill does not compile on gfx1250. Its tile
    # distributions assert on wave64 ("slice size can't evenly divide input sizes",
    # sequence<32,32,32>), so there is no in-tree paged-prefill baseline on this arch.
    candidates = {
        "flydsl": lambda: flydsl_flash_attn_paged_fwd(
            q, k_pool, v_pool, qo_indptr, kv_indptr, page_ids,
            kv_last_page_lens=last_lens, max_seqlen_q=sq, max_seqlen_k=sk, causal=causal,
        ),
    }

    # causal keeps rows above the diagonal, so roughly half the score matrix
    frac = 0.5 if causal else 1.0
    flops = 2 * 2 * batch * nhead_q * sq * sk * hdim * frac
    nbytes = (
        q.numel()
        + 2 * batch * sk * nhead_kv * hdim      # the K/V pages actually touched
        + batch * sq * nhead_q * hdim           # O
    ) * q.element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        out = out[0] if isinstance(out, (tuple, list)) else out
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32), out.to(dtypes.fp32),
            rtol=1e-2, atol=1e-2, msg=f"{name}: paged prefill",
        )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("paged FMHA unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="config input of test"
    )
    parser.add_argument("-d", "--dtype", type=dtypes.str2Dtype, nargs="*",
                        default=[dtypes.bf16], help="data types, e.g.: -d bf16")
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4],
                        help="batch sizes, e.g.: -b 1 4")
    parser.add_argument("-s", "--seqlens", type=dtypes.str2tuple, nargs="*",
                        default=[(512, 512), (1024, 1024), (4096, 4096)],
                        help="(seqlen_q, seqlen_k) pairs, e.g.: -s 512,512")
    parser.add_argument("-p", "--page-size", type=int, nargs="*", default=[1, 16, 1024],
                        help="KV cache page sizes, e.g.: -p 1 16")
    parser.add_argument("-nh", "--nheads", type=dtypes.str2tuple, nargs="*",
                        default=[(8, 8), (8, 2)],
                        help="(nhead_q, nhead_kv) pairs, e.g.: -nh 8,2")
    parser.add_argument("-hd", "--hdim", type=int, nargs="*", default=[128],
                        help="head dims, e.g.: -hd 128")
    parser.add_argument("-c", "--causal", type=int, nargs="*", default=[0, 1],
                        help="causal off/on, e.g.: -c 1")
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for batch, (sq, sk), page_size, (nhq, nhkv), hdim, causal in itertools.product(
            args.batch, args.seqlens, args.page_size, args.nheads, args.hdim, args.causal
        ):
            df.append(
                test_paged_prefill(batch, sq, sk, nhq, nhkv, hdim, page_size,
                                   bool(causal), dtype)
            )
        df = pd.DataFrame(df)
        aiter.logger.info("paged FMHA prefill summary (markdown):\n%s",
                          df.to_markdown(index=False))


if __name__ == "__main__":
    main()
