# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL sparse MLA perf on the GLM-5.3-Flash contract.

Torch is the reference and is not timed. T=7 is the BQ pad tail. T=16384 is
the prefill row. shared is exact Q-block reuse; random stays on one-query.
"""

from __future__ import annotations

import argparse
import itertools

import triton  # isort: skip  # noqa: F401  # this HIP image aborts if torch loads triton later
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.sparse_mla_qblock_kernels import (
    sparse_mla_one_query_fwd_flydsl,
    sparse_mla_qblock_fwd_flydsl,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
H_FLASH, D_FLASH, TOPK_FLASH, POOL_FLASH = 16, 512, 2048, 131072
_REF_CHUNK = 128


def make_indices(pattern: str, n_tok: int, pool: int, topk: int):
    if pattern == "shared":
        return torch.arange(topk, dtype=torch.int32).repeat(n_tok)
    seed = torch.randint(0, pool, (n_tok, 1), dtype=torch.int32)
    k = torch.arange(topk, dtype=torch.int32)
    return ((k * 17 + seed) % pool).reshape(-1).contiguous()


def run_torch(q, kv, indices, topk, chunk=_REF_CHUNK):
    n_tok = q.shape[0]
    idx = indices.view(n_tok, topk).long()
    scale = q.shape[-1] ** -0.5
    out = torch.empty(
        n_tok, q.shape[1], q.shape[2], dtype=torch.float32, device=q.device
    )
    for t0 in range(0, n_tok, chunk):
        t1 = min(t0 + chunk, n_tok)
        gathered = kv[idx[t0:t1]].float()
        scores = torch.einsum("thd,tkd->thk", q[t0:t1].float(), gathered) * scale
        probs = torch.softmax(scores, dim=-1)
        out[t0:t1] = torch.einsum("thk,tkd->thd", probs, gathered)
    return out.to(q.dtype)


@benchmark()
def test_sparse_mla_qblock(n_tok, num_heads, head_dim, topk, pool, pattern, dtype):
    q = torch.randn(n_tok, num_heads, head_dim, dtype=dtype)
    kv = torch.randn(pool, head_dim, dtype=dtype)
    indices = make_indices(pattern, n_tok, pool, topk)
    ref = run_torch(q, kv, indices, topk)
    out_one = torch.empty_like(q)
    out_qb = torch.empty_like(q)
    kv_indptr = torch.arange(
        0, (n_tok + 1) * topk, topk, device=q.device, dtype=torch.int32
    )

    def _flydsl_one_q():
        return sparse_mla_one_query_fwd_flydsl(
            q, kv, indices, out_one, kv_indptr=kv_indptr
        )

    def _flydsl_qblock():
        return sparse_mla_qblock_fwd_flydsl(q, kv, indices, out_qb, kv_indptr=kv_indptr)

    candidates = {"flydsl_one_q": _flydsl_one_q, "flydsl_qblock": _flydsl_qblock}
    flops = 4.0 * n_tok * num_heads * head_dim * topk
    elem = q.element_size()
    nbytes = (n_tok * topk * head_dim + 2 * n_tok * num_heads * head_dim) * elem
    n_iters = 11 if n_tok >= 4096 else 51
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_iters=n_iters, num_warmup=2)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: sparse_mla_qblock {pattern}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("sparse_mla_qblock unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(description="gfx942 FlyDSL sparse MLA perf")
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
    )
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[7, 16384])
    parser.add_argument("--topk", type=int, nargs="*", default=[TOPK_FLASH])
    parser.add_argument("--pool", type=int, nargs="*", default=[POOL_FLASH])
    parser.add_argument("--pattern", type=str, nargs="*", default=["shared", "random"])
    args = parser.parse_args()

    for dtype in args.dtype:
        rows = []
        for n_tok, topk, pool, pattern in itertools.product(
            args.tokens, args.topk, args.pool, args.pattern
        ):
            rows.append(
                test_sparse_mla_qblock(
                    n_tok, H_FLASH, D_FLASH, topk, pool, pattern, dtype
                )
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "sparse_mla_qblock summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
