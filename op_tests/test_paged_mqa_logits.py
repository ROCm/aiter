# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``aiter.paged_mqa_logits`` (tuned dispatch) vs each backend vs torch.

Inputs follow the DSA indexer decode call: compact Q ``[B, next_n, 32, 128]``,
a 16x16-preshuffled ``[num_blocks, 64, 1, 132]`` uint8 KV cache with scattered
pages, a preallocated ``[B * next_n, max_model_len]`` output, and context
lengths either per sequence ``[B]`` or per Q row ``[B, next_n]`` (vLLM passes
the 2D table).
"""

from __future__ import annotations

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import paged_mqa_logits as pmql
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
HEADS = 32
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
MAX_MODEL_LEN = 131072


def _preshuffle(raw, head_dim):
    from aiter.ops.shuffle import shuffle_weight

    num_blocks, kvb, one, index_dim = raw.shape
    flat = raw.reshape(num_blocks, kvb * index_dim).clone()
    keys = flat[:, : kvb * head_dim].contiguous().view(num_blocks, kvb, head_dim)
    flat[:, : kvb * head_dim] = shuffle_weight(keys, layout=(16, 16)).reshape(
        num_blocks, kvb * head_dim
    )
    return flat.view(num_blocks, kvb, one, index_dim)


def build_inputs(shape, context_len, max_model_len, seed=0):
    """Decode tensors; the KV pool is sized to the real demand so it is not
    cache resident, and each sequence's pages are scattered across it."""
    batch_size, next_n, heads, head_dim, kvb, preshuffle = shape
    torch.manual_seed(seed)
    pages = math.ceil(context_len / kvb)
    num_blocks = batch_size * pages

    kv = torch.randn((num_blocks, kvb, head_dim), dtype=torch.bfloat16)
    scale = kv.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4) / 240.0
    raw = torch.empty((num_blocks, kvb * (head_dim + 4)), dtype=torch.uint8)
    raw[:, : kvb * head_dim] = (
        (kv * (1.0 / scale)).to(dtypes.fp8).view(num_blocks, -1).view(torch.uint8)
    )
    raw[:, kvb * head_dim :] = scale.reshape(num_blocks, kvb).view(torch.uint8)
    raw = raw.view(num_blocks, kvb, 1, head_dim + 4)

    block_tables = torch.zeros(
        (batch_size, math.ceil(max_model_len / kvb)), dtype=torch.int32
    )
    block_tables[:, :pages] = (
        torch.randperm(num_blocks).view(batch_size, pages).to(torch.int32)
    )
    return {
        "q_fp8": torch.randn(
            (batch_size, next_n, heads, head_dim), dtype=torch.bfloat16
        ).to(dtypes.fp8),
        "kv_raw": raw,
        "kv_cache": _preshuffle(raw, head_dim) if preshuffle else raw,
        "weights": torch.randn((batch_size * next_n, heads), dtype=torch.float32),
        "out_logits": torch.full(
            (batch_size * next_n, max_model_len), float("-inf"), dtype=torch.float32
        ),
        "context_lens": torch.full((batch_size,), context_len, dtype=torch.int32),
        "kv_indices": block_tables,
        "max_model_len": max_model_len,
        "Preshuffle": preshuffle,
        "KVBlockSize": kvb,
    }


def run_torch(inp, context_len):
    """``[B * next_n, context_len]`` logits, -inf past each row's causal bound.
    Reference only: not timed, not in the table."""
    q = inp["q_fp8"].float()
    batch_size, next_n, _, head_dim = q.shape
    kvb = inp["KVBlockSize"]
    flat = inp["kv_raw"].reshape(inp["kv_raw"].shape[0], -1)
    keys = flat[:, : kvb * head_dim].contiguous().view(dtypes.fp8).float()
    scales = flat[:, kvb * head_dim :].contiguous().view(torch.float32)
    kv = keys.view(-1, kvb, head_dim) * scales.view(-1, kvb, 1)

    pos = torch.arange(context_len)
    blk = inp["kv_indices"][:, pos // kvb]
    q_lim = context_len - next_n + torch.arange(next_n)
    causal = pos[None, :] <= q_lim[:, None]
    out = torch.empty((batch_size * next_n, context_len), dtype=torch.float32)
    for b in range(batch_size):
        s = torch.einsum("nhd,pd->nhp", q[b], kv[blk[b], pos % kvb]).relu()
        w = inp["weights"][b * next_n : (b + 1) * next_n]
        s = (s * w[:, :, None]).sum(dim=1)
        out[b * next_n : (b + 1) * next_n] = torch.where(causal, s, float("-inf"))
    return out


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def launch_args(inp):
    return (
        inp["q_fp8"],
        inp["kv_cache"],
        inp["weights"],
        inp["out_logits"],
        inp["context_lens"],
        inp["kv_indices"],
        inp["max_model_len"],
        inp["Preshuffle"],
        inp["KVBlockSize"],
    )


def _row_context_lens(context_len, batch_size, next_n):
    """vLLM's (B, next_n) table: row j of a sequence sees context_len - next_n + 1 + j."""
    rows = context_len - next_n + 1 + torch.arange(next_n, dtype=torch.int32)
    return rows.expand(batch_size, next_n).contiguous()


@benchmark()
def test_paged_mqa_logits(batch, next_n, kv_len, context_lens_dim):
    shape = (batch, next_n, HEADS, HEAD_DIM, KV_BLOCK_SIZE, True)
    inp = build_inputs(shape, kv_len, MAX_MODEL_LEN)
    if context_lens_dim == 2:
        inp["context_lens"] = _row_context_lens(kv_len, batch, next_n)
    ref = run_torch(inp, kv_len)
    args = launch_args(inp)
    picked = pmql.get_paged_mqa_logits_config(
        batch, next_n, HEADS, HEAD_DIM, KV_BLOCK_SIZE, True
    )

    candidates = {
        "paged_mqa_logits": (pmql.paged_mqa_logits, args),
        "gluon": (pmql.run_paged_mqa_logits, (pmql.DEFAULT_CONFIG, *args)),
    }
    # FlyDSL is gfx950-only and needs H=32, D=128, KVBlockSize=64, preshuffled.
    if pmql.flydsl_supports(
        inp["q_fp8"], inp["kv_cache"], inp["weights"], True, KV_BLOCK_SIZE
    ):
        flydsl = {"backend": "flydsl", "ChunkK": 0, "WavePerEU": 0, "wg_per_cu": 0}
        candidates["flydsl"] = (pmql.run_paged_mqa_logits, (flydsl, *args))

    flops = 2 * HEADS * HEAD_DIM * batch * next_n * kv_len
    nbytes = (
        inp["q_fp8"].numel() * inp["q_fp8"].element_size()
        + batch * kv_len * (HEAD_DIM + 4)
        + inp["weights"].numel() * inp["weights"].element_size()
        + batch * next_n * kv_len * 4
    )
    ret = {"gfx": get_gfx(), "picked": picked["backend"]}
    ref_inf = torch.isneginf(ref)
    for name, (fn, fn_args) in candidates.items():
        inp["out_logits"].fill_(float("-inf"))
        # Tensors go in as args so run_perftest rotates copies past the caches;
        # in a model every layer reads its own KV cache.
        with torch.inference_mode():
            out, us = run_perftest(fn, *fn_args)
        got = out[:, :kv_len]
        assert torch.equal(torch.isneginf(got), ref_inf), f"{name}: -inf mask mismatch"
        got, want = got.masked_fill(ref_inf, 0), ref.masked_fill(ref_inf, 0)
        diff = calc_diff(want, got)
        assert diff < 1e-3, f"{name}: calc_diff={diff}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            want.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=1e-2,
            atol=5.0,
            msg=f"{name}: paged_mqa_logits",
            printLog=False,
        )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("paged_mqa_logits unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 16, 128],
        help="Sequences per decode step.\n    e.g.: -b 4",
    )
    parser.add_argument(
        "--next-n",
        type=int,
        nargs="*",
        default=[1, 3, 5],
        help="Q rows per sequence (1 + speculative tokens).\n    e.g.: --next-n 1 8",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        nargs="*",
        default=[8192],
        help="Tokens per sequence.\n    e.g.: --kv-len 8192 65536",
    )
    parser.add_argument(
        "--context-lens-dim",
        type=int,
        nargs="*",
        default=[1, 2],
        choices=[1, 2],
        help="context_lens as [B] (1) or vLLM's per-row [B, next_n] (2).",
    )
    args = parser.parse_args()

    rows = [
        test_paged_mqa_logits(batch, next_n, kv_len, dim)
        for batch, next_n, kv_len, dim in itertools.product(
            args.batch, args.next_n, args.kv_len, args.context_lens_dim
        )
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "paged_mqa_logits summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
