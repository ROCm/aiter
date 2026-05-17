# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test the v5 paged-attention decode kernel that ships in
# hsa/gfx942/pa_gqa_v5/asm_pa_gqa_v5.co and is dispatched from
# `aiter.pa_gqa_v5_decode` (mirrors the topk_per_row_decode .co pattern).
#
# Compares against the in-tree HIP backend (`paged_attention_rocm_core`) for:
#   - accuracy: max |v5 - hip| within bf16 tolerance
#   - performance: per-call median latency
#
# Default benchmark configuration (matches the v5 hard-specialised kernels):
#   bf16, num_q_heads=8, num_kv_heads=1, head_size=128,
#   block_size=16, mtp=2 (seq_len_q=2), partition_size=256
#
# Usage:
#   HIP_VISIBLE_DEVICES=0 python op_tests/test_pa_gqa_v5.py
#   HIP_VISIBLE_DEVICES=0 python op_tests/test_pa_gqa_v5.py --bs 8 --ctx 65536
#   HIP_VISIBLE_DEVICES=0 python op_tests/test_pa_gqa_v5.py --check_only
import argparse
import math
import statistics
import sys
from typing import Tuple

import pandas as pd
import torch

import aiter  # registers torch.ops.aiter.* and the JIT module loaders
from aiter.ops.attention import pa_gqa_v5_decode
from aiter.test_common import benchmark, checkAllclose
from csrc.cpp_itfs.pa.pa import (
    paged_attention_rocm as paged_attention_rocm_core,  # type: ignore
)


PARTITION_SIZE = 256
NUM_Q_HEADS    = 8
NUM_KV_HEADS   = 1
HEAD_SIZE      = 128
BLOCK_SIZE     = 16
MTP            = 2


def _make_inputs(bs: int, ctx: int, seed: int = 0,
                 layout: str = "random") -> Tuple[torch.Tensor, ...]:
    """Same generation rule as bench_paged_attn_decode.py."""
    import random
    random.seed(seed + bs * 31 + ctx)
    torch.manual_seed(seed + bs * 31 + ctx)

    n_blk_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_blk_tot     = bs * n_blk_per_seq

    q  = torch.empty(bs * MTP, NUM_Q_HEADS, HEAD_SIZE,
                     dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    kc = torch.empty(n_blk_tot, NUM_KV_HEADS, HEAD_SIZE // 8, BLOCK_SIZE, 8,
                     dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    vc = torch.empty(n_blk_tot, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE,
                     dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    sl = torch.full((bs,), ctx, dtype=torch.int32, device="cuda")
    if layout == "random":
        rows = [
            [random.randint(0, n_blk_tot - 1) for _ in range(n_blk_per_seq)]
            for _ in range(bs)
        ]
        bt = torch.tensor(rows, dtype=torch.int32, device="cuda")
    else:
        bt = torch.arange(n_blk_tot, dtype=torch.int32,
                          device="cuda").view(bs, n_blk_per_seq)
    return q, kc, vc, bt, sl


def _make_workspace(bs: int, max_num_partitions: int):
    """Pre-allocate (out, exp_sums, max_logits, tmp_out) tensors with the
    exact layout `pa_gqa_v5_decode` and `paged_attention_rocm_core` expect."""
    n = bs * MTP
    out  = torch.empty((n, NUM_Q_HEADS, HEAD_SIZE),
                       dtype=torch.bfloat16, device="cuda")
    es   = torch.empty((n, NUM_Q_HEADS, max_num_partitions),
                       dtype=torch.float32, device="cuda")
    ml   = torch.empty_like(es)
    tmp  = torch.empty((n, NUM_Q_HEADS, max_num_partitions, HEAD_SIZE),
                       dtype=torch.bfloat16, device="cuda")
    return out, es, ml, tmp


def _run_v5(q, kc, vc, bt, sl, ctx: int, *, ws=None):
    """Native aiter v5: `aiter.ops.attention.pa_gqa_v5_decode` → asm_pa_gqa_v5.co."""
    bs = q.shape[0] // MTP
    max_num_partitions = (ctx + PARTITION_SIZE - 1) // PARTITION_SIZE
    if ws is None:
        ws = _make_workspace(bs, max_num_partitions)
    out, es, ml, tmp = ws
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    pa_gqa_v5_decode(
        out, es, ml, tmp,
        q, kc, vc, bt, sl,
        NUM_KV_HEADS, scale, BLOCK_SIZE, ctx,
        PARTITION_SIZE, MTP,
    )
    return out


def _run_hip(q, kc, vc, bt, sl, ctx: int, *, ws=None):
    """Reference: in-tree HIP launcher (`paged_attention_rocm_core`)."""
    bs = q.shape[0] // MTP
    max_num_partitions = (ctx + PARTITION_SIZE - 1) // PARTITION_SIZE
    if ws is None:
        ws = _make_workspace(bs, max_num_partitions)
    out, es, ml, tmp = ws
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    paged_attention_rocm_core(
        out, es, ml, tmp,
        q, kc, vc, NUM_KV_HEADS, scale,
        bt, sl, BLOCK_SIZE, ctx,
        None, "auto",
        torch.tensor([], device="cuda"),
        torch.tensor([], device="cuda"),
        None, PARTITION_SIZE, MTP, None,
    )
    return out


def _time_fn(fn, *, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    times.sort()
    trim = max(1, iters // 10)
    return statistics.mean(times[trim:-trim])


@benchmark()
def test_pa_gqa_v5(
    bs: int,
    ctx: int,
    iters: int = 200,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    check_only: bool = False,
) -> dict:
    """Accuracy + perf comparison vs in-tree HIP."""
    ret: dict = {}

    q, kc, vc, bt, sl = _make_inputs(bs, ctx)
    max_num_partitions = (ctx + PARTITION_SIZE - 1) // PARTITION_SIZE

    # ---- accuracy ----
    out_hip = _run_hip(q, kc, vc, bt, sl, ctx)
    out_v5  = _run_v5(q, kc, vc, bt, sl, ctx)

    diff      = (out_v5.float() - out_hip.float()).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff  = (diff / (out_hip.float().abs() + 1e-3)).max().item()

    err_pct = checkAllclose(
        out_v5.float(), out_hip.float(), rtol=rtol, atol=atol,
        msg=f"[v5 vs paged_attention_rocm_core]  bs={bs} ctx={ctx}",
        printLog=False,
    )

    ret["max_abs_diff"]  = max_diff
    ret["mean_abs_diff"] = mean_diff
    ret["max_rel_diff"]  = rel_diff
    ret["err_pct"]       = err_pct
    ret["acc_pass"]      = bool(max_diff <= atol or rel_diff <= rtol)

    if check_only:
        return ret

    # ---- perf ----
    ws_hip = _make_workspace(bs, max_num_partitions)
    ws_v5  = _make_workspace(bs, max_num_partitions)

    us_hip = _time_fn(
        lambda: _run_hip(q, kc, vc, bt, sl, ctx, ws=ws_hip),
        iters=iters,
    )
    us_v5 = _time_fn(
        lambda: _run_v5(q, kc, vc, bt, sl, ctx, ws=ws_v5),
        iters=iters,
    )

    ret["us_hip"]    = round(us_hip, 2)
    ret["us_v5"]     = round(us_v5, 2)
    ret["hip/v5"]    = round(us_hip / us_v5, 3)
    ret["v5/hip"]    = round(us_v5 / us_hip, 3)
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test pa_gqa v5 (.co) vs paged_attention_rocm_core (HIP).",
)
parser.add_argument("--bs", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16],
                    help="Batch sizes to sweep")
parser.add_argument("--ctx", type=int, nargs="+",
                    default=[1024, 4096, 16384, 32768, 65536, 131072],
                    help="Context lengths to sweep")
parser.add_argument("--iters", type=int, default=200,
                    help="Timing iterations per cell")
parser.add_argument("--atol", type=float, default=1e-2)
parser.add_argument("--rtol", type=float, default=1e-2)
parser.add_argument("--check_only", action="store_true",
                    help="Skip perf timing, only check accuracy.")

args = parser.parse_args()


df = []
for bs in args.bs:
    for ctx in args.ctx:
        try:
            ret = test_pa_gqa_v5(
                bs, ctx,
                iters=args.iters,
                atol=args.atol, rtol=args.rtol,
                check_only=args.check_only,
            )
            df.append(ret)
        except torch.cuda.OutOfMemoryError:
            df.append({"bs": bs, "ctx": ctx, "OOM": True})

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("pa_gqa v5 (.co) vs paged_attention_rocm_core summary:\n%s", df_md)

if "acc_pass" in df.columns:
    n_total  = (~df["acc_pass"].isna()).sum()
    n_passed = df["acc_pass"].fillna(False).sum()
    aiter.logger.info("Accuracy: %d/%d cells pass", int(n_passed), int(n_total))
    if int(n_passed) != int(n_total):
        sys.exit(1)
