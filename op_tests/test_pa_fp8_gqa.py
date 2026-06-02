# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test the project-internal FP8 GQA paged-attention decode path dispatched from
# `PagedAttention.forward_decode`.
#
# Default configuration matches the specialised kernel:
#   Q/K/V fp8_e4m3fnuz, num_q_heads=8, num_kv_heads=1, head_size=128,
#   block_size=16, partition_size=256, Q/K per-token scale, V per-head scale.
#
# Usage:
#   HIP_VISIBLE_DEVICES=0 python op_tests/test_pa_fp8_gqa.py --check_only
#   HIP_VISIBLE_DEVICES=0 python op_tests/test_pa_fp8_gqa.py --bs 2 16 --ctx 64
import argparse
import math
import statistics
import sys
from typing import Optional, Tuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.paged_attn import PagedAttention
from aiter.ops.attention import _pa_fp8_gqa_eligible
from aiter.test_common import benchmark, checkAllclose


PARTITION_SIZE = 256
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_SIZE = 128
BLOCK_SIZE = 16
FP8 = dtypes.fp8
DEFAULT_P_SCALE = 256.0


def _make_inputs(bs: int, ctx: int, mtp: int, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + bs * 31 + ctx * 17 + mtp)

    blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * blocks_per_seq
    fp8_max = torch.finfo(FP8).max

    q_bf16 = torch.randn(
        bs * mtp, NUM_Q_HEADS, HEAD_SIZE, dtype=dtypes.bf16, device="cuda")
    k_bf16 = torch.randn(
        num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE, dtype=dtypes.bf16, device="cuda")
    v_bf16 = torch.randn_like(k_bf16)

    q_scale = (q_bf16.float().abs().amax(dim=-1) / fp8_max).clamp_min(1e-6)
    k_scale = (k_bf16.float().abs().amax(dim=2) / fp8_max).clamp_min(1e-6)
    v_scale = (
        v_bf16.float()
        .abs()
        .permute(1, 0, 2, 3)
        .reshape(NUM_KV_HEADS, -1)
        .amax(dim=1)
        / fp8_max
    ).clamp_min(1e-6)

    query = (q_bf16.float() / q_scale[:, :, None]).clamp(-fp8_max, fp8_max).to(FP8)
    k_cache = (
        (k_bf16.float() / k_scale[:, :, None, :])
        .clamp(-fp8_max, fp8_max)
        .to(FP8)
        .view(num_blocks, NUM_KV_HEADS, HEAD_SIZE // 16, 16, BLOCK_SIZE)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    value_cache_4d = (
        v_bf16.float() / v_scale[None, :, None, None]
    ).clamp(-fp8_max, fp8_max).to(FP8)
    value_cache = (
        value_cache_4d
        .view(num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )

    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda").view(bs, blocks_per_seq)
    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device="cuda")
    return query, k_cache, value_cache, block_tables, seq_lens, q_scale, k_scale, v_scale


def _reference_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    mtp: int,
    scale: float,
    p_scale: Optional[torch.Tensor] = None,
    p_scale_inv: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = dtypes.bf16,
) -> torch.Tensor:
    bs = seq_lens.numel()
    out = torch.empty(
        (query.shape[0], NUM_Q_HEADS, HEAD_SIZE), dtype=torch.float32, device=query.device)

    q_deq = query.float() * q_scale[:, :, None]
    if p_scale is None:
        p_scale = torch.full(
            (NUM_Q_HEADS,), DEFAULT_P_SCALE,
            dtype=torch.float32, device=query.device)
        p_scale_inv = torch.full(
            (NUM_Q_HEADS,), 1.0 / DEFAULT_P_SCALE,
            dtype=torch.float32, device=query.device)
    elif p_scale.numel() == 1 and p_scale_inv is not None and p_scale_inv.numel() == 1:
        p_scale = p_scale.reshape(1).expand(NUM_Q_HEADS)
        p_scale_inv = p_scale_inv.reshape(1).expand(NUM_Q_HEADS)
    for seq_idx in range(bs):
        ctx_len = int(seq_lens[seq_idx].item())
        k_tokens = torch.empty(
            (ctx_len, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32, device=query.device)
        v_tokens = torch.empty_like(k_tokens)
        for pos in range(ctx_len):
            block_id = int(block_tables[seq_idx, pos // BLOCK_SIZE].item())
            block_offset = pos % BLOCK_SIZE
            k_tokens[pos] = (
                k_cache[block_id, :, :, block_offset, :]
                .reshape(NUM_KV_HEADS, HEAD_SIZE)
                .float()
                * k_scale[block_id, :, block_offset, None]
            )
            if value_cache.dim() == 5:
                v_raw = value_cache[
                    block_id, :, block_offset // 16, :, block_offset % 16]
            else:
                v_raw = value_cache[block_id, :, :, block_offset]
            v_tokens[pos] = v_raw.float() * v_scale[:, None]

        for mtp_idx in range(mtp):
            q_idx = seq_idx * mtp + mtp_idx
            for head_idx in range(NUM_Q_HEADS):
                kv_head = head_idx // (NUM_Q_HEADS // NUM_KV_HEADS)
                logits = torch.matmul(k_tokens[:, kv_head], q_deq[q_idx, head_idx]) * scale
                probs = torch.exp(logits - logits.max())
                denom = probs.sum()
                probs = probs * p_scale[head_idx]
                probs = probs.clamp(-torch.finfo(FP8).max, torch.finfo(FP8).max).to(FP8).float()
                value = torch.matmul(probs, v_tokens[:, kv_head]) / denom
                value = value * p_scale_inv[head_idx]
                out[q_idx, head_idx] = value
    return out.to(output_dtype)


def _run_paged_attention(
    query, k_cache, value_cache, block_tables, seq_lens, q_scale, k_scale, v_scale,
    *, ctx: int, mtp: int, p_scale=None, p_scale_inv=None,
):
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    output = torch.empty_like(query, dtype=dtypes.bf16)
    assert _pa_fp8_gqa_eligible(
        query, k_cache, value_cache, output,
        NUM_KV_HEADS, BLOCK_SIZE, PARTITION_SIZE, mtp,
        None, None, q_scale, k_scale, v_scale, p_scale, p_scale_inv,
    )
    return PagedAttention.forward_decode(
        query,
        k_cache,
        value_cache,
        block_tables,
        seq_lens,
        ctx,
        kv_cache_dtype="auto",
        num_kv_heads=NUM_KV_HEADS,
        scale=scale,
        alibi_slopes=None,
        k_scale=k_scale,
        v_scale=v_scale,
        q_scale=q_scale,
        mtp=mtp,
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
    )


def _make_pscale(mode: str, device: torch.device):
    if mode in ("none", "default_256"):
        return None, None
    if mode == "scalar_256":
        p = torch.tensor(DEFAULT_P_SCALE, dtype=torch.float32, device=device)
        return p, torch.tensor(1.0 / DEFAULT_P_SCALE, dtype=torch.float32, device=device)
    if mode == "all_ones":
        p = torch.ones(NUM_Q_HEADS, dtype=torch.float32, device=device)
        return p, p.clone()
    if mode == "all_2":
        p = torch.full((NUM_Q_HEADS,), 2.0, dtype=torch.float32, device=device)
        return p, torch.full((NUM_Q_HEADS,), 0.5, dtype=torch.float32, device=device)
    if mode == "per_head_random":
        g = torch.Generator(device=device).manual_seed(20240514)
        p = 0.7 + 0.8 * torch.rand(
            NUM_Q_HEADS, generator=g, device=device, dtype=torch.float32)
        return p, 1.0 / p
    raise ValueError(mode)


def _time_fn(fn, *, warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
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
def test_pa_fp8_gqa(
    bs: int,
    ctx: int,
    mtp: int = 1,
    iters: int = 100,
    atol: float = 2e-1,
    rtol: float = 2e-1,
    check_only: bool = False,
    p_scale_mode: str = "none",
) -> dict:
    ret: dict = {}
    data = _make_inputs(bs, ctx, mtp)
    query, k_cache, value_cache, block_tables, seq_lens, q_scale, k_scale, v_scale = data
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    p_scale, p_scale_inv = _make_pscale(p_scale_mode, query.device)

    out_kernel = _run_paged_attention(
        query, k_cache, value_cache, block_tables, seq_lens, q_scale, k_scale, v_scale,
        ctx=ctx, mtp=mtp, p_scale=p_scale, p_scale_inv=p_scale_inv,
    )
    out_ref = _reference_decode(
        query, k_cache, value_cache, block_tables, seq_lens,
        q_scale, k_scale, v_scale, mtp=mtp, scale=scale,
        p_scale=p_scale, p_scale_inv=p_scale_inv,
    )

    diff = (out_kernel.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (out_ref.float().abs() + 1e-3)).max().item()
    err_pct = checkAllclose(
        out_kernel.float(), out_ref.float(), rtol=rtol, atol=atol,
        msg=(
            f"[FP8 GQA vs torch ref] bs={bs} ctx={ctx} "
            f"mtp={mtp} p_scale={p_scale_mode}"
        ),
        printLog=False,
    )

    ret["max_abs_diff"] = max_diff
    ret["mean_abs_diff"] = mean_diff
    ret["max_rel_diff"] = rel_diff
    ret["err_pct"] = err_pct
    ret["acc_pass"] = bool(max_diff <= atol or rel_diff <= rtol)

    if not check_only:
        us_kernel = _time_fn(
            lambda: _run_paged_attention(
                query, k_cache, value_cache, block_tables, seq_lens,
                q_scale, k_scale, v_scale, ctx=ctx, mtp=mtp,
                p_scale=p_scale, p_scale_inv=p_scale_inv),
            iters=iters,
        )
        ret["us_fp8_gqa"] = round(us_kernel, 2)
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test project-internal FP8 GQA paged-attention decode.",
)
parser.add_argument("--bs", type=int, nargs="+", default=[2, 16],
                    help="Batch sizes to sweep; bs=2 exercises v1, bs=16 exercises v2 for mtp=1.")
parser.add_argument("--ctx", type=int, nargs="+", default=[64],
                    help="Context lengths to sweep.")
parser.add_argument("--mtp", type=int, nargs="+", default=[1, 2],
                    help="MTP values to sweep.")
parser.add_argument("--p_scale_mode", type=str, nargs="+",
                    default=["default_256", "scalar_256", "all_2", "per_head_random"],
                    choices=[
                        "none", "default_256", "scalar_256",
                        "all_ones", "all_2", "per_head_random",
                    ],
                    help="P scale modes to sweep; none/default_256 use the kernel default.")
parser.add_argument("--iters", type=int, default=100,
                    help="Timing iterations per cell.")
parser.add_argument("--atol", type=float, default=2e-1)
parser.add_argument("--rtol", type=float, default=2e-1)
parser.add_argument("--check_only", action="store_true",
                    help="Skip perf timing, only check correctness.")
args = parser.parse_args()


if not torch.cuda.is_available():
    aiter.logger.warning("CUDA/HIP device unavailable, skip test_pa_fp8_gqa.")
    sys.exit(0)

df = []
for bs in args.bs:
    for ctx in args.ctx:
        for mtp in args.mtp:
            for p_scale_mode in args.p_scale_mode:
                try:
                    df.append(test_pa_fp8_gqa(
                        bs, ctx, mtp=mtp, iters=args.iters,
                        atol=args.atol, rtol=args.rtol,
                        check_only=args.check_only,
                        p_scale_mode=p_scale_mode,
                    ))
                except torch.cuda.OutOfMemoryError:
                    df.append({
                        "bs": bs, "ctx": ctx, "mtp": mtp,
                        "p_scale_mode": p_scale_mode, "OOM": True})

df = pd.DataFrame(df)
aiter.logger.info("FP8 GQA paged-attention summary:\n%s", df.to_markdown(index=False))

if "acc_pass" in df.columns:
    n_total = (~df["acc_pass"].isna()).sum()
    n_passed = df["acc_pass"].fillna(False).sum()
    aiter.logger.info("Accuracy: %d/%d cells pass", int(n_passed), int(n_total))
    if int(n_passed) != int(n_total):
        sys.exit(1)
