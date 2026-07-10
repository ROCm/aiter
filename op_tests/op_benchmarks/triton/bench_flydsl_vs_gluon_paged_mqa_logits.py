# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Honest A/B: FlyDSL paged MQA-logits vs the production Gluon/Triton kernel.

Both kernels are timed by the SAME code path: identical inputs (built once and
fed to both), identical ``run_perftest`` methodology (device self-time, IQR
cleaning, same warmup/iters), identical output dtype/layout, and both are
correctness-gated against the SAME torch reference before any timing is reported.

The reference kernel is the production ``deepgemm_fp8_paged_mqa_logits`` host
(aiter Triton/Gluon), run with vLLM's production defaults (ChunkK=256,
WavePerEU=2, KVBlockSize=1, Preshuffle=False) -- NOT detuned. Both the reference
host and the FlyDSL host auto-compute SplitKV to fill the device on small decode
grids, so the comparison is apples-to-apples.

Reuses the paged benchmark's fp8 cache packer to guarantee the same input build.
"""

import argparse
import random
import statistics

import torch

from aiter.test_common import run_perftest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits

# same fp8 co-pack builder the production paged benchmark uses
from op_tests.op_benchmarks.triton.bench_deepgemm_attention import (
    kv_cache_cast_to_fp8,
    cdiv,
)

torch.set_default_device("cuda")

# Production reference config (matches vLLM rocm_fp8_paged_mqa_logits for decode).
REF_CHUNK_K = 256
REF_WAVE_PER_EU = 2


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denominator).item()


def ref_fp8_paged_mqa_logits(
    q, kv_cache_fp8, weights, context_lens, block_tables, max_model_len, fp8_dtype
):
    """Vectorized torch reference (dequantizes the co-packed fp8 cache)."""
    batch_size, next_n, heads, dim = q.size()
    kvv, scale = kv_cache_fp8[..., :dim], kv_cache_fp8[..., dim:]
    scale = scale.contiguous().view(torch.float)
    qf = q.float()
    kvf = (kvv.view(fp8_dtype).float() * scale).view(kv_cache_fp8.shape[0], dim)
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = int(context_lens[i].item())
        if context_len == 0:
            continue
        pages = block_tables[i, :context_len]
        kx = kvf[pages]
        s = torch.einsum("nhd,pd->nhp", qf[i], kx)
        s = torch.relu(s)
        wl = weights[i * next_n : (i + 1) * next_n, :]
        s = (s * wl[:, :, None]).sum(dim=1)
        p = torch.arange(context_len, device=q.device)
        q_lim = (
            context_len - next_n + torch.arange(next_n, device=q.device)
        ).unsqueeze(1)
        s = torch.where(p[None, :] <= q_lim, s, float("-inf"))
        logits[i * next_n : (i + 1) * next_n, :context_len] = s
    return logits


def parity_mask(context_lens, batch_size, next_n, max_model_len):
    """Causal-valid mask (bench methodology): positions <= ctx - next_n + off."""
    positions = (
        torch.arange(max_model_len, device="cuda")
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
    return positions <= (
        context_lens[row_indices] - next_n + next_n_offset
    ).unsqueeze(1)


def build_inputs(batch_size, next_n, heads, index_dim, avg_kv_length, seed=0):
    """Mirror the production paged benchmark's input builder (blocksize==1)."""
    torch.manual_seed(seed)
    random.seed(seed)
    fp8_dtype = get_fp8_e4m3_dtype()

    max_model_len = 2 * avg_kv_length
    blocksize = 1
    num_blocks = (max_model_len + blocksize - 1) // blocksize

    var_ratio = 0.5
    context_lens = (
        torch.randint(
            int((1 - var_ratio) * avg_kv_length),
            int((1 + var_ratio) * avg_kv_length) + 1,
            (batch_size,),
        )
        .cuda()
        .to(torch.int32)
    )
    # decode with MTP needs at least next_n tokens of context.
    context_lens = torch.clamp(context_lens, min=next_n)

    q = torch.randn(
        (batch_size, next_n, heads, index_dim), device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, index_dim), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device="cuda", dtype=torch.float32
    )

    max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
    block_tables = torch.zeros(
        (batch_size, max_block_len), device="cuda", dtype=torch.int32
    )
    pool = list(range(num_blocks))
    random.shuffle(pool)
    pool_t = torch.tensor(pool, device="cuda", dtype=torch.int32)
    counter = 0
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        n = cdiv(ctx_len, blocksize)
        idx = (counter + torch.arange(n, device="cuda")) % num_blocks
        block_tables[i, :n] = pool_t[idx]
        counter += n

    q_fp8 = q.to(fp8_dtype)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, padding=False, fp8_dtype=fp8_dtype)
    return (
        q,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
    )


def time_kernel(fn, repeats, num_iters):
    us = []
    for _ in range(repeats):
        _, u = run_perftest(fn, num_iters=num_iters)
        us.append(float(u))
    return statistics.median(us), min(us), max(us)


def run_shape(batch_size, next_n, heads, index_dim, avg_kv_length, repeats, num_iters):
    (
        q,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
    ) = build_inputs(batch_size, next_n, heads, index_dim, avg_kv_length)

    ref = ref_fp8_paged_mqa_logits(
        q, kv_cache_fp8, weights, context_lens, block_tables, max_model_len, fp8_dtype
    )
    valid = parity_mask(context_lens, batch_size, next_n, max_model_len)
    neg_inf = float("-inf")
    ref_neg_mask = ref == neg_inf

    def correctness(out):
        diff = calc_diff(out.masked_fill(~valid, 0), ref.masked_fill(~valid, 0))
        mask_ok = bool(torch.equal(out == neg_inf, ref_neg_mask))
        return diff, mask_ok

    # ---- reference (production Gluon/Triton), production defaults ----
    out_ref = torch.full(
        (batch_size * next_n, max_model_len), neg_inf, device="cuda", dtype=torch.float32
    )

    def fn_ref():
        deepgemm_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_ref,
            context_lens,
            block_tables,
            max_model_len,
            ChunkK=REF_CHUNK_K,
            Preshuffle=False,
            KVBlockSize=1,
            WavePerEU=REF_WAVE_PER_EU,
        )

    # ---- FlyDSL (KVBlockSize==1, auto SplitKV) ----
    out_fly = torch.full(
        (batch_size * next_n, max_model_len), neg_inf, device="cuda", dtype=torch.float32
    )

    def fn_fly():
        flydsl_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_fly,
            context_lens,
            block_tables,
            max_model_len,
        )

    # Warm / JIT-compile once (excluded from the reported device self-time anyway).
    fn_ref()
    fn_fly()
    torch.cuda.synchronize()

    ref_diff, ref_mask_ok = correctness(out_ref)
    fly_diff, fly_mask_ok = correctness(out_fly)
    ref_pass = (ref_diff < 1e-3) and ref_mask_ok
    fly_pass = (fly_diff < 1e-3) and fly_mask_ok

    ref_med = ref_min = ref_max = fly_med = fly_min = fly_max = float("nan")
    if ref_pass and fly_pass:
        ref_med, ref_min, ref_max = time_kernel(fn_ref, repeats, num_iters)
        fly_med, fly_min, fly_max = time_kernel(fn_fly, repeats, num_iters)

    return {
        "B": batch_size,
        "nn": next_n,
        "grid": batch_size * next_n,
        "H": heads,
        "D": index_dim,
        "avg_kv": avg_kv_length,
        "ref_ms": ref_med / 1e3,
        "ref_min_ms": ref_min / 1e3,
        "ref_max_ms": ref_max / 1e3,
        "fly_ms": fly_med / 1e3,
        "fly_min_ms": fly_min / 1e3,
        "fly_max_ms": fly_max / 1e3,
        "fly/ref": (fly_med / ref_med) if ref_med == ref_med and ref_med > 0 else float("nan"),
        "ref_diff": ref_diff,
        "fly_diff": fly_diff,
        "ref_mask": ref_mask_ok,
        "fly_mask": fly_mask_ok,
        "ref_pass": ref_pass,
        "fly_pass": fly_pass,
    }


# DSA lightning-indexer decode dims (heads=64, index_dim=128) swept over long
# contexts and a concurrency (batch) range; next_n=2 is speculative decode,
# next_n=1 non-speculative. All shapes reported, wins and losses.
def default_shapes():
    H, D = 64, 128
    shapes = []
    for B in (1, 4, 16, 64, 128):
        for avg_kv in (16384, 32768, 65536):
            shapes.append((B, 2, H, D, avg_kv))
    for B in (1, 16, 128):
        shapes.append((B, 1, H, D, 32768))
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=101)
    args = parser.parse_args()

    if get_gfx() not in ("gfx942", "gfx950"):
        print(f"unsupported gfx {get_gfx()}; skipping")
        return

    print(f"# arch={get_gfx()} fp8={get_fp8_e4m3_dtype()} "
          f"repeats={args.repeats} num_iters={args.num_iters} "
          f"ref_chunk_k={REF_CHUNK_K} ref_wave_per_eu={REF_WAVE_PER_EU}")
    header = (
        "B nn grid H D avg_kv | ref_ms[min-max] fly_ms[min-max] fly/ref | "
        "ref_diff fly_diff ref_mask fly_mask ref_pass fly_pass"
    )
    print(header)
    rows = []
    for (B, nn, H, D, avg) in default_shapes():
        r = run_shape(B, nn, H, D, avg, args.repeats, args.num_iters)
        rows.append(r)
        print(
            f"{r['B']} {r['nn']} {r['grid']} {r['H']} {r['D']} {r['avg_kv']} | "
            f"{r['ref_ms']:.4f}[{r['ref_min_ms']:.4f}-{r['ref_max_ms']:.4f}] "
            f"{r['fly_ms']:.4f}[{r['fly_min_ms']:.4f}-{r['fly_max_ms']:.4f}] "
            f"{r['fly/ref']:.2f}x | "
            f"{r['ref_diff']:.2e} {r['fly_diff']:.2e} "
            f"{r['ref_mask']} {r['fly_mask']} {r['ref_pass']} {r['fly_pass']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
