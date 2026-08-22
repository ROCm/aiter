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

This A/B is fixed at ``KVBlockSize=1``: the Gluon reference faults on a
non-padded block-flat cache at ``KVBlockSize>1``, so any KVBlockSize>1 (incl.
Preshuffle) comparison would not be apples-to-apples. FlyDSL KVBlockSize>1 and
Preshuffle correctness is covered against a torch oracle in
``op_tests/flydsl_tests/test_flydsl_fp8_paged_mqa_logits.py`` instead.

Reuses the paged benchmark's fp8 cache packer to guarantee the same input build.
"""

import argparse
import random
import statistics

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import run_perftest

# same fp8 co-pack builder the production paged benchmark uses
from op_tests.op_benchmarks.triton.bench_deepgemm_attention import (
    kv_cache_cast_to_fp8,
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
    batch_size, next_n, _heads, dim = q.size()
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
    return positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(
        1
    )


def build_inputs(
    batch_size,
    next_n,
    heads,
    index_dim,
    kv_length,
    seed=0,
    var_ratio=0.0,
    pool_blocks=0,
):
    """Paged input builder (blocksize==1).

    ``var_ratio`` controls the per-sequence context-length spread around
    ``kv_length``: 0.0 gives an exact length (every sequence == kv_length), and
    v>0 draws lengths uniformly from [(1-v)*kv_length, (1+v)*kv_length].

    The pool holds one distinct block per block the batch needs, so the KV
    footprint both kernels read is ``sum(ctx)*(index_dim+4)`` -- what a real
    decode touches. Sizing it from ``max_model_len`` instead (as an earlier
    version did) makes the block table wrap: at blocksize=1 the pool is
    2*kv_length blocks while the batch needs batch_size*kv_length, so from
    batch_size=4 up the sequences share blocks and the working set collapses
    into cache. At B=16, kv_length=32768 that is 8.65 MB standing in for 69 MB,
    which flatters both kernels and, because the two spend their time
    differently, does not flatter them equally. ``pool_blocks`` overrides the
    size to study cache residency deliberately.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    fp8_dtype = get_fp8_e4m3_dtype()

    max_model_len = 2 * kv_length
    blocksize = 1

    if var_ratio == 0.0:
        context_lens = torch.full(
            (batch_size,), kv_length, device="cuda", dtype=torch.int32
        )
    else:
        context_lens = (
            torch.randint(
                int((1 - var_ratio) * kv_length),
                int((1 + var_ratio) * kv_length) + 1,
                (batch_size,),
            )
            .cuda()
            .to(torch.int32)
        )
    # decode with MTP needs at least next_n tokens of context.
    context_lens = torch.clamp(context_lens, min=next_n)

    blocks_per_seq = (context_lens.to(torch.int64) + blocksize - 1) // blocksize
    max_block_len = int(blocks_per_seq.max().item())
    needed_blocks = int(blocks_per_seq.sum().item())
    num_blocks = needed_blocks if pool_blocks <= 0 else max(pool_blocks, max_block_len)

    # The kernels ride the per-token byte offset on a 32-bit buffer voffset, so
    # the whole pool must be addressable in i32 bytes.
    pool_bytes = num_blocks * blocksize * (index_dim + 4)
    if pool_bytes >= 2**31:
        raise SystemExit(
            f"pool of {num_blocks} blocks = {pool_bytes / 1e9:.2f} GB exceeds the "
            f"i32 gather-offset limit (2.15 GB). Shrink the shape or cap the pool "
            f"with --pool-blocks (which re-introduces block sharing)."
        )

    q = torch.randn(
        (batch_size, next_n, heads, index_dim), device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, index_dim), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device="cuda", dtype=torch.float32
    )

    # Hand blocks out of a shuffled pool in sequence order (scattered across the
    # pool, as they are after real paged allocation); pad each row's tail with 0.
    pool = list(range(num_blocks))
    random.shuffle(pool)
    pool_t = torch.tensor(pool, device="cuda", dtype=torch.int32)
    starts = torch.cumsum(blocks_per_seq, 0) - blocks_per_seq
    col = torch.arange(max_block_len, device="cuda", dtype=torch.int64)
    in_row = col[None, :] < blocks_per_seq[:, None]
    draw = (starts[:, None] + col[None, :]) % num_blocks
    block_tables = torch.where(
        in_row, pool_t[draw], torch.zeros((), device="cuda", dtype=torch.int32)
    ).to(torch.int32)

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
        min(needed_blocks, num_blocks) * blocksize * (index_dim + 4),
    )


def time_kernel(fn, repeats, num_iters):
    us = []
    for _ in range(repeats):
        _, u = run_perftest(fn, num_iters=num_iters)
        us.append(float(u))
    return statistics.median(us), min(us), max(us)


def run_shape(
    batch_size,
    next_n,
    heads,
    index_dim,
    kv_length,
    repeats,
    num_iters,
    var_ratio=0.0,
    pool_blocks=0,
):
    (
        q,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
        touched_bytes,
    ) = build_inputs(
        batch_size,
        next_n,
        heads,
        index_dim,
        kv_length,
        var_ratio=var_ratio,
        pool_blocks=pool_blocks,
    )

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
        (batch_size * next_n, max_model_len),
        neg_inf,
        device="cuda",
        dtype=torch.float32,
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
        (batch_size * next_n, max_model_len),
        neg_inf,
        device="cuda",
        dtype=torch.float32,
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
        "avg_kv": kv_length,
        "kv_mb": touched_bytes / 1e6,
        "ref_ms": ref_med / 1e3,
        "ref_min_ms": ref_min / 1e3,
        "ref_max_ms": ref_max / 1e3,
        "fly_ms": fly_med / 1e3,
        "fly_min_ms": fly_min / 1e3,
        "fly_max_ms": fly_max / 1e3,
        "fly/ref": (
            (fly_med / ref_med)
            if ref_med == ref_med and ref_med > 0  # noqa: PLR0124  (NaN check)
            else float("nan")
        ),
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
    parser.add_argument(
        "--var-ratio",
        type=float,
        default=0.0,
        help="Per-sequence context-length spread around kv_len. 0.0 = exact "
        "(every sequence == kv_len); e.g. 0.5 draws lengths in [0.5, 1.5]*kv_len.",
    )
    parser.add_argument(
        "--pool-blocks",
        type=int,
        default=0,
        help="physical blocks in the cache pool; 0 == one per block the batch "
        "needs (no sharing). Smaller values make sequences share blocks, "
        "shrinking the KV footprint into cache",
    )
    args = parser.parse_args()

    if get_gfx() not in ("gfx942", "gfx950"):
        print(f"unsupported gfx {get_gfx()}; skipping")
        return

    len_mode = "exact" if args.var_ratio == 0.0 else f"uniform +/-{args.var_ratio:g}"
    print(
        f"# arch={get_gfx()} fp8={get_fp8_e4m3_dtype()} "
        f"repeats={args.repeats} num_iters={args.num_iters} "
        f"ref_chunk_k={REF_CHUNK_K} ref_wave_per_eu={REF_WAVE_PER_EU} "
        f"ctx_len={len_mode} max_model_len=2*kv_len blocksize=1 "
        f"pool={'one block per block needed' if args.pool_blocks <= 0 else args.pool_blocks}"
    )
    print(
        "# kv_mb = distinct KV bytes the batch touches "
        "(both kernels read it once per query row)"
    )
    header = (
        "B nn grid H D kv_len kv_mb | ref_ms[min-max] fly_ms[min-max] fly/ref | "
        "ref_diff fly_diff ref_mask fly_mask ref_pass fly_pass"
    )
    print(header)
    rows = []
    for B, nn, H, D, kv_len in default_shapes():
        r = run_shape(
            B,
            nn,
            H,
            D,
            kv_len,
            args.repeats,
            args.num_iters,
            var_ratio=args.var_ratio,
            pool_blocks=args.pool_blocks,
        )
        rows.append(r)
        print(
            f"{r['B']} {r['nn']} {r['grid']} {r['H']} {r['D']} {r['avg_kv']} "
            f"{r['kv_mb']:.0f} | "
            f"{r['ref_ms']:.4f}[{r['ref_min_ms']:.4f}-{r['ref_max_ms']:.4f}] "
            f"{r['fly_ms']:.4f}[{r['fly_min_ms']:.4f}-{r['fly_max_ms']:.4f}] "
            f"{r['fly/ref']:.2f}x | "
            f"{r['ref_diff']:.2e} {r['fly_diff']:.2e} "
            f"{r['ref_mask']} {r['fly_mask']} {r['ref_pass']} {r['fly_pass']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
