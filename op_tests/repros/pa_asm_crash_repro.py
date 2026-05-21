#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Standalone aiter reproducer for ASM paged-attention crash observed in ATOM
# (Kimi-K2.5-MXFP4 + Eagle3 spec-decode) at 30k req x 128 conc.
#
# Background:
#   - Inside ATOM, attention_mha.py calls aiter.pa_fwd_asm via ASM-force path.
#     At 30k req x 128 concurrency, this crashes with HIP illegal memory access,
#     async-reported via event.synchronize(). Crash req# is highly variable
#     (observed 131 / 857 / ~900 / ~3945 across runs).
#   - The crash signature (wave dump) names kernels of the form
#       pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1.co        (fp8 KV)
#       pa_bf16_gqa8_1tg_4w_mtp_msk1.co                    (bf16 KV)
#     so the offending kernel is the ASM PA backend, GQA ratio 8, with mtp.
#   - The crash reproduces with both fp8 and bf16 KV cache. Forcing
#     self.kv_scale to be a 131072-element array did NOT fix it.
#   - Gluon-attention path under identical workload (30k x 128) is stable
#     (30000/30000 PASS). So root cause is in the ASM PA backend, not in ATOM.
#
# This script is a self-contained aiter-only stress driver that:
#   * matches ATOM's pa_fwd_asm call signature (incl. qo_indptr, K_QScale,
#     V_QScale, max_qlen, high_precision=0).
#   * uses GQA-8 (num_q_heads=8, num_kv_heads=1), block_size=16, head_dim=128
#     -> selects the same ASM .co binary as production.
#   * mixes mtp qlen in {1,2,3,4} per iteration to exercise the same kernel
#     variants the bench hits.
#   * varies batch_size (mostly large, ~128) and ctx_len each iteration
#     to imitate the request mix.
#   * launches many calls on multiple CUDA streams without sync between them,
#     because the bug is async-reported and a strict launch-and-sync loop may
#     not race the way 128-conc inflight requests do.
#   * periodically forces a sync, catches the HIP error, and prints the iter,
#     batch shape, and current call params so the ASM team can inspect.
#
# Usage:
#   # bf16 KV (no quant), default:
#   python pa_asm_crash_repro.py
#
#   # fp8 KV (matches the wave-dumped kernel from crash note):
#   python pa_asm_crash_repro.py --kv-dtype fp8
#
#   # tweak shape mix:
#   python pa_asm_crash_repro.py --kv-dtype fp8 --max-iters 200000 \
#       --streams 16 --sync-every 32

import argparse
import os
import random
import sys
import time
import traceback
from typing import List, Optional, Tuple

import torch

import aiter
from aiter import dtypes
from aiter import pertoken_quant


# --------------------------------------------------------------------------- #
# KV cache helpers (same layout as test_pa_mtp.py and aiter PA convention)
# --------------------------------------------------------------------------- #
def make_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate K/V cache in aiter PA layout.

    K: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    V: [num_blocks, num_kv_heads, head_size, block_size]
    where x = 16 // dtype.itemsize.
    """
    x = 16 // dtype.itemsize
    k_shape = (num_blocks, num_kv_heads, head_size // x, block_size, x)
    v_shape = (num_blocks, num_kv_heads, head_size, block_size)
    k_cache = torch.empty(k_shape, dtype=dtype, device=device).uniform_(-1, 1)
    v_cache = torch.empty(v_shape, dtype=dtype, device=device).uniform_(-1, 1)
    return k_cache, v_cache


def asm_v_shuffle(v_cache: torch.Tensor) -> torch.Tensor:
    """ASM PA expects V re-shuffled to [B, KVH, block_size/x, head_size, x]."""
    x = 16 // v_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = v_cache.shape
    v = v_cache.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    return v.permute(0, 1, 3, 2, 4).contiguous()


def pertoken_quant_kvcache_symm(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token symmetric quant of K/V to fp8 + scale arrays in ASM layout.

    Returns:
      k_quant: same layout as K cache, in `quant_dtype`
      v_quant: same layout as V cache, in `quant_dtype`
      k_scale_asm: [num_blocks, num_kv_heads, block_size, 1] (ASM-friendly)
      v_scale_asm: same shape
    """
    num_blocks, num_kv_heads = k_cache.shape[0], k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]

    k_perm = (
        k_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_kv_heads, block_size, -1)
        .contiguous()
    )
    v_perm = (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_kv_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale_asm = pertoken_quant(k_perm, quant_dtype=quant_dtype)
    v_quant, v_scale_asm = pertoken_quant(v_perm, quant_dtype=quant_dtype)

    quant_x = 16 // quant_dtype.itemsize
    k_quant = (
        k_quant.view(num_blocks, num_kv_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    v_quant = (
        v_quant.view(num_blocks, num_kv_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    return k_quant, v_quant, k_scale_asm, v_scale_asm


# --------------------------------------------------------------------------- #
# Iteration: pick a random request mix, build inputs, fire pa_fwd_asm
# --------------------------------------------------------------------------- #
def build_iter_inputs(
    rng: random.Random,
    num_kv_heads: int,
    num_q_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Construct one paged-attention call's inputs.

    Mirrors the production mix at 128 conc:
      - batch_size: most often near 128, sometimes 32/64/96/256 to exercise
        edge cases (matches a serving step where some seqs finished).
      - ctx_len:    random in [64, 8192], occasionally up to 16384.
      - qlen:       in {1, 2, 3, 4} (eagle3 draft / MTP draft tokens).
    """
    # batch size mix: biased toward 128 (matches CONCURRENCY=128)
    batch_size = rng.choice([32, 64, 96, 128, 128, 128, 128, 128, 192, 256])

    # ctx_len mix: most short-to-medium, sometimes long
    if rng.random() < 0.08:
        ctx_len = rng.randint(8192, 16384)
    elif rng.random() < 0.4:
        ctx_len = rng.randint(64, 1024)
    else:
        ctx_len = rng.randint(1024, 8192)
    ctx_len = max(ctx_len, block_size)

    # MTP qlen distribution: weight toward 3-4 since eagle3 emits 3 draft tokens
    qlen = rng.choice([1, 2, 3, 3, 3, 4, 4])

    max_num_blocks_per_seq = (16384 + block_size - 1) // block_size
    num_blocks_per_seq = (ctx_len + block_size - 1) // block_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_lens_qo = torch.full((batch_size,), qlen, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_qo = int(qo_indptr[-1].item())
    max_qlen = qlen

    query = torch.empty(
        (total_qo, num_q_heads, head_size), dtype=dtype, device=device
    ).uniform_(-1, 1)

    seq_lens = torch.full(
        (batch_size,), ctx_len, dtype=torch.int32, device=device
    )

    # block_tables: random page assignments per request, padded to
    # max_num_blocks_per_seq.
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        idx = torch.randint(
            0, num_blocks, (num_blocks_per_seq,), dtype=torch.int32, device=device
        )
        block_tables[i, :num_blocks_per_seq] = idx

    return dict(
        query=query,
        block_tables=block_tables,
        seq_lens=seq_lens,
        qo_indptr=qo_indptr,
        max_qlen=max_qlen,
        batch_size=batch_size,
        ctx_len=ctx_len,
        qlen=qlen,
        total_qo=total_qo,
    )


def run_one(
    iter_idx: int,
    rng: random.Random,
    cfg,
    persistent,
    stream: torch.cuda.Stream,
    log_shape: bool = False,
) -> dict:
    inp = build_iter_inputs(
        rng,
        cfg.num_kv_heads,
        cfg.num_q_heads,
        cfg.head_size,
        cfg.block_size,
        cfg.num_blocks,
        cfg.compute_dtype,
    )
    if log_shape:
        print(f"[repro]   iter={iter_idx:>6}  bs={inp['batch_size']:>4}  "
              f"ctx={inp['ctx_len']:>5}  qlen={inp['qlen']}  "
              f"total_qo={inp['total_qo']:>5}  -> calling pa_fwd_asm", flush=True)

    if cfg.kv_dtype == "fp8":
        k_cache = persistent["k_quant"]
        v_cache_asm = persistent["v_quant_asm"]
        k_scale = persistent["k_scale_asm"]
        v_scale = persistent["v_scale_asm"]
    else:
        k_cache = persistent["k_cache"]
        v_cache_asm = persistent["v_cache_asm"]
        k_scale = None
        v_scale = None

    with torch.cuda.stream(stream):
        out = aiter.pa_fwd_asm(
            inp["query"],
            k_cache,
            v_cache_asm,
            inp["block_tables"],
            inp["seq_lens"],
            inp["block_tables"].stride(0),
            max_qlen=inp["max_qlen"],
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=None,
            qo_indptr=inp["qo_indptr"],
            high_precision=0,
        )

    # IMPORTANT: keep refs to *all* input tensors so the caching allocator
    # cannot recycle their backing memory while the async kernel is still
    # running. pa_fwd_asm uses raw HIP launches and may not properly mark
    # the allocator's stream-use tracking on the input blocks.
    rec = dict(
        iter=iter_idx,
        batch_size=inp["batch_size"],
        ctx_len=inp["ctx_len"],
        qlen=inp["qlen"],
        total_qo=inp["total_qo"],
        out=out,
        _query=inp["query"],
        _block_tables=inp["block_tables"],
        _seq_lens=inp["seq_lens"],
        _qo_indptr=inp["qo_indptr"],
    )
    return rec


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="ASM paged-attention crash repro")
    p.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16",
                   help="KV cache dtype (both observed to crash in production)")
    p.add_argument("--num-q-heads", type=int, default=8,
                   help="Q heads (production: 8/rank with TP=8)")
    p.add_argument("--num-kv-heads", type=int, default=1,
                   help="KV heads (production: 1 after MLA absorb -> GQA=8)")
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=8192,
                   help="Size of KV cache pool")
    p.add_argument("--max-iters", type=int, default=100000)
    p.add_argument("--streams", type=int, default=8,
                   help="Number of concurrent CUDA streams")
    p.add_argument("--sync-every", type=int, default=64,
                   help="Force device sync + check error every N iters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-each-call", action="store_true",
                   help="Print every call's shape (for bisection)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_default_device(args.device)

    class Cfg:
        pass
    cfg = Cfg()
    cfg.num_q_heads = args.num_q_heads
    cfg.num_kv_heads = args.num_kv_heads
    cfg.head_size = args.head_size
    cfg.block_size = args.block_size
    cfg.num_blocks = args.num_blocks
    cfg.kv_dtype = args.kv_dtype
    cfg.compute_dtype = torch.bfloat16  # query/output dtype

    print(f"[repro] device={args.device}")
    print(f"[repro] kv_dtype={cfg.kv_dtype}  num_q_heads={cfg.num_q_heads}  "
          f"num_kv_heads={cfg.num_kv_heads}  GQA={cfg.num_q_heads // cfg.num_kv_heads}")
    print(f"[repro] head_size={cfg.head_size}  block_size={cfg.block_size}  "
          f"num_blocks={cfg.num_blocks}")
    print(f"[repro] max_iters={args.max_iters}  streams={args.streams}  "
          f"sync_every={args.sync_every}")

    # --- allocate persistent KV cache (re-used across iterations, like a
    #     paged KV pool in production)
    k_cache, v_cache = make_kv_cache(
        cfg.num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_size,
        cfg.compute_dtype, device=args.device,
    )
    persistent = {
        "k_cache": k_cache,
        "v_cache_asm": asm_v_shuffle(v_cache),
    }
    if cfg.kv_dtype == "fp8":
        k_q, v_q, k_s_asm, v_s_asm = pertoken_quant_kvcache_symm(
            k_cache, v_cache, quant_dtype=aiter.dtypes.fp8
        )
        persistent["k_quant"] = k_q
        persistent["v_quant_asm"] = asm_v_shuffle(v_q)
        persistent["k_scale_asm"] = k_s_asm
        persistent["v_scale_asm"] = v_s_asm

    torch.cuda.synchronize()
    print(f"[repro] KV cache allocated: K {k_cache.shape} {k_cache.dtype}  "
          f"V {v_cache.shape} {v_cache.dtype}")
    if cfg.kv_dtype == "fp8":
        print(f"[repro]   quant K {persistent['k_quant'].shape} "
              f"{persistent['k_quant'].dtype}  "
              f"K_QScale {persistent['k_scale_asm'].shape} "
              f"{persistent['k_scale_asm'].dtype}")

    streams = [torch.cuda.Stream(device=args.device) for _ in range(args.streams)]
    rng = random.Random(args.seed)

    t0 = time.time()
    last_log_t = t0
    last_log_iter = 0
    keepalive: List[dict] = []  # keep refs so async kernels don't free inputs

    for i in range(args.max_iters):
        s = streams[i % args.streams]
        try:
            rec = run_one(i, rng, cfg, persistent, s, log_shape=args.log_each_call)
            keepalive.append(rec)
            # bound memory: only hold last ~2 batches per stream
            if len(keepalive) > args.streams * 4:
                keepalive.pop(0)
        except Exception as e:
            print(f"\n[repro] !! exception at iter {i} (sync-launch): {e}")
            traceback.print_exc()
            return _report_crash(i, t0, args)

        if (i + 1) % args.sync_every == 0:
            try:
                torch.cuda.synchronize()
            except Exception as e:
                print(f"\n[repro] !! HIP error surfaced at sync after iter {i}: "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
                # dump last batch shapes to help ASM team
                recent = keepalive[-min(len(keepalive), 16):]
                print(f"[repro] last {len(recent)} call shapes:")
                for r in recent:
                    print(f"  iter={r['iter']:>6}  bs={r['batch_size']:>4}  "
                          f"ctx={r['ctx_len']:>5}  qlen={r['qlen']}  "
                          f"total_qo={r['total_qo']:>5}")
                return _report_crash(i, t0, args)

            now = time.time()
            if now - last_log_t >= 5.0:
                d_iter = (i + 1) - last_log_iter
                d_t = now - last_log_t
                ips = d_iter / d_t
                print(f"[repro] iter {i + 1:>7}/{args.max_iters}  "
                      f"{ips:>7.1f} iter/s  elapsed={now - t0:>7.1f}s")
                last_log_t = now
                last_log_iter = i + 1

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\n[repro] DONE — {args.max_iters} iters OK in {dt:.1f}s "
          f"({args.max_iters / dt:.1f} iter/s). No crash observed.")
    return 0


def _report_crash(iter_idx: int, t0: float, args) -> int:
    dt = time.time() - t0
    print(f"\n[repro] CRASHED at iter {iter_idx} after {dt:.1f}s "
          f"(kv_dtype={args.kv_dtype}, streams={args.streams})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
