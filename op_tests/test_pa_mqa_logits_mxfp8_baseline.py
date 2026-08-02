#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Performance baseline: ours (MXFP8 paged) vs Triton (fp8) MQA logits, gfx950.

Builds a reproducible, detailed performance baseline over the prefill and decode
scenarios so we can analyze / tune the schedule later. For fairness we report,
per case, BOTH the operator (kernel) time and the schedule/prep time for each
implementation:

  ours (mxfp8, paged, single kernel):
    - schedule = compute_varqlen_windows (device windows)
                 + compute_prefill_schedule (persistent-grid cta_info)
    - kernel   = pa_mqa_logits_mxfp8 launch (paged KV read straight from
                 block_tables; no gather)

  triton (fp8):
    - prefill: fp8_mqa_logits needs a CONTIGUOUS K, so its "prep" is the
               paged -> contiguous gather (cp_gather_indexer_k_quant_cache).
               kernel = fp8_mqa_logits.
    - decode:  deepgemm_fp8_paged_mqa_logits reads paged KV directly (no
               gather, static grid) -> no separate host schedule step.

Scenarios (one case per config, fixed seeds):
  prefill: batch in [1..20]; per batch qlen == ctx, each qlen >= 800,
           sum(qlen) == 16384; standard causal windows [0, n+1).
  decode:  batch in {1,2,4,8,16,32,64,128}; max_ctx in {1024, 8192} with
           per-batch ctx in [0.9*max_ctx, max_ctx] (kv_block_size-aligned);
           next_n (fixed MTP) in {1,4,8}  -> 8*2*3 = 48 combos.

Usage:
    python op_tests/test_pa_mqa_logits_mxfp8_baseline.py                 # both
    python op_tests/test_pa_mqa_logits_mxfp8_baseline.py --prefill
    python op_tests/test_pa_mqa_logits_mxfp8_baseline.py --decode --iters 50
    python op_tests/test_pa_mqa_logits_mxfp8_baseline.py --out baseline  # -> baseline.csv/.json

NOTE: use an IDLE GPU (rocm-smi --showpidgpus). Card contention produces
pathological min<->median swings for the small kernels.
"""

import argparse
import json
import os
import random
import sys

import torch

# make both the op_tests dir (sibling helpers) and the repo root (aiter pkg)
# importable regardless of the invocation cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import test_pa_mqa_logits_mxfp8_opus as M  # noqa: E402

from aiter.ops.opus.pa_mqa_logits_mxfp8_opus import (  # noqa: E402
    compute_prefill_schedule,
    compute_varqlen_windows,
    pa_mqa_logits_mxfp8_fwd_decode,
    pa_mqa_logits_mxfp8_fwd_direct,
    pa_mqa_logits_mxfp8_prefill,
)
from aiter.test_common import run_perftest  # noqa: E402

dev = "cuda"
HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
# block_k selects the warp variant in the launcher: 64 -> 1-warp (prefill),
# 256 -> 4-warp (decode/varqlen). KV cache preshuffle is PAGE=64 either way.
PREFILL_BLOCK_K = 64
DECODE_BLOCK_K = 256
# triton deepgemm's own KV tiling (independent of our warp variant); keep fixed.
TRITON_DECODE_CHUNKK = 256
WEIGHT_SCALE = 1.0

PREFILL_TOTAL_QLEN = 16384
PREFILL_QMIN = 800
DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
DECODE_MAX_CTXS = [1024, 8192]
DECODE_NEXT_NS = [1, 4, 8]
# decode CTA policy: only context-split (to fill the GPU) when the query rows
# alone can't reach this many CTAs; otherwise one CTA per row (no split).
DECODE_CTA_TARGET = 256

# When True, "ours" runs through the schedule-free direct kernel (no cta_info):
# schedule time collapses to just the window build. Toggled by --direct.
USE_DIRECT = False
# decode 3D-grid blockIdx->(split,n,batch) axis mapping (L2-locality sweep); --decode-axis.
# Default 3 (batch=x): best/tied across a min-of-N sweep of the decode workloads.
DECODE_AXIS = 3


# ── workload generation ──────────────────────────────────────────────


def gen_prefill_qlens(bs, total=PREFILL_TOTAL_QLEN, qmin=PREFILL_QMIN, seed=0):
    """Split ``total`` into ``bs`` per-batch qlens, each >= qmin, sum == total."""
    assert bs * qmin <= total, f"bs={bs} * qmin={qmin} > total={total}"
    g = random.Random(seed)
    extra = total - bs * qmin
    w = [g.random() for _ in range(bs)]
    s = sum(w) or 1.0
    parts = [qmin + int(extra * wi / s) for wi in w]
    parts[0] += total - sum(parts)  # absorb rounding into the first (only grows)
    assert min(parts) >= qmin and sum(parts) == total
    return parts


def gen_decode_ctxs(batch, max_ctx, kv_block_size=KV_BLOCK_SIZE, seed=0):
    """Per-batch ctx in [0.9*max_ctx, max_ctx], rounded up to kv_block_size."""
    low = int(0.9 * max_ctx)
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(low, max_ctx + 1, (batch,), generator=g).tolist()
    cap = (max_ctx // kv_block_size) * kv_block_size
    return [min(((c + kv_block_size - 1) // kv_block_size) * kv_block_size, cap) for c in raw]


# ── shared correctness (sampled rows; full ref over 16k rows is infeasible) ──


def check_cos_sampled(out, q_dq, kv_dq, weights, rb, ls, le, ws, n_sample=8, seed=0):
    """Cosine of a random handful of query rows vs a torch FP8-dequant ref.

    Returns (cos, n_checked). cos ~ 1.0 means the kernel matches the reference
    on the sampled rows (the kernel is bit-exact w.r.t. the FP8 inputs)."""
    total = out.shape[0]
    g = random.Random(seed)
    idxs = g.sample(range(total), min(n_sample, total))
    a_all, b_all = [], []
    for r in idxs:
        b, s, e = int(rb[r]), int(ls[r]), int(le[r])
        if e <= s:
            continue
        qk = q_dq[r].float() @ kv_dq[b, s:e].float().T
        qk = torch.relu(qk) * weights[r].float()[:, None]
        a_all.append(out[r, s:e])
        b_all.append(qk.sum(dim=0) * ws)
    if not a_all:
        return 1.0, 0
    a, b = torch.cat(a_all), torch.cat(b_all)
    return M._cos(a, b).item(), len(a_all)


# ── ours: paged single kernel (schedule = windows + cta_info) ────────


def bench_ours(
    kv_cache, kv_scale, block_tables, q_fp8, q_scale, weights,
    cu_seq_q, context_lens, total_q, max_seq_len, pun, block_k, iters, warmup,
    direct=False, split_kv=1,
):
    """Time ours' schedule and kernel separately. ``block_k`` picks the warp
    variant (64=1-warp / 256=4-warp).

    - schedule path (default): schedule = compute_varqlen_windows +
      compute_prefill_schedule (cta_info); kernel = grid(n_ctas) cta_info launch.
    - direct path (``direct=True``): schedule = compute_varqlen_windows only
      (the ~us window build; NO cta_info); kernel = schedule-free launch
      grid = total_q * split_kv.

    Returns (sched_us, kernel_us, n_ctas, out, (rb, ls, le))."""

    if direct:
        def do_sched():
            return compute_varqlen_windows(cu_seq_q, context_lens, total_q)

        (rb, ls, le), sched_us = run_perftest(do_sched, num_iters=iters, num_warmup=warmup)
        n_ctas = total_q * split_kv
        out = torch.full((total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)

        def do_kernel():
            pa_mqa_logits_mxfp8_fwd_direct(
                q_fp8, q_scale, kv_cache, kv_scale, block_tables, weights,
                rb, ls, le, out, total_q, split_kv,
                WEIGHT_SCALE, block_k, KV_BLOCK_SIZE, max_seq_len,
            )

        do_kernel()
        torch.cuda.synchronize()
        _, kernel_us = run_perftest(do_kernel, num_iters=iters, num_warmup=warmup)
        return sched_us, kernel_us, int(n_ctas), out, (rb, ls, le)

    def do_sched():
        rb, ls, le = compute_varqlen_windows(cu_seq_q, context_lens, total_q)
        cta_info, n_ctas = compute_prefill_schedule(
            rb, ls, le, block_k, pun, max_seq_len
        )
        return rb, ls, le, cta_info, n_ctas

    (rb, ls, le, cta_info, n_ctas), sched_us = run_perftest(
        do_sched, num_iters=iters, num_warmup=warmup
    )

    out = torch.full((total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)

    def do_kernel():
        pa_mqa_logits_mxfp8_prefill(
            q_fp8, q_scale, kv_cache, kv_scale, block_tables, weights,
            rb, ls, le, max_seq_len,
            weight_scale=WEIGHT_SCALE, block_k=block_k, kv_block_size=KV_BLOCK_SIZE,
            parallel_unit_num=pun, out=out, cta_info=cta_info, n_ctas=n_ctas,
        )

    do_kernel()
    torch.cuda.synchronize()
    _, kernel_us = run_perftest(do_kernel, num_iters=iters, num_warmup=warmup)
    return sched_us, kernel_us, int(n_ctas), out, (rb, ls, le)


# ── triton prefill: cp_gather (paged->contiguous prep) + fp8_mqa_logits ──


def bench_triton_prefill(
    kv_bf16, block_tables, q_bf16, weights, ctxs, rb_le, iters, warmup,
):
    """Returns (gather_us, kernel_us) or (None, None) if unavailable."""
    try:
        from aiter import cp_gather_indexer_k_quant_cache, dtypes, indexer_k_quant_and_cache
        from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
    except Exception as e:  # noqa: BLE001
        print(f"    [triton prefill unavailable] {type(e).__name__}: {e}")
        return None, None

    bs, t_max, head_dim = kv_bf16.shape
    num_blocks = block_tables.numel()
    k_flat = kv_bf16.reshape(bs * t_max, head_dim)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // KV_BLOCK_SIZE].long()
    slot_mapping = (phys * KV_BLOCK_SIZE + (tt % KV_BLOCK_SIZE)).to(torch.int64)

    kv_cache_fp8 = torch.zeros(
        (num_blocks, KV_BLOCK_SIZE, head_dim + 4), dtype=dtypes.fp8, device=dev
    )
    indexer_k_quant_and_cache(k_flat, kv_cache_fp8, slot_mapping, head_dim, "ue8m0", True)

    # committed context per batch == full seq (qlen == ctx); contiguous prefix.
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(ctxs, dtype=torch.int32, device=dev).cumsum(0)
    total_committed = int(cu[-1].item())
    dst_k = torch.empty((total_committed, head_dim), dtype=dtypes.fp8, device=dev)
    dst_scale = torch.empty((total_committed, 1), dtype=torch.float32, device=dev)

    q_fp8_atom = q_bf16.to(dtypes.fp8)
    rb, le = rb_le  # host lists; causal windows are [0, le)
    rb_t = torch.tensor(rb, dtype=torch.int64, device=dev)
    le_t = torch.tensor(le, dtype=torch.int64, device=dev)
    cu_starts = cu[rb_t].to(torch.int32)
    cu_ends = (cu[rb_t] + le_t.to(torch.int32)).to(torch.int32)
    w_f32 = weights.float()

    def do_gather():
        cp_gather_indexer_k_quant_cache(
            kv_cache_fp8, dst_k, dst_scale.view(dtypes.fp8), block_tables, cu, True
        )

    do_gather()
    torch.cuda.synchronize()
    _, gather_us = run_perftest(do_gather, num_iters=iters, num_warmup=warmup)

    def do_kernel():
        return fp8_mqa_logits(
            q_fp8_atom, dst_k, dst_scale, w_f32, cu_starts, cu_ends, clean_logits=False
        )

    do_kernel()
    torch.cuda.synchronize()
    _, kernel_us = run_perftest(do_kernel, num_iters=iters, num_warmup=warmup)
    return gather_us, kernel_us


# ── triton decode: deepgemm paged (no gather, static grid) ───────────


def bench_triton_decode(
    kv_bf16, block_tables, q_bf16, weights, context_lens, t_max, num_blocks,
    next_n, iters, warmup,
):
    """Returns kernel_us or None. deepgemm reads paged KV directly => no
    separate host schedule / gather (schedule time reported as N/A)."""
    try:
        from aiter.ops.shuffle import shuffle_weight
        from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
        from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
    except Exception as e:  # noqa: BLE001
        print(f"    [triton decode unavailable] {type(e).__name__}: {e}")
        return None

    fp8_dtype = get_fp8_e4m3_dtype()
    batch_size = q_bf16.shape[0]
    head_dim = HEAD_DIM
    kbs = KV_BLOCK_SIZE

    kv_blocks = kv_bf16.reshape(num_blocks, kbs, 1, head_dim)
    sf = kv_blocks.abs().float().amax(dim=3, keepdim=True).clamp(1e-4) / 240.0
    x_scaled = (kv_blocks * (1.0 / sf)).to(fp8_dtype)

    index_dim = head_dim + 4  # deepgemm layout: [nb, block, 1, D + 4B fp32 scale]
    kv_cache_fp8 = torch.empty((num_blocks, kbs * index_dim), dtype=torch.uint8, device=dev)
    kv_cache_fp8[:, : kbs * head_dim] = x_scaled.reshape(num_blocks, kbs * head_dim).view(torch.uint8)
    kv_cache_fp8[:, kbs * head_dim :] = sf.reshape(num_blocks, kbs).view(torch.uint8)
    kv_cache_fp8 = kv_cache_fp8.view(num_blocks, kbs, 1, index_dim)

    split = kv_cache_fp8.view(num_blocks, kbs * index_dim)
    data = shuffle_weight(
        split[:, : kbs * head_dim].contiguous().view(num_blocks, kbs, head_dim)
    )
    split[:, : kbs * head_dim] = data.reshape(num_blocks, kbs * head_dim)

    q_fp8 = q_bf16.to(fp8_dtype).contiguous()  # [B, next_n, H, D]
    w_fp32 = weights.float().contiguous()
    out_fp8 = torch.full((batch_size * next_n, t_max), float("-inf"), dtype=torch.float32, device=dev)

    def launch():
        deepgemm_fp8_paged_mqa_logits(
            q_fp8, kv_cache_fp8, w_fp32, out_fp8, context_lens, block_tables, t_max,
            ChunkK=TRITON_DECODE_CHUNKK, Preshuffle=True, KVBlockSize=kbs, WavePerEU=2,
        )

    try:
        launch()
        torch.cuda.synchronize()
        _, us = run_perftest(launch, num_iters=iters, num_warmup=warmup)
        return us
    except Exception as e:  # noqa: BLE001
        print(f"    [triton decode failed] {type(e).__name__}: {e}")
        return None


# ── case runners ─────────────────────────────────────────────────────


def run_prefill_case(bs, iters, warmup, check, seed):
    qlens = gen_prefill_qlens(bs, seed=seed)  # per-batch qlen == ctx
    ctxs = list(qlens)
    total_q = sum(qlens)

    block_k = PREFILL_BLOCK_K  # 1-warp variant
    windows_per_batch = [[(0, int(ctxs[b]))] for b in range(bs)]
    kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, t_max, max_seq_len = M._build_inputs(
        bs, windows_per_batch, HEADS, HEAD_DIM, KV_BLOCK_SIZE, block_k, seed
    )

    cu = [0]
    for q in qlens:
        cu.append(cu[-1] + q)
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    context_lens = torch.tensor(ctxs, dtype=torch.int32, device=dev)

    q_bf16, q_fp8, q_scale, q_dq, weights = M._build_q(total_q, HEADS, HEAD_DIM, WEIGHT_SCALE)
    q_fp8_v = q_fp8.view(torch.float8_e4m3fn)
    kv_cache_v = kv_cache.view(torch.float8_e4m3fn)

    pun = max(512, total_q)  # prefill policy: one CTA per row (rows fill the GPU)
    sched_us, kern_us, n_ctas, out, (rb, ls, le) = bench_ours(
        kv_cache_v, kv_scale, block_tables, q_fp8_v, q_scale, weights,
        cu_seq_q, context_lens, total_q, max_seq_len, pun, block_k, iters, warmup,
        direct=USE_DIRECT, split_kv=1,  # prefill: one CTA per row (split_kv=1)
    )

    cos, n_chk = (None, 0)
    if check:
        cos, n_chk = check_cos_sampled(out, q_dq, kv_dq, weights, rb, ls, le, WEIGHT_SCALE, seed=seed)

    # triton prefill host causal windows (row n of batch b -> [0, n+1))
    rb_h, le_h = [], []
    for b in range(bs):
        for n in range(qlens[b]):
            rb_h.append(b)
            le_h.append(n + 1)
    tri_gather_us, tri_kern_us = bench_triton_prefill(
        kv_bf16, block_tables, q_bf16, weights, ctxs, (rb_h, le_h), iters, warmup
    )

    row = {
        "scenario": "prefill", "bs": bs, "total_q": total_q, "max_seq_len": max_seq_len,
        "block_k": block_k, "qlen_min": min(qlens), "qlen_max": max(qlens), "n_ctas": n_ctas,
        "cos": cos, "n_checked": n_chk,
        "ours_sched_us": sched_us, "ours_kernel_us": kern_us,
        "ours_total_us": sched_us + kern_us,
        "tri_prep_us": tri_gather_us, "tri_kernel_us": tri_kern_us,
        "tri_total_us": (tri_gather_us + tri_kern_us) if tri_kern_us is not None else None,
        "kern_gain_pct": _gain_pct(kern_us, tri_kern_us),
    }
    _print_row(row)
    del out, kv_bf16, kv_dq, kv_cache, kv_scale, q_bf16, q_fp8, q_dq
    torch.cuda.empty_cache()
    return row


def run_decode_case(batch, max_ctx, next_n, iters, warmup, check, seed):
    ctxs = gen_decode_ctxs(batch, max_ctx, KV_BLOCK_SIZE, seed=seed)
    total_q = batch * next_n

    block_k = DECODE_BLOCK_K  # 4-warp variant
    windows_per_batch = [[(0, int(ctxs[b]))] for b in range(batch)]
    kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, t_max, max_seq_len = M._build_inputs(
        batch, windows_per_batch, HEADS, HEAD_DIM, KV_BLOCK_SIZE, block_k, seed
    )
    num_blocks = block_tables.numel()

    cu = list(range(0, (batch + 1) * next_n, next_n))
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    context_lens = torch.tensor(ctxs, dtype=torch.int32, device=dev)

    q_bf16, q_fp8, q_scale, q_dq, weights = M._build_q(total_q, HEADS, HEAD_DIM, WEIGHT_SCALE)
    q_fp8_v = q_fp8.view(torch.float8_e4m3fn)
    kv_cache_v = kv_cache.view(torch.float8_e4m3fn)

    # decode CTA policy: split context only when the rows alone can't fill the
    # GPU (< DECODE_CTA_TARGET CTAs); else one CTA per row (no split).
    max_chunks = max(1, (max_seq_len + block_k - 1) // block_k)
    if total_q >= DECODE_CTA_TARGET:
        pun = total_q
        split_kv = 1
    else:
        pun = min(total_q * max_chunks, DECODE_CTA_TARGET)
        split_kv = min(max_chunks, (DECODE_CTA_TARGET + total_q - 1) // total_q)

    if USE_DIRECT:
        # 3D-grid schedule-free decode: cu_seq_q + context_lens are INPUT metadata
        # (built once in setup, like triton's context_lens / kv_indices), so there is
        # no per-call schedule step -- sched time is 0. kernel = fwd_decode (3D grid).
        sched_us = 0.0
        n_ctas = split_kv * next_n * batch
        out = torch.full((total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)

        def do_kernel():
            pa_mqa_logits_mxfp8_fwd_decode(
                q_fp8_v, q_scale, kv_cache_v, kv_scale, block_tables, weights,
                cu_seq_q, context_lens, out, batch, next_n, split_kv, DECODE_AXIS,
                WEIGHT_SCALE, block_k, KV_BLOCK_SIZE, max_seq_len,
            )

        do_kernel()
        torch.cuda.synchronize()
        _, kern_us = run_perftest(do_kernel, num_iters=iters, num_warmup=warmup)
        rb, ls, le = compute_varqlen_windows(cu_seq_q, context_lens, total_q)  # for cos check only
    else:
        sched_us, kern_us, n_ctas, out, (rb, ls, le) = bench_ours(
            kv_cache_v, kv_scale, block_tables, q_fp8_v, q_scale, weights,
            cu_seq_q, context_lens, total_q, max_seq_len, pun, block_k, iters, warmup,
            direct=False, split_kv=split_kv,
        )

    cos, n_chk = (None, 0)
    if check:
        cos, n_chk = check_cos_sampled(out, q_dq, kv_dq, weights, rb, ls, le, WEIGHT_SCALE, seed=seed)

    q_bf16_dec = q_bf16.reshape(batch, next_n, HEADS, HEAD_DIM).contiguous()
    tri_kern_us = bench_triton_decode(
        kv_bf16, block_tables, q_bf16_dec, weights, context_lens, t_max, num_blocks,
        next_n, iters, warmup,
    )

    row = {
        "scenario": "decode", "bs": batch, "max_ctx": max_ctx, "next_n": next_n,
        "total_q": total_q, "max_seq_len": max_seq_len, "block_k": block_k,
        "ctx_min": min(ctxs), "ctx_max": max(ctxs), "n_ctas": n_ctas,
        "cos": cos, "n_checked": n_chk,
        "ours_sched_us": sched_us, "ours_kernel_us": kern_us,
        "ours_total_us": sched_us + kern_us,
        "tri_prep_us": None, "tri_kernel_us": tri_kern_us,
        "tri_total_us": tri_kern_us,
        "kern_gain_pct": _gain_pct(kern_us, tri_kern_us),
    }
    _print_row(row)
    del out, kv_bf16, kv_dq, kv_cache, kv_scale, q_bf16, q_fp8, q_dq
    torch.cuda.empty_cache()
    return row


# ── printing / export ────────────────────────────────────────────────


def _gain_pct(ours_us, tri_us):
    """Our kernel's speed benefit over triton, in percent (matches the TODO's
    '快 X%' convention): (tri - ours) / tri * 100. Positive => ours is faster
    (takes that % less time than triton); negative => ours is slower."""
    if not tri_us or ours_us is None:
        return None
    return (tri_us - ours_us) / tri_us * 100.0


def _fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) and x is not None else "n/a"


def _print_row(r):
    if r["scenario"] == "prefill":
        tag = f"prefill bs={r['bs']:>2} q=[{r['qlen_min']},{r['qlen_max']}] msl={r['max_seq_len']}"
    else:
        tag = f"decode  b={r['bs']:>3} mtp={r['next_n']} ctx~{r['max_ctx']}"
    cos = f"cos={r['cos']:.4f}" if r["cos"] is not None else "cos=--"
    ours_k = _fmt(r["ours_kernel_us"])
    ours_s = _fmt(r["ours_sched_us"])
    tri_k = _fmt(r["tri_kernel_us"])
    tri_p = _fmt(r["tri_prep_us"])
    gain = r["kern_gain_pct"]
    gain_s = f"{gain:+6.1f}%" if gain is not None else "   n/a"
    print(
        f"  {tag:<44} {cos:<11} | ours k={ours_k:>9} s={ours_s:>8} "
        f"| tri k={tri_k:>9} prep={tri_p:>9} | kern gain(vs tri)={gain_s}"
    )


def export(rows, out_prefix):
    if not out_prefix:
        return
    import csv

    keys = sorted({k for r in rows for k in r})
    csv_path, json_path = out_prefix + ".csv", out_prefix + ".json"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  wrote {csv_path} and {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", action="store_true", help="run prefill scenario only")
    ap.add_argument("--decode", action="store_true", help="run decode scenario only")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true", help="skip sampled cosine check")
    ap.add_argument("--out", type=str, default=None, help="write <out>.csv/.json")
    ap.add_argument(
        "--decode-warp", type=int, default=4, choices=[1, 4],
        help="decode warp variant: 4 -> block_k=256 (default); 1 -> block_k=64 "
             "with 4x the CTA fill target (1-warp CTAs are 1/4 the size).",
    )
    ap.add_argument(
        "--direct", action="store_true",
        help="route ours through the schedule-free direct kernel (no cta_info); "
             "schedule time collapses to just the window build.",
    )
    ap.add_argument(
        "--decode-axis", type=int, default=3, choices=[0, 1, 2, 3],
        help="decode 3D-grid blockIdx->(split,n,batch) mapping: 0 split=x,n=y,batch=z; "
             "1 n=x,split=y,batch=z; 2 split=x,batch=y,n=z; 3 batch=x,n=y,split=z (default).",
    )
    args = ap.parse_args()

    # 1-warp decode: block_k=64 and 4x the CTA fill target (a 1-warp CTA uses
    # ~1/4 the resources, so ~4x co-resident CTAs are needed to saturate).
    global DECODE_BLOCK_K, DECODE_CTA_TARGET, USE_DIRECT, DECODE_AXIS
    if args.decode_warp == 1:
        DECODE_BLOCK_K = 64
        DECODE_CTA_TARGET = 1024
    USE_DIRECT = args.direct
    DECODE_AXIS = args.decode_axis
    if USE_DIRECT:
        print("[mode] ours = schedule-free DIRECT kernel (no cta_info)")

    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        print(f"[skip] baseline only supports gfx950 (current: {get_arch()}).")
        return 0

    run_pf = args.prefill or not args.decode
    run_dc = args.decode or not args.prefill
    check = not args.no_check
    rows = []

    print("=== MQA logits fp8 baseline: ours (mxfp8 paged) vs triton (fp8), gfx950 ===")
    print("    (use an IDLE GPU; kernel/schedule times are run_perftest avg-us)")

    if run_pf:
        print(f"\n-- prefill (total_qlen={PREFILL_TOTAL_QLEN}, qlen==ctx>={PREFILL_QMIN}, causal) --")
        for bs in range(1, 21):
            rows.append(run_prefill_case(bs, args.iters, args.warmup, check, args.seed + bs))

    if run_dc:
        print(
            f"\n-- decode (fixed MTP; per-batch ctx in [0.9*max, max]; "
            f"{args.decode_warp}-warp block_k={DECODE_BLOCK_K} cta_target={DECODE_CTA_TARGET}) --"
        )
        for max_ctx in DECODE_MAX_CTXS:
            for next_n in DECODE_NEXT_NS:
                for batch in DECODE_BATCHES:
                    rows.append(
                        run_decode_case(
                            batch, max_ctx, next_n, args.iters, args.warmup, check,
                            args.seed + batch + max_ctx + next_n,
                        )
                    )

    bad = [r for r in rows if r["cos"] is not None and r["cos"] < 0.99]
    if bad:
        print(f"\n  [WARN] {len(bad)} case(s) had cos < 0.99 (correctness regression?)")
    export(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
