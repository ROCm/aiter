#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test + perf baseline for the MXFP8 paged MQA logits OPUS kernel (gfx950).

Default (no flag): correctness. Validates prefill (ragged windows) and decode (MTP
tail-causal) against a pure-torch reference: vs exact FP8-dequant ref -> kernel
correctness (cos ~ 1.0); vs bf16 ref -> FP8 quant accuracy. Q/KV are MXFP8-quantized
+ preshuffled on the host into the kernel ABI (E4M3 + E8M0 block scales, block=32).

``--baseline``: perf baseline vs Triton (fp8), reproducible (fixed seeds), reporting
per case BOTH kernel time and schedule/prep time so the comparison is fair:
  ours   : prefill schedule = compute_prefill_windows (~us); decode schedule = 0
           (cu_seq_q/context_lens are input metadata). paged KV, no gather.
  triton : prefill prep = paged->contiguous gather; decode = deepgemm reads paged.
Scenarios: prefill batch 1..20 (qlen==ctx>=800, sum==16384); decode batch {1..128}
x max_ctx {1024,8192} x next_n {1,4,8}. Use an IDLE GPU (contention skews small kernels).

Usage:
    python op_tests/test_pa_mqa_logits_mxfp8_opus.py                       # correctness
    python op_tests/test_pa_mqa_logits_mxfp8_opus.py --baseline [--prefill|--decode] [--out base]
"""

import argparse
import json
import random

import torch
import torch.nn.functional as F

from aiter.ops.opus.pa_mqa_logits_mxfp8_opus import (
    compute_prefill_windows,
    pa_mqa_logits_mxfp8_decode,
    pa_mqa_logits_mxfp8_fwd_decode,
    pa_mqa_logits_mxfp8_fwd_prefill,
    pa_mqa_logits_mxfp8_prefill,
)
from aiter.test_common import run_perftest

dev = "cuda"

SCALE_BLOCK = 32
MFMA_M = 16
FP8_E4M3_MAX = 448.0
KVS_NTPW = 4

# ── perf baseline (--baseline) config ────────────────────────────────
HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
PREFILL_BLOCK_K = 64   # 1-warp
DECODE_BLOCK_K = 256   # overridden to 64 (1-warp) by --decode-warp 1 (the default)
TRITON_DECODE_CHUNKK = 256   # triton deepgemm's own KV tiling (independent of our warp variant)
WEIGHT_SCALE = 1.0
PREFILL_TOTAL_QLEN = 16384
PREFILL_QMIN = 800
DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
DECODE_MAX_CTXS = [1024, 8192]
DECODE_NEXT_NS = [1, 4, 8]
# decode CTA policy: context-split (to fill the GPU) only when the query rows alone
# can't reach this many CTAs; else one CTA per row. Overridden to 1024 by --decode-warp 1.
DECODE_CTA_TARGET = 256


# ── MXFP8 (E4M3 + E8M0) quant / dequant ──────────────────────────────


def fp8_quant_e4m3_with_e8m0(x, block_size=SCALE_BLOCK):
    """[..., d] float -> (fp8 E4M3 bytes [..., d] uint8, e8m0 [..., d/block] uint8)."""
    *prefix, d = x.shape
    assert d % block_size == 0
    x_blk = x.float().reshape(*prefix, d // block_size, block_size)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    exp_unbiased = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    exp_biased = (exp_unbiased + 127.0).clamp(0.0, 254.0).to(torch.uint8)
    e8m0 = exp_biased.squeeze(-1).contiguous()
    scale = torch.pow(2.0, exp_biased.float() - 127.0)
    x_scaled = (x_blk / scale).reshape(*prefix, d)
    fp8 = x_scaled.to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8).contiguous()
    return fp8_bytes, e8m0


def fp8_dequant_e4m3_with_e8m0(fp8_bytes, e8m0, block_size=SCALE_BLOCK):
    *prefix, d = fp8_bytes.shape
    vals = fp8_bytes.view(torch.float8_e4m3fn).float()
    scale = torch.pow(2.0, e8m0.float() - 127.0)
    return (
        vals.reshape(*prefix, d // block_size, block_size) * scale.unsqueeze(-1)
    ).reshape(*prefix, d)


# ── Host-side FP8 layout writers (kernel ABI, mirrors logits_host.cc) ──


def quant_q_fp8_preshuffle(q):
    """[T, H, head_dim] -> q_fp8 [T,H,D] uint8, q_scale [T,K_TILES,4,16,QS_PAD] uint8."""
    total_tokens, heads, head_dim = q.shape
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128
    fp8, e8m0 = fp8_quant_e4m3_with_e8m0(q.reshape(total_tokens * heads, head_dim))
    q_fp8 = fp8.reshape(total_tokens, heads, head_dim)
    q_e8m0 = e8m0.reshape(total_tokens, heads, head_dim // 32)
    qs_pad = ((m_tiles + 3) // 4) * 4
    qe = (
        q_e8m0.reshape(total_tokens, m_tiles, 16, k_tiles, 4)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
    )
    return q_fp8, F.pad(qe, (0, qs_pad - m_tiles)).contiguous()


def indexer_k_fp8_paged_preshuffle(k, slot_mapping, kv_cache, kv_scale, kv_block_size):
    """Paged-preshuffle FP8 K writer.

    Per token at (physical block p, block offset o):
      kv_cache[p, kt, c, o, :]  = 16 fp8 bytes for K[(kt*8+c)*16 : +16]  (c in 0..7)
      kv_scale[p, kt, kc, sflat] = e8m0 byte, sflat = (o%16)*4 + (o//16)  (kc in 0..3)
    """
    _num_tokens, head_dim = k.shape
    k_tiles = head_dim // 128
    fp8, e8m0 = fp8_quant_e4m3_with_e8m0(k)
    valid = slot_mapping >= 0
    sm = slot_mapping[valid].long()
    if sm.numel() == 0:
        return kv_cache, kv_scale
    fp8 = fp8[valid].view(-1, k_tiles, 8, 16)
    e8m0 = e8m0[valid].view(-1, k_tiles, 4)
    phys = sm // kv_block_size
    boff = sm % kv_block_size
    kv_cache[phys, :, :, boff, :] = fp8
    sflat = (boff % 16) * KVS_NTPW + (boff // 16)
    kv_scale[phys, :, :, sflat] = e8m0
    return kv_cache, kv_scale


# ── Reference ────────────────────────────────────────────────────────


def ref_prefill_logits(q_in, kv_in, weights, row_to_batch, ls, le, max_seq_len, ws=1.0):
    total_tokens = q_in.shape[0]
    out = torch.full(
        (total_tokens, max_seq_len), float("-inf"), device=dev, dtype=torch.float32
    )
    for r in range(total_tokens):
        b, s, e = int(row_to_batch[r]), int(ls[r]), int(le[r])
        if e <= s:
            continue
        qk = q_in[r].float() @ kv_in[b, s:e].float().T
        qk = torch.relu(qk) * weights[r].float()[:, None]
        out[r, s:e] = qk.sum(dim=0) * ws
    return out


def _cos(a, b):
    a, b = a.double(), b.double()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


# ── Shared setup: build preshuffled KV + Q for a batch of ragged windows ──


def _build_inputs(bs, windows_per_batch, heads, head_dim, kv_block_size, block_k, seed):
    torch.manual_seed(seed)
    max_end = max(
        (w if isinstance(w, int) else w[1]) for ws in windows_per_batch for w in ws
    )
    max_blocks_per_seq = max(
        (max_end + block_k - 1) // block_k * (block_k // kv_block_size),
        block_k // kv_block_size,
    )
    t_max = max_blocks_per_seq * kv_block_size
    max_seq_len = t_max
    num_blocks = max_blocks_per_seq * bs

    kv_bf16 = torch.randn(bs, t_max, head_dim, dtype=torch.bfloat16, device=dev)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        bs, max_blocks_per_seq
    )

    kv_fp8_d, kv_e8_d = fp8_quant_e4m3_with_e8m0(kv_bf16.reshape(-1, head_dim))
    kv_dq = fp8_dequant_e4m3_with_e8m0(
        kv_fp8_d.reshape(bs, t_max, head_dim),
        kv_e8_d.reshape(bs, t_max, head_dim // 32),
    )

    k_flat = kv_bf16.reshape(bs * t_max, head_dim)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // kv_block_size].long()
    slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int32)
    k_tiles = head_dim // 128
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 8, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    indexer_k_fp8_paged_preshuffle(k_flat, slot_mapping, kv_cache, kv_scale, kv_block_size)

    return (
        kv_bf16,
        kv_dq,
        kv_cache,
        kv_scale,
        block_tables,
        t_max,
        max_seq_len,
    )


def _build_q(total_tokens, heads, head_dim, weight_scale):
    q_bf16 = torch.randn(total_tokens, heads, head_dim, dtype=torch.bfloat16, device=dev)
    weights = (
        torch.randn(total_tokens, heads, dtype=torch.float32, device=dev) * 0.1
    ).to(torch.bfloat16)
    q_fp8, q_scale = quant_q_fp8_preshuffle(q_bf16)
    q_e8 = fp8_quant_e4m3_with_e8m0(q_bf16.reshape(total_tokens * heads, head_dim))[
        1
    ].reshape(total_tokens, heads, head_dim // 32)
    q_dq = fp8_dequant_e4m3_with_e8m0(q_fp8, q_e8)
    return q_bf16, q_fp8, q_scale, q_dq, weights


# ── Prefill driver ───────────────────────────────────────────────────


def run_prefill_case(
    bs,
    windows_per_batch,
    heads=64,
    head_dim=128,
    kv_block_size=64,
    block_k=64,  # prefill -> 1-wave variant (block_k=64); varqlen/decode uses 256 (4-wave)
    seed=0,
):
    (
        kv_bf16,
        kv_dq,
        kv_cache,
        kv_scale,
        block_tables,
        t_max,
        max_seq_len,
    ) = _build_inputs(bs, windows_per_batch, heads, head_dim, kv_block_size, block_k, seed)

    rb, ls, le = [], [], []
    for b in range(bs):
        for w in windows_per_batch[b]:
            s, e = (0, w) if isinstance(w, int) else (w[0], w[1])
            rb.append(b)
            ls.append(s)
            le.append(e)
    total_tokens = len(rb)
    row_to_batch = torch.tensor(rb, dtype=torch.int32, device=dev)
    local_starts = torch.tensor(ls, dtype=torch.int32, device=dev)
    local_ends = torch.tensor(le, dtype=torch.int32, device=dev)

    weight_scale = 1.5
    q_bf16, q_fp8, q_scale, q_dq, weights = _build_q(total_tokens, heads, head_dim, weight_scale)

    ref_fp8 = ref_prefill_logits(q_dq, kv_dq, weights, row_to_batch, ls, le, max_seq_len, weight_scale)
    ref_bf16 = ref_prefill_logits(q_bf16, kv_bf16, weights, row_to_batch, ls, le, max_seq_len, weight_scale)

    out = pa_mqa_logits_mxfp8_prefill(
        q_fp8.view(torch.float8_e4m3fn),
        q_scale,
        kv_cache.view(torch.float8_e4m3fn),
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
    )
    torch.cuda.synchronize()

    m = ~torch.isneginf(ref_fp8)
    cos_fp8 = _cos(out[m], ref_fp8[m]).item()
    cos_bf16 = _cos(out[m], ref_bf16[m]).item()
    oob_ok = bool(torch.isneginf(out[~m]).all().item()) if (~m).any() else True
    ok = cos_fp8 > 0.999 and cos_bf16 > 0.99 and oob_ok
    print(
        f"  [prefill] bs={bs} tt={total_tokens} cos_fp8={cos_fp8:.6f} "
        f"cos_bf16={cos_bf16:.6f} oob_neginf={oob_ok} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


# ── Varqlen / MTP driver ─────────────────────────────────────────────


def run_varqlen_case(
    bs,
    qlens,
    context_lens,
    heads=64,
    head_dim=128,
    kv_block_size=64,
    seed=0,
):
    block_k = 64  # decode is 1-warp only; block_k here just sizes the KV cache.
    # per-batch single window [0, ctx] just to size the KV cache large enough.
    windows_per_batch = [[(0, int(context_lens[b]))] for b in range(bs)]
    (
        kv_bf16,
        kv_dq,
        kv_cache,
        kv_scale,
        block_tables,
        t_max,
        max_seq_len,
    ) = _build_inputs(bs, windows_per_batch, heads, head_dim, kv_block_size, block_k, seed)

    cu = [0]
    for q in qlens:
        cu.append(cu[-1] + int(q))
    total_q = cu[-1]
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=dev)

    # reference windows (MTP tail-causal): row r in batch b, n-th token ->
    # [0, ctx_b - (qlen_b - 1 - n)).
    rb, ls, le = [], [], []
    for b in range(bs):
        ql = int(qlens[b])
        for n in range(ql):
            rb.append(b)
            ls.append(0)
            le.append(max(int(context_lens[b]) - (ql - 1 - n), 0))
    row_to_batch = torch.tensor(rb, dtype=torch.int32, device=dev)

    weight_scale = 1.5
    q_bf16, q_fp8, q_scale, q_dq, weights = _build_q(total_q, heads, head_dim, weight_scale)

    ref_fp8 = ref_prefill_logits(q_dq, kv_dq, weights, row_to_batch, ls, le, max_seq_len, weight_scale)
    ref_bf16 = ref_prefill_logits(q_bf16, kv_bf16, weights, row_to_batch, ls, le, max_seq_len, weight_scale)

    next_n_max = max(int(q) for q in qlens)
    out = pa_mqa_logits_mxfp8_decode(
        q_fp8.view(torch.float8_e4m3fn),
        q_scale,
        kv_cache.view(torch.float8_e4m3fn),
        kv_scale,
        block_tables,
        weights,
        ctx,
        max_seq_len,
        next_n_max,
        split_ctx_len=max_seq_len,
        cu_seq_q=cu_seq_q,
        weight_scale=weight_scale,
        kv_block_size=kv_block_size,
    )
    torch.cuda.synchronize()

    m = ~torch.isneginf(ref_fp8)
    cos_fp8 = _cos(out[m], ref_fp8[m]).item()
    cos_bf16 = _cos(out[m], ref_bf16[m]).item()
    ok = cos_fp8 > 0.999 and cos_bf16 > 0.99
    print(
        f"  [varqlen] bs={bs} total_q={total_q} (1-warp) "
        f"cos_fp8={cos_fp8:.6f} cos_bf16={cos_bf16:.6f} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


# =====================================================================
# Perf baseline (--baseline): ours (MXFP8 paged) vs Triton (fp8).
# =====================================================================


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


def check_cos_sampled(out, q_dq, kv_dq, weights, rb, ls, le, ws, n_sample=8, seed=0):
    """Cosine of a random handful of query rows vs a torch FP8-dequant ref (a full
    ref over 16k rows is infeasible). cos ~ 1.0 => bit-exact w.r.t. the FP8 inputs."""
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
    return _cos(a, b).item(), len(a_all)


def bench_ours(
    kv_cache, kv_scale, block_tables, q_fp8, q_scale, weights,
    cu_seq_q, context_lens, total_q, max_seq_len, block_k, iters, warmup,
):
    """Time ours' schedule and kernel separately (schedule-free direct prefill):
    schedule = compute_prefill_windows (device window build); kernel = fwd_prefill
    (1D grid = total_q). Returns (sched_us, kernel_us, n_ctas, out, (rb, ls, le))."""

    def do_sched():
        return compute_prefill_windows(cu_seq_q, context_lens, total_q)

    (rb, ls, le), sched_us = run_perftest(do_sched, num_iters=iters, num_warmup=warmup)
    out = torch.full((total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)

    def do_kernel():
        pa_mqa_logits_mxfp8_fwd_prefill(
            q_fp8, q_scale, kv_cache, kv_scale, block_tables, weights,
            rb, ls, le, out, total_q,
            WEIGHT_SCALE, block_k, KV_BLOCK_SIZE, max_seq_len,
        )

    do_kernel()
    torch.cuda.synchronize()
    _, kernel_us = run_perftest(do_kernel, num_iters=iters, num_warmup=warmup)
    return sched_us, kernel_us, int(total_q), out, (rb, ls, le)


def bench_triton_prefill(kv_bf16, block_tables, q_bf16, weights, ctxs, rb_le, iters, warmup):
    """Triton prefill: prep = paged->contiguous gather + fp8_mqa_logits. Returns
    (gather_us, kernel_us) or (None, None) if unavailable."""
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


def bench_triton_decode(
    kv_bf16, block_tables, q_bf16, weights, context_lens, t_max, num_blocks,
    next_n, iters, warmup,
):
    """Triton decode: deepgemm reads paged KV directly (no host prep). Returns kernel_us or None."""
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


def bench_prefill_case(bs, iters, warmup, check, seed):
    qlens = gen_prefill_qlens(bs, seed=seed)  # per-batch qlen == ctx
    ctxs = list(qlens)
    total_q = sum(qlens)

    block_k = PREFILL_BLOCK_K
    windows_per_batch = [[(0, int(ctxs[b]))] for b in range(bs)]
    kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, t_max, max_seq_len = _build_inputs(
        bs, windows_per_batch, HEADS, HEAD_DIM, KV_BLOCK_SIZE, block_k, seed
    )

    cu = [0]
    for q in qlens:
        cu.append(cu[-1] + q)
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    context_lens = torch.tensor(ctxs, dtype=torch.int32, device=dev)

    q_bf16, q_fp8, q_scale, q_dq, weights = _build_q(total_q, HEADS, HEAD_DIM, WEIGHT_SCALE)
    q_fp8_v = q_fp8.view(torch.float8_e4m3fn)
    kv_cache_v = kv_cache.view(torch.float8_e4m3fn)

    sched_us, kern_us, n_ctas, out, (rb, ls, le) = bench_ours(
        kv_cache_v, kv_scale, block_tables, q_fp8_v, q_scale, weights,
        cu_seq_q, context_lens, total_q, max_seq_len, block_k, iters, warmup,
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


def bench_decode_case(batch, max_ctx, next_n, iters, warmup, check, seed):
    ctxs = gen_decode_ctxs(batch, max_ctx, KV_BLOCK_SIZE, seed=seed)
    total_q = batch * next_n

    block_k = DECODE_BLOCK_K   # 1-warp (64) by default; 4-warp (256) only with --decode-warp 4
    windows_per_batch = [[(0, int(ctxs[b]))] for b in range(batch)]
    kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, t_max, max_seq_len = _build_inputs(
        batch, windows_per_batch, HEADS, HEAD_DIM, KV_BLOCK_SIZE, block_k, seed
    )
    num_blocks = block_tables.numel()

    cu = list(range(0, (batch + 1) * next_n, next_n))
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    context_lens = torch.tensor(ctxs, dtype=torch.int32, device=dev)

    q_bf16, q_fp8, q_scale, q_dq, weights = _build_q(total_q, HEADS, HEAD_DIM, WEIGHT_SCALE)
    q_fp8_v = q_fp8.view(torch.float8_e4m3fn)
    kv_cache_v = kv_cache.view(torch.float8_e4m3fn)

    # context-split only when the query rows alone can't fill the GPU.
    max_chunks = max(1, (max_seq_len + block_k - 1) // block_k)
    if total_q >= DECODE_CTA_TARGET:
        split_kv = 1
    else:
        split_kv = min(max_chunks, (DECODE_CTA_TARGET + total_q - 1) // total_q)

    # 3D-grid schedule-free decode: cu_seq_q + context_lens are INPUT metadata (built
    # once in setup, like triton's context_lens), so sched time is 0.
    sched_us = 0.0
    n_ctas = split_kv * next_n * batch
    out = torch.full((total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)

    def do_kernel():
        pa_mqa_logits_mxfp8_fwd_decode(
            q_fp8_v, q_scale, kv_cache_v, kv_scale, block_tables, weights,
            cu_seq_q, context_lens, out, batch, next_n, split_kv,
            WEIGHT_SCALE, block_k, KV_BLOCK_SIZE, max_seq_len,
        )

    do_kernel()
    torch.cuda.synchronize()
    _, kern_us = run_perftest(do_kernel, num_iters=iters, num_warmup=warmup)
    rb, ls, le = compute_prefill_windows(cu_seq_q, context_lens, total_q)  # for cos check only

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


def _gain_pct(ours_us, tri_us):
    """Our kernel's speed benefit over triton: (tri - ours) / tri * 100 (>0 = ours faster)."""
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
    gain = r["kern_gain_pct"]
    gain_s = f"{gain:+6.1f}%" if gain is not None else "   n/a"
    print(
        f"  {tag:<44} {cos:<11} | ours k={_fmt(r['ours_kernel_us']):>9} s={_fmt(r['ours_sched_us']):>8} "
        f"| tri k={_fmt(r['tri_kernel_us']):>9} prep={_fmt(r['tri_prep_us']):>9} | kern gain(vs tri)={gain_s}"
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


def run_baseline(args):
    # 1-warp decode (product default): block_k=64 and 4x the CTA fill target.
    global DECODE_BLOCK_K, DECODE_CTA_TARGET
    if args.decode_warp == 1:
        DECODE_BLOCK_K = 64
        DECODE_CTA_TARGET = 1024

    run_pf = args.prefill or not args.decode
    run_dc = args.decode or not args.prefill
    check = not args.no_check
    rows = []

    print("=== MQA logits fp8 baseline: ours (mxfp8 paged) vs triton (fp8), gfx950 ===")
    print("    (use an IDLE GPU; kernel/schedule times are run_perftest avg-us)")

    if run_pf:
        print(f"\n-- prefill (total_qlen={PREFILL_TOTAL_QLEN}, qlen==ctx>={PREFILL_QMIN}, causal) --")
        for bs in range(1, 21):
            rows.append(bench_prefill_case(bs, args.iters, args.warmup, check, args.seed + bs))

    if run_dc:
        print(
            f"\n-- decode (fixed MTP; per-batch ctx in [0.9*max, max]; "
            f"{args.decode_warp}-warp block_k={DECODE_BLOCK_K} cta_target={DECODE_CTA_TARGET}) --"
        )
        for max_ctx in DECODE_MAX_CTXS:
            for next_n in DECODE_NEXT_NS:
                for batch in DECODE_BATCHES:
                    rows.append(
                        bench_decode_case(
                            batch, max_ctx, next_n, args.iters, args.warmup, check,
                            args.seed + batch + max_ctx + next_n,
                        )
                    )

    bad = [r for r in rows if r["cos"] is not None and r["cos"] < 0.99]
    if bad:
        print(f"\n  [WARN] {len(bad)} case(s) had cos < 0.99 (correctness regression?)")
    export(rows, args.out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true",
                    help="run the ours-vs-triton perf baseline instead of correctness")
    ap.add_argument("--prefill", action="store_true", help="[baseline] prefill scenario only")
    ap.add_argument("--decode", action="store_true", help="[baseline] decode scenario only")
    ap.add_argument("--iters", type=int, default=50, help="[baseline]")
    ap.add_argument("--warmup", type=int, default=10, help="[baseline]")
    ap.add_argument("--seed", type=int, default=0, help="[baseline]")
    ap.add_argument("--no-check", action="store_true", help="[baseline] skip sampled cosine check")
    ap.add_argument("--out", type=str, default=None, help="[baseline] write <out>.csv/.json")
    ap.add_argument(
        "--decode-warp", type=int, default=1, choices=[1, 4],
        help="[baseline] decode warp variant: 1 -> block_k=64 (default, product) + 4x CTA "
             "fill target; 4 -> block_k=256 (A/B only).",
    )
    args = ap.parse_args()

    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        print(f"[skip] pa_mqa_logits_mxfp8 only supports gfx950 (current: {get_arch()}).")
        return 0

    if args.baseline:
        return run_baseline(args)

    print("=== MXFP8 paged MQA logits (opus, gfx950) ===")
    rc = 0
    # ragged prefill windows (mix zero / non-zero lower bounds, short + long)
    rc |= run_prefill_case(2, [[(0, 50), (0, 120), (0, 200)], [(0, 40), (0, 100)]], seed=0)
    rc |= run_prefill_case(3, [[(0, 30)], [(0, 200)], [(0, 100), (0, 150)]], seed=2)
    rc |= run_prefill_case(2, [[(10, 50), (64, 200)], [(0, 100), (130, 256)]], seed=4)
    rc |= run_prefill_case(2, [[(0, 1500)], [(0, 1200), (300, 1400)]], seed=6)
    rc |= run_prefill_case(2, [[(0, 2048)], [(0, 4096)]], seed=8)
    rc |= run_prefill_case(2, [[(0, 512), (0, 1024), (0, 1536)], [(0, 2000)]], seed=10)
    rc |= run_prefill_case(2, [[(100, 2048), (512, 4096)], [(0, 8192)]], seed=12)
    rc |= run_prefill_case(2, [[(0, 3000), (1000, 5000)], [(0, 6000), (2048, 6144)]], seed=14)
    rc |= run_prefill_case(2, [[(0, 4096)], [(0, 4096)]], seed=16)

    # varqlen / MTP (per-batch query length via qlens, tail-causal; may be ragged).
    # decode is 1-warp only.
    varqlen_cases = [
        (3, [1, 1, 1], [200, 512, 1000], 20),
        (2, [2, 2], [300, 1500], 22),
        (4, [4, 4, 4, 4], [256, 800, 2048, 4096], 24),
        (3, [1, 3, 2], [200, 1500, 800], 26),  # fully ragged (no single next_n)
        (3, [2, 0, 3], [384, 256, 640], 28),  # empty batch (qlen=0) mixed in
    ]
    print("  -- varqlen (1-warp) --")
    for bs, qlens, ctx, seed in varqlen_cases:
        rc |= run_varqlen_case(bs, qlens, ctx, seed=seed)

    # ── prefill corner cases ──
    print("  -- prefill corner cases --")
    rc |= run_prefill_case(1, [[(0, 300)]], seed=30)                         # single row
    rc |= run_prefill_case(1, [[(0, 256), (0, 512), (0, 257), (0, 255)]], seed=32)  # exact/±1 block_k
    # Non-4-aligned local_start (guards the out-store alignment fix -- see
    # gcnasm/opus_logits/KNOWN_ISSUE_out_store_alignment.md): windows whose start / end land mid-tile
    # and span a block_k boundary must be written fully (no leading-token drop, no below-window leak).
    rc |= run_prefill_case(2, [[(0, 1), (17, 33)], [(63, 65), (255, 257)]], seed=34)  # 1-token + unaligned small/spanning
    # Exhaustive start sweep: every local_start in [0, 130) with a fixed 40-wide window. run_prefill_case
    # already fails on any dropped in-window cell (cos_fp8 -> nan) or any leaked below-window cell (oob_neginf).
    rc |= run_prefill_case(1, [[(s, s + 40) for s in range(0, 130)]], seed=52)
    rc |= run_prefill_case(2, [[(0, 4096)], [(0, 4096)]], seed=36)  # long rows, one CTA each
    rc |= run_prefill_case(8, [[(0, 300)] for _ in range(8)], seed=38)       # many batches, 1 row each
    rc |= run_prefill_case(2, [[(0, 0), (0, 200)], [(100, 100), (0, 128)]], seed=40)  # zero-length windows mixed
    rc |= run_prefill_case(1, [[(0, 8192)]], seed=42)                        # single long row (32 tiles)

    # ── varqlen corner cases ──
    varqlen_corner_cases = [
        (2, [8, 3], [4, 500], 44),      # qlen > ctx -> some rows tail-clamped empty
        (2, [8, 8], [1024, 2048], 46),  # next_n=8 uniform
        (1, [1], [1000], 48),           # single batch, single decode token
        (4, [16, 8, 24, 4], [4096, 2048, 4096, 1024], 50),  # larger / mixed
    ]
    print("  -- varqlen corner cases (1-warp) --")
    for bs, qlens, ctx, seed in varqlen_corner_cases:
        rc |= run_varqlen_case(bs, qlens, ctx, seed=seed)

    print("  ALL PASS" if rc == 0 else "  SOME FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
