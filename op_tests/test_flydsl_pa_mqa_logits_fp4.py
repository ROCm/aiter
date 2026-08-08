#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Op-test and benchmark for canonical ``flydsl_pa_mqa_logits_fp4_decode``.
Usage:
    python op_tests/test_flydsl_pa_mqa_logits_fp4.py
    python op_tests/test_flydsl_pa_mqa_logits_fp4.py --batch 8 --ctx 8192 --next-n 1
"""

import argparse
import itertools
import random

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_decode, is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

dev = "cuda"
SEED = 42
SCALE_BLOCK = 32  # fp4 elements per scale block
MFMA_M = 16
KVS_NTPW = 4
DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128
_ITERS = 50
_WARMUP = 10


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── FP4 quant / dequant utilities ─────────────────────────────────────

# FP4 e2m1 representable values (ordered by magnitude)
_FP4_GRID_VALUES = [
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]
# LUT: grid index → fp4 e2m1 4-bit encoding
_E2M1_LUT = [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]
# Inverse LUT: fp4 e2m1 4-bit encoding → grid index
_E2M1_INV_LUT = [7, 8, 9, 10, 11, 12, 13, 14, 7, 6, 5, 4, 3, 2, 1, 0]


def fp4_quant_e2m1_with_e8m0(x: torch.Tensor, block_size: int = SCALE_BLOCK):
    *prefix, d = x.shape
    assert d % block_size == 0
    x_blk = x.float().reshape(*prefix, d // block_size, block_size)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    exp_unbiased = torch.ceil(torch.log2(amax / 6.0))
    exp_biased = (exp_unbiased + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    e8m0 = exp_biased.squeeze(-1).contiguous()
    scale = torch.pow(2.0, exp_biased.float() - 127.0)
    x_scaled = x_blk / scale
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=x.device)
    idx = (x_scaled.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.uint8, device=x.device)
    nibbles = lut[idx].reshape(*prefix, d)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), e8m0


def fp4_dequant_e2m1_with_e8m0(packed, e8m0, block_size=SCALE_BLOCK):
    *prefix, d_half = packed.shape
    d = d_half * 2
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    nibbles = torch.empty(*prefix, d, dtype=torch.uint8, device=packed.device)
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high
    inv = torch.tensor(_E2M1_INV_LUT, dtype=torch.long, device=packed.device)
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=packed.device)
    vals = grid[inv[nibbles.long()]]
    scale = torch.pow(2.0, e8m0.float() - 127.0)
    return (
        vals.reshape(*prefix, d // block_size, block_size) * scale.unsqueeze(-1)
    ).reshape(*prefix, d)


# ── Preshuffle layout helpers (kernel ABI) ────────────────────────────


def create_paged_preshuffle_kv_fp4(kv_bf16, kv_block_size, num_blocks, block_tables):
    """Create paged preshuffle FP4 E2M1 KV cache from dense bf16 KV.

    Supports head_dim as any multiple of 128 — splits K dim into k_tiles
    outer × 4 inner K_chunks (each = 32 K elements / 16 packed bytes).

    Returns:
        kv_cache: [num_blocks, K_TILES, 4 (K_chunks), kv_block_size, 16] uint8
        kv_scale: [num_blocks, K_TILES, 4 (K_chunks), kv_block_size] uint8
        kv_fp4:   [B, T, D/2] uint8 (for reference dequant)
        kv_e8m0:  [B, T, D/32] uint8 (for reference dequant)
    """
    batch, t_max, d = kv_bf16.shape
    assert d % 128 == 0, f"head_dim must be multiple of 128, got {d}"
    assert t_max % kv_block_size == 0
    t_blocks = t_max // kv_block_size
    k_tiles = d // 128
    d_packed = d // 2
    d_scales = d // 32

    kv_flat = kv_bf16.reshape(-1, d)
    kv_fp4, kv_e8m0 = fp4_quant_e2m1_with_e8m0(kv_flat, block_size=SCALE_BLOCK)
    kv_fp4 = kv_fp4.reshape(batch, t_max, d_packed)
    kv_e8m0 = kv_e8m0.reshape(batch, t_max, d_scales)

    # FP4 (cbsz=4) per-thread K layout is CONTIGUOUS: 16 bytes of one K_chunk
    # = 32 K elements at K[k*32..k*32+31]. For head_dim > 128, K splits into
    # k_tiles outer × 4 inner K_chunks. Preshuffle: split K into (k_tiles,
    # 4 K_chunks, 16 bytes), then permute K-axes ahead of token within a page.
    kv_chunks_perm = (
        kv_fp4.view(batch, t_blocks, kv_block_size, k_tiles, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(batch * t_blocks, k_tiles, 4, kv_block_size, 16)
    )
    # KVS_NTPW: nt-bytes packed together for the kernel's packed dword load
    # (4 ubyte → 1 dword). Per (D=lane_div_16, T=lane_mod_16), bytes for nts
    # 0..KVS_NTPW-1 are adjacent so one thread dword-loads all 4 nts.
    assert kv_block_size % KVS_NTPW == 0
    kv_e8m0_perm = (
        kv_e8m0.view(batch, t_blocks, kv_block_size, k_tiles, 4)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(batch * t_blocks, k_tiles, 4, kv_block_size)
        # Interleave 4 nts per token group: split [kv_block_size] into
        # (NTPW=4, T_per_nt), transpose to (T, NTPW) so 4 consecutive bytes
        # per T cover nts 0..3 → 1 dword load.
        .view(batch * t_blocks, k_tiles, 4, KVS_NTPW, kv_block_size // KVS_NTPW)
        .transpose(-1, -2)
        .contiguous()
        .view(batch * t_blocks, k_tiles, 4, kv_block_size)
    )

    phys_flat = block_tables.reshape(-1).long()
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    kv_cache[phys_flat] = kv_chunks_perm
    kv_scale[phys_flat] = kv_e8m0_perm

    return kv_cache, kv_scale, kv_fp4, kv_e8m0


# ── Reference implementation ─────────────────────────────────────────


def ref_mqa_logits_mixed(
    q_packed,
    q_scale,
    kv_fp4,
    kv_scale,
    weights,
    context_lens,
    next_n=1,
    weight_scale=1.0,
):
    """Reference: Q (FP4) + KV (FP4) dequant → einsum → relu → weight → sum.

    Shapes:
      q_packed: [B, NEXT_N, H, D/2] uint8
      q_scale:  [B, NEXT_N, H, D/32] uint8
      kv_fp4:   [B, T, D/2] uint8
      kv_scale: [B, T, D/32] uint8
      weights:  [B*NEXT_N, H] fp32
      output:   [B*NEXT_N, T_max] fp32
    For NEXT_N>1 each row n has causal limit k <= context_len - NEXT_N + n.
    """
    batch = q_packed.shape[0]
    t_max = kv_fp4.shape[1]
    heads = q_packed.shape[2]
    head_dim_packed = q_packed.shape[3]
    head_dim_scales = q_scale.shape[3]
    head_dim_local = head_dim_packed * 2
    q_dq = fp4_dequant_e2m1_with_e8m0(
        q_packed.reshape(batch * next_n, heads, head_dim_packed),
        q_scale.reshape(batch * next_n, heads, head_dim_scales),
    ).reshape(batch, next_n, heads, head_dim_local)
    kv_dq = fp4_dequant_e2m1_with_e8m0(kv_fp4, kv_scale)  # [B, T, D] float32

    ref_logits = torch.full(
        (batch * next_n, t_max), float("-inf"), device=dev, dtype=torch.float32
    )
    for b in range(batch):
        ctx = context_lens[b].item()
        if ctx == 0:
            continue
        kvi = kv_dq[b, :ctx]  # [ctx, D]
        for n in range(next_n):
            qi = q_dq[b, n]  # [H, D]
            wi = weights[b * next_n + n].float()  # [H]
            qk = qi @ kvi.T  # [H, ctx]
            qk = torch.relu(qk) * wi[:, None]
            logits_i = qk.sum(dim=0) * weight_scale  # [ctx]
            valid_max = ctx - next_n + n
            if valid_max + 1 < ctx:
                logits_i[valid_max + 1 :] = float("-inf")
            ref_logits[b * next_n + n, :ctx] = logits_i
    return ref_logits


def _make_varctx(batch, max_ctx, kv_block_size, var_ratio=0.5, seed=0):
    """Per-batch ctx lengths matching aiter bench_deepgemm_attention.py.

    max_ctx == max_model_len, so avg = max_ctx // 2; lengths rounded up to
    kv_block_size for paged-KV correctness.
    """
    avg = max_ctx // 2
    low = int((1 - var_ratio) * avg)
    high = int((1 + var_ratio) * avg)
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(low, high + 1, (batch,), generator=g).tolist()
    return [
        min(((c + kv_block_size - 1) // kv_block_size) * kv_block_size, max_ctx)
        for c in raw
    ]


# ── Gluon FP8 baseline (E2E decode calling convention) ───────────────


def _bench_gluon_fp8(
    q_bf16,  # [B, NEXT_N, H, D] bf16
    kv_bf16,  # [B, t_max, D] bf16
    weights,  # [B*NEXT_N, H]
    context_lens,  # [B] int32
    block_tables,  # [B, max_blocks_per_seq] int32 (kv_block_size blocks)
    t_max,
    num_blocks,
    kv_block_size,
    heads,
    head_dim,
    next_n,
    ref_logits,
    mask,
    num_iters,
    num_warmup,
):
    """Time the gluon FP8 decode indexer via `deepgemm_fp8_paged_mqa_logits`
    using EXACTLY the calling convention ATOM's `Indexer._score_topk_decode`
    uses in end-to-end serving:

        Preshuffle=True, KVBlockSize=kv_block_size(64), ChunkK=256,
        WavePerEU=2, and NO VarCtxSchedule (non-varctx grid).

    Built from the same dense KV as the flydsl fp4 path so the two numbers
    are directly comparable. Returns ``(us_fp8, cos_fp8)`` or ``(None, None)``
    if the gluon path is unavailable.
    """
    try:
        from aiter.ops.shuffle import shuffle_weight
        from aiter.ops.triton.attention.pa_mqa_logits import (
            deepgemm_fp8_paged_mqa_logits,
        )
        from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

        fp8_dtype = get_fp8_e4m3_dtype()
        batch_size = q_bf16.shape[0]

        # block_tables is a contiguous arange, so batch b's 64-token block j is
        # physical block b*max_blocks_per_seq + j == the flattened
        # [num_blocks, kv_block_size, D] index -> reshape dense KV directly.
        kv_blocks = kv_bf16.reshape(num_blocks, kv_block_size, 1, head_dim)
        x_amax = kv_blocks.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
        sf = x_amax / 240.0
        x_scaled = (kv_blocks * (1.0 / sf)).to(fp8_dtype)

        # deepgemm layout: [num_blocks, block, 1, D + 4B fp32 scale]. D=128 so
        # block*D is 16B-aligned -> no extra padding needed.
        index_dim = head_dim + 4
        kv_cache_fp8 = torch.empty(
            (num_blocks, kv_block_size * index_dim), dtype=torch.uint8, device=dev
        )
        kv_cache_fp8[:, : kv_block_size * head_dim] = x_scaled.reshape(
            num_blocks, kv_block_size * head_dim
        ).view(torch.uint8)
        kv_cache_fp8[:, kv_block_size * head_dim :] = sf.reshape(
            num_blocks, kv_block_size
        ).view(torch.uint8)
        kv_cache_fp8 = kv_cache_fp8.view(num_blocks, kv_block_size, 1, index_dim)

        # Preshuffle the fp8 data section (E2E Preshuffle=True); scale tail is
        # left in place.
        split = kv_cache_fp8.view(num_blocks, kv_block_size * index_dim)
        data = shuffle_weight(
            split[:, : kv_block_size * head_dim]
            .contiguous()
            .view(num_blocks, kv_block_size, head_dim)
        )
        split[:, : kv_block_size * head_dim] = data.reshape(
            num_blocks, kv_block_size * head_dim
        )

        q_fp8 = q_bf16.to(fp8_dtype).contiguous()
        w_fp32 = weights.float().contiguous()
        out_fp8 = torch.full(
            (batch_size * next_n, t_max),
            float("-inf"),
            dtype=torch.float32,
            device=dev,
        )

        def launch_fp8():
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                w_fp32,
                out_fp8,
                context_lens,
                block_tables,
                t_max,
                ChunkK=256,
                Preshuffle=True,
                KVBlockSize=kv_block_size,
                WavePerEU=2,
            )

        out_fp8.fill_(float("-inf"))
        launch_fp8()
        torch.cuda.synchronize()
        # cosine vs the same masked ref (scale-invariant, so the separate
        # weight_scale on the flydsl path does not matter here).
        vo = out_fp8[mask].double()
        vr = ref_logits[mask].double()
        cos_fp8 = (vo * vr).sum() / (vo.norm() * vr.norm() + 1e-12)

        _, us_fp8 = run_perftest(launch_fp8, num_iters=num_iters, num_warmup=num_warmup)
        return us_fp8, float(cos_fp8.item())
    except Exception as e:  # noqa: BLE001
        print(f"  [perf] gluon fp8 path unavailable ({type(e).__name__}: {e})")
        return None, None


# ── Test + Benchmark ─────────────────────────────────────────────────


def run_fixed_decode_case(
    batch,
    max_ctx,
    kv_block_size=64,
    block_k=256,
    next_n=1,
    heads=DEFAULT_HEADS,
    num_iters=20,
    num_warmup=3,
    num_warps=4,
    parallel_unit_num=None,
    head_dim=DEFAULT_HEAD_DIM,
    bench=True,
):
    """End-to-end varctx test for the Q FP4 / KV FP4 decode kernel.

    `heads` (default 64): multiple of MFMA_M=16, <= 128.
    `head_dim` (default 128): multiple of MFMA K=128.
    """
    setup_seed(SEED)
    batch_size = batch
    assert heads % 16 == 0 and heads <= 128, f"heads={heads}: multiple of 16, <= 128"
    assert head_dim % 128 == 0, f"head_dim={head_dim}: multiple of 128"
    m_tiles = heads // 16
    k_tiles = head_dim // 128
    head_dim_packed = head_dim // 2
    head_dim_scales = head_dim // 32

    # Per-batch context lengths (varctx).
    ctx_list = _make_varctx(batch_size, max_ctx, kv_block_size)
    context_lens = torch.tensor(ctx_list, dtype=torch.int32, device=dev)
    total_tokens = int(context_lens.sum().item())

    print("=" * 96)
    print(
        f"MQA Logits (Q FP4, KV FP4) varctx: batch={batch_size}, heads={heads}, "
        f"head_dim={head_dim}, max_ctx={max_ctx}, kv_block={kv_block_size}, "
        f"block_k={block_k}, next_n={next_n}"
    )
    print(
        f"  ctx_lens = {ctx_list}  (sum={total_tokens}, "
        f"avg={total_tokens // batch_size}, util={total_tokens / (batch_size * max_ctx):.1%})"
    )
    naive_ctas = batch_size * next_n * ((max_ctx + block_k - 1) // block_k)
    print("=" * 96)

    # The kernel reads KV in block_k-sized chunks, so it touches tokens up to
    # ceil(ctx/block_k)*block_k even when ctx is only kv_block_size-aligned.
    # Size block_tables/kv_cache to the block_k boundary, otherwise the chunk
    # tail reads past block_tables -> garbage phys block -> illegal access.
    max_blocks_per_seq = max(
        (max_ctx + block_k - 1) // block_k * (block_k // kv_block_size),
        block_k // kv_block_size,
    )
    num_blocks = max_blocks_per_seq * batch_size
    t_max = max_blocks_per_seq * kv_block_size

    # ---- Generate data ----
    q_bf16 = torch.randn(
        batch_size, next_n, heads, head_dim, dtype=torch.bfloat16, device=dev
    )
    kv_bf16 = torch.randn(batch_size, t_max, head_dim, dtype=torch.bfloat16, device=dev)
    weights = (
        torch.randn(batch_size * next_n, heads, dtype=torch.float32, device=dev) * 0.1
    ).to(torch.bfloat16)
    weight_scale = 1.5

    q_packed, q_e8m0 = fp4_quant_e2m1_with_e8m0(
        q_bf16.reshape(batch_size * next_n * heads, head_dim), block_size=SCALE_BLOCK
    )
    q_packed = q_packed.reshape(batch_size, next_n, heads, head_dim_packed)
    q_e8m0 = q_e8m0.reshape(batch_size, next_n, heads, head_dim_scales)

    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        batch_size, max_blocks_per_seq
    )
    kv_cache, kv_scale, kv_fp4_dense, kv_e8m0_dense = create_paged_preshuffle_kv_fp4(
        kv_bf16, kv_block_size, num_blocks, block_tables
    )

    # ---- Reference (Q FP4 + KV FP4 dequant + matmul) — per-batch ctx_lens ----
    ref_logits = ref_mqa_logits_mixed(
        q_packed,
        q_e8m0,
        kv_fp4_dense,
        kv_e8m0_dense,
        weights,
        context_lens,
        next_n=next_n,
        weight_scale=weight_scale,
    )

    # ── Pre-shuffle Q scales for kernel layout (avoids runtime v_bfe_u32) ──
    # [B, NEXT_N, H, K_TILES*4] → [B, NEXT_N, K_TILES, 4, 16, qs_pad], H
    # decomposed as (m_tiles, 16); inner mi_idx padded to qs_pad = ⌈m_tiles/4⌉×4.
    qs_pad = ((m_tiles + 3) // 4) * 4
    qe_real = (
        q_e8m0.view(torch.uint8)
        .reshape(batch_size, next_n, m_tiles, 16, k_tiles, 4)
        .permute(0, 1, 4, 5, 3, 2)
        .contiguous()
    )  # [B, NN, K_TILES, 4, 16, m_tiles]
    qe = torch.nn.functional.pad(qe_real, (0, qs_pad - m_tiles)).contiguous()
    q_decode = q_packed.reshape(
        batch_size * next_n, heads, head_dim_packed
    ).contiguous()
    q_scale_decode = qe.reshape(
        batch_size * next_n, k_tiles, 4, 16, qs_pad
    ).contiguous()

    target_ctas = 1024 if parallel_unit_num is None else parallel_unit_num
    max_chunks = max(1, (t_max + block_k - 1) // block_k)
    split_kv = min(
        max_chunks,
        max(1, (target_ctas + batch_size * next_n - 1) // (batch_size * next_n)),
    )
    print(
        f"  bounded split: target_ctas={target_ctas} num_warps={num_warps} "
        f"split_kv={split_kv} total_ctas={batch_size * next_n * split_kv} "
        f"(one-chunk grid would be {naive_ctas})"
    )

    out_logits = torch.full(
        (batch_size * next_n, t_max), float("-inf"), dtype=torch.float32, device=dev
    )

    def launch_flydsl():
        flydsl_pa_mqa_logits_fp4_decode(
            q_decode,
            q_scale_decode,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            context_lens,
            t_max,
            next_n_max=next_n,
            cu_seq_q=None,
            weight_scale=weight_scale,
            block_k=block_k,
            kv_block_size=kv_block_size,
            num_warps=num_warps,
            parallel_unit_num=parallel_unit_num,
            out=out_logits,
        )
        return out_logits

    # ---- Correctness: one launch + cosine_sim ----
    out_logits.fill_(float("-inf"))
    launch_flydsl()
    torch.cuda.synchronize()

    # Mask = positions where ref is NOT -inf (valid logit). Works for both
    # next_n=1 (full ctx valid) and next_n>1 (per-row causal cut tail).
    mask = ~torch.isneginf(ref_logits)
    valid_out = out_logits[mask].double()
    valid_ref = ref_logits[mask].double()
    cos = (valid_out * valid_ref).sum() / (valid_out.norm() * valid_ref.norm() + 1e-12)
    max_abs_err = (valid_out - valid_ref).abs().max().item()
    mean_abs_err = (valid_out - valid_ref).abs().mean().item()
    err_ratio = checkAllclose(
        valid_ref.float(),
        valid_out.float(),
        rtol=0.05,
        atol=0.05,
        msg="flydsl-qfp4-kvfp4 vs ref",
        printLog=False,
    )
    out_past_ctx = out_logits.masked_select(~mask)
    neg_inf_ok = (
        bool(torch.isneginf(out_past_ctx).all().item())
        if out_past_ctx.numel()
        else True
    )
    print(
        f"  correctness: cosine_sim={cos.item():.6f}  "
        f"max_abs_err={max_abs_err:.6f}  mean_abs_err={mean_abs_err:.6f}  "
        f"err_ratio={err_ratio:.4f}  past_ctx_neginf={neg_inf_ok}"
    )
    cos_val = cos.item()
    assert cos_val > 0.99, f"FlyDSL qfp4/kvfp4 vs ref cosine_sim={cos_val:.4f} < 0.99"
    assert neg_inf_ok, "OOB tokens were not NEG_INF — early-exit / pre-init broken"

    if not bench:
        return

    # Useful work uses real per-batch context lengths rather than max_ctx.
    flops = total_tokens * next_n * heads * (2 * head_dim + 3)
    bytes_q = batch_size * next_n * heads * (head_dim_packed + head_dim_scales)
    bytes_kv = total_tokens * (head_dim_packed + head_dim_scales)
    bytes_w = batch_size * next_n * heads * weights.element_size()
    bytes_bt = batch_size * max_blocks_per_seq * 4
    bytes_out = total_tokens * next_n * 4
    bytes_total = bytes_q + bytes_kv + bytes_w + bytes_bt + bytes_out

    candidates = {"bounded": launch_flydsl}
    ret = {"gfx": get_gfx(), "split_kv": split_kv}
    for name, fn in candidates.items():
        candidate_out, us = run_perftest(
            fn,
            num_iters=num_iters,
            num_warmup=num_warmup,
            use_cuda_event=False,
        )
        err = checkAllclose(
            ref_logits[mask].to(torch.float32),
            candidate_out[mask].to(torch.float32),
            rtol=0.05,
            atol=0.05,
            msg=f"{name}: fixed FP4 decode",
            printLog=False,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = bytes_total / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_decode(
    batch,
    max_ctx,
    next_n,
    heads,
    head_dim,
    kv_block_size,
    block_k,
    num_warps,
    target_ctas,
):
    return run_fixed_decode_case(
        batch=batch,
        max_ctx=max_ctx,
        next_n=next_n,
        heads=heads,
        head_dim=head_dim,
        kv_block_size=kv_block_size,
        block_k=block_k,
        num_warps=num_warps,
        parallel_unit_num=target_ctas,
        num_iters=_ITERS,
        num_warmup=_WARMUP,
        bench=True,
    )


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "flydsl_pa_mqa_logits_fp4_decode unsupported on %s; skipping", get_gfx()
        )
        return
    if not is_flydsl_available():
        aiter.logger.warning("flydsl is unavailable; skipping FP4 decode")
        return

    parser = argparse.ArgumentParser(
        description="MQA Logits (Q FP4, KV FP4) decode Test + Benchmark (gfx950)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[3], help="batch sizes"
    )
    parser.add_argument(
        "--ctx", type=int, nargs="*", default=[512], help="max context lengths"
    )
    parser.add_argument(
        "--next-n",
        type=int,
        nargs="*",
        default=[1, 4, 8],
        help="fixed MTP widths",
    )
    parser.add_argument("--heads", type=int, nargs="*", default=[DEFAULT_HEADS])
    parser.add_argument("--head-dim", type=int, nargs="*", default=[DEFAULT_HEAD_DIM])
    parser.add_argument("--kv-block-size", type=int, nargs="*", default=[64])
    parser.add_argument(
        "--block-k",
        type=int,
        nargs="*",
        default=[256],
        help="tokens per chunk",
    )
    parser.add_argument("--num-warps", type=int, nargs="*", default=[4])
    parser.add_argument(
        "--target-ctas",
        type=int,
        nargs="*",
        default=[1024],
        help="bounded-split CTA targets",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    global _ITERS, _WARMUP
    _ITERS, _WARMUP = args.iters, args.warmup
    rows = [
        test_decode(
            batch, ctx, next_n, heads, head_dim, kv_block, block_k, warps, target
        )
        for (
            batch,
            ctx,
            next_n,
            heads,
            head_dim,
            kv_block,
            block_k,
            warps,
            target,
        ) in itertools.product(
            args.batch,
            args.ctx,
            args.next_n,
            args.heads,
            args.head_dim,
            args.kv_block_size,
            args.block_k,
            args.num_warps,
            args.target_ctas,
        )
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl FP4 fixed decode summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
