#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test + benchmark for ``flydsl_pa_mqa_logits_fp4`` (decode / varctx).
Usage:
    python op_tests/test_flydsl_pa_mqa_logits_fp4.py
    python op_tests/test_flydsl_pa_mqa_logits_fp4.py --batch 8 --ctx 131072 --next_n 1
"""

import argparse
import random

import torch

from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4, is_flydsl_available
from aiter.test_common import checkAllclose, run_perftest

dev = "cuda"
SEED = 42
SCALE_BLOCK = 32  # fp4 elements per scale block
MFMA_M = 16
KVS_NTPW = 4
DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128


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


def _torch_ref_step(q_dq_bn, kv_dq, w_bn, next_n=1):
    """logits[bn,t] = sum_h(relu(Q[bn,h,:] · K[b,t,:]) * w[bn,h]).

    q_dq_bn: [B*NEXT_N, H, D], kv_dq: [B, T, D] (broadcast across NEXT_N),
    w_bn:    [B*NEXT_N, H]. Returns [B*NEXT_N, T]. (torch matmul baseline.)
    """
    if next_n != 1:
        b_kv, t_kv, d_kv = kv_dq.shape
        kv_dq = (
            kv_dq.unsqueeze(1)
            .expand(-1, next_n, -1, -1)
            .reshape(b_kv * next_n, t_kv, d_kv)
        )
    qk = torch.bmm(q_dq_bn, kv_dq.transpose(1, 2))  # [B*NEXT_N, H, T_max]
    qk = torch.relu(qk) * w_bn[:, :, None]
    return qk.sum(dim=1)


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


# ── Test + Benchmark ─────────────────────────────────────────────────

_PERF_SUMMARY = []


def test_pa_mqa_logits_fp4_qfp4_kvfp4(
    batch,
    max_ctx,
    kv_block_size=64,
    block_k=256,
    next_n=1,
    heads=DEFAULT_HEADS,
    num_iters=20,
    num_warmup=3,
    num_warps=4,
    parallel_unit_num=512,
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

    max_blocks_per_seq = (max_ctx + kv_block_size - 1) // kv_block_size
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

    # ---- Host schedule (precomputed once so the bench times only the launch) ----
    from aiter.ops.flydsl.kernels.pa_mqa_logits_fp4 import compute_varctx_schedule

    safe, cta_info, total_ctas = compute_varctx_schedule(
        context_lens, block_k, parallel_unit_num, t_max, next_n=next_n
    )
    print(
        f"  schedule: parallel_unit={parallel_unit_num} num_warps={num_warps} "
        f"safe_chunks_per_cta={safe}  total_ctas={total_ctas}  "
        f"(naive grid would be {naive_ctas})"
    )

    out_logits = torch.full(
        (batch_size * next_n, t_max), float("-inf"), dtype=torch.float32, device=dev
    )

    def launch_flydsl():
        flydsl_pa_mqa_logits_fp4(
            q_packed,
            qe,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            context_lens,
            t_max,
            weight_scale=weight_scale,
            next_n=next_n,
            block_k=block_k,
            kv_block_size=kv_block_size,
            num_warps=num_warps,
            parallel_unit_num=parallel_unit_num,
            out=out_logits,
            cta_info=cta_info,
            total_ctas=total_ctas,
        )

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
    assert cos.item() > 0.99, (
        f"FlyDSL qfp4/kvfp4 vs ref cosine_sim={cos.item():.4f} < 0.99"
    )
    assert neg_inf_ok, "OOB tokens were not NEG_INF — early-exit / pre-init broken"

    if not bench:
        return

    # ---- Perf: flydsl ----
    _, us_fly = run_perftest(launch_flydsl, num_iters=num_iters, num_warmup=num_warmup)
    torch.cuda.synchronize()

    # ---- Perf: torch bf16 baseline (dequant excluded — pure matmul + relu/wsum) ----
    q_dq_bf16 = (
        fp4_dequant_e2m1_with_e8m0(
            q_packed.reshape(-1, head_dim_packed),
            q_e8m0.reshape(-1, head_dim_scales),
        )
        .reshape(batch_size * next_n, heads, head_dim)
        .to(torch.bfloat16)
    )
    kv_dq_bf16 = fp4_dequant_e2m1_with_e8m0(kv_fp4_dense, kv_e8m0_dense).to(
        torch.bfloat16
    )
    w_bf16 = weights.to(torch.bfloat16)

    _, us_bf16 = run_perftest(
        _torch_ref_step,
        q_dq_bf16,
        kv_dq_bf16,
        w_bf16,
        next_n,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )

    # ---- USEFUL FLOPs / bytes (varctx — based on real ctx_lens, not max) ----
    flops = total_tokens * next_n * heads * (2 * head_dim + 3)
    bytes_q = batch_size * next_n * heads * (head_dim_packed + head_dim_scales)
    bytes_kv = total_tokens * (head_dim_packed + head_dim_scales)
    bytes_w = batch_size * next_n * heads * 4
    bytes_bt = batch_size * max_blocks_per_seq * 4
    bytes_out = total_tokens * next_n * 4
    bytes_total = bytes_q + bytes_kv + bytes_w + bytes_bt + bytes_out

    def metrics(us):
        if us <= 0:
            return 0.0, 0.0
        sec = us * 1e-6
        return flops / sec / 1e12, bytes_total / sec / 1e9

    tflops_fly, gbps_fly = metrics(us_fly)
    tflops_bf16, _ = metrics(us_bf16)

    print(
        f"\n  {'':>16} | {'us':>10} | {'TFLOPS':>8} | {'GB/s':>8} | {'vs flydsl':>10}"
    )
    print(
        f"  {'flydsl-qfp4/kvfp4':>16} | {us_fly:>10.2f} | {tflops_fly:>8.2f} | {gbps_fly:>8.1f} |"
    )
    print(
        f"  {'torch-bf16':>16} | {us_bf16:>10.2f} | {tflops_bf16:>8.2f} | {'-':>8} | "
        f"{us_bf16 / us_fly:>9.2f}x"
    )
    print()

    _PERF_SUMMARY.append(
        (
            batch_size,
            heads,
            head_dim,
            max_ctx,
            next_n,
            kv_block_size,
            block_k,
            cos.item(),
            us_fly,
            tflops_fly,
            gbps_fly,
        )
    )


def _print_perf_summary():
    print("\n" + "=" * 96)
    print("Perf summary (flydsl-qfp4/kvfp4 across shapes)")
    print("=" * 96)
    print(
        f"  {'batch':>5} | {'heads':>5} | {'h_dim':>5} | {'ctx_len':>7} | {'next_n':>6} | "
        f"{'kv_blk':>6} | {'block_k':>7} | {'cos_sim':>8} | {'us':>9} | {'TFLOPS':>7} | {'GB/s':>7}"
    )
    print("  " + "-" * 103)
    for b, h, hd, ctx, nn, kvb, blk, cos_v, us, tflops, gbps in _PERF_SUMMARY:
        print(
            f"  {b:>5} | {h:>5} | {hd:>5} | {ctx:>7} | {nn:>6} | {kvb:>6} | {blk:>7} | "
            f"{cos_v:>8.4f} | {us:>9.2f} | {tflops:>7.2f} | {gbps:>7.1f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MQA Logits (Q FP4, KV FP4) decode Test + Benchmark (gfx950)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch", type=int, default=0, help="Batch size (0 = run default sweep)"
    )
    parser.add_argument(
        "--ctx", type=int, default=0, help="Context length (0 = run default sweep)"
    )
    parser.add_argument("--kv_block_size", type=int, default=64)
    parser.add_argument(
        "--block_k",
        type=int,
        default=256,
        help="Tokens per chunk (multiple of MFMA_N=16, divisible by num_warps)",
    )
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument(
        "--num_warps",
        type=int,
        default=4,
        help="warps per CTA (pipelined kernel only); BLOCK=num_warps*64",
    )
    parser.add_argument(
        "--parallel_unit_num",
        type=int,
        default=512,
        help="target CTA count for host schedule (default 512)",
    )
    parser.add_argument(
        "--next_n",
        type=int,
        default=1,
        help="MTP queries per batch (1 = standard MQA, 2 = MTP-1)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=DEFAULT_HEADS,
        help=f"Number of Q heads (multiple of 16, <= 128). Default {DEFAULT_HEADS}.",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=DEFAULT_HEAD_DIM,
        help=f"Per-head dim (multiple of 128). Default {DEFAULT_HEAD_DIM}.",
    )
    args = parser.parse_args()

    if not is_flydsl_available():
        print("[skip] flydsl is not available in this environment.")
        return

    print(
        "[test] using pa_mqa_logits_fp4_qfp4_kvfp4 kernel "
        "(Q FP4, KV FP4, MFMA(Q_fp4, KV_fp4))"
    )

    if args.batch > 0 and args.ctx > 0 and args.next_n > 0:
        # (batch, max_ctx, next_n, heads)
        configs = [(args.batch, args.ctx, args.next_n, args.heads)]
    else:
        # Default sweep: correctness + light perf on small/moderate ragged shapes,
        # exercising next_n=1/2 and heads=64/128. Use --batch/--ctx for a big run.
        configs = [
            (2, 512, 1, 64),
            (3, 1024, 1, 64),
            (2, 512, 2, 64),
            (2, 768, 1, 128),
            (4, 2048, 1, 64),
        ]

    for b, c, nn, h in configs:
        try:
            test_pa_mqa_logits_fp4_qfp4_kvfp4(
                batch=b,
                max_ctx=c,
                next_n=nn,
                heads=h,
                kv_block_size=args.kv_block_size,
                block_k=args.block_k,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
                num_warps=args.num_warps,
                parallel_unit_num=args.parallel_unit_num,
                head_dim=args.head_dim,
            )
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
            raise
        except Exception:
            import traceback

            traceback.print_exc()
            raise

    if _PERF_SUMMARY:
        _print_perf_summary()
    print("  PASS")


if __name__ == "__main__":
    main()
