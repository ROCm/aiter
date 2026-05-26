# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf test for the QK=FP8 / PV=FP4 mixed-precision
VSA block-sparse attention kernel (gfx950).

Run from the aiter repo root, e.g.:

  HIP_VISIBLE_DEVICES=1 python3 op_tests/test_vsa_qk_fp8_pv_fp4.py
  HIP_VISIBLE_DEVICES=1 python3 op_tests/test_vsa_qk_fp8_pv_fp4.py --num-q-blks 391  # ~50k tokens
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch

import aiter
from aiter.ops.vsa_qk_fp8_pv_fp4 import (
    vsa_qk_fp8_pv_fp4_dropB,
    build_l2_aware_lim_vsa_qk_fp8_pv_fp4,
)

NUM_HEADS = 4
HEAD_DIM = 128
SPARSE_BLK = 128


# --------------------------------------------------------------------------- #
# Quantisation helpers — match the kernel's E8M0 power-of-2 scale grid.
# Same math as /home/vsa_qk_fp8_pv_fp4_hip/vsa_hybrid.py.
# --------------------------------------------------------------------------- #
_FP8_MAX = 448.0
_FP4_MAX = 6.0
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _quantize_fp8_e8m0(x: torch.Tensor, group_size: int = 32):
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-30)
    exp = torch.ceil(torch.log2(amax / _FP8_MAX)).clamp(-127.0, 127.0)
    scale = torch.exp2(exp)
    x_norm = (grouped / scale[..., None]).clamp(-_FP8_MAX, _FP8_MAX)
    x_fp8 = x_norm.to(torch.float8_e4m3fn).reshape_as(x)
    scale_byte = (exp.int() + 127).clamp(0, 255).to(torch.uint8)
    return x_fp8, scale_byte


def _quantize_fp4_e8m0_per_channel_kblock(v: torch.Tensor, kblock_size: int = 32):
    BH, T, D = v.shape
    assert T % kblock_size == 0
    grouped = v.float().reshape(BH, T // kblock_size, kblock_size, D)
    amax = grouped.abs().amax(dim=-2).clamp_min(1e-30)            # (BH, T/32, D)
    exp = torch.ceil(torch.log2(amax / _FP4_MAX)).clamp(-127.0, 127.0)
    scale = torch.exp2(exp)
    scale_row = scale.repeat_interleave(kblock_size, dim=-2)      # (BH, T, D)
    v_norm = (v.float() / scale_row).clamp(-_FP4_MAX, _FP4_MAX)

    ax = v_norm.abs()
    abs_idx = torch.where(
        ax <= 0.25, 0,
        torch.where(ax <= 0.75, 1,
        torch.where(ax <= 1.25, 2,
        torch.where(ax <= 1.75, 3,
        torch.where(ax <= 2.5,  4,
        torch.where(ax <= 3.5,  5,
        torch.where(ax <= 5.0,  6, 7))))))).to(torch.uint8)
    sign_bit = torch.where(v_norm < 0.0, 8, 0).to(torch.uint8)
    nibble = (abs_idx | sign_bit).reshape(BH, T, D)
    lo = nibble[..., 0::2].to(torch.int32)
    hi = nibble[..., 1::2].to(torch.int32)
    packed = ((hi << 4) | lo).to(torch.uint8)

    scale_byte = (exp.int() + 127).clamp(0, 255).to(torch.uint8)
    return packed, scale_byte


def _dequantize_fp8_e8m0(x_fp8: torch.Tensor, scale_byte: torch.Tensor,
                          group_size: int = 32) -> torch.Tensor:
    exp = scale_byte.to(torch.float32) - 127.0
    scale = torch.exp2(exp).unsqueeze(-1)
    x = x_fp8.to(torch.float32).reshape(*x_fp8.shape[:-1], -1, group_size)
    return (x * scale).reshape_as(x_fp8).to(torch.float32)


def _dequantize_fp4_per_channel_kblock(x_packed: torch.Tensor,
                                        scale_byte: torch.Tensor,
                                        kblock_size: int = 32) -> torch.Tensor:
    BH, T, D2 = x_packed.shape
    D = D2 * 2
    lo_idx = (x_packed & 0x0F).to(torch.int64)
    hi_idx = ((x_packed >> 4) & 0x0F).to(torch.int64)
    levels = _FP4_LEVELS.to(x_packed.device)
    lo_sign = torch.where((lo_idx & 8) != 0, -1.0, 1.0)
    hi_sign = torch.where((hi_idx & 8) != 0, -1.0, 1.0)
    lo_val = lo_sign * levels[lo_idx & 7]
    hi_val = hi_sign * levels[hi_idx & 7]
    v = torch.stack([lo_val, hi_val], dim=-1).reshape(BH, T, D)
    exp = scale_byte.to(torch.float32) - 127.0
    scale = torch.exp2(exp)                                       # (BH, T/32, D)
    scale_row = scale.repeat_interleave(kblock_size, dim=-2)     # (BH, T, D)
    return v * scale_row


# --------------------------------------------------------------------------- #
# Synthetic data — banded sliding-window mask plus a few full-dense rows,
# matching the production VSA workload (sparsity = 0.0846).
# Index/scale tensors are built DIRECTLY (no bool-mask + argsort) so the
# 1M-token case fits comfortably under 5 GB HBM.
# --------------------------------------------------------------------------- #
def make_synthetic_data(seed: int, num_q_blks: int,
                         B: int = 1, H: int = NUM_HEADS,
                         sparsity: float = 0.0846,
                         dense_frac: float = 0.03) -> dict:
    T = num_q_blks * SPARSE_BLK
    max_kv = num_q_blks
    device = "cuda"

    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    n_attended = max(1, int(round(max_kv * sparsity)))
    n_dense_rows = max(1, int(round(num_q_blks * dense_frac)))
    half = n_attended // 2

    # Banded q->KV index, width = max_kv (so dense-row promotion needs no realloc).
    qs_ = torch.arange(num_q_blks, device=device)
    lo = (qs_ - half).clamp(min=0)
    lo = torch.minimum(lo, torch.tensor(max_kv - n_attended,
                                         device=device).clamp(min=0))
    band_offsets = torch.arange(n_attended, device=device)
    banded_row = (lo.unsqueeze(1) + band_offsets.unsqueeze(0)).to(torch.int32)

    BH = B * H
    q2k_index = torch.full((B, H, num_q_blks, max_kv), -1,
                            dtype=torch.int32, device=device)
    q2k_index[..., :n_attended] = banded_row
    q2k_num = torch.full((B, H, num_q_blks), n_attended,
                          dtype=torch.int32, device=device)

    dense_q = torch.randint(0, num_q_blks, (n_dense_rows,),
                             device=device, generator=g)
    for q in torch.unique(dense_q).tolist():
        q2k_index[:, :, q, :] = torch.arange(max_kv, dtype=torch.int32,
                                              device=device)
        q2k_num[:, :, q] = max_kv

    # Generate + quantise Q / K / V (one at a time to keep peak HBM low).
    def _gen_fp8_qk():
        x = torch.randn((BH, T, HEAD_DIM), device=device,
                         generator=g, dtype=torch.float32).to(torch.bfloat16)
        x_fp8, xs = _quantize_fp8_e8m0(x, group_size=HEAD_DIM // 4)
        del x
        torch.cuda.empty_cache()
        return x_fp8.contiguous(), xs.contiguous()

    q_fp8, qs = _gen_fp8_qk()
    k_fp8, ks = _gen_fp8_qk()

    v_bf16 = torch.randn((BH, T, HEAD_DIM), device=device,
                          generator=g, dtype=torch.float32).to(torch.bfloat16)
    v_packed, vs = _quantize_fp4_e8m0_per_channel_kblock(v_bf16, kblock_size=32)
    del v_bf16
    torch.cuda.empty_cache()
    # Re-pack VS to HBM-coalesced layout (BH, T/128, D, 4) -- 4×ubyte→1×dword.
    BH_v, NB, D_v = vs.shape
    assert NB % 4 == 0
    vs = vs.reshape(BH_v, NB // 4, 4, D_v).permute(0, 1, 3, 2).contiguous()

    vbs = torch.full((num_q_blks,), SPARSE_BLK, dtype=torch.int32,
                      device=device).contiguous()
    q2k_idx_flat = q2k_index.reshape(BH * num_q_blks, -1).contiguous()
    q2k_num_flat = q2k_num.reshape(BH * num_q_blks).contiguous()

    return {
        'B': B, 'H': H, 'T': T, 'D': HEAD_DIM,
        'num_q_blks': num_q_blks, 'max_kv': max_kv,
        'q': q_fp8, 'k': k_fp8, 'v': v_packed.contiguous(),
        'qs': qs, 'ks': ks, 'vs': vs,
        'q2k_idx': q2k_idx_flat, 'q2k_num': q2k_num_flat, 'vbs': vbs,
    }


# --------------------------------------------------------------------------- #
# Spot-check FP32 reference — recompute the attention math for `n_samples`
# random (BH, q_block) tiles on CPU and compare with the kernel's output
# on those tiles.  Tractable up to T = 1M.
# --------------------------------------------------------------------------- #
def spot_check(data: dict, out: torch.Tensor, lse: torch.Tensor,
               n_samples: int = 32, seed: int = 0,
               block_m: int = 128, block_n: int = 128) -> dict:
    BH, T, D = data['q'].shape
    num_q_blks = T // block_m
    qk_scale = (1.0 / math.sqrt(D)) * 1.44269504

    Q = _dequantize_fp8_e8m0(data['q'], data['qs']).cpu()
    K = _dequantize_fp8_e8m0(data['k'], data['ks']).cpu()
    # Undo HBM-coalesced VS layout (BH, T/128, D, 4) -> (BH, T/32, D).
    vs = data['vs'].permute(0, 1, 3, 2).reshape(BH, -1, D).contiguous()
    V = _dequantize_fp4_per_channel_kblock(data['v'], vs, 32).cpu()
    q2k_idx = data['q2k_idx'].cpu()
    q2k_num = data['q2k_num'].cpu()
    vbs = data['vbs'].cpu()

    rng = torch.Generator().manual_seed(seed)
    total_tiles = BH * num_q_blks
    n_samples = min(n_samples, total_tiles)
    tile_ids = torch.randperm(total_tiles, generator=rng)[:n_samples].tolist()

    out_cpu = out.cpu().float()
    lse_cpu = lse.cpu().float()

    cn = ca2 = cb2 = lcn = lca2 = lcb2 = 0.0
    l1n = l1d = 0.0
    mae = 0.0
    nan_tiles = 0
    for tid in tile_ids:
        bh, qb = tid // num_q_blks, tid % num_q_blks
        q_lo, q_hi = qb * block_m, (qb + 1) * block_m
        Qblk = Q[bh, q_lo:q_hi]
        m_i = torch.full((block_m,), -float("inf"))
        l_i = torch.ones((block_m,))
        acc = torch.zeros((block_m, D))
        num = int(q2k_num[bh * num_q_blks + qb].item())
        for kvi in range(num):
            kv_idx = int(q2k_idx[bh * num_q_blks + qb, kvi].item())
            blk_sz = int(vbs[kv_idx].item())
            k_lo, k_hi = kv_idx * block_n, (kv_idx + 1) * block_n
            qk = Qblk @ K[bh, k_lo:k_hi].T
            mask = torch.arange(block_n) < blk_sz
            qk = torch.where(mask[None, :], qk, torch.tensor(-float("inf")))
            row_max = qk.max(dim=-1).values * qk_scale
            m_ij = torch.maximum(m_i, row_max)
            m_safe = torch.where(torch.isinf(m_ij), torch.zeros_like(m_ij), m_ij)
            p = torch.exp2(qk * qk_scale - m_safe[:, None])
            l_ij = p.sum(dim=-1)
            alpha = torch.exp2(m_i - m_safe)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + p @ V[bh, k_lo:k_hi]
            m_i = m_safe
        out_ref = acc / l_i[:, None]
        lse_ref = m_i + torch.log2(l_i)
        out_k = out_cpu[bh, q_lo:q_hi]
        lse_k = lse_cpu[bh, q_lo:q_hi]
        if torch.isnan(out_k).any() or torch.isinf(out_k).any():
            nan_tiles += 1
        d = (out_k - out_ref).abs()
        mae = max(mae, d.max().item())
        l1n += d.sum().item()
        l1d += out_ref.abs().sum().item()
        cn += (out_k * out_ref).sum().item()
        ca2 += (out_k * out_k).sum().item()
        cb2 += (out_ref * out_ref).sum().item()
        lcn += (lse_k * lse_ref).sum().item()
        lca2 += (lse_k * lse_k).sum().item()
        lcb2 += (lse_ref * lse_ref).sum().item()

    return {
        'n_samples': n_samples,
        'cos': cn / max(math.sqrt(ca2 * cb2), 1e-30),
        'max_abs': mae,
        'rel_l1': l1n / max(l1d, 1e-30),
        'cos_lse': lcn / max(math.sqrt(lca2 * lcb2), 1e-30),
        'nan_tiles': nan_tiles,
    }


# --------------------------------------------------------------------------- #
# Benchmark + accuracy entry
# --------------------------------------------------------------------------- #
def _bench(call_fn, iters: int = 30, warmup: int = 10) -> float:
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-q-blks", type=int, default=None,
                   help="Single-size run; if not given, sweeps 391..7813 (50k..1M).")
    p.add_argument("--samples", type=int, default=32,
                   help="Random tiles per size for the spot-check accuracy.")
    p.add_argument("--no-accuracy", action="store_true")
    args = p.parse_args()

    if args.num_q_blks is not None:
        sizes = [args.num_q_blks]
    else:
        sizes = [391, 781, 1563, 3125, 4688, 6250, 7813]   # 50k..1M

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"sizes (num_q_blks): {sizes}")
    print(f"tokens (T):         {[s * SPARSE_BLK for s in sizes]}")

    results = []
    for nqb in sizes:
        T = nqb * SPARSE_BLK
        print(f"\n=== T = {T:>7,} tokens (num_q_blks={nqb}) ===")
        data = make_synthetic_data(seed=args.seed, num_q_blks=nqb)
        total_tiles = data['B'] * data['H'] * data['num_q_blks']

        # Build L2-aware lim on the GPU via the new aiter helper.
        t_lim0 = time.perf_counter()
        lim, n_dense = build_l2_aware_lim_vsa_qk_fp8_pv_fp4(
            data['q2k_idx'], data['q2k_num'], data['max_kv'])
        torch.cuda.synchronize()
        t_lim1 = time.perf_counter()
        print(f"  build_l2_aware_lim: {(t_lim1 - t_lim0) * 1e3:.2f} ms  "
              f"(n_dense={n_dense}/{total_tiles})")

        out = torch.empty((data['B'] * data['H'], T, HEAD_DIM),
                          dtype=torch.bfloat16, device='cuda')
        lse = torch.empty((data['B'] * data['H'], T),
                          dtype=torch.float32, device='cuda')
        counters = torch.zeros(2, dtype=torch.int32, device='cuda')

        def call():
            return vsa_qk_fp8_pv_fp4_dropB(
                q=data['q'], k=data['k'], v=data['v'],
                qs=data['qs'], ks=data['ks'], vs=data['vs'],
                q2k_idx=data['q2k_idx'], q2k_num=data['q2k_num'],
                vbs=data['vbs'], lim=lim, n_dense=n_dense,
                B=data['B'], T=T,
                num_q_blks=data['num_q_blks'], max_kv=data['max_kv'],
                out=out, lse=lse, counters=counters,
            )

        # Warm + sanity launch.
        call()
        torch.cuda.synchronize()
        nan = torch.isnan(out.float()).any().item()
        inf = torch.isinf(out.float()).any().item()
        if nan or inf:
            print(f"  !! BAD OUTPUT: has_nan={nan} has_inf={inf}")

        ms = _bench(call, iters=30, warmup=10)
        print(f"  runtime: {ms:.3f} ms / call  "
              f"({ms * 1000 / total_tiles:.2f} us / tile)")

        row = {'T': T, 'tiles': total_tiles, 'ms': ms, 'n_dense': n_dense,
                'has_nan': nan, 'has_inf': inf}
        if not args.no_accuracy:
            t0 = time.perf_counter()
            stats = spot_check(data, out, lse, n_samples=args.samples,
                                seed=args.seed)
            t1 = time.perf_counter()
            print(f"  spot-check ({stats['n_samples']} tiles, {(t1 - t0):.1f}s): "
                  f"cos={stats['cos']:.6f}  max_abs={stats['max_abs']:.3e}  "
                  f"rel_l1={stats['rel_l1']:.3e}  cos_lse={stats['cos_lse']:.6f}")
            row.update(stats)
        results.append(row)

        del data, out, lse, lim, counters
        torch.cuda.empty_cache()

    print("\n" + "=" * 110)
    print(" SUMMARY")
    print("=" * 110)
    header = (f"{'T':>10}  {'tiles':>6}  {'ms/call':>9}  {'us/tile':>8}  "
              f"{'cos':>9}  {'max_abs':>10}  {'rel_l1':>10}  {'cos(lse)':>9}")
    print(header)
    print("-" * len(header))
    for r in results:
        row = (f"{r['T']:>10,}  {r['tiles']:>6}  "
               f"{r['ms']:>9.3f}  {r['ms']*1000/r['tiles']:>8.3f}")
        if 'cos' in r:
            row += (f"  {r['cos']:>9.6f}  {r['max_abs']:>10.4e}  "
                    f"{r['rel_l1']:>10.4e}  {r['cos_lse']:>9.6f}")
        print(row)


if __name__ == "__main__":
    main()
