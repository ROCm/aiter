#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Side-by-side perf comparison: PR #2649 `fused_qk_norm_rope_group_quant_cache`
vs FlyDSL `flydsl_qk_norm_rope_quant`.

Both kernels are run with matched shapes (D=512, RD=64, num_kv_heads=1,
fp8 + group_size=64 + e8m0 scale, GPT-J RoPE), but they do slightly
different work:

  * `fused_qk_norm_rope_group_quant_cache` writes K-nope to a paged KV
    cache via `slot_mapping` (scattered writes) and emits a separate
    bf16 `k_pe_out` tensor for the rotated K-PE. Q output is bf16
    (no Q quant).
  * `flydsl_qk_norm_rope_quant` writes Q and KV to dense contiguous
    output tensors, and BOTH Q and KV are fp8-quantised.

The two are not strictly equivalent (the cu kernel skips Q quant and
adds paged-cache indirection; the flydsl kernel quantises Q too), so
treat the comparison as an order-of-magnitude perf reference, not a
direct A/B.

Usage (on the GPU host):
    python op_tests/bench_qk_norm_rope_quant_compare.py
    python op_tests/bench_qk_norm_rope_quant_compare.py -T 1 16 256 4096 --H 128
"""

import argparse
import random

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.flydsl import flydsl_qk_norm_rope_quant
from aiter.test_common import run_perftest


def _build_cos_sin(max_pos: int, rope_dim: int, device: str, dtype: torch.dtype):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim)
    )
    pos_range = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum("i,j->ij", pos_range, inv_freq)
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _bench_flydsl(T, H, D, RD, *, device):
    dtype = torch.bfloat16
    max_pos = max(T, 64)
    cos, sin = _build_cos_sin(max_pos, RD, device, dtype)

    q = torch.randn(T, H * D, dtype=dtype, device=device) * 0.1
    kv = torch.randn(T, D, dtype=dtype, device=device) * 0.1
    kv_w = torch.randn(D, dtype=dtype, device=device).abs() + 0.5
    pos = torch.randint(0, max_pos - 1, (T,), dtype=torch.int64, device=device)

    _, us = run_perftest(
        flydsl_qk_norm_rope_quant,
        q,
        kv,
        kv_w,
        cos,
        sin,
        pos,
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        quant=True,
        quant_group_size=64,
        scale_dtype="e8m0",
    )
    # Bytes: Q in (bf16) + KV in (bf16) + kv_w (bf16, small)
    #      + Q out (fp8) + KV out (fp8) + per-(token,head,group) e8m0 byte
    G, NG = 64, D // 64
    bytes_in = T * H * D * 2 + T * D * 2 + D * 2
    bytes_out = T * H * D * 1 + T * D * 1
    scale_bytes = (T * H + T) * NG * 1
    bw = (bytes_in + bytes_out + scale_bytes) / (us * 1e-6) / 1e12  # TB/s
    return us, bw


def _bench_fused(
    T,
    H,
    D,
    RD,
    *,
    device,
    num_kv_heads=1,
    block_size=1,
    q_quant=False,
    quant_group_size=64,
):
    """PR #2649 kernel. D = head_dim (kv_lora_rank + qk_rope_head_dim).

    kv_lora_rank = D - RD must be 448 for the kernel as it dispatches
    only on head_dim == 512.

    When ``q_quant=True``, Q is fp8 + e8m0 group scale (matches flydsl's
    ``fp8_1x{G}_e8m0`` mode).
    """
    dtype = torch.bfloat16
    cache_dtype = dtypes.fp8
    nope_dim = D - RD
    assert D == 512 and RD == 64, "PR #2649 kernel hard-codes head_dim=512, rot=64"

    num_blocks = max(1, T // block_size + 1)
    total_slots = num_blocks * block_size
    slot_mapping = torch.tensor(
        random.sample(range(total_slots), T), dtype=torch.long, device=device
    )

    kv = torch.randn(T, num_kv_heads, D, dtype=dtype, device=device) * 0.1
    q = torch.randn(T, H, D, dtype=dtype, device=device) * 0.1
    k_weight = torch.ones(D, dtype=dtype, device=device)
    cos, sin = _build_cos_sin(max(T, 64), RD, device, dtype)
    pos = torch.randint(0, T if T > 1 else 2, (T,), device=device)

    kv_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, D, dtype=cache_dtype, device=device
    )
    if q_quant:
        q_out = torch.empty((T, H, D), dtype=dtypes.fp8, device=device)
        NG = D // quant_group_size
        q_scale = torch.zeros((T, H, NG), dtype=torch.uint8, device=device)
    else:
        q_out = torch.empty((T, H, D), dtype=dtype, device=device)
        q_scale = None
    k_pe_out = torch.empty(T, num_kv_heads, RD, dtype=dtype, device=device)

    _, us = run_perftest(
        aiter.fused_qk_norm_rope_group_quant_cache,
        q,
        kv,
        k_pe_out,
        k_weight,
        kv_cache,
        q_out,
        slot_mapping,
        pos,
        cos,
        sin,
        1e-6,
        False,  # is_neox=False to match flydsl GPT-J style
        True,  # is_nope_first
        q_scale=q_scale,
        quant_group_size=quant_group_size if q_quant else 64,
        scale_dtype="e8m0",
    )
    # Bytes: Q in (bf16) + KV in (bf16) + k_weight (bf16, small)
    #      + Q out (fp8 or bf16) + (optional Q scale 1B/group)
    #      + KV cache nope (fp8) + e8m0 scale (1B/group)
    #      + k_pe_out (bf16)
    K_NG = nope_dim // 64
    bytes_in = T * H * D * 2 + T * num_kv_heads * D * 2 + D * 2
    if q_quant:
        Q_NG = D // quant_group_size
        bytes_q_out = T * H * D * 1 + T * H * Q_NG * 1  # fp8 + e8m0 scale byte
    else:
        bytes_q_out = T * H * D * 2  # bf16, no scale
    bytes_k_cache = T * num_kv_heads * nope_dim * 1 + T * num_kv_heads * K_NG * 1
    bytes_k_pe = T * num_kv_heads * RD * 2
    bw = (bytes_in + bytes_q_out + bytes_k_cache + bytes_k_pe) / (us * 1e-6) / 1e12
    return us, bw


def main():
    p = argparse.ArgumentParser(
        description="Side-by-side perf: flydsl_qk_norm_rope_quant vs "
        "fused_qk_norm_rope_group_quant_cache."
    )
    p.add_argument(
        "-T",
        "--T",
        type=int,
        nargs="*",
        default=[1, 4, 16, 32, 64, 128, 256, 512, 1024, 4096],
        help="token-count sweep",
    )
    p.add_argument(
        "--H",
        type=int,
        nargs="*",
        default=[16, 128],
        help="Q-heads sweep (DeepSeek-typical: 128)",
    )
    p.add_argument("--D", type=int, default=512, help="head_dim (fixed at 512)")
    p.add_argument("--RD", type=int, default=64, help="rope_head_dim (fixed at 64)")
    p.add_argument(
        "--num_kv_heads",
        type=int,
        default=1,
        help="num KV heads (flydsl is fixed at 1)",
    )
    args = p.parse_args()

    device = "cuda"
    torch.set_default_device(device)
    torch.manual_seed(0)

    rows = []
    for H in args.H:
        for T in args.T:
            try:
                fly_us, fly_bw = _bench_flydsl(T, H, args.D, args.RD, device=device)
            except Exception as e:
                fly_us, fly_bw = float("nan"), float("nan")
                aiter.logger.warning("flydsl T=%d H=%d failed: %s", T, H, e)
            try:
                cu_bf16_us, cu_bf16_bw = _bench_fused(
                    T,
                    H,
                    args.D,
                    args.RD,
                    device=device,
                    num_kv_heads=args.num_kv_heads,
                    q_quant=False,
                )
            except Exception as e:
                cu_bf16_us, cu_bf16_bw = float("nan"), float("nan")
                aiter.logger.warning("fused-bf16 T=%d H=%d failed: %s", T, H, e)
            try:
                cu_fp8_us, cu_fp8_bw = _bench_fused(
                    T,
                    H,
                    args.D,
                    args.RD,
                    device=device,
                    num_kv_heads=args.num_kv_heads,
                    q_quant=True,
                    quant_group_size=64,
                )
            except Exception as e:
                cu_fp8_us, cu_fp8_bw = float("nan"), float("nan")
                aiter.logger.warning("fused-fp8 T=%d H=%d failed: %s", T, H, e)

            sp_bf16 = (
                (cu_bf16_us / fly_us)
                if fly_us == fly_us and fly_us > 0
                else float("nan")
            )
            sp_fp8 = (
                (cu_fp8_us / fly_us)
                if fly_us == fly_us and fly_us > 0
                else float("nan")
            )
            rows.append(
                {
                    "T": T,
                    "H": H,
                    "flydsl_us": round(fly_us, 3),
                    "fused_bf16_us": round(cu_bf16_us, 3),
                    "fused_fp8_us": round(cu_fp8_us, 3),
                    "fp8/fly": round(sp_fp8, 3),
                    "flydsl_TB/s": round(fly_bw, 3),
                    "fused_fp8_TB/s": round(cu_fp8_bw, 3),
                }
            )

    df = pd.DataFrame(rows)
    print("\n=== flydsl vs fused_qk_norm_rope_group_quant_cache (bf16 Q vs fp8 Q) ===")
    print(
        f"D={args.D}, RD={args.RD}, num_kv_heads={args.num_kv_heads}, group=64 e8m0, GPT-J RoPE"
    )
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
