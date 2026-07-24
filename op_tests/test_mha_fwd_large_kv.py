# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Forward-only large-KV correctness check for the v3 asm FMHA fwd kernels.

Purpose: verify that a single batch's K/V slice exceeding 4 GiB is addressed
correctly. `buffer_load` uses a 32-bit offset from the (per-batch) buffer base,
so anything past 4 GiB reads as 0; `global_load` uses a full 64-bit address and
must stay correct. Run this against the buffer-load .co (expect FAIL on the tail)
and the global-load .co (expect PASS).

Per-batch byte size (bf16):
    K = seqlen_k * nheads_k * hdim_q * 2
    V = seqlen_k * nheads_k * hdim_v * 2
Both must exceed 4 GiB to actually cross the boundary for K and V.

Only the forward pass is exercised (no autograd), and the reference uses a
flash-style chunked online softmax so host memory stays bounded regardless of
seqlen_k.

Example (K~7.4GB, V~4.9GB per batch):
    python3 op_tests/test_mha_fwd_large_kv.py -b 1 -n 64 -q 256 -k 300000 -c false

Routing: flash_attn_func only picks fmha_v3_fwd when seqlen_q > 128 (see mha.py);
with -q 128 you get CK mha_fwd (.jit), not hsa/gfx950/fmha_v3_fwd/*.co.

This script forces v3 asm (fmha_v3_fwd .co), not OPUS:
    AITER_DISABLE_FMHA_OPUS=1 is set before importing aiter.
"""

import argparse
import os

# hd192 default-on OPUS would bypass fmha_v3_fwd .co; this test targets asm v3.
os.environ.setdefault("AITER_DISABLE_FMHA_OPUS", "1")

import torch

import aiter
from aiter import dtypes

_4GiB = 4 * 1024**3

_V3_CO_DIR = "hsa/gfx950/fmha_v3_fwd"


def expected_v3_co(causal: bool) -> str:
    if causal:
        return "fwd_hd192_hd128_bf16_causal.co"
    return "fwd_hd192_hd128_bf16.co"


def chunked_attn_ref(q, k, v, causal, scale, kv_chunk=8192):
    """Flash-style fp32 reference. q/k/v: [b, h, s, d] (h already broadcast for GQA)."""
    b, h, sq, dq = q.shape
    sk = k.shape[2]
    dv = v.shape[3]
    qf = q.float()

    row_max = torch.full(
        (b, h, sq, 1), float("-inf"), device=q.device, dtype=torch.float32
    )
    softmax_denom = torch.zeros((b, h, sq, 1), device=q.device, dtype=torch.float32)
    acc = torch.zeros((b, h, sq, dv), device=q.device, dtype=torch.float32)

    q_idx = torch.arange(sq, device=q.device).view(1, 1, sq, 1)
    for start in range(0, sk, kv_chunk):
        end = min(start + kv_chunk, sk)
        kc = k[:, :, start:end, :].float()
        vc = v[:, :, start:end, :].float()
        s = torch.matmul(qf, kc.transpose(-1, -2)) * scale  # [b,h,sq,chunk]
        if causal:
            k_idx = torch.arange(start, end, device=q.device).view(1, 1, 1, -1)
            # align the causal diagonal to the bottom-right (sk may exceed sq)
            s = s.masked_fill(k_idx > (q_idx + (sk - sq)), float("-inf"))

        m_new = torch.maximum(row_max, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        alpha = torch.exp(row_max - m_new)
        softmax_denom = softmax_denom * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + torch.matmul(p, vc)
        row_max = m_new

    out = acc / softmax_denom.clamp_min(1e-20)
    return out.to(q.dtype)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("-b", "--batch_size", type=int, default=1)
    p.add_argument("-n", "--nheads", type=int, default=64)
    p.add_argument("-gr", "--gqa_ratio", type=int, default=1)
    p.add_argument(
        "-q",
        "--seqlen_q",
        type=int,
        default=256,
        help="must be >128 for flash_attn_func to route to fmha_v3_fwd asm",
    )
    p.add_argument("-k", "--seqlen_k", type=int, default=300000)
    p.add_argument("-d_qk_v", type=dtypes.str2tuple, default=(192, 128))
    p.add_argument("-c", "--causal", type=dtypes.str2bool, default=False)
    p.add_argument("--kv_chunk", type=int, default=8192, help="reference kv block size")
    return p.parse_args()


def main():
    args = parse_args()
    dq, dv = args.d_qk_v
    assert args.nheads % args.gqa_ratio == 0
    nheads_k = args.nheads // args.gqa_ratio
    dtype = dtypes.bf16
    device = "cuda"

    k_bytes = args.seqlen_k * nheads_k * dq * 2
    v_bytes = args.seqlen_k * nheads_k * dv * 2
    print(
        f"per-batch K = {k_bytes/1024**3:.2f} GiB "
        f"({'>' if k_bytes > _4GiB else '<='} 4 GiB), "
        f"V = {v_bytes/1024**3:.2f} GiB "
        f"({'>' if v_bytes > _4GiB else '<='} 4 GiB)"
    )
    if k_bytes <= _4GiB and v_bytes <= _4GiB:
        print(
            "WARNING: neither K nor V exceeds 4 GiB; this will NOT exercise the "
            ">4GiB path. Increase -k / -n."
        )

    if args.seqlen_q <= 128:
        print(
            "WARNING: seqlen_q <= 128 -> flash_attn_func uses CK mha_fwd, "
            "NOT fmha_v3_fwd .co. Use -q 129 or higher."
        )

    co_name = expected_v3_co(args.causal)
    print(f"AITER_DISABLE_FMHA_OPUS={os.environ.get('AITER_DISABLE_FMHA_OPUS', '0')}")
    print(
        f"expect fmha_v3_fwd (mode=0 batch): {_V3_CO_DIR}/{co_name}  "
        f"(seqlen_q={args.seqlen_q}, causal={args.causal})"
    )

    torch.manual_seed(0)
    scale = dq**-0.5

    q = torch.randn(
        args.batch_size, args.seqlen_q, args.nheads, dq, device=device, dtype=dtype
    )
    k = torch.randn(
        args.batch_size, args.seqlen_k, nheads_k, dq, device=device, dtype=dtype
    )
    v = torch.randn(
        args.batch_size, args.seqlen_k, nheads_k, dv, device=device, dtype=dtype
    )

    with torch.no_grad():
        out = aiter.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=args.causal,
            window_size=(-1, -1),
            return_lse=False,
            return_attn_probs=False,
        )
    if isinstance(out, (tuple, list)):
        out = out[0]

    # BSHD -> BHSD; broadcast kv heads for GQA
    qb = q.transpose(1, 2)
    kb = k.transpose(1, 2)
    vb = v.transpose(1, 2)
    if args.gqa_ratio > 1:
        kb = kb.repeat_interleave(args.gqa_ratio, dim=1)
        vb = vb.repeat_interleave(args.gqa_ratio, dim=1)

    with torch.no_grad():
        ref = chunked_attn_ref(qb, kb, vb, args.causal, scale, args.kv_chunk)
    ref = ref.transpose(1, 2)  # back to BSHD

    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    tol = 0.02  # bf16
    print(f"out max diff: {max_diff:.6f}  mean diff: {mean_diff:.6f}  tol: {tol}")
    print(
        f"out abs max: {out.abs().max().item():.4f}  ref abs max: {ref.abs().max().item():.4f}"
    )
    if max_diff <= tol:
        print("#TEST PASSED")
    else:
        print(
            "#TEST FAILED (tail KV likely mis-addressed -> buffer_load 4GiB overflow)"
        )


if __name__ == "__main__":
    main()
