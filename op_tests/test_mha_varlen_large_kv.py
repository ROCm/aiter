# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Forward-only large-seqlen check for the v3 asm FMHA *varlen/group* kernels.

Uses `flash_attn_varlen_func` (cu_seqlens) so it loads the group .co:
    causal=False -> fwd_hd192_hd128_bf16_group.co
    causal=True  -> fwd_hd192_hd128_bf16_causal_group.co

Builds one sequence of length S and checks only the last `--check_rows` query rows
(global positions near S-1). Reference cost is O(check_rows * S), not O(S^2).

Example (bracket 2^23 = 8,388,608):
    python3 op_tests/test_mha_varlen_large_kv.py -n 1 -s 8000000  -c true
    python3 op_tests/test_mha_varlen_large_kv.py -n 1 -s 900000  -c false

When max_seqlen_q implies Q tiles > 65535, aiter host splits gdz launches and sets
s_q_tile_base on each chunk (still fwd_*_group.co).

Forces v3 asm group .co (AITER_DISABLE_FMHA_OPUS=1 before import).
"""

import argparse
import os

os.environ.setdefault("AITER_DISABLE_FMHA_OPUS", "1")

import torch

import aiter
from aiter import dtypes

_2P23 = 2**23  # 8,388,608
_V3_CO_DIR = "hsa/gfx950/fmha_v3_fwd"


def v3_group_dispatch_mode(seqlen: int, causal: bool) -> int:
    """fmha_fwd.csv mode for group on gfx950 hd192 (always mode=1 group .co)."""
    del seqlen, causal
    return 1


def expected_group_co(causal: bool) -> str:
    base = "fwd_hd192_hd128_bf16"
    if causal:
        base += "_causal"
    return base + "_group.co"


def ref_last_rows(q, k, v, q0, causal, scale, kv_chunk=8192):
    """Exact fp32 attention for query rows [q0:]. q/k/v: [h, S, d] (kv broadcast)."""
    h, S, _dq = q.shape
    dv = v.shape[2]
    qf = q[:, q0:, :].float()
    R = qf.shape[1]

    row_max = torch.full((h, R, 1), float("-inf"), device=q.device, dtype=torch.float32)
    softmax_denom = torch.zeros((h, R, 1), device=q.device, dtype=torch.float32)
    acc = torch.zeros((h, R, dv), device=q.device, dtype=torch.float32)

    q_idx = torch.arange(q0, S, device=q.device).view(1, R, 1)
    for start in range(0, S, kv_chunk):
        end = min(start + kv_chunk, S)
        kc = k[:, start:end, :].float()
        vc = v[:, start:end, :].float()
        s = torch.matmul(qf, kc.transpose(-1, -2)) * scale
        if causal:
            k_idx = torch.arange(start, end, device=q.device).view(1, 1, -1)
            s = s.masked_fill(k_idx > q_idx, float("-inf"))
        m_new = torch.maximum(row_max, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        alpha = torch.exp(row_max - m_new)
        softmax_denom = softmax_denom * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + torch.matmul(p, vc)
        row_max = m_new
    return (acc / softmax_denom.clamp_min(1e-20)).to(q.dtype)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("-n", "--nheads", type=int, default=1)
    p.add_argument("-gr", "--gqa_ratio", type=int, default=1)
    p.add_argument(
        "-s",
        "--seqlen",
        type=int,
        default=900000,
        help="single-sequence length S (sq == sk == S)",
    )
    p.add_argument("-c", "--causal", type=dtypes.str2bool, default=False)
    p.add_argument("-d_qk_v", type=dtypes.str2tuple, default=(192, 128))
    p.add_argument("--check_rows", type=int, default=512)
    p.add_argument("--kv_chunk", type=int, default=8192)
    return p.parse_args()


def main():
    args = parse_args()
    dq, dv = args.d_qk_v
    assert args.nheads % args.gqa_ratio == 0
    nheads_k = args.nheads // args.gqa_ratio
    S = args.seqlen
    dtype = dtypes.bf16
    device = "cuda"
    scale = dq**-0.5

    v3_mode = v3_group_dispatch_mode(S, args.causal)
    co_name = expected_group_co(args.causal)

    print(
        f"seqlen S = {S:,}  ({'>' if S > _2P23 else '<='} 2^23 = {_2P23:,})  "
        f"causal={args.causal}  nheads={args.nheads}/{nheads_k}"
    )
    print(f"AITER_DISABLE_FMHA_OPUS={os.environ.get('AITER_DISABLE_FMHA_OPUS', '0')}")
    print(f"expect fmha_v3_fwd mode={v3_mode}: {_V3_CO_DIR}/{co_name}")

    torch.manual_seed(0)
    q = torch.randn(S, args.nheads, dq, device=device, dtype=dtype)
    k = torch.randn(S, nheads_k, dq, device=device, dtype=dtype)
    v = torch.randn(S, nheads_k, dv, device=device, dtype=dtype)
    cu_q = torch.tensor([0, S], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, S], device=device, dtype=torch.int32)

    with torch.no_grad():
        out = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            S,
            S,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=args.causal,
            window_size=(-1, -1, 0),
            return_lse=False,
            return_attn_probs=False,
        )
    if isinstance(out, (tuple, list)):
        out = out[0]

    q0 = max(0, S - args.check_rows)
    qb = q.transpose(0, 1)
    kb = k.transpose(0, 1)
    vb = v.transpose(0, 1)
    if args.gqa_ratio > 1:
        kb = kb.repeat_interleave(args.gqa_ratio, dim=0)
        vb = vb.repeat_interleave(args.gqa_ratio, dim=0)

    with torch.no_grad():
        ref = ref_last_rows(qb, kb, vb, q0, args.causal, scale, args.kv_chunk)
    out_last = out[q0:].transpose(0, 1).float()

    diff = (out_last - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    tol = 0.02
    n_bad = int((diff > tol).sum().item())
    finite = torch.isfinite(out_last).all().item()
    print(f"checked last {out_last.shape[1]} query rows (global pos {q0:,}..{S-1:,})")
    print(
        f"out finite: {finite}  abs max: {out_last.abs().max().item():.4f}  "
        f"ref abs max: {ref.float().abs().max().item():.4f}"
    )
    print(
        f"max diff: {max_diff:.6f}  mean diff: {mean_diff:.6f}  "
        f"bad(>{tol}): {n_bad}/{out_last.numel()}  tol: {tol}"
    )
    if finite and max_diff <= tol:
        print("#TEST PASSED")
    else:
        print("#TEST FAILED (large-seqlen / global-load group .co mismatch)")


if __name__ == "__main__":
    main()
