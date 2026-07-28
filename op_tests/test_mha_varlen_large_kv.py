# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Forward-only large-seqlen check for FMHA varlen/group hd192.

Dispatches by KV byte extent (per-head row):
  >= 4GiB  -> OPUS gfx950 (hybrid buffer K/V)
  <  4GiB  -> v3 asm group .co (fwd_hd192_hd128_bf16[_causal]_group.co)

Builds one sequence of length S and checks only the last `--check_rows` query rows
(global positions near S-1). Reference cost is O(check_rows * S), not O(S^2).

Examples:
    python3 op_tests/test_mha_varlen_large_kv.py -n 1 -s 11500000 -c false
    python3 op_tests/test_mha_varlen_large_kv.py -n 1 -s 900000  -c false
    python3 op_tests/test_mha_varlen_large_kv.py -n 1 -s 900000  -c false --force-asm

Bracket 2^23 = 8,388,608:
    python3 op_tests/test_mha_varlen_large_kv.py -n 1 -s 8000000  -c true
"""

import argparse
import os

import torch

_2P23 = 2**23  # 8,388,608
_U32_LIMIT = 1 << 32
_V3_CO_DIR = "hsa/gfx950/fmha_v3_fwd"


def kv_byte_extent(seqlen: int, dq: int, dv: int, elem_size: int = 2) -> tuple[int, int]:
    k_bytes = seqlen * dq * elem_size
    v_bytes = seqlen * dv * elem_size
    return k_bytes, v_bytes


def expect_backend(seqlen: int, dq: int, dv: int, force_asm: bool, force_opus: bool) -> str:
    if force_asm:
        return "asm_v3"
    if force_opus:
        return "opus"
    k_bytes, v_bytes = kv_byte_extent(seqlen, dq, dv)
    if k_bytes >= _U32_LIMIT or v_bytes >= _U32_LIMIT:
        return "opus"
    return "asm_v3"


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
    p.add_argument("-c", "--causal", default="false")
    p.add_argument("-d_qk_v", type=str, default="192,128")
    p.add_argument("--check_rows", type=int, default=512)
    p.add_argument("--kv_chunk", type=int, default=8192)
    p.add_argument(
        "--force-asm",
        action="store_true",
        help="set AITER_DISABLE_FMHA_OPUS=1 before import",
    )
    p.add_argument(
        "--force-opus",
        action="store_true",
        help="fail if dispatch would not use OPUS (for >=4GiB shapes)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    dq, dv = (int(x) for x in args.d_qk_v.split(","))
    assert args.nheads % args.gqa_ratio == 0
    nheads_k = args.nheads // args.gqa_ratio
    S = args.seqlen

    if args.force_asm:
        os.environ["AITER_DISABLE_FMHA_OPUS"] = "1"
    elif args.force_opus:
        os.environ.pop("AITER_DISABLE_FMHA_OPUS", None)

    import aiter
    from aiter import dtypes

    if isinstance(args.causal, str):
        args.causal = dtypes.str2bool(args.causal)

    dtype = dtypes.bf16
    device = "cuda"
    scale = dq**-0.5

    k_bytes, v_bytes = kv_byte_extent(S, dq, dv)
    backend = expect_backend(S, dq, dv, args.force_asm, args.force_opus)
    v3_mode = v3_group_dispatch_mode(S, args.causal)
    co_name = expected_group_co(args.causal)

    print(
        f"seqlen S = {S:,}  ({'>' if S > _2P23 else '<='} 2^23 = {_2P23:,})  "
        f"causal={args.causal}  nheads={args.nheads}/{nheads_k}"
    )
    print(f"KV byte extent: k={k_bytes:,}  v={v_bytes:,}  (limit={_U32_LIMIT:,})")
    print(f"AITER_DISABLE_FMHA_OPUS={os.environ.get('AITER_DISABLE_FMHA_OPUS', '0')}")
    print(f"expect backend: {backend}")
    if backend == "asm_v3":
        print(f"expect fmha_v3_fwd mode={v3_mode}: {_V3_CO_DIR}/{co_name}")
    else:
        print("expect fmha_fwd_bf16_opus_varlen (hd192 hybrid buffer)")

    if args.force_opus and backend != "opus":
        raise SystemExit(
            f"--force-opus: KV extent k={k_bytes} v={v_bytes} < 4GiB; use larger -s"
        )

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
        print(f"#TEST FAILED ({backend} large-seqlen varlen mismatch)")


if __name__ == "__main__":
    main()
