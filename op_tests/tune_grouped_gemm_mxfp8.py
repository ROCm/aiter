# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
r"""Autotune ``grouped_gemm_mxfp8`` — produce the HIP-vs-Triton crossover table.

The HIP kernel is a single fixed 256x256x128 schedule (no tunable knobs), so the
only thing tuning produces is the ``use_hip`` column: for each shape, is the HIP
tile faster than the vLLM Triton ``dot_scaled`` grouped-GEMM baseline?

Output CSV (consumed by ``aiter/ops/grouped_gemm_mxfp8.py`` once the TODO loader
is wired up)::

    gfx, K, N, M, use_hip, hip_us, triton_us, speedup

Key design (and why it is fast):
  * **TP** only changes ``(K, N)`` of the two MoE GEMMs, so we enumerate the
    real per-TP ``(K, N)`` pairs directly.
  * **EP** only changes ``G`` (local experts). The HIP and Triton kernels both
    process experts independently, so the HIP/Triton *ratio* is ~invariant to G.
    We therefore key the table on ``(K, N, M_per_expert)`` and tune with a small
    representative ``G`` (``--g``). Use ``--verify-ep`` to sanity-check that the
    crossover does not move across G before trusting this assumption.

Run (MI355X / gfx950, vLLM + ROCm Triton installed)::

    export PYTORCH_ROCM_ARCH=gfx950
    python op_tests/tune_grouped_gemm_mxfp8.py \
        --out aiter/configs/grouped_gemm_mxfp8_tuned.csv --markdown

Speed knobs: ``--iters``/``--warmup`` (lower = faster, noisier), ``--m-grid``
(restrict per-expert token buckets), ``--tp``/``--gemm`` (restrict shapes you
actually deploy), ``--g`` (representative expert count).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

import torch
import triton
import triton.language as tl

from aiter.ops.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from aiter.ops.triton.quant.quant import dynamic_mxfp8_quant

# MiniMax-M3 dims; override via CLI if you tune a different model.
DEFAULT_HIDDEN = 6144  # H
DEFAULT_INTERMEDIATE = 3072  # I (per expert)
DEFAULT_EXPERTS = 128  # E (global) — MiniMax-M3: gate.weight=[128,6144], experts 0..127

TRITON_BLOCK_M = 64  # vLLM mxfp8_native_moe default
TRITON_BLOCK_N = 128
TRITON_BLOCK_K = 128


# ─────────────────────────── Triton baseline ───────────────────────────
# Verbatim grouped-GEMM kernel from vLLM mxfp8_native_moe (dot_scaled path),
# inlined so the tuner does not depend on a vLLM import for the kernel itself.
@triton.jit
def _mxfp8_grouped_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_post = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post:
        return

    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)

    a_ptrs = a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = (
        a_scale_ptr + offs_token[:, None] * stride_asm + offs_sk[None, :] * stride_ask
    )
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )
    bs_ptrs = (
        b_scale_ptr
        + off_e * stride_bse
        + offs_n[:, None] * stride_bsn
        + offs_sk[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
        asc = tl.load(as_ptrs, mask=token_mask[:, None], other=0)
        bsc = tl.load(bs_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e4m3")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
        bs_ptrs += (BLOCK_K // 32) * stride_bsk

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


def _triton_grouped_gemm(
    a_q,
    a_scale_u8,
    w,
    w_scale_u8,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens,
    out_dtype,
):
    E, N, K = w.shape
    out = torch.empty((num_valid_tokens, N), dtype=out_dtype, device=a_q.device)
    grid = (
        triton.cdiv(sorted_token_ids.shape[0], TRITON_BLOCK_M),
        triton.cdiv(N, TRITON_BLOCK_N),
    )
    _mxfp8_grouped_gemm_kernel[grid](
        a_q,
        a_scale_u8,
        w,
        w_scale_u8,
        out,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        num_valid_tokens,
        a_q.stride(0),
        a_q.stride(1),
        a_scale_u8.stride(0),
        a_scale_u8.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale_u8.stride(0),
        w_scale_u8.stride(1),
        w_scale_u8.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=TRITON_BLOCK_M,
        BLOCK_N=TRITON_BLOCK_N,
        BLOCK_K=TRITON_BLOCK_K,
        num_warps=8,
    )
    return out


# ─────────────────────────── shapes & metadata ───────────────────────────
@dataclass(frozen=True)
class Case:
    name: str  # gemm1_w13 | gemm2_w2
    tp: int
    K: int
    N: int


def enumerate_cases(hidden: int, inter: int, tps, gemms) -> list[Case]:
    cases: list[Case] = []
    for tp in tps:
        if "gemm1" in gemms:
            cases.append(Case("gemm1_w13", tp, K=hidden, N=2 * inter // tp))
        if "gemm2" in gemms:
            cases.append(Case("gemm2_w2", tp, K=inter // tp, N=hidden))
    # Drop shapes outside the kernel envelope.
    return [c for c in cases if c.K >= 384 and c.K % 32 == 0 and c.N % 16 == 0]


def _build_hip_metadata(m_per_expert: int, n: int, g: int, device: str):
    """G experts, each with the same m_per_expert rows (multiple of 16)."""
    tiles_n = (n + 255) // 256
    tiles_m = (m_per_expert + 255) // 256
    group_offs = [0]
    tile_offs = [0]
    b2e: list[int] = []
    for gi in range(g):
        group_offs.append(group_offs[-1] + m_per_expert)
        tile_offs.append(tile_offs[-1] + tiles_m * tiles_n)
        b2e.extend([gi] * (tiles_m * tiles_n))
    return (
        torch.tensor(group_offs, dtype=torch.int64, device=device),
        torch.tensor(b2e, dtype=torch.int32, device=device),
        torch.tensor(tile_offs, dtype=torch.int32, device=device),
    )


def _build_triton_metadata(m_per_expert: int, g: int, device: str):
    """moe_align-style layout for identical per-expert token distribution."""
    m_total = m_per_expert * g
    blocks_per_expert = (m_per_expert + TRITON_BLOCK_M - 1) // TRITON_BLOCK_M
    sorted_ids: list[int] = []
    expert_ids: list[int] = []
    for gi in range(g):
        base = gi * m_per_expert
        toks = list(range(base, base + m_per_expert))
        pad = blocks_per_expert * TRITON_BLOCK_M - m_per_expert
        toks.extend([m_total] * pad)  # sentinel >= num_valid -> masked out
        sorted_ids.extend(toks)
        expert_ids.extend([gi] * blocks_per_expert)
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=device),
        torch.tensor(expert_ids, dtype=torch.int32, device=device),
        torch.tensor([len(sorted_ids)], dtype=torch.int32, device=device),
        m_total,
    )


def _prepare(m_per_expert: int, n: int, k: int, g: int, device: str, seed: int):
    torch.manual_seed(seed)
    m_total = m_per_expert * g
    a_bf16 = (
        torch.randn(m_total, k, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    a_q, a_s_u8 = dynamic_mxfp8_quant(a_bf16)
    # Quantize B per-expert: a single dynamic_mxfp8_quant over [G*N, K] overflows
    # the Triton kernel's int32 row offset once (G*N)*K > 2^31 (the "Memory
    # access fault" at large G/N). Per-expert calls keep row offsets tiny.
    w_chunks, ws_chunks = [], []
    for _ in range(g):
        wb = (
            torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
        ).contiguous()
        wq, ws = dynamic_mxfp8_quant(wb)
        w_chunks.append(wq)
        ws_chunks.append(ws)
    w = torch.stack(w_chunks, 0).contiguous()
    w_s_u8 = torch.stack(ws_chunks, 0).contiguous()
    return a_q, a_s_u8, w, w_s_u8


# ─────────────────────────── timing ───────────────────────────
def _median_us(fn, warmup: int, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return float(statistics.median(samples))


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(0)
    arch = str(getattr(p, "gcn_arch_name", None) or getattr(p, "gcnArchName", "") or "")
    return "gfx950" in arch.lower()


def _gfx_name() -> str:
    p = torch.cuda.get_device_properties(0)
    return str(
        getattr(p, "gcn_arch_name", None) or getattr(p, "gcnArchName", "") or ""
    ).split(":")[0]


# ─────────────────────────── sweep ───────────────────────────
def bench_point(
    case: Case,
    m_per_expert: int,
    g: int,
    device: str,
    seed: int,
    warmup: int,
    iters: int,
    margin: float,
    ep: int = 0,
) -> dict:
    a_q, a_s_u8, w, w_s_u8 = _prepare(m_per_expert, case.N, case.K, g, device, seed)
    a_scale_e8 = a_s_u8.view(torch.float8_e8m0fnu)
    w_scale_e8 = w_s_u8.view(torch.float8_e8m0fnu)
    go, b2e, toff = _build_hip_metadata(m_per_expert, case.N, g, device)
    sid, eid, npp, m_total = _build_triton_metadata(m_per_expert, g, device)

    def hip_fn():
        return grouped_gemm_mxfp8_hip_fwd(
            a_q, w, a_scale_e8, w_scale_e8, go, b2e, toff, torch.bfloat16
        )

    def tri_fn():
        return _triton_grouped_gemm(
            a_q, a_s_u8, w, w_s_u8, sid, eid, npp, m_total, torch.bfloat16
        )

    hip_us = _median_us(hip_fn, warmup, iters)
    tri_us = _median_us(tri_fn, warmup, iters)
    use_hip = int(hip_us < tri_us * (1.0 - margin))
    return {
        "gfx": _gfx_name(),
        "case": case.name,
        "tp": case.tp,
        "ep": ep,
        "G": g,
        "K": case.K,
        "N": case.N,
        "M": m_per_expert,
        "use_hip": use_hip,
        "hip_us": round(hip_us, 2),
        "triton_us": round(tri_us, 2),
        "speedup": round(tri_us / hip_us, 3) if hip_us > 0 else 0.0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="aiter/configs/grouped_gemm_mxfp8_tuned.csv")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--inter", type=int, default=DEFAULT_INTERMEDIATE)
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument(
        "--ep",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="expert-parallel degrees; local experts G = experts // EP",
    )
    p.add_argument(
        "--experts",
        type=int,
        default=DEFAULT_EXPERTS,
        help="global expert count E (G = E // EP)",
    )
    p.add_argument("--gemm", nargs="+", default=["gemm1", "gemm2"])
    p.add_argument(
        "--m-grid", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    p.add_argument(
        "--max-mtotal",
        type=int,
        default=32768,
        help="skip points where G*M exceeds this (memory/realism cap)",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to sleep before each point (let the GPU cool / "
        "stabilize clocks; pair with `rocm-smi --setperflevel high`)",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="HIP must be faster by this fraction to set use_hip=1",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--verify-ep",
        type=int,
        nargs="*",
        default=None,
        help="re-time one (gemm1 TP4) shape at these G values to check G-invariance",
    )
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args(argv)

    if not torch.cuda.is_available() or not _is_gfx950():
        print("Need gfx950 + CUDA/HIP.", file=sys.stderr)
        return 1

    device = "cuda"
    cases = enumerate_cases(args.hidden, args.inter, args.tp, args.gemm)
    eps = sorted(set(args.ep))
    grid_m = [m for m in args.m_grid if m % 16 == 0]
    rows: list[dict] = []
    # Pre-count valid (case, ep, m) points after the M_total cap.
    plan = [
        (case, ep, args.experts // ep, m)
        for case in cases
        for ep in eps
        for m in grid_m
        if args.experts % ep == 0 and (args.experts // ep) * m <= args.max_mtotal
    ]
    total = len(plan)
    for done, (case, ep, g, m) in enumerate(plan, 1):
        if args.cooldown > 0:
            torch.cuda.synchronize()
            time.sleep(args.cooldown)
        rec = bench_point(
            case, m, g, device, args.seed, args.warmup, args.iters, args.margin, ep=ep
        )
        rows.append(rec)
        print(
            f"[{done}/{total}] {case.name} TP{case.tp} EP{ep} G={g} "
            f"K={case.K} N={case.N} M={m}: hip={rec['hip_us']}us "
            f"triton={rec['triton_us']}us use_hip={rec['use_hip']} ({rec['speedup']}x)"
        )

    if args.verify_ep:
        print("\n--- EP invariance check (gemm1 TP4) ---")
        ref = next((c for c in cases if c.name == "gemm1_w13" and c.tp == 4), cases[0])
        for g in args.verify_ep:
            for m in (64, 512):
                if g * m > args.max_mtotal:
                    print(f"  G={g} M={m}: skipped (G*M={g * m} > --max-mtotal)")
                    continue
                r = bench_point(
                    ref, m, g, device, args.seed, args.warmup, args.iters, args.margin
                )
                print(f"  G={g} M={m}: use_hip={r['use_hip']} ({r['speedup']}x)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "gfx",
        "case",
        "tp",
        "ep",
        "G",
        "K",
        "N",
        "M",
        "use_hip",
        "hip_us",
        "triton_us",
        "speedup",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)")

    if args.markdown:
        print(
            "\n| case | TP | EP | G | K | N | M | use_hip | HIP µs | Triton µs | speedup |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|")
        for r in rows:
            print(
                f"| {r['case']} | {r['tp']} | {r['ep']} | {r['G']} | {r['K']} | {r['N']} | "
                f"{r['M']} | {r['use_hip']} | {r['hip_us']} | {r['triton_us']} | {r['speedup']}x |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
