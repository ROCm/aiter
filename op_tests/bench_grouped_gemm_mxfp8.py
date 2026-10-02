# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
r"""Micro-benchmark ``grouped_gemm_mxfp8_hip_fwd`` for PR tables (gfx950).

This is **not** the same script as ``bench_smallm_mxfp8.py`` (decode dense vs
Triton ``dot_scaled``). Here we benchmark **prefill-style grouped GEMM**
(multi-expert G>=1 cases with valid ``group_offs`` / ``tile_offs`` /
``block_to_expert``; defaults use G=8).

How to generate a PR-style table
--------------------------------
1. **Machine**: MI355X / gfx950, ROCm PyTorch, ``export PYTORCH_ROCM_ARCH=gfx950``.
2. **Install**: ``cd aiter && pip install -e .``.
3. **Sweep HIP timings + optional reference**::

     python op_tests/bench_grouped_gemm_mxfp8.py --csv grouped_mxfp8_bench.csv --ref-torch --markdown

4. **Open the CSV** in Excel/Sheets; if you later add a **Triton baseline** column
   (same bits/layout as vLLM), compute ``speedup = triton_us / hip_us`` per cell.
5. **Paste** the ``--markdown`` block into the PR (edit column headers to match
   your real baselines).

**Reference column (``--ref-torch``)**
MX dequant (same e4m3+e8m0 bits) + ``torch.matmul`` in fp32 — *much slower* than
HIP, but reproducible without vLLM. Use the ratio as “vs accurate matmul”, **not**
as “vs Triton” unless you wire Triton yourself.

**Default rows**
Multi-expert **G=8** shapes that satisfy kernel preflight (``K>=384``,
``K%32==0``, ``N%16==0``, ``M%16==0``, and per-expert ``M_g%16==0`` for the
preshuffle launch). ``M_total`` is split evenly across the G experts. Extend
``DEFAULT_BENCH_CASES`` for more rows.

**vs decode PR #3783 table**
That table comes from ``bench_smallm_mxfp8.py`` + **Triton dot_scaled** baseline
inside aiter. To claim the same style of “× vs Triton”, you must add a Triton
kernel that matches **this** grouped layout (or call into vLLM) — out of scope
for this file; the CSV/Markdown here is the mechanical scaffold.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

import torch

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

from aiter.ops.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from aiter.ops.triton.quant.quant import dynamic_mxfp8_quant
from aiter.utility.fp4_utils import e8m0_to_f32

# (label, M_total, N, K, G) — M_total is split evenly across G experts, so
# M_total must be divisible by G and each per-expert share by 16. G=8 mirrors a
# realistic MoE grouped GEMM (not a degenerate single-expert / plain GEMM).
DEFAULT_BENCH_CASES: tuple[tuple[str, int, int, int, int], ...] = (
    ("g8_smoke", 2048, 256, 384, 8),
    ("g8_m2048_n512_k1024", 2048, 512, 1024, 8),
    ("g8_m4096_n512_k1024", 4096, 512, 1024, 8),
    ("g8_m8192_n1024_k2048", 8192, 1024, 2048, 8),
)


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(0)
    arch = str(getattr(p, "gcn_arch_name", None) or getattr(p, "gcnArchName", "") or "")
    return "gfx950" in arch.lower()


def _mxfp8_dequant_blocked(
    x_fp8: torch.Tensor,
    scale_fp8: torch.Tensor,
    *,
    block: int = 32,
) -> torch.Tensor:
    *lead, k = x_fp8.shape
    nb = k // block
    s_f32 = e8m0_to_f32(scale_fp8.view(torch.uint8)).view(*lead, nb, 1)
    x_f = x_fp8.reshape(*lead, nb, block).float()
    return (x_f * s_f32).reshape(*lead, k)


def _grouped_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    per_expert_rows: list[int],
) -> torch.Tensor:
    """Grouped NT GEMM reference (G>=1): per-expert dequant matmul, concatenated."""
    outs = []
    start = 0
    for gi, rows in enumerate(per_expert_rows):
        a_dq = _mxfp8_dequant_blocked(a[start : start + rows], a_scale[start : start + rows])
        b_dq = _mxfp8_dequant_blocked(b[gi], b_scale[gi])
        outs.append(a_dq @ b_dq.transpose(0, 1))
        start += rows
    return torch.cat(outs, dim=0)


def _tile_metadata(
    per_expert_rows: list[int], n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat tile-grid prefix sums for arbitrary G.

    tiles_per_expert = ceil(M_g / 256) * ceil(N / 256); ``tile_offs`` is their
    prefix sum and ``block_to_expert`` labels every flat tile with its expert.
    """
    tiles_n = (n + 255) // 256
    tile_offs = [0]
    block_to_expert: list[int] = []
    for gi, rows in enumerate(per_expert_rows):
        per_expert_tiles = ((rows + 255) // 256) * tiles_n
        block_to_expert.extend([gi] * per_expert_tiles)
        tile_offs.append(tile_offs[-1] + per_expert_tiles)
    return (
        torch.tensor(block_to_expert, dtype=torch.int32, device="cuda"),
        torch.tensor(tile_offs, dtype=torch.int32, device="cuda"),
    )


def _median_gpu_us(
    fn,
    *,
    warmup: int,
    iters: int,
) -> float:
    """Median elapsed time in microseconds using CUDA events."""
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
        samples.append(start.elapsed_time(end) * 1000.0)  # ms -> µs
    return float(statistics.median(samples))


def _prepare_tensors(
    m_total: int,
    n: int,
    k: int,
    g: int,
    device: str,
    seed: int,
) -> tuple:
    torch.manual_seed(seed)
    a_bf16 = (
        torch.randn(m_total, k, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    a, a_s_u8 = dynamic_mxfp8_quant(a_bf16)
    a_scale = a_s_u8.view(torch.float8_e8m0fnu)
    b_bf16 = (
        torch.randn(g * n, k, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    b_flat, b_s_u8 = dynamic_mxfp8_quant(b_bf16)
    b = b_flat.view(g, n, k).contiguous()
    b_scale = b_s_u8.view(g, n, k // 32).contiguous().view(torch.float8_e8m0fnu)
    per_expert_rows = [m_total // g] * g
    group_offs = torch.tensor(
        [(m_total // g) * i for i in range(g + 1)], device=device, dtype=torch.int64
    )
    block_to_expert, tile_offs = _tile_metadata(per_expert_rows, n)
    return a, b, a_scale, b_scale, group_offs, block_to_expert, tile_offs


def _validate_case(m_total: int, n: int, k: int, g: int) -> None:
    if g < 1 or m_total % g != 0:
        raise ValueError(f"M_total={m_total} must be divisible by G={g}")
    per_expert = m_total // g
    if k < 384 or k % 32 != 0 or n % 16 != 0 or m_total % 16 != 0 or per_expert % 16 != 0:
        raise ValueError(
            f"illegal shape M={m_total} N={n} K={k} G={g} (per-expert={per_expert}) "
            "for kernel preflight (K>=384, K%32==0, N%16==0, M%16==0, M_g%16==0)"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=30, help="warmup iters (default 30)")
    p.add_argument("--iters", type=int, default=200, help="timed iters (default 200)")
    p.add_argument("--csv", type=str, default="", help="write results to this CSV path")
    p.add_argument(
        "--ref-torch",
        action="store_true",
        help="also time MX-dequant + torch.matmul ref (fp32) for a ratio column",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="print a GitHub-flavored markdown table to stdout",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for quant inputs")
    args = p.parse_args(argv)

    if not torch.cuda.is_available() or not _is_gfx950():
        print("Need gfx950 + CUDA/HIP.", file=sys.stderr)
        return 1

    device = "cuda"
    rows: list[dict[str, float | str]] = []

    for label, m_total, n, k, g in DEFAULT_BENCH_CASES:
        _validate_case(m_total, n, k, g)
        a, b, a_scale, b_scale, go, b2e, toff = _prepare_tensors(
            m_total, n, k, g, device, args.seed
        )

        def hip_fn():
            return grouped_gemm_mxfp8_hip_fwd(
                a, b, a_scale, b_scale, go, b2e, toff, torch.bfloat16
            )

        hip_us = _median_gpu_us(hip_fn, warmup=args.warmup, iters=args.iters)
        rec: dict[str, float | str] = {
            "label": label,
            "M": m_total,
            "N": n,
            "K": k,
            "G": g,
            "hip_median_us": round(hip_us, 2),
        }
        if args.ref_torch:

            per_expert_rows = [m_total // g] * g

            def ref_fn(per_expert_rows=per_expert_rows):
                return _grouped_ref(a, b, a_scale, b_scale, per_expert_rows)

            ref_us = _median_gpu_us(
                ref_fn, warmup=max(3, args.warmup // 5), iters=args.iters
            )
            rec["ref_torch_median_us"] = round(ref_us, 2)
            rec["ref_over_hip"] = round(ref_us / hip_us, 2) if hip_us > 0 else 0.0
        rows.append(rec)

    fieldnames = list(rows[0].keys())
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv}")

    for r in rows:
        extra = ""
        if args.ref_torch:
            extra = f" ref_us={r['ref_torch_median_us']} ref/hip={r['ref_over_hip']}×"
        print(
            f"{r['label']}: M={r['M']} N={r['N']} K={r['K']} G={r['G']} "
            f"hip_median_us={r['hip_median_us']}{extra}"
        )

    if args.markdown:
        print()
        hdr = "| case | M | N | K | G | HIP µs (median) |"
        sep = "|---|---:|---:|---:|---:|---:|"
        if args.ref_torch:
            hdr += " ref torch µs | ref/HIP |"
            sep += "---:|---:|"
        print(hdr)
        print(sep)
        for r in rows:
            line = (
                f"| {r['label']} | {r['M']} | {r['N']} | {r['K']} | {r['G']} | "
                f"{r['hip_median_us']} |"
            )
            if args.ref_torch:
                line += f" {r['ref_torch_median_us']} | {r['ref_over_hip']}× |"
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
