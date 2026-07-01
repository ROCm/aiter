# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for aiter's OPUS-based dense GQA flash attention (D=128, bf16, gfx950).

Validates ``fmha_fwd_hd128_bf16_opus`` against an explicit PyTorch reference (fp32-upcast
scaled-dot-product attention with grouped-query head broadcast), causal and
non-causal, across aligned and non-aligned sequence lengths.

CLI examples (from the aiter source tree)::

    PYTHONPATH=. python3 op_tests/test_fmha_fwd_hd128_bf16_opus.py
    PYTHONPATH=. python3 op_tests/test_fmha_fwd_hd128_bf16_opus.py -n 1023 4097 --causal --no-verify
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from typing import Optional

import pandas as pd
import pytest
import torch

import aiter  # noqa: F401  (registers the top-level export)
from aiter.ops.fmha_fwd_hd128_bf16_opus import fmha_fwd_hd128_bf16_opus
from aiter.test_common import benchmark, checkAllclose, perftest


def _skip(reason: str) -> bool:
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(reason)
    print(f"SKIP: {reason}")
    return True


def _get_gpu_arch() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        if hasattr(props, "gcnArchName"):
            arch_name = props.gcnArchName
            return arch_name.split(":")[0] if ":" in arch_name else arch_name
    except (AttributeError, RuntimeError):
        pass
    return None


def _skip_if_unsupported(d: int) -> bool:
    if not torch.cuda.is_available():
        return _skip("CUDA/HIP device not available")
    arch = _get_gpu_arch()
    if arch != "gfx950":
        return _skip(f"fmha_fwd_hd128_bf16_opus requires gfx950, found {arch}")
    if d != 128:
        return _skip(f"Only D=128 is compiled, requested D={d}")
    return False


# ---------------------------------------------------------------------------
# PyTorch reference: fp32-upcast SDPA with GQA head broadcast.
# ---------------------------------------------------------------------------


def _ref_gqa(
    q: torch.Tensor,  # [B, N, H, D]
    k: torch.Tensor,  # [B, N, H_KV, D]
    v: torch.Tensor,  # [B, N, H_KV, D]
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    b, n, h, d = q.shape
    h_kv = k.shape[2]
    group = h // h_kv

    qf = q.to(torch.float32).permute(0, 2, 1, 3)  # [B, H, N, D]
    kf = k.to(torch.float32).repeat_interleave(group, dim=2).permute(0, 2, 1, 3)
    vf = v.to(torch.float32).repeat_interleave(group, dim=2).permute(0, 2, 1, 3)

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_scale  # [B, H, N, N]
    if causal:
        idx = torch.arange(n, device=q.device)
        mask = idx[None, :] > idx[:, None]  # True where key j > query i
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vf)  # [B, H, N, D]
    return o.permute(0, 2, 1, 3).to(q.dtype)  # [B, N, H, D]


def _make_inputs(b, n, h, h_kv, d, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(b, n, h, d, generator=g) * 0.5).to(device, dtype=torch.bfloat16)
    k = (torch.randn(b, n, h_kv, d, generator=g) * 0.5).to(device, dtype=torch.bfloat16)
    v = (torch.randn(b, n, h_kv, d, generator=g) * 0.5).to(device, dtype=torch.bfloat16)
    return q, k, v


@perftest()
def _profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


@benchmark()
def run_fmha_fwd_hd128_bf16_opus(
    b: int,
    n: int,
    h: int,
    h_kv: int,
    d: int,
    causal: bool,
    *,
    seed: int = 0,
    verify: bool = True,
    bench: bool = True,
) -> Optional[dict]:
    if _skip_if_unsupported(d=d):
        return None

    device = torch.device("cuda", 0)
    q, k, v = _make_inputs(b, n, h, h_kv, d, device, seed=seed)
    softmax_scale = 1.0 / math.sqrt(d)

    row: dict = {}
    if verify:
        ref = _ref_gqa(q, k, v, causal, softmax_scale)
        got = fmha_fwd_hd128_bf16_opus(
            q, k, v, causal=causal, softmax_scale=softmax_scale
        )
        max_abs = (got.float() - ref.float()).abs().max().item()
        row["max_abs_err"] = round(float(max_abs), 5)
        checkAllclose(
            got,
            ref,
            rtol=5e-2,
            atol=5e-2,
            msg=f"[B={b} N={n} H={h} H_KV={h_kv} D={d} causal={causal}]",
        )

    if bench:
        _, lat_us = _profile_func(
            fmha_fwd_hd128_bf16_opus,
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        flops = 4.0 * b * h * n * n * d / (2.0 if causal else 1.0)
        row["latency_us"] = round(float(lat_us), 2)
        row["TFLOPS"] = round(float(flops / max(lat_us * 1e-6, 1e-12) / 1e12), 2)

    return row


# ---------------------------------------------------------------------------
# pytest sweep (CI) — a few shapes incl. non-aligned N, causal + non-causal.
# ---------------------------------------------------------------------------

_PYTEST_SHAPES = [
    # (B, N, H, H_KV)
    (1, 128, 8, 2),
    (2, 100, 16, 4),
    (1, 1023, 8, 8),
    (1, 257, 16, 1),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "b,n,h,h_kv",
    _PYTEST_SHAPES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_fmha_fwd_hd128_bf16_opus(b, n, h, h_kv, causal):
    run_fmha_fwd_hd128_bf16_opus(
        b=b, n=n, h=h, h_kv=h_kv, d=128, causal=causal, verify=True, bench=False
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="fmha_fwd_hd128_bf16_opus correctness + benchmark driver.",
)
parser.add_argument("-b", "--batch", type=int, nargs="*", default=[2])
parser.add_argument("--h_q", type=int, nargs="*", default=[16])
parser.add_argument("--h_kv", type=int, nargs="*", default=[4])
parser.add_argument(
    "-n",
    "--n_tokens",
    type=int,
    nargs="*",
    default=[64, 100, 127, 128, 200, 1000, 1023, 4096, 4097],
)
parser.add_argument("-d", "--head_dim", type=int, default=128)
parser.add_argument("--causal", action="store_true")
parser.add_argument("--no-causal", dest="causal", action="store_false")
parser.set_defaults(causal=None)
parser.add_argument("--no-verify", action="store_true")
parser.add_argument("--no-bench", action="store_true")
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    causal_modes = [True, False] if args.causal is None else [args.causal]

    rows = []
    for b, h, h_kv, n, causal in itertools.product(
        args.batch, args.h_q, args.h_kv, args.n_tokens, causal_modes
    ):
        row = run_fmha_fwd_hd128_bf16_opus(
            b=b,
            n=n,
            h=h,
            h_kv=h_kv,
            d=args.head_dim,
            causal=causal,
            seed=args.seed,
            verify=not args.no_verify,
            bench=not args.no_bench,
        )
        if row:
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        drop_cols = [c for c in ("verify", "bench", "seed") if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        print()
        print(df.to_string(index=False))
    sys.exit(0)
