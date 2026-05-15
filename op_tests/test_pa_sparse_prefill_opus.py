# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for aiter's OPUS-based sparse paged prefill attention.

We validate ``pa_sparse_prefill_opus`` (the gfx950 OPUS kernel for
two-region sparse paged prefill attention) against an explicit PyTorch
reference: per-token online-softmax + per-head sink.

The same harness drives:

* A pytest-parametrised correctness sweep (CI).
* A standalone CLI for single-point runs with optional benchmark and
  TFLOPS reporting, mirroring the style of ``op_tests/test_batch_prefill.py``.

Example CLI usage (inside the aiter source tree)::

    # bf16, sparse CSR, default 1024 tokens / 128 heads, with bench
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_opus.py

    # dense CSR for both prefix and extend
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_opus.py --dense

    # custom shape, both bf16 and fp16, no correctness check
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_opus.py \\
        -n 256 1024 --h_q 128 --dtype bf16 fp16 --no-verify

Reference design notes (Q&A):

* Why does the reference cast q/k/v to fp32? The GPU kernel accumulates in
  fp32 (``D_ACC = float``). Computing the reference in bf16/fp16 would
  let softmax / accumulation errors dominate the comparison; fp32 inside
  the ref keeps the diff focused on the kernel's intermediate casts and
  is the same convention used by ``test_batch_prefill.py``,
  ``aiter/test_mha_common.py::attention_ref(upcast=True)`` etc.
* Why are ``kv_indptr`` / ``kv_indices`` cast to int64 in the ref? Only
  for the PyTorch ``index_select`` call -- the API requires Long. The
  kernel's ABI still consumes int32 indices on the GPU side; the cast
  happens entirely on CPU/GPU but on the ref path only.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from typing import Optional, Tuple

import pandas as pd
import pytest
import torch

import aiter  # noqa: F401  (registers the top-level export)
from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus
from aiter.test_common import benchmark, checkAllclose, perftest


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


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
        return _skip(f"pa_sparse_prefill_opus requires gfx950, found {arch}")
    if d != 512:
        return _skip(f"Only D=512 is compiled, requested D={d}")
    return False


# ---------------------------------------------------------------------------
# PyTorch reference: per-token online-softmax + per-head sink.
# ---------------------------------------------------------------------------


def _ref_pa_sparse_prefill_opus(
    q: torch.Tensor,                  # [N, H, D]
    unified_kv: torch.Tensor,         # [total_pages, D]
    kv_indices_prefix: torch.Tensor,  # [nnz_prefix] int32
    kv_indptr_prefix: torch.Tensor,   # [N+1] int32
    kv: torch.Tensor,                 # [total_tokens, D]
    kv_indices_extend: torch.Tensor,  # [nnz_extend] int32
    kv_indptr_extend: torch.Tensor,   # [N+1] int32
    attn_sink: torch.Tensor,          # [H] fp32
    softmax_scale: float,
) -> torch.Tensor:
    """Per-token reference. Matches the GPU kernel: online-softmax over
    ``concat(prefix, extend)`` with a per-head sink contributing only to the
    denominator.

    Computation is done in fp32 to mirror the kernel's fp32 accumulator;
    ``index_select`` requires Long indices on the PyTorch side.
    """
    n, h, d = q.shape
    out = torch.zeros_like(q)

    q_f32 = q.to(torch.float32)
    ukv_f32 = unified_kv.to(torch.float32)
    kv_f32 = kv.to(torch.float32)
    sink_f32 = attn_sink.to(torch.float32)

    p_indptr = kv_indptr_prefix.to(torch.int64).cpu().tolist()
    e_indptr = kv_indptr_extend.to(torch.int64).cpu().tolist()
    p_idx = kv_indices_prefix.to(torch.int64)
    e_idx = kv_indices_extend.to(torch.int64)

    for i in range(n):
        ps, pe = p_indptr[i], p_indptr[i + 1]
        es, ee = e_indptr[i], e_indptr[i + 1]
        rows = []
        if pe > ps:
            rows.append(ukv_f32.index_select(0, p_idx[ps:pe]))
        if ee > es:
            rows.append(kv_f32.index_select(0, e_idx[es:ee]))
        if not rows:
            # All-empty CSR row: numerator is 0, denom is exp(sink); output 0.
            continue
        kv_rows = torch.cat(rows, dim=0)  # [nnz_i, D]
        scores = q_f32[i] @ kv_rows.t() * softmax_scale  # [H, nnz_i]
        sink_col = sink_f32.unsqueeze(1)  # [H, 1]
        scores_with_sink = torch.cat([scores, sink_col], dim=1)  # [H, nnz_i+1]
        max_score = scores_with_sink.amax(dim=1, keepdim=True)
        exp_scores = torch.exp(scores - max_score)
        exp_sink = torch.exp(sink_col - max_score)
        denom = exp_scores.sum(dim=1, keepdim=True) + exp_sink
        p = exp_scores / denom
        out[i] = (p @ kv_rows).to(q.dtype)

    return out


# ---------------------------------------------------------------------------
# CSR index generators
# ---------------------------------------------------------------------------


def _random_csr(
    n: int,
    total_rows: int,
    *,
    max_nnz_per_row: Optional[int] = None,
    allow_empty: bool = True,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    cap = max_nnz_per_row if max_nnz_per_row is not None else total_rows
    cap = min(cap, total_rows)
    lo = 0 if allow_empty else 1

    lens = torch.randint(lo, cap + 1, (n,), generator=g, dtype=torch.int32)
    indptr = torch.zeros(n + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, dim=0)
    nnz = int(indptr[-1].item())

    indices = torch.empty(nnz, dtype=torch.int32)
    for i in range(n):
        s, e = int(indptr[i].item()), int(indptr[i + 1].item())
        row_len = e - s
        if row_len == 0:
            continue
        perm = torch.randperm(total_rows, generator=g)[:row_len]
        indices[s:e] = perm.to(torch.int32)
    return indptr.to(device), indices.to(device)


def _dense_csr(
    n: int, total_rows: int, *, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    indptr = torch.arange(
        0, (n + 1) * total_rows, total_rows, dtype=torch.int32
    )
    indices = torch.arange(total_rows, dtype=torch.int32).repeat(n)
    return indptr.to(device), indices.to(device)


def _empty_csr(
    n: int, *, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(n + 1, dtype=torch.int32, device=device),
        torch.zeros(0, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------

# Single sparsity knob applied symmetrically to both prefix and extend CSRs.
# 'empty' is only exercised by the pytest sweep (CLI uses --dense flag).
_MODES = ("sparse", "dense", "empty")


def _make_inputs(
    n: int,
    h: int,
    d: int,
    total_pages: int,
    total_tokens: int,
    dtype: torch.dtype,
    *,
    mode: str = "sparse",
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict:
    assert mode in _MODES
    torch.manual_seed(seed)
    device = torch.device(device)

    q = (torch.randn(n, h, d, device=device, dtype=torch.float32) * 0.5).to(dtype)
    unified_kv = (
        torch.randn(total_pages, d, device=device, dtype=torch.float32) * 0.5
    ).to(dtype)
    kv = (
        torch.randn(total_tokens, d, device=device, dtype=torch.float32) * 0.5
    ).to(dtype)
    attn_sink = (torch.randn(h, device=device, dtype=torch.float32) * 0.25)

    def _csr(total_rows: int, seed_offset: int):
        if mode == "sparse":
            return _random_csr(
                n, total_rows,
                max_nnz_per_row=min(total_rows, 96),
                device=device, seed=seed * 2 + seed_offset,
            )
        if mode == "dense":
            return _dense_csr(n, total_rows, device=device)
        return _empty_csr(n, device=device)

    ip_p, ix_p = _csr(total_pages, 1)
    ip_e, ix_e = _csr(total_tokens, 2)

    return dict(
        q=q,
        unified_kv=unified_kv,
        kv_indices_prefix=ix_p,
        kv_indptr_prefix=ip_p,
        kv=kv,
        kv_indices_extend=ix_e,
        kv_indptr_extend=ip_e,
        attn_sink=attn_sink,
    )


# ---------------------------------------------------------------------------
# perftest-wrapped kernel call (same shape as test_batch_prefill.py)
# ---------------------------------------------------------------------------


@perftest()
def _profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


def _get_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 2e-2, 2e-2  # bf16 default


# ---------------------------------------------------------------------------
# Single-case driver -- both pytest and CLI go through this.
# `@benchmark()` collects the kwargs into a row dict and merges in whatever
# this function returns, so the CLI can build a pandas DataFrame.
# ---------------------------------------------------------------------------


@benchmark()
def run_pa_sparse_prefill_opus(
    n: int,
    h: int,
    d: int,
    total_pages: int,
    total_tokens: int,
    dtype: torch.dtype,
    *,
    mode: str = "sparse",
    seed: int = 0,
    verify: bool = True,
    bench: bool = True,
) -> Optional[dict]:
    if _skip_if_unsupported(d=d):
        return None

    inputs = _make_inputs(
        n, h, d, total_pages, total_tokens, dtype,
        mode=mode, seed=seed,
    )
    softmax_scale = 1.0 / math.sqrt(d)

    nnz_p = int(inputs["kv_indices_prefix"].numel())
    nnz_e = int(inputs["kv_indices_extend"].numel())

    row: dict = {"nnz_prefix": nnz_p, "nnz_extend": nnz_e}

    if verify:
        ref = _ref_pa_sparse_prefill_opus(**inputs, softmax_scale=softmax_scale)
        got = pa_sparse_prefill_opus(**inputs, softmax_scale=softmax_scale)
        rtol, atol = _get_tolerances(dtype)
        checkAllclose(
            got, ref, rtol=rtol, atol=atol,
            msg=(
                f"[N={n} H={h} D={d} tp={total_pages} tt={total_tokens} "
                f"dtype={dtype} mode={mode}]"
            ),
        )

    if bench:
        # `@perftest()` returns (data, avg_us_per_iter).
        _, lat_us = _profile_func(
            pa_sparse_prefill_opus,
            **inputs,
            softmax_scale=softmax_scale,
        )
        # Sparse attention FLOPS: 4 * H * total_nnz * D
        total_nnz = nnz_p + nnz_e
        flops = 4.0 * h * total_nnz * d
        tflops = flops / max(lat_us * 1e-6, 1e-12) / 1e12
        row["latency_us"] = round(float(lat_us), 2)
        row["TFLOPS"] = round(float(tflops), 2)

    return row


# ---------------------------------------------------------------------------
# pytest parametrised correctness sweep (CI).
# Single sparsity knob; 'empty' is included so empty-CSR edge cases stay
# covered even though the CLI only exposes --dense.
# ---------------------------------------------------------------------------


_PYTEST_SHAPES = [
    # (N, H, total_pages, total_tokens)
    (17, 64, 256, 256),
    (64, 128, 1024, 1024),
    (256, 128, 2048, 2048),
]
_PYTEST_DTYPES = [torch.bfloat16, torch.float16]
_PYTEST_MODES = ["sparse", "dense", "empty"]


@pytest.mark.parametrize(
    "dtype", _PYTEST_DTYPES, ids=lambda d: str(d).split(".")[-1]
)
@pytest.mark.parametrize(
    "n,h,total_pages,total_tokens",
    _PYTEST_SHAPES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
@pytest.mark.parametrize("mode", _PYTEST_MODES)
def test_pa_sparse_prefill_opus(dtype, n, h, total_pages, total_tokens, mode):
    # bench=False keeps pytest fast; CLI path does the timing.
    run_pa_sparse_prefill_opus(
        n=n, h=h, d=512,
        total_pages=total_pages, total_tokens=total_tokens,
        dtype=dtype,
        mode=mode,
        seed=(hash((n, h, total_pages, total_tokens, str(dtype), mode)) & 0xFFFF),
        verify=True, bench=False,
    )


# ---------------------------------------------------------------------------
# CLI (mirrors test_batch_prefill.py style).
# ---------------------------------------------------------------------------


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "pa_sparse_prefill_opus correctness + benchmark driver.\n"
        "All list arguments are swept via itertools.product."
    ),
)
parser.add_argument(
    "-n", "--n_tokens",
    type=int, nargs="*", default=[1024],
    help="number of query tokens N (default: 1024)",
)
parser.add_argument(
    "--h_q",
    type=int, nargs="*", default=[128],
    help="number of query heads H_Q (default: 128)",
)
parser.add_argument(
    "-d", "--head_dim",
    type=int, default=512,
    help="head dim D, kernel currently only compiled for 512 (default: 512)",
)
parser.add_argument(
    "--total_pages",
    type=int, default=None,
    help="rows in unified_kv (default: matches -n)",
)
parser.add_argument(
    "--total_tokens",
    type=int, default=None,
    help="rows in extend kv (default: matches -n)",
)
parser.add_argument(
    "--dtype",
    type=str, nargs="*", default=["bf16"], choices=["bf16", "fp16"],
    help="attention dtype(s) to sweep (default: bf16)",
)
parser.add_argument(
    "--dense",
    action="store_true",
    help=(
        "use dense CSR for both prefix and extend (every token sees every "
        "page / every kv row). Default is random sparse for both."
    ),
)
parser.add_argument(
    "--no-verify",
    action="store_true",
    help="skip the PyTorch correctness check (benchmark-only mode)",
)
parser.add_argument(
    "--no-bench",
    action="store_true",
    help="skip the per-call latency benchmark",
)
parser.add_argument(
    "--seed",
    type=int, default=0,
    help="RNG seed for input + CSR generation",
)


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}


if __name__ == "__main__":
    args = parser.parse_args()
    mode = "dense" if args.dense else "sparse"

    rows = []
    for n, h, dtype_str in itertools.product(args.n_tokens, args.h_q, args.dtype):
        total_pages = args.total_pages if args.total_pages is not None else n
        total_tokens = args.total_tokens if args.total_tokens is not None else n
        row = run_pa_sparse_prefill_opus(
            n=n,
            h=h,
            d=args.head_dim,
            total_pages=total_pages,
            total_tokens=total_tokens,
            dtype=_DTYPE_MAP[dtype_str],
            mode=mode,
            seed=args.seed,
            verify=not args.no_verify,
            bench=not args.no_bench,
        )
        if row:
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        # Stringify dtype for prettier print.
        if "dtype" in df.columns:
            df["dtype"] = df["dtype"].map(
                lambda t: str(t).split(".")[-1] if hasattr(t, "is_floating_point") else str(t)
            )
        # Drop columns that don't carry signal in the default sweep.
        drop_cols = [c for c in ("verify", "bench", "seed") if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        print()
        print(df.to_string(index=False))
        sys.exit(0)
    sys.exit(0)
