# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for aiter's OPUS-based sparse paged prefill attention.

We validate both precision variants of the gfx950 OPUS two-region sparse
paged prefill kernel against explicit PyTorch references (per-token
online-softmax + per-head sink):

* ``pa_sparse_prefill_opus`` -- bf16/fp16 Q/K/V/O in a single ``D=512``
  head-dim tensor.
* ``pa_sparse_prefill_fp8_opus`` -- split-precision DSA inputs: NoPE part
  in fp8 (with embedded per-32-block E8M0 scales, 448 + 14 scales + pad =
  512 fp8 slots/row) plus a bf16 RoPE part (64), bf16 output.

A single ``prec`` axis (``"bf16"`` / ``"fp16"`` / ``"fp8"``) drives both
paths through the same shape/mode sweep, so fp8 is exercised with the same
test parameters as bf16/fp16.

The same harness drives:

* A pytest-parametrised correctness sweep (CI).
* A standalone CLI for single-point runs with optional benchmark and
  TFLOPS reporting, mirroring the style of ``op_tests/test_batch_prefill.py``.

Example CLI usage (inside the aiter source tree)::

    # default sweep: N x H_Q x total_pages x {sparse, dense} x {bf16, fp8}, with bench
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_opus.py

    # only dense CSR for both prefix and extend
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_opus.py --mode dense

    # single shape, fp8 only, no correctness check
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_opus.py \\
        -n 1024 --h_q 128 --prec fp8 --mode sparse --no-verify

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
from aiter.ops.pa_sparse_prefill_opus import (
    pa_sparse_prefill_opus,
    pa_sparse_prefill_fp8_opus,
)
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


def _skip_if_unsupported(d: int, prec: str = "fp8") -> bool:
    if not torch.cuda.is_available():
        return _skip("CUDA/HIP device not available")
    arch = _get_gpu_arch()
    if arch not in ("gfx950", "gfx1250"):
        return _skip(f"pa_sparse_prefill_opus requires gfx950/gfx1250, found {arch}")
    # Only the split-precision fp8 variant is ported to gfx1250 (wave32/WMMA);
    # the bf16/fp16 single-tensor kernel is gfx950-only.
    if arch == "gfx1250" and prec != "fp8":
        return _skip(f"gfx1250 only supports the fp8 variant, not {prec}")
    if d != 512:
        return _skip(f"Only D=512 is compiled, requested D={d}")
    return False


# ---------------------------------------------------------------------------
# PyTorch reference: per-token online-softmax + per-head sink.
# ---------------------------------------------------------------------------


def _ref_pa_sparse_prefill_opus(
    q: torch.Tensor,  # [N, H, D]
    unified_kv: torch.Tensor,  # [total_pages, D]
    kv_indices_prefix: torch.Tensor,  # [nnz_prefix] int32
    kv_indptr_prefix: torch.Tensor,  # [N+1] int32
    kv: torch.Tensor,  # [total_tokens, D]
    kv_indices_extend: torch.Tensor,  # [nnz_extend] int32
    kv_indptr_extend: torch.Tensor,  # [N+1] int32
    attn_sink: torch.Tensor,  # [H] fp32
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
# FP8 DSA packing + reference (NoPE fp8 / RoPE bf16).
#
# The NoPE stream packs, per row of 512 fp8 slots:
#   [ NoPE fp8 (448) | E8M0 block scales (14) | fp8 zero-pad (50) ]
# with one E8M0 (power-of-two) scale per 32-element NoPE block. The RoPE
# stream is a separate ``[*, 64]`` bf16 tensor. The kernel runs NoPE QK^T as
# scaled MXFP8 MFMA, RoPE QK^T and PV at bf16, and accumulates in fp32.
# ---------------------------------------------------------------------------

_FP8_D_NOPE = 448
_FP8_D_NOPE_PADDED = 512
_FP8_D_ROPE = 64
_FP8_D_HEAD = _FP8_D_NOPE + _FP8_D_ROPE  # 512
_FP8_NBLK = _FP8_D_NOPE // 32  # 14
_FP8_BLK = 32
_FP8_MAX = 448.0  # e4m3fn max normal
_FP8_KV_TILE_SIZE = 64  # KV_TILE_SIZE of the fp8 16mx1_16nx4 kernel


def _quantize_nope(real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[R, 448]`` real values into a packed ``[R, 512]`` fp8 row
    (NoPE fp8 + E8M0 block scales + zero pad) and return ``(packed_fp8, deq)``
    where ``deq`` (``[R, 448]`` fp32) is the dequantized NoPE the kernel sees.
    """
    r = real.shape[0]
    blk = real.reshape(r, _FP8_NBLK, _FP8_BLK).to(torch.float32)
    amax = blk.abs().amax(dim=-1)  # [R, NBLK]

    # Per-block E8M0 exponent chosen so the block max maps to (224, 448], i.e.
    # strictly inside the e4m3fn finite range (overflow -> NaN on cast).
    e_unbiased = torch.ceil(torch.log2(amax.clamp(min=1e-30) / _FP8_MAX)).to(
        torch.int32
    )
    e_unbiased = torch.where(amax == 0, torch.zeros_like(e_unbiased), e_unbiased)
    e_byte = (e_unbiased + 127).clamp(0, 255).to(torch.uint8)  # [R, NBLK]
    s = torch.exp2(e_unbiased.to(torch.float32)).unsqueeze(-1)  # [R, NBLK, 1]

    q = (blk / s).to(torch.float8_e4m3fn)  # [R, NBLK, BLOCK]
    deq = (q.to(torch.float32) * s).reshape(r, _FP8_D_NOPE)

    packed = torch.zeros(r, _FP8_D_NOPE_PADDED, dtype=torch.uint8, device=real.device)
    packed[:, :_FP8_D_NOPE] = q.reshape(r, _FP8_D_NOPE).view(torch.uint8)
    packed[:, _FP8_D_NOPE : _FP8_D_NOPE + _FP8_NBLK] = e_byte
    return packed.view(torch.float8_e4m3fn), deq


def _ref_pa_sparse_prefill_fp8(
    q_fp32: torch.Tensor,  # [N, H, 512] fp32 (dequant NoPE + RoPE)
    ukv_fp32: torch.Tensor,  # [total_pages, 512] fp32
    kv_fp32: torch.Tensor,  # [total_tokens, 512] fp32
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """fp8 reference: identical attention math to the bf16 ref, but operating
    on the already-dequantized ``concat(dequant_NoPE, RoPE)`` rows the kernel
    consumes (so only the kernel's bf16 intermediates / MFMA rounding differ).
    """
    n, h, _ = q_fp32.shape
    out = torch.zeros(n, h, _FP8_D_HEAD, dtype=torch.bfloat16, device=q_fp32.device)
    pp = kv_indptr_prefix.to(torch.int64).cpu().tolist()
    pe = kv_indptr_extend.to(torch.int64).cpu().tolist()
    pidx = kv_indices_prefix.to(torch.int64)
    eidx = kv_indices_extend.to(torch.int64)
    sink_f = attn_sink.to(torch.float32)

    for i in range(n):
        rows = []
        if pp[i + 1] > pp[i]:
            rows.append(ukv_fp32.index_select(0, pidx[pp[i] : pp[i + 1]]))
        if pe[i + 1] > pe[i]:
            rows.append(kv_fp32.index_select(0, eidx[pe[i] : pe[i + 1]]))
        if not rows:
            continue
        kv_rows = torch.cat(rows, dim=0)  # [nnz, 512]
        scores = q_fp32[i] @ kv_rows.t() * softmax_scale  # [H, nnz]
        sink_col = sink_f.unsqueeze(1)
        m = torch.cat([scores, sink_col], dim=1).amax(dim=1, keepdim=True)
        e_s = torch.exp(scores - m)
        e_sink = torch.exp(sink_col - m)
        denom = e_s.sum(dim=1, keepdim=True) + e_sink
        out[i] = ((e_s / denom) @ kv_rows).to(torch.bfloat16)
    return out


# ---------------------------------------------------------------------------
# CSR index generators
# ---------------------------------------------------------------------------

# Must match the KV_TILE_SIZE template default in
# csrc/include/pa_sparse_prefill_opus.h. The kernel inner loop advances the
# K/V dimension in chunks of this size, so the trailing-tile branches (full /
# half / over-tile) are most likely to break when nnz_per_row sits at one of
# these boundary values.
_KV_TILE_SIZE = 32


def _boundary_nnz(kv_tile_size: int, total_rows: int) -> list:
    """Tile-boundary nnz values seeded into the leading rows of a sparse CSR,
    mirroring gcnasm/opus_attn/sparse_paged_attn/pa_host.cc::
    init_sparse_kv_indices. Clamped into [0, total_rows]."""
    cands = [
        0,
        1,
        kv_tile_size - 1,
        kv_tile_size,
        kv_tile_size + 1,
        2 * kv_tile_size,
        2 * kv_tile_size + 1,
        total_rows,
    ]
    return [max(0, min(v, total_rows)) for v in cands]


def _random_csr(
    n: int,
    total_rows: int,
    *,
    allow_empty: bool = True,
    kv_tile_size: int = _KV_TILE_SIZE,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random CSR with deterministic tile-boundary nnz on the first rows.

    Length distribution: ``randint(0, total_rows)`` -- no artificial cap, so a
    sparse sweep can produce anything from empty rows up to nearly-dense rows.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    lo = 0 if allow_empty else 1

    lens = torch.randint(lo, total_rows + 1, (n,), generator=g, dtype=torch.int32)
    # Seed the leading rows with tile-boundary lengths -- guarantees every
    # sparse sweep exercises the kernel's full/half/over-tile branches and
    # (when allow_empty) the sink-only empty-row path.
    boundary = _boundary_nnz(kv_tile_size, total_rows)
    if not allow_empty:
        boundary = [max(b, 1) for b in boundary]
    for i, v in enumerate(boundary[:n]):
        lens[i] = v

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

    # Cheap sanity asserts (CPU-side, O(n) after generation).
    assert int(indptr[0].item()) == 0
    assert int(indptr[-1].item()) == nnz
    assert bool(torch.all(indptr[1:] >= indptr[:-1]).item())
    if nnz > 0:
        assert int(indices.min().item()) >= 0
        assert int(indices.max().item()) < total_rows

    return indptr.to(device), indices.to(device)


def _dense_csr(
    n: int, total_rows: int, *, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    indptr = torch.arange(0, (n + 1) * total_rows, total_rows, dtype=torch.int32)
    indices = torch.arange(total_rows, dtype=torch.int32).repeat(n)
    return indptr.to(device), indices.to(device)


def _empty_csr(n: int, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(n + 1, dtype=torch.int32, device=device),
        torch.zeros(0, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------

# Single sparsity knob applied symmetrically to both prefix and extend CSRs.
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
    kv = (torch.randn(total_tokens, d, device=device, dtype=torch.float32) * 0.5).to(
        dtype
    )
    attn_sink = torch.randn(h, device=device, dtype=torch.float32) * 0.25

    def _csr(total_rows: int, seed_offset: int):
        if mode == "sparse":
            return _random_csr(
                n,
                total_rows,
                device=device,
                seed=seed * 2 + seed_offset,
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


def _make_inputs_fp8(
    n: int,
    h: int,
    total_pages: int,
    total_tokens: int,
    *,
    mode: str = "sparse",
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict:
    """Build split NoPE-fp8 / RoPE-bf16 inputs plus the matching fp32
    reference rows. Returns ``{"kernel": ..., "ref": ...}`` where ``kernel``
    holds the tensors passed to ``pa_sparse_prefill_fp8_opus`` and ``ref``
    holds the dequantized ``*_fp32`` rows passed to ``_ref_pa_sparse_prefill_fp8``.
    """
    assert mode in _MODES
    torch.manual_seed(seed)
    device = torch.device(device)

    def _streams(rows: int):
        nope_fp8, deq = _quantize_nope(
            torch.randn(rows, _FP8_D_NOPE, device=device) * 0.5
        )
        rope = (torch.randn(rows, _FP8_D_ROPE, device=device) * 0.5).to(torch.bfloat16)
        row_fp32 = torch.cat([deq, rope.to(torch.float32)], dim=1)  # [rows, 512]
        return nope_fp8, rope, row_fp32

    qn, qr, q_fp32 = _streams(n * h)
    qn = qn.reshape(n, h, _FP8_D_NOPE_PADDED)
    qr = qr.reshape(n, h, _FP8_D_ROPE)
    q_fp32 = q_fp32.reshape(n, h, _FP8_D_HEAD)
    ukn, ukr, ukv_fp32 = _streams(total_pages)
    kn, kr, kv_fp32 = _streams(total_tokens)

    attn_sink = torch.randn(h, device=device, dtype=torch.float32) * 0.25

    def _csr(total_rows: int, seed_offset: int):
        if mode == "sparse":
            return _random_csr(
                n,
                total_rows,
                device=device,
                kv_tile_size=_FP8_KV_TILE_SIZE,
                seed=seed * 2 + seed_offset,
            )
        if mode == "dense":
            return _dense_csr(n, total_rows, device=device)
        return _empty_csr(n, device=device)

    ip_p, ix_p = _csr(total_pages, 1)
    ip_e, ix_e = _csr(total_tokens, 2)

    kernel = dict(
        q_nope=qn,
        q_rope=qr,
        unified_kv_nope=ukn,
        unified_kv_rope=ukr,
        kv_indices_prefix=ix_p,
        kv_indptr_prefix=ip_p,
        kv_nope=kn,
        kv_rope=kr,
        kv_indices_extend=ix_e,
        kv_indptr_extend=ip_e,
        attn_sink=attn_sink,
    )
    ref = dict(
        q_fp32=q_fp32,
        ukv_fp32=ukv_fp32,
        kv_fp32=kv_fp32,
        kv_indices_prefix=ix_p,
        kv_indptr_prefix=ip_p,
        kv_indices_extend=ix_e,
        kv_indptr_extend=ip_e,
        attn_sink=attn_sink,
    )
    return dict(kernel=kernel, ref=ref)


# ---------------------------------------------------------------------------
# perftest-wrapped kernel call (same shape as test_batch_prefill.py)
# ---------------------------------------------------------------------------


@perftest()
def _profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


# Supported precisions. "bf16"/"fp16" use the single-tensor Q/K/V/O kernel;
# "fp8" uses the split NoPE-fp8 / RoPE-bf16 DSA kernel.
_PRECS = ("bf16", "fp16", "fp8")
_PREC_TO_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _get_tolerances(prec: str) -> Tuple[float, float]:
    if prec == "fp16":
        return 1e-2, 1e-2
    if prec == "fp8":
        return 3e-2, 3e-2
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
    prec: str,
    *,
    mode: str = "sparse",
    seed: int = 0,
    verify: bool = True,
    bench: bool = True,
) -> Optional[dict]:
    assert prec in _PRECS, f"unknown prec {prec!r}"
    if _skip_if_unsupported(d=d, prec=prec):
        return None

    softmax_scale = 1.0 / math.sqrt(d)
    msg = (
        f"[N={n} H={h} D={d} total_pages={total_pages} total_tokens={total_tokens} "
        f"prec={prec} mode={mode}]"
    )

    if prec == "fp8":
        data = _make_inputs_fp8(n, h, total_pages, total_tokens, mode=mode, seed=seed)
        kernel_inputs = data["kernel"]
        kernel_fn = pa_sparse_prefill_fp8_opus
        ref_fn, ref_inputs = _ref_pa_sparse_prefill_fp8, data["ref"]
    else:
        kernel_inputs = _make_inputs(
            n,
            h,
            d,
            total_pages,
            total_tokens,
            _PREC_TO_DTYPE[prec],
            mode=mode,
            seed=seed,
        )
        kernel_fn = pa_sparse_prefill_opus
        ref_fn, ref_inputs = _ref_pa_sparse_prefill_opus, kernel_inputs

    nnz_p = int(kernel_inputs["kv_indices_prefix"].numel())
    nnz_e = int(kernel_inputs["kv_indices_extend"].numel())
    row: dict = {"nnz_prefix": nnz_p, "nnz_extend": nnz_e}

    if verify:
        ref = ref_fn(**ref_inputs, softmax_scale=softmax_scale)
        got = kernel_fn(**kernel_inputs, softmax_scale=softmax_scale)
        rtol, atol = _get_tolerances(prec)
        checkAllclose(got, ref, rtol=rtol, atol=atol, msg=msg)

    if bench:
        # `@perftest()` returns (data, avg_us_per_iter).
        _, lat_us = _profile_func(
            kernel_fn, **kernel_inputs, softmax_scale=softmax_scale
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
# ---------------------------------------------------------------------------


_PYTEST_SHAPES = [
    # (N, H, total_pages, total_tokens)
    (64, 16, 256, 256),
    (128, 32, 256, 256),
    (64, 64, 1024, 1024),
    (256, 128, 2048, 2048),
]
_PYTEST_PRECS = ["bf16", "fp16", "fp8"]
_PYTEST_MODES = ["sparse", "dense", "empty"]


@pytest.mark.parametrize("prec", _PYTEST_PRECS)
@pytest.mark.parametrize(
    "n,h,total_pages,total_tokens",
    _PYTEST_SHAPES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
@pytest.mark.parametrize("mode", _PYTEST_MODES)
def test_pa_sparse_prefill_opus(prec, n, h, total_pages, total_tokens, mode):
    # bench=False keeps pytest fast; CLI path does the timing.
    run_pa_sparse_prefill_opus(
        n=n,
        h=h,
        d=512,
        total_pages=total_pages,
        total_tokens=total_tokens,
        prec=prec,
        mode=mode,
        seed=(hash((n, h, total_pages, total_tokens, prec, mode)) & 0xFFFF),
        verify=True,
        bench=False,
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
    "-n",
    "--n_tokens",
    type=int,
    nargs="*",
    default=[1024, 4096],
    help="number of query tokens N (default: [1024, 4096])",
)
parser.add_argument(
    "--h_q",
    type=int,
    nargs="*",
    default=[16, 32, 64, 128],
    help="number of query heads H_Q (default: [16, 32, 64, 128])",
)
parser.add_argument(
    "-d",
    "--head_dim",
    type=int,
    default=512,
    help="head dim D, kernel currently only compiled for 512 (default: 512)",
)
parser.add_argument(
    "--total_pages",
    type=int,
    nargs="*",
    default=[4096, 16384],
    help=(
        "rows in unified_kv (default: [1024, 4096, 16384]). "
        "Pass 0 to mirror -n for that sweep point."
    ),
)
parser.add_argument(
    "--total_tokens",
    type=int,
    default=None,
    help="rows in extend kv (default: matches -n)",
)
parser.add_argument(
    "--prec",
    type=str,
    nargs="*",
    default=["bf16", "fp8"],
    choices=list(_PRECS),
    help=(
        "precision(s) to sweep (default: [bf16, fp8]).\n"
        "  bf16/fp16: single-tensor Q/K/V/O kernel\n"
        "  fp8      : split NoPE-fp8 / RoPE-bf16 DSA kernel"
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    nargs="*",
    default=["sparse", "dense"],
    choices=list(_MODES),
    help=(
        "CSR mode(s) to sweep for both prefix and extend.\n"
        "  sparse: random nnz/row in [0, total_rows] with leading rows\n"
        "          seeded at KV-tile boundaries (0, 1, T-1, T, T+1, ...)\n"
        "  dense : every token sees every page / every kv row\n"
        "  empty : all-empty CSR rows (sink-only output)\n"
        "Default: [sparse, dense]."
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
    type=int,
    default=0,
    help="RNG seed for input + CSR generation",
)


if __name__ == "__main__":
    args = parser.parse_args()

    rows = []
    for n, h, prec, mode, pages_arg in itertools.product(
        args.n_tokens,
        args.h_q,
        args.prec,
        args.mode,
        args.total_pages,
    ):
        # 0 is the sentinel for "mirror -n" on a per-sweep-point basis.
        total_pages = pages_arg if pages_arg > 0 else n
        total_tokens = args.total_tokens if args.total_tokens is not None else n
        row = run_pa_sparse_prefill_opus(
            n=n,
            h=h,
            d=args.head_dim,
            total_pages=total_pages,
            total_tokens=total_tokens,
            prec=prec,
            mode=mode,
            seed=args.seed,
            verify=not args.no_verify,
            bench=not args.no_bench,
        )
        if row:
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        # Drop columns that don't carry signal in the default sweep.
        drop_cols = [c for c in ("verify", "bench", "seed") if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        print()
        print(df.to_string(index=False))
        sys.exit(0)
    sys.exit(0)
