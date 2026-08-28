# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance tests for sparse prefill attention.

Each case uses one shared Q/KV/CSR input and compares the applicable
implementations: OPUS (bf16/fp16 or split fp8), gfx1250 MLA asm (split fp8),
and the portable Triton kernel. Pytest runs correctness coverage; the CLI also
supports benchmark and explicit CSR-boundary sweeps.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os

import pandas as pd
import pytest
import torch
import triton

import aiter  # noqa: F401  (registers the top-level export)
from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_fp8_asm
from aiter.ops.pa_sparse_prefill_opus import (
    pa_sparse_prefill_fp8_opus,
    pa_sparse_prefill_opus,
)
from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _sparse_attn_prefill_kernel,
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


def _get_gpu_arch() -> str | None:
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


_SUPPORTED_ARCHS = ("gfx950", "gfx1250")


def _skip_if_unsupported(d: int, prec: str | None = None) -> bool:
    if not torch.cuda.is_available():
        return _skip("CUDA/HIP device not available")
    arch = _get_gpu_arch()
    if arch not in _SUPPORTED_ARCHS:
        return _skip(
            f"pa_sparse_prefill_opus requires one of {_SUPPORTED_ARCHS}, found {arch}"
        )
    if d != 512:
        return _skip(f"Only D=512 is compiled, requested D={d}")
    # The gfx1250 code object only carries the bf16 traits.
    if arch == "gfx1250" and prec == "fp16":
        return _skip("gfx1250 only provides the bf16 variant")
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
    n, _h, _d = q.shape
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


def _quantize_nope(real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

# Cover the KV tile sizes used by the OPUS and asm candidates. The kernel inner
# loop advances the K/V dimension in chunks of these sizes, so the trailing-tile
# branches are most likely to break when nnz_per_row sits at one of these
# boundary values.
_CSR_TILE_SIZES = (32, 64)


def _boundary_nnz(kv_tile_sizes, total_rows: int) -> list:
    """Tile-boundary nnz values seeded into the leading rows of a sparse CSR,
    mirroring gcnasm/opus_attn/sparse_paged_attn/pa_host.cc::
    init_sparse_kv_indices. Clamped into [0, total_rows]."""
    if isinstance(kv_tile_sizes, int):
        kv_tile_sizes = (kv_tile_sizes,)
    cands = {0, 1, total_rows}
    for tile_size in kv_tile_sizes:
        cands.update(
            (
                tile_size - 1,
                tile_size,
                tile_size + 1,
                2 * tile_size,
                2 * tile_size + 1,
            )
        )
    return [max(0, min(v, total_rows)) for v in sorted(cands)]


def _random_csr(
    n: int,
    total_rows: int,
    *,
    allow_empty: bool = True,
    kv_tile_size=_CSR_TILE_SIZES,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor]:
    indptr = torch.arange(0, (n + 1) * total_rows, total_rows, dtype=torch.int32)
    indices = torch.arange(total_rows, dtype=torch.int32).repeat(n)
    return indptr.to(device), indices.to(device)


def _empty_csr(n: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(n + 1, dtype=torch.int32, device=device),
        torch.zeros(0, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------

# Single sparsity knob applied symmetrically to both prefix and extend CSRs.
_MODES = ("sparse", "dense", "empty")


def _fixed_csr(n: int, nnz: int, total_rows: int, *, device):
    """CSR with exactly ``nnz`` entries on every row."""
    if nnz == 0:
        return _empty_csr(n, device=device)
    indptr = torch.arange(n + 1, dtype=torch.int32, device=device) * nnz
    indices = (
        torch.arange(nnz, dtype=torch.int32, device=device) % max(total_rows, 1)
    ).repeat(n)
    return indptr, indices


def _make_sparse_case(
    n: int,
    h: int,
    d: int,
    total_pages: int,
    total_tokens: int,
    *,
    mode: str = "sparse",
    nnz_prefix: int | None = None,
    nnz_extend: int | None = None,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict:
    """Generate one logical sparse-attention problem shared by all candidates."""
    assert mode in _MODES or mode == "fixed"
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q_fp32 = torch.randn(n, h, d, device=device, generator=gen) * 0.5
    unified_kv_fp32 = torch.randn(total_pages, d, device=device, generator=gen) * 0.5
    kv_fp32 = torch.randn(total_tokens, d, device=device, generator=gen) * 0.5
    attn_sink = torch.randn(h, device=device, generator=gen) * 0.25

    def _csr(total_rows: int, nnz: int | None, seed_offset: int):
        if nnz is not None:
            return _fixed_csr(n, nnz, total_rows, device=device)
        if mode == "sparse":
            return _random_csr(
                n,
                total_rows,
                device=device,
                kv_tile_size=_CSR_TILE_SIZES,
                seed=seed * 2 + seed_offset,
            )
        if mode == "dense":
            return _dense_csr(n, total_rows, device=device)
        return _empty_csr(n, device=device)

    ip_p, ix_p = _csr(total_pages, nnz_prefix, 1)
    ip_e, ix_e = _csr(total_tokens, nnz_extend, 2)
    return {
        "q_fp32": q_fp32,
        "unified_kv_fp32": unified_kv_fp32,
        "kv_fp32": kv_fp32,
        "kv_indices_prefix": ix_p,
        "kv_indptr_prefix": ip_p,
        "kv_indices_extend": ix_e,
        "kv_indptr_extend": ip_e,
        "attn_sink": attn_sink,
    }


def _make_single_inputs(case: dict, dtype: torch.dtype) -> dict:
    return {
        "q": case["q_fp32"].to(dtype),
        "unified_kv": case["unified_kv_fp32"].to(dtype),
        "kv_indices_prefix": case["kv_indices_prefix"],
        "kv_indptr_prefix": case["kv_indptr_prefix"],
        "kv": case["kv_fp32"].to(dtype),
        "kv_indices_extend": case["kv_indices_extend"],
        "kv_indptr_extend": case["kv_indptr_extend"],
        "attn_sink": case["attn_sink"],
    }


def _make_split_inputs(case: dict) -> dict:
    def split_rows(rows: torch.Tensor):
        flat = rows.reshape(-1, _FP8_D_HEAD)
        nope, deq = _quantize_nope(flat[:, :_FP8_D_NOPE])
        rope = flat[:, _FP8_D_NOPE:].to(torch.bfloat16)
        return nope, rope, torch.cat([deq, rope.to(torch.float32)], dim=1)

    qn, qr, q_fp32 = split_rows(case["q_fp32"])
    ukn, ukr, ukv_fp32 = split_rows(case["unified_kv_fp32"])
    kn, kr, kv_fp32 = split_rows(case["kv_fp32"])
    n, h, _ = case["q_fp32"].shape
    return {
        "kernel": {
            "q_nope": qn.reshape(n, h, _FP8_D_NOPE_PADDED),
            "q_rope": qr.reshape(n, h, _FP8_D_ROPE),
            "unified_kv_nope": ukn,
            "unified_kv_rope": ukr,
            "kv_indices_prefix": case["kv_indices_prefix"],
            "kv_indptr_prefix": case["kv_indptr_prefix"],
            "kv_nope": kn,
            "kv_rope": kr,
            "kv_indices_extend": case["kv_indices_extend"],
            "kv_indptr_extend": case["kv_indptr_extend"],
            "attn_sink": case["attn_sink"],
        },
        "ref": {
            "q_fp32": q_fp32.reshape(n, h, _FP8_D_HEAD),
            "ukv_fp32": ukv_fp32,
            "kv_fp32": kv_fp32,
            "kv_indices_prefix": case["kv_indices_prefix"],
            "kv_indptr_prefix": case["kv_indptr_prefix"],
            "kv_indices_extend": case["kv_indices_extend"],
            "kv_indptr_extend": case["kv_indptr_extend"],
            "attn_sink": case["attn_sink"],
        },
    }


# ---------------------------------------------------------------------------
# Portable Triton candidate.
# ---------------------------------------------------------------------------

_ASM_HEADS = 128
_ERR_TOL = 0.05


def _merge_two_sources(ukv, kv, ix_p, ip_p, ix_e, ip_e):
    """Merge the two CSR sources for the portable Triton candidate."""
    total_pages = ukv.shape[0]
    pool = torch.cat([ukv, kv], dim=0)
    lens_p = (ip_p[1:] - ip_p[:-1]).to(torch.int64)
    lens_e = (ip_e[1:] - ip_e[:-1]).to(torch.int64)
    indptr = torch.zeros(ip_p.numel(), dtype=torch.int32, device=ukv.device)
    indptr[1:] = torch.cumsum(lens_p + lens_e, 0).to(torch.int32)
    parts = []
    pp = ip_p.to(torch.int64).tolist()
    pe = ip_e.to(torch.int64).tolist()
    for i in range(len(pp) - 1):
        parts.append(ix_p[pp[i] : pp[i + 1]])
        parts.append(ix_e[pe[i] : pe[i + 1]] + total_pages)
    indices = (
        torch.cat(parts).to(torch.int32)
        if parts
        else torch.zeros(0, dtype=torch.int32, device=ukv.device)
    )
    return pool, indices, indptr


def _run_triton(q, pool, indices, indptr, attn_sink, softmax_scale):
    """Run the portable Triton sparse-prefill candidate."""
    out = torch.empty_like(q)
    num_queries, num_heads, head_dim = q.shape

    def grid(META):
        return (num_queries, triton.cdiv(num_heads, META["BLOCK_H"]))

    _sparse_attn_prefill_kernel[grid](
        q,
        pool,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        pool.stride(0),
        pool.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        pool.shape[0],
        softmax_scale,
        HAS_ATTN_SINK=True,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out


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


def _get_tolerances(prec: str) -> tuple[float, float]:
    if prec == "fp16":
        return 1e-2, 1e-2
    if prec == "fp8":
        return 3e-2, 3e-2
    return 2e-2, 2e-2  # bf16 default


# ---------------------------------------------------------------------------
# Single-case driver -- all candidates share this one logical input case.
# `@benchmark()` collects the kwargs into a row dict and merges in whatever
# this function returns, so the CLI can build a pandas DataFrame.
# ---------------------------------------------------------------------------


@benchmark()
def run_sparse_prefill(
    n: int,
    h: int,
    d: int,
    total_pages: int,
    total_tokens: int,
    prec: str,
    *,
    mode: str = "sparse",
    nnz_prefix: int | None = None,
    nnz_extend: int | None = None,
    seed: int = 0,
    verify: bool = True,
    bench: bool = True,
) -> dict | None:
    assert prec in _PRECS, f"unknown prec {prec!r}"
    if _skip_if_unsupported(d=d, prec=prec):
        return None

    softmax_scale = 1.0 / math.sqrt(d)
    msg = (
        f"[N={n} H={h} D={d} total_pages={total_pages} total_tokens={total_tokens} "
        f"prec={prec} mode={mode}]"
    )

    case = _make_sparse_case(
        n,
        h,
        d,
        total_pages,
        total_tokens,
        mode=mode,
        nnz_prefix=nnz_prefix,
        nnz_extend=nnz_extend,
        seed=seed,
    )
    nnz_p = int(case["kv_indices_prefix"].numel())
    nnz_e = int(case["kv_indices_extend"].numel())
    total_nnz = nnz_p + nnz_e
    candidates = []

    if prec == "fp8":
        split = _make_split_inputs(case)
        candidates.append(
            (
                "opus",
                lambda: pa_sparse_prefill_fp8_opus(
                    **split["kernel"], softmax_scale=softmax_scale
                ),
                _ref_pa_sparse_prefill_fp8(**split["ref"], softmax_scale=softmax_scale),
                "fp8",
                1,
            )
        )
    else:
        single = _make_single_inputs(case, _PREC_TO_DTYPE[prec])
        candidates.append(
            (
                "opus",
                lambda: pa_sparse_prefill_opus(**single, softmax_scale=softmax_scale),
                _ref_pa_sparse_prefill_opus(**single, softmax_scale=softmax_scale),
                prec,
                torch.tensor([], dtype=_PREC_TO_DTYPE[prec]).element_size(),
            )
        )

    bf16_inputs = _make_single_inputs(case, torch.bfloat16)
    pool, merged_indices, merged_indptr = _merge_two_sources(
        bf16_inputs["unified_kv"],
        bf16_inputs["kv"],
        case["kv_indices_prefix"],
        case["kv_indptr_prefix"],
        case["kv_indices_extend"],
        case["kv_indptr_extend"],
    )
    triton_ref = _ref_pa_sparse_prefill_opus(**bf16_inputs, softmax_scale=softmax_scale)
    candidates.append(
        (
            "triton",
            lambda: _run_triton(
                bf16_inputs["q"],
                pool,
                merged_indices,
                merged_indptr,
                case["attn_sink"],
                softmax_scale,
            ),
            triton_ref,
            "bf16",
            2,
        )
    )

    if h == _ASM_HEADS and _get_gpu_arch() == "gfx1250":
        split = _make_split_inputs(case)
        candidates.append(
            (
                "asm",
                lambda: mla_sparse_prefill_fp8_asm(
                    **split["kernel"], softmax_scale=softmax_scale
                ),
                _ref_pa_sparse_prefill_fp8(**split["ref"], softmax_scale=softmax_scale),
                "fp8",
                1,
            )
        )

    # Report per-row nnz rather than the pool-wide total, so the column matches
    # the --nnz-prefix/--nnz-extend the sweep was asked for instead of scaling
    # with N. Exact for fixed/dense/empty (every row has the same count); a mean
    # for sparse, whose rows vary. total_nnz below keeps the full count -- the
    # FLOPs and bytes figures need the real work done, not the per-row figure.
    row: dict = {"nnz_prefix": nnz_p // n, "nnz_extend": nnz_e // n}
    for name, invoke, ref, candidate_prec, kv_esz in candidates:
        if verify:
            rtol, atol = _get_tolerances(candidate_prec)
            err = checkAllclose(
                invoke(),
                ref,
                rtol=rtol,
                atol=atol,
                tol_err_ratio=_ERR_TOL,
                msg=f"{name}: {msg}",
            )
            row[f"{name} err"] = err

        if bench:
            _, lat_us = _profile_func(invoke)
            flops = 4.0 * h * total_nnz * d
            tflops = flops / max(lat_us * 1e-6, 1e-12) / 1e12
            row[f"{name} us"] = round(float(lat_us), 2)
            row[f"{name} TFLOPS"] = round(float(tflops), 2)
            row[f"{name} TB/s"] = round(
                (n * h * d * kv_esz + total_nnz * d * kv_esz + n * h * d * 2)
                / max(lat_us, 1e-12)
                / 1e6,
                2,
            )

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
def test_pa_sparse_prefill(prec, n, h, total_pages, total_tokens, mode):
    # bench=False keeps pytest fast; CLI path does the timing.
    run_sparse_prefill(
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
        "PA OPUS and gfx1250 asm MLA sparse-prefill correctness + benchmark "
        "driver.\nAll list arguments are swept via itertools.product."
    ),
)
parser.add_argument(
    "-n",
    "--n_tokens",
    type=int,
    nargs="*",
    default=[512, 1024, 2048, 4096],
    help="number of query tokens N (default: [512, 1024, 2048, 4096])",
)
parser.add_argument(
    "--h_q",
    type=int,
    nargs="*",
    default=[128],
    help=(
        "number of query heads H_Q (default: [128]).\n"
        "The asm candidate only registers at H_Q=128, so the default keeps the\n"
        "three-way opus/triton/asm comparison. Pass 16/32/64 to sweep the\n"
        "opus-vs-triton pair at other head counts."
    ),
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
    default=[],
    help=(
        "rows in unified_kv. Pass 0 to mirror -n for that sweep point.\n"
        "Empty by default, which switches the mode/total_pages sweep off so a\n"
        "bare run only does the explicit-nnz sweep; pass values to enable it."
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
    default=["fp8"],
    choices=list(_PRECS),
    help=(
        "precision(s) to sweep (default: [fp8], the only one the asm\n"
        "candidate implements).\n"
        "  bf16/fp16: single-tensor Q/K/V/O kernel\n"
        "  fp8      : split NoPE-fp8 / RoPE-bf16 DSA kernel"
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    nargs="*",
    default=[],
    choices=list(_MODES),
    help=(
        "CSR mode(s) to sweep for both prefix and extend.\n"
        "  sparse: random nnz/row in [0, total_rows] with leading rows\n"
        "          seeded at KV-tile boundaries (0, 1, T-1, T, T+1, ...)\n"
        "  dense : every token sees every page / every kv row\n"
        "  empty : all-empty CSR rows (sink-only output)\n"
        "Empty by default (see --total_pages); this sweep needs both set."
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
parser.add_argument(
    "--nnz",
    type=int,
    nargs="*",
    default=[],
    help="explicit per-row nnz values for the shared boundary sweep",
)
parser.add_argument(
    "--nnz-prefix",
    type=int,
    nargs="*",
    default=[256, 1024, 4096, 8192, 16384],
    help=(
        "explicit per-row prefix nnz values for the shared shape sweep\n"
        "(default: [256, 1024, 4096, 8192, 16384])"
    ),
)
parser.add_argument(
    "--nnz-extend",
    type=int,
    nargs="*",
    default=[128],
    help=(
        "explicit per-row extend nnz values for the shared shape sweep\n"
        "(default: [128])"
    ),
)
parser.add_argument(
    "--pool",
    type=int,
    default=4096,
    help=(
        "prefix/extend pool rows for explicit CSR sweeps. Raised to the largest\n"
        "swept nnz when that is bigger, so the CSR never wraps and re-gathers\n"
        "the same rows (default: 4096, i.e. 16384 for the default nnz sweep)."
    ),
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
        row = run_sparse_prefill(
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

    if args.nnz:
        pool = max(args.pool, max(args.nnz))
        for n, h, prec, nnz in itertools.product(
            args.n_tokens, args.h_q, args.prec, args.nnz
        ):
            for npx, nex in ((nnz, 0), (0, nnz), (nnz, nnz)):
                row = run_sparse_prefill(
                    n=n,
                    h=h,
                    d=args.head_dim,
                    total_pages=pool,
                    total_tokens=pool,
                    prec=prec,
                    mode="fixed",
                    nnz_prefix=npx,
                    nnz_extend=nex,
                    seed=args.seed,
                    verify=not args.no_verify,
                    bench=not args.no_bench,
                )
                if row:
                    rows.append(row)

    if args.nnz_prefix and args.nnz_extend:
        pool = max(args.pool, max(args.nnz_prefix), max(args.nnz_extend))
        for n, h, prec, npx, nex in itertools.product(
            args.n_tokens,
            args.h_q,
            args.prec,
            args.nnz_prefix,
            args.nnz_extend,
        ):
            row = run_sparse_prefill(
                n=n,
                h=h,
                d=args.head_dim,
                total_pages=pool,
                total_tokens=pool,
                prec=prec,
                mode="fixed",
                nnz_prefix=npx,
                nnz_extend=nex,
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
