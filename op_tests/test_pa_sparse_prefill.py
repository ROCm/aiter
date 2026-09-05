# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance comparison for sparse paged prefill attention.

Three backends implement the same two-region sparse attention -- a paged
prefix source (``unified_kv``) plus a flat extend source (``kv``), joined by
one online softmax with a per-head sink. Every backend in a case runs on the
same inputs and is checked against the same PyTorch reference, so the reported
latencies are directly comparable.

The ``prec`` axis picks both the input format and the backends that run:

* ``bf16`` -- single ``D=512`` Q/K/V/O tensor: ``opus``, ``triton``.
* ``fp8``  -- split NoPE-fp8 / RoPE-bf16 DSA inputs: ``opus``, ``asm``.

``triton`` is the ``pa_prefill_sparse`` dispatcher, which on gfx1250 reaches a
gluon kernel that takes both KV sources natively. ``asm`` needs ``H_Q == 128``.
Both are gfx1250-only here and drop out of the sweep elsewhere.

Example CLI usage::

    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill.py
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill.py --mode dense
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill.py --backend opus \\
        -n 1024 --h_q 128 --prec fp8 --no-verify
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import pandas as pd
import pytest
import torch

import aiter  # noqa: F401  (registers the top-level export)
from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_fp8_asm
from aiter.ops.pa_sparse_prefill_opus import (
    pa_sparse_prefill_fp8_opus,
    pa_sparse_prefill_opus,
)
from aiter.test_common import (
    DATA_DISTS,
    benchmark,
    checkAllclose,
    fill,
    make_generator,
    perftest,
    print_json_table,
)

try:
    from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse
except ImportError:  # its gluon kernels need Triton >= 3.6
    pa_prefill_sparse = None

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


def _skip_if_unsupported(d: int) -> bool:
    if not torch.cuda.is_available():
        return _skip("CUDA/HIP device not available")
    arch = _get_gpu_arch()
    if arch not in _SUPPORTED_ARCHS:
        return _skip(
            f"pa_sparse_prefill_opus requires one of {_SUPPORTED_ARCHS}, found {arch}"
        )
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
    """Online softmax over ``concat(prefix, extend)``, sink in the denominator
    only. fp32 throughout to mirror the kernel's fp32 accumulator.
    """
    n, _h, _d = q.shape
    out = torch.zeros_like(q)

    q_f32 = q.to(torch.float32)
    ukv_f32 = unified_kv.to(torch.float32)
    kv_f32 = kv.to(torch.float32)
    sink_f32 = attn_sink.to(torch.float32)

    # int64 only because index_select requires Long; the kernel ABI is int32.
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
            continue  # sink-only row: numerator 0, output stays 0
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
# FP8 DSA packing + reference. Each NoPE row of 512 fp8 slots holds
#   [ NoPE fp8 (448) | E8M0 block scales (14) | zero-pad (50) ]
# ---------------------------------------------------------------------------

_FP8_D_NOPE = 448
_FP8_D_NOPE_PADDED = 512
_FP8_D_ROPE = 64
_FP8_D_HEAD = _FP8_D_NOPE + _FP8_D_ROPE
_FP8_NBLK = _FP8_D_NOPE // 32
_FP8_BLK = 32
_FP8_MAX = 448.0  # e4m3fn max normal


def _quantize_nope(real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[R, 448]`` reals into a ``[R, 512]`` fp8 row, returning it
    alongside the dequantized ``[R, 448]`` fp32 values the kernel sees.
    """
    r = real.shape[0]
    blk = real.reshape(r, _FP8_NBLK, _FP8_BLK).to(torch.float32)
    amax = blk.abs().amax(dim=-1)  # [R, NBLK]

    # Lands the block max in (224, 448]; overflowing e4m3fn would cast to NaN.
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
    """Same math as the bf16 reference, over the already-dequantized rows the
    kernel consumes, so quantization error is excluded from the comparison.
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


def _random_csr(
    n: int,
    total_rows: int,
    *,
    allow_empty: bool = True,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random nnz/row in ``[0, total_rows]``."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    lo = 0 if allow_empty else 1

    lens = torch.randint(lo, total_rows + 1, (n,), generator=g, dtype=torch.int32)

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

# Applied symmetrically to both the prefix and the extend CSR.
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
    data_init: str = "norm",
) -> dict:
    assert mode in _MODES
    device = torch.device(device)
    gen = make_generator(seed, device=device)

    q = (
        fill((n * h, d), data_init, gen, dtype=torch.float32, device=device)
        .view(n, h, d)
        .mul_(0.5)
    ).to(dtype)
    unified_kv = (
        fill(
            (total_pages, d), data_init, gen, dtype=torch.float32, device=device
        )
        * 0.5
    ).to(dtype)
    kv = (
        fill(
            (total_tokens, d), data_init, gen, dtype=torch.float32, device=device
        )
        * 0.5
    ).to(dtype)
    attn_sink = (
        fill((h,), data_init, gen, dtype=torch.float32, device=device) * 0.25
    )

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

    return {
        "q": q,
        "unified_kv": unified_kv,
        "kv_indices_prefix": ix_p,
        "kv_indptr_prefix": ip_p,
        "kv": kv,
        "kv_indices_extend": ix_e,
        "kv_indptr_extend": ip_e,
        "attn_sink": attn_sink,
    }


def _make_inputs_fp8(
    n: int,
    h: int,
    total_pages: int,
    total_tokens: int,
    *,
    mode: str = "sparse",
    device: torch.device | str = "cuda",
    seed: int = 0,
    data_init: str = "norm",
) -> dict:
    """Returns ``{"kernel": ..., "ref": ...}``: the split fp8/bf16 tensors the
    kernels take, and the dequantized fp32 rows the reference takes.
    """
    assert mode in _MODES
    device = torch.device(device)
    gen = make_generator(seed, device=device)

    def _streams(rows: int):
        nope_fp8, deq = _quantize_nope(
            fill(
                (rows, _FP8_D_NOPE),
                data_init,
                gen,
                dtype=torch.float32,
                device=device,
            )
            * 0.5
        )
        rope = (
            fill(
                (rows, _FP8_D_ROPE),
                data_init,
                gen,
                dtype=torch.float32,
                device=device,
            )
            * 0.5
        )
        rope = rope.to(torch.bfloat16)
        row_fp32 = torch.cat([deq, rope.to(torch.float32)], dim=1)  # [rows, 512]
        return nope_fp8, rope, row_fp32

    qn, qr, q_fp32 = _streams(n * h)
    qn = qn.reshape(n, h, _FP8_D_NOPE_PADDED)
    qr = qr.reshape(n, h, _FP8_D_ROPE)
    q_fp32 = q_fp32.reshape(n, h, _FP8_D_HEAD)
    ukn, ukr, ukv_fp32 = _streams(total_pages)
    kn, kr, kv_fp32 = _streams(total_tokens)

    attn_sink = (
        fill((h,), data_init, gen, dtype=torch.float32, device=device) * 0.25
    )

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

    kernel = {
        "q_nope": qn,
        "q_rope": qr,
        "unified_kv_nope": ukn,
        "unified_kv_rope": ukr,
        "kv_indices_prefix": ix_p,
        "kv_indptr_prefix": ip_p,
        "kv_nope": kn,
        "kv_rope": kr,
        "kv_indices_extend": ix_e,
        "kv_indptr_extend": ip_e,
        "attn_sink": attn_sink,
    }
    ref = {
        "q_fp32": q_fp32,
        "ukv_fp32": ukv_fp32,
        "kv_fp32": kv_fp32,
        "kv_indices_prefix": ix_p,
        "kv_indptr_prefix": ip_p,
        "kv_indices_extend": ix_e,
        "kv_indptr_extend": ip_e,
        "attn_sink": attn_sink,
    }
    return {"kernel": kernel, "ref": ref}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_BACKENDS = ("opus", "asm", "triton")

_PREC_BACKENDS = {
    "bf16": ("opus", "triton"),
    "fp8": ("opus", "asm"),
}

# Baked into the asm code object; its Q address math needs gridDim.y == 1.
_ASM_HEADS = 128

# Only gfx1250's pa_prefill_sparse reads a second KV source; the rest reject it.
_TRITON_ARCHS = ("gfx1250",) if pa_prefill_sparse is not None else ()

# Mismatch ratio allowed per backend; same as test_pa_decode_sparse.py.
_ERR_TOL = 0.01


# ---------------------------------------------------------------------------
# perftest-wrapped kernel call
# ---------------------------------------------------------------------------


@perftest()
def _profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


_PRECS = ("bf16", "fp8")
_PREC_TO_DTYPE = {"bf16": torch.bfloat16}

# |got - ref| <= _ATOL + _RTOL * |ref|. Must stay tight in absolute terms: the
# output magnitude shrinks as nnz/row grows (|out| ~ 0.05 at 4k nnz/row), so a
# loose atol would accept even all-zeros. Measured worst case is 3.9e-3, itself
# half an ulp of the bf16 output.
_RTOL = 1e-2
_ATOL = 1e-2


# ---------------------------------------------------------------------------
# Single-case driver. `@benchmark()` turns the kwargs into a row dict and
# merges in whatever this returns.
# ---------------------------------------------------------------------------


@benchmark()
def run_pa_sparse_prefill(
    n: int,
    h: int,
    d: int,
    total_pages: int,
    total_tokens: int,
    prec: str,
    *,
    mode: str = "sparse",
    backends: tuple = _BACKENDS,
    seed: int = 0,
    data_init: str = "norm",
    verify: bool = True,
    bench: bool = True,
) -> dict | None:
    assert prec in _PRECS, f"unknown prec {prec!r}"
    if _skip_if_unsupported(d=d):
        return None

    softmax_scale = 1.0 / math.sqrt(d)
    msg = (
        f"[N={n} H={h} D={d} total_pages={total_pages} total_tokens={total_tokens} "
        f"prec={prec} mode={mode} data_init={data_init} seed={seed}]"
    )
    wanted = [b for b in _PREC_BACKENDS[prec] if b in backends]

    # Lambdas, not partials: @perftest() deep-copies args, cloning bound tensors.
    candidates: list = []

    if prec == "fp8":
        data = _make_inputs_fp8(
            n,
            h,
            total_pages,
            total_tokens,
            mode=mode,
            seed=seed,
            data_init=data_init,
        )
        kernel_inputs = data["kernel"]
        ref_fn, ref_inputs = _ref_pa_sparse_prefill_fp8, data["ref"]
        if "opus" in wanted:
            candidates.append(
                (
                    "opus",
                    lambda: pa_sparse_prefill_fp8_opus(
                        **kernel_inputs, softmax_scale=softmax_scale
                    ),
                )
            )
        if "asm" in wanted and h == _ASM_HEADS and _get_gpu_arch() == "gfx1250":
            candidates.append(
                (
                    "asm",
                    lambda: mla_sparse_prefill_fp8_asm(
                        **kernel_inputs, softmax_scale=softmax_scale
                    ),
                )
            )
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
            data_init=data_init,
        )
        ref_fn, ref_inputs = _ref_pa_sparse_prefill_opus, kernel_inputs
        if "opus" in wanted:
            candidates.append(
                (
                    "opus",
                    lambda: pa_sparse_prefill_opus(
                        **kernel_inputs, softmax_scale=softmax_scale
                    ),
                )
            )
        if "triton" in wanted and _get_gpu_arch() in _TRITON_ARCHS:
            # These CSRs never hold -1; the wrapper's default guess costs ~1.5x.
            candidates.append(
                (
                    "triton",
                    lambda: pa_prefill_sparse(
                        **kernel_inputs,
                        softmax_scale=softmax_scale,
                        has_invalid=False,
                    ),
                )
            )

    if not candidates:
        return None

    total_nnz = int(kernel_inputs["kv_indices_prefix"].numel()) + int(
        kernel_inputs["kv_indices_extend"].numel()
    )
    row: dict = {}

    ref = ref_fn(**ref_inputs, softmax_scale=softmax_scale) if verify else None

    for name, invoke in candidates:
        if verify:
            err = checkAllclose(
                invoke(),
                ref,
                rtol=_RTOL,
                atol=_ATOL,
                tol_err_ratio=_ERR_TOL,
                msg=f"{name}: {msg}",
            )
            # float() so a skipped backend's NaN holes don't split the dtype.
            row[f"{name} err"] = float(err)
            # checkAllclose only raises on catastrophic mismatches, not on this.
            assert err <= _ERR_TOL, (
                f"{name}: {msg} mismatch ratio {err:.3%} exceeds "
                f"{_ERR_TOL:.1%} at rtol={_RTOL} atol={_ATOL}"
            )

        if bench:
            _, lat_us = _profile_func(invoke)  # (data, avg_us_per_iter)
            flops = 4.0 * h * total_nnz * d
            tflops = flops / max(lat_us * 1e-6, 1e-12) / 1e12
            row[f"{name} us"] = round(float(lat_us), 2)
            row[f"{name} TFLOPS"] = round(float(tflops), 2)

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
_PYTEST_PRECS = ["bf16", "fp8"]
_PYTEST_MODES = ["sparse", "dense", "empty"]


@pytest.mark.parametrize("prec", _PYTEST_PRECS)
@pytest.mark.parametrize(
    "n,h,total_pages,total_tokens",
    _PYTEST_SHAPES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
@pytest.mark.parametrize("mode", _PYTEST_MODES)
def test_pa_sparse_prefill(prec, n, h, total_pages, total_tokens, mode):
    run_pa_sparse_prefill(
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
        "opus / asm / triton sparse-prefill correctness + benchmark driver.\n"
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
    help=(
        "number of query heads H_Q (default: [16, 32, 64, 128]).\n"
        f"The asm candidate is built for H_Q={_ASM_HEADS} only and drops out\n"
        "of the other sweep points; opus and triton run at every value."
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
    default=[4096, 16384],
    help=(
        "rows in unified_kv (default: [4096, 16384]). "
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
        "  bf16: single-tensor Q/K/V/O kernel      -> opus, triton\n"
        "  fp8 : split NoPE-fp8 / RoPE-bf16 kernel -> opus, asm"
    ),
)
parser.add_argument(
    "--backend",
    type=str,
    nargs="*",
    default=list(_BACKENDS),
    choices=list(_BACKENDS),
    help=(
        "backend(s) to run, intersected with what each precision supports\n"
        "and with what the running arch provides (default: all)."
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
        "  sparse: random nnz/row in [0, total_rows]\n"
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
parser.add_argument(
    "--data-init",
    nargs="+",
    choices=list(DATA_DISTS),
    default=["norm"],
    help="DATA initialization distribution(s) for Q, KV and attention sink",
)


if __name__ == "__main__":
    args = parser.parse_args()

    rows = []
    # product varies its last argument fastest -> this is also the row order.
    for prec, mode, h, n, pages_arg, data_init in itertools.product(
        args.prec,
        args.mode,
        args.h_q,
        args.n_tokens,
        args.total_pages,
        args.data_init,
    ):
        total_pages = pages_arg if pages_arg > 0 else n  # 0 is "mirror -n"
        total_tokens = args.total_tokens if args.total_tokens is not None else n
        row = run_pa_sparse_prefill(
            n=n,
            h=h,
            d=args.head_dim,
            total_pages=total_pages,
            total_tokens=total_tokens,
            prec=prec,
            mode=mode,
            backends=tuple(args.backend),
            seed=args.seed,
            data_init=data_init,
            verify=not args.no_verify,
            bench=not args.no_bench,
        )
        if row:
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        drop_cols = [
            c for c in ("verify", "bench", "seed", "backends") if c in df.columns
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        # Column order otherwise follows whichever row first ran a backend.
        lead = [
            c for c in ("prec", "mode", "data_init", "h", "n") if c in df.columns
        ]
        rest = [c for c in df.columns if c not in lead]
        metrics = [c for b in _BACKENDS for c in rest if c.startswith(f"{b} ")]
        df = df[lead + [c for c in rest if c not in metrics] + metrics]
        print_json_table("pa_sparse_prefill summary", df)
        sys.exit(0)
    sys.exit(0)
