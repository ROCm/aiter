# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for aiter's OPUS-based split-precision (NoPE fp8 / RoPE bf16) sparse
paged prefill attention (``pa_sparse_prefill_fp8_opus``).

The DSA fp8 layout packs, per NoPE row of 512 fp8 slots:

    [ NoPE fp8 (448) | E8M0 block scales (14) | fp8 zero-pad (50) ]

with one E8M0 (power-of-two) scale per 32-element NoPE block. The RoPE stream
is a separate ``[*, 64]`` bf16 tensor. The kernel runs the NoPE QK^T as scaled
MXFP8 MFMA, the RoPE QK^T and PV at bf16, and accumulates in fp32.

The reference dequantizes the *same* packed fp8/E8M0 bytes the kernel consumes
(``fp8.float() * 2**(E-127)``) and runs the attention in fp32, so the only
sources of divergence are the kernel's bf16 intermediates and MFMA rounding.

CLI::

    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_fp8_opus.py
    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill_fp8_opus.py --mode dense
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

import aiter  # noqa: F401
from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus
from aiter.test_common import benchmark, checkAllclose, perftest

D_NOPE = 448
D_NOPE_PADDED = 512
D_ROPE = 64
D_HEAD = D_NOPE + D_ROPE  # 512
_BLOCK = 32
_NBLK = D_NOPE // _BLOCK  # 14
_FP8_MAX = 448.0  # e4m3fn max normal
_KV_TILE_SIZE = 64


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


def _skip_if_unsupported() -> bool:
    if not torch.cuda.is_available():
        return _skip("CUDA/HIP device not available")
    arch = _get_gpu_arch()
    if arch != "gfx950":
        return _skip(f"pa_sparse_prefill_fp8_opus requires gfx950, found {arch}")
    return False


# ---------------------------------------------------------------------------
# DSA fp8 packing: real floats -> (packed fp8 NoPE row, dequantized NoPE)
# ---------------------------------------------------------------------------


def _quantize_nope(real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[R, 448]`` real values into a packed ``[R, 512]`` fp8 row
    (NoPE fp8 + E8M0 block scales + zero pad) and return ``(packed_fp8, deq)``
    where ``deq`` (``[R, 448]`` fp32) is the dequantized NoPE the kernel sees.
    """
    r = real.shape[0]
    blk = real.reshape(r, _NBLK, _BLOCK).to(torch.float32)
    amax = blk.abs().amax(dim=-1)  # [R, NBLK]

    # Per-block E8M0 exponent chosen so the block max maps to (224, 448], i.e.
    # strictly inside the e4m3fn finite range (overflow -> NaN on cast).
    e_unbiased = torch.ceil(torch.log2(amax.clamp(min=1e-30) / _FP8_MAX)).to(torch.int32)
    zero_blk = amax == 0
    e_unbiased = torch.where(zero_blk, torch.zeros_like(e_unbiased), e_unbiased)
    e_byte = (e_unbiased + 127).clamp(0, 255).to(torch.uint8)  # [R, NBLK]
    s = torch.exp2(e_unbiased.to(torch.float32)).unsqueeze(-1)  # [R, NBLK, 1]

    q = (blk / s).to(torch.float8_e4m3fn)  # [R, NBLK, BLOCK]
    deq = q.to(torch.float32) * s  # dequantized exactly as the kernel does
    deq = deq.reshape(r, D_NOPE)

    packed = torch.zeros(r, D_NOPE_PADDED, dtype=torch.uint8, device=real.device)
    packed[:, :D_NOPE] = q.reshape(r, D_NOPE).view(torch.uint8)
    packed[:, D_NOPE : D_NOPE + _NBLK] = e_byte
    return packed.view(torch.float8_e4m3fn), deq


# ---------------------------------------------------------------------------
# PyTorch reference (fp32) over concat(dequant NoPE, RoPE).
# ---------------------------------------------------------------------------


def _ref(
    q_dense: torch.Tensor,  # [N, H, 512] fp32 (deq nope + rope)
    ukv_dense: torch.Tensor,  # [total_pages, 512] fp32
    kv_dense: torch.Tensor,  # [total_tokens, 512] fp32
    ip_p: torch.Tensor,
    ix_p: torch.Tensor,
    ip_e: torch.Tensor,
    ix_e: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    n, h, _ = q_dense.shape
    out = torch.zeros(n, h, D_HEAD, dtype=torch.float32, device=q_dense.device)
    pp = ip_p.to(torch.int64).cpu().tolist()
    pe = ip_e.to(torch.int64).cpu().tolist()
    pidx = ix_p.to(torch.int64)
    eidx = ix_e.to(torch.int64)
    sink_f = sink.to(torch.float32)

    for i in range(n):
        rows = []
        if pp[i + 1] > pp[i]:
            rows.append(ukv_dense.index_select(0, pidx[pp[i] : pp[i + 1]]))
        if pe[i + 1] > pe[i]:
            rows.append(kv_dense.index_select(0, eidx[pe[i] : pe[i + 1]]))
        if not rows:
            continue
        kv_rows = torch.cat(rows, dim=0)  # [nnz, 512]
        scores = q_dense[i] @ kv_rows.t() * softmax_scale  # [H, nnz]
        sink_col = sink_f.unsqueeze(1)
        m = torch.cat([scores, sink_col], dim=1).amax(dim=1, keepdim=True)
        e_s = torch.exp(scores - m)
        e_sink = torch.exp(sink_col - m)
        denom = e_s.sum(dim=1, keepdim=True) + e_sink
        out[i] = (e_s / denom) @ kv_rows
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# CSR generators (mirror the bf16 test harness)
# ---------------------------------------------------------------------------


def _boundary_nnz(total_rows: int) -> list:
    cands = [0, 1, _KV_TILE_SIZE - 1, _KV_TILE_SIZE, _KV_TILE_SIZE + 1,
             2 * _KV_TILE_SIZE, 2 * _KV_TILE_SIZE + 1, total_rows]
    return [max(0, min(v, total_rows)) for v in cands]


def _random_csr(n, total_rows, *, device, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    lens = torch.randint(0, total_rows + 1, (n,), generator=g, dtype=torch.int32)
    for i, v in enumerate(_boundary_nnz(total_rows)[:n]):
        lens[i] = v
    indptr = torch.zeros(n + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, dim=0)
    nnz = int(indptr[-1].item())
    indices = torch.empty(nnz, dtype=torch.int32)
    for i in range(n):
        s, e = int(indptr[i]), int(indptr[i + 1])
        if e > s:
            indices[s:e] = torch.randperm(total_rows, generator=g)[: e - s].to(torch.int32)
    return indptr.to(device), indices.to(device)


def _dense_csr(n, total_rows, *, device):
    indptr = torch.arange(0, (n + 1) * total_rows, total_rows, dtype=torch.int32)
    indices = torch.arange(total_rows, dtype=torch.int32).repeat(n)
    return indptr.to(device), indices.to(device)


def _empty_csr(n, *, device):
    return (torch.zeros(n + 1, dtype=torch.int32, device=device),
            torch.zeros(0, dtype=torch.int32, device=device))


_MODES = ("sparse", "dense", "empty")


def _make_inputs(n, h, total_pages, total_tokens, *, mode="sparse", device="cuda", seed=0):
    torch.manual_seed(seed)
    device = torch.device(device)

    def _streams(rows, scale):
        real = torch.randn(rows, D_NOPE, device=device) * scale
        nope_fp8, deq = _quantize_nope(real)
        rope = (torch.randn(rows, D_ROPE, device=device) * scale).to(torch.bfloat16)
        dense = torch.cat([deq, rope.to(torch.float32)], dim=1)  # [rows, 512]
        return nope_fp8, rope, dense

    qn, qr, q_dense2d = _streams(n * h, 0.5)
    qn = qn.reshape(n, h, D_NOPE_PADDED)
    qr = qr.reshape(n, h, D_ROPE)
    q_dense = q_dense2d.reshape(n, h, D_HEAD)

    ukn, ukr, uk_dense = _streams(total_pages, 0.5)
    kn, kr, k_dense = _streams(total_tokens, 0.5)

    sink = torch.randn(h, device=device, dtype=torch.float32) * 0.25

    def _csr(total_rows, off):
        if mode == "sparse":
            return _random_csr(n, total_rows, device=device, seed=seed * 2 + off)
        if mode == "dense":
            return _dense_csr(n, total_rows, device=device)
        return _empty_csr(n, device=device)

    ip_p, ix_p = _csr(total_pages, 1)
    ip_e, ix_e = _csr(total_tokens, 2)

    return dict(
        kernel=dict(
            q_nope=qn, q_rope=qr, unified_kv_nope=ukn, unified_kv_rope=ukr,
            kv_indices_prefix=ix_p, kv_indptr_prefix=ip_p,
            kv_nope=kn, kv_rope=kr, kv_indices_extend=ix_e, kv_indptr_extend=ip_e,
            attn_sink=sink,
        ),
        ref=dict(
            q_dense=q_dense, ukv_dense=uk_dense, kv_dense=k_dense,
            ip_p=ip_p, ix_p=ix_p, ip_e=ip_e, ix_e=ix_e, sink=sink,
        ),
    )


@perftest()
def _profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


@benchmark()
def run_case(n, h, total_pages, total_tokens, *, mode="sparse", seed=0,
             verify=True, bench=True) -> Optional[dict]:
    if _skip_if_unsupported():
        return None
    data = _make_inputs(n, h, total_pages, total_tokens, mode=mode, seed=seed)
    softmax_scale = 1.0 / math.sqrt(D_HEAD)
    nnz_p = int(data["kernel"]["kv_indices_prefix"].numel())
    nnz_e = int(data["kernel"]["kv_indices_extend"].numel())
    row: dict = {"nnz_prefix": nnz_p, "nnz_extend": nnz_e}

    if verify:
        ref = _ref(**data["ref"], softmax_scale=softmax_scale)
        got = pa_sparse_prefill_fp8_opus(**data["kernel"], softmax_scale=softmax_scale)
        checkAllclose(
            got, ref, rtol=3e-2, atol=3e-2,
            msg=f"[N={n} H={h} pages={total_pages} tokens={total_tokens} mode={mode}]",
        )

    if bench:
        _, lat_us = _profile_func(
            pa_sparse_prefill_fp8_opus, **data["kernel"], softmax_scale=softmax_scale
        )
        total_nnz = nnz_p + nnz_e
        flops = 4.0 * h * total_nnz * D_HEAD
        row["latency_us"] = round(float(lat_us), 2)
        row["TFLOPS"] = round(float(flops / max(lat_us * 1e-6, 1e-12) / 1e12), 2)
    return row


_PYTEST_SHAPES = [
    (64, 16, 256, 256),
    (128, 16, 256, 256),
    (64, 32, 1024, 1024),
]
_PYTEST_MODES = ["sparse", "dense", "empty"]


@pytest.mark.parametrize(
    "n,h,total_pages,total_tokens",
    _PYTEST_SHAPES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
@pytest.mark.parametrize("mode", _PYTEST_MODES)
def test_pa_sparse_prefill_fp8_opus(n, h, total_pages, total_tokens, mode):
    run_case(
        n=n, h=h, total_pages=total_pages, total_tokens=total_tokens, mode=mode,
        seed=(hash((n, h, total_pages, total_tokens, mode)) & 0xFFFF),
        verify=True, bench=False,
    )


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-n", "--n_tokens", type=int, nargs="*", default=[1024, 4096])
parser.add_argument("--h_q", type=int, nargs="*", default=[16, 32])
parser.add_argument("--total_pages", type=int, nargs="*", default=[4096])
parser.add_argument("--total_tokens", type=int, default=None)
parser.add_argument("--mode", type=str, nargs="*", default=["sparse", "dense"], choices=list(_MODES))
parser.add_argument("--no-verify", action="store_true")
parser.add_argument("--no-bench", action="store_true")
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    rows = []
    for n, h, mode, pages in itertools.product(
        args.n_tokens, args.h_q, args.mode, args.total_pages
    ):
        total_tokens = args.total_tokens if args.total_tokens is not None else n
        r = run_case(
            n=n, h=h, total_pages=pages if pages > 0 else n, total_tokens=total_tokens,
            mode=mode, seed=args.seed, verify=not args.no_verify, bench=not args.no_bench,
        )
        if r:
            rows.append(r)
    if rows:
        df = pd.DataFrame(rows)
        drop_cols = [c for c in ("verify", "bench", "seed") if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        print()
        print(df.to_string(index=False))
    sys.exit(0)
