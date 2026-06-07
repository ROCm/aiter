# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

r"""FP8 (DeepSeek-V4 / asm-v4 layout) path for OPUS sparse paged prefill attention.

Head dim D = 512 = NOPE(448, FP8 e4m3, per-64-tile e8m0 scale) + ROPE(64, BF16),
for both Q and KV, mirroring aiter PR #3112 (asm v4 MLA decode).

This test drives the *correctness oracle* and the data path end-to-end:

  bf16 ground-truth q/kv  --quantize_to_v4_fp8-->  (nope fp8, rope bf16, scale fp32)
        |                                                    |
        v (fp8-dequant reference)                            v (kernel-under-test)
  dequantize_v4_fp8 -> existing fp32 SDPA reference     dequant -> attention kernel
                         \________________ checkAllclose ________________/

Milestone v1 (this file): the "kernel-under-test" dequants on-GPU and runs the
*existing, proven* bf16 ``pa_sparse_prefill_opus``. This validates the v4 fp8
format, the scales, and the reference wiring on real gfx950 hardware before the
fused FP8 ``__global__`` lands. When the fused kernel is ready, only
``_run_kernel_under_test`` changes (swap in ``pa_sparse_prefill_opus_fp8``).
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

import aiter  # noqa: F401
from aiter.ops.pa_sparse_prefill_opus import (
    pa_sparse_prefill_opus,
    pa_sparse_prefill_opus_fp8,
    pa_sparse_prefill_opus_fp8_fused,
)
from aiter.test_common import benchmark, checkAllclose, perftest

# Reuse the bf16 test's validated reference + CSR generators + skip helpers.
from test_pa_sparse_prefill_opus import (  # noqa: E402
    _dense_csr,
    _empty_csr,
    _random_csr,
    _ref_pa_sparse_prefill_opus,
    _skip_if_unsupported,
)
from pa_sparse_prefill_opus_fp8_quant import (  # noqa: E402
    D_FULL,
    D_NOPE,
    D_ROPE,
    NUM_TILES,
    dequantize_v4_fp8,
    quantize_to_v4_fp8,
)

_MODES = ("sparse", "dense", "empty")


def _make_fp8_inputs(
    n: int,
    h: int,
    total_pages: int,
    total_tokens: int,
    *,
    mode: str = "sparse",
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict:
    """Build bf16 ground-truth q/unified_kv/kv, quantize to the v4 fp8 layout,
    and return everything both the reference (dequant) and the kernel need.

    Returns a dict with:
      bf16 ground truth: q_bf16, unified_kv_bf16, kv_bf16
      fp8 packed:        q_nope/q_rope/q_scale, ukv_nope/ukv_rope/ukv_scale,
                         kv_nope/kv_rope/kv_scale
      csr + sink:        kv_indices/indptr (prefix+extend), attn_sink
    """
    assert mode in _MODES
    torch.manual_seed(seed)
    device = torch.device(device)

    q = (torch.randn(n, h, D_FULL, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    unified_kv = (torch.randn(total_pages, D_FULL, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    kv = (torch.randn(total_tokens, D_FULL, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    attn_sink = torch.randn(h, device=device, dtype=torch.float32) * 0.25

    def _csr(total_rows: int, seed_offset: int):
        if mode == "sparse":
            return _random_csr(n, total_rows, device=device, seed=seed * 2 + seed_offset)
        if mode == "dense":
            return _dense_csr(n, total_rows, device=device)
        return _empty_csr(n, device=device)

    ip_p, ix_p = _csr(total_pages, 1)
    ip_e, ix_e = _csr(total_tokens, 2)

    q_nope, q_rope, q_scale = quantize_to_v4_fp8(q)
    ukv_nope, ukv_rope, ukv_scale = quantize_to_v4_fp8(unified_kv)
    kv_nope, kv_rope, kv_scale = quantize_to_v4_fp8(kv)

    return dict(
        q_bf16=q,
        unified_kv_bf16=unified_kv,
        kv_bf16=kv,
        q_nope=q_nope, q_rope=q_rope, q_scale=q_scale,
        ukv_nope=ukv_nope, ukv_rope=ukv_rope, ukv_scale=ukv_scale,
        kv_nope=kv_nope, kv_rope=kv_rope, kv_scale=kv_scale,
        kv_indices_prefix=ix_p, kv_indptr_prefix=ip_p,
        kv_indices_extend=ix_e, kv_indptr_extend=ip_e,
        attn_sink=attn_sink,
    )


def _ref_from_fp8(inp: dict, softmax_scale: float) -> torch.Tensor:
    """Dequantize the fp8 tensors the kernel sees, then run the validated
    fp32 SDPA reference. Isolates 'kernel math' from 'fp8 quant noise'."""
    q = dequantize_v4_fp8(inp["q_nope"], inp["q_rope"], inp["q_scale"])
    ukv = dequantize_v4_fp8(inp["ukv_nope"], inp["ukv_rope"], inp["ukv_scale"])
    kv = dequantize_v4_fp8(inp["kv_nope"], inp["kv_rope"], inp["kv_scale"])
    return _ref_pa_sparse_prefill_opus(
        q=q,
        unified_kv=ukv,
        kv_indices_prefix=inp["kv_indices_prefix"],
        kv_indptr_prefix=inp["kv_indptr_prefix"],
        kv=kv,
        kv_indices_extend=inp["kv_indices_extend"],
        kv_indptr_extend=inp["kv_indptr_extend"],
        attn_sink=inp["attn_sink"],
        softmax_scale=softmax_scale,
    )


def _run_kernel_under_test(inp: dict, softmax_scale: float) -> torch.Tensor:
    """The fp8 device op: on-GPU dequant (standalone kernel) -> bf16 attention."""
    return pa_sparse_prefill_opus_fp8(
        q_nope=inp["q_nope"],
        q_rope=inp["q_rope"],
        q_scale=inp["q_scale"],
        unified_kv_nope=inp["ukv_nope"],
        unified_kv_rope=inp["ukv_rope"],
        unified_kv_scale=inp["ukv_scale"],
        kv_nope=inp["kv_nope"],
        kv_rope=inp["kv_rope"],
        kv_scale=inp["kv_scale"],
        kv_indices_prefix=inp["kv_indices_prefix"],
        kv_indptr_prefix=inp["kv_indptr_prefix"],
        kv_indices_extend=inp["kv_indices_extend"],
        kv_indptr_extend=inp["kv_indptr_extend"],
        attn_sink=inp["attn_sink"],
        softmax_scale=softmax_scale,
    )


@benchmark()
def run_fp8(
    n: int,
    h: int,
    total_pages: int,
    total_tokens: int,
    *,
    mode: str = "sparse",
    seed: int = 0,
    verify: bool = True,
) -> Optional[dict]:
    if _skip_if_unsupported(d=D_FULL):
        return None
    inp = _make_fp8_inputs(n, h, total_pages, total_tokens, mode=mode, seed=seed)
    softmax_scale = 1.0 / math.sqrt(D_FULL)

    row = {
        "nnz_prefix": int(inp["kv_indices_prefix"].numel()),
        "nnz_extend": int(inp["kv_indices_extend"].numel()),
    }
    if verify:
        ref = _ref_from_fp8(inp, softmax_scale)
        got = _run_kernel_under_test(inp, softmax_scale)
        # fp8 e4m3 tolerance (matches the v4 nm accuracy convention ~3e-2).
        checkAllclose(
            got, ref, rtol=4e-2, atol=4e-2,
            msg=f"[fp8 N={n} H={h} D={D_FULL} pages={total_pages} tokens={total_tokens} mode={mode}]",
        )
    return row


_PYTEST_SHAPES = [
    (64, 16, 256, 256),
    (128, 32, 256, 256),
    (64, 64, 1024, 1024),
    (256, 128, 2048, 2048),
]
_PYTEST_MODES = ["sparse", "dense", "empty"]


@pytest.mark.parametrize("mode", _PYTEST_MODES)
@pytest.mark.parametrize(
    "n,h,total_pages,total_tokens",
    _PYTEST_SHAPES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_pa_sparse_prefill_opus_fp8(n, h, total_pages, total_tokens, mode):
    run_fp8(
        n=n, h=h, total_pages=total_pages, total_tokens=total_tokens,
        mode=mode, seed=(hash((n, h, total_pages, total_tokens, mode)) & 0xFFFF),
        verify=True,
    )


def _run_fused(inp: dict, softmax_scale: float) -> torch.Tensor:
    return pa_sparse_prefill_opus_fp8_fused(
        q_nope=inp["q_nope"], q_rope=inp["q_rope"], q_scale=inp["q_scale"],
        unified_kv_nope=inp["ukv_nope"], unified_kv_rope=inp["ukv_rope"], unified_kv_scale=inp["ukv_scale"],
        kv_nope=inp["kv_nope"], kv_rope=inp["kv_rope"], kv_scale=inp["kv_scale"],
        kv_indices_prefix=inp["kv_indices_prefix"], kv_indptr_prefix=inp["kv_indptr_prefix"],
        kv_indices_extend=inp["kv_indices_extend"], kv_indptr_extend=inp["kv_indptr_extend"],
        attn_sink=inp["attn_sink"], softmax_scale=softmax_scale,
    )


@pytest.mark.parametrize("mode", _PYTEST_MODES)
@pytest.mark.parametrize(
    "n,h,total_pages,total_tokens",
    _PYTEST_SHAPES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_pa_sparse_prefill_opus_fp8_fused(n, h, total_pages, total_tokens, mode):
    if _skip_if_unsupported(d=D_FULL):
        return
    inp = _make_fp8_inputs(n, h, total_pages, total_tokens, mode=mode,
                           seed=(hash((n, h, total_pages, total_tokens, mode)) & 0xFFFF))
    softmax_scale = 1.0 / math.sqrt(D_FULL)
    ref = _ref_from_fp8(inp, softmax_scale)
    got = _run_fused(inp, softmax_scale)
    checkAllclose(
        got, ref, rtol=4e-2, atol=4e-2,
        msg=f"[fused fp8 N={n} H={h} pages={total_pages} tokens={total_tokens} mode={mode}]",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--n_tokens", type=int, nargs="*", default=[1024])
    p.add_argument("--h_q", type=int, nargs="*", default=[16, 32, 64, 128])
    p.add_argument("--total_pages", type=int, nargs="*", default=[4096])
    p.add_argument("--mode", type=str, nargs="*", default=["sparse", "dense"], choices=list(_MODES))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rows = []
    for n, h, mode, pages in itertools.product(args.n_tokens, args.h_q, args.mode, args.total_pages):
        r = run_fp8(n=n, h=h, total_pages=pages, total_tokens=n, mode=mode, seed=args.seed)
        if r:
            rows.append(r)
    if rows:
        print()
        print(pd.DataFrame(rows).to_string(index=False))
    sys.exit(0)
