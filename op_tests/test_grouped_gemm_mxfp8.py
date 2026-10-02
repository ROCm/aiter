# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and guard tests for ``grouped_gemm_mxfp8_hip_fwd`` (gfx950).

Reference: dequantize the **same** MXFP8 e4m3 payload + e8m0 1×32 block scales the
kernel consumes, then bf16 matmul in fp32 — same bar style as decode small-M MXFP8 PRs.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

from aiter.ops.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from aiter.ops.triton.quant.quant import dynamic_mxfp8_quant
from aiter.utility.fp4_utils import e8m0_to_f32


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(0)
    arch = str(getattr(p, "gcn_arch_name", None) or getattr(p, "gcnArchName", "") or "")
    return "gfx950" in arch.lower()


requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(),
    reason="grouped_gemm_mxfp8 HIP tile requires gfx950 (device path under __gfx950__)",
)


def _mxfp8_dequant_blocked(
    x_fp8: torch.Tensor,
    scale_fp8: torch.Tensor,
    *,
    block: int = 32,
) -> torch.Tensor:
    """OCP MXFP8: per-block scale (e8m0) × fp8 e4m3 block values → fp32."""
    assert x_fp8.shape[-1] % block == 0
    *lead, k = x_fp8.shape
    nb = k // block
    assert scale_fp8.shape == tuple(lead + [nb]), (scale_fp8.shape, tuple(lead + [nb]))
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
    """Grouped NT GEMM reference (G>=1): for each expert ``gi`` the rows
    ``[start, start+rows)`` of A multiply that expert's weight ``b[gi]``::

        C[m, n] = sum_k dq(a)[m, k] * dq(b)[gi, n, k]   for m in expert gi
    """
    outs = []
    start = 0
    for gi, rows in enumerate(per_expert_rows):
        a_dq = _mxfp8_dequant_blocked(a[start : start + rows], a_scale[start : start + rows])
        b_dq = _mxfp8_dequant_blocked(b[gi], b_scale[gi])
        outs.append(a_dq @ b_dq.transpose(0, 1))
        start += rows
    return torch.cat(outs, dim=0)


def _build_grouped_metadata(
    per_expert_rows: list[int],
    n: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the flat-grid grouped-GEMM metadata the kernel consumes.

    * ``group_offs`` [G+1] int64 — prefix sum of per-expert rows.
    * ``tile_offs``  [G+1] int32 — prefix sum of per-expert tile counts, where
      tiles_per_expert = ceil(M_g / 256) * ceil(N / 256).
    * ``block_to_expert`` [total_tiles] int32 — expert id for each flat tile.
    """
    tiles_n = (n + 255) // 256
    group_offs = [0]
    tile_offs = [0]
    block_to_expert: list[int] = []
    for gi, rows in enumerate(per_expert_rows):
        group_offs.append(group_offs[-1] + rows)
        per_expert_tiles = ((rows + 255) // 256) * tiles_n
        block_to_expert.extend([gi] * per_expert_tiles)
        tile_offs.append(tile_offs[-1] + per_expert_tiles)
    return (
        torch.tensor(group_offs, device=device, dtype=torch.int64),
        torch.tensor(block_to_expert, device=device, dtype=torch.int32),
        torch.tensor(tile_offs, device=device, dtype=torch.int32),
    )


def _prepare_grouped_inputs(
    per_expert_rows: list[int],
    n: int,
    k: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize random bf16 A/B into the MXFP8 e4m3 + e8m0 layout the kernel wants."""
    g = len(per_expert_rows)
    m_total = sum(per_expert_rows)
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
    return a, b, a_scale, b_scale


# Multi-expert mix: per-expert row counts vary and each is a multiple of 16
# (scale-preshuffle alignment). Some experts span >1 M-tile (>256 rows), some
# are partial (<256) — exercising the grouped tile->expert dispatch rather than
# a degenerate single-GEMM (G=1) case. Sum = 1712 (divisible by 16).
_MULTI_EXPERT_ROWS = [256, 128, 272, 64, 512, 16, 320, 144]


@requires_gfx950
def test_grouped_gemm_mxfp8_multi_expert_matches_mx_dequant_ref():
    """Multi-expert (G=8) correctness vs MX-dequant reference."""
    device = "cuda"
    n, k = 256, 384
    per_expert_rows = _MULTI_EXPERT_ROWS
    g = len(per_expert_rows)
    m_total = sum(per_expert_rows)
    assert m_total % 16 == 0 and (g * n) % 16 == 0

    a, b, a_scale, b_scale = _prepare_grouped_inputs(
        per_expert_rows, n, k, device, seed=1
    )
    group_offs, block_to_expert, tile_offs = _build_grouped_metadata(
        per_expert_rows, n, device
    )

    out = grouped_gemm_mxfp8_hip_fwd(
        a,
        b,
        a_scale,
        b_scale,
        group_offs,
        block_to_expert,
        tile_offs,
        torch.bfloat16,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()

    ref = _grouped_ref(a, b, a_scale, b_scale, per_expert_rows)
    got = out.float().reshape(-1)
    r = ref.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(r.unsqueeze(0), got.unsqueeze(0)).item()
    assert cos >= 0.999, f"cosine_similarity={cos}"
    den = r.abs().clamp(min=1e-4)
    max_rel = ((got - r).abs() / den).max().item()
    assert max_rel < 0.05, f"max_rel_err={max_rel}"


@requires_gfx950
def test_grouped_gemm_mxfp8_deterministic():
    """Repeated launches on identical inputs must be bitwise-identical.

    Grouped GEMM kernels are prone to LDS / accumulator races that surface as
    run-to-run output drift; comparing many launches with ``torch.equal`` makes
    such a race a hard failure rather than an occasional cosine wobble.
    """
    device = "cuda"
    n, k = 256, 384
    per_expert_rows = _MULTI_EXPERT_ROWS

    a, b, a_scale, b_scale = _prepare_grouped_inputs(
        per_expert_rows, n, k, device, seed=7
    )
    group_offs, block_to_expert, tile_offs = _build_grouped_metadata(
        per_expert_rows, n, device
    )

    def _run() -> torch.Tensor:
        o = grouped_gemm_mxfp8_hip_fwd(
            a,
            b,
            a_scale,
            b_scale,
            group_offs,
            block_to_expert,
            tile_offs,
            torch.bfloat16,
        )
        torch.cuda.synchronize()
        return o

    ref = _run()
    assert torch.isfinite(ref).all()
    for i in range(1, 50):
        cur = _run()
        assert torch.equal(cur, ref), f"non-deterministic output at iteration {i}"


def _bad_k_inputs(m_total: int, n: int, k_bad: int, g: int, device: str) -> tuple:
    """Zero-filled multi-expert inputs for envelope/guard tests (K rejected on
    the host before any compute). Per-expert rows split evenly across G."""
    assert m_total % g == 0
    rows = m_total // g
    a = torch.zeros(m_total, k_bad, device=device, dtype=torch.float8_e4m3fn)
    b = torch.zeros(g, n, k_bad, device=device, dtype=torch.float8_e4m3fn)
    sc = (k_bad + 31) // 32
    z = torch.zeros(m_total, sc, device=device, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    zb = torch.zeros(g, n, sc, device=device, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    go = torch.tensor(
        [rows * i for i in range(g + 1)], device=device, dtype=torch.int64
    )
    to = torch.tensor(list(range(g + 1)), device=device, dtype=torch.int32)
    b2e = torch.arange(g, device=device, dtype=torch.int32)
    return a, b, z, zb, go, b2e, to


@requires_gfx950
def test_grouped_gemm_mxfp8_k_below_min_raises():
    device = "cuda"
    # 320 % 32 == 0 but < 384; G=2 multi-expert.
    a, b, z, zb, go, b2e, to = _bad_k_inputs(256, 128, 320, 2, device)
    with pytest.raises(RuntimeError, match="K must be >= 384"):
        grouped_gemm_mxfp8_hip_fwd(a, b, z, zb, go, b2e, to, torch.bfloat16)


@requires_gfx950
def test_grouped_gemm_mxfp8_k_not_multiple_of_32_raises():
    device = "cuda"
    # 370 % 32 != 0; G=2 multi-expert.
    a, b, z, zb, go, b2e, to = _bad_k_inputs(256, 128, 370, 2, device)
    with pytest.raises(RuntimeError, match="K must be multiple"):
        grouped_gemm_mxfp8_hip_fwd(a, b, z, zb, go, b2e, to, torch.bfloat16)
