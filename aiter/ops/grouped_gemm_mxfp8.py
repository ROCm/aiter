# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""JIT HIP MX-FP8 grouped GEMM (MoE prefill path, gfx950 tile).

Bound to pybind symbol ``grouped_gemm_mxfp8_hip_fwd`` in
``csrc/grouped_gemm_mxfp8/grouped_gemm_mxfp8.cu`` (``.cu`` is the aiter
JIT convention; ROCm builds hipify it like other ``csrc`` kernels).

``grouped_gemm_mxfp8`` is the dispatch entry point: it returns the HIP result
when the fixed 256x256x128 tile is both *supported* (envelope) and *expected to
win* (large-M prefill regime), otherwise it returns ``None`` so the caller can
fall back to its existing Triton path.
"""

from __future__ import annotations

import csv
import functools
import os
from typing import Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

# Backend override: "auto" (default, tuned table + heuristic), "hip" (force HIP
# whenever supported), or "triton" (always fall back).
_BACKEND_ENV = "AITER_GROUPED_GEMM_MXFP8_BACKEND"

# MX scale block (e8m0 scales one value per 32-element K group).
_MX_BLOCK_SIZE = 32

# HIP-vs-Triton crossover table produced by op_tests/tune_grouped_gemm_mxfp8.py.
_TUNED_CSV = os.path.join(
    os.path.dirname(__file__), os.pardir, "configs", "grouped_gemm_mxfp8_tuned.csv"
)


def _gen_grouped_gemm_mxfp8_fake(
    a: Tensor,
    b: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    group_offs: Tensor,
    block_to_expert: Tensor,
    tile_offs: Tensor,
    out_dtype: torch.dtype,
) -> Tensor:
    m_total = a.size(0)
    n = b.size(1)
    return torch.empty((m_total, n), dtype=out_dtype, device=a.device)


@compile_ops(
    "module_grouped_gemm_mxfp8",
    fc_name="grouped_gemm_mxfp8_hip_fwd",
    gen_fake=_gen_grouped_gemm_mxfp8_fake,
)
def grouped_gemm_mxfp8_hip_fwd(
    a: Tensor,
    b: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    group_offs: Tensor,
    block_to_expert: Tensor,
    tile_offs: Tensor,
    out_dtype: torch.dtype,
) -> Tensor: ...


def _supported(
    a: Tensor,
    b: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    group_offs: Tensor,
    block_to_expert: Tensor,
    tile_offs: Tensor,
    out_dtype: torch.dtype,
) -> bool:
    """Envelope check: mirrors the host-side TORCH_CHECKs in the .cu so the
    dispatch never calls into a kernel that would throw or silently truncate."""
    if get_gfx() != "gfx950":
        return False
    if a.dim() != 2 or b.dim() != 3:
        return False
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return False
    if a_scale.dtype != torch.float8_e8m0fnu or b_scale.dtype != torch.float8_e8m0fnu:
        return False
    if out_dtype not in (torch.bfloat16, torch.float16):
        return False

    m_total, k = a.shape
    g, n = b.shape[0], b.shape[1]
    if b.shape[2] != k:
        return False
    # Kernel pipeline: K multiple of the MX block and >= 384 (>=3 K-iters).
    if k % _MX_BLOCK_SIZE != 0 or k < 384:
        return False
    if n % 16 != 0:
        return False
    # Scale preshuffle launches dim3(m_total / 16) / dim3((g * n) / 16): both
    # row counts must be divisible by 16 or tail rows are dropped.
    if m_total % 16 != 0 or (g * n) % 16 != 0:
        return False
    if group_offs.numel() != g + 1 or tile_offs.numel() != g + 1:
        return False
    if group_offs.dtype != torch.int64:
        return False
    if block_to_expert.dtype != torch.int32 or tile_offs.dtype != torch.int32:
        return False
    if not (
        a.is_contiguous()
        and b.is_contiguous()
        and a_scale.is_contiguous()
        and b_scale.is_contiguous()
    ):
        return False
    return True


@functools.lru_cache(maxsize=1)
def _load_tuned_table() -> tuple[dict, dict]:
    """Parse the tuned CSV into two lookups, both mapping to sorted
    ``[(M, use_hip)]`` lists:

    * ``exact``: keyed on ``(gfx, K, N, G)`` — the precise EP/expert-count row.
    * ``agnostic``: keyed on ``(gfx, K, N)`` — majority vote across G, used when
      the runtime G is not in the table. The crossover is ~invariant to G
      (verified across G=32/64/128), so this fallback is safe.

    Returns ``({}, {})`` if the CSV is absent."""
    exact: dict[tuple[str, int, int, int], list[tuple[int, int]]] = {}
    by_m: dict[tuple[str, int, int], dict[int, list[int]]] = {}
    path = os.path.normpath(_TUNED_CSV)
    if not os.path.exists(path):
        return exact, {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                gfx, k, n, g = row["gfx"], int(row["K"]), int(row["N"]), int(row["G"])
                m, use_hip = int(row["M"]), int(row["use_hip"])
            except (KeyError, ValueError):
                continue
            exact.setdefault((gfx, k, n, g), []).append((m, use_hip))
            by_m.setdefault((gfx, k, n), {}).setdefault(m, []).append(use_hip)
    for entries in exact.values():
        entries.sort()
    agnostic: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for key, mmap in by_m.items():
        agnostic[key] = sorted(
            (m, 1 if 2 * sum(flags) >= len(flags) else 0) for m, flags in mmap.items()
        )
    return exact, agnostic


def _floor_bucket(rows: list, per_expert: float) -> bool:
    """``use_hip`` flag at the largest M bucket <= per-expert tokens (clamped to
    the smallest bucket when below the table's range)."""
    flag = rows[0][1]
    for m, use_hip in rows:
        if m <= per_expert:
            flag = use_hip
        else:
            break
    return bool(flag)


def _lookup_tuned(
    gfx: str, k: int, n: int, g: int, per_expert: float
) -> Optional[bool]:
    """Tuned-table lookup. Tries the exact ``(gfx, K, N, G)`` row, then the
    G-agnostic ``(gfx, K, N)`` aggregate. Returns ``None`` on table miss."""
    exact, agnostic = _load_tuned_table()
    rows = exact.get((gfx, k, n, g)) or agnostic.get((gfx, k, n))
    if not rows:
        return None
    return _floor_bucket(rows, per_expert)


def _hip_preferred(m_total: int, n: int, k: int, g: int) -> bool:
    """Decide HIP vs Triton from per-expert token count. Consults the tuned
    crossover table first; falls back to a shape heuristic on table miss.

    The fixed 256x256x128 tile only beats Triton ``tl.dot_scaled`` once each
    expert has enough rows to fill the 256-row M tile, and the crossover M rises
    as N shrinks. Small-K gemm2 (K<=768, e.g. TP>=4 down-proj) never wins on the
    MI355X sweep, so the heuristic excludes it."""
    per_expert = m_total / max(g, 1)
    tuned = _lookup_tuned(get_gfx(), k, n, g, per_expert)
    if tuned is not None:
        return tuned
    # Fallback (shape not in tuned table): conservative, bench-derived.
    if k < 1024:
        return False
    crossover = 128 if n >= 3072 else (256 if n >= 1536 else 512)
    return per_expert >= crossover


def grouped_gemm_mxfp8(
    a: Tensor,
    b: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    group_offs: Tensor,
    block_to_expert: Tensor,
    tile_offs: Tensor,
    out_dtype: torch.dtype,
) -> Optional[Tensor]:
    """Dispatch wrapper. Returns the HIP grouped-GEMM output, or ``None`` to
    signal the caller to use its own (Triton) fallback.

    Contract (matches aiter decode small-M path #3783): ``None`` means "not
    engaging HIP" — either the shape is outside the kernel envelope, or the
    fixed tile is not expected to beat Triton for this M regime.
    """
    backend = os.environ.get(_BACKEND_ENV, "auto").lower()
    if backend == "triton":
        return None
    if backend not in ("auto", "hip"):
        backend = "auto"

    if not _supported(
        a, b, a_scale, b_scale, group_offs, block_to_expert, tile_offs, out_dtype
    ):
        return None

    if backend == "auto":
        m_total, k = a.shape
        g, n = b.shape[0], b.shape[1]
        if not _hip_preferred(m_total, n, k, g):
            return None

    return grouped_gemm_mxfp8_hip_fwd(
        a, b, a_scale, b_scale, group_offs, block_to_expert, tile_offs, out_dtype
    )
