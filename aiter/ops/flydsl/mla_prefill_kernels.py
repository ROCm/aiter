# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL dense absorbed MLA prefill API.

SILOTIGER-957 / Kimi: non-causal context over latent 576→512 without head
padding. Kernel lives in ``kernels.mla_prefill_dense``.
"""

from __future__ import annotations

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.mla_prefill_dense import (
    DEFAULT_BLOCK_N,
    DEFAULT_WAVES_PER_EU,
    QK_HEAD_DIM,
    SUPPORTED_GFX,
    V_HEAD_DIM,
    flydsl_mla_prefill_dense_fwd,
)

__all__ = [
    "QK_HEAD_DIM",
    "V_HEAD_DIM",
    "flydsl_mla_prefill_fwd",
    "flydsl_mla_prefill_supported",
]


def flydsl_mla_prefill_supported(
    *,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.bfloat16,
    qk_dim: int = QK_HEAD_DIM,
    v_dim: int = V_HEAD_DIM,
) -> bool:
    """Return True when this build can run the dense absorb prefill kernel."""
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        return False
    if q_dtype != torch.bfloat16 or kv_dtype != torch.bfloat16:
        return False
    if qk_dim != QK_HEAD_DIM or v_dim != V_HEAD_DIM:
        return False
    return True


def flydsl_mla_prefill_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float | None = None,
    *,
    is_causal: bool = False,
    block_n: int = DEFAULT_BLOCK_N,
    waves_per_eu: int = DEFAULT_WAVES_PER_EU,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Dense absorb MLA prefill (bf16). See ``flydsl_mla_prefill_dense_fwd``."""
    if not flydsl_mla_prefill_supported(
        q_dtype=q.dtype,
        kv_dtype=kv_buffer.dtype,
        qk_dim=q.shape[-1],
        v_dim=o.shape[-1],
    ):
        raise RuntimeError(
            "flydsl_mla_prefill_fwd unsupported for "
            f"gfx={get_gfx()} q={q.dtype} kv={kv_buffer.dtype} "
            f"qk={q.shape[-1]} v={o.shape[-1]}"
        )
    return flydsl_mla_prefill_dense_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        is_causal=is_causal,
        block_n=block_n,
        waves_per_eu=waves_per_eu,
        stream=stream,
    )
