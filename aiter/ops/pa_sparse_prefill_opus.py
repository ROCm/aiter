# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus-pinned MLA sparse prefill entries, kept for existing callers.

New code should use :mod:`aiter.ops.mla_sparse_prefill`: ``mla_sparse_prefill`` and
``mla_sparse_prefill_v4`` take the same arguments and also choose the backend.
"""

import torch

from ..jit.utils.torch_guard import torch_compile_guard
from .mla_sparse_prefill import _opus_sparse_prefill, _opus_sparse_prefill_v4
from .opus.mla import (
    opus_mla_v4_prefill_a8w8_gfx950_fwd as pa_sparse_prefill_fp8_gfx950_opus_fwd,
)
from .opus.mla import (
    opus_mla_v4_prefill_a8w8_gfx1250_fwd as pa_sparse_prefill_fp8_gfx1250_opus_fwd,
)
from .opus.mla import (
    opus_mla_v4_prefill_a16w16_gfx950_fwd as pa_sparse_prefill_gfx950_opus_fwd,
)
from .opus.mla import (
    opus_mla_v4_prefill_a16w16_gfx1250_fwd as pa_sparse_prefill_gfx1250_opus_fwd,
)


def pa_sparse_prefill_gfx950_opus_split_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    partial_o: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int,
) -> None:
    """:func:`pa_sparse_prefill_gfx950_opus_fwd` with split-K, in the former argument order."""
    pa_sparse_prefill_gfx950_opus_fwd(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        softmax_scale,
        num_splits,
        partial_o,
        partial_max,
        partial_sum,
    )


def _pa_sparse_prefill_opus_fake(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_sparse_prefill_opus_fake)
def pa_sparse_prefill_opus(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
) -> torch.Tensor:
    """:func:`aiter.ops.mla_sparse_prefill.mla_sparse_prefill` on the opus backend."""
    return _opus_sparse_prefill(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
        num_kv_splits,
    )


def _pa_sparse_prefill_fp8_opus_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_sparse_prefill_fp8_opus_fake)
def pa_sparse_prefill_fp8_opus(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`aiter.ops.mla_sparse_prefill.mla_sparse_prefill_v4` on the opus backend."""
    return _opus_sparse_prefill_v4(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
    )


__all__ = [
    "pa_sparse_prefill_fp8_gfx950_opus_fwd",
    "pa_sparse_prefill_fp8_gfx1250_opus_fwd",
    "pa_sparse_prefill_fp8_opus",
    "pa_sparse_prefill_gfx950_opus_fwd",
    "pa_sparse_prefill_gfx950_opus_split_fwd",
    "pa_sparse_prefill_gfx1250_opus_fwd",
    "pa_sparse_prefill_opus",
]
