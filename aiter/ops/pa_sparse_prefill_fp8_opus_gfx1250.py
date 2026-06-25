# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250-ONLY split-precision (NoPE fp8 / RoPE bf16) sparse paged prefill
attention for DeepSeek-V4 DSA.

Parallel implementation to ``pa_sparse_prefill_fp8_opus`` (gfx950), but compiled
as its own module and gated to gfx1250 (wave32 WMMA). Same signature/semantics.

See ``aiter/csrc/include/pa_sparse_prefill_fp8_opus_gfx1250.h`` for the C++ API.
"""

import torch
from typing import Optional

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

MD_NAME = "module_pa_sparse_prefill_fp8_opus_gfx1250"


@compile_ops("module_pa_sparse_prefill_fp8_opus_gfx1250", develop=True)
def pa_sparse_prefill_fp8_opus_gfx1250_fwd(
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
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _pa_sparse_prefill_fp8_opus_gfx1250_fake(
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
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(
    mutates_args=["out"], gen_fake=_pa_sparse_prefill_fp8_opus_gfx1250_fake
)
def pa_sparse_prefill_fp8_opus_gfx1250(
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
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """gfx1250 sparse prefill attention with split fp8 NoPE / bf16 RoPE inputs.

    Same signature/semantics as ``pa_sparse_prefill_fp8_opus`` but gated to
    gfx1250. Returns ``out`` (``[T, H, 512]`` bf16).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx1250":
        raise RuntimeError(
            f"pa_sparse_prefill_fp8_opus_gfx1250 requires gfx1250, got {gfx}"
        )

    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")

    t, h = q_nope.shape[0], q_nope.shape[1]
    if out is None:
        out = torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)
    elif out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )

    pa_sparse_prefill_fp8_opus_gfx1250_fwd(
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
        out,
        float(softmax_scale),
    )
    return out


__all__ = [
    "pa_sparse_prefill_fp8_opus_gfx1250_fwd",
    "pa_sparse_prefill_fp8_opus_gfx1250",
]
