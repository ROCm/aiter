# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250-only backend for the split-precision (NoPE fp8 / RoPE bf16) sparse
paged prefill attention kernel.

This module only exposes the low-level ``*_fwd`` binding for the gfx1250
wave32/WMMA kernel. The public, arch-agnostic entry point is
``aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus``, which detects
the runtime arch and dispatches here on gfx1250.

See ``aiter/csrc/include/pa_sparse_prefill_fp8_opus_gfx1250.h`` for the C++ API.
"""

import torch

from ..jit.core import compile_ops


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


__all__ = [
    "pa_sparse_prefill_fp8_opus_gfx1250_fwd",
]
