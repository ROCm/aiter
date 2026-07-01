# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS-based dense GQA flash attention (head dim D=128) for gfx950.

Grouped-query scaled-dot-product attention with arbitrary sequence length
(non KV-tile / Q-tile aligned N supported), causal or non-causal, bf16.

The user-facing entry is :func:`fmha_fwd_hd128_bf16_opus`; it forwards to the JIT-compiled HIP
kernel via :func:`fmha_fwd_hd128_bf16_opus_fwd`.

Tensor layout (row-major, last dim contiguous):

* ``q``   : ``[B, N, H,    D]`` bf16
* ``k``   : ``[B, N, H_KV, D]`` bf16
* ``v``   : ``[B, N, H_KV, D]`` bf16
* ``out`` : ``[B, N, H,    D]`` bf16

``H`` must be a multiple of ``H_KV`` (the GQA group size). Only ``D == 128``
and bf16 are compiled.
"""

import math
from typing import Optional

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

MD_NAME = "module_fmha_fwd_hd128_bf16_opus"


@compile_ops("module_fmha_fwd_hd128_bf16_opus", develop=True)
def fmha_fwd_hd128_bf16_opus_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def _fmha_fwd_hd128_bf16_opus_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@torch_compile_guard(mutates_args=["out"], gen_fake=_fmha_fwd_hd128_bf16_opus_fake)
def fmha_fwd_hd128_bf16_opus(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense GQA attention (D=128, bf16), backed by the OPUS gfx950 HIP kernel.

    Args:
      q:             ``[B, N, H, D]`` bf16 query.
      k:             ``[B, N, H_KV, D]`` bf16 key.
      v:             ``[B, N, H_KV, D]`` bf16 value.
      causal:        causal mask if True, else full attention.
      softmax_scale: scale applied to QK^T. Defaults to ``1/sqrt(D)``.
      out:           optional ``[B, N, H, D]`` output buffer; allocated if None.

    Returns:
      ``out`` (``[B, N, H, D]`` same dtype as ``q``).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"fmha_fwd_hd128_bf16_opus requires gfx950, got {gfx}")

    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"fmha_fwd_hd128_bf16_opus expects bf16 q, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError(
            f"k/v dtype must match q: k={k.dtype}, v={v.dtype}, q={q.dtype}"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise RuntimeError("q/k/v must be 4-D [B, N, H(_KV), D]")
    if q.size(-1) != 128:
        raise RuntimeError(
            f"fmha_fwd_hd128_bf16_opus only supports D=128, got D={q.size(-1)}"
        )

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={q.dtype}"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))

    fmha_fwd_hd128_bf16_opus_fwd(q, k, v, out, bool(causal), float(softmax_scale))
    return out


__all__ = [
    "fmha_fwd_hd128_bf16_opus_fwd",
    "fmha_fwd_hd128_bf16_opus",
]
