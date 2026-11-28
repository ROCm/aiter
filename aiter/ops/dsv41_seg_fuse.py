# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1 shared-expert segment fusion kernels.

Its own JIT module (``module_dsv41_seg_fuse``): nothing here is imported by
``aiter/__init__`` and nothing here touches an existing module's sources, so a
process that never calls these ops never loads or builds it.

``silu_mul_mxfp8``: the shared expert's clamped SiLU-mul and the group-32 MXFP8
quant of its BF16 result in one launch, bit for bit ``silu_and_mul`` followed by
a group-32 MXFP8 quant of the BF16 result (finite inputs): the quant is the
one ATOM's ``quantize_fp8`` and Mega-MHC's ``out_dtype="mxfp8"`` epilogue use
(amax of the BF16 values, floor 1e-10; E8M0 = ceil_pow2(amax / 448); E4M3 RNE).
Unlike ``silu_and_mul_quant`` (FP32 scale, quantizes the FP32 product) the
product is rounded to BF16 before the quant and the scale is E8M0, so the
codes match the two-launch pair exactly.

Kernel ported from the DeepSeek-V4.1 DEP8 AgentX PoC (plugin_poc PR #129,
Bin Ding).
"""

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility.dtypes import get_dtype_fp8


@compile_ops("module_dsv41_seg_fuse", fc_name="silu_mul_mxfp8", develop=True)
def _silu_mul_mxfp8_kernel(
    out: Tensor, scale: Tensor, input: Tensor, limit: float = 0.0
) -> None: ...


def silu_mul_mxfp8(
    input: Tensor,
    limit: float = 0.0,
    out: Tensor | None = None,
    scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """``input`` ``[..., 2 * d]`` bf16 (``[gate | up]``) -> ``(q, scale)``:
    ``q`` ``[..., d]`` e4m3 and ``scale`` ``[..., d / 32]`` e8m0, row-major.

    gfx950 only (the kernel body is compiled for gfx950 alone).
    """
    if input.dtype != torch.bfloat16 or not input.is_contiguous():
        raise ValueError("silu_mul_mxfp8 needs a contiguous BF16 input")
    d2 = input.shape[-1]
    if d2 <= 0 or d2 % 64:
        raise ValueError("silu_mul_mxfp8 needs input [..., 2 * d], d a multiple of 32")
    d = d2 // 2
    lead = input.shape[:-1]
    if out is None:
        out = torch.empty((*lead, d), dtype=get_dtype_fp8(), device=input.device)
    if scale is None:
        scale = torch.empty(
            (*lead, d // 32), dtype=torch.float8_e8m0fnu, device=input.device
        )
    _silu_mul_mxfp8_kernel(out, scale.view(torch.uint8), input, float(limit))
    return out, scale
