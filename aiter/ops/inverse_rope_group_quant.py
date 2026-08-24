# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 attention-output inverse RoPE fused with FP8 group quant.

Its own JIT module: this sits on the attention *output* path and shares no
kernel with the QK-norm/RoPE input-path ops, so folding it into their 6k-line
translation unit would only make every edit there rebuild it.
"""

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility.dtypes import get_dtype_fp8


@compile_ops(
    "module_inverse_rope_group_quant",
    fc_name="inverse_rope_group_quant",
    develop=True,
)
def _inverse_rope_group_quant_kernel(
    o: Tensor,
    x_fp8: Tensor,
    x_scale: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    num_groups: int,
    quant_group_size: int = 128,
    scale_shuffle: bool = False,
) -> None: ...


def inverse_rope_group_quant(
    o: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    num_groups: int,
    quant_group_size: int = 128,
    scale_shuffle: bool = False,
    x_fp8: Tensor | None = None,
    x_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Inverse-RoPE V4 attention output and emit FP8 group-quant rows.

    Args:
        o: ``[S, H, head_dim]`` bf16/fp16 attention output before inverse RoPE.
        positions: ``[S]`` absolute positions.
        cos_cache/sin_cache: ``[max_pos, rd//2]``.
        num_groups: output-LoRA local groups ``G``.
        quant_group_size: quant block along ``D``; V4 wo_a path uses 128.
        scale_shuffle: when True, emit scale in MFMA tile-shuffled layout
            ``[G, S_pad, Ks_pad]`` for ``V_MFMA_SCALE_F32_16x16x128_F8``.
            When False, emit row-major ``[S, G, Ks]``.

    Returns:
        ``(x_fp8, x_scale)``.
    """
    assert o.dim() == 3, f"o must be [S,H,Dh], got {tuple(o.shape)}"
    S, H, head_dim = o.shape
    assert (
        H * head_dim
    ) % num_groups == 0, (
        f"H*head_dim={H * head_dim} must be divisible by num_groups={num_groups}"
    )
    D = (H * head_dim) // num_groups
    assert (
        D % quant_group_size == 0
    ), f"per-group D={D} must be divisible by quant_group_size={quant_group_size}"

    for name, cache in (("cos_cache", cos_cache), ("sin_cache", sin_cache)):
        assert cache.dim() == 2, (
            f"{name} must be 2D [max_pos, rd//2], got {tuple(cache.shape)}; "
            "reshape a cache carrying singleton batch/head dims at the call site"
        )

    from .. import dtypes

    if x_fp8 is None:
        x_fp8 = torch.empty((S, num_groups, D), dtype=get_dtype_fp8(), device=o.device)
    scale_groups = D // quant_group_size
    if x_scale is None:
        if scale_shuffle:
            S_pad = ((S + 31) // 32) * 32
            Ks_pad = ((scale_groups + 7) // 8) * 8
            x_scale = torch.full(
                (num_groups, S_pad, Ks_pad),
                0x7F,
                dtype=dtypes.fp8_e8m0,
                device=o.device,
            )
        else:
            x_scale = torch.empty(
                (S, num_groups, scale_groups),
                dtype=dtypes.fp8_e8m0,
                device=o.device,
            )

    _inverse_rope_group_quant_kernel(
        o,
        x_fp8,
        x_scale,
        positions,
        cos_cache,
        sin_cache,
        num_groups,
        quant_group_size,
        scale_shuffle,
    )
    return x_fp8, x_scale
