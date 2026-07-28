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
    transpose_scale: bool = False,
) -> None: ...


def inverse_rope_group_quant(
    o: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    num_groups: int,
    quant_group_size: int = 128,
    transpose_scale: bool = False,
    x_fp8: Tensor | None = None,
    x_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Inverse-RoPE V4 attention output and emit FP8 group-quant rows.

    Args:
        o: ``[S, H, head_dim]`` bf16/fp16 attention output before inverse RoPE.
        positions: ``[S]`` absolute positions.
        cos_cache/sin_cache: ``[max_pos, rd//2]``; a caller holding the singleton
            batch/head dims (ATOM's ``_V4RoPE``) reshapes at the call site, so
            this op stays framework-agnostic.
        num_groups: output-LoRA local groups ``G``.
        quant_group_size: quant block along ``D``; V4 wo_a path uses 128.
        transpose_scale: store the scale column-major, which is what the
            preshuffled-B blockscale GEMM reads; plain ``gemm_a8w8_blockscale``
            wants it false. Only the scale moves, ``x_fp8`` stays row-major.

    Returns:
        ``(x_fp8, x_scale)`` with logical shapes ``[S, G, D]`` and
        ``[S, G, D/group]``. Under ``transpose_scale`` the scale's storage is
        ``[G, D/group, S]``, so ``x_scale[:, g, :]`` is compact column-major.
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
        if transpose_scale:
            # Group-major [G, Ks, S] rather than [Ks, S, G] so each group's
            # [S, Ks] slice is itself compact column-major: wo_a runs one GEMM
            # per group, and a group-minor layout would hand it strides
            # (G, S*G) instead.
            x_scale = torch.empty(
                (num_groups, scale_groups, S),
                dtype=dtypes.fp8_e8m0,
                device=o.device,
            ).permute(2, 0, 1)
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
        transpose_scale,
    )
    return x_fp8, x_scale
