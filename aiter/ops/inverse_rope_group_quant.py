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

# Mirrors aiter::ScaleLayout in csrc/include/inverse_rope_group_quant.h.
# "mfma" is a legacy alias for mfma_tile (gfx950 V_MFMA_SCALE_16x16x128 tile).
SCALE_LAYOUTS = {"row": 0, "mfma": 1, "mfma_tile": 1, "n32k4": 2}


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
    scale_layout: int = 0,
) -> None: ...


def scale_shape(
    s: int, num_groups: int, scale_groups: int, scale_layout: str = "row"
) -> tuple[int, ...]:
    """Shape of the ``x_scale`` buffer a given layout wants.

    Exposed so a caller pre-allocating its own scale (a HIP graph capture, or a
    persistent workspace) does not have to restate the padding rules.
    """
    if scale_layout in ("mfma", "mfma_tile"):
        return (num_groups, ((s + 31) // 32) * 32, ((scale_groups + 7) // 8) * 8)
    if scale_layout == "n32k4":
        return ((s + 31) // 32, num_groups, scale_groups * 32)
    return (s, num_groups, scale_groups)


def inverse_rope_group_quant(
    o: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    num_groups: int,
    quant_group_size: int = 128,
    scale_layout: str = "row",
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
        scale_layout: e8m0 scale storage, one of

            * ``"row"``: row-major ``[S, G, Ks]``.
            * ``"mfma_tile"`` (``"mfma"`` alias): gfx950
              ``V_MFMA_SCALE_F32_16x16x128_F8`` 256B tile layout
              ``[G, S_pad, Ks_pad]``. Not the gfx1250 WMMA layout.
            * ``"n32k4"``: gfx1250 WMMA scaleB layout ``[ceil(S,32)/32, G, Ks*32]``
              (``shuffle_scale_n32k4`` on weights). Needs ``quant_group_size == 32``
              and ``Ks % 4 == 0``; the ``n32`` is super-row height, not group size.

    Returns:
        ``(x_fp8, x_scale)``.
    """
    assert (
        scale_layout in SCALE_LAYOUTS
    ), f"scale_layout must be one of {sorted(SCALE_LAYOUTS)}, got {scale_layout!r}"
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
    if scale_layout == "n32k4":
        # A lane's WMMA scaleB operand is 4 e8m0 of one K=128 step, so the
        # groups come in fours and each must span 128/4 = 32 elements. Other
        # group sizes still produce a buffer that matches the layout formula --
        # nothing downstream would flag it -- but the GEMM would read four
        # different K steps' scales as one step's.
        assert quant_group_size == 32, (
            "n32k4 scale is only defined for quant_group_size == 32 (the "
            "consumer's WMMA-K=128 step is 4 groups of 32), got "
            f"{quant_group_size}"
        )
        assert (
            scale_groups % 4 == 0
        ), f"n32k4 scale needs Ks % 4 == 0, got Ks={scale_groups}"
    if x_scale is None:
        shape = scale_shape(S, num_groups, scale_groups, scale_layout)
        if scale_layout == "row":
            x_scale = torch.empty(shape, dtype=dtypes.fp8_e8m0, device=o.device)
        else:
            # Both padded layouts round S up to 32, and the tail rows are never
            # written; 0x7F (exponent 127, dequant scale 1.0) keeps a GEMM that
            # reads past S benign.
            x_scale = torch.full(shape, 0x7F, dtype=dtypes.fp8_e8m0, device=o.device)

    _inverse_rope_group_quant_kernel(
        o,
        x_fp8,
        x_scale,
        positions,
        cos_cache,
        sin_cache,
        num_groups,
        quant_group_size,
        SCALE_LAYOUTS[scale_layout],
    )
    return x_fp8, x_scale
