# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hand-written HIP prefill-phase FP8 MQA indexer logits kernel for gfx950.

The dense half of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning indexer:

    logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]

K has already been gathered into a contiguous ``[N, 128]`` buffer, so there is no
block table. Drop-in for ``aiter.ops.triton.attention.fp8_mqa_logits``, so the two
can be A/B'd on identical inputs.

Fixed at n_heads=32, head_dim=128 -- the shipped GLM-5-FP8 indexer shape.
``is_supported()`` gates on that, so a caller that also has to serve other shapes
can route them to the Triton kernel rather than trip a TORCH_CHECK.
"""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

MD_NAME = "module_fp8_mqa_logits"

SUPPORTED_GFX = ("gfx950",)
NUM_HEADS = 32
HEAD_DIM = 128


def is_supported(num_heads: int, head_dim: int) -> bool:
    """True when this kernel can run this shape on this device."""
    return (
        get_gfx() in SUPPORTED_GFX and num_heads == NUM_HEADS and head_dim == HEAD_DIM
    )


@compile_ops(MD_NAME, fc_name="fp8_mqa_logits")
def fp8_mqa_logits(
    q_fp8: Tensor,
    k_fp8: Tensor,
    kv_scale: Tensor,
    weights: Tensor,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
    BlockM: int = 0,
    SplitN: int = 0,
    num_warps: int = 0,
    TotalCuCount: int = 256,
    clean_logits: bool = True,
    unroll2: int = -1,
    reverse_rows: int = -1,
    out: Tensor | None = None,
) -> Tensor:
    """Prefill indexer logits over a contiguous K buffer.

    q_fp8       [M, 32, 128] fp8   k_fp8    [N, 128] fp8
    kv_scale    [N] f32            weights  [M, 32] f32
    cu_seqlen_ks/ke [M] i32 -- row m is valid on [ks[m], ke[m]). Either bound may
    legally sit outside [0, N); the row is then empty over the part that does.

    The zero-valued tunables (BlockM, SplitN, num_warps) mean "use the host
    heuristic"; unroll2 and reverse_rows are tri-state, -1 for the heuristic and
    0/1 to force off/on. Writes into `out` when given (and returns it), otherwise allocates
    [M, N] f32; outside each row's window the kernel writes -inf when clean_logits,
    and leaves the buffer untouched otherwise.
    """
