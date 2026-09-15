# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hand-written HIP decode-phase FP8 paged MQA indexer logits kernel for gfx950.

The paged half of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning indexer:

    logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]

K and its per-token dequant scale are co-packed in a paged cache and gathered
through a block table. Drop-in for ``deepgemm_fp8_paged_mqa_logits``, so the two
can be A/B'd on identical inputs.

Fixed at n_heads=32, head_dim=128 -- the shipped GLM-5-FP8 indexer shape.
``is_supported()`` gates on that, so a caller that also has to serve other shapes
can route them to the Triton kernel rather than trip a TORCH_CHECK.
"""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

MD_NAME = "module_fp8_paged_mqa_logits"

SUPPORTED_GFX = ("gfx950",)
NUM_HEADS = 32
HEAD_DIM = 128
# KVBlockSize folds into the kernel as a compile-time constant, and each value is
# tied to one cache layout -- the pairing production uses.
SUPPORTED_KV_BLOCK_SIZES = (1, 64)


def is_supported(num_heads: int, head_dim: int, kv_block_size: int = 1) -> bool:
    """True when this kernel can run this shape/layout on this device."""
    return (
        get_gfx() in SUPPORTED_GFX
        and num_heads == NUM_HEADS
        and head_dim == HEAD_DIM
        and kv_block_size in SUPPORTED_KV_BLOCK_SIZES
    )


@compile_ops(MD_NAME, fc_name="fp8_paged_mqa_logits")
def fp8_paged_mqa_logits(
    q_fp8: Tensor,
    kv_cache_fp8: Tensor,
    weights: Tensor,
    context_lens: Tensor,
    block_tables: Tensor,
    max_model_len: int,
    ChunkK: int = 0,
    SplitKV: int = 0,
    num_warps: int = 0,
    TotalCuCount: int = 256,
    RowsPerBlock: int = 0,
    out: Tensor | None = None,
) -> Tensor:
    """Decode indexer logits over a paged KV cache.

    q_fp8        [batch, next_n, 32, 128] fp8
    kv_cache_fp8 [num_blocks, block_size, 1, index_dim] -- block_size*128 fp8 K
                 bytes followed by block_size f32 dequant scales, so
                 index_dim == 132 for the fp8 layout.
                 block_size must be 1 or 64, and the two take different layouts,
                 exactly as in production: block_size=1 takes the plain cache,
                 block_size=64 takes the `shuffle_weight(layout=(16, 16))`
                 preshuffled one.
    weights      [batch*next_n, 32] f32   context_lens [batch] i32
    block_tables [batch, max_blocks_per_seq] i32

    The zero-valued tunables (ChunkK, SplitKV, num_warps, RowsPerBlock) mean "use
    the host heuristic". Writes into `out` when given (and returns it), otherwise
    allocates [batch*next_n, max_model_len] f32. Only the causal window
    p <= context_lens[b] - next_n + n is written; the -inf outside it is the
    caller's, exactly as for `deepgemm_fp8_paged_mqa_logits`.
    """
