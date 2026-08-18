# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import Optional

from ..jit.core import compile_ops


@compile_ops("module_minimax_m3_fused_qknorm_rope_cache_shuffle", develop=True)
def minimax_m3_qknorm_rope_cache_shuffle_insert(
    qkv: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    rotary_dim: int,
    eps: float,
    slot_mapping: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    q_out: Tensor,
    index_q_norm_weight: Optional[Tensor] = None,
    index_k_norm_weight: Optional[Tensor] = None,
    index_slot_mapping: Optional[Tensor] = None,
    index_cache: Optional[Tensor] = None,
    index_q_out: Optional[Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    skip_index_branch: bool = False,
) -> None:
    """Fused MiniMax-M3 QK-norm + partial NeoX RoPE + page-16 SHUFFLE KV insert.

    Consumes the sparse layer's packed projection row
    ``[q | k | v | index_q | index_k]`` (all head_dim=128) and, in one pass:

    * writes Gemma-normed + roped ``q`` into ``q_out`` and ``index_q`` into
      ``index_q_out``,
    * scatters normed + roped ``k`` and verbatim ``v`` into the paged caches in
      the layout ``pa_decode_gluon`` reads::

          k_cache: [num_pages, num_kv_heads, head_dim // x,  page_size, x]
          v_cache: [num_pages, num_kv_heads, page_size // x, head_dim,  x]

      where ``x = 16 // k_cache.element_size()`` and ``page_size`` is
      ``k_cache.shape[3]`` (16 for the gluon decode path),
    * scatters normed + roped ``index_k`` into ``index_cache``
      ``[num_pages, page_size, 128]``.

    ``k``/``v``/``index_k`` are *not* written back into ``qkv``; they are
    consumed only by the cache scatter. ``skip_index_branch=True`` skips all
    index work (the row still carries the index sub-blocks).

    ``k_scale``/``v_scale`` are optional single-element fp32 **device** tensors
    applied as ``value / scale`` when ``kv_cache_dtype`` is quantized, matching
    ``reshape_and_cache``.
    """
    ...
