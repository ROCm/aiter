# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Standalone Python entry for the qkptph/vph (PER_TOKEN_HEAD) FP8 causal paged
# batch-prefill ASM kernel. This routes ONLY to the asm kernel (module
# module_mha_batch_prefill_asm, built ENABLE_CK=0) and never falls back to CK.
#
# Tensor layouts (5D vec_k_col_v K, col-major V, per-token-head descales, SGLang
# 1D page table) mirror the AITERKER-112 mha_batch_prefill reference interface.

from typing import Optional

from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_mha_batch_prefill_asm", fc_name="mha_batch_prefill_asm")
def mha_batch_prefill_asm(
    q: Tensor,                    # [total_q, hq, d] fp8
    k: Tensor,                    # [num_pages, hk, d/x, page, x] fp8
    v: Tensor,                    # [num_pages, hk, d, page] fp8 (col-major)
    cu_seqlens_q: Tensor,         # [b+1] int32 (QTP)
    kv_indptr: Tensor,            # [b+1] int32 (LTP)
    kv_page_indices: Tensor,      # [num_pages] int32 (LTD)
    seqlens_kvcache: Tensor,      # [b] int32 per-batch KV token len
    out: Tensor,                  # [total_q, hq, dv] bf16 (written in place)
    q_descale_per_token: Tensor,  # [total_q, hq] f32
    k_descale_per_token: Tensor,  # [num_pages, page, hk] f32
    v_descale_per_head: Tensor,   # [hk] f32
    batch: int,
    num_heads: int,
    num_heads_k: int,
    head_size_q: int,
    head_size_v: int,
    page_block_size: int,
    num_total_pages: int,
    max_seqlen_q: int,
    softmax_scale: float,
    p_scale: Optional[Tensor] = None,  # [hq] f32
) -> Tensor: ...
