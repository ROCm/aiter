# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA decode launcher.

Routes to the appropriate kernel implementation based on num_qo_heads
and dtype, then prepares tensors and dispatches via ``@flyc.jit``.
"""

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_lds_size_per_cu


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def flydsl_mla_fwd_decode(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    kv_buffer: torch.Tensor,  # [num_page, page_size, num_kv_heads, head_size]
    qo_indptr: torch.Tensor,  # [batch_size+1]
    kv_indptr: torch.Tensor,  # [batch_size+1]
    kv_page_indices: torch.Tensor,  # [num_page_used]
    kv_last_page_lens: torch.Tensor,  # [batch_size]
    work_indptr: torch.Tensor,  # metadata
    work_info_set: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    split_output: torch.Tensor,  # [num_partial_slots, 1, num_heads, v_head_dim]
    split_lse: torch.Tensor,  # [num_partial_slots, 1, num_heads, 1]
    final_output: torch.Tensor,  # [num_seqs, num_heads, v_head_dim]
) -> None:
    """Launch the FlyDSL MLA decode forward kernel.

    Signature matches ``hk_mla_fwd_decode_fwd`` so it can be a drop-in
    replacement in ``aiter/mla.py``.
    """
    num_heads = query.size(1)
    q_dtype = query.dtype
    kv_dtype = kv_buffer.dtype

    # ----- dispatch to the right kernel based on (num_heads, dtype) -----
    if num_heads == 128 and _is_fp8(q_dtype) and _is_fp8(kv_dtype):
        from .kernels.mla_fwd_decode_m16x8_fp8_fp8 import (
            OCCUPANCY,
            QK_HEAD_DIM,
            V_HEAD_DIM,
            launch_mla_fwd_decode_m16x8_fp8_fp8,
        )

        num_seqs = query.size(0)
        num_pages = kv_buffer.size(0)
        num_partial = split_output.size(0)

        query_flat = query.reshape(num_seqs * num_heads, QK_HEAD_DIM)
        kv_flat = kv_buffer.reshape(num_pages, QK_HEAD_DIM)
        final_flat = final_output.reshape(num_seqs * num_heads, V_HEAD_DIM)
        split_o_flat = split_output.reshape(num_partial * num_heads, V_HEAD_DIM)
        split_lse_flat = split_lse.reshape(num_partial * num_heads)

        work_indptr_flat = work_indptr.contiguous()
        work_info_flat = work_info_set.contiguous().view(-1)
        kv_idx_flat = kv_page_indices.contiguous()

        num_cus = get_cu_num()
        lds_size = get_lds_size_per_cu() // OCCUPANCY

        launch_mla_fwd_decode_m16x8_fp8_fp8(
            query_flat,
            kv_flat,
            kv_idx_flat,
            work_indptr_flat,
            work_info_flat,
            final_flat,
            split_o_flat,
            split_lse_flat,
            softmax_scale,
            num_cus,
            lds_size,
            stream=torch.cuda.current_stream(),
        )
    else:
        raise NotImplementedError(
            f"flydsl_mla_fwd_decode: unsupported num_heads={num_heads}, "
            f"q_dtype={q_dtype}, kv_dtype={kv_dtype}"
        )
