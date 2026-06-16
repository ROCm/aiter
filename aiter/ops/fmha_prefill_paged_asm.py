# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import math
import torch
from csrc.cpp_itfs.torch_utils import torch_to_c_types

_impl = None


def _get_impl():
    global _impl
    if _impl is None:
        from csrc.cpp_itfs.fmha_prefill_paged_asm import compile
        _impl = compile()
    return _impl


@functools.lru_cache()
def _sched_groups(num_cu: int, nhead_q: int, max_seqlen_q: int) -> int:
    num_tiles = (max_seqlen_q + 255) // 256   # kTileQ = 256
    return max(1, min(num_cu // max(nhead_q, 1), num_tiles))


def fmha_prefill_paged_asm_launch(
    q,                      # [total_q, nhead_q, 128] fp8
    k,                      # paged K pool, vec_k_col_v layout fp8
    v,                      # paged V pool, vec_k_col_v layout fp8
    cu_seqlens_q,           # [batch+1] int32
    kv_indptr,              # [batch+1] int32  (LTP)
    kv_page_indices,        # [num_pages_used] int32  (LTD)
    kv_last_page_lens,      # [batch] int32  (unused by kernel, kept for API compat)
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,        # always True for this kernel
    q_descale_per_token,    # [total_q, nhead_q] fp32
    k_descale_per_token,    # [num_pages, page_size, nhead_k] fp32
    v_descale_per_head,     # [nhead_k] fp32
    p_scale,                # [nhead_q] fp32, optional
    p_scale_inv,            # unused, kept for API compat
    out=None,
):
    batch   = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    nhead_q = q.shape[1]
    nhead_k = v_descale_per_head.shape[0]

    # K pool: [num_pages, nhead_k, hd/kVecX, page_size, kVecX]  kVecX=16
    num_pages = k.shape[0]
    page_size = 16   # kPageSize fixed in constants.py

    # Byte strides in K/V pool (fp8 = 1 byte)
    # K: [pages, nhead_k, 8, page_size, 16]  → head stride = 8*page_size*16 bytes
    k_head_stride_bytes = (128 // 16) * page_size * 16 * 1   # = 128 * page_size
    # K: page stride = nhead_k * k_head_stride_bytes
    k_page_stride_bytes = nhead_k * k_head_stride_bytes
    # V pool: [pages, nhead_k, hd, page_size] col-major (KV cache manager native layout).
    # Kernel consumes col-major V directly — no permute needed.
    v_head_stride_bytes = 128 * page_size * 1                 # bytes per kv-head (hd * page_size * bpe)
    v_page_stride_bytes = nhead_k * 128 * page_size * 1       # bytes per page (col-major)

    # Output: [total_q, nhead_q, 128] bf16 = 2 bytes per element
    if out is None:
        out = torch.empty(total_q, nhead_q, 128, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty(nhead_q, total_q, dtype=torch.float32, device=q.device)

    # Output head byte stride: stride between heads = 128 * sizeof(bf16) = 256 bytes
    o_head_stride_bytes = out.stride(1) * out.element_size()

    # The kernel uses fixed stride s_Bs = max_seqlen_q * q_token_stride for Q/O addressing,
    # so it only works correctly when every batch entry has exactly max_seqlen_q tokens.
    seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    if any(s != max_seqlen_q for s in seqlens_q):
        return None, None, None, None  # signal caller to fall back to CK

    props = torch.cuda.get_device_properties(q.device)
    num_cu = props.multi_processor_count
    sched_groups = _sched_groups(num_cu, nhead_q, max_seqlen_q)

    # Kernel expects q_descale in [batch, nhead_q, max_seqlen_q] layout.
    # All seqlens equal max_seqlen_q here, so a simple reshape+permute suffices.
    q_descale_hm = (q_descale_per_token
                    .reshape(batch, max_seqlen_q, nhead_q)
                    .permute(0, 2, 1)
                    .contiguous())  # [batch, nhead_q, max_seqlen_q]

    _get_impl()(
        *torch_to_c_types(
            out,
            q,
            k,
            v,
            lse,
            softmax_scale,
            max_seqlen_q,
            batch,
            max_seqlen_k,
            nhead_q,
            nhead_k,
            num_pages,
            page_size,
            kv_page_indices,
            kv_indptr,
            q_descale_hm,
            k_descale_per_token,
            v_descale_per_head,
            p_scale,
            k_page_stride_bytes,
            v_page_stride_bytes,
            k_head_stride_bytes,
            v_head_stride_bytes,
            o_head_stride_bytes,
            total_q,
            sched_groups,
            torch.cuda.current_stream(q.device),
        )
    )
    return out, lse, None, None
