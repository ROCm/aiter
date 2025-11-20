# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

import torch
import triton
from aiter.ops.triton._triton_kernels.pa_prefill import _fwd_kernel, _fwd_kernel_alibi
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

BASE_BLOCK = 64
NUM_WARPS = 4


@torch.inference_mode()
def context_attention_fwd(
    q,
    k,
    v,
    o,
    kv_cache_dtype: str,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_input_len,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    skip_decode=False,
):
    """
    Paged attention prefill for multi-token context processing with paged KV cache.
    Supports variable-length sequences, GQA, FP8 quantization, ALiBi, and sliding window.

    Args:
        q (torch.Tensor): Query tensor with shape (total_tokens, num_q_heads, head_dim).
        k (torch.Tensor): Key tensor for prefill tokens with shape (total_tokens, num_kv_heads, head_dim).
        v (torch.Tensor): Value tensor for prefill tokens with shape (total_tokens, num_kv_heads, head_dim).
        o (torch.Tensor): Pre-allocated output tensor with shape (total_tokens, num_q_heads, head_dim).
        kv_cache_dtype (str): KV cache data type ("auto", "fp8", "fp8_e4m3").
        k_cache (torch.Tensor): Paged key cache with shape
            (num_blocks, num_kv_heads, head_dim//x, block_size, x) for vectorized layout.
        v_cache (torch.Tensor): Paged value cache with shape
            (num_blocks, num_kv_heads, head_dim, block_size).
        b_loc (torch.Tensor): Block location table mapping tokens to cache blocks with shape
            (batch_size, max_blocks_per_seq).
        b_start_loc (torch.Tensor): Start token index for each sequence with shape (batch_size + 1,).
        b_seq_len (torch.Tensor): Sequence length for each sequence with shape (batch_size,).
        max_input_len (int): Maximum input length across all sequences in batch.
        k_scale (torch.Tensor): Quantization scale for key cache.
        v_scale (torch.Tensor): Quantization scale for value cache.
        alibi_slopes (Optional[torch.Tensor]): ALiBi position bias slopes with shape (num_q_heads,).
        sliding_window (Optional[int]): Sliding window size for local attention. 0 or None disables.
        sm_scale (Optional[float]): Softmax scale, defaults to 1/sqrt(head_dim).
        skip_decode (bool): Skip decode-only sequences (single-token) in mixed batch.

    Returns:
        None. Results written in-place to o.
    """

    _LOGGER.info(
        f"PA_PREFILL: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
    )
    q_dtype_is_f32 = q.dtype is torch.float32
    # need to reduce num. blocks when using fp32
    # due to increased use of GPU shared memory
    # if q.dtype is torch.float32:
    BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK

    IN_PRECISION = None

    if (
        torch.finfo(k_cache.dtype).bits == 8 or torch.finfo(v_cache.dtype).bits == 8
    ) and kv_cache_dtype == "auto":
        raise ValueError(
            "kv_cache_dtype='auto' unsupported for\
            FP8 KV Cache prefill kernel"
        )

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    # round up Lk to a power of 2 - this is required for Triton block size
    Lk_padded = triton.next_power_of_2(Lk)

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]

    assert batch + 1 == len(b_start_loc)
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

    # 0 means "disable"
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if alibi_slopes is not None:
        _fwd_kernel_alibi[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            b_loc,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            alibi_slopes,
            v_cache.shape[3],
            k_cache.shape[4],
            o,
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(4),  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),  # [num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            IN_PRECISION=IN_PRECISION,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            SKIP_DECODE=skip_decode,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return

    _fwd_kernel[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        b_loc,
        sm_scale,
        k_scale,
        v_scale,
        b_start_loc,
        b_seq_len,
        v_cache.shape[3],
        k_cache.shape[4],
        o,
        b_loc.stride(0),
        b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_cache.stride(4),  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),  # [num_blocks, num_kv_heads, head_size, block_size]
        num_queries_per_kv=num_queries_per_kv,
        IN_PRECISION=IN_PRECISION,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        BLOCK_N=BLOCK,
        SLIDING_WINDOW=sliding_window,
        SKIP_DECODE=skip_decode,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return
