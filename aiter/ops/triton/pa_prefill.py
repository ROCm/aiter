# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

import torch
import triton
import triton.language as tl
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
    #TODO: Add Doc
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
