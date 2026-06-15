# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.gather_kv_b_proj import (
    _next_pow2,
    _triton_gather_kv_b_proj,
)


def gather_kv_b_proj(
    k_buffer: torch.Tensor,  # [num_block, block_size, hidden_dim]
    k_scale: torch.Tensor,  # [1]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # len(kv_indices) = kv_indptr[-1]
    kv_prefix_sum_context_lens: torch.Tensor,  # [batch_size + 1]
    kv_proj_weight: torch.Tensor,  # [tp_heads * (qk_nope_head_dim + v_head_dim), kv_c_dim]
    kv_proj_scale: torch.Tensor,  # [weight_n] per-output-row, or [N//128, K//128] block
    k_prefix: torch.Tensor,  # [total_kv, tp_k_head_num, qk_nope_head_dim + kv_pe_dim]
    v_prefix: torch.Tensor,  # [total_kv, tp_k_head_num, v_head_dim]
    weight_preshuffle: bool = False,
    shuffled_kv_cache: bool = False,
):
    num_block, block_size, hidden_dim = k_buffer.shape
    batch_size = kv_indptr.shape[0] - 1
    weight_n, packed_weight_k = kv_proj_weight.shape
    fp4_weight_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    is_fp4_weight = (
        fp4_weight_dtype is not None and kv_proj_weight.dtype == fp4_weight_dtype
    )
    weight_k = packed_weight_k * 2 if is_fp4_weight else packed_weight_k
    total_kv_k, tp_k_head_num_k, qk_nope_pe_dim = k_prefix.shape
    total_kv_v, tp_k_head_num_v, v_head_dim = v_prefix.shape

    qk_nope_head_dim = weight_n // tp_k_head_num_k - v_head_dim

    # Three scale modes:
    #   - kv_proj_scale is None   : weight is unquantized (e.g. bf16
    #     kv_b_proj on Kimi-K2.5-MXFP4). Kernel skips scale-load and
    #     scale-multiply entirely (NO_SCALE branch).
    #   - kv_proj_scale.dim() == 1 (or [N, 1]) : per-row scale.
    #   - else                     : per-block scale ([N//128, K//128]).
    no_scale = kv_proj_scale is None
    if no_scale:
        # Triton requires a non-None tensor pointer for every kernel argument
        # even if NO_SCALE=True makes it unread. Pass the weight tensor as a
        # placeholder; the kernel does not load from it in this branch.
        kv_proj_scale = kv_proj_weight
        per_row_scale = False  # ignored when NO_SCALE=True
    else:
        per_row_scale = kv_proj_scale.dim() == 1 or (
            kv_proj_scale.dim() == 2 and kv_proj_scale.shape[1] == 1
        )
        if per_row_scale:
            assert kv_proj_scale.numel() == weight_n, (
                f"per-row kv_proj_scale must have shape ({weight_n},) or ({weight_n}, 1), "
                f"got {tuple(kv_proj_scale.shape)}"
            )
        else:
            scale_n, scale_k = kv_proj_scale.shape
            scale_k_granularity = weight_k // scale_k
            scale_n_granularity = weight_n // scale_n
            if is_fp4_weight:
                if weight_preshuffle:
                    assert scale_k >= (weight_k + 31) // 32, (
                        "Preshuffled FP4 gather_kv_b_proj expects padded per-1x32 scale columns, "
                        f"got scale cols {scale_k} for logical K {weight_k}"
                    )
                    assert scale_n >= weight_n, (
                        "Preshuffled FP4 gather_kv_b_proj expects padded per-output-row MXFP4 scales, "
                        f"got scale rows {scale_n} for weight rows {weight_n}"
                    )
                else:
                    assert scale_k_granularity == 32, (
                        "FP4 gather_kv_b_proj expects per-1x32 weight scales, "
                        f"got K granularity {scale_k_granularity}"
                    )
                    assert scale_n_granularity == 1, (
                        "FP4 gather_kv_b_proj expects per-output-row MXFP4 scales, "
                        f"got N granularity {scale_n_granularity}"
                    )
            else:
                assert scale_k_granularity == 128
                assert scale_n_granularity == 128

    if shuffled_kv_cache:
        # FP4 *kv_buffer* is not supported; the kv buffer must be bf16/fp8. The
        # weight may still be MXFP4 (handled by the FP4 weight path below).
        assert k_buffer.dtype in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ), f"shuffled_kv_cache gather expects a bf16/fp8 kv buffer, got {k_buffer.dtype}"
        assert block_size % 16 == 0, (
            f"shuffled_kv_cache gather requires block_size % 16 == 0 (16-token "
            f"shuffle groups), got block_size={block_size}"
        )
        # The shuffle keeps each token's data within its own block, so a chunk
        # must span exactly one block (KBlocksPerChunkK == 1).
        ChunkK = block_size
    elif is_fp4_weight:
        ChunkK = 64
    else:
        ChunkK = 16 if k_buffer.dtype in [torch.float16, torch.bfloat16] else 32

    assert total_kv_k == total_kv_v
    assert tp_k_head_num_k == tp_k_head_num_v
    assert ChunkK % block_size == 0

    padded_k = _next_pow2(qk_nope_head_dim)
    padded_v = _next_pow2(v_head_dim)

    grid = (batch_size * tp_k_head_num_k,)
    if is_fp4_weight:
        # Use the actual output token count, not kv_indices capacity. Serving
        # paths may pass a preallocated kv_indices buffer that is much larger
        # than the valid range described by kv_indptr/k_prefix.
        max_kv_chunks = max(1, (total_kv_k + ChunkK - 1) // ChunkK)
        fp4_scale_k_granularity = 32 if weight_preshuffle else 128
        fp4_grid = (batch_size * tp_k_head_num_k * max_kv_chunks,)
        _triton_gather_kv_b_proj[fp4_grid](
            batch_size,
            k_buffer,
            k_scale,
            kv_indptr,
            kv_indices,
            kv_prefix_sum_context_lens,
            kv_proj_weight.view(torch.uint8),
            kv_proj_scale.view(torch.uint8),
            k_prefix,
            v_prefix,
            KBlockSize=block_size,
            TpNumHeads=tp_k_head_num_k,
            QkNopeHeadDim=qk_nope_head_dim,
            VHeadDim=v_head_dim,
            KV_CDim=weight_k,
            KV_PeDim=qk_nope_pe_dim - qk_nope_head_dim,
            ChunkK=ChunkK,
            PaddedK=padded_k,
            PaddedV=padded_v,
            ScaleCols=scale_k if not no_scale and not per_row_scale else 1,
            IS_FP4=True,
            Fp4ScaleKGranularity=fp4_scale_k_granularity,
            WEIGHT_PRESHUFFLE=weight_preshuffle,
            SHUFFLED_KV_CACHE=shuffled_kv_cache,
            num_stages=3,
        )
        return

    _triton_gather_kv_b_proj[grid](
        batch_size,
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix,
        v_prefix,
        KBlockSize=block_size,
        TpNumHeads=tp_k_head_num_k,
        QkNopeHeadDim=qk_nope_head_dim,
        VHeadDim=v_head_dim,
        KV_CDim=weight_k,
        KV_PeDim=qk_nope_pe_dim - qk_nope_head_dim,
        ChunkK=ChunkK,
        PaddedK=padded_k,
        PaddedV=padded_v,
        IS_FP4=False,
        WEIGHT_PRESHUFFLE=weight_preshuffle,
        PER_ROW_SCALE=per_row_scale,
        NO_SCALE=no_scale,
        SHUFFLED_KV_CACHE=shuffled_kv_cache,
        num_stages=3,
    )
