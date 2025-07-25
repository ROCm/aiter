# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import functools
import json
import torch
import triton
import triton.language as tl

import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.pid_preprocessing import remap_xcd
from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward
from aiter.ops.triton.mha_fused_bwd import flash_attn_fused_backward
from aiter.ops.triton.utils.mha_kernel_utils import (
    _compute_fp8_scaling_factors,
    _is_fp8,
)

global _USE_FUSED_BWD_KERNEL
_USE_FUSED_BWD_KERNEL = False


def mha_set_use_fused_bwd_kernel(value: bool):
    global _USE_FUSED_BWD_KERNEL
    _USE_FUSED_BWD_KERNEL = value


_USE_INT64_STRIDES = True


def mha_set_use_int64_strides(value: bool):
    """Use 64-bit integer strides to prevent integer overflows with very large tensors."""
    global _USE_INT64_STRIDES
    _USE_INT64_STRIDES = value


def _cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype,
    layout,
    clamp_val=1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor to FP8 format, returning an FP8 tensor and a descale factor.
    Args:
        - x (torch.Tensor): shape [batch, seq_len, heads, dim]
    Returns:
        - x_fp8 (torch.Tensor): FP8 tensor with the same shape as x
        - descale_factor (torch.Tensor): tensor of shape [batch, 1, heads, 1]
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}"
        )
    reduce_dims = (1, 3)  # seq_len and dim dimensions

    # Compute the absolute max along reduce_dims, clamped to avoid 0-scale
    x_abs_max = x.abs().amax(dim=reduce_dims)
    x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

    # Unsqueeze back to a shape suitable for broadcast
    unsqueeze_dims = sorted(reduce_dims)
    for d in unsqueeze_dims:
        x_abs_max = x_abs_max.unsqueeze(d)

    # compute scale and descale
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / x_abs_max
    descale_factor = x_abs_max / fp8_max

    # cast to FP8, optionally setting requires_grad
    x_fp8 = (x * scale).to(fp8_dtype)

    return x_fp8, descale_factor


def _cast_varlen_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    cu_seqlens,
    clamp_val: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor of sequences with variable seq_len into fp8.
    Args:
        - x (torch.Tensor): shape [total_seq_len, heads, dim]
    Returns:
        - x_fp8 (torch.Tensor): shape [total_seq_len, heads, dim]
        - descale_factors (torch.Tensor): shape [batch, heads]
    """
    # validate tensor shape
    if len(x.shape) != 3:
        raise ValueError(
            f"tensor should have shape [total_seqlen, heads, dim], got {x.shape}"
        )
    num_heads = x.shape[1]

    # Get batch size from cu_seqlens
    batch = cu_seqlens.shape[0] - 1
    fp8_max = torch.finfo(fp8_dtype).max

    # Compute scale and descale factors per sequence
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros(
        (batch, num_heads), device=x.device, dtype=torch.float32
    )

    for i in range(batch):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        x_slice = x[start:end]  # Slice for current sequence

        # Standard tensor (0: seq_len, 2: head_dim)
        x_abs_max = x_slice.abs().amax(dim=(0, 2))  # [heads]

        # apply minimum clamping
        x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

        # compute scale and descale factors
        scale_i = fp8_max / x_abs_max
        descale_i = x_abs_max / fp8_max

        # store descale factors
        descale_factors[i, :] = descale_i

        scale_reshape = scale_i.reshape(1, num_heads, 1)

        # scale and cast to FP8
        x_fp8[start:end] = (x_slice * scale_reshape).to(fp8_dtype)

    return x_fp8, descale_factors


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def _compute_alibi_block(
    alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False
):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def compute_window_bounds(
    q_start,
    q_end,
    diag,
    seqlen_k,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Calculate the window boundaries for a query block."""
    # Left boundary
    if WINDOW_SIZE_LEFT < 0:
        left_min = 0
        left_max = 0
    else:
        left_min = tl.maximum(0, q_start + diag - WINDOW_SIZE_LEFT)
        left_max = tl.maximum(0, q_end + diag - WINDOW_SIZE_LEFT)

    # Right boundary
    if IS_CAUSAL:
        # Causal cap: col ≤ row + diag
        right_min = tl.minimum(seqlen_k - 1, q_start + diag)
        right_max = tl.minimum(seqlen_k - 1, q_end + diag)
    else:
        if WINDOW_SIZE_RIGHT < 0:
            right_min = tl.minimum(seqlen_k - 1, q_start + diag + WINDOW_SIZE_RIGHT)
            right_max = tl.minimum(seqlen_k - 1, q_end + diag + WINDOW_SIZE_RIGHT)
        else:
            # Non-causal doesn't have the diagonal constraint
            right_min = tl.minimum(seqlen_k - 1, q_start + diag + WINDOW_SIZE_RIGHT)
            right_max = tl.minimum(seqlen_k - 1, q_end + diag + WINDOW_SIZE_RIGHT)

    return left_min, left_max, right_min, right_max


@triton.jit
def classify_window_blocks(
    left_min, left_max, right_min, right_max, BLOCK_N: tl.constexpr
):
    """Classify blocks based on window boundaries."""
    # First and last blocks that have ANY overlap with window
    first_block = left_min // BLOCK_N
    last_block = right_max // BLOCK_N

    # First block that is FULLY visible for all rows in Q block
    full_left_block = left_max // BLOCK_N + (left_max % BLOCK_N != 0)
    clipped_left = tl.minimum(full_left_block, last_block + 1)

    # Last block that is FULLY visible for all rows in Q block
    last_full_block_candidate = right_min // BLOCK_N
    if (last_full_block_candidate + 1) * BLOCK_N - 1 > right_min:
        last_full_block_candidate -= 1
    full_right_block = tl.maximum(last_full_block_candidate, clipped_left - 1)

    # Calculate counts
    n_front_skip_blocks = first_block
    n_front_masked_blocks = tl.maximum(0, clipped_left - first_block)
    n_full_blocks = tl.maximum(0, full_right_block - clipped_left + 1)
    n_back_masked_blocks = tl.maximum(0, last_block - full_right_block)

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        clipped_left,
    )


@triton.jit
def handle_padded_last_block(
    n_extra_tokens,
    last_block,
    total_k_blocks,
    clipped_left,
    n_front_masked_blocks,
    n_full_blocks,
    n_back_masked_blocks,
):
    """Adjust block counts when last K block has padding."""
    padded_last_k = (n_extra_tokens != 0) & (last_block == total_k_blocks - 1)

    if padded_last_k & (n_back_masked_blocks == 0):
        last_block_in_front = clipped_left > last_block
        if last_block_in_front:
            n_front_masked_blocks = tl.maximum(0, n_front_masked_blocks - 1)
        else:
            n_full_blocks = tl.maximum(0, n_full_blocks - 1)
        n_back_masked_blocks = 1

    return n_front_masked_blocks, n_full_blocks, n_back_masked_blocks


@triton.jit
def compute_padding_info(seqlen_k, BLOCK_N: tl.constexpr):
    """Calculate padding information for the last K block."""
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    else:
        n_extra_tokens = 0
    return n_extra_tokens


@triton.jit
def compute_block_masking(
    seqlen_k,
    seqlen_q,
    start_m,
    IS_CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Classify K blocks for attention computation with sliding window support.
    """
    # common
    q_start = start_m * BLOCK_M
    q_end = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)
    diag = seqlen_k - seqlen_q
    total_k_blocks = _cdiv_fn(seqlen_k, BLOCK_N)
    n_extra_tokens = compute_padding_info(seqlen_k, BLOCK_N)

    if USE_SLIDING_WINDOW:
        # get window bounds
        left_min, left_max, right_min, right_max = compute_window_bounds(
            q_start,
            q_end,
            diag,
            seqlen_k,
            WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT,
            IS_CAUSAL,
        )

        # window vanishes → early exit
        if right_max < left_min:
            return 0, 0, 0, 0, n_extra_tokens

        # classify blocks
        (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            clipped_left,
        ) = classify_window_blocks(left_min, left_max, right_min, right_max, BLOCK_N)

        # handle padded last block if needed
        if n_extra_tokens != 0:
            last_block = right_max // BLOCK_N
            n_front_masked_blocks, n_full_blocks, n_back_masked_blocks = (
                handle_padded_last_block(
                    n_extra_tokens,
                    last_block,
                    total_k_blocks,
                    clipped_left,
                    n_front_masked_blocks,
                    n_full_blocks,
                    n_back_masked_blocks,
                )
            )
        return (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            n_extra_tokens,
        )
    else:
        # Original causal/non-causal logic
        if IS_CAUSAL:
            n_blocks_seqlen = _cdiv_fn(
                (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
            )
            n_blocks = min(total_k_blocks, n_blocks_seqlen)

            if n_blocks <= 0:
                return 0, 0, 0, 0, n_extra_tokens

            padded_last_k = n_extra_tokens != 0
            is_modulo_mn = (not padded_last_k) & (seqlen_q % BLOCK_M == 0)

            n_back_masked_blocks = BLOCK_M // BLOCK_N + tl.where(is_modulo_mn, 0, 1)
            n_back_masked_blocks = tl.minimum(n_back_masked_blocks, n_blocks)

            n_front_skip_blocks = 0
            n_front_masked_blocks = 0
            n_full_blocks = n_blocks - n_back_masked_blocks
        else:
            # Non-causal mode
            n_front_skip_blocks = 0
            n_front_masked_blocks = 0
            if n_extra_tokens != 0:
                n_back_masked_blocks = 1
                n_full_blocks = total_k_blocks - 1
            else:
                n_back_masked_blocks = 0
                n_full_blocks = total_k_blocks

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    )


@triton.jit
def _attn_fwd_inner_no_mask(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    stride_sn,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    sd_mask_ptrs,
    dropout_mask_ptrs,
    philox_seed,
    philox_ptrs,
    block_min,
    block_max,
    alibi_slope,
    descale_q,
    descale_k,
    descale_v,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Fast path for blocks that don't need masking"""
    RCP_LN2: tl.constexpr = 1.4426950408889634

    for start_n in range(block_min, block_max, BLOCK_N):
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = _load_fn(k_ptrs, k_offs_k, None, BLOCK_DMODEL, seqlen_k)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        if IS_FP8:
            qk += tl.dot(q, k) * descale_q * descale_k
        else:
            qk += tl.dot(q, k)

        if alibi_slope is not None:
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = _compute_alibi_block(
                alibi_slope, seqlen_q, seqlen_k, global_m_positions, global_n_positions
            )
            qk += alibi_block / SM_SCALE

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]
        p = tl.math.exp2(q_shifted)

        l_ij = tl.sum(p, 1)

        if ENABLE_DROPOUT:
            q_mask = OFFS_M[:, None] < seqlen_q
            k_mask = (start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k
            p_mask = q_mask & k_mask

            rng_output = tl.rand(philox_seed, philox_ptrs)
            dropout_mask = rng_output > dropout_p
            tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            q_mask = OFFS_M[:, None] < seqlen_q
            k_mask = (start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k
            p_mask = q_mask & k_mask
            tl.store(sd_mask_ptrs, p, mask=p_mask)

        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = tl.math.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]

        v_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        v = _load_fn(v_ptrs, None, v_offs_k, seqlen_k, BLOCK_DMODEL)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        if IS_FP8:
            scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
            acc += (
                tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v
            )
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_inner_mask(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    stride_sn,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    sd_mask_ptrs,
    dropout_mask_ptrs,
    philox_seed,
    philox_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    n_extra_tokens,
    alibi_slope,
    descale_q,
    descale_k,
    descale_v,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
):
    """Path for blocks that need masking (causal, padding, or sliding window)"""
    RCP_LN2: tl.constexpr = 1.4426950408889634

    for start_n in range(block_min, block_max, BLOCK_N):
        k_offs_n = start_n + tl.arange(0, BLOCK_N)
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = _load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)

        # Padding mask for last block
        bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
        boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
        size_n = start_n + OFFS_N[None, :]
        mask_partial = size_n < boundary_m[:, None]
        mask = tl.where(bound_cond, mask_partial, mask)

        if IS_FP8:
            qk += tl.dot(q, k) * descale_q * descale_k
        else:
            qk += tl.dot(q, k)

        # Apply masking based on mode
        if USE_SLIDING_WINDOW:
            if IS_CAUSAL:
                # Causal sliding window
                row_idx = OFFS_M
                col_idx = k_offs_n
                row_idx_expanded = row_idx[:, None]
                col_idx_expanded = col_idx[None, :]

                causal_offset = seqlen_k - seqlen_q
                causal_mask = col_idx_expanded > (row_idx_expanded + causal_offset)

                if WINDOW_SIZE_LEFT < 0:
                    window_mask = col_idx_expanded > (
                        row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT
                    )
                else:
                    left_bound = row_idx_expanded + causal_offset - WINDOW_SIZE_LEFT
                    right_bound = row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT
                    window_mask = (col_idx_expanded < left_bound) | (
                        col_idx_expanded > right_bound
                    )

                mask = mask & ~(causal_mask | window_mask)
            else:
                # Non-causal sliding window
                row_idx = OFFS_M
                col_idx = k_offs_n
                row_idx_expanded = row_idx[:, None]
                col_idx_expanded = col_idx[None, :]

                sk = seqlen_k
                sq = seqlen_q

                if WINDOW_SIZE_LEFT < 0:
                    window_mask = col_idx_expanded > (
                        row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT
                    )
                else:
                    sk_full = tl.full((1, BLOCK_N), sk, dtype=tl.int32)
                    right_bound_val = row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT
                    right_bound = tl.minimum(right_bound_val, sk_full)
                    left_bound = row_idx_expanded + sk - sq - WINDOW_SIZE_LEFT
                    window_mask = (col_idx_expanded > right_bound) | (
                        col_idx_expanded < left_bound
                    )

                mask = mask & ~window_mask
        else:
            if IS_CAUSAL:
                causal_boundary = start_n + offs_n_causal
                causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
                mask = mask and causal_mask

        qk = tl.where(mask, qk, float("-inf"))

        if alibi_slope is not None:
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = _compute_alibi_block(
                alibi_slope, seqlen_q, seqlen_k, global_m_positions, global_n_positions
            )
            qk += alibi_block / SM_SCALE

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2

        # Handle -inf cases for sliding window
        if USE_SLIDING_WINDOW:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"),
                float("-inf"),
                qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None],
            )
        else:
            q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        q_mask = OFFS_M[:, None] < seqlen_q
        k_mask = k_offs_n[None, :] < seqlen_k
        p_mask = q_mask & k_mask

        if ENABLE_DROPOUT:
            rng_output = tl.rand(philox_seed, philox_ptrs)
            dropout_mask = rng_output > dropout_p
            tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            tl.store(sd_mask_ptrs, p, mask=p_mask)

        # Handle -inf cases for sliding window
        if USE_SLIDING_WINDOW:
            m_diff_scaled = tl.where(
                m_ij == float("-inf"),
                float("-inf"),
                m_i * SM_SCALE * RCP_LN2 - m_ij_scaled,
            )
        else:
            m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled

        alpha = tl.math.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]

        v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        if IS_FP8:
            scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
            acc += (
                tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v
            )
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    v_ptr: torch.Tensor,
    descale_q_ptr: torch.Tensor,
    descale_k_ptr: torch.Tensor,
    descale_v_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    alibi_slopes_ptr: torch.Tensor,
    s_dmask_ptr: torch.Tensor,
    dropout_mask_ptr: torch.Tensor,
    softmax_lse_ptr: torch.Tensor,
    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    stride_alibi_z_in,
    stride_alibi_h_in,
    stride_sd_z_in,
    stride_sd_h_in,
    stride_sd_m_in,
    stride_sd_n_in,
    stride_lse_z_in,
    stride_lse_h_in,
    stride_lse_m_in,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    VARLEN: tl.constexpr,
    BATCH,
    NUM_XCD: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
):
    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    # calculate offsets
    wid = tl.program_id(
        0
    )  # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
    # num blocks along seqlen

    off_q_head = wid % NUM_Q_HEADS
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    # offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)

    # NOTE:
    # Workaround for int64 strides, In the absence of strides being int64, parts of the offset
    # computation is done in 32 bit and overflows resulting in segfaults
    # If input strides are defined as int64, it disables vectorized loads which drops perf
    # If we define new strides as stride_x = stride_x_in.to(tl.int64), that does not work
    # because strides are tl.constexpr and cannot be upcasted
    # If we define new strides as stride_x: tl.int64 = stride_x_in, segfault remains
    # The permanent solution is to enable upcasting of tl.constexpr
    # In the meantime, the following workaround provides correctness and does not drop perf
    if USE_INT64_STRIDES:
        stride_qz = tl.cast(stride_qz_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qk = tl.cast(stride_qk_in, tl.int64)
        stride_kz = tl.cast(stride_kz_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kk = tl.cast(stride_kk_in, tl.int64)
        stride_vz = tl.cast(stride_vz_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vk = tl.cast(stride_vk_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
        stride_oz = tl.cast(stride_oz_in, tl.int64)
        stride_oh = tl.cast(stride_oh_in, tl.int64)
        stride_om = tl.cast(stride_om_in, tl.int64)
        stride_on = tl.cast(stride_on_in, tl.int64)
        stride_alibi_z = tl.cast(stride_alibi_z_in, tl.int64)
        stride_alibi_h = tl.cast(stride_alibi_h_in, tl.int64)

        # NOTE: philox offset is need in dropout pointer calculations
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_sd_z = tl.cast(stride_sd_z_in, tl.int64)
        stride_sd_h = tl.cast(stride_sd_h_in, tl.int64)
        stride_sd_m = tl.cast(stride_sd_m_in, tl.int64)
        stride_sd_n = tl.cast(stride_sd_n_in, tl.int64)
        stride_lse_z = tl.cast(stride_lse_z_in, tl.int64)
        stride_lse_h = tl.cast(stride_lse_h_in, tl.int64)
        stride_lse_m = tl.cast(stride_lse_m_in, tl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_qh = stride_qh_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_alibi_z = stride_alibi_z_in
        stride_alibi_h = stride_alibi_h_in
        philox_offset_base = philox_offset_base_in
        stride_sd_z = stride_sd_z_in
        stride_sd_h = stride_sd_h_in
        stride_sd_m = stride_sd_m_in
        stride_sd_n = stride_sd_n_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.

        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = _cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )

        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)

        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            offs_out = (
                off_z * stride_oz
                + off_q_head * stride_oh
                + cu_seqlens_q_start * stride_om
                + offs_m[:, None] * stride_om
                + offs_d[None, :] * stride_on
            )
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=out_ptr.type.element_ty)
            out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
            tl.store(out_ptr + offs_out, acc, mask=out_mask)

            if softmax_lse_ptr is not None:
                offs_lse = (
                    off_z * stride_lse_z
                    + off_q_head * stride_lse_h
                    + cu_seqlens_q_start * stride_lse_m
                    + offs_m * stride_lse_m
                )
                lse_mask = offs_m < SEQLEN_Q
                lse = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
                tl.store(softmax_lse_ptr + offs_lse, lse, mask=lse_mask)
                # TODO: Should dropout and return encoded softmax be handled here too?

            return

    grp_sz: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    if grp_sz != 1:  # Grouped Query Attention
        off_k_head = off_q_head // grp_sz
    else:
        off_k_head = off_q_head

    # q,k,v offsets
    q_offs = (
        off_z * stride_qz
        + off_q_head * stride_qh
        + cu_seqlens_q_start * stride_qm
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q_ptrs = q_ptr + q_offs

    k_offs = (
        off_z * stride_kz
        + off_k_head * stride_kh
        + cu_seqlens_k_start * stride_kn
        + offs_d[:, None] * stride_kk
        + offs_n[None, :] * stride_kn
    )
    k_ptrs = k_ptr + k_offs

    v_offs = (
        off_z * stride_vz
        + off_k_head * stride_vh
        + cu_seqlens_k_start * stride_vn
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vk
    )
    v_ptrs = v_ptr + v_offs

    # alibi slopes
    if alibi_slopes_ptr is not None:
        alibi_offs = off_z * stride_alibi_z + off_q_head * stride_alibi_h
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offs)
    else:
        alibi_slope = None

    # s_dmask (return_scores)
    if s_dmask_ptr is not None:
        s_dmask_offs = (
            off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + offs_m[:, None] * stride_sd_m
            + offs_n[None, :] * stride_sd_n
        )
        s_dmask_ptrs = s_dmask_ptr + s_dmask_offs
    else:
        s_dmask_ptrs = None

    # dropout
    if dropout_mask_ptr is not None:
        dropout_mask_offs = (
            off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + offs_m[:, None] * stride_sd_m
            + offs_n[None, :] * stride_sd_n
        )
        dropout_mask_ptrs = dropout_mask_ptr + dropout_mask_offs
        philox_ptrs = (
            philox_offset_base
            + off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + offs_m[:, None] * stride_sd_m
            + offs_n[None, :] * stride_sd_n
        )
    else:
        dropout_mask_ptrs = None
        philox_ptrs = None

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
    if BLOCK_DMODEL == BLOCK_DMODEL_POW2:
        q_mask = offs_m[:, None] < seqlen_q
    else:
        q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    if IS_FP8:
        descale_q = tl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
        descale_k = tl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
        descale_v = tl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)
    else:
        descale_q, descale_k, descale_v = 1.0, 1.0, 1.0

    # figure out block masking
    (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    ) = compute_block_masking(
        seqlen_k,
        seqlen_q,
        start_m,
        IS_CAUSAL,
        USE_SLIDING_WINDOW,
        WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT,
        BLOCK_M,
        BLOCK_N,
    )

    ## Check if all blocks are skipped
    total_visible_blocks = n_front_masked_blocks + n_full_blocks + n_back_masked_blocks
    if total_visible_blocks == 0:
        # Write zeros to output and LSE
        offs_out = (
            off_z * stride_oz
            + off_q_head * stride_oh
            + cu_seqlens_q_start * stride_om
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_on
        )
        out_zeros = acc.to(out_ptr.dtype.element_ty)
        out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
        tl.store(out_ptr + offs_out, out_zeros, mask=out_mask)

        if softmax_lse_ptr is not None:
            offs_lse = (
                off_z * stride_lse_z
                + off_q_head * stride_lse_h
                + cu_seqlens_q_start * stride_lse_m
                + offs_m * stride_lse_m
            )
            lse_mask = offs_m < seqlen_q
            lse = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
            tl.store(softmax_lse_ptr + offs_lse, lse, mask=lse_mask)
        return

    # Process front masked blocks (sliding window only)
    if n_front_masked_blocks > 0 and USE_SLIDING_WINDOW:
        block_min = n_front_skip_blocks * BLOCK_N
        block_max = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        k_ptrs_front = k_ptrs + block_min * stride_kn
        v_ptrs_front = v_ptrs + block_min * stride_vn
        s_dmask_ptrs_front = (
            s_dmask_ptrs + block_min * stride_sd_n if RETURN_SCORES else None
        )
        dropout_mask_ptrs_front = (
            dropout_mask_ptrs + block_min * stride_sd_n if ENABLE_DROPOUT else None
        )

        acc, l_i, m_i = _attn_fwd_inner_mask(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs_front,
            v_ptrs_front,
            stride_kn,
            stride_vn,
            stride_sd_n,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            s_dmask_ptrs_front,
            dropout_mask_ptrs_front,
            philox_seed,
            philox_ptrs,
            block_min,
            block_max,
            offs_n + (seqlen_q - seqlen_k) if IS_CAUSAL else 0,
            0,  # n_extra_tokens (not relevant for front blocks)
            alibi_slope,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            sm_scale,
            IS_CAUSAL,
            ENABLE_DROPOUT,
            RETURN_SCORES,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
        )

    # Process full blocks (no masking needed)
    if n_full_blocks > 0:
        block_min = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        block_max = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N
        k_ptrs_full = k_ptrs + block_min * stride_kn
        v_ptrs_full = v_ptrs + block_min * stride_vn
        s_dmask_ptrs_full = (
            s_dmask_ptrs + block_min * stride_sd_n if RETURN_SCORES else None
        )
        dropout_mask_ptrs_full = (
            dropout_mask_ptrs + block_min * stride_sd_n if ENABLE_DROPOUT else None
        )

        acc, l_i, m_i = _attn_fwd_inner_no_mask(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs_full,
            v_ptrs_full,
            stride_kn,
            stride_vn,
            stride_sd_n,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            s_dmask_ptrs_full,
            dropout_mask_ptrs_full,
            philox_seed,
            philox_ptrs,
            block_min,
            block_max,
            alibi_slope,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            sm_scale,
            ENABLE_DROPOUT,
            RETURN_SCORES,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
        )

    # Process back masked blocks
    if n_back_masked_blocks > 0:
        block_min = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N
        block_max = (
            n_front_skip_blocks
            + n_front_masked_blocks
            + n_full_blocks
            + n_back_masked_blocks
        ) * BLOCK_N
        k_ptrs_back = k_ptrs + block_min * stride_kn
        v_ptrs_back = v_ptrs + block_min * stride_vn
        s_dmask_ptrs_back = (
            s_dmask_ptrs + block_min * stride_sd_n if RETURN_SCORES else None
        )
        dropout_mask_ptrs_back = (
            dropout_mask_ptrs + block_min * stride_sd_n if ENABLE_DROPOUT else None
        )

        acc, l_i, m_i = _attn_fwd_inner_mask(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs_back,
            v_ptrs_back,
            stride_kn,
            stride_vn,
            stride_sd_n,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            s_dmask_ptrs_back,
            dropout_mask_ptrs_back,
            philox_seed,
            philox_ptrs,
            block_min,
            block_max,
            offs_n + (seqlen_q - seqlen_k) if IS_CAUSAL else 0,
            n_extra_tokens,
            alibi_slope,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            sm_scale,
            IS_CAUSAL,
            ENABLE_DROPOUT,
            RETURN_SCORES,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
        )

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if USE_SLIDING_WINDOW:
        invalid_mask = m_i == float("-inf")
        l_i_safe = tl.where(invalid_mask, 1.0, l_i)
        l_recip = 1 / l_i_safe[:, None]
    else:
        l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full(
                (BLOCK_DMODEL_POW2,), causal_start_idx, dtype=tl.int32
            )
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    overflow_size = end_m_idx - seqlen_q
    if softmax_lse_ptr is not None:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        # mi_base2 = m_i * RCP_LN2
        mi_base2 = m_i * RCP_LN2 * sm_scale
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2

        if IS_CAUSAL:
            # zero out nans caused by -infs when doing causal
            lse_causal_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
            softmax_lse = tl.where(lse_causal_mask, 0.0, softmax_lse)

        # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
        # This is only true for the last M block. For others, overflow_size will be -ve
        offs_lse = (
            off_z * stride_lse_z
            + off_q_head * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
            + offs_m * stride_lse_m
        )
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
            lse_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(
                softmax_lse_ptr + offs_lse, softmax_lse, mask=lse_mask
            )  # the log of the normalization constant
        else:
            tl.store(
                softmax_lse_ptr + offs_lse, softmax_lse
            )  # the log of the normalization constant

    # write back O
    offs_out = (
        off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    out_mask = tl.full([BLOCK_M, BLOCK_DMODEL_POW2], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_d[None, :] < BLOCK_DMODEL)
    op = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_out, op, mask=out_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    enable_dropout: bool,
    dtype: torch.dtype,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    if enable_dropout or dtype == torch.float32:
        return _get_config._config_dict["default"]["fwd"]["dropout_or_fp32"]
    else:
        return _get_config._config_dict["default"]["fwd"]["default"]


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    config: Optional[dict[str, any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if bias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    # FP8
    IS_FP8 = _is_fp8(q)
    FP8_MAX: tl.constexpr = torch.finfo(q.dtype).max
    is_varlen = True if cu_seqlens_q is not None else False

    if IS_FP8:
        o = torch.zeros_like(q, dtype=torch.float32)
    else:
        o = torch.zeros_like(q)
    if is_varlen:
        # Layout for q,k,v is thd ie [total_tokens, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
            q.shape[2],
        )
        seqlen_k, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        seqlen_k = k.shape[1]
        num_k_heads = k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    # padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_DMODEL_POW2 = max(BLOCK_DMODEL_POW2, 16)

    # softmax_lse [batch, num_q_heads, seqlen_q]
    if is_varlen:
        softmax_lse = torch.zeros(
            (q.shape[0], num_q_heads), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = (
            0,
            softmax_lse.stride(1),
            softmax_lse.stride(0),
        )
    else:
        softmax_lse = torch.zeros(
            (batch, num_q_heads, max_seqlen_q), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # exp_scores [batch, num_q_heads, seqlen_q, seqlen_k]
    enable_dropout = dropout_p > 0.0
    if enable_dropout:
        philox_seed = torch.randint(0, 0xFFFFFF, (1,))[
            0
        ].item()  # No specific reason to restrict range to 0xffffff
        philox_offset = torch.randint(0, 0xFFFFFF, (1,))[
            0
        ].item()  # Pass in an int, not Tensor
    else:
        philox_seed = 0
        philox_offset = 0
    if return_softmax or enable_dropout:
        s_dmask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        s_dmask = None
        dropout_mask = None

    if config is None:
        config = _get_config(enable_dropout, q.dtype)

    """
    # Tuned for MI300x
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "waves_per_eu": 2,
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 1,
    }
    # Dropout significantly increases VGPR usage so use small tiles
    if enable_dropout or q.dtype == torch.float32:
        config = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "waves_per_eu": 1,
            "num_warps": 2,
            "num_ctas": 1,
            "num_stages": 1,
        }
    """

    grid = lambda META: (  # noqa: E731
        batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),
    )

    _attn_fwd[grid](
        q,
        k,
        v,
        descale_q,
        descale_k,
        descale_v,
        o,
        alibi_slopes,
        s_dmask,
        dropout_mask,
        softmax_lse,
        *q_strides,
        *k_strides,
        *v_strides,
        descale_q.stride(0) if descale_q is not None else 0,
        descale_k.stride(0) if descale_k is not None else 0,
        descale_v.stride(0) if descale_v is not None else 0,
        *o_strides,
        alibi_slopes.stride(0) if alibi_slopes is not None else 0,
        alibi_slopes.stride(1) if alibi_slopes is not None else 0,
        s_dmask.stride(0) if s_dmask is not None else 0,
        s_dmask.stride(1) if s_dmask is not None else 0,
        s_dmask.stride(2) if s_dmask is not None else 0,
        s_dmask.stride(3) if s_dmask is not None else 0,
        stride_lse_z if softmax_lse is not None else 0,
        stride_lse_h if softmax_lse is not None else 0,
        stride_lse_m if softmax_lse is not None else 0,
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p,
        philox_seed,
        philox_offset,
        SEQLEN_Q=max_seqlen_q,
        SEQLEN_K=max_seqlen_k,
        IS_CAUSAL=causal,
        USE_SLIDING_WINDOW=window_size_left != -1 or window_size_right != -1,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BLOCK_DMODEL=head_sz,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        RETURN_SCORES=return_softmax,
        ENABLE_DROPOUT=enable_dropout,
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX,
        VARLEN=is_varlen,
        BATCH=batch,
        NUM_XCD=8,
        USE_INT64_STRIDES=_USE_INT64_STRIDES,
        **config,
    )

    return o, softmax_lse, s_dmask, philox_seed, philox_offset


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                config=config,
            )
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.bias = bias
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])

        print("Using fused backward kernel:", _USE_FUSED_BWD_KERNEL)

        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    config: Optional[dict[str, any]] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """

    return _FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        config,
    )


class _FlashAttnFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        # cast input to fp8
        fp8_dtype = arch_info.get_fp8_e4m3_dtype()
        q_fp8, descale_q = _cast_to_fp8(q, fp8_dtype, "bshd")
        k_fp8, descale_k = _cast_to_fp8(k, fp8_dtype, "bshd")
        v_fp8, descale_v = _cast_to_fp8(v, fp8_dtype, "bshd")

        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q_fp8,
                k_fp8,
                v_fp8,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=None,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                config=config,
            )
        )

        if is_grad:
            ctx.save_for_backward(
                q_fp8,
                k_fp8,
                v_fp8,
                out_padded,
                softmax_lse,
                descale_q,
                descale_k,
                descale_v,
            )
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes

        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q_fp8, k_fp8, v_fp8, out, softmax_lse, descale_q, descale_k, descale_v = (
            ctx.saved_tensors
        )
        dq, dk, dv = (
            torch.zeros_like(q_fp8, dtype=torch.float32),
            torch.zeros_like(k_fp8, dtype=torch.float32),
            torch.zeros_like(v_fp8, dtype=torch.float32),
        )
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])

        fp8_dtype = arch_info.get_fp8_e4m3_dtype()
        do_padded_fp8, descale_do = _cast_to_fp8(do_padded, fp8_dtype, "bshd")
        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q_fp8.shape[1],
                max_seqlen_k=k_fp8.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q_fp8.shape[1],
                max_seqlen_k=k_fp8.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )

        # dq = dq[..., : q_fp8.shape[-1]]  # We could have padded the head dimension
        # dk = dk[..., : k_fp8.shape[-1]]
        # dv = dv[..., : v_fp8.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_fp8_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    config: Optional[dict[str, any]] = None,
):
    return _FlashAttnFP8Func.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        config,
    )


class _FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        out,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0.0,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                config=config,
            )
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k
            )
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
        out = out_padded[..., :head_size_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        head_size_og = do.size(2)
        do_padded = do
        if head_size_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_og % 8])

        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
    out=None,
    config: Optional[dict[str, any]] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return _FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        out,
        torch.is_grad_enabled(),
        config,
    )


class _FlashAttnVarlenFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        # cast input to fp8
        fp8_dtype = arch_info.get_fp8_e4m3_dtype()
        q_fp8, descale_q = _cast_varlen_to_fp8(q, fp8_dtype, cu_seqlens=cu_seqlens_q)
        k_fp8, descale_k = _cast_varlen_to_fp8(k, fp8_dtype, cu_seqlens=cu_seqlens_k)
        v_fp8, descale_v = _cast_varlen_to_fp8(v, fp8_dtype, cu_seqlens=cu_seqlens_k)

        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q_fp8,
                k_fp8,
                v_fp8,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=None,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                config=config,
            )
        )
        if is_grad:
            ctx.save_for_backward(
                q_fp8,
                k_fp8,
                v_fp8,
                out_padded,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                descale_q,
                descale_k,
                descale_v,
            )
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes

        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (
            q_fp8,
            k_fp8,
            v_fp8,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            descale_q,
            descale_k,
            descale_v,
        ) = ctx.saved_tensors
        dq, dk, dv = (
            torch.zeros_like(q_fp8, dtype=torch.float32),
            torch.zeros_like(k_fp8, dtype=torch.float32),
            torch.zeros_like(v_fp8, dtype=torch.float32),
        )
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])

        fp8_dtype = arch_info.get_fp8_e4m3_dtype()
        do_padded_fp8, descale_do = _cast_varlen_to_fp8(
            do_padded, fp8_dtype, "thd", cu_seqlens_q
        )
        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        dq = dq[..., : q_fp8.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k_fp8.shape[-1]]
        dv = dv[..., : v_fp8.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_fp8_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
    config: Optional[dict[str, any]] = None,
):
    return _FlashAttnVarlenFP8Func.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
        config,
    )
