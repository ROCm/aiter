# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sorted_stage2_kernel(
    a2_ptr,
    w2_ptr,
    out_ptr,
    sorted_token_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    token_num: tl.constexpr,
    model_dim: tl.constexpr,
    inter_dim: tl.constexpr,
    topk: tl.constexpr,
    stride_a2m: tl.constexpr,
    stride_a2k: tl.constexpr,
    stride_w2e: tl.constexpr,
    stride_w2h: tl.constexpr,
    stride_w2i: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(model_dim, BLOCK_N)
    pid_m = pid // n_blocks
    pid_n = pid - pid_m * n_blocks

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    num_valid = tl.load(num_valid_ids_ptr)
    packed = tl.load(
        sorted_token_ids_ptr + row_offsets,
        mask=row_offsets < num_valid,
        other=token_num,
    )
    token_ids = packed & 0xFFFFFF
    valid_rows = (row_offsets < num_valid) & (token_ids < token_num)
    expert_id = tl.load(sorted_expert_ids_ptr + pid_m)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, inter_dim, BLOCK_K):
        ks = k0 + k_offsets
        a = tl.load(
            a2_ptr + row_offsets[:, None] * stride_a2m + ks[None, :] * stride_a2k,
            mask=valid_rows[:, None] & (ks[None, :] < inter_dim),
            other=0.0,
        )
        b = tl.load(
            w2_ptr
            + expert_id * stride_w2e
            + col_offsets[None, :] * stride_w2h
            + ks[:, None] * stride_w2i,
            mask=(col_offsets[None, :] < model_dim) & (ks[:, None] < inter_dim),
            other=0.0,
        )
        acc += tl.dot(a, b)

    weights = tl.load(sorted_weights_ptr + row_offsets, mask=valid_rows, other=0.0)
    acc *= weights[:, None]
    tl.atomic_add(
        out_ptr + token_ids[:, None] * stride_om + col_offsets[None, :] * stride_on,
        acc,
        sem="relaxed",
        mask=valid_rows[:, None] & (col_offsets[None, :] < model_dim),
    )


def sorted_row_indices(
    sorted_token_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_valid = int(num_valid_ids.reshape(-1)[0].item())
    packed = sorted_token_ids[:num_valid]
    token_ids = packed & 0xFFFFFF
    slot_ids = packed >> 24
    valid = (token_ids < token_num) & (slot_ids < topk)
    flat_indices = token_ids[valid].to(torch.long) * topk + slot_ids[valid].to(
        torch.long
    )
    return flat_indices, valid


def pack_a2_sorted(
    a2_current: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    token_num, _, inter_dim = a2_current.shape
    num_valid = int(num_valid_ids.reshape(-1)[0].item())
    flat_indices, valid = sorted_row_indices(
        sorted_token_ids, num_valid_ids, token_num, topk
    )
    a2_sorted = torch.zeros(
        (num_valid, inter_dim), dtype=a2_current.dtype, device=a2_current.device
    )
    a2_sorted[valid] = a2_current.reshape(token_num * topk, inter_dim).index_select(
        0, flat_indices
    )
    return a2_sorted


def unpack_a2_sorted(
    a2_sorted: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
) -> torch.Tensor:
    flat_indices, valid = sorted_row_indices(
        sorted_token_ids, num_valid_ids, token_num, topk
    )
    a2_current = torch.zeros(
        (token_num * topk, a2_sorted.shape[-1]),
        dtype=a2_sorted.dtype,
        device=a2_sorted.device,
    )
    a2_current[flat_indices] = a2_sorted[valid]
    return a2_current.view(token_num, topk, a2_sorted.shape[-1])


def sorted_stage2_triton(
    a2_sorted: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _, model_dim, inter_dim = w2.shape
    if out is None:
        out = torch.zeros((token_num, model_dim), dtype=torch.float32, device=w2.device)
    if out.dtype != torch.float32:
        raise ValueError("sorted_stage2_triton currently expects a fp32 output buffer")
    if not a2_sorted.is_contiguous():
        a2_sorted = a2_sorted.contiguous()
    if not w2.is_contiguous():
        w2 = w2.contiguous()

    num_valid = int(num_valid_ids.reshape(-1)[0].item())
    grid = (triton.cdiv(num_valid, block_m) * triton.cdiv(model_dim, block_n),)
    _sorted_stage2_kernel[grid](
        a2_sorted,
        w2,
        out,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        token_num,
        model_dim,
        inter_dim,
        topk,
        a2_sorted.stride(0),
        a2_sorted.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out
