# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Host-side SpargeAttn / VFA block-sparse attention preparation.

Self-contained backport of ``build_attention_lut`` (and the SpargeAttn meansim
block-selection it rides on) from the ``tianxing/sage_vfa`` branch, vendored
here so the all-fp8 block-sparse + VFA asm kernel
(``flash_attn_fp8_sparse_vfa_pertensor_func``) can be driven from a Q/K-derived
ragged LUT on this branch (which ships the wrapper but not the LUT builder).

Everything ``build_attention_lut`` needs -- the mean-pool proxy kernel, the
SpargeAttn block-map kernels and the LUT-fill kernel -- is inlined so this
module has no new cross-file aiter dependencies (and no symbol collisions with
this branch's pre-existing ``_triton_kernels/attention/block_lut.py``).

Turns Q/K into the ragged block-sparse look-up table ``(kv_block_indices,
lut_start, lut_count)`` plus the matching ``freeze_softmax_max_count`` for the
VFA running-max-freeze path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ============================================================================
# Triton kernels (vendored; see tianxing/sage_vfa
# aiter/ops/triton/_triton_kernels/{pool.py, attention/block_lut.py})
# ============================================================================
@triton.jit
def _triton_bmm_pool_sim_simmean(
    x_ptr,
    pool_ptr,
    sim_ptr,
    simthreshd1_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    SKIP_SIM: tl.constexpr = False,
):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    _, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb * BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = (
        x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    )
    x = tl.load(x_ptrs, mask=xmask)
    BS_ = BS if (N - nb * BS) >= BS else (N - nb * BS)

    x_fp32 = x.to(tl.float32)
    is_nan = x_fp32 != x_fp32
    x_fp32 = tl.where(is_nan, 0.0, x_fp32)

    pool = tl.sum(x_fp32, axis=0) / BS_
    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)

    if not SKIP_SIM:
        cur_h1 = tl.load(simthreshd1_ptr + h)
        x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
        x = (x / x_norm).to(tl.float16)
        is_nan = x != x
        x = tl.where(is_nan, 0.0, x)

        grams = tl.dot(x, tl.trans(x))
        sum_value = tl.sum(grams).to(tl.float32)
        cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

        sim_offset = b * H * NB + h * NB + nb
        tl.store(sim_ptr + sim_offset, cur_sim)


@triton.jit()
def _block_attn_mask_to_lut_kernel(
    mask_ptr,
    lut_start_ptr,
    lut_count_ptr,
    kv_block_indices_ptr,
    stride_mask_b,
    stride_mask_h,
    stride_mask_qb,
    stride_mask_kb,
    num_heads,
    num_q_blocks,
    num_kv_blocks,
    BLOCK_KB: tl.constexpr,
):
    linear_idx = tl.program_id(0)
    num_entries = num_heads * num_q_blocks
    b = linear_idx // num_entries
    remainder = linear_idx % num_entries
    h = remainder // num_q_blocks
    qb = remainder % num_q_blocks

    base = tl.load(lut_start_ptr + linear_idx)
    write_offset = 0

    for start_kb in range(0, num_kv_blocks, BLOCK_KB):
        kb_offs = start_kb + tl.arange(0, BLOCK_KB)
        in_bounds = kb_offs < num_kv_blocks

        row_base = b * stride_mask_b + h * stride_mask_h + qb * stride_mask_qb
        mask_ptrs = mask_ptr + row_base + kb_offs * stride_mask_kb
        mask_chunk = tl.load(mask_ptrs, mask=in_bounds, other=0)

        mask_vals = (mask_chunk != 0).to(tl.int32)
        mask_vals = tl.where(in_bounds, mask_vals, 0)
        cumsum = tl.cumsum(mask_vals, axis=0)
        store_offsets = base + write_offset + cumsum - 1
        chunk_kb = (start_kb + tl.arange(0, BLOCK_KB)).to(tl.int32)
        tl.store(
            kv_block_indices_ptr + store_offsets,
            chunk_kb,
            mask=mask_vals != 0,
        )
        write_offset = write_offset + tl.sum(mask_vals)


@triton.jit
def _triton_fill_block_map_kernel(
    final_map,
    num_to_select,
    sorted_indices,
    NK: tl.constexpr,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    H, Q = tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (
        (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    )
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)


@triton.jit
def _triton_fill_causal_mask_kernel(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    K = tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)


# ============================================================================
# Host wrappers
# ============================================================================
def get_pool_sim_triton_simmean(
    x: torch.Tensor,
    block_size: int,
    simthreshd1: torch.Tensor,
    attention_scored_only: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mean-pool each block and flag internally-similar (meansim) blocks."""
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    if attention_scored_only:
        sim_blocks = None
        sim_arg = pool
    else:
        sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
        sim_arg = sim_blocks
    grid = (B, H, nblock)
    _triton_bmm_pool_sim_simmean[grid](
        x,
        pool,
        sim_arg,
        simthreshd1,
        N=N,
        D=D,
        BS=block_size,
        SKIP_SIM=attention_scored_only,
    )
    return pool, sim_blocks


def block_attn_mask_to_lut(
    block_attn_mask: torch.Tensor,
    lut_start: torch.Tensor,
    lut_count: torch.Tensor,
    kv_block_indices: torch.Tensor,
    BLOCK_KB: int = 128,
):
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    num_programs = batch * num_heads * num_q_blocks
    grid = (num_programs,)
    _block_attn_mask_to_lut_kernel[grid](
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        stride_mask_b=block_attn_mask.stride(0),
        stride_mask_h=block_attn_mask.stride(1),
        stride_mask_qb=block_attn_mask.stride(2),
        stride_mask_kb=block_attn_mask.stride(3),
        num_heads=num_heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        BLOCK_KB=BLOCK_KB,
    )


def fill_block_map_triton(
    final_map: torch.Tensor,
    num_to_select: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    _triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map


def fill_causal_mask_triton(mask: torch.Tensor, BqdivBk: float) -> torch.Tensor:
    assert mask.dim() == 2
    _triton_fill_causal_mask_kernel[mask.shape](mask, BqdivBk)
    return mask


def block_attn_mask_to_ragged_lut(
    block_attn_mask: torch.Tensor,
    num_heads: Optional[int] = None,
    return_none_if_dense: bool = False,
    BLOCK_KB: int = 128,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert a dense block attention mask to a ragged ``(kv_idx, start, count)`` LUT."""
    device = block_attn_mask.device

    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        batch, num_q_blocks, num_kv_blocks = block_attn_mask.shape
        if return_none_if_dense and block_attn_mask.all():
            return None
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(
            batch, num_heads, num_q_blocks, num_kv_blocks
        )

    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    if return_none_if_dense and block_attn_mask.all():
        return None

    counts = block_attn_mask.to(torch.int32).sum(dim=-1)
    lut_count = counts.reshape(-1)
    lut_start = torch.cumsum(lut_count, dim=0) - lut_count

    max_count = batch * num_heads * num_q_blocks * num_kv_blocks
    kv_block_indices = torch.empty(max_count, dtype=torch.int32, device=device)
    block_attn_mask_to_lut(
        block_attn_mask.contiguous(),
        lut_start,
        lut_count,
        kv_block_indices,
        BLOCK_KB=BLOCK_KB,
    )
    return kv_block_indices, lut_start, lut_count


# ============================================================================
# SpargeAttn block-sparse mask construction (meansim proxy)
# ============================================================================
def get_block_map_meansim(
    q: torch.Tensor,
    k: torch.Tensor,
    is_causal: bool = False,
    BLKQ: int = 64,
    BLKK: int = 64,
    simthreshd1: torch.Tensor = None,
    cdfthreshd: torch.Tensor = None,
    attention_sink: bool = False,
    attention_scored_only: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Build a SpargeAttn block-sparse mask via the mean-similarity proxy."""
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_q, sim_q = get_pool_sim_triton_simmean(
        q, BLKQ, simthreshd1, attention_scored_only
    )
    pooled_k, sim_k = get_pool_sim_triton_simmean(
        k, BLKK, simthreshd1, attention_scored_only
    )
    pooled_score = pooled_q @ pooled_k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if attention_scored_only:
        return None, pooled_score

    neg_inf = pooled_score.new_full((), float("-inf"))
    sim_k = sim_k.unsqueeze(-2).expand(-1, -1, nq, -1)
    sim_q = sim_q.unsqueeze(-1).expand(-1, -1, -1, nk)

    prob = torch.where(sim_k, pooled_score, neg_inf)
    causal_mask = None
    if is_causal:
        causal_mask = fill_causal_mask_triton(
            torch.empty(nq, nk, device=q.device, dtype=torch.bool), BLKQ / BLKK
        )
        prob = torch.where(causal_mask[None, None], prob, neg_inf)
    prob = prob.softmax(-1)

    sorted_score = torch.sort(prob, dim=-1, descending=True)
    cdf = sorted_score.values.cumsum(dim=-1)
    H, K = cdf.shape[1], cdf.shape[-1]
    ge = cdf >= cdfthreshd.view(1, H, 1, 1)
    idx = ge.to(torch.uint8).argmax(dim=-1)
    num_to_select = torch.where(ge.any(dim=-1), idx, idx.new_full((), K))

    final_map = fill_block_map_triton(
        torch.zeros_like(prob, dtype=torch.bool), num_to_select, sorted_score.indices
    )
    final_map = final_map | ~sim_k | ~sim_q
    if is_causal:
        final_map = final_map * causal_mask[None, None]
    if attention_sink:
        final_map[:, :, :, 0] = 1
    return final_map, pooled_score


def _num_text_blocks(text_len: int, block_m: int, block_n: int) -> Tuple[int, int]:
    return (
        (text_len + block_m - 1) // block_m,
        (text_len + block_n - 1) // block_n,
    )


def _assemble_full_block_mask(
    image_block_mask: torch.Tensor,
    image_len_q: int,
    image_len_k: int,
    text_len: int,
    block_m: int,
    block_n: int,
) -> torch.Tensor:
    if text_len == 0:
        return image_block_mask

    B, H, n_iq, n_ik = image_block_mask.shape
    n_text_q, n_text_k = _num_text_blocks(text_len, block_m, block_n)

    full = torch.zeros(
        (B, H, n_iq + n_text_q, n_ik + n_text_k),
        dtype=image_block_mask.dtype,
        device=image_block_mask.device,
    )
    full[:, :, :n_iq, :n_ik] = image_block_mask
    full[:, :, :, -n_text_k:] = True
    full[:, :, -n_text_q:, :] = True
    if image_len_q % block_m != 0:
        full[:, :, image_len_q // block_m, :] = True
    if image_len_k % block_n != 0:
        full[:, :, :, image_len_k // block_n] = True
    return full


def block_attn_mask_to_ragged_lut_topn_front(
    block_attn_mask: torch.Tensor,
    pooled_score: torch.Tensor,
    sample_n: int,
    num_heads: Optional[int] = None,
    force_front_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a ragged LUT with the top-``sample_n`` scored K blocks emitted first."""
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        B, Q, K = block_attn_mask.shape
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(B, num_heads, Q, K)
        if force_front_mask is not None and force_front_mask.dim() == 3:
            force_front_mask = force_front_mask.unsqueeze(1).expand(B, num_heads, Q, K)

    B, H, Q, K = block_attn_mask.shape
    assert pooled_score.shape[:3] == (B, H, Q) and pooled_score.shape[-1] == K, (
        f"pooled_score shape {tuple(pooled_score.shape)} does not match mask "
        f"{(B, H, Q, K)}"
    )
    device = block_attn_mask.device

    attended = block_attn_mask.to(torch.bool)
    lut_count = attended.sum(-1).to(torch.int32).reshape(-1)
    lut_start = torch.cumsum(lut_count, 0) - lut_count

    if force_front_mask is None:
        force_front = torch.zeros_like(attended)
    else:
        force_front = force_front_mask.to(torch.bool).expand(B, H, Q, K) & attended

    neg_inf = pooled_score.new_full((), float("-inf"))
    masked_score = torch.where(attended, pooled_score.to(torch.float32), neg_inf)

    is_topn = torch.zeros((B, H, Q, K), dtype=torch.bool, device=device)
    n = min(sample_n, K)
    if n > 0:
        sample_score = torch.where(force_front, neg_inf, masked_score)
        topk = sample_score.topk(n, dim=-1)
        is_topn.scatter_(-1, topk.indices, topk.values > neg_inf)

    col = torch.arange(K, device=device).view(1, 1, 1, K)
    priority = torch.where(
        ~attended,
        3,
        torch.where(is_topn, 0, torch.where(force_front, 1, 2)),
    )
    tiebreak = torch.where(is_topn, -masked_score, col.to(torch.float32))

    o1 = torch.argsort(tiebreak, dim=-1, stable=True)
    order = torch.gather(
        o1, -1, torch.argsort(torch.gather(priority, -1, o1), dim=-1, stable=True)
    )

    R = B * H * Q
    rows = order.reshape(R, K)
    col = torch.arange(K, device=device)
    keep = col[None, :] < lut_count[:, None]
    sink = R * K
    dest = torch.where(
        keep,
        lut_start[:, None].to(torch.long) + col[None, :],
        col.new_full((), sink),
    )
    packed = torch.empty(sink + 1, dtype=torch.int32, device=device)
    packed.scatter_(0, dest.reshape(-1), rows.reshape(-1).to(torch.int32))
    kv_block_indices = packed[:sink]
    return kv_block_indices, lut_start, lut_count


def build_attention_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    simthreshd1: torch.Tensor,
    cdfthreshd: torch.Tensor,
    use_vfa: bool = True,
    n_sample: int = 8,
    is_causal: bool = False,
    static_block_mask: Optional[torch.Tensor] = None,
    text_len: int = 0,
    block_m: int = 128,
    block_n: int = 128,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Build a SpargeAttn block-sparse ragged LUT (optionally VFA-front-loaded).

    ``simthreshd1``/``cdfthreshd`` are per-head ``(H,)`` fp32 tensors. With
    ``use_vfa=True`` the top-``n_sample`` selected blocks are front-loaded per
    segment and the returned ``freeze_softmax_max_count`` is
    ``n_sample (+ n_text_blocks)``; with ``use_vfa=False`` blocks are emitted in
    ascending order and the freeze count is ``-1`` (never frozen).
    """
    image_q = q[:, :, : q.shape[2] - text_len, :] if text_len > 0 else q
    image_k = k[:, :, : k.shape[2] - text_len, :] if text_len > 0 else k
    image_len_q = q.shape[2] - text_len
    image_len_k = k.shape[2] - text_len
    n_text_k = _num_text_blocks(text_len, block_m, block_n)[1] if text_len > 0 else 0

    image_mask, image_score = get_block_map_meansim(
        image_q,
        image_k,
        is_causal=is_causal,
        BLKQ=block_m,
        BLKK=block_n,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
    )

    if static_block_mask is not None:
        image_mask = image_mask | static_block_mask[None, None, ...]

    full_mask = _assemble_full_block_mask(
        image_mask, image_len_q, image_len_k, text_len, block_m, block_n
    )

    if not use_vfa:
        return block_attn_mask_to_ragged_lut(full_mask), -1

    B, H, n_iq, n_ik = image_mask.shape
    n_tq, n_tk = full_mask.shape[-2], full_mask.shape[-1]
    full_score = full_mask.new_full(
        (B, H, n_tq, n_tk), float("-inf"), dtype=torch.float32
    )
    full_score[:, :, :n_iq, :n_ik] = image_score.to(torch.float32)

    force_front = None
    if n_text_k > 0:
        force_front = torch.zeros((B, H, n_tq, n_tk), dtype=torch.bool, device=q.device)
        force_front[:, :, :, -n_text_k:] = True

    block_lut = block_attn_mask_to_ragged_lut_topn_front(
        full_mask, full_score, n_sample, force_front_mask=force_front
    )
    return block_lut, n_sample + n_text_k
