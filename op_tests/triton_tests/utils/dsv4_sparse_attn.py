# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared inputs for the DSv4 sparse-attention tests and benchmarks.

The packed ("2buff") operand is a 512 B row and a separate RoPE plane:

  packed row [512 B] = [ NoPE 448 x fp8-e4m3
                       | 14 E8M0 scale bytes, each 64-elt group's scale
                         written TWICE (s0,s0,s1,s1,...,s6,s6)
                       | 50 B pad ]
  rope plane [64] bf16
"""

import torch

V4_DIM_NOPE = 448
V4_DIM_ROPE = 64
V4_DIM_QK = V4_DIM_NOPE + V4_DIM_ROPE  # 512
V4_TILE = 64
V4_NUM_TILES = V4_DIM_NOPE // V4_TILE  # 7
V4_FP8 = torch.float8_e4m3fn


def v4_pack_2buff(x_bf16):
    """``[..., 512]`` bf16 (NoPE||RoPE) -> ``(packed [..., 512] fp8, rope [..., 64] bf16)``."""
    assert x_bf16.shape[-1] == V4_DIM_QK
    lead = x_bf16.shape[:-1]
    nope = x_bf16[..., :V4_DIM_NOPE].float()
    rope = x_bf16[..., V4_DIM_NOPE:].contiguous()

    tiled = nope.reshape(*lead, V4_NUM_TILES, V4_TILE)
    fp8_max = float(torch.finfo(V4_FP8).max)
    # amax/fp8_max rounded UP to a power of two, exactly as E8M0 stores it.
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    nope_fp8 = (tiled / scale.unsqueeze(-1)).to(V4_FP8).reshape(*lead, V4_DIM_NOPE)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)

    packed = torch.zeros((*lead, V4_DIM_QK), dtype=torch.uint8, device=x_bf16.device)
    packed[..., :V4_DIM_NOPE] = nope_fp8.view(torch.uint8)
    # The kernel reads each group's scale twice (its scaled-MMA blocks are 32
    # elements wide, the quant group is 64), so duplicate every byte.
    packed[..., V4_DIM_NOPE : V4_DIM_NOPE + 2 * V4_NUM_TILES] = e8m0.repeat_interleave(
        2, dim=-1
    )
    return packed.view(V4_FP8), rope


def v4_unpack_2buff(packed, rope):
    """Inverse of ``v4_pack_2buff`` -> ``[..., 512]`` bf16."""
    lead = packed.shape[:-1]
    u8 = packed.view(torch.uint8)
    nope = (
        u8[..., :V4_DIM_NOPE].view(V4_FP8).float().reshape(*lead, V4_NUM_TILES, V4_TILE)
    )
    # One byte per group: read the first of each duplicated pair.
    exps = u8[..., V4_DIM_NOPE : V4_DIM_NOPE + 2 * V4_NUM_TILES : 2].to(torch.int32)
    scale = torch.pow(2.0, (exps - 127).float())
    out = torch.empty((*lead, V4_DIM_QK), dtype=torch.bfloat16, device=packed.device)
    out[..., :V4_DIM_NOPE] = (
        (nope * scale.unsqueeze(-1)).reshape(*lead, V4_DIM_NOPE).to(torch.bfloat16)
    )
    out[..., V4_DIM_NOPE:] = rope
    return out


def make_pa_prefill_sparse_inputs(
    T: int,
    H: int,
    D: int,
    prefix_len_per_token: int,
    extend_len_per_token: int,
    total_pages: int,
    total_extend_tokens: int,
    dtype=torch.bfloat16,
    seed: int = 0,
    include_sentinels: bool = False,
    variable_len: bool = False,
):
    """Random ``pa_prefill_sparse`` operands: q, both pools, ragged indices, sink.

    Returns ``(q, unified_kv, p_indices, p_indptr, kv, e_indices, e_indptr,
    attn_sink, softmax_scale)``. ``include_sentinels`` sets 1/16 of the slots
    to ``-1``.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")

    q = torch.randn(T, H, D, dtype=dtype, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=dtype, device=device) * 0.5
    kv = torch.randn(total_extend_tokens, D, dtype=dtype, device=device) * 0.5
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    def ragged(per_token, pool):
        if variable_len:
            lens = torch.randint(
                low=1, high=per_token + 1, size=(T,), device=device, dtype=torch.int64
            )
        else:
            lens = torch.full((T,), per_token, device=device, dtype=torch.int64)
        indptr = torch.zeros(T + 1, device=device, dtype=torch.int64)
        indptr[1:] = lens.cumsum(0)
        total = int(indptr[-1].item())
        idx = torch.randint(
            low=0, high=pool, size=(total,), device=device, dtype=torch.int32
        )
        return idx, indptr, total

    p_indices, p_indptr, total_p = ragged(prefix_len_per_token, total_pages)
    e_indices, e_indptr, total_e = ragged(extend_len_per_token, total_extend_tokens)

    if include_sentinels:
        if total_p > 0:
            n_s = max(1, total_p // 16)
            pos = torch.randperm(total_p, device=device)[:n_s]
            p_indices[pos] = -1
        if total_e > 0:
            n_s = max(1, total_e // 16)
            pos = torch.randperm(total_e, device=device)[:n_s]
            e_indices[pos] = -1

    return (
        q,
        unified_kv,
        p_indices,
        p_indptr.to(torch.int32),
        kv,
        e_indices,
        e_indptr.to(torch.int32),
        attn_sink,
        float(D) ** -0.5,
    )
