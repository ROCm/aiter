# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Gate chunk cumsum for chunk_delta_attn."""

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices
from .utils import (
    autotune_cache_kwargs,
    check_shared_mem,
    chunk_delta_attn_autotune_configs,
    exp,
    input_guard,
    softplus,
)

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=chunk_delta_attn_autotune_configs(
        [triton.Config({"BS": BS}, num_warps=nw) for BS in BS_LIST for nw in [2, 4, 8]],
        default_config=triton.Config({"BS": 64}, num_warps=2),
    ),
    key=["H", "S", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gate_cumsum_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    CACHE_INPUT: tl.constexpr = "",
    CACHE_OUTPUT: tl.constexpr = "",
):
    i_s, i_t, i_bh = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
    p_s = s + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
    p_o = o + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]

    b_s = tl.load(p_s, mask=m_s, other=0.0, cache_modifier=CACHE_INPUT).to(tl.float32)

    if HAS_BIAS:
        b_bias = tl.load(dt_bias + i_h * S + o_s, mask=o_s < S, other=0.0).to(
            tl.float32
        )
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_s, cache_modifier=CACHE_OUTPUT)


@input_guard
def chunk_gate_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float | None = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype | None = torch.float,
    cache_input: str = "",
    cache_output: str = "",
) -> torch.Tensor:
    """Chunk-local gate cumsum.

    Args:
        cache_input: Cache modifier for loads (e.g. ".cs"), empty string to disable.
        cache_output: Cache modifier for stores (e.g. ".wt"), empty string to disable.
    """
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Varlen mode requires B=1"
    assert g.dim() == 4, f"Expected g.dim()==4, got {g.dim()}"
    B, T, H, S = g.shape
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_out = torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return (triton.cdiv(S, meta["BS"]), NT, B * H)

    chunk_gate_cumsum_kernel[grid](
        s=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=g_out,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        S=S,
        BT=BT,
        CACHE_INPUT=cache_input,
        CACHE_OUTPUT=cache_output,
    )
    return g_out
