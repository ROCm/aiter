# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from aiter.ops.triton.utils.types import _is_fp8
from aiter.ops.triton.utils.logger import AiterTritonLogger

from aiter.ops.triton._triton_kernels.mha_onekernel_bwd import (
    _bwd_preprocess,
    bwd_kernel_causal,
    bwd_kernel_noncausal,
    _get_config,
)

_LOGGER = AiterTritonLogger()


# NOTE: triton fails to import tl.constexprs so create them here for the file
DROPOUT_USE_PYTORCH = False
DROPOUT_DUMP = False

tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)


def flash_attn_onekernel_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dbias: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
    USE_INT64_STRIDES: Optional[bool] = False,
    sink: Optional[torch.Tensor] = None,
    dsink: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, any]] = None,
):
    """
    Flash Attention one-kernel backward pass with positional encoding support.
    Computes dQ, dK, dV in separate passes without atomics. Supports Q/K head dimensions
    larger than V for positional encoding.

    Args:
        do (torch.Tensor): Output gradient. Shape (batch, seqlen_q, num_q_heads, v_head_dim)
            or (total_tokens, num_q_heads, v_head_dim) for varlen.
        q (torch.Tensor): Query tensor with shape (batch, seqlen_q, num_q_heads, qk_head_dim).
            qk_head_dim may be larger than v_head_dim for positional encoding.
        k (torch.Tensor): Key tensor with shape (batch, seqlen_k, num_k_heads, qk_head_dim).
        v (torch.Tensor): Value tensor with shape (batch, seqlen_k, num_k_heads, v_head_dim).
        o (torch.Tensor): Output from forward pass with same shape as do.
        softmax_lse (torch.Tensor): Log-sum-exp from forward pass with shape
            (batch, num_q_heads, seqlen_q) or (total_tokens, num_q_heads) for varlen.
        dq (torch.Tensor): Pre-allocated query gradient with same shape as q.
        dk (torch.Tensor): Pre-allocated key gradient with same shape as k.
        dv (torch.Tensor): Pre-allocated value gradient with same shape as v.
        dbias (torch.Tensor): Bias gradient (not supported, must be None).
        sm_scale (float): Softmax scale, typically 1/sqrt(head_dim).
        alibi_slopes (Optional[torch.Tensor]): ALiBi position bias slopes with shape (num_q_heads,).
        causal (bool): Apply causal masking.
        cu_seqlens_q (Optional[torch.Tensor]): Cumulative sequence lengths for query with shape
            (batch + 1,). Enables variable-length mode.
        cu_seqlens_k (Optional[torch.Tensor]): Cumulative sequence lengths for key with shape
            (batch + 1,).
        max_seqlen_q (int): Maximum query sequence length in batch.
        max_seqlen_k (int): Maximum key sequence length in batch.
        dropout_p (float): Dropout probability. 0.0 disables dropout.
        philox_seed (Optional[int]): Random seed for dropout.
        philox_offset (Optional[int]): Random offset for dropout.
        descale_q (Optional[torch.Tensor]): FP8 descaling factor for q.
        descale_k (Optional[torch.Tensor]): FP8 descaling factor for k.
        descale_v (Optional[torch.Tensor]): FP8 descaling factor for v.
        descale_do (Optional[torch.Tensor]): FP8 descaling factor for do.
        USE_INT64_STRIDES (Optional[bool]): Use 64-bit stride indexing for large tensors.
        sink (Optional[torch.Tensor]): Attention sink scores (one per Q head). Shape (num_q_heads,).
        dsink (Optional[torch.Tensor]): Pre-allocated sink gradient with same shape as sink.
        config (Optional[Dict[str, any]]): Kernel tuning parameters (preprocess_kernel,
            onekernel, onekernel_pe).

    Returns:
        torch.Tensor: Delta tensor (element-wise product of do and o) with shape matching softmax_lse.
    """
    _LOGGER.info(
        f"FLASH_ATTN_ONEKERNEL_BKWD: do={tuple(do.shape)} q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} "
        + f"dq={tuple(dq.shape)}  dk={tuple(dk.shape)}  dv={tuple(dv.shape)}"
    )
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )

    IS_FP8 = _is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        descale_strides = (
            descale_q.stride(0),
            descale_k.stride(0),
            descale_v.stride(0),
            descale_do.stride(0),
        )
    else:
        FP8_MAX = None
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = (
            stride_descale_do_z
        ) = None
        descale_strides = (
            stride_descale_q_z,
            stride_descale_k_z,
            stride_descale_v_z,
            stride_descale_do_z,
        )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    # get strides and shape
    if IS_VARLEN:
        # Layout is thd.
        # q and k are [total_tokens, num_head, head_dim_qk].
        # v is [total_tokens, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        dv_strides = (0, dv.stride(1), dv.stride(0), dv.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v]
        batch, seqlen_q, num_q_heads = q.shape[:-1]
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        dv_strides = (dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_D_MODEL_PE_POW2 = (
        0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    )
    assert (pe_head_dim == 0 and BLOCK_D_MODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_D_MODEL_POW2 and pe_head_dim == BLOCK_D_MODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."
    assert (not IS_FP8) or (
        IS_FP8 and pe_head_dim == 0
    ), "Positional encoding doesn't support FP8."

    assert (sink is None) or (
        sink is not None and sink.dim() == 1 and sink.shape[0] == num_q_heads
    ), "Sink must be 1D and have one element per query head."
    assert (dsink is None) or (
        dsink is not None and dsink.dim() == 1 and dsink.shape[0] == num_q_heads
    ), "Sink gradient must be 1D and have one element per query head."
    assert (sink is None) == (
        dsink is None
    ), "Sink and its gradient must be both present or absent."

    # Configs
    if config is None:
        config = _get_config()

    # init delta
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        # [total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta.stride(1), delta.stride(0))
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta.stride()

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    _bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        descale_strides[3],
        cu_seqlens_q,
        max_seqlen_q,
        descale_do,
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=v_head_dim,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
        IS_FP8=IS_FP8,
    )

    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    seqlen = max(max_seqlen_q, max_seqlen_k)

    # "onekernel_pe" is for Positional Encoding (PE) causal case, it's going to be
    # used if present. Otherwise, fallback to default "onekernel" config.
    config_onekernel = (
        config["onekernel_pe"]
        if (pe_head_dim > 0 and causal and "onekernel_pe" in config)
        else config["onekernel"]
    )
    grid = (
        num_k_heads,
        triton.cdiv(seqlen, config_onekernel["BLOCK_N1"]),
        batch,
    )

    if causal:
        bwd_kernel_causal[grid](
            q,
            k,
            v,
            sink,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            dsink,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            HEAD_DIM=v_head_dim,
            ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
            PE_HEAD_DIM=pe_head_dim,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            ENABLE_SINK=sink is not None,
            **config_onekernel,
        )
    else:
        bwd_kernel_noncausal[grid](
            q,
            k,
            v,
            sink,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            dsink,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            HEAD_DIM=v_head_dim,
            ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
            PE_HEAD_DIM=pe_head_dim,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            ENABLE_SINK=sink is not None,
            **config_onekernel,
        )

    return delta
