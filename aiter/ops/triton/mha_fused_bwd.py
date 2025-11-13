# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import torch
import triton

from aiter.ops.triton.utils.types import _is_fp8
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.mha_fused_bwd import (
    _bwd_preprocess,
    _bwd_kernel_dkdvdq_causal,
    _bwd_kernel_dkdvdq_noncausal,
    _get_config,
)
from aiter.ops.triton.utils.device_info import get_num_xcds


_LOGGER = AiterTritonLogger()


def flash_attn_fused_backward(
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
    config: Optional[Dict[str, any]] = None,
):
    """
    Flash Attention fused backward pass computing dQ, dK, dV in a single kernel using atomics.
    Supports variable-length sequences, GQA, FP8 quantization, and dropout.

    Args:
        do (torch.Tensor): Output gradient. Shape (batch, seqlen_q, num_q_heads, head_dim)
            or (total_tokens, num_q_heads, head_dim) for varlen.
        q (torch.Tensor): Query tensor from forward pass with same shape as do.
        k (torch.Tensor): Key tensor with shape (batch, seqlen_k, num_k_heads, head_dim)
            or (total_tokens_k, num_k_heads, head_dim) for varlen.
        v (torch.Tensor): Value tensor with same shape as k.
        o (torch.Tensor): Output from forward pass with same shape as q.
        softmax_lse (torch.Tensor): Log-sum-exp from forward pass with shape
            (batch, num_q_heads, seqlen_q) or (total_tokens, num_q_heads) for varlen.
        dq (torch.Tensor): Pre-allocated query gradient with same shape as q.
        dk (torch.Tensor): Pre-allocated key gradient with same shape as k.
        dv (torch.Tensor): Pre-allocated value gradient with same shape as v.
        dbias (torch.Tensor): Bias gradient (not supported, must be None).
        sm_scale (float): Softmax scale, typically 1/sqrt(head_dim).
        alibi_slopes (Optional[torch.Tensor]): ALiBi position bias slopes.
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
        config (Optional[Dict[str, any]]): Kernel tuning parameters (preprocess_kernel,
            dkdvdq_kernel_N64, dkdvdq_kernel_N128).

    Returns:
        torch.Tensor: Delta tensor (element-wise product of do and o) with shape matching softmax_lse.
    """
    _LOGGER.info(
        f"FLASH_ATTN_FUSED_BKWD: do={tuple(do.shape)} q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} "
        + f"dq={tuple(dq.shape)}  dk={tuple(dk.shape)}  dv={tuple(dv.shape)}"
    )
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")
    if q.shape[-1] == k.shape[-1] and k.shape[-1] > v.shape[-1]:
        raise ValueError(
            "'Fused' backward doesn't support Positional Encoding (PE). Please use 'one kernel' backward implementation for PE."
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
        # Layout for q,k,v is thd ie [total tokens, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
            q.shape[2],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

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
    if config is None:
        config = _get_config()

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
        BLOCK_D_MODEL=head_sz,
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

    # Fuses dk,dv and dq computations into one kernel using atomics
    if BLOCK_D_MODEL_POW2 > 160 or q.dtype == torch.float32:
        config_dkdvdq = config["dkdvdq_kernel_N64"]
    else:
        config_dkdvdq = config["dkdvdq_kernel_N128"]

    num_k_pids = (max_seqlen_k + config_dkdvdq["BLOCK_N"] - 1) // config_dkdvdq[
        "BLOCK_N"
    ]
    # NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if causal:
        grid_dkdvdq = (batch * num_q_heads * num_k_pids,)

        _bwd_kernel_dkdvdq_causal[grid_dkdvdq](
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BATCH=batch,
            NUM_K_PIDS=num_k_pids,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            NUM_XCD=get_num_xcds(),
            **config_dkdvdq,
        )
    else:
        # in non causal inner loop over grouped q heads
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,)
        _bwd_kernel_dkdvdq_noncausal[grid_dkdvdq](
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BATCH=batch,
            NUM_K_PIDS=num_k_pids,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **config_dkdvdq,
        )

    return delta
