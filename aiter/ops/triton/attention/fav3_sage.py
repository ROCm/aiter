# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple
import torch
import aiter
import triton
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    sage_quant,
    sage_fwd,
    map_dims,
)
from aiter.ops.triton.utils._triton import arch_info


def get_sage_fwd_configs():
    arch = arch_info.get_arch()
    if arch == "gfx950":
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "waves_per_eu": 2,
            "PRE_LOAD_V": False,
            "num_stages": 5,
            "num_warps": 8,
        }
        
    elif arch == "gfx942":
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "waves_per_eu": 2,
            "PRE_LOAD_V": False,
            "num_stages": 2,
            "num_warps": 8,
        }
    else:
        # return tuned config for MI300X by default
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "waves_per_eu": 2,
            "PRE_LOAD_V": False,
            "num_stages": 2,
            "num_warps": 8,
        }


class _FAv3SageWrapperFunc(torch.autograd.Function):
    """
    Sage Attention v1 wrapper that maintains high-precision inputs/outputs.

    This wrapper allows users to pass BF16/FP32 tensors and automatically handles
    the quantization internally, maintaining backward compatibility with
    high-precision training workflows.

    Forward: BF16/FP32 -> Int8 (Q & K) + FP16 V -> sage_attn -> FP32 output
    Backward: not supported yet
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        deterministic: bool,
        sm_margin: int,
        return_lse: bool = True,
        layout: str = "bshd",
        config: Optional[dict] = None,
    ):
        # 1. Dimension Mapping & Config Setup
        bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd_map)
        _, seqlen_k, num_kv_heads, _ = map_dims(k.shape, bshd_map)

        if config is None:
            config = get_sage_fwd_configs()

        BLKQ, BLKK = config["BLOCK_M"], config["BLOCK_N"]

        # 2. Validation: Early Exit for unsupported features
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0 or sm_margin != 0:
            raise NotImplementedError(
                "softcap/sm_margin not supported in FP8 high-precision API"
            )

        if (q.requires_grad or k.requires_grad or v.requires_grad) and not return_lse:
            raise ValueError(
                "return_lse must be True during training (requires_grad=True)"
            )

        # 3. Quantization
        # Note: softmax_scale is integrated into quantization descaling
        softmax_scale = softmax_scale or (head_dim**-0.5)
        fp8_dtype = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_dtype).max

        q_int8, q_descale, k_int8, k_descale, v_fp8, v_descale = sage_quant(
            q,
            k,
            v,
            fp8_dtype,
            fp8_max,
            sm_scale=softmax_scale,
            BLKQ=BLKQ,
            BLKK=BLKK,
            layout=layout,
        )

        # 4. Verify Descale Shapes (Grouped scaling for GQA/MQA)
        num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
        num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

        expected_q_ds = (batch, num_q_heads, num_q_blocks)
        expected_k_ds = (batch, num_kv_heads, num_k_blocks)

        assert (
            q_descale.shape == expected_q_ds
        ), f"q_descale shape {q_descale.shape} != {expected_q_ds}"
        assert (
            k_descale.shape == expected_k_ds
        ), f"k_descale shape {k_descale.shape} != {expected_k_ds}"

        # 5. Execution
        out, softmax_lse = fav3_sage_func(
            q_int8,
            k_int8,
            v_fp8,
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            window_size,
            attention_chunk,
            softcap,
            sm_margin,
            return_lse,
            layout,
            config,
        )

        # 6. Context Saving for Backward
        if return_lse:
            ctx.save_for_backward(
                q_int8, k_int8, v_fp8, out, softmax_lse, q_descale, k_descale
            )
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.deterministic = deterministic
            ctx.sm_margin = sm_margin
            ctx.input_dtype = q.dtype
            ctx.layout = layout

        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        return (
            None,  # q
            None,  # k
            None,  # v
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # deterministic
            None,  # sm_margin
            None,  # return_lse
            None,  # layout
            None,  # config
        )


def fav3_sage_wrapper_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    inference_mode: bool = True,
    layout: str = "bshd",
    config: Optional[dict] = None,
):
    """
    SageAttention v1 high-precision entry point.

    This function accepts high-precision (BF16/FP32) tensors and internally
    quantizes them to Int8/BF16 for computation. The output and gradients remain
    in high precision (FP32 for output, input dtype for gradients).

    This API is designed for seamless integration with existing training code
    that uses BF16/FP32 tensors, providing FP8 acceleration without requiring
    manual quantization.

    Args:
        q: Query tensor [batch, seqlen, num_q_heads, head_dim] (BF16/FP32)
        k: Key tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP32)
        v: Value tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP32)
        k_mean: Mean of k to conduct k-smoothing
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        qv: Extra query-value tensor (not yet supported)
        window_size: Sliding window attention size (left, right)
        attention_chunk: Chunking parameter (0 or 1 only)
        softcap: Softcapping value (not yet supported)
        num_splits: Number of splits for parallel processing (not yet supported)
        pack_gqa: GQA packing flag (not yet supported)
        deterministic: Whether to use deterministic backward (not yet supported)
        sm_margin: SM margin parameter (not yet supported)
        inference_mode: do not return softmax_lse
        layout: bshd or bhsd layout for the inputs
        config: Optional kernel configuration dict with keys BLOCK_M, BLOCK_N,
                waves_per_eu, PRE_LOAD_V, num_stages, num_warps

    Returns:
        out: Output tensor [batch, seqlen, num_q_heads, head_dim] or [batch, num_q_heads, seqlen, head_dim] (FP32)

    Note:
        - Supports GQA/MQA (num_q_heads != num_kv_heads)
        - Automatically handles grouped quantization for GQA/MQA queries
        - backward is not yet supported
        - qv, softcap, num_splits, pack_gqa, and sm_margin are not yet supported in FP8 mode
    """

    # Check that inputs are high precision
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"sage_attn_v1_func expects high-precision inputs (fp16/bf16/fp32), got q.dtype={q.dtype}. "
        f"If you already have Int8 tensors, use sage_attn_v1_func() with q_descale/k_descale parameters instead."
    )
    assert k.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"sage_attn_v1_func expects high-precision inputs (fp16/bf16/fp32), got k.dtype={k.dtype}. "
        f"If you already have Int8 tensors, use sage_attn_v1_func() with q_descale/k_descale parameters instead."
    )
    assert v.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], f"sage_attn_v1_func expects high-precision inputs (fp16/bf16/fp32), got v.dtype={v.dtype}. "

    if sm_margin != 0:
        raise NotImplementedError(
            "sm_margin != 0 not supported in Sage Attention v1 API"
        )

    return_lse = not inference_mode

    return _FAv3SageWrapperFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        sm_margin,
        return_lse,
        layout,
        config,
    )


def fav3_sage_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    sm_margin: int = 0,
    return_lse: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
):
    """
    SageAttention v1.

    Args:
        q: Query tensor [batch, seqlen, num_q_heads, head_dim] (int8)
        k: Key tensor [batch, seqlen, num_kv_heads, head_dim] (int8)
        v: Value tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP16)
        k_mean: Mean of k to conduct k-smoothing
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        qv: Extra query-value tensor (not yet supported)
        window_size: Sliding window attention size (left, right)
        attention_chunk: Chunking parameter (0 or 1 only)
        softcap: Softcapping value (not yet supported)
        num_splits: Number of splits for parallel processing (not yet supported)
        pack_gqa: GQA packing flag (not yet supported)
        deterministic: Whether to use deterministic backward (not yet supported)
        sm_margin: SM margin parameter (not yet supported)
        inference_model: do not return softmax_lse
        layout: bshd or bhsd layout for the inputs
        config: Optional kernel configuration dict with keys BLOCK_M, BLOCK_N,
                waves_per_eu, PRE_LOAD_V, num_stages, num_warps

    Returns:
        out: Output tensor [batch, seqlen, num_q_heads, head_dim] or [batch, num_q_heads, seqlen, head_dim] (FP32)
    """

    # --- 1. Layout & Dimension Mapping ---
    # bshd: [0,1,2,3], bhsd: [0,2,1,3]
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    batch, seqlen_q, nheads_q, head_size_qk = map_dims(q.shape, bshd_map)
    _, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd_map)
    _, seqlen_v, nheads_v, head_size_v = map_dims(v.shape, bshd_map)

    # --- 2. Feature & Input Validation ---
    if attention_chunk not in (0, 1) or softcap != 0.0 or sm_margin != 0:
        raise NotImplementedError(
            "Feature (chunking/softcap/sm_margin) not supported in this API."
        )

    assert q.dtype == torch.int8 and k.dtype == torch.int8, "Q and K must be int8"
    assert seqlen_k == seqlen_v, f"K/V seqlen mismatch: {seqlen_k} vs {seqlen_v}"
    assert nheads_k == nheads_v, f"K/V head mismatch: {nheads_k} vs {nheads_v}"
    assert (
        nheads_q % nheads_k == 0
    ), f"GQA/MQA error: {nheads_q} not divisible by {nheads_k}"

    # --- 3. Configuration & Descale Setup ---
    if config is None:
        config = get_sage_fwd_configs()

    BLKQ, BLKK = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
    num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

    assert q_descale.shape == (batch, nheads_q, num_q_blocks)
    assert k_descale.shape == (batch, nheads_k, num_k_blocks)

    # --- 4. Output Allocation ---
    out_dtype = torch.bfloat16
    if layout == "thd":
        out = torch.zeros(
            (q.shape[0], q.shape[1], v.shape[-1]), dtype=out_dtype, device=q.device
        )
        softmax_lse = (
            torch.zeros((nheads_q, q.shape[0]), device=q.device, dtype=torch.float32)
            if return_lse
            else None
        )
    else:
        out_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
        out = torch.zeros(out_shape, dtype=out_dtype, device=q.device)
        softmax_lse = (
            torch.zeros(
                (batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32
            )
            if return_lse
            else None
        )

    # --- 5. Stride Extraction ---
    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd_map)
    stride_vb, stride_vn, stride_vh, stride_vd = map_dims(v.stride(), bshd_map)
    stride_ob, stride_om, stride_oh, stride_od = map_dims(out.stride(), bshd_map)

    stride_lse_z, stride_lse_h, stride_lse_m = (
        softmax_lse.stride() if return_lse else (0, 0, 0)
    )
    stride_qsz, stride_qsh, stride_qsblk = q_descale.stride()
    stride_ksz, stride_ksh, stride_ksblk = k_descale.stride()
    stride_vsz, stride_vsh, _ = v_descale.stride()

    # --- 6. Padding & Metadata ---
    padded_d_model_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_model_v = max(16, 1 << (head_size_v - 1).bit_length())

    window_size_left, window_size_right = int(window_size[0]), int(window_size[1])
    use_sliding_window = window_size_left != -1 or window_size_right != -1

    # --- 7. Kernel Launch ---
    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]) * nheads_q * batch, )

    sage_fwd[grid](
        q,
        k,
        v,
        None,
        q_descale,
        k_descale,
        v_descale,
        stride_qsz,
        stride_qsh,
        stride_qsblk,
        stride_ksz,
        stride_ksh,
        stride_ksblk,
        stride_vsz,
        stride_vsh,
        softmax_lse,
        out,
        None,
        None,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        0,
        0,
        0,
        0,  # stride_bz, stride_bh, stride_bm, stride_bn
        0,
        0,  # stride_az, stride_ah
        0,
        0,
        0,
        0,  # stride_sz, stride_sh, stride_sm, stride_sn
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        None,
        None,
        None,
        None,
        dropout_p=0.0,
        philox_seed=None,
        philox_offset_base=None,
        RETURN_LSE=return_lse,
        BATCH=batch,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        SM_SCALE=softmax_scale,
        IS_CAUSAL=causal,
        USE_SLIDING_WINDOW=use_sliding_window,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        IS_VARLEN=False,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False,
        USE_ALIBI=False,
        ENABLE_DROPOUT=False,
        USE_EXP2=True,
        RETURN_SCORES=False,
        USE_SEQUSED=False,
        **config,
    )

    if return_lse:
        return out, softmax_lse
    else:
        return out, None
