# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple
import torch
import aiter

from aiter.ops.triton._triton_kernels.sage_attn_triton_amd import (
    fav3_sage,
    get_sage_fwd_configs,
)

from aiter.ops.triton._triton_kernels.sage_attn_triton_amd.utils import (
    map_dims,
)

from aiter.ops.triton._triton_kernels.sage_attn_triton_amd import sage_quant, sage_quant_v2


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
        k_mean: torch.Tensor | None,
        softmax_scale: float | None,
        causal: bool,
        qv: Optional[torch.Tensor],
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        num_splits: int,
        pack_gqa: Optional[bool],
        deterministic: bool,
        sm_margin: int,
        return_lse: bool = True,
        layout: str = "bshd",
        config: Optional[dict] = None,
        sage_version: fav3_sage.Sage_version = fav3_sage.Sage_version.V1
    ):
        bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd)
        _, seqlen_k, num_kv_heads, _ = map_dims(k.shape, bshd)
        if sage_version == fav3_sage.Sage_version.V2:
            head_dim *= 2

        # Quantize K, V to int8, and convert v to float16

        # Use provided config or get default config
        if config is None:
            config = get_sage_fwd_configs()
        # assert len(config) == 1, f"Number of best config is expected to be 1, got {len(config)}"
        # config = config[0].all_kwargs()
        BLKQ = config["BLOCK_M"]
        BLKK = config["BLOCK_N"]

        softmax_scale = head_dim**-0.5
        ## following quantization already considered softmax scale and RCP_LN2
        fp8_dtype = aiter.dtypes.fp8
        FP8_MAX = torch.finfo(fp8_dtype).max

        if sage_version == fav3_sage.Sage_version.V1:
            q_quantized, q_descale, k_quantized, k_descale, v_quantized, v_descale = sage_quant(
                q,
                k,
                v,
                fp8_dtype,
                FP8_MAX,
                sm_scale=softmax_scale,
                BLKQ=BLKQ,
                BLKK=BLKK,
                layout=layout,
            )
        elif sage_version == fav3_sage.Sage_version.V2:
            q_quantized, q_descale, k_quantized, k_descale, v_quantized, v_descale = sage_quant_v2(
                q,
                k,
                v,
                fp8_dtype,
                FP8_MAX,
                sm_scale=softmax_scale,
                BLKQ=BLKQ,
                BLKK=BLKK,
                layout=layout,
            )

        # For GQA/MQA: quantize query with grouped scaling
        # group_size = (
        #    num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None
        # )

        # Verify descale shapes for GQA/MQA
        num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
        num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

        if sage_version == fav3_sage.Sage_version.V1:
            assert q_descale.shape == (
                batch,
                num_q_heads,
                num_q_blocks,
            ), f"q_descale shape {q_descale.shape} != expected {(batch, num_q_heads, num_q_blocks)}"
            assert k_descale.shape == (
                batch,
                num_kv_heads,
                num_k_blocks,
            ), f"k_descale shape {k_descale.shape} != expected {(batch, num_kv_heads, num_k_blocks)}"
        elif sage_version == fav3_sage.Sage_version.V2:
            assert map_dims(q_descale.shape, bshd) == [
                batch,
                seqlen_q,
                num_q_heads,
                head_dim // 32
            ], f"q_descale shape {map_dims(q_descale.shape, bshd)} != expected {(batch, seqlen_q, num_q_heads, head_dim // 32)}"
            assert map_dims(k_descale.shape, bshd) == [
                batch,
                seqlen_k,
                num_kv_heads,
                head_dim // 32
            ], f"k_descale shape {map_dims(k_descale.shape, bshd)} != expected {(batch, seqlen_k, num_kv_heads, head_dim // 32)}"


        # Validate unsupported features
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0:
            raise NotImplementedError(
                "softcap not implemented in FP8 high-precision API"
            )
        if sm_margin != 0:
            raise NotImplementedError(
                "sm_margin != 0 not supported in FP8 high-precision API"
            )

        if q.requires_grad or k.requires_grad or v.requires_grad:
            assert (
                return_lse
            ), f"in train mode, return_lse is expected to be True, got {return_lse}"

        # Call flash attention forward
        out, softmax_lse = fav3_sage.fwd(
            q_quantized,
            k_quantized,
            v_quantized,
            None,
            None,
            None,
            None,  # k_new, v_new, qv, out
            None,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
            None,
            None,
            None,
            None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k
            None,
            None,
            None,  # rotary_cos, rotary_sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,  # v_descale
            FP8_MAX,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,
            1,
            None,
            sm_margin,  # scheduler_metadata, num_splits, pack_gqa, sm_margin
            return_lse,
            layout,
            config,
            sage_version=sage_version
        )

        if not return_lse:
            return out

        # Save tensors needed for backward
        ctx.save_for_backward(
            q_quantized, k_quantized, v_quantized, out, softmax_lse, q_descale, k_descale
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
            None,  # qv
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # num_splits
            None,  # pack_gqa
            None,  # deterministic
            None,  # sm_margin
        )


def fav3_sage_wrapper_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_mean: torch.Tensor = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
    inference_mode: bool = True,
    layout: str = "bshd",
    config: Optional[dict] = None,
    sage_version: fav3_sage.Sage_version = fav3_sage.Sage_version.V1
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

    if qv is not None:
        raise NotImplementedError("qv not supported in Sage Attention v1 API")
    if softcap != 0.0:
        raise NotImplementedError("softcap not supported in Sage Attention v1 API")
    if num_splits != 1:
        raise NotImplementedError(
            "num_splits != 1 not supported in Sage Attention v1 API"
        )
    if pack_gqa is not None:
        raise NotImplementedError("pack_gqa not supported in Sage Attention v1 API")
    if sm_margin != 0:
        raise NotImplementedError(
            "sm_margin != 0 not supported in Sage Attention v1 API"
        )

    return_lse = not inference_mode
    return _FAv3SageWrapperFunc.apply(
        q,
        k,
        v,
        k_mean,
        softmax_scale,
        causal,
        qv,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_lse,
        layout,
        config,
        sage_version=sage_version
    )


def fav3_sage_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    FP8_MAX: float = 240.0,
    k_mean: torch.Tensor = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
    inference_mode: bool = True,
    layout: str = "bshd",
    config: Optional[dict] = None,
    sage_version: fav3_sage.Sage_version = fav3_sage.Sage_version.V1
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

    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd)

    if sage_version == fav3_sage.Sage_version.V2:
        head_dim *= 2

    _, seqlen_k, num_kv_heads, _ = map_dims(k.shape, bshd)
    # Quantize K, V to int8, and convert v to float16

    # Use provided config or get default config
    if config is None:
        config = get_sage_fwd_configs()
    # assert len(config) == 1, f"Number of best config is expected to be 1, got {len(config)}"
    # config = config[0].all_kwargs()
    BLKQ = config["BLOCK_M"]
    BLKK = config["BLOCK_N"]

    if sage_version == fav3_sage.Sage_version.V1:
        assert q.dtype == torch.int8, f"expected dtype of q to be int8, got {q.dtype}"
        assert k.dtype == torch.int8, f"expected dtype of k to be int8, got {k.dtype}"
    elif sage_version == fav3_sage.Sage_version.V2:
        assert q.dtype == torch.uint8, f"expected dtype of q to be uint8, got {q.dtype}"
        assert k.dtype == torch.uint8, f"expected dtype of k to be uint8, got {k.dtype}"


    # Verify descale shapes for GQA/MQA
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
    num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

    if sage_version == fav3_sage.Sage_version.V1:
        assert q_descale.shape == (
            batch,
            num_q_heads,
            num_q_blocks,
        ), f"q_descale shape {q_descale.shape} != expected {(batch, num_q_heads, num_q_blocks)}"
        assert k_descale.shape == (
            batch,
            num_kv_heads,
            num_k_blocks,
        ), f"k_descale shape {k_descale.shape} != expected {(batch, num_kv_heads, num_k_blocks)}"
    elif sage_version == fav3_sage.Sage_version.V2:
        assert map_dims(q_descale.shape, bshd) == [
            batch,
            seqlen_q,
            num_q_heads,
            head_dim // 32
        ], f"q_descale shape {map_dims(q_descale.shape, bshd)} != expected {(batch, seqlen_q, num_q_heads, head_dim // 32)}"
        assert map_dims(k_descale.shape, bshd) == [
            batch,
            seqlen_k,
            num_kv_heads,
            head_dim // 32
        ], f"k_descale shape {map_dims(k_descale.shape, bshd)} != expected {(batch, seqlen_k, num_kv_heads, head_dim // 32)}"


    # Validate unsupported features
    if attention_chunk not in (0, 1):
        raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
    if softcap != 0.0:
        raise NotImplementedError("softcap not implemented in FP8 high-precision API")
    if sm_margin != 0:
        raise NotImplementedError(
            "sm_margin != 0 not supported in FP8 high-precision API"
        )

    return_lse = not inference_mode
    # Call flash attention forward
    out, _ = fav3_sage.fwd(
        q,
        k,
        v,
        None,
        None,
        None,
        None,  # k_new, v_new, qv, out
        None,
        None,
        None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
        None,
        None,
        None,
        None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
        None,
        None,
        None,  # page_table, kv_batch_idx, leftpad_k
        None,
        None,
        None,  # rotary_cos, rotary_sin, seqlens_rotary
        q_descale,
        k_descale,
        v_descale,  # v_descale
        FP8_MAX,
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        attention_chunk,
        softcap,
        False,  # rotary_interleaved
        None,
        1,
        None,
        sm_margin,  # scheduler_metadata, num_splits, pack_gqa, sm_margin
        return_lse,
        layout,
        config,
        sage_version=sage_version
    )

    return out
