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
    else:
        # Default tuned config for MI300X (gfx942)
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "waves_per_eu": 2,
            "PRE_LOAD_V": False,
            "num_stages": 2,
            "num_warps": 8,
        }

class _FAv3SageWrapperFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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

        # 2. Validation
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0 or sm_margin != 0:
            raise NotImplementedError("softcap/sm_margin not supported in Int8 API")

        # 3. Quantization (Int8 Q/K, FP16/BF16 V)
        softmax_scale = (head_dim**-0.5)
        fp8_dtype = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_dtype).max

        q_quantized, q_descale, k_quantized, k_descale, v_quantized, v_descale = (
            sage_quant(
                q, k, v,
                fp8_dtype,
                fp8_max,
                sm_scale=softmax_scale,
                BLKQ=BLKQ,
                BLKK=BLKK,
                layout=layout,
            )
        )

        # 4. Execution
        out, softmax_lse = fav3_sage_func(
            q_quantized, k_quantized, v_quantized,
            q_descale, k_descale, v_descale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
            return_lse=return_lse,
            layout=layout,
            config=config,
        )

        if return_lse:
            ctx.save_for_backward(q, k, v, out, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.layout = layout

        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        return (None,) * 12 # No backward support

def fav3_sage_wrapper_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
    return _FAv3SageWrapperFunc.apply(
        q, k, v, causal, window_size, attention_chunk, 
        softcap, deterministic, sm_margin, not inference_mode, 
        layout, config
    )

def fav3_sage_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    config: dict,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    sm_margin: int = 0,
    return_lse: bool = False,
    layout: str = "bshd"
):
    
    assert config is not None, "Specific config has been used for descale computations, and cannot be None!"
    
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_size_qk = map_dims(q.shape, bshd_map)
    _, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd_map)
    _, _, _, head_size_v = map_dims(v.shape, bshd_map)

    

    # Output Allocation
    out = torch.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=torch.bfloat16, device=q.device)
    softmax_lse = torch.zeros((batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32) if return_lse else None

    # Stride Extraction
    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd_map)
    stride_vb, stride_vn, stride_vh, stride_vd = map_dims(v.stride(), bshd_map)
    stride_ob, stride_om, stride_oh, stride_od = map_dims(out.stride(), bshd_map)
    
    stride_qsz, stride_qsh, stride_qsblk = q_descale.stride()
    stride_ksz, stride_ksh, stride_ksblk = k_descale.stride()
    stride_vsz, stride_vsh, _ = v_descale.stride()
    stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride() if return_lse else (0, 0, 0)

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)

    sage_fwd[grid](
        q, k, v, None, q_descale, k_descale, v_descale,
        stride_qsz, stride_qsh, stride_qsblk,
        stride_ksz, stride_ksh, stride_ksblk,
        stride_vsz, stride_vsh,
        softmax_lse, out, None, None,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        stride_lse_z, stride_lse_h, stride_lse_m,
        None, None, None, None,
        dropout_p=0.0, philox_seed=None, philox_offset_base=None,
        RETURN_LSE=return_lse,
        HQ=nheads_q, HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q, MAX_SEQLENS_K=seqlen_k,
        IS_CAUSAL=causal,
        USE_SLIDING_WINDOW=(window_size[0] != -1),
        WINDOW_SIZE_LEFT=int(window_size[0]),
        WINDOW_SIZE_RIGHT=int(window_size[1]),
        BLOCK_DMODEL_QK=max(16, 1 << (head_size_qk - 1).bit_length()),
        BLOCK_DMODEL_V=max(16, 1 << (head_size_v - 1).bit_length()),
        **config,
    )

    return out, softmax_lse