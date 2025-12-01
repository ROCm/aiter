# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton._triton_kernels.attn_qk_int8_per_block import (
    _attn_fwd,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


@functools.lru_cache(maxsize=1024)
def _get_config(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-ATTN-QK-INT8-PER-BLOCK.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = "default"

    # Select config based on sequence length
    if seq_len < 8192 and "small" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small"]
    elif seq_len <= 32768:
        # Check for M-specific medium configs based on BLOCK_SIZE_M
        BLK_M = triton.next_power_of_2(min(seq_len // 64, 128))
        if BLK_M == 64 and "medium_M64" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M64"]
        elif BLK_M == 128 and "medium_M128" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M128"]
    elif seq_len <= 65536 and "large" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["large"]
    elif seq_len > 65536:
        BLK_M = triton.next_power_of_2(min(seq_len // 512, 256))
        if f"xlarge_M{BLK_M}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"xlarge_M{BLK_M}"]
        elif "xlarge" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["xlarge"]

    return _get_config._config_dict[key]["any"]


def attn_qk_int8_per_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    tensor_layout: str = "HND",
    attn_mask: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float16,
    return_lse: bool = False,
    config: Optional[dict] = None,
):
    """
    Computes attention with INT8 quantized Q and K matrices using per-block scales.

    Args:
        q (torch.Tensor): INT8 query tensor with shape (B, H, qo_len, head_dim) for HND layout
            or (B, qo_len, H, head_dim) for NHD layout.
        k (torch.Tensor): INT8 key tensor with shape (B, H, kv_len, head_dim) for HND layout
            or (B, kv_len, H, head_dim) for NHD layout.
        v (torch.Tensor): FP16 value tensor with shape (B, H, kv_len, head_dim) for HND layout
            or (B, kv_len, H, head_dim) for NHD layout.
        q_scale (torch.Tensor): Per-block scale for Q with shape (B, H, qo_len // BLOCK_M, 1).
        k_scale (torch.Tensor): Per-block scale for K with shape (B, H, kv_len // BLOCK_N, 1).
        tensor_layout (str): Tensor layout, either "HND" or "NHD". Default: "HND".
        attn_mask (Optional[torch.Tensor]): Optional attention mask with shape (B, H, qo_len, kv_len).
        output_dtype (Optional[torch.dtype]): Output datatype (FP16 or BF16). Default: torch.float16.
        return_lse (bool): Whether to return log-sum-exp values. Default: False.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N, num_warps, etc.).

    Returns:
        tuple: (output tensor, lse tensor) if return_lse=True, else (output tensor, empty tensor).
    """
    _LOGGER.info(
        f"ATTN_QK_INT8_PER_BLOCK: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"q_scale={tuple(q_scale.shape)} k_scale={tuple(k_scale.shape)} layout={tensor_layout}"
    )

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if attn_mask is not None:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = (
            attn_mask.stride(0),
            attn_mask.stride(1),
            attn_mask.stride(2),
            attn_mask.stride(3),
        )
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device="cpu")

    if config is None:
        config = _get_config(b, h_qo, qo_len, head_dim)

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]
    stage = config.get("stage", 1)

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        o,
        attn_mask,
        lse,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        stride_bz_mask,
        stride_h_mask,
        stride_m_mask,
        stride_n_mask,
        qo_len,
        kv_len,
        h_qo,
        num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        RETURN_LSE=return_lse,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 3),
        waves_per_eu=config.get("waves_per_eu", 2),
    )

    return o, lse

