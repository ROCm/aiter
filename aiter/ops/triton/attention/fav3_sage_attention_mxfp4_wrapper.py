# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import map_dims
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
    sage_fwd_mxfp4,
    sage_quant_mxfp4,
)


def get_sage_fwd_configs_mxfp4():
    """Returns tuned config for MXFP4 on supported architectures."""
    arch = arch_info.get_arch()
    # MXFP4 is primarily targeted at gfx950
    if arch != "gfx950":
        raise RuntimeError(f"MXFP4 is not supported on {arch}")
    return {
        "BLOCK_M": 256,
        "BLOCK_N": 128,
        "waves_per_eu": 2,
        "PRE_LOAD_V": False,
        "num_stages": 3,
        "num_warps": 8,
    }


class _FAv3SageMXFP4WrapperFunc(torch.autograd.Function):
    """
    Sage Attention v2 MXFP4 wrapper maintaining high-precision I/O.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        layout: str = "bshd",
        q_smooth: bool = False,
        hadamard_rotation: bool = True,
        config: Optional[dict] = None,
        R: torch.Tensor = None,
    ):
        bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        bhsd_map = [0, 2, 1, 3] if layout == "bshd" else [0, 1, 2, 3]
        batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd_map)
        _, seqlen_k, num_kv_heads, _ = map_dims(k.shape, bshd_map)

        if config is None:
            config = get_sage_fwd_configs_mxfp4()

        (
            q_quantized,
            q_descale,
            k_quantized,
            k_descale,
            v_quantized,
            v_descale,
            delta_s,
        ) = sage_quant_mxfp4(
            q,
            k,
            v,
            hadamard_rotation=hadamard_rotation,
            R=R,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_R=128 if R is None else R.shape[-1],
            q_smoothing=q_smooth,
            layout=layout,
        )

        qd_mapped = map_dims(q_descale.shape, bhsd_map)
        kd_mapped = map_dims(k_descale.shape, bhsd_map)

        expected_q_ds = (batch, num_q_heads, seqlen_q, head_dim // 32)
        expected_k_ds = (batch, num_kv_heads, seqlen_k, head_dim // 32)

        assert tuple(qd_mapped) == expected_q_ds, "q_descale mismatch"
        assert tuple(kd_mapped) == expected_k_ds, "k_descale mismatch"

        out = fav3_sage_mxfp4_func(
            q=q_quantized,
            k=k_quantized,
            v=v_quantized,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            bias=delta_s,
            causal=causal,
            layout=layout,
            config=config,
        )

        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        # Backward remains unimplemented
        assert False, "backward not implemented"
        return (None,) * 9


def fav3_sage_mxfp4_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    layout: str = "bshd",
    q_smooth: bool = False,
    hadamard_rotation: bool = False,
    config: Optional[dict] = None,
    R: torch.Tensor = None,
):
    """High-precision entry point for MXFP4 SageAttention."""
    for tensor, name in zip([q, k, v], ["q", "k", "v"]):
        assert tensor.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], f"Expected high-precision for {name}, got {tensor.dtype}"

    return _FAv3SageMXFP4WrapperFunc.apply(
        q, k, v, causal, layout, q_smooth, hadamard_rotation, config, R
    )


def fav3_sage_mxfp4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    bias: torch.Tensor = None,
    causal: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
):
    """Direct MXFP4 kernel execution with unused parameters removed."""
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_size_qk = map_dims(q.shape, bshd_map)

    # MXFP4 head size adjustment (elements per byte)
    head_size_qk *= 2
    _, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd_map)
    _, _, _, head_size_v = map_dims(v.shape, bshd_map)

    # Validations
    assert q.dtype == torch.uint8 and k.dtype == torch.uint8, "MXFP4 Q/K must be uint8"
    assert nheads_q % nheads_k == 0, "GQA/MQA ratio mismatch"
    assert layout in ["bhsd", "bshd"], "Only bhsd and bshd supported for now."

    if config is None:
        config = get_sage_fwd_configs_mxfp4()

    # Allocation
    out = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
        dtype=torch.bfloat16,
        device=q.device,
    )

    # Tensor Strides
    stride_qb, stride_qm, stride_qh, _ = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, _ = map_dims(k.stride(), bshd_map)
    stride_vb, stride_vn, stride_vh, _ = map_dims(v.stride(), bshd_map)
    stride_ob, stride_om, stride_oh, _ = map_dims(out.stride(), bshd_map)

    # delta s is the bias
    if bias is not None:
        USE_BIAS = True
        stride_bz, stride_bm, stride_bh, stride_bn = map_dims(bias.stride(), bshd_map)
    else:
        USE_BIAS = False
        stride_bz, stride_bm, stride_bh, stride_bn = 0, 0, 0, 0

    # Descale Strides
    stride_qsz, stride_qsm, stride_qsh, _ = map_dims(q_descale.stride(), bshd_map)
    stride_ksz, stride_ksn, stride_ksh, _ = map_dims(k_descale.stride(), bshd_map)
    stride_vsz, stride_vsh, _ = v_descale.stride()

    # Kernel padding logic
    padded_d_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_v = max(16, 1 << (head_size_v - 1).bit_length())

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)

    sage_fwd_mxfp4[grid](
        Q=q,
        K=k,
        V=v,
        bias=bias,
        Q_Descale=q_descale,
        K_Descale=k_descale,
        V_Descale=v_descale,
        stride_qsz=stride_qsz,
        stride_qsh=stride_qsh,
        stride_qsm=stride_qsm,
        stride_ksz=stride_ksz,
        stride_ksh=stride_ksh,
        stride_ksn=stride_ksn,
        stride_vsz=stride_vsz,
        stride_vsh=stride_vsh,
        Out=out,
        stride_qz=stride_qb,
        stride_qh=stride_qh,
        stride_qm=stride_qm,
        stride_kz=stride_kb,
        stride_kh=stride_kh,
        stride_kn=stride_kn,
        stride_vz=stride_vb,
        stride_vh=stride_vh,
        stride_vk=stride_vn,
        stride_oz=stride_ob,
        stride_oh=stride_oh,
        stride_om=stride_om,
        stride_bz=stride_bz,
        stride_bh=stride_bh,
        stride_bm=stride_bm,
        stride_bn=stride_bn,  # Bias strides
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        Q_DTYPE_STR="e2m1",
        K_DTYPE_STR="e2m1",
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        IS_VARLEN=False,
        IS_CAUSAL=causal,
        BLOCK_DMODEL_QK=padded_d_qk,
        BLOCK_DMODEL_V=padded_d_v,
        USE_BIAS=USE_BIAS,
        **config,
    )

    return out
