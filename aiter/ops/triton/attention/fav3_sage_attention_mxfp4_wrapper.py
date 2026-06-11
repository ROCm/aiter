# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple
import torch
import triton
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import map_dims
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
    sage_fwd_mxfp4,
)
import math
import torch.nn.functional as F
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant_mxfp4,
    sage_quant_lloymax,
    sage_quant_lloymax_packed,
)
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_lloymax import (
    sage_fwd_lloymax,
)


import aiter


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
        BLOCK_R: int = 128,
        block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        bhsd_map = [0, 2, 1, 3] if layout == "bshd" else [0, 1, 2, 3]
        batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd_map)
        _, seqlen_k, num_kv_heads, _ = map_dims(k.shape, bshd_map)

        if config is None:
            config = get_sage_fwd_configs_mxfp4()

        FP8_TYPE = aiter.dtypes.fp8
        FP8_MAX = torch.finfo(FP8_TYPE).max

        assert hadamard_rotation, "hadamard_rotation=False not supported at the moment"
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
            FP8_TYPE,
            FP8_MAX,
            BLKQ=config["BLOCK_M"],
            BLKK=64,
            layout=layout,
            R=R,
            BLOCK_R=BLOCK_R,
            q_smoothing=q_smooth,
        )
        # TODO: fused quant has perf downgrade
        # fused_sage_quant_mxfp4(
        #     q,
        #     k,
        #     v,
        #     hadamard_rotation=hadamard_rotation,
        #     R=R,
        #     BLOCK_M=config["BLOCK_M"],
        #     BLOCK_R=BLOCK_R if R is None else R.shape[-1],
        #     q_smoothing=q_smooth,
        #     layout=layout,
        # )

        qd_mapped = map_dims(q_descale.shape, bhsd_map)
        kd_mapped = map_dims(k_descale.shape, bhsd_map)

        expected_q_ds = (batch, num_q_heads, seqlen_q, head_dim // 32)
        expected_k_ds = (batch, num_kv_heads, seqlen_k, head_dim // 32)

        assert tuple(qd_mapped) == expected_q_ds, "q_descale mismatch"
        assert tuple(kd_mapped) == expected_k_ds, "k_descale mismatch"

        if block_lut is not None:
            kv_block_indices, lut_start, lut_count = block_lut
            use_block_sparse = True
            if causal:
                raise NotImplementedError(
                    "The Triton block-sparse attention path selected by block_lut "
                    "does not support causal masking."
                    "require causal=False."
                )
        else:
            kv_block_indices = lut_start = lut_count = None
            use_block_sparse = False

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
            kv_block_indices=kv_block_indices,
            lut_start=lut_start,
            lut_count=lut_count,
            use_block_sparse=use_block_sparse,
        )

        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        # Backward remains unimplemented
        assert False, "backward not implemented"
        return (None,) * 10


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
    BLOCK_R: int = 128,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """High-precision entry point for MXFP4 SageAttention."""
    for tensor, name in zip([q, k, v], ["q", "k", "v"]):
        assert tensor.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], f"Expected high-precision for {name}, got {tensor.dtype}"

    return _FAv3SageMXFP4WrapperFunc.apply(
        q,
        k,
        v,
        causal,
        layout,
        q_smooth,
        hadamard_rotation,
        config,
        R,
        BLOCK_R,
        block_lut,
    )


def fav3_sage_lloymax_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    layout: str = "bshd",
    R: torch.Tensor = None,
    BLOCK_R: int = 128,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """
    Sage attention with Lloyd-Max Q/K quantization.

    Quantization: Hadamard rotation → per-vector norm separation → Lloyd-Max
    nearest-centroid lookup (pre-generated codebook for head_dim=128, 4-bit).
    Uses the sage_fwd_lloymax Triton kernel which replaces tl.dot_scaled
    (hardwired e2m1 on MI355) with codebook lookup + BF16 matmul.

    Accuracy (vs Sage v2 MXFP4 current):
      +16.3% reduction in Q/K quantization error
      +17.2% reduction in end-to-end attention output error (measured)
      93.1% of vectors improved

    Supports:
      • Dense causal / non-causal attention
      • Block-sparse (Sparge) via block_lut

    Args:
      block_lut: (kv_block_indices, lut_start, lut_count) for Sparge sparse attention.
    """
    for tensor, name in zip([q, k, v], ["q", "k", "v"]):
        assert tensor.dtype in [torch.float16, torch.bfloat16, torch.float32], (
            f"Expected high-precision for {name}, got {tensor.dtype}"
        )

    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX  = torch.finfo(FP8_TYPE).max
    config   = get_sage_fwd_configs_mxfp4()

    # Quantize Q/K with Lloyd-Max (packed uint8 + float16 norms)
    q_packed, q_norms, k_packed, k_norms, v_fp8, v_scale, _ = sage_quant_lloymax_packed(
        q, k, v,
        FP8_TYPE=FP8_TYPE,
        FP8_MAX=FP8_MAX,
        BLKQ=config["BLOCK_M"],
        BLKK=64,
        layout=layout,
        R=R,
        BLOCK_R=BLOCK_R,
    )

    # Load codebook (cached by get_codebook)
    from aiter.ops.triton.attention.turboquant.codebook import get_codebook
    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    head_dim = map_dims(q.shape, bshd)[-1]
    codebook = get_codebook(head_dim, 4, device=q.device).float().contiguous()

    use_block_sparse = block_lut is not None
    if use_block_sparse:
        kv_block_indices, lut_start, lut_count = block_lut
    else:
        kv_block_indices = lut_start = lut_count = None

    return fav3_sage_lloymax_func(
        q_packed, q_norms,
        k_packed, k_norms,
        v_fp8, v_scale,
        codebook,
        causal=causal,
        layout=layout,
        config=config,
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
        use_block_sparse=use_block_sparse,
    )


def fav3_sage_lloymax_func(
    q_packed: torch.Tensor,
    q_norms: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v: torch.Tensor,
    v_descale: torch.Tensor,
    codebook: torch.Tensor,
    bias: torch.Tensor = None,
    causal: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
    kv_block_indices: Optional[torch.Tensor] = None,
    lut_start: Optional[torch.Tensor] = None,
    lut_count: Optional[torch.Tensor] = None,
    use_block_sparse: bool = False,
):
    """
    Launch the sage_fwd_lloymax Triton kernel.

    Inputs:
      q_packed / k_packed : uint8 (B, S, H, D//2) — packed 4-bit Lloyd-Max indices
      q_norms  / k_norms  : float16 (B*H, S)      — pre-scaled per-vector norms
                            (stride-1 along S for fast sequence access in kernel)
      v        : fp8 (B, S_k, H_kv, D_v)
      v_descale: float32 (B, H_kv, D_v)
      codebook : float32 (16,) — Lloyd-Max centroids
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_dim_half = map_dims(q_packed.shape, bshd_map)
    _, seqlen_k, nheads_k, head_dim_v        = map_dims(v.shape, bshd_map)

    assert q_packed.dtype == torch.uint8, "q_packed must be uint8"
    assert k_packed.dtype == torch.uint8, "k_packed must be uint8"
    assert nheads_q % nheads_k == 0,      "GQA ratio mismatch"
    assert layout in ("bshd", "bhsd")
    # q_norms shape: (B*H_q, S_q) contiguous; k_norms: (B*H_k, S_k) contiguous
    assert q_norms.shape == (batch * nheads_q, seqlen_q), \
        f"q_norms shape mismatch: {q_norms.shape} vs ({batch*nheads_q}, {seqlen_q})"
    assert k_norms.shape == (batch * nheads_k, seqlen_k), \
        f"k_norms shape mismatch: {k_norms.shape} vs ({batch*nheads_k}, {seqlen_k})"

    if config is None:
        config = get_sage_fwd_configs_mxfp4()

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]

    # Output tensor (same spatial layout as Q, but V head dim)
    out_shape = list(q_packed.shape)
    out_shape[-1] = head_dim_v
    out = torch.zeros(out_shape, dtype=torch.bfloat16, device=q_packed.device)

    # Strides for packed Q/K (last dim = D//2, stride 1)
    stride_qz, stride_qm, stride_qh, _ = map_dims(q_packed.stride(), bshd_map)
    stride_kz, stride_kn, stride_kh, _ = map_dims(k_packed.stride(), bshd_map)
    # V / V_descale / Out strides
    stride_vz, stride_vn, stride_vh, _ = map_dims(v.stride(), bshd_map)
    stride_vsz, stride_vsh, _          = v_descale.stride()
    stride_oz, stride_om, stride_oh, _ = map_dims(out.stride(), bshd_map)

    USE_BIAS = bias is not None
    if USE_BIAS:
        stride_bz, stride_bh, stride_bm, stride_bn = bias.stride()
    else:
        bias = torch.empty(0, device=q_packed.device)
        stride_bz = stride_bh = stride_bm = stride_bn = 0

    if use_block_sparse:
        if any(x is None for x in (kv_block_indices, lut_start, lut_count)):
            raise ValueError("kv_block_indices, lut_start, lut_count required for block-sparse")
    else:
        kv_block_indices = lut_start = lut_count = torch.empty(0, device=q_packed.device)

    padded_dv = max(16, 1 << (head_dim_v - 1).bit_length())
    PADDED_HEAD_V = padded_dv != head_dim_v

    grid = (triton.cdiv(seqlen_q, BLOCK_M), nheads_q, batch)

    sage_fwd_lloymax[grid](
        q_packed, q_norms,
        k_packed, k_norms,
        v, v_descale,
        codebook,
        out,
        # Q packed strides
        stride_qz, stride_qh, stride_qm,
        # K packed strides
        stride_kz, stride_kh, stride_kn,
        # V strides
        stride_vz, stride_vh, stride_vn,
        # V_descale strides
        stride_vsz, stride_vsh,
        # Out strides
        stride_oz, stride_oh, stride_om,
        # Bias
        bias, stride_bz, stride_bh, stride_bm, stride_bn,
        # LUT
        kv_block_indices, lut_start, lut_count,
        # Sequence
        seqlen_q, seqlen_k,
        # GQA
        HQ=nheads_q, HK=nheads_k,
        # Head dims
        HALF_D=head_dim_half,
        BLOCK_DV=padded_dv,
        ACTUAL_DV=head_dim_v,
        # Tuning
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        PRE_LOAD_V=config.get("PRE_LOAD_V", False),
        # Flags
        IS_CAUSAL=causal,
        USE_BLOCK_SPARSE=use_block_sparse,
        USE_BIAS=USE_BIAS,
        PADDED_HEAD_V=PADDED_HEAD_V,
        num_warps=config.get("num_warps", 8),
        num_stages=config.get("num_stages", 3),
    )

    return out


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
    kv_block_indices: Optional[torch.Tensor] = None,
    lut_start: Optional[torch.Tensor] = None,
    lut_count: Optional[torch.Tensor] = None,
    use_block_sparse: bool = False,
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
        stride_bz, stride_bh, stride_bm, stride_bn = bias.stride()
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

    # Block sparse logic
    if use_block_sparse:
        if kv_block_indices is None or lut_start is None or lut_count is None:
            raise ValueError(
                "kv_block_indices, lut_start, and lut_count must be provided "
                "when use_block_sparse=True"
            )
        if causal:
            raise NotImplementedError(
                "The Triton block-sparse attention path selected by block_lut "
                "does not support causal masking."
                "require causal=False."
            )
    else:
        kv_block_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
        lut_start = torch.zeros(1, dtype=torch.int32, device=q.device)
        lut_count = torch.zeros(1, dtype=torch.int32, device=q.device)

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
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
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
        USE_BLOCK_SPARSE=use_block_sparse,
        **config,
    )

    return out
