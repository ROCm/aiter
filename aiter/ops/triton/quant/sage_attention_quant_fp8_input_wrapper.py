# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# sage_quant_mxfp4_fp8_input: same contract as sage_quant_mxfp4 but accepts
# fp8 q, k, and optionally fp8 v tensors. The fp8->fp32 widening happens
# inside the Triton kernels on the first tl.load, so no intermediate bf16
# tensor is written to HBM.
#
# Optional q_input_scale / k_input_scale apply per-tensor comms descale on-chip
# (x_bf16 = x_fp8 * scale) before Hadamard rotation and mxfp4 quantization.
#
# If v is fp8, v_scale must be provided (computed when v was quantized upstream)
# and sage_quant_v_kernel is skipped entirely — v passes through unchanged.
# If v is bf16/fp16, v_scale must be None and the existing quantization path runs.
#
# k-smoothing (subtract k.mean) cannot be done in fp8 on the host, so the
# mean is computed in fp32 and subtracted inside the kernel.

import torch
import triton
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import map_dims
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    sage_quant_v_kernel,
)
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant_fp8_input import (
    _rotate_quantize_qk_fp8_kernel,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import create_hadamard_matrix


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def sage_quant_mxfp4_fp8_input(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    R=None,
    BLOCK_R=128,
    q_input_scale=None,
    k_input_scale=None,
    v_scale=None,
):
    """
    Quantize fp8 q/k and bf16-or-fp8 v for mxfp4 sage attention.

    q, k must be fp8 (float8_e4m3fn or float8_e4m3fnuz).
    q_input_scale, k_input_scale: optional fp32 scalars for per-tensor comms
      descale (``x_bf16 = x_fp8 * scale``). Omit for direct fp8 casts.
    v can be:
      - bf16 or fp16: quantized to fp8 internally; v_scale must be None.
      - fp8: passed through unchanged; v_scale must be provided. Scalar
        per-tensor comms scale or [B, H, D] from sage_quant_mxfp4-style quant.

    Returns: q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s
      - q_fp4, k_fp4 : uint8 packed e2m1 fp4
      - q_scale, k_scale : uint8 e8m0 block scales  [*, D/32]
      - v_fp8  : fp8 v (quantized here if input was bf16, else passed through)
      - v_scale: fp32 per-(B,H,D) scale
      - delta_s: None (q_smoothing not yet supported for fp8 input path)
    """
    assert q.dtype in _FP8_DTYPES, f"q must be fp8, got {q.dtype}"
    assert k.dtype in _FP8_DTYPES, f"k must be fp8, got {k.dtype}"
    assert not q_smoothing, "q_smoothing is not supported for fp8 input"

    v_is_fp8 = v.dtype in _FP8_DTYPES
    if v_is_fp8:
        assert v_scale is not None, "v_scale must be provided when v is fp8"
    else:
        assert v_scale is None, "v_scale must be None when v is bf16/fp16 (computed internally)"

    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    b, s_q, h_q, d = map_dims(q.shape, bshd_map)
    _, s_k, h_k, _ = map_dims(k.shape, bshd_map)

    if sm_scale is None:
        sm_scale = d ** -0.5

    if R is None:
        R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=torch.bfloat16) / (
            BLOCK_R ** 0.5
        )

    apply_input_scale = q_input_scale is not None or k_input_scale is not None
    if apply_input_scale:
        if q_input_scale is None or k_input_scale is None:
            raise ValueError(
                "q_input_scale and k_input_scale must both be set for per-tensor fp8 comms descale."
            )
        q_input_scale_t = q_input_scale.reshape(()).to(device=q.device, dtype=torch.float32)
        k_input_scale_t = k_input_scale.reshape(()).to(device=q.device, dtype=torch.float32)
    else:
        q_input_scale_t = torch.ones((), device=q.device, dtype=torch.float32)
        k_input_scale_t = torch.ones((), device=q.device, dtype=torch.float32)

    # K mean for smoothing: use dequantized magnitudes when comms scales apply.
    if apply_input_scale:
        k_for_mean = k.to(torch.float32) * k_input_scale_t
    else:
        k_for_mean = k.to(torch.float32)
    k_mean = k_for_mean.mean(dim=1 if layout == "bshd" else 2)  # [B, H, D]

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd_map)

    Q_NUM_BLKS = (s_q + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (s_k + BLKK - 1) // BLKK

    # Match sage_quant_mxfp4: kernels write d//32 E8M0 groups, not d//BLOCK_R.
    scale_groups = d // 32
    Q_q = q.new_empty((*q.shape[:-1], d // 2), dtype=torch.uint8)
    Q_descale = q.new_empty((*q.shape[:-1], scale_groups), dtype=torch.uint8)
    K_q = k.new_empty((*k.shape[:-1], d // 2), dtype=torch.uint8)
    K_descale = k.new_empty((*k.shape[:-1], scale_groups), dtype=torch.uint8)

    stride_qqb, stride_qqm, stride_qqh, stride_qqd = map_dims(Q_q.stride(), bshd_map)
    stride_kqb, stride_kqn, stride_kqh, stride_kqd = map_dims(K_q.stride(), bshd_map)
    stride_qsb, stride_qsm, stride_qsh, stride_qsd = map_dims(Q_descale.stride(), bshd_map)
    stride_ksb, stride_ksn, stride_ksh, stride_ksd = map_dims(K_descale.stride(), bshd_map)

    # Single fused kernel handles all Q and K blocks concurrently.
    # Both Q and K use BLKQ as the rotation block size (same as original sage_quant_mxfp4).
    # BLKK is only used for the V kernel below.
    K_ROT_NUM_BLKS = (s_k + BLKQ - 1) // BLKQ
    total_pids = b * h_q * Q_NUM_BLKS + b * h_k * K_ROT_NUM_BLKS
    _rotate_quantize_qk_fp8_kernel[(total_pids,)](
        q,
        Q_q,
        Q_descale,
        k,
        K_q,
        K_descale,
        k_mean,
        R,
        q_input_scale_t,
        k_input_scale_t,
        apply_input_scale,
        sm_scale * 1.4426950408889634,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_qqb, stride_qqm, stride_qqh, stride_qqd,
        stride_qsb, stride_qsm, stride_qsh, stride_qsd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_kqb, stride_kqn, stride_kqh, stride_kqd,
        stride_ksb, stride_ksn, stride_ksh, stride_ksd,
        k_mean.stride(0), k_mean.stride(1), k_mean.stride(2),
        b, h_q, h_k, s_q, s_k, d,
        BLOCK_M=BLKQ,
        BLOCK_R=BLOCK_R,
        D=d,
        num_warps=4,
        num_stages=5,
    )

    if v_is_fp8:
        # v is already quantized upstream — pass through, skip kernel launch.
        v_fp8 = v
        if v_scale.ndim == 0 or v_scale.numel() == 1:
            v_scale = v_scale.reshape(()).expand(b, h_k, d).contiguous()
    else:
        # Quantize bf16/fp16 v to fp8.
        # sage_quant_v_kernel expects strides in (B, H, S, D) order regardless of
        # layout, so H and S are swapped for bshd tensors.
        if layout == "bshd":
            stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
                v.stride(0), v.stride(2), v.stride(1), v.stride(3)
            )
        else:
            stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
                v.stride(0), v.stride(1), v.stride(2), v.stride(3)
            )
        v_fp8 = torch.empty_like(v, dtype=FP8_TYPE)
        v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

        v_task_count = b * h_k * K_NUM_BLKS
        sage_quant_v_kernel[(v_task_count,)](
            v,
            v_fp8,
            v_scale,
            stride_bz_v, stride_h_v, stride_seq_v, stride_d_v,
            v_scale.stride(0), v_scale.stride(1),
            b, h_k,
            K_NUM_BLKS,
            s_k,
            D=d,
            BLK_K=BLKK,
            num_stages=3,
            num_warps=8,
        )

    return Q_q, Q_descale, K_q, K_descale, v_fp8, v_scale, None
