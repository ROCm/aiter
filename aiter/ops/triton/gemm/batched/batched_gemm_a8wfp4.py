# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8wfp4 import (
    _batched_gemm_a8wfp4_kernel,
    _batched_gemm_a8wfp4_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False

# a_dtype -> (elems per A byte, tl.dot_scaled format string).
_A_DTYPE = {
    "fp8": (1, "e4m3"),
    "fp4": (2, "e2m1"),
}


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value


def batched_gemm_a8wfp4(
    x,
    w,
    x_scales,
    w_scales,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    a_dtype: str = "fp8",
    layout: str = "bmn",
):
    """
    Batched MXFP GEMM Y[b] = dequant(X[b]) @ dequant(W[b])^T with per-1x32 e8m0
    microscales on both operands.

    Variants (a_dtype): "fp8" (MXFP8 E4M3 A x MXFP4 B, a8w4) or "fp4" (MXFP4 A x
    MXFP4 B, a4w4). Both A and B scales are folded into a scaled MFMA.

    Args:
        x (torch.Tensor): logical (B, M, K // A_PACK) uint8 A codes, where A_PACK is
            2 for "fp4" and 1 for "fp8". Arbitrary strides are honored, so the mbn
            layout is a `.transpose(0, 1)` view of a physical [M, B, *] buffer.
        w (torch.Tensor): (B, N, K // 2) packed MXFP4 weight codes, internally
            transposed to (B, K // 2, N).
        x_scales (torch.Tensor): logical (B, M, K // 32) e8m0 scale for x.
        w_scales (torch.Tensor): (B, N, K // 32) e8m0 scale for w.
        dtype (torch.dtype): output dtype (bf16 or fp16).
        y (Optional[torch.Tensor]): pre-allocated logical (B, M, N) output. When None,
            it is allocated per `layout`.
        config (Optional[dict]): kernel tuning params. When None, chosen by `_get_config`.
        a_dtype (str): "fp8" (default, a8w4) or "fp4" (a4w4).
        layout (str): "bmn" (contiguous [B, M, N] output) or "mbn" (physical [M, B, N]
            output returned as a [B, M, N] view; deepseek-v4 grouped output). Only used
            to allocate `y` when it is None.

    Returns:
        torch.Tensor: logical (B, M, N) output.
    """
    _LOGGER.info(
        f"BATCHED_GEMM_A8WFP4: x={tuple(x.shape)} w={tuple(w.shape)} "
        f"x_scale={tuple(x_scales.shape)} w_scale={tuple(w_scales.shape)} a_dtype={a_dtype}"
    )

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"
    if a_dtype not in _A_DTYPE:
        raise ValueError(f"a_dtype must be one of {sorted(_A_DTYPE)}; got {a_dtype!r}")
    if layout not in ("bmn", "mbn"):
        raise ValueError(f"layout must be 'bmn' or 'mbn'; got {layout!r}")

    A_PACK, A_FORMAT = _A_DTYPE[a_dtype]

    w = w.transpose(1, 2)  # (B, N, K//2) -> (B, K//2, N)
    Bx, M, _ = x.shape
    Bw, _, N = w.shape
    K = x.shape[-1] * A_PACK
    assert Bx == Bw, f"batch mismatch: x={Bx}, w={Bw}"
    Batch = Bx

    if y is None:
        if layout == "mbn":
            y = torch.empty((M, Batch, N), dtype=dtype, device=x.device).transpose(0, 1)
        else:
            y = torch.empty((Batch, M, N), dtype=dtype, device=x.device)
    assert y.shape == (Batch, M, N), f"y must be (B,M,N)={(Batch, M, N)}, got {tuple(y.shape)}"

    if config is None:
        config, _ = _get_config(M, N, K)

    if config["NUM_KSPLIT"] > 1:
        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (Batch, config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (Batch, config["NUM_KSPLIT"], M, N),
                dtype=torch.float32,
                device=y.device,
            )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        Batch,
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _batched_gemm_a8wfp4_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        y.stride(2) if config["NUM_KSPLIT"] == 1 else y_pp.stride(3),
        x_scales.stride(0),
        x_scales.stride(1),
        x_scales.stride(2),
        w_scales.stride(0),
        w_scales.stride(1),
        w_scales.stride(2),
        A_PACK=A_PACK,
        A_FORMAT=A_FORMAT,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            Batch,
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _batched_gemm_a8wfp4_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y_pp.stride(3),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            config["NUM_KSPLIT"],
        )

    return y
