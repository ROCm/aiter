# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import copy

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_afp8wfp8 import (
    _batched_gemm_afp8wfp8_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# tl.dot_scaled reads one E8M0 scale for each 32 values along K.
_SCALE_GROUP_SIZE = 32


def batched_gemm_afp8wfp8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    w_scale_group_size: tuple[int, int] = (1, 32),
) -> torch.Tensor:
    """Compute Y[b] = X[b] @ W[b].T for MXFP8 X and W.

    X and W are E4M3. Each 32 values of X along K share one E8M0 scale. W has
    one E8M0 scale for each (1, 32) or (32, 32) block of its (N, K) matrix. All
    tensors can have any positive strides. Thus a token-major (M, B, K) buffer
    goes in through a transposed view, with no copy.

    Args:
        x: E4M3 activations, (B, M, K). A uint8 view is accepted.
        w: E4M3 weights, (B, N, K). A uint8 view is accepted.
        x_scales: E8M0 scales of x, (B, M, K // 32). A uint8 view is accepted.
        w_scales: E8M0 scales of w, (B, ceil(N / n), K // 32), where (n, 32) is
            w_scale_group_size. A uint8 view is accepted.
        dtype: Output dtype: bf16, fp16 or fp32.
        y: Optional output, (B, M, N), of dtype.
        config: Optional kernel config. If it is None, the config comes from
            the tuned JSON files for B, M, N and K.
        w_scale_group_size: (N, K) elements for each weight scale: (1, 32) or
            (32, 32).

    Returns:
        y, (B, M, N).

    Limits: K must be a multiple of 32. The kernel has no split-K.
    """
    _LOGGER.info(
        "BATCHED_GEMM_AFP8WFP8: x=%s w=%s x_scales=%s w_scales=%s",
        tuple(x.shape),
        tuple(w.shape),
        tuple(x_scales.shape),
        tuple(w_scales.shape),
    )
    assert x.ndim == w.ndim == x_scales.ndim == w_scales.ndim == 3
    batch, M, K = x.shape
    batch_w, N, K_w = w.shape
    assert batch == batch_w and K == K_w, "x and w must share B and K"
    assert K % _SCALE_GROUP_SIZE == 0, f"K={K} must be a multiple of 32"
    w_scale_group_size = tuple(w_scale_group_size)
    assert w_scale_group_size in ((1, 32), (32, 32)), w_scale_group_size
    group_n = w_scale_group_size[0]
    assert tuple(x_scales.shape) == (batch, M, K // _SCALE_GROUP_SIZE)
    assert tuple(w_scales.shape) == (
        batch,
        triton.cdiv(N, group_n),
        K // _SCALE_GROUP_SIZE,
    ), f"w_scales must be (B, ceil(N / {group_n}), K // 32)"
    assert all(t.dtype in (torch.uint8, torch.float8_e4m3fn) for t in (x, w))
    assert all(
        t.dtype in (torch.uint8, torch.float8_e8m0fnu) for t in (x_scales, w_scales)
    )
    assert dtype in (torch.bfloat16, torch.float16, torch.float32)
    if y is None:
        y = torch.empty((batch, M, N), dtype=dtype, device=x.device)
    else:
        assert tuple(y.shape) == (batch, M, N) and y.dtype == dtype
    if M == 0 or batch == 0:
        return y

    if config is None:
        config, _ = _get_config(batch, M, N, K, w_scale_group_size)
    config = copy.deepcopy(config)
    # Native E4M3 pointers keep the scaled-dot layouts; scales go in as bytes.
    x, w = x.view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn)
    x_scales, w_scales = x_scales.view(torch.uint8), w_scales.view(torch.uint8)
    w = w.transpose(1, 2)
    grid = (
        batch,
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _batched_gemm_afp8wfp8_kernel[grid](
        x,
        w,
        y,
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
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        x_scales.stride(2),
        w_scales.stride(0),
        w_scales.stride(1),
        w_scales.stride(2),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        B_SCALE_N_GROUP=group_n,
        cache_modifier=config["cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
        matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
    )
    return y
