# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a16w8_blockscale import (
    _batched_gemm_a16w8_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
from aiter.jit.utils.torch_guard import torch_compile_guard

_LOGGER = AiterTritonLogger()


def batched_gemm_a16w8_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
    prequant: Optional[bool] = False,
) -> torch.Tensor:
    if y is None:
        Bx, M, _ = x.shape
        _, N, _ = w.shape
        return torch.empty((Bx, M, N), dtype=dtype, device=x.device)
    return y


@torch_compile_guard(gen_fake=batched_gemm_a16w8_fake_tensor)
def batched_gemm_a16w8_(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
    prequant: Optional[bool] = False,
) -> torch.Tensor:
    """
    Computes batched A16W8 matrix multiplication Y[i] = X[i] @ W[i]^T with blockscale quantization.

    Args:
        X: Batch tensor X with shape (B, M, K) in FP16/BF16.
        W: Batch tensor W with shape (B, N, K) in FP8.
        W_scale: Scale tensor for W with shape (B, **scale_n, *scale_k).
        bias (Optional[torch.Tensor]): Bias batch with shape (B, N) or (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        prequant: If True, quantize X to FP8 on-the-fly. If False, keep X as FP16/BF16.

    Returns:
        Y: The output batch tensor with shape (B, M, N).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k
    **scale_n = (N + scale_block_size_n - 1) // scale_block_size_n
    """
    _LOGGER.info(
        f"BATCHED_GEMM_A16W8: x={tuple(x.shape)} w={tuple(w.shape)} w_scale={tuple(w_scale.shape)} prequant={prequant}"
    )

    Bx, M, K = x.shape
    Bw, N, K_w = w.shape

    # Check constraints
    assert Bx == Bw, f"Batch size mismatch: x has {Bx}, w has {Bw}"
    assert K == K_w, f"K dimension mismatch: x has {K}, w has {K_w}"
    assert x.device == w.device, "x and w must be on the same device"
    assert x.device.type == "cuda", "Inputs must be on CUDA device"

    B = Bx

    if config is None:
        config, _ = _get_config(M, N, K)
    else:
        config = deserialize_str(config)

    # Allocate output if not provided
    if y is None:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)
    else:
        assert (
            y.shape[0] == B and y.shape[1] == M and y.shape[2] == N
        ), f"Output dimension error {y.shape} vs expected ({B}, {M}, {N})"

    # Transpose w and w_scale for computation
    # W goes from (B, N, K) -> (B, K, N)
    w = w.transpose(1, 2).contiguous()
    # W_scale goes from (B, scale_n, scale_k) -> (B, scale_k, scale_n)
    w_scale = w_scale.transpose(1, 2).contiguous()

    # Scale block sizes
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[1]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[2]))

    # Handle bias
    has_bias = bias is not None
    if has_bias:
        # Ensure bias has correct shape (B, N)
        if bias.dim() == 3:
            assert bias.shape == (
                B,
                1,
                N,
            ), f"Bias shape {bias.shape} incompatible with (B, 1, N)"
            bias = bias.squeeze(1)
        assert bias.shape == (B, N), f"Bias shape {bias.shape} must be (B, N)"
        bias_ptr = bias
        stride_biasb = bias.stride(0)
    else:
        bias_ptr = x  # Dummy pointer (won't be used)
        stride_biasb = 0

    # Get dtype bounds for FP8 quantization
    DTYPE_MAX = (
        torch.finfo(w.dtype).max
        if torch.is_floating_point(w)
        else torch.iinfo(w.dtype).max
    )

    def grid(META):
        return (
            B,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _batched_gemm_a16w8_kernel[grid](
        x,
        w,
        y,
        w_scale,
        bias_ptr,
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
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        stride_biasb,
        HAS_BIAS=has_bias,
        PREQUANT=prequant,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        **config,
    )

    return y


def batched_gemm_a16w8(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    prequant: Optional[bool] = False,
) -> torch.Tensor:
    """
    Public API for batched GEMM A16W8.

    Computes batched matrix multiplication Y[i] = X[i] @ W[i]^T with FP16/BF16 activations and FP8 weights.
    """
    config_hashable = serialize_dict(config) if config else None
    return batched_gemm_a16w8_(x, w, w_scale, bias, dtype, y, config_hashable, prequant)
