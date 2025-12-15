# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
from torch import Tensor
import triton
import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger
<<<<<<< HEAD
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_string
from aiter.ops.triton._triton_kernels.gemm_afp4wfp4_pre_quant_atomic import (
    _gemm_afp4_wfp4_pre_quant_kernel,
    _get_config,
=======
from aiter.ops.triton.gemm_a16wfp4 import (
    gemm_a16wfp4,
>>>>>>> main
)
from aiter.jit.utils.torch_guard import torch_compile_guard

_LOGGER = AiterTritonLogger()


def gemm_afp4wfp4_pre_quant_fake_tensor(
    x: Tensor,
    w: Tensor,
    w_scales: Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
<<<<<<< HEAD
    config: Optional[str] = None,
) -> Tensor:
    if y is None:
        M, _ = x.shape
        N, _ = w.shape
        return torch.zeros((M, N), dtype=dtype, device=x.device)
    return y

@torch_compile_guard(gen_fake=gemm_afp4wfp4_pre_quant_fake_tensor)
def gemm_afp4wfp4_pre_quant_(
    x: Tensor,
    w: Tensor,
    w_scales: Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
) -> Tensor:
=======
    config: Optional[dict] = None,
):
>>>>>>> main
    _LOGGER.info(
        "gemm_afp4wfp4_pre_quant will be deprecated in future AITER release, please switch to gemm_a16wfp4"
    )
<<<<<<< HEAD

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape

    # inner kernel expects (K, N)
    w = w.T

    if y is None:
        y = torch.zeros((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)
    else:
        config = deserialize_string(config)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_afp4_wfp4_pre_quant_kernel[grid](
        x,
        w,
        y,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0,
        y.stride(0),
        y.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    return y

def gemm_afp4wfp4_pre_quant(
    x: Tensor,
    w: Tensor,
    w_scales: Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with on-the-fly FP4 quantization of activations.
    X is quantized to MXFP4 during computation, W is pre-quantized FP4. Uses atomic operations for split-K reduction.

    Args:
        x (torch.Tensor): Higher precision input matrix with shape (M, K) (BF16 or FP16).
            Quantized to FP4 E2M1 on-the-fly during GEMM.
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K), internally transposed.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
            Must be zero-initialized for atomic operations.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    config_hashable = serialize_dict(config) if config else None
    return gemm_afp4wfp4_pre_quant_(x, w, w_scales, dtype, y, config_hashable)
=======
    return gemm_a16wfp4(x, w, w_scales, True, dtype, y, config)
>>>>>>> main
