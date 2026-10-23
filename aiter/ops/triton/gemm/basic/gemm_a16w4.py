# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w4 import (
    _gemm_a16w4_kernel,
    _get_config,
)
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_a16w4(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    w_zeros: torch.Tensor | None = None,
    x_perm: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    y: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    Computes Y = X @ dequant(W)^T with 16-bit activations and group-wise int4 weights
    (W4A16, for GPTQ and AWQ checkpoints).

    The dequantized weight is W[n, k] = (q[n, k] - zero[n, g]) * w_scales[n, g],
    with g = k // group_size and group_size = K // w_scales.shape[1].

    Key parameters:
        x (torch.Tensor): FP16 or BF16 input matrix with shape (M, K).
        w (torch.Tensor): Packed uint4 weights q with shape (N, K // 2) and dtype
            uint8. Byte j of row n holds q[n, 2j] in bits 0-3 and q[n, 2j + 1]
            in bits 4-7.
        w_scales (torch.Tensor): Scales with shape (N, K // group_size), in FP16,
            BF16 or FP32. Use one column for per-channel scales.
        w_zeros (Optional[torch.Tensor]): Zero points with shape
            (N, K // group_size) and dtype uint8, values 0 to 15. If None, the
            zero point is 8 (symmetric quantization).
        x_perm (Optional[torch.Tensor]): Int32 tensor with shape (K,) for
            act-order checkpoints. Column k of W multiplies column x_perm[k] of X.
        bias (Optional[torch.Tensor]): Bias with shape (N,), added to Y.
        dtype (Optional[torch.dtype]): Output datatype. Default: x.dtype.
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M,
            BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, num_warps,
            num_stages, waves_per_eu, matrix_instr_nonkdim, cache_modifier).

    Returns:
        y (torch.Tensor): Output with shape (M, N).

    Limits: K must be a multiple of 8 and of group_size. group_size must be K
    (per-channel) or a power of two, 16 or more. Tuned configs exist for
    gfx942 and gfx1100 only. The helpers gptq_to_gemm_a16w4 and
    awq_to_gemm_a16w4 in aiter.ops.triton.utils.shuffle make these inputs from
    GPTQ and AWQ checkpoint tensors.
    """
    _LOGGER.info(
        "GEMM_A16W4: x=%s w=%s w_scales=%s w_zeros=%s x_perm=%s",
        tuple(x.shape),
        tuple(w.shape),
        tuple(w_scales.shape),
        None if w_zeros is None else tuple(w_zeros.shape),
        None if x_perm is None else tuple(x_perm.shape),
    )

    M, K = x.shape
    N = w.shape[0]
    assert x.dtype in (torch.float16, torch.bfloat16), "x must be FP16 or BF16"
    assert w.dtype == torch.uint8, "w must be uint8 with two uint4 values per byte"
    assert w.shape[1] * 2 == K, "w must have shape (N, K // 2)"
    assert K % 8 == 0, "K must be a multiple of 8"
    assert w_scales.dim() == 2 and w_scales.shape[0] == N
    num_groups = w_scales.shape[1]
    assert (
        K % num_groups == 0
    ), f"K={K} must be a multiple of the group size (w_scales has {num_groups} groups)"
    group_size = K // num_groups
    if num_groups > 1:
        assert group_size >= 16 and group_size & (group_size - 1) == 0, (
            f"group_size = K // w_scales.shape[1] = {group_size} must be a power of "
            "two, 16 or more (or K, with one column of w_scales)"
        )
    if w_zeros is not None:
        assert w_zeros.shape == w_scales.shape, "w_zeros must match w_scales"
        assert w_zeros.dtype == torch.uint8, "w_zeros must be uint8"
    if x_perm is not None:
        assert x_perm.shape == (K,) and x_perm.dtype == torch.int32
        # The kernel reads x_perm and bias with unit stride.
        x_perm = x_perm.contiguous()
    if bias is not None:
        assert bias.shape == (N,)
        bias = bias.contiguous()
    if dtype is None:
        dtype = x.dtype

    # Eight uint4 values of K in each int32 word.
    w = w.view(torch.int32)

    if config is None:
        config, _ = _get_config(M, N, K)

    compute_splitk_params(config, K)
    config["SCALE_PER_TILE"] = (
        num_groups == 1 or group_size % config["BLOCK_SIZE_K"] == 0
    )
    # Scaling the dot result is exact and faster for small M tiles. With large
    # M tiles, the extra accumulator costs registers.
    config["SCALE_AFTER_DOT"] = (
        config["SCALE_PER_TILE"] and config["BLOCK_SIZE_M"] <= 64
    )

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    if M == 0:
        return y

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device
        )
    else:
        y_pp = None

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16w4_kernel[grid](
        x,
        w,
        y if y_pp is None else y_pp,
        w_scales,
        w_scales if w_zeros is None else w_zeros,
        x if x_perm is None else x_perm,
        x if bias is None else bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if y_pp is None else y_pp.stride(0),
        y.stride(0) if y_pp is None else y_pp.stride(1),
        y.stride(1) if y_pp is None else y_pp.stride(2),
        w_scales.stride(0),
        w_scales.stride(1),
        0 if w_zeros is None else w_zeros.stride(0),
        0 if w_zeros is None else w_zeros.stride(1),
        GROUP_K=group_size,
        HAS_ZEROS=w_zeros is not None,
        HAS_PERM=x_perm is not None,
        ADD_BIAS=bias is not None and y_pp is None,
        **config,
    )

    if y_pp is not None:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            bias,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=bias is not None,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_a16w4_reduce_kernel",
        )

    return y
