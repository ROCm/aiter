# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.activation import _get_activation_from_str
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _gemm_a16_w16_gated_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    remap_xcd(pid, GRID_MN)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32

    """
    Our effective block size is actually BLOCK_N // 2.
    Per Triton program, we compute the matmul for TWO tiles of C of shape (BLOCK_M, BLOCK_N // 2) -
    one on the left side of C and one on the right side.
    """
    offs_bn0 = (
        pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    ) % (N // 2)
    offs_bn1 = (
        (pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2))
        % (N // 2)
    ) + (N // 2)
    b0_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn0[None, :] * stride_bn)
    b1_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn1[None, :] * stride_bn)
    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=acc_dtype)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b0 = tl.load(b0_ptrs, cache_modifier=cache_modifier)
            b1 = tl.load(b1_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b0 = tl.load(
                b0_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )
            b1 = tl.load(
                b1_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )

        acc0 += tl.dot(a, b0, input_precision="ieee")
        acc1 += tl.dot(a, b1, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b0_ptrs += BLOCK_SIZE_K * stride_bk
        b1_ptrs += BLOCK_SIZE_K * stride_bk

    if use_activation:
        acc0 = activation(acc0)

    acc_gated = acc0 * acc1
    c = acc_gated.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < (N // 2))
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-gated.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = (
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-gated-N={N}-K={K}.json"
        )
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    bounds = [64, 128, 256, 512, 2048]
    for bound in bounds:
        if M <= bound and f"M_LEQ_{bound}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"M_LEQ_{bound}"]
    else:
        return _get_config._config_dict[key]["M_GEQ_4096"]


def gemm_a16w16_gated(
    x,
    w,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """
    Computes the 16 bit matmul Y = X x W
    Uses the first half of the output (along the N dim) as a gate for the second half (e.g for SwiGLU)

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, N//2).
    If this is none, then it's created by this API and returned as output.
    - activation: Optional activation function to apply to the output. One of ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu")

    Returns:
    - Y: The output matrix with shape (M, N//2).
    """
    _LOGGER.info(f"GEMM_A16W16_GATED: x={tuple(x.shape)} w={tuple(w.shape)}")

    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
    M, K = x.shape
    N, K = w.shape

    assert N % 2 == 0, "Weight shape incompatible with gating (N not divisible by 2)"

    w = w.T

    if y is None:
        y = torch.empty((M, N // 2), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a16_w16_gated_kernel[grid](
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        **config,
    )

    return y
