# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import os
import warnings
import triton
import triton.language as tl
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton import arch_info
from ..utils.core import AITER_TRITON_CONFIGS_PATH
from .activation import _get_activation_from_str


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _ff_a16w16_fused_ungated(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_w1k,
    stride_w1n,
    stride_w2n,
    stride_w2k,
    stride_ym,
    stride_yk,
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

    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_w1k > 0)
    tl.assume(stride_w1n > 0)
    tl.assume(stride_w2k > 0)
    tl.assume(stride_w2n > 0)
    tl.assume(stride_ym > 0)
    tl.assume(stride_yk > 0)

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

    # Create pointers for first block of x and w1 input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_xm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    acc_dtype = tl.float32 if y_ptr.type.element_ty != tl.int8 else tl.int32

    offs_w1n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_w1n[None, :] * stride_w1n)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            x = tl.load(x_ptrs, mask=offs_xm[:, None] < M)
            w1 = tl.load(
                w1_ptrs,
                mask=offs_w1n[None, :] < N,
                cache_modifier=cache_modifier,
            )
        else:
            x = tl.load(
                x_ptrs,
                mask=(offs_xm[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            w1 = tl.load(
                w1_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_w1n[None, :] < N),
                other=0.0,
                cache_modifier=cache_modifier,
            )

        acc += tl.dot(x, w1, input_precision="ieee")

        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w1_ptrs += BLOCK_SIZE_K * stride_w1k

    if use_activation:
        acc = activation(acc)

    acc = acc.to(w2_ptr.type.element_ty)

    offs_w2n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    w2_ptrs = w2_ptr + (offs_w2n[:, None] * stride_w2n + offs_k[None, :] * stride_w2k)

    offs_ym = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_ptrs = y_ptr + (offs_ym[:, None] * stride_ym + offs_k[None, :] * stride_yk)

    # Stagger k-loop start position based on N block index (to minimize contention)
    k_cyclic_offset = pid_n % tl.cdiv(K, BLOCK_SIZE_K)
    w2_ptrs += k_cyclic_offset * stride_w2k * BLOCK_SIZE_K
    y_ptrs += k_cyclic_offset * stride_yk * BLOCK_SIZE_K

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            w2 = tl.load(
                w2_ptrs,
                mask=offs_w2n[:, None] < N,
            )
        else:
            w2 = tl.load(
                w2_ptrs,
                mask=(offs_w2n[:, None] < N)
                & ((offs_k[None, :] + k_cyclic_offset * BLOCK_SIZE_K) < K),
                other=0.0,
            )
        partial_sum_y = tl.dot(acc, w2)
        # tl.device_print("w2:", w2)
        # tl.device_print("partial y:", partial_sum_y)
        y_mask = (offs_ym[:, None] < M) & ((offs_k[None, :] + BLOCK_SIZE_K * k) < K)
        tl.atomic_add(y_ptrs, partial_sum_y, mask=y_mask, sem="relaxed", scope="gpu")
        # tl.store(y_ptrs, partial_sum_y, mask=y_mask)
        k_cyclic_offset += 1
        if k_cyclic_offset >= tl.cdiv(K, BLOCK_SIZE_K):
            k_cyclic_offset = 0
            w2_ptrs -= BLOCK_SIZE_K * stride_w2k * (tl.cdiv(K, BLOCK_SIZE_K) - 1)
            y_ptrs -= BLOCK_SIZE_K * stride_yk * (tl.cdiv(K, BLOCK_SIZE_K) - 1)
        else:
            w2_ptrs += BLOCK_SIZE_K * stride_w2k
            y_ptrs += BLOCK_SIZE_K * stride_yk


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-FF-A16W16-fused.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = (
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-FF-A16W16-fused-N={N}-K={K}.json"
        )
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    bounds = [4, 8, 64]
    for bound in bounds:
        if M <= bound and f"M_LEQ_{bound}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"M_LEQ_{bound}"]
    else:
        return _get_config._config_dict[key]["M_GEQ_4096"]
