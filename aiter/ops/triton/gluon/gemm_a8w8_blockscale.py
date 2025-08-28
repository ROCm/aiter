# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
from aiter.ops.triton.utils.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from triton import language as tl

_LOGGER = AiterTritonLogger()
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gemm_a8w8_blockscale_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called direcgly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N
    """

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = gl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 16],  # 128 * 128
        threads_per_warp=[8, 8],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 4],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True, warps_per_cta=[2, 2]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        # SPLITK_BLOCK_SIZE = gl.cdiv(K, NUM_KSPLIT)
        num_k_iter = gl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        smem_a = gl.allocate_shared_memory(
            a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
        )

        smem_b = gl.allocate_shared_memory(
            b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
        )

        # Create pointers for first block of A and B input matrices
        offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
        offs_ak_split = pid_k * SPLITK_BLOCK_SIZE + offs_ak
        offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
        offs_bk_split = pid_k * SPLITK_BLOCK_SIZE + offs_bk

        smem_scale_a = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [BLOCK_SIZE_M], layout=shared_a_scale
        )

        smem_scale_b = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [BLOCK_SIZE_N], layout=shared_b_scale
        )

        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
        )
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
        )

        offs_a = offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak

        # Create pointers for the scales
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        offs_a_scale = offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k

        if EVEN_K:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=offs_am[:, None] < M,
                cache=cache_modifier,
            )
        else:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=(offs_ak[None, :] < K - (pid_k * num_k_iter * BLOCK_SIZE_K))
                & (offs_am[:, None] < M),
                cache=cache_modifier,
            )
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        offs_b = offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        offs_bsn = offs_bn // GROUP_N
        offs_b_scale = offs_k_scale * stride_bscale_k + offs_bsn * stride_bscale_n

        if EVEN_K:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=offs_bn[None, :] < N,
                cache=cache_modifier,
            )
        else:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=(offs_bk[:, None] < K - (pid_k * num_k_iter * BLOCK_SIZE_K))
                & (offs_bn[None, :] < N),
                cache=cache_modifier,
            )
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr,
            offsets=offs_b_scale,
            cache=cache_modifier,
        )
        smem_a.store(a)
        smem_scale_a.store(a_scale)

        acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
        acc = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
        )
        zeros = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
        )

        offs_ks_step = BLOCK_SIZE_K // GROUP_K  # could be replaced by a constant 1

        for k in range(pid_k * num_k_iter, ((pid_k + 1) * num_k_iter) - 1):
            # Advance the ptrs to the next K block.
            offs_a += BLOCK_SIZE_K * stride_ak
            offs_b += BLOCK_SIZE_K * stride_bk
            a_scale_ptr += offs_ks_step * stride_ascale_k
            b_scale_ptr += offs_ks_step * stride_bscale_k

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = gl.amd.cdna4.buffer_load(
                    ptr=a_ptr,
                    offsets=offs_a,
                    mask=offs_am[:, None] < M,
                    cache=cache_modifier,
                )
            else:
                a = gl.amd.cdna4.buffer_load(
                    ptr=a_ptr,
                    offsets=offs_a,
                    mask=(offs_ak[None, :] < K - (k + 1) * BLOCK_SIZE_K)
                    & (offs_am[:, None] < M),
                    cache=cache_modifier,
                )
            smem_b.store(b)
            smem_scale_b.store(b_scale)
            cur_a = smem_a.load(layout=dot_a_layout)
            cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, mfma_layout))
            a_scale = gl.amd.cdna4.buffer_load(
                ptr=a_scale_ptr,
                offsets=offs_a_scale,
                cache=cache_modifier,
            )
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, mfma_layout))
            if EVEN_K:
                b = gl.amd.cdna4.buffer_load(
                    ptr=b_ptr,
                    offsets=offs_b,
                    mask=offs_bn[None, :] < N,
                    cache=cache_modifier,
                )
            else:
                b = gl.amd.cdna4.buffer_load(
                    ptr=b_ptr,
                    offsets=offs_b,
                    mask=(offs_bk[:, None] < K - (k + 1) * BLOCK_SIZE_K)
                    & (offs_bn[None, :] < N),
                    cache=cache_modifier,
                )
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr,
                offsets=offs_b_scale,
                cache=cache_modifier,
            )
            cur_b = smem_b.load(layout=dot_b_layout)

            mfma_out = gl.amd.cdna4.mfma(cur_a, cur_b, zeros)
            acc += mfma_out * cur_a_scale[:, None] * cur_b_scale[None, :]

            smem_a.store(a)
            smem_scale_a.store(a_scale)

        # ======= Epilogue ========
        smem_b.store(b)
        smem_scale_b.store(b_scale)
        cur_a = smem_a.load(layout=dot_a_layout)
        cur_b = smem_b.load(layout=dot_b_layout)
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, mfma_layout))
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, mfma_layout))

        zeros = gl.amd.cdna4.mfma(cur_a, cur_b, zeros)
        acc += zeros * cur_a_scale[:, None] * cur_b_scale[None, :]

        c = acc.to(c_ptr.type.element_ty)

        # # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
        )
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        c_offs = (
            stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        gl.amd.cdna4.buffer_store(
            stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask
        )


@triton.jit
def _gemm_a8w8_blockscale_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    ACTUAL_KSPLIT: gl.constexpr,
    MAX_KSPLIT: gl.constexpr,
):

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N)
    offs_k = gl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = gl.load(c_in_ptrs)
    else:
        c = gl.load(
            c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT
        )  # , other=0.0)
    c = gl.sum(c, axis=0)

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    gl.store(c_out_ptrs, c)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        if int(dev.split("MI")[1].replace("X", "")) < 350:
            raise ValueError(
                "Gluon implementation is not supported on this device (requires CDNA4)."
            )
        _get_config._config_dict = {}
        fpath = (
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE.json"
        )
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    return _get_config._config_dict[key]["any"]


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: Scale tensor for X with shape (M, *scale_k).
    - W_scale: Scale tensor for W with shape (**scale_n, *scale_k).

    Returns:
    - Y: The output matrix with shape (M, N).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k
    **scale_n = (N + scale_block_size_n - 1) // scale_block_size_n
    """
    _LOGGER.info(
        f"GEMM_A8W8_BLOCKSCALE: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, K = w.shape

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    w = w.T
    w_scale = w_scale.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        y_pp = None

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))

    assert (
        config["GROUP_K"] == config["BLOCK_SIZE_K"]
    ), "GROUP_K must equal BLOCK_SIZE_K"
    assert (
        config["BLOCK_SIZE_M"] == 128
    ), "Before modifying the hparams, also make sure to modify the Gluon layouts. Afterwards, modify this assert."
    assert (
        config["BLOCK_SIZE_N"] == 128
    ), "Before modifying the hparams, also make sure to modify the Gluon layouts. Afterwards, modify this assert."
    assert (
        config["BLOCK_SIZE_K"] == 128
    ), "Before modifying the hparams, also make sure to modify the Gluon layouts. Afterwards, modify this assert."

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a8w8_blockscale_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_a8w8_blockscale_reduce_kernel[grid_reduce](
            y_pp,
            y,
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
        )

    return y
