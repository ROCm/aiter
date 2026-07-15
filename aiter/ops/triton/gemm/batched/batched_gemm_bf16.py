# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_bf16 import (
    _batched_gemm_bf16_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _batched_gemm_splitk_reduce_kernel,
)
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils._triton.arch_info import get_arch

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:
        return False


def batched_gemm_bf16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    YQ: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
):
    """
    Computes batched 16 bit matrix multiplication Y[i] = X[i] @ W[i]^T with optional bias.

    Args:
        XQ (torch.Tensor): Input batch with shape (B, M, K) (BF16 or FP16).
        WQ (torch.Tensor): Weight batch with shape (B, N, K), internally transposed.
        bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        splitK (Optional[int]): Not supported. Must be None.
        YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N).
        config (Optional[dict]): Kernel tuning parameters.
        kernel_type (str): [gluon only] Kernel variant ("bandwidth_bound", "compute_bound").
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output batch with shape (B, M, N).
    """
    _LOGGER.info(f"BATCHED_GEMM_BF16: x={tuple(XQ.shape)} w={tuple(WQ.shape)}")

    assert XQ.shape[0] == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert XQ.shape[2] == WQ.shape[2], "Incompatible K dimensions!!!"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_bf16"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    # Transpose N and K dimensions of WQ: (B, N, K) -> (B, K, N)
    WQ = WQ.transpose(1, 2)

    B = XQ.shape[0]
    M = XQ.shape[1]
    K = XQ.shape[2]
    N = WQ.shape[2]

    has_bias = bias is not None

    if backend is None:
        backend = "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_bf16 import (
            _KERNEL_MAP,
        )
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
            create_shared_layouts,
            create_wmma_layouts,
        )

        assert (
            kernel_type in _KERNEL_MAP
        ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP.keys())}"
        _LOGGER.info(
            f"BATCHED_GEMM_BF16 [gluon/gfx1250]: x={tuple(XQ.shape)} w={tuple(WQ.shape)} "
            f"kernel={kernel_type}"
        )

        if config is None:
            config, _ = get_gemm_config(
                "BATCHED_GEMM-A16W16", M, N, K, B=B, backend="gluon"
            )

        kernel_type_from_config = config.pop("kernel_type", None)
        if kernel_type_from_config is not None:
            kernel_type = kernel_type_from_config

        BLOCK_M = config["BLOCK_SIZE_M"]
        BLOCK_N = config["BLOCK_SIZE_N"]
        BLOCK_K = config["BLOCK_SIZE_K"]
        NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
        num_warps = config["num_warps"]
        waves_per_eu = config["waves_per_eu"]
        cache_modifier = config.get("cache_modifier", None)
        NUM_KSPLIT = config.get("NUM_KSPLIT", 1)
        SPLITK_BLOCK_SIZE = triton.cdiv(K, NUM_KSPLIT)

        num_k_tiles = triton.cdiv(SPLITK_BLOCK_SIZE, BLOCK_K)
        _MIN_BUFFERS = {"bandwidth_bound": 1, "compute_bound": 2}
        _DEPTH_SLACK = {"compute_bound": 2}

        if kernel_type_from_config is None:
            depth_cap = num_k_tiles - _DEPTH_SLACK.get(kernel_type, 0)
            if depth_cap < _MIN_BUFFERS[kernel_type]:
                needed = _MIN_BUFFERS[kernel_type] + _DEPTH_SLACK.get(kernel_type, 0)
                _LOGGER.warning(
                    f"BATCHED_GEMM_BF16 [gluon/gfx1250]: kernel_type='{kernel_type}' needs "
                    f"num_k_tiles>={needed} but num_k_tiles={num_k_tiles} "
                    f"(K={K}, BLOCK_K={BLOCK_K}); falling back to kernel_type='bandwidth_bound'."
                )
                kernel_type = "bandwidth_bound"
                depth_cap = num_k_tiles
        else:
            depth_cap = num_k_tiles - _DEPTH_SLACK.get(kernel_type, 0)

        NUM_BUFFERS = min(NUM_BUFFERS, depth_cap)

        if YQ is None:
            YQ = torch.empty((B, M, N), dtype=dtype, device=XQ.device)

        if NUM_KSPLIT > 1:
            y_pp = torch.empty(
                (NUM_KSPLIT, B, M, N),
                dtype=torch.float32,
                device=XQ.device,
            )
        else:
            y_pp = None

        # Operand layout
        if XQ.stride(2) == 1:
            layout = "T"
        elif XQ.stride(1) == 1:
            layout = "N"
        else:
            raise ValueError(
                f"XQ must be contiguous in at least one of M/K dims, got strides {XQ.stride()}"
            )

        if WQ.stride(2) == 1:
            layout += "T"
        elif WQ.stride(1) == 1:
            layout += "N"
        else:
            raise ValueError(
                f"WQ must be contiguous in at least one of K/N dims, got strides {WQ.stride()}"
            )

        wmma_layout, operand_a, operand_b = create_wmma_layouts(num_warps)
        shared_a, shared_b = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, layout)

        out_tensor = YQ if NUM_KSPLIT == 1 else y_pp

        grid = (
            B,
            NUM_KSPLIT * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        )

        num_ksplit = NUM_KSPLIT
        splitk_block_size = SPLITK_BLOCK_SIZE

        _KERNEL_MAP[kernel_type][grid](
            XQ,
            WQ,
            out_tensor,
            bias,
            M,
            N,
            K,
            XQ.stride(0),
            XQ.stride(1),
            XQ.stride(2),
            WQ.stride(0),
            WQ.stride(1),
            WQ.stride(2),
            YQ.stride(0) if num_ksplit == 1 else y_pp.stride(1),
            YQ.stride(1) if num_ksplit == 1 else y_pp.stride(2),
            YQ.stride(2) if num_ksplit == 1 else y_pp.stride(3),
            0 if num_ksplit == 1 else y_pp.stride(0),
            bias.stride(0) if has_bias else 0,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            NUM_BUFFERS=NUM_BUFFERS,
            LAYOUT=layout,
            SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b,
            WMMA_LAYOUT=wmma_layout,
            OPERAND_LAYOUT_A=operand_a,
            OPERAND_LAYOUT_B=operand_b,
            ADD_BIAS=has_bias,
            NUM_KSPLIT=num_ksplit,
            SPLITK_BLOCK_SIZE=splitk_block_size,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            cache_modifier=cache_modifier,
        )

    else:
        # ---- Triton backend ----
        if YQ is None:
            YQ = torch.empty((B, M, N), dtype=dtype, device=XQ.device)

        if config is None:
            config, _ = _get_config(M, N, K, B=B)

        num_ksplit = config["NUM_KSPLIT"]
        splitk_block_size = config["SPLITK_BLOCK_SIZE"]

        if num_ksplit > 1:
            y_pp = torch.empty(
                (num_ksplit, B, M, N),
                dtype=torch.float32,
                device=XQ.device,
            )
        else:
            y_pp = None

        grid = lambda META: (  # noqa: E731
            B,
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        _batched_gemm_bf16_kernel[grid](
            XQ,
            WQ,
            YQ if num_ksplit == 1 else y_pp,
            bias,
            M,
            N,
            K,
            XQ.stride(0),
            XQ.stride(1),
            XQ.stride(2),
            WQ.stride(0),
            WQ.stride(1),
            WQ.stride(2),
            YQ.stride(0) if num_ksplit == 1 else y_pp.stride(1),
            YQ.stride(1) if num_ksplit == 1 else y_pp.stride(2),
            YQ.stride(2) if num_ksplit == 1 else y_pp.stride(3),
            0 if num_ksplit == 1 else y_pp.stride(0),
            bias.stride(0) if has_bias else 0,
            has_bias,
            **config,
        )

    # ---- Shared split-K reduction ----
    if num_ksplit > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, splitk_block_size)

        grid_reduce = (
            B,
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _batched_gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            YQ,
            bias,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y_pp.stride(3),
            YQ.stride(0),
            YQ.stride(1),
            YQ.stride(2),
            bias.stride(0) if has_bias else 0,
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(num_ksplit),
            ADD_BIAS=has_bias,
            activation="",
            use_activation=False,
            KERNEL_NAME="_batched_gemm_bf16_reduce_kernel",
        )

    return YQ
