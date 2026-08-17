# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16_persistent import (
    gemm_a16w16_persistent_kernel_,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    """Check if the gluon backend is available for the current GPU architecture."""
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def gemm_a16w16_persistent(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    activation: str | None = None,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
) -> torch.Tensor:
    """
    Computes 16 bit matrix multiplication Y = X @ W^T

    Uses the gluon backend automatically on supported architectures (gfx1250)
    and the triton backend everywhere else. Pass ``backend`` to force a choice.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters.
        activation (Optional[str]): Activation function ("gelu", "gelu_tanh", "silu",
            "silu_exp2", "relu").
        kernel_type (str): Ignored. Kept for signature parity with gemm_a16w16;
            the persistent kernels have no bandwidth/compute-bound variants.
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    if backend is None:
        backend = "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
    M, K = x.shape
    N, _ = w.shape

    if config is None:
        config, _ = get_gemm_config("GEMM-A16W16-PERSISTENT", M, N, K)
        if backend == "triton":
            # fills in SPLITK_BLOCK_SIZE / cache_modifier defaults and clamps
            # BLOCK_SIZE_K, which the triton kernel needs; the gluon kernel does not
            config = compute_splitk_params(config, K)

    assert config.get("NUM_KSPLIT", 1) == 1, (
        f"gemm_a16w16_persistent does not support split-K yet (got NUM_KSPLIT="
        f"{config.get('NUM_KSPLIT')}); use gemm_a16w16 instead"
    )

    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        import triton.experimental.gluon.language as gl

        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16_persistent import (
            gemm_a16w16_persistent_kernel_,
        )

        _LOGGER.info(
            f"GEMM_A16W16_PERSISTENT [gluon/gfx1250]: x={tuple(x.shape)} w={tuple(w.shape)}"
        )
        assert x.dtype in (
            torch.float16,
            torch.bfloat16,
        ), f"Activations (x) must be fp16 or bf16, got {x.dtype}"
        assert w.dtype in (
            torch.float16,
            torch.bfloat16,
        ), f"Weights (w) must be fp16 or bf16, got {w.dtype}"
        BLOCK_M = config["BLOCK_M"]
        BLOCK_N = config["BLOCK_N"]
        BLOCK_K = config["BLOCK_K"]
        NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
        GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)
        num_warps = config["num_warps"]

        w = w.T

        # Clamp the pipeline depth
        num_k_tiles = triton.cdiv(K, BLOCK_K)
        NUM_BUFFERS = max(2, min(NUM_BUFFERS, num_k_tiles + 1))

        if y is None:
            y = torch.empty((M, N), dtype=dtype, device=x.device)

        shared_a = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
        )
        shared_b = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
        )

        # Persistent, NUM_WGS processes num_tiles
        NUM_WGS = torch.cuda.get_device_properties(x.device).multi_processor_count
        num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        grid = (min(NUM_WGS, num_tiles),)

        gemm_a16w16_persistent_kernel_[grid](
            x,
            w,
            bias,
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
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_BUFFERS=NUM_BUFFERS,
            SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b,
            activation=_get_activation_from_str(activation) if activation else None,
            USE_ACTIVATION=activation is not None,
            ADD_BIAS=(bias is not None),
            NUM_WGS=NUM_WGS,
            num_warps=num_warps,
        )

        return y

    _LOGGER.info(f"GEMM_A16W16_PERSISTENT [triton]: x={tuple(x.shape)} w={tuple(w.shape)}")

    w = w.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # Persistent 
    NUM_WGS = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = lambda META: (
        min(
            NUM_WGS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    gemm_a16w16_persistent_kernel_[grid](
        x,
        w,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0,  # stride_ck
        y.stride(0),
        y.stride(1),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=(bias is not None),
        SKIP_REDUCE=False,
        NUM_WGS=NUM_WGS,
        **config,
    )

    return y
