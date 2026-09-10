# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)

_CONFIG_NAME = "BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT"


def _is_gluon_available():
    """Check if the gluon backend is available for the current GPU architecture."""
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def _get_gluon_config(B: int, M: int, N: int, K: int):
    """Load the tuned gluon config for this shape from
    configs/<arch>/gluon/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/.
    """
    config, _ = get_gemm_config(_CONFIG_NAME, M, N, K, backend="gluon", B=B)
    return config


def _batched_gemm_a8w8_ptg_gluon(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    YQ: torch.Tensor,
    B: int,
    M: int,
    N: int,
    K: int,
    group_size: int,
    bias: torch.Tensor | None,
    transpose_bm: bool,
    transpose_bm_in: bool,
    config: dict | None,
):
    """Gluon/gfx1250 path.

    ``WQ`` is passed in as (B, N, K) and is NOT transposed here: the kernel
    reads (N, K) tiles directly through its tensor descriptor.
    """
    import triton.experimental.gluon.language as gl

    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501
        _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_gluon_kernel as _kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501
        create_blocked_a,
        create_shared_layouts,
        create_wmma_layouts,
    )

    cfg = dict(_get_gluon_config(B, M, N, K) if config is None else config)

    BLOCK_M = cfg["BLOCK_SIZE_M"]
    BLOCK_N = cfg["BLOCK_SIZE_N"]
    BLOCK_K = cfg["BLOCK_SIZE_K"]
    num_warps = cfg.get("num_warps", 4)
    waves_per_eu = cfg.get("waves_per_eu", 0)
    NUM_BUFFERS = cfg.get("NUM_BUFFERS", 2)
    QUANT_IN_DOT = bool(cfg.get("QUANT_IN_DOT", 1))

    # Clamp the tile to the problem. BLOCK_K must stay a whole number of
    # quantization groups (the in-kernel amax reduces over exactly one group),
    # so it is clamped to the group-aligned round-up of K rather than to K.
    k_aligned = triton.cdiv(K, group_size) * group_size
    BLOCK_K = max(group_size, min(BLOCK_K, k_aligned))
    BLOCK_N = min(BLOCK_N, max(16, triton.next_power_of_2(N)))
    if M > 0:
        BLOCK_M = min(BLOCK_M, max(16, triton.next_power_of_2(M)))

    assert (
        BLOCK_K % group_size == 0
    ), f"BLOCK_SIZE_K ({BLOCK_K}) must be a multiple of group_size ({group_size})"
    K_GROUPS = BLOCK_K // group_size

    num_k_tiles = triton.cdiv(K, BLOCK_K)
    NUM_BUFFERS = max(1, min(NUM_BUFFERS, num_k_tiles))

    DTYPE_MAX = torch.finfo(WQ.dtype).max
    fp8_ty = gl.float8e4nv if WQ.dtype == torch.float8_e4m3fn else gl.float8e5

    wmma_layout, opa, opb = create_wmma_layouts(num_warps, BLOCK_M, BLOCK_N)
    blocked_a = create_blocked_a(BLOCK_M, group_size, num_warps)
    shared_a, shared_b = create_shared_layouts(
        BLOCK_M, BLOCK_N, BLOCK_K, X.element_size() * 8, WQ.element_size() * 8
    )

    grid = (B, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))

    _kernel[grid](
        X,
        WQ,
        YQ,
        w_scale,
        bias,
        M,
        N,
        K,
        X.stride(0) if not transpose_bm_in else X.stride(1),
        X.stride(1) if not transpose_bm_in else X.stride(0),
        X.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0) if not transpose_bm else YQ.stride(1),
        YQ.stride(1) if not transpose_bm else YQ.stride(0),
        YQ.stride(2),
        bias.stride(0) if bias is not None else 0,
        HAS_BIAS=bias is not None,
        DTYPE_MAX=DTYPE_MAX,
        FP8_TY=fp8_ty,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        QGROUP=group_size,
        NUM_BUFFERS=NUM_BUFFERS,
        K_GROUPS=K_GROUPS,
        BLOCKED_A=blocked_a,
        SHARED_A=shared_a,
        SHARED_B=shared_b,
        WMMA_LAYOUT=wmma_layout,
        OPA=opa,
        OPB=opb,
        QUANT_IN_DOT=QUANT_IN_DOT,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return YQ


def batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    splitK: int | None = None,
    YQ: torch.Tensor | None = None,
    transpose_bm: bool | None = False,
    transpose_bm_in: bool | None = False,
    config: dict | None = None,
    backend: str | None = None,
):
    """
    Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization.
    X is quantized to INT8 during computation using per-token grouped quantization.
    W is pre-quantized INT8 with per-batch-element scaling.

    Uses the gluon backend automatically on supported architectures (gfx1250)
    and the triton backend everywhere else. Pass ``backend`` to force a choice.

    Args:
        X (torch.Tensor): Higher precision input batch with shape (B, M, K) or (M, B, K) if transpose_bm_in=True.
            Quantized to INT8 on-the-fly during GEMM.
        WQ (torch.Tensor): Pre-quantized INT8 weight batch with shape (B, N, K), internally transposed.
        w_scale (torch.Tensor): Per-batch scale for WQ with shape (1,).
        group_size (int): Group size for per-token grouped quantization of X. Must be power of 2.
        bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        splitK (Optional[int]): Not supported. Must be None.
        YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N) or (M, B, N) if transpose_bm=True.
        transpose_bm (Optional[bool]): Transpose batch and M dimensions in output.
        transpose_bm_in (Optional[bool]): Transpose batch and M dimensions in input.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M).
        backend (Optional[str]): "triton", "gluon", or None (auto-detect; gluon
            wherever it is available).

    Returns:
        torch.Tensor: Output batch with shape (B, M, N) or (M, B, N) if transpose_bm=True.
    """

    # Check constraints.
    if not transpose_bm_in:
        B = X.shape[0]
        M = X.shape[1]
    else:
        M = X.shape[0]
        B = X.shape[1]
    K = X.shape[2]
    N = WQ.shape[1]

    assert B == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert K == WQ.shape[2], "Incompatible K dimensions!!!"
    assert (
        triton.next_power_of_2(group_size) == group_size
    ), "group_size mush be power of 2"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

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
        # The gluon kernel is fp8-only (it emits WMMA on fp8 operands).
        # gfx1250 uses the OCP fp8 encodings; anything else (int8, fnuz)
        # goes down the triton path.
        if WQ.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            _LOGGER.warning(
                f"BATCHED_GEMM_A8W8_PTG [gluon]: WQ dtype {WQ.dtype} is not fp8; "
                "falling back to the triton backend."
            )
            backend = "triton"

    has_bias = bias is not None
    if YQ is None:
        if transpose_bm:
            YQ = torch.empty((M, B, N), dtype=dtype, device=X.device)
        else:
            YQ = torch.empty((B, M, N), dtype=dtype, device=X.device)
    else:
        if transpose_bm:
            assert (
                YQ.shape[0] == M and YQ.shape[1] == B and YQ.shape[2] == N
            ), "Output dimension error"
        else:
            assert (
                YQ.shape[0] == B and YQ.shape[1] == M and YQ.shape[2] == N
            ), "Output dimension error"

    if backend == "gluon":
        _LOGGER.info(
            f"BATCHED_GEMM_A8W8_PTG [gluon/gfx1250]: X={tuple(X.shape)} WQ={tuple(WQ.shape)}"
        )
        return _batched_gemm_a8w8_ptg_gluon(
            X,
            WQ,
            w_scale,
            YQ,
            B,
            M,
            N,
            K,
            group_size,
            bias,
            bool(transpose_bm),
            bool(transpose_bm_in),
            config,
        )

    _LOGGER.info(
        f"BATCHED_GEMM_A8W8_PTG [triton]: X={tuple(X.shape)} WQ={tuple(WQ.shape)}"
    )

    WQ = WQ.transpose(1, 2)

    if config is None:
        config, _ = _get_config(M, N, K)
    else:
        config = dict(config)
    config["BLOCK_SIZE_K"] = group_size

    grid = lambda META: (
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    DTYPE_MAX = (
        torch.finfo(WQ.dtype).max
        if torch.is_floating_point(WQ)
        else torch.iinfo(WQ.dtype).max
    )

    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel[
        grid
    ](
        X,
        WQ,
        YQ,
        w_scale,
        bias,
        M,
        N,
        K,
        X.stride(0) if not transpose_bm_in else X.stride(1),
        X.stride(1) if not transpose_bm_in else X.stride(0),
        X.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0) if not transpose_bm else YQ.stride(1),
        YQ.stride(1) if not transpose_bm else YQ.stride(0),
        YQ.stride(2),
        bias.stride(0) if has_bias else 0,
        has_bias,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        **config,
    )

    return YQ
