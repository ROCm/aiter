# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import (
    _gemm_a8w8_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import (
    get_scaled_dot_format_string,
    torch_to_triton_dtype,
)

_LOGGER = AiterTritonLogger()

# Architectures that ship a gluon a8w8 kernel. They are *different* kernels:
#   gfx950  -> _gluon_kernels/gfx950/gemm/basic/gemm_a8w8.py  (MFMA + preshuffle)
#   gfx1250 -> _gluon_kernels/gfx1250/gemm/basic/gemm_a8w8.py (TDM + WMMA)
_GLUON_SUPPORTED_ARCHS = ("gfx950", "gfx1250")

# Preshuffled weights are only implemented by the gfx950 gluon kernel.
_GLUON_PRESHUFFLE_ARCHS = ("gfx950",)

# The gfx1250 gluon kernel is an fp8 WMMA kernel; int8 has no gfx1250 WMMA
# path here, so those calls stay on triton.
_GFX1250_GLUON_IN_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_gluon_available():
    """Check if the gluon backend is available for the current GPU architecture."""
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def _gfx1250_gluon_unsupported_reason(x, w, y, skip_reduce):
    """Why the gfx1250 gluon kernel cannot serve this call, or None if it can.

    Auto-dispatch (``backend=None``) falls back to triton on a non-None
    reason; an explicit ``backend="gluon"`` raises instead.
    """
    if x.dtype not in _GFX1250_GLUON_IN_DTYPES:
        return f"input dtype {x.dtype} (gfx1250 gluon a8w8 is fp8-only)"
    if w.dtype != x.dtype:
        return f"mixed input dtypes x={x.dtype} w={w.dtype}"
    if skip_reduce:
        return "skip_reduce=True is a triton-only split-K feature"
    # The WMMA instruction is 16x16x128, and the TDM pipeline needs at least
    # double buffering, so the K loop needs >= 2 tiles of >= 128 elements.
    if x.shape[1] <= 128:
        return f"K={x.shape[1]} is too small (needs >= 2 tiles of BLOCK_K >= 128)"
    if x.stride(1) != 1:
        return f"x must be K-contiguous, got strides {x.stride()}"
    if w.stride(1) != 1:
        return f"w must be K-contiguous, got strides {w.stride()}"
    if y is not None and y.stride(1) != 1:
        return f"y must be N-contiguous, got strides {y.stride()}"
    return None


def _gemm_a8w8_gluon_gfx1250(x, w, x_scale, w_scale, bias, dtype, y, config):
    """Launch the gfx1250 TDM/WMMA gluon a8w8 kernel.

    ``x``: (M, K) fp8, ``w``: (N, K) fp8 (NOT pre-transposed),
    ``x_scale``: (M,) or (M, 1), ``w_scale``: (N,) or (1, N).
    """
    import triton.experimental.gluon.language as gl

    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a8w8 import (
        _DEPTH_SLACK,
        _KERNEL_MAP,
        _MIN_BUFFERS,
        DEFAULT_CONFIG,
        _pad_interval,
        create_shared_layouts,
        create_wmma_layouts,
    )

    M, K = x.shape
    N, _ = w.shape

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    num_warps = cfg["num_warps"]
    kernel_type = cfg["kernel_type"]
    assert kernel_type in _KERNEL_MAP, f"Unknown kernel_type '{kernel_type}'"

    NUM_KSPLIT = cfg.get("NUM_KSPLIT", 1)
    if NUM_KSPLIT > 1:
        # Each partition owns a whole number of BLOCK_K tiles.
        SPLITK_BLOCK_SIZE = triton.cdiv(triton.cdiv(K, NUM_KSPLIT), BLOCK_K) * BLOCK_K
        ACTUAL_KSPLIT = triton.cdiv(K, SPLITK_BLOCK_SIZE)
    else:
        SPLITK_BLOCK_SIZE = K
        ACTUAL_KSPLIT = 1

    # The pipeline prologue/epilogue walk a fixed NUM_BUFFERS tiles, so the
    # depth must fit the SHORTEST split-K partition -- the last one can be
    # shorter than SPLITK_BLOCK_SIZE when K does not divide evenly.
    last_K = K - (ACTUAL_KSPLIT - 1) * SPLITK_BLOCK_SIZE
    num_k_tiles = min(
        triton.cdiv(SPLITK_BLOCK_SIZE, BLOCK_K), triton.cdiv(last_K, BLOCK_K)
    )
    depth_cap = num_k_tiles - _DEPTH_SLACK[kernel_type]
    if depth_cap < _MIN_BUFFERS[kernel_type]:
        # Not enough K tiles for the requested variant's reach; the
        # bandwidth_bound variant needs the least.
        kernel_type = "bandwidth_bound"
        depth_cap = num_k_tiles
    if depth_cap < _MIN_BUFFERS[kernel_type]:
        # Shrink BLOCK_K until the shortest partition has enough tiles.
        while BLOCK_K > 128 and depth_cap < _MIN_BUFFERS[kernel_type]:
            BLOCK_K //= 2
            num_k_tiles = min(
                triton.cdiv(SPLITK_BLOCK_SIZE, BLOCK_K), triton.cdiv(last_K, BLOCK_K)
            )
            depth_cap = num_k_tiles
    if depth_cap < _MIN_BUFFERS[kernel_type]:
        raise ValueError(
            f"GEMM_A8W8 [gluon/gfx1250]: not enough K tiles for the pipeline: "
            f"K={K} BLOCK_K={BLOCK_K} NUM_KSPLIT={ACTUAL_KSPLIT} gives "
            f"{num_k_tiles} tile(s) in the shortest partition, need >= "
            f"{_MIN_BUFFERS[kernel_type]}"
        )
    NUM_BUFFERS = max(_MIN_BUFFERS[kernel_type], min(cfg["NUM_BUFFERS"], depth_cap))

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if ACTUAL_KSPLIT > 1:
        y_pp = torch.empty((ACTUAL_KSPLIT, M, N), dtype=torch.float32, device=x.device)
        out_t = y_pp
        stride_ck = y_pp.stride(0)
        stride_cm, stride_cn = y_pp.stride(1), y_pp.stride(2)
    else:
        out_t = y
        stride_ck = 0
        stride_cm, stride_cn = y.stride(0), y.stride(1)

    x_scale = x_scale.reshape(-1)
    w_scale = w_scale.reshape(-1)

    wmma_layout, operand_a, operand_b = create_wmma_layouts(
        num_warps, warps_n=cfg["WARPS_N"], instr_k=cfg["INSTR_K"]
    )
    shared_a, shared_b = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K)
    shared_c = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BLOCK_N, out_t.element_size() * 8), 8]],
        [BLOCK_M, BLOCK_N],
        [1, 0],
    )

    grid = (
        ACTUAL_KSPLIT * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        1,
    )

    _KERNEL_MAP[kernel_type][grid](
        x,
        w,
        out_t,
        x_scale,
        w_scale,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        stride_ck,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_KSPLIT=ACTUAL_KSPLIT,
        SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        WMMA_LAYOUT=wmma_layout,
        OPERAND_LAYOUT_A=operand_a,
        OPERAND_LAYOUT_B=operand_b,
        SHARED_LAYOUT_C=shared_c,
        ADD_BIAS=(bias is not None) and ACTUAL_KSPLIT == 1,
        USE_TDM_STORE=cfg["USE_TDM_STORE"],
        num_warps=num_warps,
    )

    if ACTUAL_KSPLIT > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
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
            BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
            BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT=ACTUAL_KSPLIT,
            MAX_KSPLIT=triton.next_power_of_2(ACTUAL_KSPLIT),
            ADD_BIAS=bias is not None,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_a8w8_reduce_kernel",
        )

    return y


def gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    backend: str | None = None,
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias.
    INT8 inputs are scaled back to higher precision using per-tensor scale factors.

    Uses the gluon backend automatically on supported architectures
    (gfx950, gfx1250) and the triton backend everywhere else. Pass ``backend``
    to force a choice.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        skip_reduce (Optional[bool]): [triton only] Skip reduction of split-K partial
            results. Enables kernel fusion with downstream operations (FP8/FP4
            quantization, RMSNorm). Returns shape (NUM_KSPLIT, M, N) instead of (M, N).
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output with shape (M, N) or (NUM_KSPLIT, M, N) if skip_reduce=True.
    """

    _LOGGER.info(
        f"GEMM_A8W8: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    auto_backend = backend is None
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

    # gfx1250 has its own gluon kernel that takes w untransposed and only
    # covers fp8; decide before the shared `w = w.T` below.
    if backend == "gluon" and "gfx1250" in get_arch():
        reason = _gfx1250_gluon_unsupported_reason(x, w, y, skip_reduce)
        if reason is None:
            M, K = x.shape
            N, _ = w.shape
            if config is None:
                config, _ = get_gemm_config("GEMM-A8W8", M, N, K, backend="gluon")
            _LOGGER.info(
                f"GEMM_A8W8 [gluon/gfx1250]: x={tuple(x.shape)} w={tuple(w.shape)}"
            )
            return _gemm_a8w8_gluon_gfx1250(
                x, w, x_scale, w_scale, bias, dtype, y, config
            )
        if not auto_backend:
            raise ValueError(
                f"gfx1250 gluon a8w8 does not support this call: {reason}"
            )
        _LOGGER.info(
            f"GEMM_A8W8: falling back to the triton backend on gfx1250 ({reason})"
        )
        backend = "triton"

    M, K = x.shape
    N, K = w.shape

    w = w.T

    if backend == "gluon":
        assert x.dtype == w.dtype, "Input types must be the same"

    if config is None:
        if backend == "gluon":
            config, _ = get_gemm_config("GEMM-A8W8", M, N, K, backend="gluon")
        else:
            config, _ = _get_config(M, N, K)
    if y is None and (config.get("NUM_KSPLIT", 1) == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if backend == "gluon":
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8 import (
            _gemm_a8w8_kernel as _gluon_gemm_a8w8_kernel,
        )

        _LOGGER.info(
            f"GEMM_A8W8 [gluon/{get_arch()}]: x={tuple(x.shape)} w={tuple(w.shape)}"
        )

        fp8_format = (
            None
            if x.dtype == torch.int8
            else get_scaled_dot_format_string(torch_to_triton_dtype[x.dtype])
        )
        grid = (
            triton.cdiv(M, config["BLOCK_SIZE_M"])
            * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        )
        _gluon_gemm_a8w8_kernel[grid](
            x,
            w,
            x_scale,
            w_scale,
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
            bias is not None,
            NUM_XCDS=get_num_xcds(),
            NUM_WARPS=config["num_warps"],
            **config,
            FP8_FORMAT=fp8_format,
        )
        return y

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=y.device if y is not None else x.device,
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
    _gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
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
        (bias is not None) and (config["NUM_KSPLIT"] == 1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
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
            BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
            BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT=ACTUAL_KSPLIT,
            MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=bias is not None,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_a8w8_reduce_kernel",
        )

    return y


def gemm_a8w8_preshuffle(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias,
    taking weights in a pre-shuffled layout for better memory access.

    Args:
        x (torch.Tensor): INT8/FP8 input matrix with shape (M, K).
        w (torch.Tensor): INT8/FP8 weight matrix pre-shuffled to (N*16, K//16),
            internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        torch.Tensor: Output with shape (M, N) in higher precision format.
    """
    assert (
        get_arch() in _GLUON_PRESHUFFLE_ARCHS
    ), f"gemm_a8w8_preshuffle requires one of {_GLUON_PRESHUFFLE_ARCHS}, got '{get_arch()}'"
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8 import (
        _gemm_a8w8_preshuffled_kernel as _gluon_gemm_a8w8_preshuffled_kernel,
    )

    _LOGGER.info(
        f"GEMM_A8W8 PRESHUFFLE [gluon/{get_arch()}]: x={tuple(x.shape)} w={tuple(w.shape)}"
    )

    M, K = x.shape
    N, K = w.shape
    N = N * 16
    K = K // 16

    if config is None:
        config, _ = get_gemm_config("GEMM-A8W8", M, N, K, backend="gluon")

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    assert (
        K % config["BLOCK_SIZE_K"] == 0
    ), "K must be multiple of BLOCK_SIZE_K for preshuffling"

    fp8_format = (
        None
        if x.dtype == torch.int8
        else get_scaled_dot_format_string(torch_to_triton_dtype[x.dtype])
    )
    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gluon_gemm_a8w8_preshuffled_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
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
        bias is not None,
        NUM_XCDS=get_num_xcds(),
        NUM_WARPS=config["num_warps"],
        **config,
        FP8_FORMAT=fp8_format,
    )

    return y
