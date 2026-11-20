# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import (
    _gemm_a8w8_kernel,
    _gemm_a8w8_preshuffle_kernel,
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

_GLUON_SUPPORTED_ARCHS = ("gfx950", "gfx1250", "gfx1250-strict")
_GLUON_PTPC_ARCHS = ("gfx1250", "gfx1250-strict")


def _scale_stride(scale: torch.Tensor, size: int) -> int:
    assert scale.numel() == size, f"scale has {scale.numel()} elements, expected {size}"
    for dim, extent in enumerate(scale.shape):
        if extent == size:
            return scale.stride(dim)
    return 1


def _gemm_a8w8_ptpc_gfx1250(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
    y: torch.Tensor | None,
    config: dict | None,
    skip_reduce: bool,
) -> torch.Tensor:
    """Launch the gfx1250 Gluon PTPC kernel: x is (M, K) FP8 and w is (N, K) FP8."""
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a8w8 import (
        _KERNEL_MAP,
        _MIN_BUFFERS,
        create_ptpc_layouts,
    )

    M, K = x.shape
    N, _ = w.shape
    assert (
        x.dtype == torch.float8_e4m3fn and w.dtype == torch.float8_e4m3fn
    ), "The gfx1250 Gluon kernel supports FP8 E4M3 inputs only"
    assert x.stride(1) == 1 and w.stride(1) == 1, "x and w must be contiguous along K"

    if config is None:
        config, _ = get_gemm_config("GEMM-A8W8", M, N, K, backend="gluon")
    kernel_type = config["kernel_type"]
    assert (
        kernel_type in _KERNEL_MAP
    ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP)}"

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    num_warps = config["num_warps"]
    num_ctas = config["num_ctas"]
    register_store = config["REGISTER_STORE"]

    # Split-K: each split does a whole number of BLOCK_K stages.
    k_stages = triton.cdiv(K, BLOCK_K)
    stages_per_split = triton.cdiv(k_stages, max(config["NUM_KSPLIT"], 1))
    splitk_block_size = stages_per_split * BLOCK_K
    num_ksplit = triton.cdiv(K, splitk_block_size)
    if num_ksplit == 1:
        splitk_block_size = 0

    # The pipeline cannot be deeper than the number of K stages of a split.
    num_buffers = min(config["NUM_BUFFERS"], stages_per_split)
    if num_buffers < _MIN_BUFFERS[kernel_type]:
        _LOGGER.warning(
            "GEMM_A8W8 [gluon/gfx1250]: kernel_type='%s' needs NUM_BUFFERS>=%s "
            "but K=%s, BLOCK_K=%s gives %s; uses kernel_type='bandwidth_bound'.",
            kernel_type,
            _MIN_BUFFERS[kernel_type],
            K,
            BLOCK_K,
            num_buffers,
        )
        kernel_type = "bandwidth_bound"
    # The coalesced register store has no fp32 layout. The sliced_k tile is already coalesced.
    if register_store in (2, 3) and (kernel_type == "sliced_k" or num_ksplit > 1):
        register_store = 1

    # Split-K writes fp32 partials, then a reduce kernel adds them and the bias.
    if num_ksplit > 1:
        c = torch.empty((num_ksplit, M, N), dtype=torch.float32, device=x.device)
    else:
        c = y if y is not None else torch.empty((M, N), dtype=dtype, device=x.device)

    layouts = create_ptpc_layouts(
        kernel_type,
        num_warps,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_ctas,
        config["WARPS_N"],
        register_store,
        config["CTAS_M"],
        config["PARTITION"],
    )
    _LOGGER.info(
        "GEMM_A8W8 [gluon/gfx1250]: x=%s w=%s kernel=%s",
        tuple(x.shape),
        tuple(w.shape),
        kernel_type,
    )

    if kernel_type == "bandwidth_bound":
        variant_args = {"NUM_KSUB": config["NUM_KSUB"]}
    else:
        variant_args = {"SLICED_K": kernel_type == "sliced_k"}
    x_scale_stride = _scale_stride(x_scale, M)
    w_scale_stride = _scale_stride(w_scale, N)
    grid_2d = num_ctas == 1 and num_ksplit == 1 and config["GROUP_SIZE_M"] == 0
    if grid_2d:
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    else:
        grid = (num_ksplit * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _KERNEL_MAP[kernel_type][grid](
        x,
        w,
        c,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        c.stride(-2),
        c.stride(-1),
        c.stride(0) if num_ksplit > 1 else 0,
        x_scale_stride,
        w_scale_stride,
        bias,
        _scale_stride(bias, N) if bias is not None else 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=num_buffers,
        NUM_CTAS=num_ctas,
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        FUSED_LOAD=config["FUSED_LOAD"] if num_warps >= 2 else 0,
        REGISTER_STORE=register_store,
        ADD_BIAS=bias is not None and num_ksplit == 1,
        NUM_KSPLIT=num_ksplit,
        SPLITK_BLOCK_SIZE=splitk_block_size,
        SCALE_TDM=(
            config["SCALE_TDM"]
            if num_ksplit == 1 and x_scale_stride == 1 and w_scale_stride == 1
            else 0
        ),
        GRID_2D=grid_2d,
        **variant_args,
        num_warps=num_warps,
        num_ctas=num_ctas,
        waves_per_eu=config["waves_per_eu"],
        **layouts,
    )

    if num_ksplit == 1 or skip_reduce:
        return c
    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    REDUCE_BLOCK = 32
    _gemm_splitk_reduce_kernel[
        (triton.cdiv(M, REDUCE_BLOCK), triton.cdiv(N, REDUCE_BLOCK))
    ](
        c,
        y,
        bias,
        M,
        N,
        c.stride(0),
        c.stride(1),
        c.stride(2),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE_M=REDUCE_BLOCK,
        BLOCK_SIZE_N=REDUCE_BLOCK,
        ACTUAL_KSPLIT=num_ksplit,
        MAX_KSPLIT=triton.next_power_of_2(num_ksplit),
        ADD_BIAS=bias is not None,
        activation="",
        use_activation=False,
        KERNEL_NAME="_gemm_a8w8_ptpc_reduce_kernel",
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
    backend: str = "triton",
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias.
    INT8 inputs are scaled back to higher precision using per-tensor scale factors.

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
        skip_reduce (Optional[bool]): Skip reduction of split-K partial results.
            Enables kernel fusion with downstream operations (FP8/FP4 quantization,
            RMSNorm). Returns shape (NUM_KSPLIT, M, N) instead of (M, N).
        backend (str): "triton" (default) or "gluon". On gfx1250 the gluon backend
            runs the PTPC kernel. Its limits are: FP8 E4M3 inputs, and x and w
            contiguous along K. N can be any size.

    Returns:
        torch.Tensor: Output with shape (M, N) or (NUM_KSPLIT, M, N) if skip_reduce=True.
    """

    _LOGGER.info(
        "GEMM_A8W8: x=%s w=%s x_scale=%s w_scale=%s",
        tuple(x.shape),
        tuple(w.shape),
        tuple(x_scale.shape),
        tuple(w_scale.shape),
    )

    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    M, K = x.shape
    N, K = w.shape

    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert (
            get_arch() in _GLUON_SUPPORTED_ARCHS
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        assert x.dtype == w.dtype, "Input types must be the same"
        if get_arch() in _GLUON_PTPC_ARCHS:
            return _gemm_a8w8_ptpc_gfx1250(
                x, w, x_scale, w_scale, bias, dtype, y, config, skip_reduce
            )

    w = w.T

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
            "GEMM_A8W8 [gluon/%s]: x=%s w=%s",
            get_arch(),
            tuple(x.shape),
            tuple(w.shape),
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
        NUM_XCDS=get_num_xcds(),
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
    skip_reduce: bool | None = False,
    backend: str = "triton",
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
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        skip_reduce (Optional[bool]): Skip reduction of split-K partial results.
            Enables kernel fusion with downstream operations (FP8/FP4 quantization,
            RMSNorm). Returns shape (NUM_KSPLIT, M, N) instead of (M, N).
        backend (str): "triton" (default) or "gluon".

    Returns:
        torch.Tensor: Output with shape (M, N) or (NUM_KSPLIT, M, N) if skip_reduce=True.
    """

    _LOGGER.info(
        "GEMM_A8W8 PRESHUFFLE: x=%s w=%s x_scale=%s w_scale=%s",
        tuple(x.shape),
        tuple(w.shape),
        tuple(x_scale.shape),
        tuple(w_scale.shape),
    )

    M, K = x.shape
    N, K = w.shape
    N = N * 16
    K = K // 16

    assert x.shape[1] == K, "Incompatible dimensions!!!"
    assert K % 32 == 0, "K must be a multiple of 32 for preshuffled weights"

    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert (
            get_arch() in _GLUON_SUPPORTED_ARCHS
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        assert x.dtype == w.dtype, "Input types must be the same"

    if config is None:
        if backend == "gluon":
            config, _ = get_gemm_config("GEMM-A8W8", M, N, K, backend="gluon")
        else:
            config, _ = _get_config(M, N, K, shuffle=True)

    if y is None and (config.get("NUM_KSPLIT", 1) == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if backend == "gluon":
        assert (
            K % config["BLOCK_SIZE_K"] == 0
        ), "K must be multiple of BLOCK_SIZE_K for preshuffling"

        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8 import (
            _gemm_a8w8_preshuffled_kernel as _gluon_gemm_a8w8_preshuffled_kernel,
        )

        _LOGGER.info(
            "GEMM_A8W8 PRESHUFFLE [gluon/%s]: x=%s w=%s",
            get_arch(),
            tuple(x.shape),
            tuple(w.shape),
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
    _gemm_a8w8_preshuffle_kernel[grid](
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
        NUM_XCDS=get_num_xcds(),
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
            KERNEL_NAME="_gemm_a8w8_preshuffle_reduce_kernel",
        )

    return y
