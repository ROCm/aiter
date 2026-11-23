# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import copy

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import _fp8_format, _is_gluon_available
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_PRESHUFFLE_CONFIG = "BATCHED_GEMM-AFP8WFP8_PRESHUFFLED"
# n32k4; with 1x128 x scales and 128x128 w scales it only selects the compact w-scale path
_SCALE_MODE = 2


def batched_gemm_afp8wfp8_preshuffle(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    x_format: str | None = None,
    w_format: str | None = None,
    is_x_scale_transposed: bool = True,
    y_scales: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    gfx1250 (gluon) batched MXFP8 GEMM: Y[b] = X[b] @ W[b]^T with fp8 X and W, 1x128
    e8m0 scales on X and 128x128 e8m0 scales on W. Each operand can be a strided view
    with a unit last stride, for example a token-major (M, B, K) x as x.transpose(0, 1).
    Y can also be fp8 with 1x128 e8m0 scales, the format of X.

    Args:
        x (torch.Tensor): e4m3 (or uint8 view) activations, (B, M, K).
        w (torch.Tensor): e4m3 (or uint8 view) weights, each batch preshuffled with
            aiter.ops.shuffle.shuffle_weight and viewed as (B, N // 16, K * 16).
        x_scales (torch.Tensor): e8m0 (uint8) scales of x, one per 128 K elements:
            (B, K // 128, M) with unit M stride, or with is_x_scale_transposed=False
            (B, M, K // 128) with unit K stride.
        w_scales (torch.Tensor): e8m0 (uint8) scales of w, (B, ceil(N / 128), K // 128)
            with unit K stride.
        dtype (torch.dtype): output dtype (bf16, fp16 or float8_e4m3fn). An fp8 Y gets
            one e8m0 scale per 1x128 block. The kernel rounds Y to bf16 first, so the
            result is the same as per_group_quant_hip (group_size=128, e8m0 scales) on
            the bf16 Y.
        y (Optional[torch.Tensor]): preallocated (B, M, N) output with unit N stride.
            Its dtype replaces dtype.
        config (Optional[dict]): tuning keys (BLOCK_SIZE_M/N/K, num_warps, NUM_BUFFERS,
            num_ctas, GROUP_M, B2B_WMMA, NUM_KSUB, EPI_SPLIT); default from the
            BATCHED_GEMM-AFP8WFP8_PRESHUFFLED config files.
        x_format, w_format (str | None): "e4m3" or "e5m2"; None infers from the tensor
            dtype (float8_e4m3fn / float8_e5m2; uint8 views default to "e4m3").
        is_x_scale_transposed (bool): x_scales is (B, K // 128, M). False: x_scales is
            (B, M, K // 128).
        y_scales (Optional[torch.Tensor]): preallocated e8m0 (uint8) scales of an fp8 Y,
            (B, M, N // 128) with any strides. If None, the scales are a new
            (B, N // 128, M) tensor with unit M stride, returned as its (B, M, N // 128)
            transpose.

    Returns:
        y (torch.Tensor): (B, M, N) output. For an fp8 Y: the tuple (y, y_scales).
    """
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

    if not _is_gluon_available():
        raise RuntimeError("batched_gemm_afp8wfp8_preshuffle is gfx1250 (gluon) only")
    a_fmt = x_format or _fp8_format(x)
    b_fmt = w_format or _fp8_format(w)
    for f in (a_fmt, b_fmt):
        if f not in ("e4m3", "e5m2"):
            raise ValueError(f"fp8 format must be 'e4m3' or 'e5m2', got {f!r}")

    B, M, K = x.shape
    N = w.shape[1] * 16
    assert x.stride(2) == 1, f"x needs a unit K stride, got strides {x.stride()}"
    assert (
        w.shape == (B, N // 16, K * 16) and w.stride(2) == 1
    ), f"w must be shuffle_weight(w).view(B, N // 16, K * 16), got {tuple(w.shape)}"
    assert K % 128 == 0, f"K must be a multiple of 128, got {K}"

    if config is None:
        config, _ = get_gemm_config(_PRESHUFFLE_CONFIG, M, N, K, B=B, backend="gluon")
    else:
        config = copy.deepcopy(config)

    xs_shape = (B, K // 128, M) if is_x_scale_transposed else (B, M, K // 128)
    assert tuple(x_scales.shape) == xs_shape, (
        f"x_scales must be {xs_shape} for is_x_scale_transposed="
        f"{is_x_scale_transposed}, got {tuple(x_scales.shape)}"
    )
    if x_scales.shape[2] == 1 and x_scales.stride(2) != 1:
        # torch leaves any stride on a size-1 dim, but TDM needs unit inner stride
        padded = x_scales.new_empty((B, x_scales.shape[1], 16))
        padded[..., :1] = x_scales
        x_scales = padded[..., :1]
    assert (
        x_scales.stride(2) == 1
    ), f"x_scales needs a unit last stride, got strides {x_scales.stride()}"
    config["BLOCK_SIZE_K"], config["NUM_KSUB"] = 128, 1

    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp8 import (
        _gemm_mxfp8_preshuffle_kernel,
        cluster_shape,
        get_gemm_afp8wfp8_preshuffle_layouts,
    )

    if y is None:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)
    assert (
        tuple(y.shape) == (B, M, N) and y.stride(2) == 1
    ), f"y must be (B, M, N) with a unit N stride, got {tuple(y.shape)} {y.stride()}"
    quant_y = y.dtype == torch.float8_e4m3fn
    if y.element_size() == 1 and not quant_y:
        raise ValueError(f"an 8-bit y must be float8_e4m3fn, got {y.dtype}")
    if quant_y:
        assert N % 128 == 0, f"an fp8 y needs N % 128 == 0, got N={N}"
        if y_scales is None:
            y_scales = torch.empty(
                (B, N // 128, M), dtype=torch.uint8, device=x.device
            ).transpose(1, 2)
        ys_shape = (B, M, N // 128)
        assert tuple(y_scales.shape) == ys_shape and y_scales.element_size() == 1, (
            f"y_scales must be e8m0 {ys_shape}, got {y_scales.dtype} "
            f"{tuple(y_scales.shape)}"
        )
    elif y_scales is not None:
        raise ValueError(f"y_scales needs a float8_e4m3fn y, got {y.dtype}")
    config["NUM_BUFFERS"] = min(config["NUM_BUFFERS"], K // 128)

    num_ctas = config["num_ctas"]
    _, ctas_n = cluster_shape(num_ctas, config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    expected = (B, triton.cdiv(N, 128), K // 128)
    if tuple(w_scales.shape) != expected or w_scales.stride(2) != 1:
        raise ValueError(
            f"w_scales for 128x128 blocks must be {expected} with a unit last stride, "
            f"got {tuple(w_scales.shape)} strides {w_scales.stride()}"
        )
    if (config["BLOCK_SIZE_N"] // ctas_n) % 128:
        raise ValueError(
            f"128x128 w scales need a multiple of 128 columns per CTA tile; config "
            f"{config} has {config['BLOCK_SIZE_N'] // ctas_n}"
        )
    # The sliced epilogue stores through a (N / BNc, M, BNc) view of y, BNc = per-CTA N.
    if config.get("EPI_SPLIT", 1) > 1 and (
        N % (config["BLOCK_SIZE_N"] // ctas_n) != 0 or config["NUM_BUFFERS"] < 2
    ):
        config["EPI_SPLIT"] = 1
    if quant_y:
        # each epilogue slice must hold whole 1x128 scale blocks
        bnc = config["BLOCK_SIZE_N"] // ctas_n
        while (bnc // config.get("EPI_SPLIT", 1)) % 128:
            config["EPI_SPLIT"] //= 2

    layouts = get_gemm_afp8wfp8_preshuffle_layouts(
        config["num_warps"],
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        num_ctas,
        _SCALE_MODE,
        config["NUM_KSUB"],
        0,
        config.get("EPI_SPLIT", 1),
        128,
        (128, 128),
        a_scale_rowmajor=not is_x_scale_transposed,
        c_elem_bytes=y.element_size(),
    )
    _LOGGER.info(
        "BATCHED_GEMM_AFP8WFP8_PRESHUFFLE: x=%s w=%s config=%s",
        tuple(x.shape),
        tuple(w.shape),
        config,
    )
    x_k = x.view(torch.uint8)
    w_k = w.view(torch.uint8)
    xs, ws = x_scales.view(torch.uint8), w_scales.view(torch.uint8)
    c_scale_args = {}
    if quant_y:
        ys = y_scales.view(torch.uint8)
        c_scale_args = {
            "C_SCALE_GROUP": 128,
            "c_scale_ptr": ys,
            "stride_cs_m": ys.stride(1),
            "stride_cs_n": ys.stride(2),
            "stride_csb": ys.stride(0),
        }

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        B,
    )
    _gemm_mxfp8_preshuffle_kernel[grid](
        x_k,
        w_k,
        y,
        xs,
        ws,
        M,
        N,
        K,
        x_k.stride(1),
        x_k.stride(2),
        w_k.stride(1),
        w_k.stride(2),
        y.stride(1),
        y.stride(1),
        y.stride(2),
        xs.stride(1),
        xs.stride(2),
        ws.stride(1),
        ws.stride(2),
        x_k.stride(0),
        w_k.stride(0),
        y.stride(0),
        xs.stride(0),
        ws.stride(0),
        SCALE_MODE=_SCALE_MODE,
        A_DATA_PRESHUFFLE=0,
        A_FMT=a_fmt,
        B_FMT=b_fmt,
        A_SCALE_GROUP=128,
        W_SCALE_BN=128,
        W_SCALE_BK=128,
        BATCHED=1,
        A_SCALE_ROWMAJOR=int(not is_x_scale_transposed),
        A_SCALE_K_COLS=1 if is_x_scale_transposed else triton.next_power_of_2(K // 128),
        **c_scale_args,
        **config,
        **layouts,
    )
    return (y, y_scales) if quant_y else y
