# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Aiter tensor/launch adapter for the unmodified pyhip blockscale kernel."""

import functools

import torch
from torch import Tensor

from .gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
    TILE_K,
    TILE_M,
    TILE_N,
    kernel_fits_shape,
    kernelInstance,
    kernels_by_name,
)


def is_supported(XQ: Tensor, WQ: Tensor, Out: Tensor, preshuffle_b: bool) -> bool:
    """Whether this backend can honor the call; tuned dispatch asserts this."""
    if XQ.ndim != 2 or WQ.ndim != 2 or Out.ndim != 2:
        return False
    M, K = XQ.shape
    N = WQ.shape[0]
    if WQ.shape[1] != K or tuple(Out.shape) != (M, N):
        return False
    if not XQ.is_cuda or WQ.device != XQ.device or Out.device != XQ.device:
        return False
    fp8_storage = (torch.float8_e4m3fn, torch.int8, torch.uint8)
    if XQ.dtype not in fp8_storage or WQ.dtype not in fp8_storage:
        return False
    if Out.dtype != torch.bfloat16:
        return False
    gfx = torch.cuda.get_device_properties(XQ.device).gcnArchName.split(":", 1)[0]
    return kernel_fits_shape(kernelInstance(preshuffle_b, False, False), M, N, K, gfx)


def _prepare_scales(
    x_scale: Tensor,
    w_scale: Tensor,
    M: int,
    N: int,
    K: int,
    preshuffle_b: bool,
) -> tuple[Tensor, Tensor]:
    """Normalize existing aiter scale contracts to the kernel's physical layout."""
    KB = K // 128
    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        raise ValueError("FlyDSL blockscale requires FP32 scales")
    if tuple(w_scale.shape) != ((N + 127) // 128, KB):
        raise ValueError("w_scale must have shape (ceil(N / 128), K / 128)")
    if preshuffle_b:
        if tuple(x_scale.shape) == (M, KB):
            if x_scale.is_contiguous():
                # per_group_quant_hip and the tuner return column-major bytes
                # in a tensor reshaped back to (M, KB). Do not transpose twice.
                sa = x_scale.view(-1)
            elif x_scale.stride(0) == 1:
                sa = x_scale.transpose(0, 1).contiguous().view(-1)
            else:
                raise ValueError("preshuffled x_scale must use column-major storage")
        elif tuple(x_scale.shape) == (KB, M):
            sa = x_scale.contiguous().view(-1)
        else:
            raise ValueError("preshuffled x_scale must have shape (M, KB) or (KB, M)")
    else:
        if tuple(x_scale.shape) != (M, KB):
            raise ValueError("x_scale must have shape (M, K / 128)")
        # Kept inside the timed call: plain-layout tuning must include this cost.
        sa = x_scale.transpose(0, 1).contiguous().view(-1)
    return sa, w_scale.contiguous().view(-1)


@functools.lru_cache(maxsize=1024)
def _compile_gemm(N: int, K: int, kernel_name: str, device_index: int):
    from .kernels.gemm_a8w8_blockscale_8wave import compile_gemm_fp8_8wave

    ki = kernels_by_name[kernel_name]
    launch = compile_gemm_fp8_8wave(
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        preshuffle_b=ki.preshuffle_b,
        with_scale=True,
        useTileDMA=ki.use_tile_dma,
        split_m=ki.split_m,
    )
    # Match the prototype's flyc.compile[{"opt_level": 2}] without a global knob.
    launch.compile_hints["opt_level"] = 2
    return launch


def run_gemm_a8w8_blockscale(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
    preshuffle_b: bool = False,
) -> Tensor:
    """Run the named candidate into ``Out``; never silently benchmark a fallback.

    Plain B: XQ[M,K], WQ[N,K], row-major x_scale[M,K/128].
    Preshuffled B: shuffle_weight(WQ, (16,16)) and column-major x_scale.
    Both use w_scale[ceil(N/128),K/128] and BF16 output on gfx950.
    """
    ki = kernels_by_name.get(kernel_name)
    if ki is None:
        raise ValueError(f"Unknown FlyDSL blockscale kernel: {kernel_name!r}")
    if ki.preshuffle_b != preshuffle_b:
        raise ValueError("FlyDSL blockscale kernelName does not match the B layout")
    if not is_supported(XQ, WQ, Out, preshuffle_b):
        raise ValueError("Unsupported gfx, shape or dtype for FlyDSL blockscale")
    if x_scale.device != XQ.device or w_scale.device != XQ.device:
        raise ValueError("FlyDSL blockscale scales must be on the input device")

    import flydsl.expr as fx

    from .kernels.tensor_shim import _run_compiled

    M, K = XQ.shape
    N = WQ.shape[0]
    with torch.cuda.device(XQ.device):
        sa, sb = _prepare_scales(x_scale, w_scale, M, N, K, preshuffle_b)
        launch = _compile_gemm(N, K, kernel_name, XQ.device.index)
        out_contiguous = Out.contiguous()
        _run_compiled(
            launch,
            XQ.contiguous().view(torch.int8).view(-1),
            WQ.contiguous().view(torch.int8).view(-1),
            out_contiguous.view(-1),
            sa,
            sb,
            M,
            fx.Stream(torch.cuda.current_stream(XQ.device)),
        )
        if out_contiguous is not Out:
            Out.copy_(out_contiguous)
    return Out
