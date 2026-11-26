# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Aiter adapter for the FlyDSL FP8 blockscale kernels.

IDs 0/1: the gfx950 rebased half-M, raw-DMA kernel (BF16 or FP16 output).
IDs 2/3: the gfx942 2x4-LDS pf1 kernel ported from PyHIP (BF16 or FP16 output).
"""

import functools

import torch
from torch import Tensor

from .aot_launch import aot_factory
from .gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
    TILE_K,
    TILE_M,
    TILE_N,
    kernel_fits_shape,
    kernelInstance,
    kernels_by_name,
)

# The FP8 format each kernel's MFMAs read; int8/uint8 are raw storage of it.
_FP8_DTYPE = {"gfx950": torch.float8_e4m3fn, "gfx942": torch.float8_e4m3fnuz}


def is_supported(
    XQ: Tensor,
    WQ: Tensor,
    Out: Tensor,
    preshuffle_b: bool,
    kernel_name: str | None = None,
) -> bool:
    """Whether the named kernel can honor the call; tuned dispatch asserts this.

    Without ``kernel_name`` it asks about the gfx950 kernel of this B layout.
    """
    if kernel_name is None:
        ker_obj = kernelInstance(preshuffle_b, "gfx950")
    else:
        ker_obj = kernels_by_name.get(kernel_name)
        if ker_obj is None or ker_obj.preshuffle_b != preshuffle_b:
            return False
    if XQ.ndim != 2 or WQ.ndim != 2 or Out.ndim != 2:
        return False
    M, K = XQ.shape
    N = WQ.shape[0]
    if WQ.shape[1] != K or tuple(Out.shape) != (M, N):
        return False
    if not XQ.is_cuda or WQ.device != XQ.device or Out.device != XQ.device:
        return False
    fp8_storage = (_FP8_DTYPE[ker_obj.gfx], torch.int8, torch.uint8)
    if XQ.dtype not in fp8_storage or WQ.dtype not in fp8_storage:
        return False
    if Out.dtype not in (torch.bfloat16, torch.float16):
        return False
    gfx = torch.cuda.get_device_properties(XQ.device).gcnArchName.split(":", 1)[0]
    return kernel_fits_shape(ker_obj, M, N, K, gfx)


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
    if x_scale.nelement() != M * KB:
        raise ValueError("x_scale tensor must have the correct number of elements")
    # X scale must be [KB, M] layout, M is the innner-most dimension not matter preshuffle_b is True or False.
    # The gemm op A8w8 blockscale  has implicit rule:
    # x_scale =  orignal_scale.tranposed(0,1).contineous() if preshuffle_b else orignal_scale
    # here let us assume x_scale already transposed when preshuffle_b and plain when not preshuffle_b
    if not preshuffle_b:
        x_scale = x_scale.view(M, KB)
        x_scale = x_scale.transpose(0, 1).contiguous()
    elif x_scale.shape == (M, KB) and x_scale.stride() == (1, M):
        # when preshuffled in sglang, x_scale already tranposed in buffer but view as [M, KB]: [1, M]
        # q_scale = aiter_per1x128_quant(x, transpose_scale=True) ## [KB, M]:[M, 1]
        # q_scale = q_scale.view(M, BK) [M, KB]:[KB, 1] ##buffer tranposed but view as plain buffer.
        # x_scale = torch.as_strided(q_scale, (M, KB), (1, M)) ##[M, KB]: [1, M]

        # [M, KB]: [1, M] not continguous, here just need to do transpose on sglang scale tensor
        x_scale = x_scale.T.contiguous()
    elif not x_scale.is_contiguous():
        raise ValueError(
            "preshuffle x_scale must be contiguous [K/128, M] bytes "
            "or an (M, K/128) view with stride (1, M)"
        )
    sa = x_scale.view(-1)
    return sa, w_scale.contiguous().view(-1)


@aot_factory(
    "gemm_a8w8_blockscale",
    sources=[
        "gemm_tune/flydsl_gemm_a8w8_blockscale_common.py",
        "kernels/gemm_a8w8_blockscale_8wave_gfx942.py",
        "kernels/gemm_a8w8_blockscale_8wave_gfx950.py",
    ],
)
@functools.lru_cache(maxsize=1024)
def _compile_gemm(
    N: int,
    K: int,
    kernel_name: str,
    device: torch.device,
    out_dtype: torch.dtype = torch.bfloat16,
):
    ker_obj = kernels_by_name[kernel_name]
    if ker_obj.gfx == "gfx942":
        from .kernels.gemm_a8w8_blockscale_8wave_gfx942 import (
            compile_gemm_fp8_blockscale_gfx942 as compile_gemm_fp8_blockscale,
        )
    else:
        from .kernels.gemm_a8w8_blockscale_8wave_gfx950 import (
            compile_gemm_fp8_blockscale_gfx950 as compile_gemm_fp8_blockscale,
        )

    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FlyDSL blockscale requires BF16 or FP16 output")
    # The output tensor specializes the epilogue. Keep separate launchers since
    # each caches the first typed CompiledFunction it compiles.
    launch = compile_gemm_fp8_blockscale(
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        preshuffle_b=ker_obj.preshuffle_b,
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
    Both use w_scale[ceil(N/128),K/128] and BF16/FP16 output; IDs 0/1 run on
    gfx950, IDs 2/3 on gfx942.
    """
    ker_obj = kernels_by_name.get(kernel_name)
    if ker_obj is None:
        raise ValueError(f"Unknown FlyDSL blockscale kernel: {kernel_name!r}")
    if ker_obj.preshuffle_b != preshuffle_b:
        raise ValueError("FlyDSL blockscale kernelName does not match the B layout")
    if not is_supported(XQ, WQ, Out, preshuffle_b, kernel_name):
        raise ValueError("Unsupported gfx, shape or dtype for FlyDSL blockscale")
    if x_scale.device != XQ.device or w_scale.device != XQ.device:
        raise ValueError("FlyDSL blockscale scales must be on the input device")

    from .kernels.tensor_shim import _run_compiled, current_stream_arg

    M, K = XQ.shape
    N = WQ.shape[0]
    with torch.cuda.device(XQ.device):
        sa, sb = _prepare_scales(x_scale, w_scale, M, N, K, preshuffle_b)
        out_contiguous = Out.contiguous()
        # Every tensor is contiguous and only Out's dtype varies, so the
        # compile arguments determine the launcher's argument signature. The
        # device keeps one launcher per device on the JIT path; on disk the
        # entry is shared by every device.
        # Keep matrix dimensions separate in the launch ABI: flattening a
        # >2**31-element matrix overflows a dynamic i32 FlyDSL extent.
        _run_compiled(
            _compile_gemm.lazy(N, K, kernel_name, XQ.device, Out.dtype),
            XQ.contiguous().view(torch.int8),
            WQ.contiguous().view(torch.int8),
            out_contiguous,
            sa,
            sb,
            M,
            current_stream_arg(XQ.device),
        )
        if out_contiguous is not Out:
            Out.copy_(out_contiguous)
    return Out
