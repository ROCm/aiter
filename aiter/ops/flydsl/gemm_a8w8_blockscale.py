# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Aiter adapter for the rebased half-M, raw-DMA FP8 blockscale kernel."""

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

#: M at or below which the 8-wave prefill tile computes mostly padding; the
#: decode kernel splits N and K instead.
DECODE_M_MAX = 32
#: Smallest K with a measured op-level gain.  At K=512 the routed op is flat
#: (0.99-1.03x over 1..32 rows at N=320/512; graph, A/A 1.00), so those calls
#: keep the existing backend.
DECODE_K_MIN = 1024
#: Largest K with no measured loss against the tuned CK/asm/Triton rows.  At
#: K=6144, 30 of 96 sweep points are 1-3% slower (worst re-measured median
#: 0.98x), so those rows keep their kernel; every production decode shape here
#: has K <= 4096.
DECODE_K_MAX = 4096
DECODE_N = 16


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
    if Out.dtype not in (torch.bfloat16, torch.float16):
        return False
    gfx = torch.cuda.get_device_properties(XQ.device).gcnArchName.split(":", 1)[0]
    return kernel_fits_shape(kernelInstance(preshuffle_b), M, N, K, gfx)


def decode_supported(XQ: Tensor, WQ: Tensor, Out: Tensor, preshuffle_b: bool) -> bool:
    """Whether the small-M decode kernel can honor this call on gfx950.

    The decode kernel keeps the 8-wave kernel's scale and dtype contract; it
    only changes the output tiling.  Its FP8 fragment layout needs N % 16 and
    its 128-element K blocks need K % 256, both already true for every tuned
    blockscale row of this backend.
    """
    if preshuffle_b or not is_supported(XQ, WQ, Out, preshuffle_b):
        return False
    M, K = XQ.shape
    return (
        M <= DECODE_M_MAX
        and DECODE_K_MIN <= K <= DECODE_K_MAX
        and K % 256 == 0
        and WQ.shape[0] % 16 == 0
    )


def _prepare_decode_scale(x_scale: Tensor, w_scale: Tensor, M: int, N: int, K: int):
    """The decode kernel's scales, without a transpose or copy.

    The routed op's plain-scale contract is vLLM's row-major ``[M, K/128]``
    x_scale; the decode kernel reads exactly that layout.  The 8-wave path's
    ``_prepare_scales`` keeps its own ``[K/128, M]`` contract unchanged.
    """
    KB = K // 128
    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        raise ValueError("FlyDSL blockscale requires FP32 scales")
    if tuple(w_scale.shape) != ((N + 127) // 128, KB):
        raise ValueError("w_scale must have shape (ceil(N / 128), K / 128)")
    if x_scale.nelement() != M * KB:
        raise ValueError("x_scale tensor must have the correct number of elements")
    if tuple(x_scale.shape) != (M, KB):
        x_scale = x_scale.view(M, KB)
    if not x_scale.is_contiguous():
        raise ValueError("decode x_scale must be row-major contiguous (M, K / 128)")
    return x_scale.view(-1), w_scale.contiguous().view(-1)


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


@aot_factory("gemm_a8w8_blockscale")
@functools.lru_cache(maxsize=1024)
def _compile_gemm(
    N: int,
    K: int,
    kernel_name: str,
    device: torch.device,
    out_dtype: torch.dtype = torch.bfloat16,
):
    from .kernels.gemm_a8w8_blockscale_8wave import compile_gemm_fp8_8wave

    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FlyDSL blockscale requires BF16 or FP16 output")
    # The output tensor specializes the epilogue. Keep separate launchers since
    # each caches the first typed CompiledFunction it compiles.
    ki = kernels_by_name[kernel_name]
    launch = compile_gemm_fp8_8wave(
        TILE_M,
        TILE_N,
        TILE_K,
        N,
        K,
        preshuffle_b=ki.preshuffle_b,
    )
    # Match the prototype's flyc.compile[{"opt_level": 2}] without a global knob.
    launch.compile_hints["opt_level"] = 2
    return launch


@aot_factory("gemm_a8w8_blockscale")
@functools.lru_cache(maxsize=1024)
def _compile_decode_gemm(
    N: int,
    K: int,
    tile_m: int,
    device: torch.device,
    out_dtype: torch.dtype = torch.bfloat16,
):
    import flydsl.expr as fx

    from .kernels.gemm_a8w8_blockscale_decode import (
        compile_gemm_fp8_decode,
        decode_k_waves,
        decode_waves_per_eu,
    )

    if out_dtype == torch.bfloat16:
        elem = fx.BFloat16
    elif out_dtype == torch.float16:
        elem = fx.Float16
    else:
        raise ValueError("FlyDSL blockscale requires BF16 or FP16 output")
    waves = decode_k_waves(N, K, tile_m, DECODE_N)
    launch = compile_gemm_fp8_decode(
        N,
        K,
        tile_m,
        DECODE_N,
        elem,
        num_stages=4,
        k_waves=waves,
    )
    launch.compile_hints["opt_level"] = 2
    # The heuristic and this hint agree on the wave count (compile_hints are
    # not part of the launcher's cache key, so keep them a pure function of
    # the arguments).  ``waves_per_eu`` is FlyDSL's supported occupancy knob;
    # ``maxnreg`` is rejected by this version.
    launch.compile_hints["waves_per_eu"] = decode_waves_per_eu(waves)
    return launch


def _decode_tile_m(M: int) -> int:
    return 32 if M > 16 else 16


def _run_launcher(
    launch,
    XQ: Tensor,
    WQ: Tensor,
    Out: Tensor,
    sa: Tensor,
    sb: Tensor,
    M: int,
) -> Tensor:
    from .kernels.tensor_shim import _run_compiled, current_stream_arg

    out_contiguous = Out.contiguous()
    # Keep matrix dimensions separate in the launch ABI: flattening a
    # >2**31-element matrix overflows a dynamic i32 FlyDSL extent.
    _run_compiled(
        launch,
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


def run_gemm_a8w8_blockscale_decode(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
) -> Tensor:
    """Run the small-M plain-weight decode kernel into ``Out``; never fall back silently."""
    if not decode_supported(XQ, WQ, Out, False):
        raise ValueError("Unsupported gfx, shape or dtype for FlyDSL decode GEMM")
    if x_scale.device != XQ.device or w_scale.device != XQ.device:
        raise ValueError("FlyDSL blockscale scales must be on the input device")

    M, K = XQ.shape
    N = WQ.shape[0]
    with torch.cuda.device(XQ.device):
        sa, sb = _prepare_decode_scale(x_scale, w_scale, M, N, K)
        launch = _compile_decode_gemm.lazy(
            N, K, _decode_tile_m(M), XQ.device, Out.dtype
        )
        return _run_launcher(launch, XQ, WQ, Out, sa, sb, M)


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
    Both use w_scale[ceil(N/128),K/128] and BF16/FP16 output on gfx950.
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

    M, K = XQ.shape
    N = WQ.shape[0]
    with torch.cuda.device(XQ.device):
        sa, sb = _prepare_scales(x_scale, w_scale, M, N, K, preshuffle_b)
        # Every tensor is contiguous and only Out's dtype varies, so the
        # compile arguments determine the launcher's argument signature. The
        # device keeps one launcher per device on the JIT path; on disk the
        # entry is shared by every device.
        launch = _compile_gemm.lazy(N, K, kernel_name, XQ.device, Out.dtype)
        return _run_launcher(launch, XQ, WQ, Out, sa, sb, M)
