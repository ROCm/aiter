# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _convert_from_mxfp8_kernel,
    _convert_to_mxfp8_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "convert_from_mxfp8",
    "convert_to_mxfp8",
]

_LOGGER = AiterTritonLogger()


def convert_to_mxfp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    quant_block_size: int = 32,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: bool | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MXFP8 format.

    Args:
        x: Input tensor of shape (M, N), dtype float32 or bfloat16.
        fp8_dtype: Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2).
        quant_block_size: Block size for quantization scaling (default 32).
        is_2d_block: Whether to use 2D block scaling.
        use_sr: Whether to use stochastic rounding.
        use_asm: Whether to use the inline-assembly path. The ASM instructions
            (``v_cvt_scalef32_*``) are gfx950-only; ``None`` (default)
            auto-selects ASM on gfx950 and the portable path elsewhere.
        block_m: Tile size along M dimension.
        block_n: Tile size along N dimension.

    Returns:
        Tuple of (quantized_tensor, scales) where quantized_tensor has fp8_dtype
        and scales has uint8 dtype with e8m0 format.
    """
    _LOGGER.info(f"CONVERT_TO_MXFP8: x={tuple(x.shape)}")
    # The ASM path (v_cvt_scalef32_*) is gfx950-only and _pack_fp8 only accepts
    # e4m3fn; anything else must take the portable path.
    asm_supported = (
        arch_info.get_arch() == "gfx950" and fp8_dtype == torch.float8_e4m3fn
    )
    if use_asm is None:
        use_asm = asm_supported
    elif use_asm and not asm_supported:
        raise ValueError(
            "use_asm=True requires gfx950 and fp8_dtype=torch.float8_e4m3fn"
        )
    # Stochastic rounding on the ASM path passes mismatched RNG/operand shapes
    # and cannot compile; only the portable path supports it for now.
    if use_sr and use_asm:
        raise ValueError("use_sr=True is not supported with use_asm=True")
    assert x.ndim == 2, "Input must be 2D"
    M, N = x.shape
    # The kernel loads/stores full [block_m, block_n] tiles without masking and
    # reshapes each into QUANT_BLOCK_SIZE groups, so M and N must tile exactly.
    assert M % block_m == 0 and N % block_n == 0, (
        f"MXFP8 convert requires M,N aligned to tile ({block_m},{block_n}); "
        f"got ({M},{N})"
    )

    y = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    if is_2d_block:
        scale_m = triton.cdiv(M, quant_block_size)
    else:
        scale_m = M
    scale_n = triton.cdiv(N, quant_block_size)
    s = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _convert_to_mxfp8_kernel[grid](
        x,
        y,
        s,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        s.stride(0),
        s.stride(1),
        0,
        0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )
    return y, s


def convert_from_mxfp8(
    x: torch.Tensor,
    s: torch.Tensor,
    output_dtype: torch.dtype,
    quant_block_size: int = 32,
    is_2d_block: bool = False,
    use_asm: bool | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    Dequantize a tensor from MXFP8 format.

    Args:
        x: Quantized input tensor of shape (M, N), dtype float8.
        s: Scale tensor with uint8 dtype (e8m0 format).
        output_dtype: Target output dtype (torch.float32 or torch.bfloat16).
        quant_block_size: Block size for quantization scaling (default 32).
        is_2d_block: Whether 2D block scaling was used.
        use_asm: Whether to use the inline-assembly path. The ASM instructions
            (``v_cvt_scalef32_*``) are gfx950-only; ``None`` (default)
            auto-selects ASM on gfx950 and the portable path elsewhere.
        block_m: Tile size along M dimension.
        block_n: Tile size along N dimension.

    Returns:
        Dequantized tensor with output_dtype.
    """
    _LOGGER.info(f"CONVERT_FROM_MXFP8: x={tuple(x.shape)}")
    # The ASM path (v_cvt_scalef32_*) is gfx950-only and _unpack_fp8 only accepts
    # e4m3fn input; anything else must take the portable path.
    asm_supported = arch_info.get_arch() == "gfx950" and x.dtype == torch.float8_e4m3fn
    if use_asm is None:
        use_asm = asm_supported
    elif use_asm and not asm_supported:
        raise ValueError("use_asm=True requires gfx950 and x.dtype=torch.float8_e4m3fn")
    assert x.ndim == 2, "Input must be 2D"
    M, N = x.shape
    # Kernel loads/stores full [block_m, block_n] tiles without masking.
    assert M % block_m == 0 and N % block_n == 0, (
        f"MXFP8 convert requires M,N aligned to tile ({block_m},{block_n}); "
        f"got ({M},{N})"
    )
    # The scale tensor is read with unmasked full-tile loads, so its shape must
    # match exactly what convert_to_mxfp8 produced.
    expected_scale_shape = (
        triton.cdiv(M, quant_block_size) if is_2d_block else M,
        triton.cdiv(N, quant_block_size),
    )
    assert (
        tuple(s.shape) == expected_scale_shape
    ), f"scale shape {tuple(s.shape)} != {expected_scale_shape}"
    assert s.dtype == torch.uint8, f"scale dtype must be uint8, got {s.dtype}"
    assert s.device == x.device, "input and scale must be on the same device"

    y = torch.empty((M, N), dtype=output_dtype, device=x.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _convert_from_mxfp8_kernel[grid](
        x,
        y,
        s,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        s.stride(0),
        s.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
    )
    return y
