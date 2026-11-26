# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _convert_from_mxfp8_kernel,
    _convert_to_mxfp8_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.fast_launch import fast_launch
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "convert_from_mxfp8",
    "convert_to_mxfp8",
]

_LOGGER = AiterTritonLogger()

# repeated small casts (e.g. one per projection GEMM) skip Triton's per-call
# argument binding; sizes are keyed by Triton's specialization classes
_convert_to_mxfp8_launch = fast_launch(_convert_to_mxfp8_kernel, by_specialization=True)


def convert_to_mxfp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    quant_block_size: int = 32,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: bool | None = None,
    block_m: int = 64,
    block_n: int = 64,
    scale_mode: str = "torchao",
    dequant_dtype: torch.dtype | None = None,
    round_input_bf16: bool = False,
    preshuffle: bool = False,
    allow_unaligned: bool = False,
    out_cols: int | None = None,
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
        scale_mode: ``"torchao"`` (default): e8m0 exponent
            ``floor(log2(rne_mbits(amax))) - target_max_pow2``. ``"te"``:
            Transformer Engine ``MXFP8BlockScaling`` (Megatron-LM MXFP8),
            ``ceil(log2(amax / max_pos))`` clamped to [0, 254] (0 for amax 0),
            then a saturating round-to-nearest-even cast of ``x * 2^(127-s)``;
            bit-exact with TE. ``"te"`` takes the portable path (no ASM, no SR).
        dequant_dtype: with ``scale_mode="te"`` and e4m3, write the dequantized
            values ``fp8 * 2^(s-127)`` in this dtype (e.g. bfloat16, which holds
            them exactly) instead of the fp8 data, in one pass. ``x`` may be a
            strided view (e.g. a transpose) in every mode.
        round_input_bf16: quantize the round-to-nearest-even bf16 value of each
            fp32 input element, bit-identical to quantizing ``x.bfloat16()``
            without materializing it.
        preshuffle: write the fp8 data in ``shuffle_weight(y, layout=(16, 16))``
            order (bit-identical to shuffling afterwards, without the extra
            pass); not with ``dequant_dtype`` or ``is_2d_block``.
        allow_unaligned: take M and N that are not multiples of the tile (masked
            loads and stores, portable path, 1-D blocks): the same bits as
            zero-padding ``x`` to the tile, without the padded copy.
        out_cols: with ``allow_unaligned``, the number of output columns (>= N);
            the columns past N hold the cast of zeros, as for an ``x``
            zero-padded to ``out_cols`` columns.

    Returns:
        Tuple of (quantized_tensor, scales) where quantized_tensor has fp8_dtype
        and scales has uint8 dtype with e8m0 format.

    Note:
        M and N must be exact multiples of ``block_m`` / ``block_n`` (default
        64).  The kernel loads full tiles without masking; unaligned shapes must
        be padded before calling.  This is a deliberate constraint that enables
        the gfx950 ASM path.

        Scale rounding uses torchao round-to-nearest-even, which differs from
        ``_downcast_to_mxfp`` in ``quant_moe.py`` (round-up / round-down, fnuz,
        arbitrary shapes).  Quantising the same tensor through both will produce
        different bits — see the kernel file header for the rationale.
    """
    _LOGGER.info("CONVERT_TO_MXFP8: x=%s", tuple(x.shape))
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(
            f"fp8_dtype must be torch.float8_e4m3fn or torch.float8_e5m2, got {fp8_dtype}"
        )
    if scale_mode not in ("torchao", "te"):
        raise ValueError(f"scale_mode must be 'torchao' or 'te', got {scale_mode!r}")
    te_scale = scale_mode == "te"
    if dequant_dtype is not None and not (
        te_scale and fp8_dtype == torch.float8_e4m3fn
    ):
        raise ValueError("dequant_dtype needs scale_mode='te' and float8_e4m3fn")
    if preshuffle and (dequant_dtype is not None or is_2d_block):
        raise ValueError("preshuffle writes 1-D-block fp8 data only")
    if round_input_bf16 and x.dtype != torch.float32:
        raise ValueError("round_input_bf16 needs an fp32 input")
    if te_scale and (use_asm or use_sr):
        raise ValueError("scale_mode='te' supports neither use_asm nor use_sr")
    # The ASM path (v_cvt_scalef32_*) is gfx950-only and _pack_fp8 only accepts
    # e4m3fn; anything else must take the portable path.
    asm_supported = (
        arch_info.get_arch() == "gfx950"
        and fp8_dtype == torch.float8_e4m3fn
        and not te_scale
    )
    asm_auto = use_asm is None
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
    n_out = N if out_cols is None else out_cols
    masked = allow_unaligned and (M % block_m or n_out % block_n or n_out != N)
    if out_cols is not None and (not allow_unaligned or out_cols < N):
        raise ValueError("out_cols needs allow_unaligned and out_cols >= N")
    if masked and use_asm and asm_auto:
        use_asm = False
    if masked and (use_asm or is_2d_block or use_sr):
        raise ValueError("unaligned shapes take the portable 1-D path, without SR")
    # Unmasked, the kernel loads/stores full [block_m, block_n] tiles and
    # reshapes each into QUANT_BLOCK_SIZE groups, so M and N must tile exactly.
    assert masked or (M % block_m == 0 and N % block_n == 0), (
        f"MXFP8 convert requires M,N aligned to tile ({block_m},{block_n}); "
        f"got ({M},{N})"
    )

    y = torch.empty((M, n_out), dtype=dequant_dtype or fp8_dtype, device=x.device)
    if is_2d_block:
        scale_m = triton.cdiv(M, quant_block_size)
    else:
        scale_m = M
    scale_n = triton.cdiv(n_out, quant_block_size)
    s = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(n_out, block_n))
    # num_warps scales with tile area so it remains valid when block_m/block_n
    # differ from the default 64×64 (MI308X benchmark: nw=1/2/4 differ < 1%
    # at default tile, nw≥8 regresses).
    num_warps = min(16, max(1, block_m * block_n // 1024))
    _convert_to_mxfp8_launch[grid](
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
        M,
        N,
        n_out,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_SR=use_sr,
        USE_ASM=use_asm,
        TE_SCALE=te_scale,
        DEQUANT_OUT=dequant_dtype is not None,
        ROUND_BF16=round_input_bf16,
        PRESHUFFLE=preshuffle,
        MASKED=bool(masked),
        num_warps=num_warps,
        waves_per_eu=2,
        num_stages=2,
    )
    if preshuffle:
        # the attributes shuffle_weight sets
        y.is_shuffled = True
        y.aiter_shuffle_layout = (16, 16)
        y.aiter_shuffle_is_guinterleave = False
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
    _LOGGER.info("CONVERT_FROM_MXFP8: x=%s", tuple(x.shape))
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
    num_warps = min(16, max(1, block_m * block_n // 1024))
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
        num_warps=num_warps,
        waves_per_eu=2,
        num_stages=2,
    )
    return y
