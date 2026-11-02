# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.quant.dequant_hadamard_quant_mxfp4 import (
    _dequant_hadamard_quant_mxfp4_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.mxfp4 import (
    MXFP4_SCALE_KCHUNK,
    MXFP4_SCALE_STRIPE,
    MXFP4_SHUFFLE_TILE_ROWS,
    MXFP4_SHUFFLE_UNIT_BYTES,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.mxfp4 import (
    mxfp4_data_shuffle_supported,
    mxfp4_scale_swizzle_supported,
    prepare_h16_sign,
    reserve_mxfp4_philox_streams,
)

__all__ = ["dequant_hadamard_quant_mxfp4"]

_LOGGER = AiterTritonLogger()
_MXFP4_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_CONFIG_NAME = "DEQUANT-HADAMARD-QUANT-MXFP4"


def _philox_stream(
    M: int,
    K: int,
    use_sr: bool,
    philox_seed: int | None,
    philox_offset: int,
) -> tuple[int, int]:
    seed, offsets = reserve_mxfp4_philox_streams(
        M * K, (use_sr,), philox_seed, philox_offset
    )
    return seed, offsets[0]


def dequant_hadamard_quant_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    sign_vector: torch.Tensor,
    *,
    block_size: int = 32,
    g: int = 16,
    use_sr: bool = False,
    philox_seed: int | None = None,
    philox_offset: int = 0,
    swizzle_scale: bool = False,
    shuffle_data: bool = False,
    in_scale_swizzled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse packed MXFP4 dequantization, transpose, H16, and requantization.

    The input represents a logical ``(M, K)`` matrix with one row-major E8M0
    scale per 32 values. The result represents ``H16(data.T)`` with signed,
    normalized Hadamard groups along the transposed matrix's final dimension.

    Args:
        data_fp4: 2-D packed E2M1 tensor of shape ``(M, K // 2)``.
            Raw ``uint8`` and canonical ``float4_e2m1fn_x2`` are accepted.
        scales: E8M0 tensor of shape ``(M, K // 32)``, or its swizzled gfx950
            layout when ``in_scale_swizzled=True``. Raw ``uint8`` and
            canonical ``float8_e8m0fnu`` are accepted. Swizzled scales must
            be contiguous; row-major scales may be strided.
        sign_vector: A same-device float32 or bfloat16 vector containing exactly
            16 finite values, each equal to ``-1`` or ``1``.
        block_size: MXFP4 scale block size. Only 32 is supported.
        g: Hadamard group size. Only 16 is supported.
        use_sr: Stochastically round the requantized E2M1 payload.
        philox_seed: Philox seed required when ``use_sr`` is enabled.
        philox_offset: Starting counter of the SR stream; the launch consumes
            ``K * M // 8`` counters.
        swizzle_scale: Store output scales in the gfx950 GEMM layout.
        shuffle_data: Store the output payload in the gfx950 B layout.
        in_scale_swizzled: Read ``scales`` from the swizzled gfx950 layout.

    Returns:
        ``(output_fp4, output_scales)``. The payload has shape ``(K, M // 2)``
        in both plain and shuffled layouts. The scales have shape
        ``(K, M // 32)``, or the swizzled ``(K // 32, M)`` when
        ``swizzle_scale=True``.

    Decoded values are narrowed to bfloat16 before H16, matching Lumen's
    fused training path. An input E8M0 NaN sentinel (raw 255) poisons the
    corresponding complete 32-value output scale block as raw 255 with a
    zero payload. The gfx950 JSON configuration supplies ``num_warps`` and
    ``num_stages``.
    """
    _LOGGER.info(
        "DEQUANT_HADAMARD_QUANT_MXFP4: shape=%s use_sr=%s swizzle_scale=%s "
        "shuffle_data=%s in_scale_swizzled=%s",
        tuple(data_fp4.shape),
        use_sr,
        swizzle_scale,
        shuffle_data,
        in_scale_swizzled,
    )
    if block_size != _MXFP4_BLOCK_SIZE:
        raise ValueError(f"block_size must be {_MXFP4_BLOCK_SIZE}, got {block_size}")
    if g != _HADAMARD_SIZE:
        raise ValueError(f"g must be {_HADAMARD_SIZE}, got {g}")
    if data_fp4.dim() != 2:
        raise ValueError(f"data_fp4 must be 2-D, got {data_fp4.dim()}-D")
    if data_fp4.dtype not in (
        torch.uint8,
        getattr(torch, "float4_e2m1fn_x2", torch.uint8),
    ):
        raise TypeError(
            "data_fp4 must have dtype torch.uint8 or "
            f"torch.float4_e2m1fn_x2, got {data_fp4.dtype}"
        )
    data_bytes = data_fp4.view(torch.uint8)
    M, K_packed = data_bytes.shape
    K = K_packed * 2
    if M == 0 or K == 0:
        raise ValueError(f"logical input dimensions must be non-zero, got ({M}, {K})")
    if M % block_size != 0 or K % block_size != 0:
        raise ValueError(f"logical input shape must be divisible by 32, got ({M}, {K})")

    if scales.dim() != 2:
        raise ValueError(f"scales must be 2-D, got {scales.dim()}-D")
    if scales.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8)):
        raise TypeError(
            "scales must have dtype torch.uint8 or torch.float8_e8m0fnu, "
            f"got {scales.dtype}"
        )
    for flag in (use_sr, swizzle_scale, shuffle_data, in_scale_swizzled):
        if not isinstance(flag, bool):
            raise TypeError("layout and stochastic-rounding flags must be bool")
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError("dequant-H16 MXFP4 requantization requires gfx950")

    input_scale_cols = K // block_size
    output_scale_cols = M // block_size
    if in_scale_swizzled:
        if not mxfp4_scale_swizzle_supported(M, input_scale_cols):
            raise ValueError(
                f"input scale shape {(M, input_scale_cols)} cannot be swizzled"
            )
        expected_scale_shape = (
            M // MXFP4_SCALE_STRIPE,
            input_scale_cols * MXFP4_SCALE_STRIPE,
        )
    else:
        expected_scale_shape = (M, input_scale_cols)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales shape must be {expected_scale_shape}, got {tuple(scales.shape)}"
        )
    if in_scale_swizzled and not scales.is_contiguous():
        raise ValueError("swizzled scales must be contiguous")
    scale_bytes = scales.view(torch.uint8)
    if scales.device != data_fp4.device:
        raise ValueError("scales must be on the same device as data_fp4")
    if swizzle_scale and not mxfp4_scale_swizzle_supported(K, output_scale_cols):
        raise ValueError(
            f"output scale shape {(K, output_scale_cols)} cannot be swizzled"
        )
    if shuffle_data and not mxfp4_data_shuffle_supported(K, M // 2):
        raise ValueError(f"output payload shape {(K, M // 2)} cannot be shuffled")

    sign_flat = prepare_h16_sign(sign_vector, data_fp4, g)
    philox_seed, philox_offset = _philox_stream(
        M, K, use_sr, philox_seed, philox_offset
    )

    output = torch.empty((K, M // 2), dtype=torch.uint8, device=data_fp4.device)
    output_scale_shape = (
        (K // MXFP4_SCALE_STRIPE, output_scale_cols * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (K, output_scale_cols)
    )
    output_scales = torch.empty(
        output_scale_shape, dtype=torch.uint8, device=data_fp4.device
    )

    config_dir = resolve_config_dir("fusions", _CONFIG_NAME)
    config = load_config_json(f"{config_dir}/DEFAULT.json")[
        "_dequant_hadamard_quant_mxfp4_kernel"
    ]

    grid = (M // block_size, K // block_size)
    _dequant_hadamard_quant_mxfp4_kernel[grid](
        data_bytes,
        scale_bytes,
        sign_flat,
        output,
        output_scales,
        *data_bytes.stride(),
        *scale_bytes.stride(),
        *output.stride(),
        *output_scales.stride(),
        M,
        K,
        philox_seed,
        philox_offset,
        BLOCK_SIZE=block_size,
        HADAMARD_SIZE=g,
        USE_SR=use_sr,
        SWIZZLE_SCALE=swizzle_scale,
        SHUFFLE_DATA=shuffle_data,
        IN_SCALE_SWIZZLED=in_scale_swizzled,
        SCALE_STRIPE=MXFP4_SCALE_STRIPE,
        SCALE_KCHUNK=MXFP4_SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=MXFP4_SHUFFLE_UNIT_BYTES,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return output, output_scales
