# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.quant.dual_layout_mxfp4 import (
    _dual_layout_quant_mxfp4_kernel,
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
)

__all__ = ["dual_layout_quant_mxfp4"]


_LOGGER = AiterTritonLogger()
_MXFP4_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_MAX_PHILOX_COUNTER = (1 << 63) - 1
_CONFIG_NAME = "DUAL-LAYOUT-MXFP4"


def _philox_streams(
    M: int,
    N: int,
    use_sr_row: bool,
    use_sr_transposed: bool,
    philox_seed: int | None,
    philox_offset: int,
) -> tuple[int, int, int]:
    use_sr = use_sr_row or use_sr_transposed
    if not use_sr:
        if philox_seed is not None or philox_offset != 0:
            raise ValueError(
                "Philox arguments are only valid when stochastic rounding is enabled"
            )
        return 0, 0, 0

    if philox_seed is None:
        raise ValueError("philox_seed is required when stochastic rounding is enabled")
    if type(philox_seed) is not int or type(philox_offset) is not int:
        raise TypeError("philox_seed and philox_offset must be integers")
    if not 0 <= philox_seed <= _MAX_PHILOX_COUNTER:
        raise ValueError("philox_seed must be in [0, 2**63 - 1]")

    counters_per_layout = M * N // 8
    enabled_layouts = int(use_sr_row) + int(use_sr_transposed)
    reserved_counters = enabled_layouts * counters_per_layout
    if not 0 <= philox_offset <= _MAX_PHILOX_COUNTER - reserved_counters + 1:
        raise ValueError(
            "philox_offset must be non-negative and leave room for enabled layouts"
        )

    next_offset = philox_offset
    row_offset = next_offset if use_sr_row else 0
    if use_sr_row:
        next_offset += counters_per_layout
    transposed_offset = next_offset if use_sr_transposed else 0
    return philox_seed, row_offset, transposed_offset


def dual_layout_quant_mxfp4(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    *,
    block_size: int = 32,
    g: int = 16,
    use_sr_row: bool = False,
    use_sr_transposed: bool = False,
    philox_seed: int | None = None,
    philox_offset: int = 0,
    swizzle_scale: bool = False,
    shuffle_col: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit MXFP4 ``x`` and signed-H16 ``x.T`` layouts from one read.

    Args:
        x: Contiguous 2-D bfloat16 or float32 tensor with 32-aligned dimensions.
        sign_vector: Same-device vector containing 16 values equal to +1 or -1.
        block_size: MXFP4 scale block size; only 32 is supported.
        g: Hadamard group size; only 16 is supported.
        use_sr_row: Stochastically round the row-wise E2M1 payload.
        use_sr_transposed: Stochastically round the rotated-transposed payload.
        philox_seed: Philox seed required when either SR flag is enabled.
        philox_offset: Starting counter for the enabled SR streams.
        swizzle_scale: Store both scale tensors in the gfx950 GEMM layout.
        shuffle_col: Store the rotated-transposed payload in the gfx950 B layout.

    The gfx950 JSON configuration supplies ``num_warps`` and ``num_stages``.

    Returns:
        ``(row_fp4, row_scales, transposed_fp4, transposed_scales)``: packed
        E2M1 payloads of shape ``(M, N // 2)`` and ``(N, M // 2)``, with E8M0
        scales of shape ``(M, N // 32)`` and ``(N, M // 32)``.
        ``swizzle_scale=True`` stores each scale tensor in the padded gfx950
        GEMM scale arrangement instead of its logical row-major shape;
        ``shuffle_col=True`` likewise stores ``transposed_fp4`` in the
        preshuffled gfx950 B-operand arrangement.
    """
    _LOGGER.info(
        "DUAL_LAYOUT_QUANT_MXFP4: shape=%s sr_row=%s sr_transposed=%s "
        "swizzle_scale=%s shuffle_col=%s",
        tuple(x.shape),
        use_sr_row,
        use_sr_transposed,
        swizzle_scale,
        shuffle_col,
    )
    if block_size != _MXFP4_BLOCK_SIZE:
        raise ValueError(f"block_size must be {_MXFP4_BLOCK_SIZE}, got {block_size}")
    if g != _HADAMARD_SIZE:
        raise ValueError(f"g must be {_HADAMARD_SIZE}, got {g}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D, got {x.dim()}-D")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"x must have dtype torch.bfloat16 or torch.float32, got {x.dtype}"
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not isinstance(use_sr_row, bool) or not isinstance(use_sr_transposed, bool):
        raise TypeError("use_sr_row and use_sr_transposed must be bool")
    if not isinstance(swizzle_scale, bool) or not isinstance(shuffle_col, bool):
        raise TypeError("swizzle_scale and shuffle_col must be bool")

    M, N = x.shape
    if M == 0 or N == 0:
        raise ValueError(f"x dimensions must be non-zero, got {tuple(x.shape)}")
    if M % _MXFP4_BLOCK_SIZE != 0 or N % _MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "x dimensions must both be divisible by "
            f"{_MXFP4_BLOCK_SIZE}, got {tuple(x.shape)}"
        )
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError("dual-layout MXFP4 quantization requires gfx950")

    row_scale_cols = N // block_size
    transposed_scale_cols = M // block_size
    if swizzle_scale:
        if not mxfp4_scale_swizzle_supported(M, row_scale_cols):
            raise ValueError(
                f"row scale shape {(M, row_scale_cols)} cannot be swizzled"
            )
        if not mxfp4_scale_swizzle_supported(N, transposed_scale_cols):
            raise ValueError(
                "transposed scale shape "
                f"{(N, transposed_scale_cols)} cannot be swizzled"
            )
    if shuffle_col and not mxfp4_data_shuffle_supported(N, M // 2):
        raise ValueError(f"transposed payload shape {(N, M // 2)} cannot be shuffled")

    sign_vector = prepare_h16_sign(sign_vector, x, g)
    philox_seed, philox_offset_row, philox_offset_transposed = _philox_streams(
        M,
        N,
        use_sr_row,
        use_sr_transposed,
        philox_seed,
        philox_offset,
    )

    row_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    row_scale_shape = (
        (M // MXFP4_SCALE_STRIPE, row_scale_cols * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (M, row_scale_cols)
    )
    row_scales = torch.empty(row_scale_shape, dtype=torch.uint8, device=x.device)
    transposed_fp4 = torch.empty((N, M // 2), dtype=torch.uint8, device=x.device)
    transposed_scale_shape = (
        (
            N // MXFP4_SCALE_STRIPE,
            transposed_scale_cols * MXFP4_SCALE_STRIPE,
        )
        if swizzle_scale
        else (N, transposed_scale_cols)
    )
    transposed_scales = torch.empty(
        transposed_scale_shape, dtype=torch.uint8, device=x.device
    )

    config_dir = resolve_config_dir("fusions", _CONFIG_NAME)
    config = load_config_json(f"{config_dir}/DEFAULT.json")[
        "_dual_layout_quant_mxfp4_kernel"
    ]
    grid = (M // _MXFP4_BLOCK_SIZE, N // _MXFP4_BLOCK_SIZE)
    _dual_layout_quant_mxfp4_kernel[grid](
        x,
        row_fp4,
        row_scales,
        transposed_fp4,
        transposed_scales,
        sign_vector,
        *x.stride(),
        *row_fp4.stride(),
        *row_scales.stride(),
        *transposed_fp4.stride(),
        *transposed_scales.stride(),
        M,
        N,
        philox_seed,
        philox_offset_row,
        philox_offset_transposed,
        BLOCK_SIZE=_MXFP4_BLOCK_SIZE,
        HADAMARD_SIZE=_HADAMARD_SIZE,
        USE_SR_ROW=use_sr_row,
        USE_SR_TRANSPOSED=use_sr_transposed,
        SWIZZLE_SCALE=swizzle_scale,
        SHUFFLE_COL=shuffle_col,
        SCALE_STRIPE=MXFP4_SCALE_STRIPE,
        SCALE_KCHUNK=MXFP4_SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=MXFP4_SHUFFLE_UNIT_BYTES,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return row_fp4, row_scales, transposed_fp4, transposed_scales
