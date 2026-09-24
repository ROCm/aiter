# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MXFP8 grouped (MoE) GEMM.

Performs all expert GEMMs in a single Triton kernel launch using MXFP8
quantisation. E8M0 microscales (one uint8 per ``quant_block_size`` elements
along K) are converted to FP32 power-of-two factors inside the kernel.

Convention (TN layout):
    ``out[tokens_for_e] = lhs[tokens_for_e] @ rhs[e]^T``
where both lhs and rhs are stored in FP8 with per-block E8M0 scales.

Implementation reuses ``_moe_gemm_a8w8`` with ``USE_FNUZ=True`` for fnuz
FP8 (gfx942) and the standard OCP path on gfx950+.  The group_sizes
interface is converted to the ExptData routing format internally.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a8w8 import _moe_gemm_a8w8
from aiter.ops.triton.moe.moe_utils import group_sizes_to_expt_tensors
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

__all__ = ["moe_gemm_mxfp8"]

_LOGGER = AiterTritonLogger()

# Tile values must come from configs/<arch>/triton/moe/mxfp8_fnuz/DEFAULT.json.
_MXFP8_FALLBACK = triton.Config({}, num_warps=4, num_stages=1)


def moe_gemm_mxfp8(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    group_sizes: torch.Tensor,
    quant_block_size: int = 32,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused MXFP8 grouped GEMM for MoE.

    Args:
        lhs: FP8 activation ``[total_tokens, K]``.
        rhs: FP8 weight ``[E, N, K]``.
        x_scale: E8M0 uint8 activation scales ``[total_tokens, K // qbs]``.
        w_scale: E8M0 uint8 weight scales ``[E, N, K // qbs]``.
        group_sizes: Expert token counts ``[E]`` (int).
        quant_block_size: MXFP block size (default 32).
        bias: Optional per-expert bias ``[E, N]``.
        out_dtype: Output dtype (default BF16).

    Returns:
        Output tensor ``[total_tokens, N]``.
    """
    _LOGGER.info(
        "MOE_GEMM_MXFP8: lhs=%s rhs=%s x_scale=%s w_scale=%s block_size=%d",
        tuple(lhs.shape),
        tuple(rhs.shape),
        tuple(x_scale.shape),
        tuple(w_scale.shape),
        quant_block_size,
    )

    total_tokens = lhs.shape[0]
    E, N, K = rhs.shape

    assert lhs.shape[1] == K, "K dimension mismatch"
    assert quant_block_size == 32, (
        f"quant_block_size must be 32 (got {quant_block_size}): "
        "_moe_gemm_a8w8 hardcodes MX_PACK_DIVISOR=32"
    )
    assert (
        K % quant_block_size == 0
    ), f"K ({K}) must be divisible by quant_block_size ({quant_block_size})"

    out = torch.empty(total_tokens, N, dtype=out_dtype, device=lhs.device)
    if total_tokens == 0:
        return out

    BLOCK_K = quant_block_size

    cfg = get_tuned_kernel_config(
        "moe", "MXFP8_FNUZ", "moe_gemm_mxfp8", _MXFP8_FALLBACK
    )
    if "BLOCK_M" not in cfg.kwargs or "BLOCK_N" not in cfg.kwargs:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        raise FileNotFoundError(
            f"No MXFP8 MoE GEMM tile config for arch '{get_arch()}'. "
            "Add configs/<arch>/triton/moe/mxfp8_fnuz/DEFAULT.json."
        )
    BLOCK_M = cfg.kwargs["BLOCK_M"]
    BLOCK_N = cfg.kwargs["BLOCK_N"]

    expt_hist, expt_offs, expt_offs_sum, expt_data, grid_m = (
        group_sizes_to_expt_tensors(group_sizes, BLOCK_M)
    )
    if grid_m == 0:
        return out

    grid_n = triton.cdiv(N, BLOCK_N)

    # Permute rhs/w_scale from NK to KN layout as required by _moe_gemm_a8w8.
    rhs_kn = rhs.permute(0, 2, 1).contiguous()
    w_scale_kn = w_scale.permute(0, 2, 1).contiguous()
    bias_stride = N if bias is not None else 0

    _moe_gemm_a8w8[(grid_m * grid_n,)](
        # output
        out,
        out.stride(0),  # stride_y_k  (SPLIT_K=1, unused)
        out.stride(0),  # stride_y_m
        out.stride(1),  # stride_y_n
        # X (activations)
        lhs,
        lhs.stride(0),
        lhs.stride(1),
        # XMxScale
        x_scale,
        x_scale.stride(0),
        x_scale.stride(1),
        # W (weights, KN layout)
        rhs_kn,
        rhs_kn.stride(0),
        rhs_kn.stride(1),
        rhs_kn.stride(2),
        # WMxScale (KN layout)
        w_scale_kn,
        w_scale_kn.stride(0),
        w_scale_kn.stride(1),
        w_scale_kn.stride(2),
        # static scales (not used)
        None,
        None,
        None,
        # bias
        bias,
        bias_stride,
        # Gammas (not used)
        None,
        # shapes
        N,
        K,
        # routing
        None,  # GatherIndx — tokens already in contiguous expert order
        expt_hist,
        expt_offs,
        expt_offs_sum,
        expt_data,
        # grid
        grid_m,
        grid_n,
        # fused ops (disabled)
        False,  # APPLY_SWIGLU
        None,
        None,  # alpha, limit
        1,  # ACTIVATION_REDUCTION_N
        False,  # SWIGLU_ADD_RESIDUAL
        E,  # N_EXPTS_ACT
        # tile sizes
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        1,  # GROUP_M
        1,  # XCD_SWIZZLE
        None,  # SWIZZLE_MX_SCALE
        K % BLOCK_K == 0,  # EVEN_K
        K % BLOCK_K or BLOCK_K,  # MASK_K_LIMIT: remainder size of last K block
        1,  # SPLIT_K
        "",  # W_CACHE_MODIFIER
        False,  # UPCAST_INDICES
        True,  # USE_FNUZ — fnuz FP8 (float8_e4m3fnuz) path
    )
    return out
