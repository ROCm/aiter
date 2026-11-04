# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_NORM_MODES = {"aiter": 0, "gluon": 1}
_LDS_BYTES = 320 * 1024
# The norm prologue reproduces 2048-wide rows (128 lanes of 16, or 4 waves).
_SEG_K = 2048


def _get_config(M: int, N: int, K: int):
    return get_gemm_config("FUSED-RMSNORM-GEMM-A16W16", M, N, K, backend="gluon")


def _lds_bytes(num_seg, block_m, block_n, block_k, num_buffers, k_groups):
    # Padded tiles: 8 bf16 of pad every BLOCK_K // k_groups elements.
    group = block_k // k_groups

    def padded(n):
        return n * 2 + n // group * 16

    return num_seg * padded(block_m * _SEG_K) + num_buffers * padded(block_n * block_k)


def fused_rmsnorm_gemm_a16w16_supported(
    M: int, N: int, H: int, num_seg: int, dtype: torch.dtype
) -> bool:
    """Whether fused_rmsnorm_gemm_a16w16 runs these shapes: gfx1250, bf16,
    segments of H = 2048, and a config whose A rows and W tiles fit in LDS."""
    if get_arch() != "gfx1250" or dtype != torch.bfloat16 or H != _SEG_K:
        return False
    if M <= 0 or num_seg not in (1, 2):
        return False
    config, _ = _get_config(M, N, num_seg * H)
    k_groups = config["num_warps"] if config.get("K_SPLIT", True) else 1
    need = _lds_bytes(
        num_seg, config["BLOCK_M"], config["BLOCK_N"], config["BLOCK_K"], 2, k_groups
    )
    return need <= _LDS_BYTES


def fused_rmsnorm_gemm_a16w16(
    a0: torch.Tensor,
    w: torch.Tensor,
    norm_weight0: torch.Tensor | None,
    eps0: float | None,
    a1: torch.Tensor | None = None,
    norm_weight1: torch.Tensor | None = None,
    eps1: float | None = None,
    row_index: torch.Tensor | None = None,
    norm_mode: str = "aiter",
    y: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    Computes Y = cat(norm_0(A_0), norm_1(A_1)) @ W^T in one gfx1250 Gluon
    kernel, for small M. Each norm is a Gemma RMSNorm,
    bf16(x * rsqrt(mean(x^2) + eps) * (1 + weight)), with the rstd of the norm
    kernel bit for bit. The kernel normalizes A in its prologue, so the normed
    A is never written to memory.

    Args:
        a0 (torch.Tensor): Segment 0 of A, (M, H). With row_index, a table
            (V, H), and segment 0 is a0[row_index].
        w (torch.Tensor): Weight (N, H), or (N, 2 * H) with a1.
        norm_weight0 (Optional[torch.Tensor]): Norm weight (H,) of segment 0.
            None: segment 0 is used as is (a table normed beforehand); needs
            row_index.
        eps0 (Optional[float]): Norm epsilon of segment 0.
        a1 (Optional[torch.Tensor]): Segment 1 of A, (M, H).
        norm_weight1 (Optional[torch.Tensor]): Norm weight (H,) of segment 1.
        eps1 (Optional[float]): Norm epsilon of segment 1.
        row_index (Optional[torch.Tensor]): int32 (M,) row ids into a0. An id
            outside [0, V) gives a zero row.
        norm_mode (str): The norm kernel whose rstd is reproduced: "aiter"
            (fused_qk_rmsnorm_group_quant) or "gluon" (fused_rmsnorm_add).
        y (Optional[torch.Tensor]): Output (M, N), bf16, rows contiguous.
        config (Optional[dict]): Kernel tuning parameters.

    Returns:
        torch.Tensor: Y, (M, N) bf16.

    Limits: gfx1250, bf16, H = 2048, contiguous rows, 4 * k warps. The whole
    A rows of a program sit in LDS (see fused_rmsnorm_gemm_a16w16_supported).
    With the config key K_SPLIT the warps split K, and Y differs from the
    unfused GEMM by the fp32 summation order; without it Y is the unfused
    result bit for bit.
    """
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_rmsnorm_gemm_a16w16 import (
        _fused_rmsnorm_gemm_a16w16_kernel,
    )

    num_seg = 1 if a1 is None else 2
    H = a0.shape[1]
    M = a0.shape[0] if row_index is None else row_index.shape[0]
    N, K = w.shape
    assert H == _SEG_K, f"segments must be {_SEG_K} wide, got {H}"
    assert K == num_seg * H, f"w is {tuple(w.shape)}, want K = {num_seg} * {H}"
    assert a0.dtype == w.dtype == torch.bfloat16, "bf16 only"
    assert a0.stride(1) == 1 and w.stride(1) == 1, "rows must be contiguous"
    assert norm_mode in _NORM_MODES, f"norm_mode must be one of {list(_NORM_MODES)}"
    if a1 is not None:
        assert a1.shape == (M, H) and a1.dtype == a0.dtype and a1.stride(1) == 1
        assert norm_weight1 is not None and eps1 is not None
    if norm_weight0 is None:
        assert row_index is not None, "an unnormed segment 0 must be gathered"
    else:
        assert eps0 is not None
    if row_index is not None:
        assert row_index.dtype == torch.int32 and row_index.dim() == 1
        assert row_index.is_contiguous()

    if y is None:
        y = torch.empty((M, N), dtype=a0.dtype, device=a0.device)
    assert y.shape == (M, N) and y.dtype == a0.dtype and y.stride(1) == 1
    if M == 0:
        return y
    if config is None:
        config, _ = _get_config(M, N, K)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    num_warps = config["num_warps"]
    k_split = config.get("K_SPLIT", True)
    k_groups = num_warps if k_split else 1
    NUM_BUFFERS = min(config["NUM_BUFFERS"], K // BLOCK_K)
    while NUM_BUFFERS > 2 and (
        _lds_bytes(num_seg, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, k_groups)
        > _LDS_BYTES
    ):
        NUM_BUFFERS -= 1

    # Rank 3: with K_SPLIT the warps split the first axis (the K group), else
    # every warp holds the whole tile.
    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[
            [1 << i if k_split else 0, 0, 0] for i in range(num_warps.bit_length() - 1)
        ],
        reg_bases=[],
        instr_shape=[16, 16, 32],
        rank=3,
    )

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _LOGGER.info(
        "FUSED_RMSNORM_GEMM_A16W16: M=%s N=%s K=%s mode=%s", M, N, K, norm_mode
    )
    _fused_rmsnorm_gemm_a16w16_kernel[grid](
        a0,
        a1 if a1 is not None else a0,
        row_index if row_index is not None else a0,
        norm_weight0 if norm_weight0 is not None else a0,
        norm_weight1 if norm_weight1 is not None else a0,
        w,
        y,
        M,
        N,
        a0.shape[0],
        a0.stride(0),
        a1.stride(0) if a1 is not None else 0,
        w.stride(0),
        y.stride(0),
        float(eps0 if eps0 is not None else 0.0),
        float(eps1 if eps1 is not None else 0.0),
        SEG_K=H,
        NUM_SEG=num_seg,
        GATHER_0=row_index is not None,
        NORM_MODE=_NORM_MODES[norm_mode],
        NORM_0=norm_weight0 is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        K_SPLIT=k_split,
        WMMA_LAYOUT=wmma_layout,
        num_warps=num_warps,
        waves_per_eu=config.get("waves_per_eu", 0),
    )
    return y
