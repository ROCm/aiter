# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_a16w16_split_cat import (
    _fused_gemm_a16w16_split_cat,
)
from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_a16w16_quant_x import (
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_gemm_a16w16_split_cat(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    S1: int,
    S2: int,
    dtype: Optional[torch.dtype] = torch.float8_e4m3fn,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the 16-bit matmul C = X @ W^T (bf16/fp16 inputs, no quant scales).
    Then splits the product C into C1 and C2 with sizes S1 and S2 along the last
    dimension, and concatenates Y onto C1 along the last dimension.

    This is the bf16 analog of ``fused_gemm_afp4wfp4_split_cat`` used to expand
    the compact MLA latent KV (kv_b_proj) into per-head K/V in one kernel,
    folding the nope/v split, the k_pe concat, and the fp8 cast into the GEMM
    epilogue.

    Equivalent to the following sequence:
        c = (x @ w.T).view(-1, y.shape[1], S1 + S2)
        c1, c2 = c.split([S1, S2], dim=-1)
        c1 = torch.cat([c1, y], dim=-1)
        return c1.to(dtype), c2.to(dtype)

    Key parameters:
    - x: Matrix X with shape (M, K).
    - w: Matrix W with shape (N, K) (transposed internally to (K, N)).
    - y: Tensor Y with shape (M, D, S3) — k_pe, broadcast across the D heads.

    Returns:
    - c1: Output matrix K with shape (M, D, S1 + S3), dtype ``dtype``.
    - c2: Output matrix V with shape (M, D, S2), dtype ``dtype``.

    NOTE: N must be D * (S1 + S2).
    """
    _LOGGER.info(
        f"FUSED_GEMM_A16W16_SPLIT_CAT: x={tuple(x.shape)} w={tuple(w.shape)} y={tuple(y.shape)}"
    )

    M, K = x.shape
    N, Kw = w.shape
    My, D, S3 = y.shape

    # Check constraints.
    assert K == Kw, "Incompatible dimensions!!!"
    assert My == M, "Incompatible dimensions!!!"
    assert N == D * (S1 + S2), "N is not D * (S1 + S2)"

    # Transpose w: (N, K) -> (K, N)
    w = w.T

    if config is None:
        config, _ = _get_config(M, N, K)

    # This kernel does the split/cat epilogue inline; the split-K reduce path
    # is not supported here, so always run with a single K partition.
    config.pop("NUM_KSPLIT", None)
    config.pop("SPLITK_BLOCK_SIZE", None)
    config.setdefault("cache_modifier", None)

    c1 = torch.empty((M, D, S1 + S3), dtype=dtype, device=x.device)
    c2 = torch.empty((M, D, S2), dtype=dtype, device=x.device)

    # S3 block size: chosen so that across all N blocks the whole D*S3 of y is
    # covered (mirrors the afp4 split_cat reference).
    config["BLOCK_SIZE_S3"] = triton.next_power_of_2(
        triton.cdiv(D * S3, triton.cdiv(N, config["BLOCK_SIZE_N"]))
    )

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _fused_gemm_a16w16_split_cat[grid](
        x,
        w,
        y,
        c1,
        c2,
        M,
        N,
        K,
        D,
        S1,
        S2,
        S3,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        c1.stride(0),
        c1.stride(1),
        c1.stride(2),
        c2.stride(0),
        c2.stride(1),
        c2.stride(2),
        **config,
    )

    return c1, c2
