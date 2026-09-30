# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def batched_gemm_a16w8(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    YQ: torch.Tensor | None = None,
    transpose_bm: bool = False,
    config: dict | None = None,
):
    """
    [gluon/gfx1250] Batched Y[i] = X[i] @ (WQ[i] * w_scale)^T.

    Drop-in for batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant
    on the MLA absorbed-weight BMMs, except X is kept in bf16 (not re-quantized
    to fp8 per token-group): WQ is upcast to bf16 in registers and the GEMM
    runs as bf16 WMMA. These BMMs are store-bound, so this costs no speed.

    Args:
        X (torch.Tensor): (B, M, K) bf16, K-contiguous (other dims any stride).
        WQ (torch.Tensor): (B, N, K) fp8, K-contiguous.
        w_scale (torch.Tensor): per-tensor fp32 scale for WQ (numel 1).
        YQ (Optional[torch.Tensor]): output, (B, M, N) or (M, B, N) if transpose_bm.
        transpose_bm (bool): write the output as (M, B, N).

    Returns:
        torch.Tensor: (B, M, N), or (M, B, N) if transpose_bm.
    """
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
        create_wmma_layouts,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_a16w8 import (
        _batched_gemm_a16w8_gfx1250_persistent_kernel,
    )

    B, M, K = X.shape
    N = WQ.shape[1]
    assert WQ.shape[0] == B and WQ.shape[2] == K, "Incompatible X/WQ shapes"
    assert X.stride(2) == 1 and WQ.stride(2) == 1, "X and WQ must be K-contiguous"
    assert dtype in (torch.bfloat16, torch.float16)

    if YQ is None:
        if transpose_bm:
            YQ = torch.empty((M, B, N), dtype=dtype, device=X.device)
        else:
            YQ = torch.empty((B, M, N), dtype=dtype, device=X.device)
    Y = YQ.transpose(0, 1) if transpose_bm else YQ
    assert Y.shape == (B, M, N) and Y.stride(2) == 1, "Output dimension error"

    if config is None:
        config, _ = get_gemm_config("BATCHED_GEMM-A16W8", M, N, K, backend="gluon")
    BM = config["BLOCK_SIZE_M"]
    BN = config["BLOCK_SIZE_N"]
    BK = config["BLOCK_SIZE_K"]
    num_warps = config["num_warps"]
    store_mode = config.get("STORE_MODE", 1)

    wmma_layout, operand_a, operand_b = create_wmma_layouts(num_warps)
    shared_a, shared_b = create_shared_layouts(BM, BN, BK, "TN", 16)
    c_pad = config.get("C_PAD", 16)
    shared_c = gl.PaddedSharedLayout.with_identity_for([[BN, c_pad]], [BM, BN], [1, 0])
    tpn = min(32, BN // 8)
    store_layout = gl.BlockedLayout([1, 8], [32 // tpn, tpn], [num_warps, 1], [1, 0])

    num_tiles = B * triton.cdiv(M, BM) * triton.cdiv(N, BN)
    num_wgs = min(num_tiles, _num_cus(X.device.index) * config.get("WG_PER_CU", 1))

    _batched_gemm_a16w8_gfx1250_persistent_kernel[(num_wgs,)](
        X,
        WQ,
        Y,
        w_scale,
        B,
        M,
        N,
        K,
        X.stride(0),
        X.stride(1),
        WQ.stride(0),
        WQ.stride(1),
        Y.stride(0),
        Y.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        NUM_BUFFERS=config.get("NUM_BUFFERS", 2),
        NUM_WGS=num_wgs,
        STORE_MODE=store_mode,
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        SHARED_LAYOUT_C=shared_c,
        STORE_LAYOUT=store_layout,
        WMMA_LAYOUT=wmma_layout,
        OPERAND_LAYOUT_A=operand_a,
        OPERAND_LAYOUT_B=operand_b,
        num_warps=num_warps,
        waves_per_eu=config.get("waves_per_eu", 1),
    )
    return YQ
