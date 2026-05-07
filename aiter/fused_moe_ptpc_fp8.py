# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 MoE HSACO: prebuilt ``.co`` via ``hsaco_tools.get_kernel`` (no pyhip package).

``fused_moe()`` calls ``run_moe_small_batch_hsaco`` only when ``AITER_MOE_SMALL_BATCH=1`` and
``1 <= B <= MOE_HSACO_SMALL_BATCH_MAX_B``. No HSACO hooks in ``get_2stage_cfgs``.

``fused_moe_`` may call ``fused_moe_hsaco_batch1_fwd`` for a second-chance B=1 path (no
``moe_sorting``) if the top-level fast path returned ``None``.

Requires matching artifacts under ``hsa/<gfx>/fmoe_pyhip/`` (README there).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from csrc.cpp_itfs.hsaco_tools import get_kernel
from csrc.cpp_itfs.utils import AITER_CORE_DIR

def fused_moe_ptpc_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
) -> Optional[torch.Tensor]:
    """Prebuilt ``.co`` MoE for ``1 <= B <= MOE_HSACO_SMALL_BATCH_MAX_B``; else ``None``."""
    B = int(hidden_states.shape[0])
    if not (1 <= B <= 32):
        return None
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation != ActivationType.Silu
    ):
        return None
    if not (
        (quant_type == QuantType.per_Token and w1.dtype == torch.float8_e4m3fnuz)
    ):
        return None

    from aiter.fused_moe import moe_sorting

    E, N1, K1 = w1.shape
    N2, K2 = w2.shape[1], w2.shape[2]
    TOPK = topk_ids.shape[1]
    assert N1 == 2 * K2

    topk_w_f32 = (
        topk_weight
        if topk_weight.dtype == torch.float32
        else topk_weight.float()
    )

    gemm1_out = torch.empty(
        [B, TOPK, N1 // 2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    print(f"********************************* Batch size:{B} *********************************")
    if B == 1:
        assert N1 == 2 * K2
        print("Get moe_gemm_batch1_gate kernel start")
        moe_gemm_batch1_gate = get_kernel(f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_pyhip/moe_gemm_batch1-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=True:moe_gemm_batch1")
        print("Get moe_gemm_batch1_gate kernel end")
        print("Get moe_gemm_batch1_down kernel start")
        moe_gemm_batch1_down = get_kernel(f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_pyhip/moe_gemm_batch1-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=False:moe_gemm_batch1")
        print("Get moe_gemm_batch1_down kernel end")

        # ``weight_dtype`` / ``with_silu`` are pyhip JIT compile-time args (see
        # ``pyhip/src/contrib/moe.py:moe_gemm_batch1``); the prebuilt ``.co`` basename
        # already encodes them. ``hsaco_tools.CallableKernel`` only accepts tensors,
        # None, int, or float — not ``torch.dtype`` or ``bool``.
        # topk_w_f32 = (
        #     topk_weight
        #     if topk_weight.dtype == torch.float32
        #     else topk_weight.float()
        # )
        if moe_gemm_batch1_gate is None or moe_gemm_batch1_down is None:
            return None
        cur_out = torch.zeros(
            [1, N2], dtype=hidden_states.dtype, device=hidden_states.device
        )
        print("Call moe_gemm_batch1_gate kernel start")

        moe_gemm_batch1_gate(
            [N1 // 32, TOPK],
            [256],
            hidden_states,
            w1,
            gemm1_out,
            topk_ids,
            topk_w_f32,
            w1_scale,
            1,
            N1,
            K1,
        )

        print("Call moe_gemm_batch1_gate kernel end")
        print("Call moe_gemm_batch1_down kernel start")
        moe_gemm_batch1_down(
            [N2 // 32, TOPK],
            [64],
            gemm1_out,
            w2,
            cur_out,
            topk_ids,
            topk_w_f32,
            w2_scale,
            1,
            N2,
            K2,
        )
        print("Call moe_gemm_batch1_down kernel end")
    else:
        BLOCK_M = 16
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
            topk_ids,
            topk_weight,
            E,
            K1,
            hidden_states.dtype,
            BLOCK_M,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
        )
        grid = int(sorted_expert_ids.shape[0])
        if B * TOPK <= E:
            grid = min(grid, B * TOPK)
        cur_out = torch.zeros(
            (B, N2),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        print("Get moe_gemm_batch kernel start")
        moe_gemm_batch = get_kernel(f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_pyhip/moe_gemm_batch-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=True:moe_gemm_batch")
        print("Get moe_gemm_batch kernel end")
        print("Get moe_gemm_batch and moe_2stage_splitk kernels start")
        moe_2stage_splitk = get_kernel(f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_pyhip/moe_2stage_splitk-1-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-with_silu=False-BLOCK_TILE_SIZE_M=16-BLOCK_TILE_SIZE_N=64-fp8_ptpc=True:moe_2stage_splitk")
        print("Get moe_gemm_batch and moe_2stage_splitk kernels end")
        if moe_gemm_batch is None or moe_2stage_splitk is None:
            return None

        # Compile-time args (``weight_dtype``, ``with_silu``, TOPK/K/N/tiling, ``fp8_ptpc``)
        # are encoded in the ``.co`` basename. ``CallableKernel`` only accepts tensors,
        # ``None``, ``int``, or ``float`` (see ``hsaco_tools.py``).
        print("Call moe_gemm_batch kernel start")
        moe_gemm_batch(
            [N1 // 32, grid],
            [256],
            hidden_states,
            w1,
            gemm1_out,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w1_scale,
            B,
            N1,
            K1,
            TOPK,
        )
        print("Call moe_gemm_batch kernel end")
        BLOCK_TILE_SIZE_N = 64
        print("Call moe_2stage_splitk kernel start")
        moe_2stage_splitk(
            [N2 // BLOCK_TILE_SIZE_N, grid],
            [64],
            gemm1_out,
            w2,
            cur_out,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w2_scale,
            B,
        )
        print("Call moe_2stage_splitk kernel end")

        
    return cur_out


if __name__ == "__main__":
    # Matches ``fused_moe_hsaco_batch1_fwd`` hardcoded gfx942 kernel layout (see that function).
    HIDDEN_SIZE = 4096
    INTER_SIZE = 1024
    TP = 8
    INTER_TP = INTER_SIZE // TP
    TOPK = 10
    E = 512
    # B = 2 # 1 ~ 32
    batch_sizes = [1, 2, 4, 8, 10, 12, 16, 32]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float8_e4m3fnuz
    for B in batch_sizes:
        hidden_states = torch.randn(B, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        w1_bf16 = torch.randn(E, 2 * INTER_TP, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        w2_bf16 = torch.randn(E, HIDDEN_SIZE, INTER_TP, dtype=torch.bfloat16, device=device)

        torch_quant = aiter.get_torch_quant(QuantType.per_Token)
        w1, w1_scale = torch_quant(w1_bf16, quant_dtype=weight_dtype)
        w2, w2_scale = torch_quant(w2_bf16, quant_dtype=weight_dtype)

        topk_weight = torch.randn(B, TOPK, dtype=torch.float32, device=device)
        topk_ids = torch.randint(0, E, (B, TOPK), dtype=torch.int32, device=device)

        expert_mask = None
        num_local_tokens = None
        moe_sorting_dispatch_policy = 0
        cur_out = fused_moe_ptpc_fp8(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            ActivationType.Silu,
            QuantType.per_Token,
            w1_scale,
            w2_scale,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
        )
        if cur_out is None:
            print("cur_out: None (gfx942 / kernel / shape gate not satisfied or HSACO unavailable)")
        else:
            print("cur_out: ", cur_out.shape)
            print("cur_out: ", cur_out)