# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3.5 397B MoE PTPC FP8 TP8 run with prebuilt hasco artifacts on gfx942:
``fused_moe()`` calls ``fused_moe_ptpc_fp8`` only when the following conditions are met:
- B=1~32, E=512, N1=256, K1=4096, N2=4096, K2=128, TOPK=10
- if (os.environ.get("AITER_MOE_SMALL_BATCH", "0") == "1"
    and 1 <= hidden_states.shape[0] <= 32
    and hidden_states.dtype == torch.bfloat16
    and expert_mask is None
    and activation == ActivationType.Silu
    and (quant_type == QuantType.per_Token and w1.dtype == torch.float8_e4m3fnuz)
    and get_gfx() == "gfx942"
    and topk_ids.shape[1] == 10
    and w1.shape[1] == 256
    and w1.shape[2] == 4096
    and w2.shape[1] == 4096
    and w2.shape[2] == 128)
- Requires matching artifacts under ``hsa/gfx942/fmoe_ptpc_fp8/``, and loading via ``csrc.cpp_itfs.hsaco_tools.get_kernel``.
"""

import os
from typing import Any, Optional

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.fused_moe import moe_sorting
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
        (quant_type == QuantType.per_Token and w1.dtype == torch.float8_e4m3fnuz and get_gfx() == "gfx942")
    ):
        return None

    E, N1, K1 = w1.shape
    N2, K2 = w2.shape[1], w2.shape[2]
    TOPK = topk_ids.shape[1]
    fp8_ptpc = w1.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    num_CU = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
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
    if B == 1:
        assert N1 == 2 * K2
        try:
            moe_gemm_batch1_gate = get_kernel(
                f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
                f"moe_gemm_batch1-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=True:moe_gemm_batch1"
            )
            moe_gemm_batch1_down = get_kernel(
                f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
                f"moe_gemm_batch1-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=False:moe_gemm_batch1"
            )
            cur_out = torch.zeros(
                [1, N2], dtype=hidden_states.dtype, device=hidden_states.device
            )
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
        except Exception as e:
            msg = (
                f"fused_moe_ptpc_fp8 (B=1): HSACO kernel load or launch failed: {e}. "
                f"Check artifacts under {AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
            )
            logger.warning(
                msg + "; fallback to default fused_moe_() instead."
            )
            return None
    elif 2 <= B <= 32:
        # Stage 1: Shared ``moe_sorting`` + ``moe_gemm_batch``;
        # stage 2: Choose between ``moe_2stage_down_loopn`` and ``moe_2stage_splitk`` based on ``use_down_loopn`` condition.
        BLOCK_M = 16
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
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
            grid = B * TOPK

        try:
            moe_gemm_batch = get_kernel(
                f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
                f"moe_gemm_batch-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=True:moe_gemm_batch"
            )
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
        except Exception as e:
            msg = (
                f"fused_moe_ptpc_fp8 (B={B}): moe_gemm_batch HSACO kernel load or launch failed: {e}. "
                f"Check artifacts under {AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
            )
            logger.warning(
                msg + "; fallback to default fused_moe_() instead."
            )
            return None


        BLOCK_N = 1024
        use_down_loopn = (
            fp8_ptpc
            and (N2 // BLOCK_N) * grid >= num_CU
            and N2 % BLOCK_N == 0
            and 16 <= B <= 32
        )

        if use_down_loopn:
            try:
                moe_2stage_down_loopn = get_kernel(
                    f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
                    f"moe_2stage_down_loopn-1-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-"
                    f"BLOCK_TILE_SIZE_M=16-BLOCK_TILE_SIZE_N=16-fp8_ptpc=True-BLOCK_N=1024-"
                    f"atomic_write=False-STAGES=3:moe_2stage_down_loopn"
                )
                gemm2_out = torch.empty(
                    [B, TOPK, N2],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                moe_2stage_down_loopn(
                    [N2 // BLOCK_N, grid],
                    [256],
                    gemm1_out,
                    w2,
                    gemm2_out,
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    w2_scale,
                    B,
                )
                cur_out = torch.sum(gemm2_out, dim=1)
            except Exception as e:
                msg = (
                    f"fused_moe_ptpc_fp8 (B={B}): moe_2stage_down_loopn HSACO kernel load or launch failed: {e}. "
                    f"Check artifacts under {AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
                )
                logger.warning(
                    msg + "; fallback to default fused_moe_() instead."
                )
                return None
        else:
            try:
                moe_2stage_splitk = get_kernel(
                    f"{AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
                    f"moe_2stage_splitk-1-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-"
                    f"with_silu=False-BLOCK_TILE_SIZE_M=16-BLOCK_TILE_SIZE_N=64-fp8_ptpc=True:moe_2stage_splitk"
                )
                BLOCK_TILE_SIZE_N = 64
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
            except Exception as e:
                msg = (
                    f"fused_moe_ptpc_fp8 (B={B}): moe_2stage_splitk HSACO kernel load or launch failed: {e}. "
                    f"Check artifacts under {AITER_CORE_DIR}/hsa/gfx942/fmoe_ptpc_fp8/"
                )
                logger.warning(
                    msg + "; fallback to default fused_moe_() instead."
                )
                return None

    return cur_out
