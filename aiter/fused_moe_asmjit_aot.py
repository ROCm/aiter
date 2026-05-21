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
- Requires matching artifacts under ``hsa/gfx942/fmoe_asmjit/``, and loading via ``csrc.cpp_itfs.hsaco_tools.get_kernel``.
"""

import os
from typing import Any, Optional

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.fused_moe import moe_sorting
from csrc.cpp_itfs.hsaco_tools import hsaco

from dataclasses import dataclass

@dataclass
class Config:
    BLOCK_M: int
    use_down_loopn: bool
    use_prefill: bool

    def to_string(self):
        return str(self.BLOCK_M) + "_" + str(self.use_down_loopn) + "_" + str(self.use_prefill)

    @classmethod
    def from_string(cls, data: str):
        parts = data.split("_")
        return cls(*[eval(p) for p in parts])

def get_tune_space():
    return [Config(16, True, False).to_string(), Config(64, True, True).to_string(), Config(128, True, True).to_string()]

def fused_moe_asmjit_aot(
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
    config_string: str,
) -> Optional[torch.Tensor]:

    # decode kernel configs from kernel name
    kcfgs = Config.from_string(config_string)

    B = int(hidden_states.shape[0])
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation != ActivationType.Silu
    ):
        return None
    if not (w1.dtype == torch.float8_e4m3fnuz and get_gfx() == "gfx942"):
        return None

    if (quant_type != QuantType.per_Token and quant_type != QuantType.per_Tensor):
        return None

    qtype_str = str(quant_type).split(".")[1]

    E, N1, K1 = w1.shape
    N2, K2 = w2.shape[1], w2.shape[2]
    TOPK = topk_ids.shape[1]
    fp8_ptpc = w1.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) and (quant_type == QuantType.per_Token)
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

    if kcfgs.use_prefill:
        BLOCK_TILE_SIZE_N = 128
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
            topk_ids,
            topk_weight,
            E,
            N2,     # reduce dim is same with output dim
            hidden_states.dtype,
            kcfgs.BLOCK_M,
            None,
            None,
            0,
        )
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_Token)
        hidden_states_q, hidden_states_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=w1.dtype,
            num_rows=None,
        )
        hsaco.fmoe_asmjit.moe_2stage_gateup([N1 // BLOCK_TILE_SIZE_N * sorted_expert_ids.shape[0]], [256],
                    hidden_states_q, w1, 
                    gemm1_out, 
                    sorted_ids, 
                    sorted_expert_ids, 
                    num_valid_ids, 
                    hidden_states_scale,
                    w1_scale, B, N1 // BLOCK_TILE_SIZE_N * sorted_expert_ids.shape[0],
                    weight_dtype = str(w1.dtype),
                    TOPK = TOPK,
                    K=K1,
                    N=N1,
                    BLOCK_TILE_SIZE_M = kcfgs.BLOCK_M,
                    BLOCK_TILE_SIZE_N = BLOCK_TILE_SIZE_N,
                    quant_type_w = f"QuantType.{qtype_str}",
                    )
        gemm1_out_q, gemm1_out_scale = quant_func(
            gemm1_out.view(B * TOPK, -1),
            scale=None,
            quant_dtype=w2.dtype,
            num_rows=None,
        )
        gemm2_out = torch.empty(B, TOPK, N2, dtype=torch.bfloat16, device=gemm1_out_q.device)
        hsaco.fmoe_asmjit.moe_2stage_down([1, sorted_expert_ids.shape[0]], [256],
                    gemm1_out_q, w2, 
                    gemm2_out, #cur_out,
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    gemm1_out_scale,
                    w2_scale,
                    B,
                    sorted_expert_ids.shape[0],
                    weight_dtype = str(w2.dtype),
                    TOPK = TOPK,
                    K = K2,
                    N = N2,
                    with_silu = False,
                    BLOCK_TILE_SIZE_M = kcfgs.BLOCK_M,
                    BLOCK_TILE_SIZE_N = BLOCK_TILE_SIZE_N,
                    quant_type_w = f"QuantType.{qtype_str}")
        num_WG = num_CU * 4
        num_tokens_wg = B // num_WG
        num_extra_tokens = B % num_WG
        hsaco.fmoe_asmjit.moe_gemm_final_reduce_bf16([num_WG], [64],
                                gemm2_out,
                                cur_out,
                                num_tokens_wg, num_extra_tokens, B,
                                TOPK = TOPK, OC = N2)
        return cur_out

    if B == 1:
        assert N1 == 2 * K2
        cur_out = torch.zeros(
            [1, N2], dtype=hidden_states.dtype, device=hidden_states.device
        )
        hsaco.fmoe_asmjit.moe_gemm_batch1(
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
            weight_dtype = torch.float8_e4m3fnuz,
            with_silu=True,
            quant_type_str = qtype_str
        )
        hsaco.fmoe_asmjit.moe_gemm_batch1(
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
            weight_dtype = torch.float8_e4m3fnuz,
            with_silu=False,
            quant_type_str = qtype_str                
        )
    elif 2 <= B <= 32:
        # Stage 1: Shared ``moe_sorting`` + ``moe_gemm_batch``;
        # stage 2: Choose between ``moe_2stage_down_loopn`` and ``moe_2stage_splitk`` based on ``use_down_loopn`` condition.
        BLOCK_M = kcfgs.BLOCK_M
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

        hsaco.fmoe_asmjit.moe_gemm_batch(
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
            weight_dtype=torch.float8_e4m3fnuz,
            with_silu=True,
            quant_type_str=qtype_str
        )

        BLOCK_N = 1024
        if kcfgs.use_down_loopn:
            # extra checks
            use_down_loopn = (
                fp8_ptpc
                and (N2 // BLOCK_N) * grid >= num_CU
                and N2 % BLOCK_N == 0
                and 16 <= B <= 32
            )
        else:
            use_down_loopn = False

        if use_down_loopn:
            gemm2_out = torch.empty(
                [B, TOPK, N2],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            hsaco.fmoe_asmjit.moe_2stage_down_loopn(
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
                weight_dtype=torch.float8_e4m3fnuz,
                TOPK=TOPK,
                K=K2,
                N=N2,
                BLOCK_TILE_SIZE_M=16,
                BLOCK_TILE_SIZE_N=16,
                fp8_ptpc=True,
                BLOCK_N=BLOCK_N,
                atomic_write=False,
                STAGES=3
            )
            cur_out = torch.sum(gemm2_out, dim=1)
        else:
            BLOCK_TILE_SIZE_N = 64
            hsaco.fmoe_asmjit.moe_2stage_splitk(
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
                weight_dtype=torch.float8_e4m3fnuz,
                TOPK=TOPK,
                K=K2,
                N=N2,
                with_silu=False,
                BLOCK_TILE_SIZE_M=16,
                BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
                quant_type_str=qtype_str
            )
    else:
        return None
    return cur_out
