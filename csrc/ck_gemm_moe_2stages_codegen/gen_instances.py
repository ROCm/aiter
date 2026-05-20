# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
import itertools
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list

ACT_OP_MAP = {
    "gelu": 0,
    "silu": 1,
    "swiglustep": 2,
}

STG_INSTANCE_IMPL = """// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common{quanttype}.cuh"

using A0DataType = {A0DataType};
using B0DataType = {B0DataType};
using AccDataType = {AccDataType};
using EDataType = {EDataType};
using CDEElementOp = {CDEElementOp};
const bool Nswizzle = {Nswizzle};
const bool PerTensorQuant = {Quant} == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = {MulRoutedWeight};
const int ActOP = {ActOP};
CK_MOE_STAGE{Stage}_GEMM_DEFINE({BlockSize}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, V{PipelineVer})
"""

# stage 2 explicit MNPerXDLArg variant (NPerBlock=32 nopad path requires MNPerXDLArg=32)
STG_INSTANCE_IMPL_MNX = """// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common{quanttype}.cuh"

using A0DataType = {A0DataType};
using B0DataType = {B0DataType};
using AccDataType = {AccDataType};
using EDataType = {EDataType};
using CDEElementOp = {CDEElementOp};
const bool Nswizzle = {Nswizzle};
const bool PerTensorQuant = {Quant} == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = {MulRoutedWeight};
const int ActOP = {ActOP};
CK_MOE_STAGE2_GEMM_DEFINE_MNX({BlockSize}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, V{PipelineVer}, {MNPerXDLArg})
"""


LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

#define GENERATE_LOOKUP_TABLE()                                                                                      \\
   {                                                                                                                             \\"""

LOOKUP_template = """
       {{"{kernel_tag}",                                                                                                       \\
        ck_moe_stage{Stage}_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V{PipelineVer}, {BlockSize}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>}},                       \\"""

# stage 2 explicit MNPerXDLArg variant for runtime kernel resolution (symbol matches MNX .cu)
# NT-switch wrapper template signature = 16 args (ActOP, MNPerXDLArg); NT is runtime via std::optional<bool> nt
LOOKUP_template_MNX = """
       {{"{kernel_tag}",                                                                                                       \\
        ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V{PipelineVer}, {BlockSize}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}, {MNPerXDLArg}>}},                       \\"""

LOOKUP_end = """
   }

"""


gemm1_heuristic_dispatch_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage1_heuristic_dispatch(int block_m, int inter_dim, at::ScalarType x_dtype, at::ScalarType w_dtype, at::ScalarType y_dtype, int act_op, int quant, bool mul_routed_weight_stage, bool is_shuffled)
{{
"""

gemm2_heuristic_dispatch_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage2_heuristic_dispatch(int block_m, int inter_dim, at::ScalarType x_dtype, at::ScalarType w_dtype, at::ScalarType y_dtype, int act_op, int quant, bool mul_routed_weight_stage, bool is_shuffled)
{{
"""

heuristic_dispatch_end = """
    TORCH_CHECK(
        false,
        "Unsupported kernel config for moe heuristic dispatch");
}}

"""

A16W16_A8W8_gemm1_gfx950_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {ActOP} == act_op
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 32)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 128)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 256)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 256, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""


A16W16_gemm1_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {ActOP} == act_op
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 32)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 128)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 256)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 256, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

A8W8_gemm1_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {ActOP} == act_op
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 16)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 16, 64, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 32)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 128)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 256)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 256, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

A8W4_gemm1_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {ActOP} == act_op
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 32)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 128)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 256)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 256, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

A4W4_gemm1_heuristic_dispatch = """
#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {ActOP} == act_op
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant
        && {Preshuffle} == is_shuffled)
    {{
        if (block_m == 32)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 64, 32, 32, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 128)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
#endif

"""

A4W4_bns_gemm1_heuristic_dispatch = """
#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {ActOP} == act_op
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant
        && {Preshuffle} == is_shuffled)
    {{
        if (block_m == 32)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 64, 128/sizeof({A0DataType}), 2, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 128)
        {{
            return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 64, 128/sizeof({A0DataType}), 2, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
#endif

"""

A8W8_blockscale_gemm1_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {ActOP} == act_op
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (inter_dim % 128 != 0 && inter_dim % 64 == 0)
        {{
            // NPerBlock=64: inter_dim is 64-aligned but not 128-aligned (e.g. tp=4 inter_dim=320)
            // scale index block_n_id*64/128 (integer div) maps pairs of tiles to one per_1x128 scale
            if (block_m == 16)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 16, 64, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 32)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 64)
            {{
                // V1 used instead of V3: V3 requires MRepeat>=4, but MPerBlock=64,MWaves=1 gives MRepeat=4 (borderline)
                // NWaves=4 gives NXdlPerWave=1 (acceptable tradeoff for correctness)
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                TORCH_CHECK(
                    false,
                    "Unsupported block_m value for NPerBlock=64 blockscale dispatch: ",
                    block_m);
            }}
        }}
        else if (inter_dim % 128 != 0 && inter_dim % 64 != 0 && inter_dim % 32 == 0)
        {{
            // NPerBlock=32: inter_dim is 32-aligned but not 64-aligned (e.g. tp=8 inter_dim=160, inter_dim=96)
            // scale index block_n_id*32/128 (integer div) maps groups of 4 tiles to one per_1x128 scale
            // W1_RESULTS §1.2: ScaleSliceSizeN = ceil(NPerBlock=32, ScaleBlockN=128) = 1
            // W1_RESULTS §2 candidate A: BLOCKSIZE=128, MPerBlock=16, KPerBlock=128, MWaves=1, NWaves=2, PipelineVer=V1
            // [W3 EXTEND] sub-case ladder for block_m in {{16, 32, 64}} — fixes W2-V2-block_m_64 caveat
            // [W3] block_m=32/64 instance bound to lead R01 P1-A finalize (T1 §3.8 candidate A id=8 + B' fallback id=9)
            if (block_m == 16)
            {{
                // [W2 V01-supp PASS] candidate A: V1, 128, 16, 32, 128/sizeof(A0DataType), 1, 2
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 128, 16, 32, 128/sizeof({A0DataType}), 1, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 32)
            {{
                // [W3 NEW] lead R01 P1-A binding: A (id=8) — V1, 128, 32, 32, 128/sizeof(A0DataType), 1, 2
                // mimic id=7 minimal drift; MXDLPerWave=2, M-axis line 1607 extension carries [unverified inference] caveat (T1 §3.5)
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 128, 32, 32, 128/sizeof({A0DataType}), 1, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 64)
            {{
                // [W3 NEW] lead R01 P1-A binding: B' fallback (id=9) — V1, 256, 64, 32, 128/sizeof(A0DataType), 2, 2
                // BLOCKSIZE=256/MWaves=2/MXDLPerWave=2 low-risk; rejected T1 main B (BLOCKSIZE=128/MWaves=1/MXDLPerWave=4)
                // PipelineVer=1 (single-LDS) — Run_2Lds caveat (W1 §4.3 #3) NOT triggered
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 32, 128/sizeof({A0DataType}), 2, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                TORCH_CHECK(
                    false,
                    "Unsupported block_m value for NPerBlock=32 blockscale dispatch: ",
                    block_m);
            }}
        }}
        else
        {{
            if (block_m == 16)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 16, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 32)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 64)
            {{
                return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                TORCH_CHECK(
                    false,
                    "Unsupported block_m value for moe heuristic dispatch: ",
                    block_m);
            }}
        }}
    }}
"""

A16W16_gemm2_gfx950_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 32)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 64)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 128, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 128)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 256)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

# NOTE: temporarily not using KPerBlock=64 for inter_dim=192 cases due to gfx950 fp8 mfma instruction limitation
A8W8_gemm2_gfx950_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 32)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 128)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 256)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

A16W16_gemm2_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 32)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 64)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 128)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 256)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

A8W8_gemm2_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 16)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 64, 16, 64, 64, 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 32)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 64)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 128)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else if (block_m == 256)
        {{
            if (inter_dim <= 192)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 64, 64, 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

A8W4_gemm2_heuristic_dispatch = """
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 32)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 128)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 256)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 256, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""


A4W4_gemm2_heuristic_dispatch = """
#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant
        && {Preshuffle} == is_shuffled)
    {{
        if (inter_dim <= 256)
        {{
            if (block_m == 32)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 64, 32, 32, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 64)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 64, 64, 128, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 128)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 64, 128, 128, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                TORCH_CHECK(
                    false,
                    "Unsupported block_m value for moe heuristic dispatch: ",
                    block_m);
            }}
        }}
        else
        {{
            if (block_m == 32)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 64)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 128)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                TORCH_CHECK(
                    false,
                    "Unsupported block_m value for moe heuristic dispatch: ",
                    block_m);
            }}
        }}
    }}
#endif
"""

A4W4_bns_gemm2_heuristic_dispatch = """
#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant
        && {Preshuffle} == is_shuffled)
    {{
        if (inter_dim <= 256)
        {{
            if (block_m == 32)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 64, 32, 32, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 64)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 64, 64, 64, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 128)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 64, 128, 128, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                TORCH_CHECK(
                    false,
                    "Unsupported block_m value for moe heuristic dispatch: ",
                    block_m);
            }}
        }}
        else
        {{
            if (block_m == 32)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 64)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 64, 128/sizeof({A0DataType}), 2, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else if (block_m == 128)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 128, 64, 128/sizeof({A0DataType}), 2, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            }}
            else
            {{
                TORCH_CHECK(
                    false,
                    "Unsupported block_m value for moe heuristic dispatch: ",
                    block_m);
            }}
        }}
    }}
#endif
"""

# gfx942 (default): 含 NPerBlock=32 KPerBlock=32 nopad outer-if (T15 + T16 spec)
# K=160 (32-aligned, 64-non-aligned) 命中 nopad; K=192 (64-aligned) 不命中走 pad fallback
# 18th template arg = MNPerXDLArg=32 (T17 wrapper template extension; default 16 backward compat)
A8W8_blockscale_gemm2_heuristic_dispatch = """

    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (inter_dim % 128 != 0 && inter_dim % 64 != 0 && inter_dim % 32 == 0)
        {{
            // [T15+T16+T17 NEW] NPerBlock=32 / KPerBlock=32 nopad path (gfx942 only)
            // T11 verdict: KPack=16, MPerXdl=NPerXdl=32 唯一可行 (KPerThread=KPerBlock/K0PerXdlops=32/2=16=KPack)
            // gfx950 不会到达此分支 (由 codegen-time tag 拆分到 a8w8blkscale_gfx950)
            // K=160 命中 (160%128!=0, 160%64!=0, 160%32==0), K=192 (192%64==0) 不命中
            // 17th-18th template args: ActOP, MNPerXDLArg=32 (T17 wrapper extension; default 16 backward compat)
            // block_m=16 fallback to else: MPerBlock=16 + MPerXdl=32 → MXDLPerWave=0.5 invalid (T15 §3.3 caveat)
            if (block_m == 32)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 64, 32, 32, 32/sizeof({A0DataType}), 1, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}, 32>;
            }}
            else if (block_m == 64)
            {{
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 128, 64, 32, 32/sizeof({A0DataType}), 2, 1, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}, 32>;
            }}
            // block_m=16 falls through to pad path below
        }}
        if (block_m == 16)
        {{
            if (inter_dim % 256 == 0)
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 16, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            else
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 128, 16, 128, 128/sizeof({A0DataType}), 1, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 32)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""

# gfx950 path: 仅 NPerBlock=128 / KPerBlock>=128 (KPack=32 强制); K=160 强制 pad 到 128 整数倍
# 字面同现 a8w8blkscale stage 2 行为 (W3 pre-T17 状态)
A8W8_blockscale_gemm2_gfx950_heuristic_dispatch = """

    if (dtype_checker<{A0DataType}>{{}}(x_dtype)
        && dtype_checker<{B0DataType}>{{}}(w_dtype)
        && dtype_checker<{EDataType}>{{}}(y_dtype)
        && {MulRoutedWeight} == mul_routed_weight_stage
        && {Quant} == quant)
    {{
        if (block_m == 16)
        {{
            if (inter_dim % 256 == 0)
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 16, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
            else
                return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 128, 16, 128, 128/sizeof({A0DataType}), 1, 2, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 32)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 64)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else
        {{
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }}
    }}
"""


heuristic_dispatch_dict = {
    "a8w8_gfx950": [
        A16W16_A8W8_gemm1_gfx950_heuristic_dispatch,
        A8W8_gemm2_gfx950_heuristic_dispatch,
    ],
    "a8w8": [
        A8W8_gemm1_heuristic_dispatch,
        A8W8_gemm2_heuristic_dispatch,
    ],
    "a8w8blkscale": [
        A8W8_blockscale_gemm1_heuristic_dispatch,
        A8W8_blockscale_gemm2_heuristic_dispatch,
    ],
    "a8w8blkscale_gfx950": [
        A8W8_blockscale_gemm1_heuristic_dispatch,
        A8W8_blockscale_gemm2_gfx950_heuristic_dispatch,
    ],
    "a16w16_gfx950": [
        A16W16_A8W8_gemm1_gfx950_heuristic_dispatch,
        A16W16_gemm2_gfx950_heuristic_dispatch,
    ],
    "a16w16": [
        A16W16_gemm1_heuristic_dispatch,
        A16W16_gemm2_heuristic_dispatch,
    ],
    "a8w4": [
        A8W4_gemm1_heuristic_dispatch,
        A8W4_gemm2_heuristic_dispatch,
    ],
    "a4w4": [
        A4W4_gemm1_heuristic_dispatch,
        A4W4_gemm2_heuristic_dispatch,
    ],
    "a4w4_bns": [
        A4W4_bns_gemm1_heuristic_dispatch,
        A4W4_bns_gemm2_heuristic_dispatch,
    ],
}


def generate_instance_and_lookUpTable_head(working_path):
    f_lookUpTable = os.path.join(working_path, "gemm_moe_ck2stages_lookup.h")
    if os.path.exists(f_lookUpTable):
        os.remove(f_lookUpTable)
    with open(f_lookUpTable, "w") as f_lookup:
        f_lookup.write(LOOKUP_head)

    f_gemm1_heuristic_dispatch = os.path.join(
        working_path, "ck2stages_moe_stage1_heuristic_dispatch.hpp"
    )
    if os.path.exists(f_gemm1_heuristic_dispatch):
        os.remove(f_gemm1_heuristic_dispatch)
    with open(f_gemm1_heuristic_dispatch, "w") as f_h:
        f_h.write(gemm1_heuristic_dispatch_head)

    f_gemm2_heuristic_dispatch = os.path.join(
        working_path, "ck2stages_moe_stage2_heuristic_dispatch.hpp"
    )
    if os.path.exists(f_gemm2_heuristic_dispatch):
        os.remove(f_gemm2_heuristic_dispatch)
    with open(f_gemm2_heuristic_dispatch, "w") as f_h:
        f_h.write(gemm2_heuristic_dispatch_head)


def generate_instance_and_lookUpTable_end(working_path):
    f_lookUpTable = os.path.join(working_path, "gemm_moe_ck2stages_lookup.h")
    with open(f_lookUpTable, "a") as f_lookup:
        f_lookup.write(LOOKUP_end)

    f_gemm1_heuristic_dispatch = os.path.join(
        working_path, "ck2stages_moe_stage1_heuristic_dispatch.hpp"
    )
    with open(f_gemm1_heuristic_dispatch, "a") as f_h:
        f_h.write(heuristic_dispatch_end)

    f_gemm2_heuristic_dispatch = os.path.join(
        working_path, "ck2stages_moe_stage2_heuristic_dispatch.hpp"
    )
    with open(f_gemm2_heuristic_dispatch, "a") as f_h:
        f_h.write(heuristic_dispatch_end)


class ck_moe_2stage_gemm_codegen:
    def __init__(
        self,
        working_path,
        a_dtype,
        b_dtype,
        c_dtype,
        quant_type,
        activation,
        mul_routed_weight_stage,
        preshuffle,
        splitk,
    ):
        self.working_path = working_path
        self.a_dtype = a_dtype.upper()
        self.b_dtype = b_dtype.upper()
        self.c_dtype = c_dtype.upper()
        self.quant_type = quant_type
        self.activation = activation
        self.mul_routed_weight_stage = mul_routed_weight_stage
        self.nswizzle = False
        self.preshuffle = preshuffle
        self.splitk = splitk

    def generate_instance_and_lookUpTable(self):
        _, gemm1_kernel_list = get_gemm1_kernels_list(
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.nswizzle,
            self.quant_type,
            self.activation,
            self.mul_routed_weight_stage == 1,
            self.preshuffle,
            self.splitk,
        )
        tag, gemm2_kernel_list = get_gemm2_kernels_list(
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.nswizzle,
            self.quant_type,
            self.mul_routed_weight_stage == 2,
            self.preshuffle,
        )
        kernel_list = list(gemm1_kernel_list.values()) + list(
            gemm2_kernel_list.values()
        )

        f_lookUpTable = os.path.join(self.working_path, "gemm_moe_ck2stages_lookup.h")

        with open(f_lookUpTable, "a") as f_lookup:
            for kernel in kernel_list:
                ## generate instance
                os.makedirs(os.path.join(self.working_path, "instances"), exist_ok=True)
                f_instance = os.path.join(
                    self.working_path, "instances", f"{kernel.name}.cu"
                )
                # if os.path.exists(f_instance):
                #     os.remove(f_instance)
                if self.quant_type in [4, 5]:
                    quanttype = "_blockscale"
                elif "FP4" in self.a_dtype:
                    if "bns" in tag:
                        quanttype = "_mxfp4_bns"
                    else:
                        quanttype = "_mxfp4"
                else:
                    quanttype = ""
                gemm1_fp32 = (
                    self.splitk and (kernel.stage == 1) and (quanttype == "_blockscale")
                )
                if not os.path.exists(f_instance):
                    with open(f_instance, "a") as f_ins:
                        # Select MNX variant if stage==2 and kernel has non-default MNPerXDLArg
                        use_mnx = (
                            kernel.stage == 2
                            and getattr(kernel, "MNPerXDLArg", 16) != 16
                        )
                        impl_template = STG_INSTANCE_IMPL_MNX if use_mnx else STG_INSTANCE_IMPL
                        fmt_kwargs = dict(
                            quanttype=quanttype,
                            A0DataType=self.a_dtype,
                            B0DataType=self.b_dtype,
                            AccDataType="F32" if self.a_dtype != "I8" else "I32",
                            EDataType="F32" if gemm1_fp32 else self.c_dtype,
                            CDEElementOp=kernel.CDEElementOp,
                            Nswizzle=str(self.nswizzle).lower(),
                            Quant=self.quant_type,
                            ActOP=(
                                ACT_OP_MAP[self.activation] if kernel.stage == 1 else 0
                            ),
                            Stage=kernel.stage,
                            BlockSize=kernel.BLOCK_SIZE,
                            MPerBlock=kernel.MPerBlock,
                            NPerBlock=kernel.NPerBlock,
                            KPerBlock=kernel.KPerBlock,
                            MWaves=kernel.MWaves,
                            NWaves=kernel.NWaves,
                            PipelineVer=kernel.GemmPipelineVersion,
                            MulRoutedWeight=str(
                                self.mul_routed_weight_stage == kernel.stage
                            ).lower(),
                        )
                        if use_mnx:
                            fmt_kwargs["MNPerXDLArg"] = kernel.MNPerXDLArg
                        stage_instance = impl_template.format(**fmt_kwargs)
                        if "FP4" in self.b_dtype:
                            stage_instance = (
                                "#ifndef __gfx942__\n" + stage_instance + "\n#endif\n"
                            )
                        f_ins.write(stage_instance)

                ## generate lookUpTable
                use_mnx = (
                    kernel.stage == 2 and getattr(kernel, "MNPerXDLArg", 16) != 16
                )
                lookup_template_choice = LOOKUP_template_MNX if use_mnx else LOOKUP_template
                lookup_kwargs = dict(
                    kernel_tag=kernel.name,
                    A0DataType=self.a_dtype,
                    B0DataType=self.b_dtype,
                    AccDataType="F32" if self.a_dtype != "I8" else "I32",
                    EDataType="F32" if gemm1_fp32 else self.c_dtype,
                    CDEElementOp=kernel.CDEElementOp,
                    Nswizzle=str(self.nswizzle).lower(),
                    Quant=self.quant_type,
                    ActOP=ACT_OP_MAP[self.activation] if kernel.stage == 1 else 0,
                    Stage=kernel.stage,
                    BlockSize=kernel.BLOCK_SIZE,
                    MPerBlock=kernel.MPerBlock,
                    NPerBlock=kernel.NPerBlock,
                    KPerBlock=kernel.KPerBlock,
                    MWaves=kernel.MWaves,
                    NWaves=kernel.NWaves,
                    PipelineVer=kernel.GemmPipelineVersion,
                    MulRoutedWeight=str(
                        self.mul_routed_weight_stage == kernel.stage
                    ).lower(),
                )
                if use_mnx:
                    lookup_kwargs["MNPerXDLArg"] = kernel.MNPerXDLArg
                lookup_ele = lookup_template_choice.format(**lookup_kwargs)
                f_lookup.write(lookup_ele)

        f_gemm1_heuristic_dispatch = os.path.join(
            self.working_path, "ck2stages_moe_stage1_heuristic_dispatch.hpp"
        )
        gemm1_heuristic_dispatch, gemm2_heuristic_dispatch = heuristic_dispatch_dict[
            tag
        ]
        with open(f_gemm1_heuristic_dispatch, "a") as f_h:
            gemm1_fp32 = self.splitk and (quanttype == "_blockscale")
            gemm1_heuristic_dispatch_str = gemm1_heuristic_dispatch.format(
                A0DataType=self.a_dtype,
                B0DataType=self.b_dtype,
                AccDataType="F32" if self.a_dtype != "I8" else "I32",
                EDataType="F32" if gemm1_fp32 else self.c_dtype,
                CDEElementOp=kernel_list[0].CDEElementOp,
                Nswizzle=str(self.nswizzle).lower(),
                Quant=self.quant_type,
                ActOP=str(ACT_OP_MAP[self.activation]),
                MulRoutedWeight=str(self.mul_routed_weight_stage == 1).lower(),
                Preshuffle=str(self.preshuffle).lower(),
            )
            f_h.write(gemm1_heuristic_dispatch_str)

        f_gemm2_heuristic_dispatch = os.path.join(
            self.working_path, "ck2stages_moe_stage2_heuristic_dispatch.hpp"
        )
        with open(f_gemm2_heuristic_dispatch, "a") as f_h:
            gemm2_heuristic_dispatch_str = gemm2_heuristic_dispatch.format(
                A0DataType=self.a_dtype,
                B0DataType=self.b_dtype,
                AccDataType="F32" if self.a_dtype != "I8" else "I32",
                EDataType=self.c_dtype,
                CDEElementOp=kernel_list[-1].CDEElementOp,
                Nswizzle=str(self.nswizzle).lower(),
                Quant=self.quant_type,
                ActOP=0,
                MulRoutedWeight=str(self.mul_routed_weight_stage == 2).lower(),
                Preshuffle=str(self.preshuffle).lower(),
            )
            f_h.write(gemm2_heuristic_dispatch_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ck 2stage gemm instance.")

    # Add arguments
    parser.add_argument(
        "-a",
        "--a_dtype",
        nargs="*",
        required=False,
        type=str,
        choices=["f8", "i8", "f16", "b16", "fp4x2"],
        help="select input dtype",
    )

    parser.add_argument(
        "-b",
        "--b_dtype",
        nargs="*",
        required=False,
        type=str,
        choices=["f8", "i8", "f16", "b16", "i4", "fp4x2"],
        help="select weight dtype",
    )

    parser.add_argument(
        "-c",
        "--c_dtype",
        default="b16",
        required=False,
        type=str,
        choices=["f16", "b16"],
        help="select out dtype",
    )

    parser.add_argument(
        "-q",
        "--quant_type",
        default="per_tensor",
        required=False,
        type=str,
        choices=[
            "per_tensor",
            "per_token",
            "per_128x128",
            "per_1x128",
            "per_1x32",
            "no",
        ],
        help="select quant_type",
    )

    parser.add_argument(
        "-act",
        "--activation",
        default="silu",
        required=False,
        type=str,
        choices=["silu", "gelu", "swiglustep"],
        help="select activation",
    )

    parser.add_argument(
        "-m",
        "--mul_routed_weight_stage",
        default=2,
        required=False,
        type=int,
        choices=[1, 2],
        help="select quant_type",
    )

    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    parser.add_argument(
        "-p",
        "--preshuffle",
        action="store_true",
        help="enable pre-shuffle weight mode",
    )

    parser.add_argument(
        "--issplitk",
        action="store_true",
        help="enable moe_stage1 splitk mode",
    )

    args = parser.parse_args()
    args.quant_type = (
        "per_1x128" if args.quant_type == "per_128x128" else args.quant_type
    )

    quant_dict = {
        "no": 0,
        "per_tensor": 1,
        "per_token": 2,
        "per_1x32": 3,
        "per_1x128": 4,
    }

    generate_instance_and_lookUpTable_head(args.working_path)
    # build all
    if args.b_dtype is None:
        # quanted moe
        b_quant_dtypes = ["f8", "i8", "i4", "fp4x2"]
        c_dtypes = ["f16", "b16"]
        acts = ["silu", "gelu"]
        routed_weight_l = [1, 2]
        general_quant_l = ["per_tensor", "per_token"]
        preshuffle_mode_l = [True, False]
        for (
            b_dtype,
            c_dtype,
            act,
            routed_weight,
            quant,
            preshuffle_mode,
        ) in itertools.product(
            b_quant_dtypes,
            c_dtypes,
            acts,
            routed_weight_l,
            general_quant_l,
            preshuffle_mode_l,
        ):
            a_dtype = b_dtype if b_dtype != "i4" else "f8"
            quant = quant if b_dtype != "fp4x2" else "per_1x32"
            preshuffle_mode = preshuffle_mode if quant == "per_1x32" else True
            codegen = ck_moe_2stage_gemm_codegen(
                args.working_path,
                a_dtype,
                b_dtype,
                c_dtype,
                quant_dict[quant],
                act,
                routed_weight,
                preshuffle_mode,
                False,  # splitk
            )
            codegen.generate_instance_and_lookUpTable()

        # blk-quant moe
        blk_quant_l = ["per_1x128"]
        blk_splitk_l = [False, True]
        for c_dtype, act, routed_weight, quant, splitk in itertools.product(
            c_dtypes, acts, routed_weight_l, blk_quant_l, blk_splitk_l
        ):
            codegen = ck_moe_2stage_gemm_codegen(
                args.working_path,
                "f8",
                "f8",
                c_dtype,
                quant_dict[quant],
                act,
                routed_weight,
                preshuffle_mode,
                splitk,
            )
            codegen.generate_instance_and_lookUpTable()

        # no-quant moe
        b_quant_dtypes = [
            "f16",
            "b16",
        ]
        for (
            b_dtype,
            act,
            routed_weight,
        ) in itertools.product(b_quant_dtypes, acts, routed_weight_l):
            c_dtype = a_dtype = b_dtype

            codegen = ck_moe_2stage_gemm_codegen(
                args.working_path,
                a_dtype,
                b_dtype,
                c_dtype,
                quant_dict["no"],
                act,
                routed_weight,
                preshuffle_mode,
                False,  # splitk
            )
            codegen.generate_instance_and_lookUpTable()

        # no-quant moe: swiglustep needs both preshuffle=True and preshuffle=False
        # (preshuffle=False is required for gfx950/MI350X which uses un-shuffled weights)
        for (
            b_dtype,
            routed_weight,
            preshuffle,
        ) in itertools.product(b_quant_dtypes, routed_weight_l, [True, False]):
            c_dtype = a_dtype = b_dtype

            codegen = ck_moe_2stage_gemm_codegen(
                args.working_path,
                a_dtype,
                b_dtype,
                c_dtype,
                quant_dict["no"],
                "swiglustep",
                routed_weight,
                preshuffle,
                False,  # splitk
            )
            codegen.generate_instance_and_lookUpTable()
    else:
        for b_dtype in args.b_dtype:
            a_dtype = b_dtype if b_dtype != "i4" else "f8"
            codegen = ck_moe_2stage_gemm_codegen(
                args.working_path,
                a_dtype,
                b_dtype,
                args.c_dtype,
                quant_dict[args.quant_type],
                args.activation,
                args.mul_routed_weight_stage,
                args.preshuffle,
                args.issplitk,
            )
            codegen.generate_instance_and_lookUpTable()

    generate_instance_and_lookUpTable_end(args.working_path)
