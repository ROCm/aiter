// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck_gemm_common.cuh"

using A0DataType          = F8;
using B0DataType          = I4;
using AccDataType         = F32;
using EDataType           = B16;
using CDEElementOp        = MulABScaleExpertWeightWin4;
const bool Nswizzle       = false;
const bool PerTensorQuant = false;

const auto V1 = ck::BlockGemmPipelineVersion::v1;
const auto V3 = ck::BlockGemmPipelineVersion::v3;
CK_MOE_STAGE2_GEMM_DEFINE(32, 128 / sizeof(A0DataType), 1, 4, V1, true)
CK_MOE_STAGE2_GEMM_DEFINE(64, 128 / sizeof(A0DataType), 1, 4, V1, true)
CK_MOE_STAGE2_GEMM_DEFINE(128, 128 / sizeof(A0DataType), 1, 4, V1, true)
CK_MOE_STAGE2_GEMM_DEFINE(256, 128 / sizeof(A0DataType), 1, 4, V1, true)

CK_MOE_STAGE2_GEMM_DEFINE(32, 128 / sizeof(A0DataType), 1, 4, V1, false)
CK_MOE_STAGE2_GEMM_DEFINE(64, 128 / sizeof(A0DataType), 1, 4, V1, false)
CK_MOE_STAGE2_GEMM_DEFINE(128, 128 / sizeof(A0DataType), 1, 4, V1, false)
CK_MOE_STAGE2_GEMM_DEFINE(256, 128 / sizeof(A0DataType), 1, 4, V1, false)

// CK_MOE_STAGE2_GEMM_DEFINE(128, 128/sizeof(A0DataType), 1, 4, V3,true)

// CK_MOE_STAGE2_GEMM_DEFINE(128, 128/sizeof(A0DataType), 1, 4, V3,false)