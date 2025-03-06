// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck_gemm_common.cuh"

using A0DataType = F8;
using B0DataType = F8;
using AccDataType = F32;
using EDataType = F16;
using CDEElementOp = MulABScale;

CK_MOE_STAGE1_GEMM_DEFINE(32, 256,  1, 1, false)
CK_MOE_STAGE1_GEMM_DEFINE(64, 256,  2, 1, false)
CK_MOE_STAGE1_GEMM_DEFINE(128, 128, 2, 2, false)
// CK_MOE_STAGE1_GEMM_DEFINE(32, 256,  1, 1, true)
// CK_MOE_STAGE1_GEMM_DEFINE(64, 256,  2, 1, true)
// CK_MOE_STAGE1_GEMM_DEFINE(128, 128, 2, 2, true)