// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck_gemm_common.cuh"

using A0DataType = F8;
using B0DataType = F8;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = MulABScale;

CK_MOE_STAGE1_GEMM_DEFINE(32, 256,  2, 2, false)
CK_MOE_STAGE1_GEMM_DEFINE(64, 256,  4, 2, false)
CK_MOE_STAGE1_GEMM_DEFINE(128, 128, 4, 4, false)
CK_MOE_STAGE1_GEMM_DEFINE(32, 256,  2, 2, true)
CK_MOE_STAGE1_GEMM_DEFINE(64, 256,  4, 2, true)
CK_MOE_STAGE1_GEMM_DEFINE(128, 128, 4, 4, true)
