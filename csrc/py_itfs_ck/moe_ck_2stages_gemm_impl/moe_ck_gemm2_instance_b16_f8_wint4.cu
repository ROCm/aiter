// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck_gemm_common.cuh"

using A0DataType = F8;
using B0DataType = I4;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = MulABScaleExpertWeight;
const bool Nswizzle = false;
//CK_MOE_STAGE2_GEMM_DEFINE(32)
CK_MOE_STAGE2_GEMM_DEFINE(64)
CK_MOE_STAGE2_GEMM_DEFINE(128)
