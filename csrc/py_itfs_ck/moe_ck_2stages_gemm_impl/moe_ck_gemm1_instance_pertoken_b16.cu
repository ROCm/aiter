// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck_gemm_common.cuh"

using A0DataType = B16;
using B0DataType = B16;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = TypeCast;
const bool Nswizzle = false;
const bool PerTensorQuant = false;
CK_MOE_STAGE1_GEMM_DEFINE(32, 256/sizeof(A0DataType), 1, 4)
CK_MOE_STAGE1_GEMM_DEFINE(32, 128/sizeof(A0DataType), 1, 4)
CK_MOE_STAGE1_GEMM_DEFINE(64, 256/sizeof(A0DataType), 1, 4)
CK_MOE_STAGE1_GEMM_DEFINE(64, 128/sizeof(A0DataType), 1, 4)
CK_MOE_STAGE1_GEMM_DEFINE(128, 128/sizeof(A0DataType), 2, 2)

