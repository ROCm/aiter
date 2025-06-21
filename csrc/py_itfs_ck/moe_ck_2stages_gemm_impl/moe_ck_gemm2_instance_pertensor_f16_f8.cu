// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck_gemm_common.cuh"

using A0DataType = F8;
using B0DataType = F8;
using AccDataType = F32;
using EDataType = F16;
using CDEElementOp = MulABScaleExpertWeight;
const bool Nswizzle = false;
const bool PerTensorQuant = true;

const auto V1 = ck::BlockGemmPipelineVersion::v1;
const auto V3 = ck::BlockGemmPipelineVersion::v3;

CK_MOE_STAGE2_GEMM_DEFINE(32, 256/sizeof(A0DataType), 1, 4, V1, true)
CK_MOE_STAGE2_GEMM_DEFINE(64, 256/sizeof(A0DataType), 1, 4, V1, true)
CK_MOE_STAGE2_GEMM_DEFINE(128, 128/sizeof(A0DataType), 1, 4, V1, true)
CK_MOE_STAGE2_GEMM_DEFINE(256, 128/sizeof(A0DataType), 1, 4, V1, true)

CK_MOE_STAGE2_GEMM_DEFINE(32, 256/sizeof(A0DataType), 1, 4, V1, false)
CK_MOE_STAGE2_GEMM_DEFINE(64, 256/sizeof(A0DataType), 1, 4, V1, false)
CK_MOE_STAGE2_GEMM_DEFINE(128, 128/sizeof(A0DataType), 1, 4, V1, false)
CK_MOE_STAGE2_GEMM_DEFINE(256, 128/sizeof(A0DataType), 1, 4, V1, false)

#if defined(__gfx942__)
CK_MOE_STAGE2_GEMM_DEFINE(64, 128/sizeof(A0DataType), 1, 4, V3,true)

CK_MOE_STAGE2_GEMM_DEFINE(64, 128/sizeof(A0DataType), 1, 4, V3,false)
#endif

CK_MOE_STAGE2_GEMM_DEFINE(128, 128/sizeof(A0DataType), 1, 4, V3,true)

CK_MOE_STAGE2_GEMM_DEFINE(128, 128/sizeof(A0DataType), 1, 4, V3,false)

CK_MOE_STAGE2_GEMM_DEFINE(256, 128/sizeof(A0DataType), 1, 4, V3,true)

CK_MOE_STAGE2_GEMM_DEFINE(256, 128/sizeof(A0DataType), 1, 4, V3,false)


