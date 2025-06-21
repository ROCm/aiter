// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck_gemm_common_mxfp4.cuh"

using A0DataType       = F4;
using A1DataType       = XDataType;
using B0DataType       = F4;
using B1DataType       = XDataType;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = MulABScaleExpertWeight;
const auto V1 = ck::BlockGemmPipelineVersion::v1;
const auto V3 = ck::BlockGemmPipelineVersion::v3;

const bool Nswizzle = false;
const bool PerTensorQuant = true;

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V1, true, 0);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V1, true, 0);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V1, true, 0);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V1, true, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V1, false, 0);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V1, false, 0);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V1, false, 0);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V1, false, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V1, true, 1);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V1, true, 1);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V1, true, 1);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V1, true, 1);


CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V1, false, 1);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V1, false, 1);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V1, false, 1);
CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V1, false, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V3, true, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V3, false, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V3, true, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(32, 128/sizeof(A0DataType), 1, 4, V3, false, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V3, true, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V3, false, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V3, true, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(64, 128/sizeof(A0DataType), 2, 2,  V3, false, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V3, true, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V3, false, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V3, true, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(128, 128/sizeof(A0DataType), 2, 2,  V3, false, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V3, true, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V3, false, 0);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V3, true, 1);

CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(256, 128/sizeof(A0DataType), 2, 2,  V3, false, 1);




