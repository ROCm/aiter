# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-owned MoE OperationPlan providers."""

from .stage1 import (
    FQ_ACTIVATION_OP_ID,
    INT4_STAGE1_GEMM_OP_ID,
    MIXED_STAGE1_GEMM_OP_ID,
    MoeStage1OperationCase,
    MoeStage1Role,
    Stage1ExternalPostprocessMetadata,
    Stage1FqMetadata,
    Stage1GemmMetadata,
    resolve_moe_stage1_operation_plan,
)

__all__ = [
    "FQ_ACTIVATION_OP_ID",
    "INT4_STAGE1_GEMM_OP_ID",
    "MIXED_STAGE1_GEMM_OP_ID",
    "MoeStage1OperationCase",
    "MoeStage1Role",
    "Stage1ExternalPostprocessMetadata",
    "Stage1FqMetadata",
    "Stage1GemmMetadata",
    "resolve_moe_stage1_operation_plan",
]
