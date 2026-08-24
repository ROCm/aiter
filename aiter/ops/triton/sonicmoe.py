# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SonicMoE: Pure-Triton grouped GEMM MoE with full autograd support.
# Ported from sonic-moe/sonicmoe/functional_rocm/.

from aiter.ops.triton._triton_kernels.moe.sonicmoe import (
    moe_TC_softmax_topk_layer,
    moe_general_routing_inputs,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.enums import (
    ActivationType as SonicMoEActivationType,
    is_glu as sonicmoe_is_glu,
)

__all__ = [
    "moe_TC_softmax_topk_layer",
    "moe_general_routing_inputs",
    "SonicMoEActivationType",
    "sonicmoe_is_glu",
]
