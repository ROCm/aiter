# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Recurrence-level Kimi Delta Attention with chunkwise context parallelism.

``kda_recurrence``           The KDA core as one autograd op on aiter's Triton or
                             FlyDSL chunk fwd/bwd, with ``cp_context``.
``chunk_kda``                FLA / Megatron ``kda_backend`` call contract.
``patch_megatron_kda_backend``  Registers ``kda_backend="aiter"`` in Megatron-Core.
``build_kda_cp_context``     FLA-compatible contiguous CP partition, without FLA.
"""

from aiter.ops.kda.context_parallel import (
    KdaCPBoundary,
    KdaCPContext,
    all_gather_boundary_packets,
    build_kda_cp_context,
    kda_cp_boundary,
    kda_state_chain,
)
from aiter.ops.kda.megatron import chunk_kda, make_chunk_kda, patch_megatron_kda_backend
from aiter.ops.kda.recurrence import (
    KdaKernels,
    KdaOptions,
    get_kda_kernels,
    kda_cp_backward_finish,
    kda_cp_backward_packet,
    kda_cp_forward_finish,
    kda_cp_forward_local,
    kda_recurrence,
    kda_transition,
)

__all__ = [
    "KdaCPBoundary",
    "KdaCPContext",
    "KdaKernels",
    "KdaOptions",
    "all_gather_boundary_packets",
    "build_kda_cp_context",
    "chunk_kda",
    "get_kda_kernels",
    "kda_cp_backward_finish",
    "kda_cp_backward_packet",
    "kda_cp_boundary",
    "kda_cp_forward_finish",
    "kda_cp_forward_local",
    "kda_recurrence",
    "kda_state_chain",
    "kda_transition",
    "make_chunk_kda",
    "patch_megatron_kda_backend",
]
