# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused a16w4/a16wi4/a16w16 (bf16 A x mxfp4/int4/bf16 W) 2-stage MoE kernels.

Standalone CDNA4 (gfx950) MFMA pipeline. bf16 A (no A-scale), W1/W2 upconverted
to bf16 in-kernel, non-scaled ``MFMA(16,16,32,bf16)``:

  - stage1 (:mod:`gemm1`): fused gate+up GEMM + SiLU/SiTUv2 -> bf16 intermediate
    ``[sorted_size, inter_dim]`` stored by sorted position (no requant, no scale).
  - stage2 (:mod:`gemm2`): down-projection GEMM + routing-weighted atomic bf16
    scatter to ``[tokens, model_dim]``.

Reuses the standard sorting/cumsum/m_indices contract and the
shuffle_weight+e8m0_shuffle W layout. Self-contained: shared helpers live in
:mod:`common`.
"""

from .gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid
from .host import (
    a16wi4_recommend_block_m,
    a16wi4_scale_to_kernel_layout,
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
    pick_a16w4_config,
    resolve_a16w4_gemm1_config,
    resolve_a16w4_gemm2_config,
)

__all__ = [
    "a16wi4_recommend_block_m",
    "a16wi4_scale_to_kernel_layout",
    "compile_gemm1_a16w4_port",
    "compile_gemm2_a16w4_port",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "gemm1_a16w4_grid",
    "gemm2_a16w4_grid",
    "pick_a16w4_config",
    "resolve_a16w4_gemm1_config",
    "resolve_a16w4_gemm2_config",
]
