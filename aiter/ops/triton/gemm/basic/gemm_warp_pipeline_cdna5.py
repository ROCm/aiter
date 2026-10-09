# SPDX-License-Identifier: MIT
"""Native K128 CDNA5 warp-pipeline GEMM with M-contiguous A scales by default.

Requires upstream Triton with CDNA5 warp-pipeline support. Preparation accepts
canonical scales, or old Gluon/FlyDSL scale buffers with
``a_scales_transposed=True``. Output uses two padded strip buffers.
"""

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.warp_pipeline import (
    gemm_warp_pipeline_cdna5,
    prepare_gemm_warp_pipeline,
)

__all__ = ["gemm_warp_pipeline_cdna5", "prepare_gemm_warp_pipeline"]
