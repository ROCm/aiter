# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What each FlyDSL AOT cache is built from: the one place that says so.

``aiter/ops/flydsl/aot_cache.py`` names an entry by the contents of the source
files and the values of the environment variables its cache declares, so an
edit to a declared file, or another value of a declared variable, makes that
cache miss and no other. A launcher only names its cache
(``@aot_factory("mixed_moe_gemm1")``); what that cache declares is here.

Paths are relative to ``aiter/ops/flydsl`` (or absolute); a directory stands
for every ``.py`` below it. The file that creates a cache is always one of
its sources, and so are ``COMMON_SOURCES``; ``COMMON_ENV`` are inputs of every
cache.

A file or a variable that is missing here invalidates nothing. A prebuild and
``tests/test_flydsl_aot_launch.py`` raise for the omissions they can see: a
declared file under ``kernels/`` that imports an undeclared one, and a
variable a declared file reads by a literal name that is neither declared
here, nor in the cache's fingerprint, nor in ``ENV_NOT_CODE``.

This module is data only (tuples, so that no cache can change another's
list) and imports nothing: ``aot_cache`` reads it when a cache is created.
"""

# Sources of every cache: the helpers nearly every kernel is built with, and
# the per-arch hardware constants kernels size themselves by (LDS capacity,
# CUs per XCD).
COMMON_SOURCES = (
    "kernels/kernels_common.py",
    "kernels/tensor_shim.py",
    "kernels/buffer_ops.py",
    "../../jit/utils/chip_info.py",
    "../../jit/utils/lds_capacity.py",
)

# Variables the common sources read: inputs of every cache.
COMMON_ENV = (
    "AITER_FLYDSL_KERNARG_PRELOAD",
    "AITER_FLYDSL_KERNARG_PRELOAD_COUNT",
)

# Variables aiter's FlyDSL code reads that cannot change generated code, or
# that reach an entry's identity another way. The check of a cache's sources
# (``aot_cache._undeclared_env``) accepts them without a declaration.
ENV_NOT_CODE = frozenset(
    {
        # Whether to launch; the cache's own switches.
        "COMPILE_ONLY",
        "AITER_FLYDSL_AOT_CACHE",
        "AITER_FLYDSL_AOT_CACHE_DIR",
        "AITER_FLYDSL_AOT_MAX_RETRIES",
        "AITER_FLYDSL_AOT_TIMEOUT",
        "AITER_FLYDSL_AOT_TRACEBACK",
        # The target: hashed as the arch and CU count they select.
        "FLYDSL_GPU_ARCH",
        "GPU_ARCHS",
        "AITER_GPU_ARCHS",
        "CU_NUM",
        "AITER_AOT_REPLAY",
        # Read by dispatch helpers of moe_common.py (which kernels import
        # for GateMode): what they choose reaches a kernel as a factory
        # argument (the output dtype, the SwiGLU limit), which is its key.
        "AITER_FLYDSL_STAGE2_FP8",
        "AITER_MXFP4_INTERMEDIATE",
        "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
        # A host-side argument audit of the GDN wrapper.
        "AITER_K5_OPT_CHECK",
        # FlyDSL's own cache.
        "FLYDSL_RUNTIME_CACHE_DIR",
        "FLYDSL_RUNTIME_ENABLE_CACHE",
        "FLYDSL_RUNTIME_RUN_ONLY",
    }
)

# Lists that several caches share.
_A16WMIX_MOE = (
    "kernels/moe_2stage_a16wmix",
    "kernels/act.py",
    "kernels/dpp_utils.py",
    "kernels/layout_utils.py",
    "kernels/mxfp4_gemm_common.py",
)
_GFX942_MOE = ("kernels/moe_gemm_2stage/gfx9_fp8",)
_MXSCALE_GEMM = (
    "gemm_a8w8_mxfp8_8wave.py",
    "gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py",
    "kernels/gemm",
    "kernels/mfma_preshuffle_pipeline.py",
    "kernels/communication_ops_utils.py",
)
_MX_MOE = (
    "kernels/mixed_moe_gemm_2stage_common.py",
    "kernels/act.py",
    "kernels/layout_utils.py",
    "kernels/mfma_epilogues.py",
    "kernels/mfma_preshuffle_pipeline.py",
    "moe_common.py",
)
_GFX942_MOE_ENV = (
    "AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED",
    "AITER_FLYDSL_MOE_BF16_RTE_SIMPLIFIED",
)

# Cache name -> {"sources": (...), "env": (...)}; either may be left out.
AOT_CACHES = {
    # GEMM
    "preshuffle_gemm_a8": {
        "sources": (
            "kernels/preshuffle_gemm.py",
            "kernels/mfma_preshuffle_pipeline.py",
            "kernels/splitk_epilogue.py",
            "kernels/communication_ops_utils.py",
            "kernels/gemm_a16w16_gfx950_utils.py",
        ),
    },
    "gemm_a8w8_blockscale": {
        "sources": (
            "gemm_tune/flydsl_gemm_a8w8_blockscale_common.py",
            "kernels/gemm_a8w8_blockscale_8wave_gfx942.py",
            "kernels/gemm_a8w8_blockscale_8wave_gfx950.py",
        ),
    },
    "gemm_a8w8_mxfp8_8wave": {
        "sources": (
            "kernels/mxfp8_gemm_8wave.py",
            "kernels/gemm_a8w8_8wave.py",
            "kernels/mfma_preshuffle_pipeline.py",
        ),
    },
    "gemm_a8w8_fp8_8wave": {
        "sources": (
            "kernels/gemm_a8w8_8wave.py",
            "kernels/mfma_preshuffle_pipeline.py",
        ),
    },
    "gemm_a8w8_int8_4wave": {
        "sources": (
            "kernels/gemm_a8w8_8wave.py",
            "kernels/mfma_preshuffle_pipeline.py",
            "kernels/gemm_a8w8_int8_4wave.py",
        ),
    },
    "mxscale_preshuffle_gemm": {
        "sources": _MXSCALE_GEMM,
    },
    "mxscale_preshuffle_gemm_fused": {
        "sources": _MXSCALE_GEMM,
    },
    "mxscale_preshuffle_splitk_reduce": {
        "sources": _MXSCALE_GEMM,
    },
    "bmm_a8w8_mxscale_gfx950": {
        "sources": (
            "kernels/bmm_a8w8_mxscale_gfx950.py",
            "kernels/mfma_preshuffle_pipeline.py",
            "../../utility/mx_types.py",
            "kernels/quant_utils.py",
        ),
    },
    "gemm_a16w16_gfx950": {
        "sources": ("kernels/gemm_a16w16_gfx950_utils.py",),
    },
    "gemm_a16w16_band_gfx950": {},
    "gemm_a16w16_skinny_gfx950": {},
    # 2-stage MoE
    "moe_gfx942_gemm": {
        "sources": _GFX942_MOE,
        "env": _GFX942_MOE_ENV,
    },
    "moe_gfx942_sorted_sum": {
        "sources": _GFX942_MOE,
        "env": _GFX942_MOE_ENV,
    },
    "moe_gfx942_absmax": {
        "sources": _GFX942_MOE,
        "env": _GFX942_MOE_ENV,
    },
    "moe_gfx942_quant_per_tensor": {
        "sources": _GFX942_MOE,
        "env": _GFX942_MOE_ENV,
    },
    "moe_a16wmix_gemm1": {
        "sources": _A16WMIX_MOE,
    },
    "moe_a16wmix_gemm2": {
        "sources": _A16WMIX_MOE,
    },
    "mixed_moe_gemm1": {
        "sources": _MX_MOE,
    },
    "mixed_moe_gemm2": {
        "sources": _MX_MOE,
    },
    "mixed_fhmoe_gemm1": {
        "sources": _MX_MOE,
    },
    "mixed_fhmoe_gemm2": {
        "sources": _MX_MOE,
    },
    "mxfp4_moe_gemm1": {
        "sources": (
            "kernels/mxfp4_gemm1.py",
            "kernels/mxfp4_gemm_common.py",
            "kernels/act.py",
            "kernels/dpp_utils.py",
            "kernels/layout_utils.py",
        ),
    },
    "mxmoe_gemm2": {
        "sources": (
            "kernels/mxmoe_gemm_v2.py",
            "kernels/mxfp4_gemm_common.py",
            "kernels/dpp_utils.py",
            "kernels/act.py",
            "kernels/layout_utils.py",
        ),
        "env": (
            "MXFP4_G2_ASCALE_PF",
            "MXFP4_G2_BF16_LDS",
            "MXFP4_G2_BHOIST",
            "MXFP4_G2_PREFETCH_IDS",
            "MXFP4_G2_SPART",
        ),
    },
    "moe_reduction": {
        "sources": (
            "kernels/mxfp4_gemm_common.py",
            "kernels/act.py",
            "kernels/dpp_utils.py",
            "kernels/layout_utils.py",
        ),
    },
    # Attention, GDN, conv3d
    "pa_decode_plan": {},
    "chunk_gated_delta_h": {
        "sources": (
            "kernels/gdr_prefill/chunk_gated_delta_h.py",
            "kernels/gdr_common.py",
        ),
        "env": ("FLYDSL_K5_OPT_BV",),
    },
    "gdn_fused_prefill": {},
    "flash_attn_fp8_gfx950": {
        "sources": ("kernels/fmha_gfx950",),
    },
    "conv3d_implicit_gfx950": {
        "sources": (
            "kernels/conv/conv3d_gfx950_utils.py",
            "kernels/conv/conv3d_im2col.py",
        ),
    },
    "conv3d_transpose_ncdhw_ndhwc": {
        "sources": ("kernels/conv/conv3d_gfx950_utils.py",),
    },
    # gfx950 intranode MegaMoE
    "mega_moe_prepare": {
        "sources": (
            "kernels/communication_ops_utils.py",
            "kernels/dpp_utils.py",
            "kernels/mega_moe/dispatch.py",
            "kernels/mega_moe/quant.py",
        ),
    },
    "mega_moe_stage1_bundle": {
        "sources": (
            "kernels/act.py",
            "kernels/communication_ops_utils.py",
            "kernels/dpp_utils.py",
            "kernels/mega_moe/dispatch.py",
            "kernels/mega_moe/gemm1.py",
            "kernels/mega_moe/gemm_util.py",
            "kernels/mega_moe/mega_moe_config.py",
        ),
        "env": ("AITER_MEGA_FIXED_SLOT_MAX_MTPR",),
    },
    "mega_moe_small_m_stage1": {
        "sources": (
            "kernels/act.py",
            "kernels/communication_ops_utils.py",
            "kernels/dpp_utils.py",
            "kernels/flydsl_dispatch_combine_intranode_kernel.py",
            "kernels/flydsl_dispatch_combine_intranode_op.py",
            "kernels/layout_utils.py",
            "kernels/mxfp4_gemm_common.py",
            "kernels/mega_moe/dispatch.py",
            "kernels/mega_moe/gemm1.py",
            "kernels/mega_moe/gemm2.py",
            "kernels/mega_moe/gemm_util.py",
            "kernels/mega_moe/mega_moe_config.py",
            "kernels/mega_moe/mega_moe_fused_gemm.py",
            "kernels/mega_moe/mega_moe_stage1.py",
            "kernels/mega_moe/mega_moe_stage2.py",
            "kernels/mega_moe/weights.py",
        ),
        "env": ("AITER_MEGA_FIXED_SLOT_MAX_MTPR",),
    },
    "mega_moe_large_m_stage1": {
        "sources": (
            "kernels/act.py",
            "kernels/communication_ops_utils.py",
            "kernels/dpp_utils.py",
            "kernels/flydsl_dispatch_combine_intranode_kernel.py",
            "kernels/flydsl_dispatch_combine_intranode_op.py",
            "kernels/layout_utils.py",
            "kernels/mxfp4_gemm_common.py",
            "kernels/mega_moe/dispatch.py",
            "kernels/mega_moe/gemm1.py",
            "kernels/mega_moe/gemm2.py",
            "kernels/mega_moe/gemm_util.py",
            "kernels/mega_moe/mega_moe_config.py",
            "kernels/mega_moe/mega_moe_fused_gemm.py",
            "kernels/mega_moe/mega_moe_stage1.py",
            "kernels/mega_moe/mega_moe_stage2.py",
            "kernels/mega_moe/quant.py",
            "kernels/mega_moe/weights.py",
        ),
        "env": ("AITER_MEGA_FIXED_SLOT_MAX_MTPR",),
    },
    "mega_moe_stage2": {
        "sources": (
            "kernels/act.py",
            "kernels/communication_ops_utils.py",
            "kernels/dpp_utils.py",
            "kernels/layout_utils.py",
            "kernels/mxfp4_gemm_common.py",
            "kernels/mega_moe/gemm2.py",
            "kernels/mega_moe/gemm_util.py",
        ),
    },
    "mega_moe_stage2_aligned_pair": {
        "sources": (
            "kernels/act.py",
            "kernels/communication_ops_utils.py",
            "kernels/dpp_utils.py",
            "kernels/layout_utils.py",
            "kernels/mxfp4_gemm_common.py",
            "kernels/mega_moe/gemm2.py",
            "kernels/mega_moe/gemm_util.py",
            "kernels/mega_moe/mega_moe_stage2.py",
        ),
    },
    "mega_moe_mx_quant": {},
    "mega_moe_combine": {
        "sources": (
            "kernels/communication_ops_utils.py",
            "kernels/flydsl_dispatch_combine_intranode_kernel.py",
        ),
    },
}
