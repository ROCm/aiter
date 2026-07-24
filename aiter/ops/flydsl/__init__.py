# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
"""

from packaging.version import Version

from .moe_common import GateMode
from .utils import is_flydsl_available

_MIN_FLYDSL_VERSION = Version("0.2.4")

__all__ = [
    "GateMode",
    "is_flydsl_available",
]

if is_flydsl_available():
    import flydsl as _flydsl

    installed_flydsl_version = getattr(_flydsl, "__version__", None)
    if installed_flydsl_version is None:
        raise ImportError(
            "`flydsl` is importable but its version cannot be determined."
        )

    _base_version = Version(installed_flydsl_version.split("+")[0])
    if _base_version < _MIN_FLYDSL_VERSION:
        raise ImportError(
            "Unsupported `flydsl` version: "
            f"expected >=`{_MIN_FLYDSL_VERSION}`, "
            f"got `{installed_flydsl_version}`."
        )

    from .fmha_kernels import flydsl_flash_attn_func
    from .gemm_kernels import flydsl_hgemm, flydsl_preshuffle_gemm_a8
    from .kernels.fp8_mqa_logits import (
        DEFAULT_VARIANT as FP8_MQA_LOGITS_DEFAULT_VARIANT,
    )
    from .kernels.fp8_mqa_logits import (
        KERNEL_VARIANTS as FP8_MQA_LOGITS_VARIANTS,
    )
    from .kernels.fp8_mqa_logits import (
        flydsl_fp8_mqa_logits,
    )
    from .kernels.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle import (
        flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
    )
    from .kernels.pa_mqa_logits_fp4 import (
        flydsl_pa_mqa_logits_fp4,
    )
    from .kernels.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_logits_fp4_prefill,
    )
    from .kernels.qk_norm_rope_quant import flydsl_qk_norm_rope_quant
    from .moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    # from .linear_attention_kernels import flydsl_gdr_decode

    __all__ += [
        "FP8_MQA_LOGITS_DEFAULT_VARIANT",
        "FP8_MQA_LOGITS_VARIANTS",
        "flydsl_flash_attn_func",
        "flydsl_fp8_mqa_logits",
        "flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle",
        "flydsl_hgemm",
        "flydsl_moe_stage1",
        "flydsl_moe_stage2",
        "flydsl_pa_mqa_logits_fp4",
        "flydsl_pa_mqa_logits_fp4_prefill",
        "flydsl_preshuffle_gemm_a8",
        "flydsl_qk_norm_rope_quant",
        # "flydsl_gdr_decode",
    ]
