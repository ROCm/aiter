# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
"""

from packaging.version import Version

from .utils import is_flydsl_available
from .moe_common import GateMode

# Kimi-K3 note: the K3 FlyDSL MoE kernels (mixed_moe_gemm_2stage.py) call
# rocdl._split_mfma_operands(loc=...), an API flydsl removed in 0.2.4
# (0.2.2 still accepts loc=). rocm main bumped _MIN_FLYDSL_VERSION to 0.2.4
# for unrelated kernels; the K3 path needs 0.2.2, so pin it here.
_MIN_FLYDSL_VERSION = Version("0.2.2")

__all__ = [
    "is_flydsl_available",
    "GateMode",
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

    from .gemm_kernels import flydsl_hgemm, flydsl_preshuffle_gemm_a8
    from .moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    from .fmha_kernels import flydsl_flash_attn_func
    from .kernels.qk_norm_rope_quant import flydsl_qk_norm_rope_quant
    from .kernels.pa_mqa_logits_fp4 import (
        flydsl_pa_mqa_logits_fp4,
    )
    from .kernels.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_logits_fp4_prefill,
    )
    from .kernels.fp8_mqa_logits import (
        flydsl_fp8_mqa_logits,
        KERNEL_VARIANTS as FP8_MQA_LOGITS_VARIANTS,
        DEFAULT_VARIANT as FP8_MQA_LOGITS_DEFAULT_VARIANT,
    )

    # from .linear_attention_kernels import flydsl_gdr_decode

    __all__ += [
        "flydsl_preshuffle_gemm_a8",
        "flydsl_moe_stage1",
        "flydsl_moe_stage2",
        "flydsl_hgemm",
        "flydsl_flash_attn_func",
        "flydsl_qk_norm_rope_quant",
        "flydsl_pa_mqa_logits_fp4",
        "flydsl_pa_mqa_logits_fp4_prefill",
        "flydsl_fp8_mqa_logits",
        "FP8_MQA_LOGITS_VARIANTS",
        "FP8_MQA_LOGITS_DEFAULT_VARIANT",
        # "flydsl_gdr_decode",
    ]
