# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL."""

from importlib import import_module

import flydsl as _flydsl
from packaging.version import Version

from .moe_common import GateMode

_MIN_FLYDSL_VERSION = Version("0.2.4")

installed_flydsl_version = getattr(_flydsl, "__version__", None)
if installed_flydsl_version is None:
    raise ImportError("`flydsl` is importable but its version cannot be determined.")

_base_version = Version(installed_flydsl_version.split("+")[0])
if _base_version < _MIN_FLYDSL_VERSION:
    raise ImportError(
        "Unsupported `flydsl` version: "
        f"expected >=`{_MIN_FLYDSL_VERSION}`, "
        f"got `{installed_flydsl_version}`."
    )

_LAZY_IMPORTS = {
    "FP8_MQA_LOGITS_DEFAULT_VARIANT": (
        ".kernels.mqa_logits.fp8_mqa_logits",
        "DEFAULT_VARIANT",
    ),
    "FP8_MQA_LOGITS_VARIANTS": (
        ".kernels.mqa_logits.fp8_mqa_logits",
        "KERNEL_VARIANTS",
    ),
    "compute_varqlen_windows": (
        ".kernels.mqa_logits.pa_mqa_logits_fp4_prefill",
        "compute_varqlen_windows",
    ),
    "flydsl_flash_attn_func": (".fmha_kernels", "flydsl_flash_attn_func"),
    "flydsl_fp8_mqa_logits": (
        ".kernels.mqa_logits.fp8_mqa_logits",
        "flydsl_fp8_mqa_logits",
    ),
    "flydsl_hgemm": (".gemm_kernels", "flydsl_hgemm"),
    "flydsl_mla_reduce_v1": (".mla_reduce_kernels", "flydsl_mla_reduce_v1"),
    "flydsl_moe_stage1": (".moe_kernels", "flydsl_moe_stage1"),
    "flydsl_moe_stage2": (".moe_kernels", "flydsl_moe_stage2"),
    "flydsl_pa_mqa_logits_fp4": (
        ".kernels.mqa_logits.pa_mqa_logits_fp4",
        "flydsl_pa_mqa_logits_fp4",
    ),
    "flydsl_pa_mqa_logits_fp4_prefill": (
        ".kernels.mqa_logits.pa_mqa_logits_fp4_prefill",
        "flydsl_pa_mqa_logits_fp4_prefill",
    ),
    "flydsl_pa_mqa_logits_fp4_varqlen": (
        ".kernels.mqa_logits.pa_mqa_logits_fp4_prefill",
        "flydsl_pa_mqa_logits_fp4_varqlen",
    ),
    "flydsl_preshuffle_gemm_a8": (
        ".gemm_kernels",
        "flydsl_preshuffle_gemm_a8",
    ),
    "flydsl_qk_norm_rope_quant": (
        ".kernels.qk_norm_rope_quant",
        "flydsl_qk_norm_rope_quant",
    ),
}

__all__ = [
    "FP8_MQA_LOGITS_DEFAULT_VARIANT",
    "FP8_MQA_LOGITS_VARIANTS",
    "GateMode",
    "compute_varqlen_windows",
    "flydsl_flash_attn_func",
    "flydsl_fp8_mqa_logits",
    "flydsl_hgemm",
    "flydsl_mla_reduce_v1",
    "flydsl_moe_stage1",
    "flydsl_moe_stage2",
    "flydsl_pa_mqa_logits_fp4",
    "flydsl_pa_mqa_logits_fp4_prefill",
    "flydsl_pa_mqa_logits_fp4_varqlen",
    "flydsl_preshuffle_gemm_a8",
    "flydsl_qk_norm_rope_quant",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
