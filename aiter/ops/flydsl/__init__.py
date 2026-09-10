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
    "FP4_LITETOPK_SUPPORTED_TOPKS": (
        ".pa_mqa_litetopk_fp4",
        "FP4_LITETOPK_SUPPORTED_TOPKS",
    ),
    "FP4LiteTopKResult": (
        ".pa_mqa_litetopk_fp4",
        "FP4LiteTopKResult",
    ),
    "FP4LiteTopKWorkspace": (
        ".pa_mqa_litetopk_fp4",
        "FP4LiteTopKWorkspace",
    ),
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
    "allocate_fp4_litetopk_workspace": (
        ".pa_mqa_litetopk_fp4",
        "allocate_fp4_litetopk_workspace",
    ),
    "fp4_litetopk_workspace_nbytes": (
        ".pa_mqa_litetopk_fp4",
        "fp4_litetopk_workspace_nbytes",
    ),
    "fp4_litetopk_workspace_size": (
        ".pa_mqa_litetopk_fp4",
        "fp4_litetopk_workspace_size",
    ),
    "flydsl_pa_mqa_litetopk_fp4_prefill": (
        ".pa_mqa_litetopk_fp4",
        "flydsl_pa_mqa_litetopk_fp4_prefill",
    ),
    "prepare_fp4_litetopk_seed": (
        ".pa_mqa_litetopk_fp4",
        "prepare_fp4_litetopk_seed",
    ),
    "flydsl_flash_attn_func": (".fmha_kernels", "flydsl_flash_attn_func"),
    "flydsl_fp8_mqa_logits": (
        ".kernels.mqa_logits.fp8_mqa_logits",
        "flydsl_fp8_mqa_logits",
    ),
    "flydsl_hgemm": (".gemm_kernels", "flydsl_hgemm"),
    "flydsl_hstu_attention_fwd": (
        ".hstu_attention_kernels",
        "flydsl_hstu_attention_fwd",
    ),
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
    "FP4_LITETOPK_SUPPORTED_TOPKS",
    "FP8_MQA_LOGITS_DEFAULT_VARIANT",
    "FP8_MQA_LOGITS_VARIANTS",
    "FP4LiteTopKResult",
    "FP4LiteTopKWorkspace",
    "GateMode",
    "allocate_fp4_litetopk_workspace",
    "compute_varqlen_windows",
    "flydsl_flash_attn_func",
    "flydsl_fp8_mqa_logits",
    "flydsl_hgemm",
    "flydsl_hstu_attention_fwd",
    "flydsl_mla_reduce_v1",
    "flydsl_moe_stage1",
    "flydsl_moe_stage2",
    "flydsl_pa_mqa_litetopk_fp4_prefill",
    "flydsl_pa_mqa_logits_fp4",
    "flydsl_pa_mqa_logits_fp4_prefill",
    "flydsl_pa_mqa_logits_fp4_varqlen",
    "flydsl_preshuffle_gemm_a8",
    "flydsl_qk_norm_rope_quant",
    "fp4_litetopk_workspace_nbytes",
    "fp4_litetopk_workspace_size",
    "prepare_fp4_litetopk_seed",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
