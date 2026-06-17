# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 codegen -- emit launchers for gfx942-targeted kid families."""

import os
from pathlib import Path

from opus_gemm_common import OpusGemmInstance

from codegen.common import (
    WARP_SIZE,
    _GFX942_A16W16_TAGS,
    _NOSPLIT,
    _SPLITK,
    W3_KERNEL_PAIRS,
    register_arch_map,
    register_emit,
)
from codegen.template_env import render as _render


# gfx942 pipeline header derived from W3_KERNEL_PAIRS: splitk_X reuses
# nosplit_X's .cuh (paired template); splitk_fused has its own.
def _gfx942_pipeline(tag):
    return f"gfx942/opus_gemm_pipeline_{tag}.cuh"


# Traits header carries the traits struct + kargs struct definitions for a given pipeline tag.
GFX942_TRAITS_HEADER = "gfx942/opus_gemm_traits_a16w16.cuh"

# gfx942 a16w16 tags all share one traits class name (no arch suffix).
GFX942_TRAITS_NAME = "opus_gemm_a16w16_traits"

# gfx942 a16w16 family supports only the 16x16x16 BF16 MFMA shape.
VALID_GFX942_BF16_MFMA = {(16, 16, 16)}

PIPELINE_HEADER_MAP = {
    "a16w16_fused_reduce": _gfx942_pipeline("a16w16_fused_reduce"),
    "a16w16_kbuf1_large_tile": _gfx942_pipeline("a16w16_kbuf1_large_tile"),
    **{nosplit: _gfx942_pipeline(nosplit) for nosplit in _NOSPLIT},
    **{
        splitk: _gfx942_pipeline(nosplit) for nosplit, splitk in W3_KERNEL_PAIRS.items()
    },
}

TRAITS_HEADER_MAP = {tag: GFX942_TRAITS_HEADER for tag in _GFX942_A16W16_TAGS}

TRAITS_NAME_MAP = {tag: GFX942_TRAITS_NAME for tag in _GFX942_A16W16_TAGS}

KARGS_NAME_MAP = {
    "a16w16_fused_reduce": "opus_gemm_splitk_fused_kargs",
    "a16w16_kbuf1_large_tile": "opus_gemm_noscale_kargs",
    **{tag: "opus_gemm_splitk_kargs" for tag in _SPLITK},
    **{tag: "opus_gemm_noscale_kargs" for tag in _NOSPLIT},
}

KERNEL_FUNC_MAP = {
    "a16w16_fused_reduce": "gemm_a16w16_fused_reduce_kernel",
    "a16w16_kbuf1_large_tile": "gemm_a16w16_kbuf1_large_tile_kernel",
    # gfx942 paired tags: nosplit_tag's kernel symbol; splitk_tag reuses it.
    **{nosplit: f"gemm_{nosplit}_kernel" for nosplit in W3_KERNEL_PAIRS.keys()},
    **{splitk: f"gemm_{nosplit}_kernel" for nosplit, splitk in W3_KERNEL_PAIRS.items()},
}

register_arch_map("gfx942", "pipeline_header", PIPELINE_HEADER_MAP)
register_arch_map("gfx942", "traits_header", TRAITS_HEADER_MAP)
register_arch_map("gfx942", "traits_name", TRAITS_NAME_MAP)
register_arch_map("gfx942", "kargs_name", KARGS_NAME_MAP)
register_arch_map("gfx942", "kernel_func", KERNEL_FUNC_MAP)


def gen_splitk_gfx942_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    fused,
    kargs_template_vars,
    SPLITK_REDUCE_FAST_ARCHES,
    V3_NVEC_ROWS,
    V2_SUPPORTED_SPLITKS,
    A16W16_TUNE_HOST_EXTRA,
    record_one_instantiation,
    **_unused,
):
    """gfx942 a16w16 splitk launcher emit (fused / non-fused)."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    err_label = "a16w16_fused_reduce" if fused else k.kernel_tag
    v2_enabled = (not fused) and (k.arch_prefix in SPLITK_REDUCE_FAST_ARCHES)
    target_wg_expr = "2 * cu_cached" if k.kernel_tag.endswith("_p1") else "cu_cached"
    INSTANCE_IMPL = _render(
        "impl_splitk_gfx942.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
        k=k,
        traits_name=traits_name,
        kargs_name=kargs_name,
        kernel_func=kernel_func,
        err_label=err_label,
        fused=fused,
        v2_enabled=v2_enabled,
        V3_NVEC_ROWS=V3_NVEC_ROWS,
        V2_SUPPORTED_SPLITKS=V2_SUPPORTED_SPLITKS,
        da=da,
        db=db,
        target_wg_expr=target_wg_expr,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    if fused:
        # fused: instantiate both Y dtypes (bf16, float) so runtime dispatch links.
        for CDtype in k.output_dtypes:
            cg._host_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "host_extra_params": A16W16_TUNE_HOST_EXTRA,
                }
            )
            # fused kernel has a second D_OUT template param for in-kernel Y-dtype dispatch.
            cg._device_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "kernel_func": kernel_func,
                    "kargs_name": kargs_name,
                    "kargs_explicit_param": ", __bf16",
                    "extra_device_decls": [
                        {
                            "kargs_explicit_param": ", float",
                        }
                    ],
                }
            )
    else:
        record_one_instantiation(
            cg,
            k,
            kernel_func,
            kargs_name,
            A16W16_TUNE_HOST_EXTRA,
            kargs_explicit_param,
        )


def gen_a16w16_nosplit_gfx942_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    kargs_template_vars,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    A16W16_TUNE_TAGS,
    **_unused,
):
    """gfx942 a16w16 non-splitK launcher emit (kbuf1_large_tile / kbuf3 / kbuf2v /
    kbuf2v_bk128 / kbuf1). Single Traits, no bias, no cachectl, no HAS_OOB tail."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_tune_tags = k.kernel_tag in A16W16_TUNE_TAGS
    INSTANCE_IMPL = _render(
        "impl_nosplit_gfx942.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
        kernel_func=kernel_func,
        k=k,
        traits_name=traits_name,
        kargs_name=kargs_name,
        has_tune_tags=has_tune_tags,
        da=da,
        db=db,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    inst_extra_param = (
        ",\n    std::optional<aiter_tensor_t>,\n    int"
        if k.kernel_tag in A16W16_TUNE_TAGS
        else ""
    )
    for CDtype in k.output_dtypes:
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_extra_params": inst_extra_param}
        )
        cg._device_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "kernel_func": kernel_func,
                "kargs_name": kargs_name,
                "kargs_explicit_param": kargs_explicit_param,
            }
        )


# ---------- Self-register at import time ----------
# gfx942 splitk family: 5 tags.
_GFX942_SPLITK_TAGS = (
    "a16w16_kbuf3_sk",
    "a16w16_kbuf2v_sk",
    "a16w16_kbuf2v_bk128_sk",
    "a16w16_kbuf1_sk",
    "a16w16_fused_reduce",
)
for _tag in _GFX942_SPLITK_TAGS:
    register_emit("gfx942", _tag, gen_splitk_gfx942_instance)

# gfx942 a16w16 non-splitK family: 5 tags.
_GFX942_NOSPLIT_TAGS = (
    "a16w16_kbuf1_large_tile",
    "a16w16_kbuf3",
    "a16w16_kbuf2v",
    "a16w16_kbuf2v_bk128",
    "a16w16_kbuf1",
)
for _tag in _GFX942_NOSPLIT_TAGS:
    register_emit("gfx942", _tag, gen_a16w16_nosplit_gfx942_instance)


# ---------------- gfx942 a16w16 validator ----------------
# Coverage: basic physical limits only. Detailed LDS depth / layout checks
# live in gfx942/opus_gemm_traits_a16w16.cuh static_asserts (hipcc enforces).

# gfx942 (CDNA3 / MI300X) hardware LDS budget per WG.
_GFX942_LDS_PER_WG_BYTES = 64 * 1024


def _validate_a16w16_gfx942(k: OpusGemmInstance):
    """Validate a gfx942 a16w16 instance -- basic physical limits only."""
    errors = []

    # MFMA shape: gfx942 a16w16 family is locked to 16x16x16 BF16.
    if (k.W_M, k.W_N, k.W_K) not in VALID_GFX942_BF16_MFMA:
        errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_GFX942_BF16_MFMA}")

    # BLOCK_SIZE physical cap (hardware: 1024 max; gfx942 a16w16 we cap at 512).
    if k.BLOCK_SIZE > 512:
        errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} exceeds 512")

    # AGPR/VGPR register-file caps (hardware: 256 each, 512 combined).
    E_M = (k.B_M // 2) // (k.W_M * k.T_M) if (k.W_M * k.T_M) else 0
    E_N = (k.B_N // 2) // (k.W_N * k.T_N) if (k.W_N * k.T_N) else 0
    E_K = k.B_K // k.W_K if k.W_K else 0
    agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
    total_agprs = 4 * E_M * E_N * agpr_per_mfma
    vgpr_est = 4 * E_K * (E_M + 2 * E_N) + 80
    if total_agprs >= 256:
        errors.append(f"AGPR={total_agprs} must be < 256")
    if vgpr_est > 256:
        errors.append(f"VGPR_est={vgpr_est} exceeds 256")
    if vgpr_est + total_agprs > 512:
        errors.append(f"VGPR+AGPR={vgpr_est + total_agprs} exceeds 512")

    # Loose LDS bound: 2 * B_M * B_K + 2 * B_N * B_K bytes for bf16 (1-deep
    # per slot, ignores pipeline depth + padding). Anything past 64 KiB is
    # physically impossible; finer-grained checks live in traits.cuh.
    lds_min_bytes = 2 * (k.B_M + k.B_N) * k.B_K
    if lds_min_bytes > _GFX942_LDS_PER_WG_BYTES:
        errors.append(
            f"LDS lower bound={lds_min_bytes // 1024}KiB exceeds "
            f"{_GFX942_LDS_PER_WG_BYTES // 1024}KiB (gfx942 budget)"
        )

    if errors:
        msg = f"Invalid gfx942 a16w16 instance '{k.name}':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    return {
        "E_M": E_M,
        "E_N": E_N,
        "E_K": E_K,
        "agprs": total_agprs,
        "vgpr_est": vgpr_est,
        "lds_bytes": lds_min_bytes,
        "min_k": 2 * k.B_K,
    }
