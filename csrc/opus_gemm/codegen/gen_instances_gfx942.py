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


# gfx942 pipeline header derived from W3_KERNEL_PAIRS: splitk_X reuses
# nosplit_X's .cuh (paired template).
def _gfx942_pipeline(tag):
    return f"gfx942/opus_gemm_pipeline_{tag}.cuh"


# Traits header carries the traits struct + kargs struct definitions for a given pipeline tag.
GFX942_TRAITS_HEADER = "gfx942/opus_gemm_traits_a16w16.cuh"

# gfx942 a16w16 tags all share one traits class name (no arch suffix).
GFX942_TRAITS_NAME = "opus_gemm_a16w16_traits"

# gfx942 a16w16 family supports only the 16x16x16 BF16 MFMA shape.
VALID_GFX942_BF16_MFMA = {(16, 16, 16)}

# SplitK tags normally use host tile geometry; overrides remap device traits.
GFX942_SPLITK_TRAITS_OVERRIDES = {
    "a16w16_em3en4_lds1_pgr2_sk": {
        "block": (96, 128),
        "lds_depth": 1,
    },
}

GFX942_QUAD_MFMA32_SPLITK_TAG = "a16w16_quad_mfma32_kbuf1_sk"
GFX942_SPLITK_TAGS = _SPLITK + ("a16w16_em3en4_lds1_pgr2_sk",)
GFX942_EVEN_LOOP_SPLITK_TAGS = (
    "a16w16_kbuf2v_sk",
    "a16w16_kbuf2v_bk128_sk",
    GFX942_QUAD_MFMA32_SPLITK_TAG,
)


def _splitk_traits_geometry(k):
    override = GFX942_SPLITK_TRAITS_OVERRIDES.get(k.kernel_tag)
    if override is None:
        return k.B_M, k.B_N, 2
    trait_bm, trait_bn = override["block"]
    return trait_bm, trait_bn, override["lds_depth"]


def _splitk_workspace_types(k):
    dtype = getattr(k, "splitk_workspace_dtype", "fp32_t")
    if dtype == "bf16_t":
        return "bf16_t", "__bf16"
    return "fp32_t", "float"


def _uses_bf16_workspace(k):
    return getattr(k, "splitk_workspace_dtype", "fp32_t") == "bf16_t"


PIPELINE_HEADER_MAP = {
    "a16w16_em3en4_lds1_pgr2_sk": _gfx942_pipeline("a16w16_em3en4_lds1_pgr2_sk"),
    "a16w16_kbuf1_large_tile": _gfx942_pipeline("a16w16_kbuf1_large_tile"),
    "a16w16_quad_mfma32_kbuf1": _gfx942_pipeline("a16w16_quad_mfma32_kbuf1"),
    "a16w16_wave_k_coop": _gfx942_pipeline("a16w16_wave_k_coop"),
    "a16w16_wave_k_coop_accum": _gfx942_pipeline("a16w16_wave_k_coop"),
    **{nosplit: _gfx942_pipeline(nosplit) for nosplit in _NOSPLIT},
    **{
        splitk: _gfx942_pipeline(nosplit) for nosplit, splitk in W3_KERNEL_PAIRS.items()
    },
}

TRAITS_HEADER_MAP = {tag: GFX942_TRAITS_HEADER for tag in _GFX942_A16W16_TAGS}

TRAITS_NAME_MAP = {tag: GFX942_TRAITS_NAME for tag in _GFX942_A16W16_TAGS}

KARGS_NAME_MAP = {
    "a16w16_em3en4_lds1_pgr2_sk": "opus_gemm_splitk_kargs",
    "a16w16_kbuf1_large_tile": "opus_gemm_noscale_kargs",
    "a16w16_quad_mfma32_kbuf1": "opus_gemm_noscale_kargs",
    "a16w16_wave_k_coop": "opus_gemm_noscale_kargs",
    "a16w16_wave_k_coop_accum": "opus_gemm_noscale_kargs",
    **{tag: "opus_gemm_splitk_kargs" for tag in _SPLITK},
    **{tag: "opus_gemm_noscale_kargs" for tag in _NOSPLIT},
}

KERNEL_FUNC_MAP = {
    "a16w16_em3en4_lds1_pgr2_sk": "gemm_a16w16_em3en4_lds1_pgr2_sk_kernel",
    "a16w16_kbuf1_large_tile": "gemm_a16w16_kbuf1_large_tile_kernel",
    "a16w16_quad_mfma32_kbuf1": "gemm_a16w16_quad_mfma32_kbuf1_kernel",
    "a16w16_wave_k_coop": "gemm_a16w16_wave_k_coop_kernel",
    "a16w16_wave_k_coop_accum": "gemm_a16w16_wave_k_coop_accum_kernel",
    # gfx942 paired tags: nosplit_tag's kernel symbol; splitk_tag reuses it.
    **{nosplit: f"gemm_{nosplit}_kernel" for nosplit in W3_KERNEL_PAIRS.keys()},
    **{splitk: f"gemm_{nosplit}_kernel" for nosplit, splitk in W3_KERNEL_PAIRS.items()},
}

SPLITK_REDUCE_SUPPORTED_SPLITKS = tuple(range(1, 17))

# Exact-N row-block reduce: (VEC, N_VEC, ROWS_PER_BLOCK), BLOCK = N_VEC * ROWS_PER_BLOCK.
EXACT_N_ROWBLOCK_REDUCE_CONFIGS = (
    (8, 8, 8),  # N=64,  8 rows/wg
    (8, 16, 4),  # N=128, 4 rows/wg
    (8, 32, 2),  # N=256, 2 rows/wg
    (8, 64, 1),  # N=512, 1 row/wg
    (8, 128, 4),  # N=1024, 4 rows/wg
    (8, 128, 2),  # N=1024, 2 rows/wg
    (8, 128, 1),  # N=1024, 1 row/wg
    (8, 256, 1),  # N=2048, 1 row/wg
)


def splitk_reduce_extra_forward_decls():
    return (
        "template<int VEC_, int BLOCK_, typename D_OUT,\n"
        "         bool HAS_BIAS_, typename D_BIAS_, bool HAS_OOB_>\n"
        "__global__ void splitk_reduce_kernel_bf16ws_fallback(\n"
        "    const opus_splitk_ws_handle* ws_handle, D_OUT* c_out,\n"
        "    int split_k, int M, int N, int batch,\n"
        "    int padded_M, int padded_N,\n"
        "    const D_BIAS_* bias, int stride_bias_batch);\n"
        "template<int SPLIT_K, int N_VEC, int ROWS_PER_BLOCK, int VEC_,\n"
        "         typename D_WS, typename D_OUT, bool HAS_BIAS_, typename D_BIAS_>\n"
        "__global__ void splitk_reduce_kernel_exact_n_rowblock(\n"
        "    const opus_splitk_ws_handle* ws_handle, D_OUT* c_out,\n"
        "    int M, int N, int batch,\n"
        "    int padded_M, int padded_N,\n"
        "    const D_BIAS_* bias, int stride_bias_batch);\n"
    )


def splitk_reduce_extra_device_instantiations():
    contents = "// Exact-N row-block reduce instantiations (BLOCK=N_VEC*ROWS)\n"
    for out_type in ("__bf16", "float"):
        contents += (
            f"template __global__ void splitk_reduce_kernel_bf16ws_fallback<16, 64, {out_type}, true,  {out_type}, true>(\n"
            f"    const opus_splitk_ws_handle*, {out_type}*, int, int, int, int, int, int,\n"
            f"    const {out_type}*, int);\n"
            f"template __global__ void splitk_reduce_kernel_bf16ws_fallback<16, 64, {out_type}, false, {out_type}, true>(\n"
            f"    const opus_splitk_ws_handle*, {out_type}*, int, int, int, int, int, int,\n"
            f"    const {out_type}*, int);\n"
        )
    for vec, nvec, rows in EXACT_N_ROWBLOCK_REDUCE_CONFIGS:
        for sk in SPLITK_REDUCE_SUPPORTED_SPLITKS:
            for ws_type in ("float", "__bf16"):
                contents += (
                    f"template __global__ void splitk_reduce_kernel_exact_n_rowblock<{sk}, {nvec}, {rows}, {vec}, {ws_type}, __bf16, true,  __bf16>(\n"
                    "    const opus_splitk_ws_handle*, __bf16*, int, int, int, int, int,\n"
                    "    const __bf16*, int);\n"
                    f"template __global__ void splitk_reduce_kernel_exact_n_rowblock<{sk}, {nvec}, {rows}, {vec}, {ws_type}, __bf16, false, __bf16>(\n"
                    "    const opus_splitk_ws_handle*, __bf16*, int, int, int, int, int,\n"
                    "    const __bf16*, int);\n"
                )
    return contents


SPLITK_REDUCE_EXTRA_MAP = {
    "forward_decls": splitk_reduce_extra_forward_decls,
    "device_instantiations": splitk_reduce_extra_device_instantiations,
}

register_arch_map("gfx942", "pipeline_header", PIPELINE_HEADER_MAP)
register_arch_map("gfx942", "traits_header", TRAITS_HEADER_MAP)
register_arch_map("gfx942", "traits_name", TRAITS_NAME_MAP)
register_arch_map("gfx942", "kargs_name", KARGS_NAME_MAP)
register_arch_map("gfx942", "kernel_func", KERNEL_FUNC_MAP)
register_arch_map("gfx942", "splitk_reduce_extra", SPLITK_REDUCE_EXTRA_MAP)


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
    kargs_template_vars,
    BIAS_HOST_VALIDATE,
    A16W16_TUNE_HOST_EXTRA,
    make_host_decl,
    make_device_decl,
    record_one_instantiation,
    **_unused,
):
    """gfx942 a16w16 splitk launcher emit."""
    is_quad_mfma32_splitk = k.kernel_tag == GFX942_QUAD_MFMA32_SPLITK_TAG
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    if is_quad_mfma32_splitk:
        kargs_explicit_param = f", {k.GROUP_M}, opus_gemm_splitk_kargs"
        fwd_decl_kargs_tpl = ", int COL_MAJOR_GROUP_M, typename Kargs"
        fwd_decl_kargs_fnarg = "Kargs"
    bf16ws = _uses_bf16_workspace(k)
    workspace_dtype, workspace_ptr_type = _splitk_workspace_types(k)
    # gfx942 a16w16_traits: 7 params <BLOCK_SIZE, BLOCK, DTYPE, VEC, TILE, WAVE, LDS_DEPTH=2>.
    trait_bm, trait_bn, lds_depth = _splitk_traits_geometry(k)
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{trait_bm}, {trait_bn}, {k.B_K}>,
    opus::tuple<{da}, {db}, {workspace_dtype}, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.T_M}, {k.T_N}, 1>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {lds_depth}>;
"""

    err_label = k.kernel_tag
    kernel_fwd_decl = (
        f"template<typename Traits{fwd_decl_kargs_tpl}>\n"
        f"__global__ void {kernel_func}({fwd_decl_kargs_fnarg} kargs);"
    )
    if is_quad_mfma32_splitk:
        kernel_launch_body = (
            f"\n    {kernel_func}<{k.name}_Traits<D_C>, {k.GROUP_M}, opus_gemm_splitk_kargs>"
            f"<<<grid_main, block_main, 0, stream>>>(kargs);"
        )
    else:
        kernel_launch_body = (
            f"\n    {kernel_func}<{k.name}_Traits<D_C>>"
            f"<<<grid_main, block_main, 0, stream>>>(kargs);"
        )
    reduce_rowblock_prelude = """
    // Exact-N row-block fast path: static split_k, guarded M tail.
    const bool reduce_rowblock_align = (padded_N == N);
"""
    rowblock_rows_by_n = {}
    for vec, nvec, rows in EXACT_N_ROWBLOCK_REDUCE_CONFIGS:
        rowblock_rows_by_n.setdefault(vec * nvec, []).append(rows)

    def reduce_rowblock_branch(hasbias):
        hb = "true" if hasbias else "false"
        bias_arg = (
            "reinterpret_cast<const __bf16*>(ptr_bias_), stride_bias_batch_"
            if hasbias
            else "nullptr, 0"
        )
        branches = []
        first = True
        for vec, nvec, rows in EXACT_N_ROWBLOCK_REDUCE_CONFIGS:
            n_exact = vec * nvec
            block_size = nvec * rows
            rows_for_n = rowblock_rows_by_n[n_exact]
            m_condition = "" if rows == rows_for_n[-1] else f" && (M % {rows} == 0)"
            for sk in SPLITK_REDUCE_SUPPORTED_SPLITKS:
                kw = "if" if first else "else if"
                first = False
                branches.append(
                    f"""            {kw} (reduce_rowblock_align && N == {n_exact}{m_condition} && split_k == {sk}) {{{{{{{{
            dim3 grid_rowblock(1, (M + {rows} - 1) / {rows}, batch);
            dim3 block_rowblock({block_size});
            splitk_reduce_kernel_exact_n_rowblock<{sk}, {nvec}, {rows}, {vec}, {workspace_ptr_type}, __bf16, {{hb}}, __bf16>
                <<<grid_rowblock, block_rowblock, 0, stream>>>(
                    ws_handle_device_,
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    M, N, batch, padded_M, padded_N,
                    {{bias_arg}});
        }}}}}}}}""".format(
                        hb=hb, bias_arg=bias_arg
                    )
                )
        return "\n".join(branches) + " else "

    def _baseline_call(dtype, hasbias, indent):
        hb = "true" if hasbias else "false"
        bias_args = (
            f"\n{indent}            reinterpret_cast<const {dtype}*>(ptr_bias_),\n"
            f"{indent}            stride_bias_batch_);"
            if hasbias
            else f"\n{indent}            nullptr, 0);"
        )
        reduce_kernel = (
            "splitk_reduce_kernel_bf16ws_fallback"
            if bf16ws
            else "splitk_reduce_kernel_fallback"
        )
        return (
            f"{indent}{reduce_kernel}<REDUCE_VEC, REDUCE_BS, {dtype}, {hb}, {dtype}, true>\n"
            f"{indent}    <<<grid_reduce, block_reduce, 0, stream>>>(\n"
            f"{indent}        ws_handle_device_,\n"
            f"{indent}        reinterpret_cast<{dtype}*>(Y.data_ptr()),\n"
            f"{indent}        split_k, M, N, batch, padded_M, padded_N,"
            f"{bias_args}"
        )

    bf16_t = _baseline_call("__bf16", True, "                ")
    bf16_f = _baseline_call("__bf16", False, "                ")
    fp32_t = _baseline_call("float", True, "            ")
    fp32_f = _baseline_call("float", False, "            ")
    bf16_y_check = ""
    bf16ws_fallback_decl = ""
    bf16ws_host_redirect = ""
    if bf16ws:
        fp32ws_name = k.name.replace("_bf16ws", "")
        exact_reduce_shape_conditions = " ||\n        ".join(
            f"(N == {n_exact})"
            for n_exact in sorted(
                {vec * nvec for vec, nvec, _ in EXACT_N_ROWBLOCK_REDUCE_CONFIGS}
            )
        )
        if is_quad_mfma32_splitk:
            bf16ws_host_redirect = f"""
    const bool bf16ws_exact_reduce_shape =
        {exact_reduce_shape_conditions};
    AITER_CHECK(bf16ws_exact_reduce_shape,
        "{err_label} bf16 workspace currently supports only exact-N rowblock "
        "reduce shapes");
"""
        else:
            bf16ws_fallback_decl = f"""
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void {fp32ws_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);
#endif
"""
            bf16ws_host_redirect = f"""
    const bool bf16ws_exact_reduce_shape =
        {exact_reduce_shape_conditions};
    if (!bf16ws_exact_reduce_shape) {{
        {fp32ws_name}<D_C>(XQ, WQ, Y, bias, splitK);
        return;
    }}
"""
        bf16_y_check = (
            "    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,\n"
            f'    "{err_label} bf16 workspace currently supports only bf16 Y");\n'
        )
    reduce_launch = f"""
    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                  batch * M, 1);
    dim3 block_reduce(REDUCE_BS);
{reduce_rowblock_prelude}
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
    if (bias.has_value()) {{{{
{reduce_rowblock_branch(True)}{{{{
{bf16_t}
        }}}}
    }}}} else {{{{
{reduce_rowblock_branch(False)}{{{{
{bf16_f}
        }}}}
    }}}}
    }}}} else {{{{
    // fp32 output: exact-N row-block reduce is bf16-only; use baseline.
    if (bias.has_value()) {{{{
{fp32_t}
    }}}} else {{{{
{fp32_f}
    }}}}
    }}}}
"""
    INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#include <type_traits>
#endif
#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
{kernel_fwd_decl}
#else
#include "{pipeline_header}"
#endif
{bf16ws_fallback_decl}
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK)
{{{{
    static_assert(std::is_same<D_C, fp32_t>::value,
    "{err_label} splitK launcher uses the fp32 tune-dispatch table");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

{bf16ws_host_redirect}
{bf16_y_check}
    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16
            || Y.dtype() == AITER_DTYPE_fp32,
    "{err_label} requires Y dtype bf16 or fp32");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1 && batch >= 1,
    "M, N, K, batch must be >= 1");
    AITER_CHECK(K % 2 == 0,
    "K=", K, " must be even (a16w16 family rejects odd K due to a "
    "latent K-tail accumulation bug; pass an even K)");
    // The gfx942 a16w16 splitk pipeline does not yet implement mask_va_tail
    // (the per-lane K-tail zeroing that gfx950's flatmm_splitk uses). When
    // K is not a multiple of B_K the last K-tile's buffer_load wraps past
    // the row into the next M-row's data, corrupting the accumulator
    // (observed max|err|~44 on bf16). Reject K%B_K!=0 until the
    // mask_va_tail port lands; callers must pad K to a multiple of B_K.
    AITER_CHECK(K % {k.B_K} == 0,
    "K=", K, " must be a multiple of B_K={k.B_K} for {err_label} "
    "(K-tail masking not yet implemented on gfx942 splitk)");
{BIAS_HOST_VALIDATE}
    using Traits = {k.name}_Traits<D_C>;

    // splitK semantics for gfx942 splitk launchers:
    //   splitK >  1 -> caller-pinned (tuner / explicit override). Used verbatim
    //                  (subject to the iters-per-split auto-clamp below).
    //   splitK <= 0 -> caller wants the launcher to auto-pick. Production
    //                  dispatcher (opus_gemm.cu) takes this path so the call
    //                  site stays gfx950-style (`fn(..., 0)`) without
    //                  arch-aware splitK plumbing leaking up.
    //   splitK == 1 -> caller explicitly requested no K-split. Honored.
    int split_k;
    if (splitK > 0) {{{{
    split_k = splitK;
    }}}} else {{{{
    // Auto-pick: target ~1 WG per CU. cu_num cached thread_local so we
    // do not pay hipGetDeviceProperties on every launch.
    static thread_local int cu_cached = -1;
    if (cu_cached < 0) {{{{
        int dev = 0;
        hipDeviceProp_t prop{{{{}}}};
        if (hipGetDevice(&dev) == hipSuccess &&
            hipGetDeviceProperties(&prop, dev) == hipSuccess) {{{{
            cu_cached = prop.multiProcessorCount;
        }}}}
        if (cu_cached <= 0) cu_cached = 64;  // safe gfx942 lower bound
    }}}}
    int tiles_mn = ((M + {k.B_M} - 1) / {k.B_M})
                 * ((N + {k.B_N} - 1) / {k.B_N}) * batch;
    if (tiles_mn <= 0) tiles_mn = 1;
    // P1 variant wants 2 wg/CU co-residency for TLP -> aim for 2x cu_num grid.
    int target_wg_dbuf2 = {"2 * cu_cached" if k.kernel_tag.endswith("_p1") else "cu_cached"};
    split_k = (target_wg_dbuf2 + tiles_mn - 1) / tiles_mn;
    if (split_k < 1)  split_k = 1;
    if (split_k > 16) split_k = 16;  // matches tuner enumeration ceiling
    }}}}

    // Host-side auto-clamp: split-barrier pipeline requires at least 2
    // K-tile iterations per split (one in LDS + one prefetched). Applies to
    // both caller-pinned and auto-picked split_k. P1 (depth=2 K-dbuf) additionally
    // requires loops even per split.
    int total_iters = (K + {k.B_K} - 1) / {k.B_K};
    constexpr int min_iters_per_split = 2;
    constexpr bool require_even_loops_dbuf2 = {"true" if k.kernel_tag in GFX942_EVEN_LOOP_SPLITK_TAGS else "false"};
    while (split_k > 1) {{{{
    int iters_full = (total_iters + split_k - 1) / split_k;
    int last_loops = total_iters - (split_k - 1) * iters_full;
    bool parity_ok = !require_even_loops_dbuf2
                   || (iters_full % 2 == 0 && last_loops % 2 == 0);
    if (iters_full >= min_iters_per_split && last_loops >= min_iters_per_split && parity_ok) break;
    split_k--;
    }}}}
    AITER_CHECK(total_iters >= min_iters_per_split,
    "K=", K, " too small for {err_label} B_K={k.B_K}: need K >= ",
    {k.B_K} * min_iters_per_split);
    if (require_even_loops_dbuf2) {{{{
    int iters_full = (total_iters + split_k - 1) / split_k;
    int last_loops = total_iters - (split_k - 1) * iters_full;
    AITER_CHECK(iters_full % 2 == 0 && last_loops % 2 == 0,
        "{err_label} needs even loops per split; K=", K,
        " split_k=", split_k, " gives loops=(", iters_full, ",", last_loops, ")");
    }}}}

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    int padded_M    = num_tiles_m * {k.B_M};
    int padded_N    = num_tiles_n * {k.B_N};

    auto stream = aiter::getCurrentHIPStream();
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
    const bool capturing = (capture_status != hipStreamCaptureStatusNone);
    extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);
    extern const opus_splitk_ws_handle* opus_splitk_ws_device_handle(hipStream_t, bool);
    extern void opus_splitk_ws_sync_to_device(hipStream_t);
    auto* ws_handle_ = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

    size_t ws_bytes = (size_t)split_k * (size_t)batch
                * (size_t)padded_M * (size_t)padded_N * sizeof(typename Traits::D_C);
    if (ws_handle_->ptr == nullptr || ws_bytes > ws_handle_->bytes)
    {{
    AITER_CHECK(!capturing,
        "{err_label} workspace grow inside HIP graph capture is not "
        "supported. Call aiter.opus_gemm_workspace_init() on the capture "
        "stream and warm with the largest expected GEMM before capturing.");

    if (ws_handle_->ptr != nullptr)
    {{
        HIP_CALL(hipDeviceSynchronize());
        HIP_CALL(hipFree(ws_handle_->ptr));
    }}
    const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
    size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
    void* new_ptr = nullptr;
    HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
    ws_handle_->ptr = new_ptr;
    ws_handle_->bytes = grow_bytes;
    opus_splitk_ws_sync_to_device(stream);
    }}
    const auto* ws_handle_device_ =
        opus_splitk_ws_device_handle(stream, /*allow_create=*/!capturing);

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a         = XQ.data_ptr();
    kargs.ptr_b         = WQ.data_ptr();
    kargs.ws_handle     = ws_handle_device_;
    kargs.ptr_c         = Y.data_ptr();
    kargs.ptr_bias      = ptr_bias_;
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
    kargs.split_k = split_k;
    kargs.stride_a        = XQ.stride(1);
    kargs.stride_b        = WQ.stride(1);
    kargs.stride_ws       = padded_N;
    kargs.stride_c        = N;
    kargs.stride_a_batch  = XQ.stride(0);
    kargs.stride_b_batch  = WQ.stride(0);
    kargs.stride_ws_batch = padded_M * padded_N;
    kargs.stride_c_batch  = M * N;
    kargs.stride_bias_batch = stride_bias_batch_;
    dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
    dim3 block_main({k.BLOCK_SIZE});

{kernel_launch_body}{reduce_launch}
}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    record_one_instantiation(
        cg,
        k,
        kernel_func,
        kargs_name,
        A16W16_TUNE_HOST_EXTRA,
        kargs_explicit_param,
    )


def _emit_a16w16_nosplit_launcher(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    instance_impl_preamble,
    instance_impl_host_tu_split,
    A16W16_TUNE_TAGS,
    fwd_decl_kargs_tpl,
    fwd_decl_kargs_fnarg,
    traits_extra,
    k_check,
    grid_decl,
    launch_block,
    device_decl_for_dtype,
):
    extra_param = (
        ",\n    std::optional<aiter_tensor_t> bias," "\n    int /*splitK*/"
        if k.kernel_tag in A16W16_TUNE_TAGS
        else ""
    )

    bias_kargs_block = (
        "    AITER_CHECK(!bias.has_value(),\n"
        '        "bias not supported on this a16w16 kid");\n'
        if k.kernel_tag in A16W16_TUNE_TAGS
        else ""
    )

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra}>;
"""

    preamble = instance_impl_preamble()
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )
    INSTANCE_IMPL = f"""{preamble}
{host_tu_split}
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y{extra_param})
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    {kargs_name} kargs{{}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = XQ.stride(1);
    kargs.stride_b = WQ.stride(1);
    kargs.stride_c = N;
    kargs.stride_a_batch = XQ.stride(0);
    kargs.stride_b_batch = WQ.stride(0);
    kargs.stride_c_batch = M * N;
{bias_kargs_block}
    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
{grid_decl}
    dim3 block({k.BLOCK_SIZE});
{launch_block}

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    inst_extra_param = (
        ",\n    std::optional<aiter_tensor_t>,\n    int"
        if k.kernel_tag in A16W16_TUNE_TAGS
        else ""
    )
    for CDtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{CDtype}>(\n"
            f"    aiter_tensor_t &XQ,\n"
            f"    aiter_tensor_t &WQ,\n"
            f"    aiter_tensor_t &Y{inst_extra_param});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
        )
        cg._device_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "device_decl": device_decl_for_dtype(CDtype),
            }
        )


def gen_a16w16_quad_mfma32_gfx942_instance(
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
    instance_impl_preamble,
    instance_impl_host_tu_split,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    A16W16_TUNE_TAGS,
    **_unused,
):
    """gfx942 quad MFMA32 launcher emit."""
    fwd_decl_kargs_fnarg = "Kargs"
    fwd_decl_kargs_tpl = ", int COL_MAJOR_GROUP_M, typename Kargs"
    traits_extra = (
        f",\n        opus::seq<{k.T_M}, {k.T_N}, 1>,"
        f"\n        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>"
    )
    min_k = 2 * k.B_K
    k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""

    launch_block = f"""
    auto stream = aiter::getCurrentHIPStream();
    if (num_tiles_m <= 8 && num_tiles_n > 8) {{
        {kernel_func}<{k.name}_Traits<D_C>, 2><<<grid, block, 0, stream>>>(kargs);
    }} else if (num_tiles_n <= 8) {{
        {kernel_func}<{k.name}_Traits<D_C>, 8><<<grid, block, 0, stream>>>(kargs);
    }} else {{
        {kernel_func}<{k.name}_Traits<D_C>, 0><<<grid, block, 0, stream>>>(kargs);
    }}"""
    grid_decl = "    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);"

    def device_decl_for_dtype(CDtype):
        return (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>, 0, opus_gemm_noscale_kargs>({kargs_name});\n"
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>, 2, opus_gemm_noscale_kargs>({kargs_name});\n"
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>, 8, opus_gemm_noscale_kargs>({kargs_name});\n"
        )

    _emit_a16w16_nosplit_launcher(
        cg,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
        instance_impl_preamble,
        instance_impl_host_tu_split,
        A16W16_TUNE_TAGS,
        fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg,
        traits_extra,
        k_check,
        grid_decl,
        launch_block,
        device_decl_for_dtype,
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
    instance_impl_preamble,
    instance_impl_host_tu_split,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    A16W16_TUNE_TAGS,
    **_unused,
):
    """gfx942 a16w16 non-splitK launcher emit (kbuf2v / kbuf2v_bk128 /
    kbuf1 / wave_k_coop)."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    is_wkc_accum = k.kernel_tag == "a16w16_wave_k_coop_accum"
    is_wkc = k.kernel_tag in ("a16w16_wave_k_coop", "a16w16_wave_k_coop_accum")
    waves_per_wg = k.BLOCK_SIZE // 64
    t_k = waves_per_wg if is_wkc else 1
    lds_depth_suffix = ", 1" if is_wkc else ""
    traits_extra = (
        f",\n        opus::seq<{k.T_M}, {k.T_N}, {t_k}>,"
        f"\n        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>"
        f"{lds_depth_suffix}"
    )
    if is_wkc:
        wg_k_tile = k.B_K * t_k
        k_check = f"""
    AITER_CHECK(K % {wg_k_tile} == 0,
        "K=", K, " must be divisible by B_K*T_K={wg_k_tile} for wave-K-coop");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""
    else:
        min_k = 2 * k.B_K
        k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""

    launch_block = f"""
    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);"""
    if is_wkc_accum:
        grid_decl = "    dim3 grid(num_tiles_n * 8, num_tiles_m, batch);"
    elif is_wkc:
        grid_decl = "    dim3 grid(num_tiles_n, num_tiles_m, batch);"
    else:
        grid_decl = "    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);"

    def device_decl_for_dtype(CDtype):
        return (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>{kargs_explicit_param}>({kargs_name});\n"
        )

    _emit_a16w16_nosplit_launcher(
        cg,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
        instance_impl_preamble,
        instance_impl_host_tu_split,
        A16W16_TUNE_TAGS,
        fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg,
        traits_extra,
        k_check,
        grid_decl,
        launch_block,
        device_decl_for_dtype,
    )


# ---------- Self-register at import time ----------
# gfx942 splitk family.
for _tag in GFX942_SPLITK_TAGS:
    register_emit("gfx942", _tag, gen_splitk_gfx942_instance)

register_emit(
    "gfx942",
    "a16w16_quad_mfma32_kbuf1",
    gen_a16w16_quad_mfma32_gfx942_instance,
)
# gfx942 a16w16 non-splitK family.
_GFX942_NOSPLIT_TAGS = (
    "a16w16_kbuf1_large_tile",
    "a16w16_kbuf2v",
    "a16w16_kbuf2v_bk128",
    "a16w16_kbuf1",
    "a16w16_wave_k_coop",
    "a16w16_wave_k_coop_accum",
)
for _tag in _GFX942_NOSPLIT_TAGS:
    register_emit("gfx942", _tag, gen_a16w16_nosplit_gfx942_instance)


# ---------------- gfx942 a16w16 validator ----------------
# Coverage: basic physical limits only. Detailed LDS depth / layout checks
# live in gfx942/opus_gemm_traits_a16w16.cuh static_asserts (hipcc enforces).

# gfx942 (CDNA3 / MI300X) hardware LDS budget per WG.
_GFX942_LDS_PER_WG_BYTES = 64 * 1024


def _validate_a16w16_em3en4_gfx942(k: OpusGemmInstance):
    """Validate gfx942 EM3EN4: host 128x96, device 96x128 LDSB1."""
    errors = []

    if getattr(k, "arch_prefix", "") != "gfx942":
        errors.append("EM3EN4 LDS1/PGR2 path is gfx942-only")
    if k.kernel_tag != "a16w16_em3en4_lds1_pgr2_sk":
        errors.append(f"kernel_tag={k.kernel_tag} must be a16w16_em3en4_lds1_pgr2_sk")
    # Host tile is 128M x 96N; device traits are remapped to 96 x 128.
    if k.BLOCK_SIZE != 256 or (k.B_M, k.B_N) != (128, 96) or k.B_K != 128:
        errors.append(
            f"BLOCK=({k.BLOCK_SIZE},{k.B_M},{k.B_N},{k.B_K}) must be (256,128,96,128)"
        )
    if (k.T_M, k.T_N) != (2, 2):
        errors.append(f"T=({k.T_M},{k.T_N}) must be (2,2)")
    if (k.W_M, k.W_N, k.W_K) != (16, 16, 16):
        errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) must be (16,16,16)")

    sizeof_da = 2
    expected_vec = 16 // sizeof_da
    if k.VEC_A != expected_vec or k.VEC_B != expected_vec:
        errors.append(f"VEC_A/B must be {expected_vec}")
    if k.VEC_C != 4:
        errors.append("VEC_C must be 4 for fp32 workspace stores")

    # Device traits geometry (physical B_M=96, B_N=128).
    trait_bm, trait_bn = 96, 128
    smem_linear_wave = WARP_SIZE * expected_vec
    smem_sub = smem_linear_wave // k.B_K if k.B_K else 0
    if not smem_sub or trait_bm % smem_sub or trait_bn % smem_sub:
        errors.append("B_M/B_N must be divisible by smem_sub")

    E_M = trait_bm // (k.W_M * k.T_M)
    E_N = trait_bn // (k.W_N * k.T_N)
    E_K = k.B_K // k.W_K
    if (E_M, E_N) != (3, 4) or E_K not in (4, 8):
        errors.append(f"E=({E_M},{E_N},{E_K}) must be (3,4,{{4,8}})")

    agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
    total_agprs = 4 * E_M * E_N * agpr_per_mfma
    smem_padding = 16
    total_lds = (trait_bm + trait_bn) * (k.B_K + smem_padding) * sizeof_da
    if total_lds > _GFX942_LDS_PER_WG_BYTES:
        errors.append(
            f"LDS={total_lds // 1024}KiB exceeds {_GFX942_LDS_PER_WG_BYTES // 1024}KiB"
        )

    if errors:
        raise ValueError(
            f"Invalid a16w16_em3en4 instance '{k.name}':\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return {
        "E_M": E_M,
        "E_N": E_N,
        "E_K": E_K,
        "agprs": total_agprs,
        "vgpr_est": 4 * E_K * (E_M + 2 * E_N) + 80,
        "lds_bytes": total_lds,
        "min_k": 2 * k.B_K,
    }


def _validate_a16w16_wave_k_coop_gfx942(k: OpusGemmInstance):
    """Validate gfx942 wave-K-coop: all waves split K, then reduce in LDS."""
    errors = []
    if getattr(k, "arch_prefix", "") != "gfx942":
        errors.append("wave-K-coop is gfx942-only")
    if k.kernel_tag not in ("a16w16_wave_k_coop", "a16w16_wave_k_coop_accum"):
        errors.append(f"kernel_tag={k.kernel_tag} must be a16w16_wave_k_coop/_accum")
    if k.BLOCK_SIZE not in (256, 512):
        errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} must be 256 or 512")
    waves_per_wg = k.BLOCK_SIZE // WARP_SIZE
    if (k.T_M, k.T_N) != (1, 1):
        errors.append(
            f"T_M/T_N=({k.T_M},{k.T_N}) must be (1,1); WKC uses all waves for K"
        )
    if (k.W_M, k.W_N, k.W_K) != (16, 16, 16):
        errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) must be (16,16,16)")
    if k.B_M % k.T_M or k.B_N % k.T_N:
        errors.append("B_M/B_N must be divisible by T_M/T_N")
    tile_m = k.B_M // k.T_M if k.T_M else 0
    tile_n = k.B_N // k.T_N if k.T_N else 0
    if tile_m % k.W_M or tile_n % k.W_N:
        errors.append("wave-local M/N tiles must be divisible by W_M/W_N=16")
    if k.B_K % k.W_K:
        errors.append("B_K must be divisible by W_K=16")

    sizeof_da = 2
    expected_vec = 16 // sizeof_da
    if k.VEC_A != expected_vec or k.VEC_B != expected_vec:
        errors.append(f"VEC_A/B must be {expected_vec}")

    E_M = tile_m // k.W_M
    E_N = tile_n // k.W_N
    E_K = k.B_K // k.W_K
    agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
    total_agprs = 4 * E_M * E_N * agpr_per_mfma

    t_k = waves_per_wg
    a_pad = 4
    a_bytes = tile_m * (k.B_K + a_pad) * sizeof_da * k.T_M * t_k
    b_bytes = tile_n * (k.B_K + a_pad) * sizeof_da * k.T_N * t_k
    partial_bytes = k.B_M * k.B_N * 4 * t_k
    ab_bytes = a_bytes + b_bytes
    alias_partial = ab_bytes + partial_bytes > _GFX942_LDS_PER_WG_BYTES
    alias_bytes = max(ab_bytes, partial_bytes)
    lds_bytes = alias_bytes if alias_partial else ab_bytes + partial_bytes
    if alias_partial and ab_bytes > _GFX942_LDS_PER_WG_BYTES:
        errors.append(f"A/B LDS={ab_bytes // 1024}KiB exceeds 64KiB")
    if lds_bytes > _GFX942_LDS_PER_WG_BYTES:
        errors.append(f"LDS={lds_bytes // 1024}KiB exceeds 64KiB")

    if errors:
        raise ValueError(
            f"Invalid a16w16_wave_k_coop instance '{k.name}':\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    return {
        "E_M": E_M,
        "E_N": E_N,
        "E_K": E_K,
        "agprs": total_agprs,
        "vgpr_est": 4 * E_K * (E_M + 2 * E_N) + 80,
        "lds_bytes": lds_bytes,
        "min_k": t_k * k.B_K,
    }


def _validate_a16w16_quad_mfma32_gfx942(k: OpusGemmInstance):
    """Validate the gfx942 quad MFMA32 path.

    This is deliberately separate from _validate_a16w16_gfx942 so the legacy
    16x16x16 family remains locked to its original MFMA shape.
    """
    errors = []
    if getattr(k, "arch_prefix", "") != "gfx942":
        errors.append("quad_mfma32 path is gfx942-only")
    valid_tags = ("a16w16_quad_mfma32_kbuf1", GFX942_QUAD_MFMA32_SPLITK_TAG)
    if k.kernel_tag not in valid_tags:
        errors.append(f"kernel_tag={k.kernel_tag} must be one of {valid_tags}")
    valid_blocks = {
        (256, 256, 256, 32): (2, 2, 2, 2, 4),
    }
    block_key = (k.BLOCK_SIZE, k.B_M, k.B_N, k.B_K)
    if block_key not in valid_blocks:
        errors.append(
            f"BLOCK=({k.BLOCK_SIZE},{k.B_M},{k.B_N},{k.B_K}) must be one of "
            f"{tuple(valid_blocks)}"
        )
    if (k.W_M, k.W_N, k.W_K) != (32, 32, 8):
        errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) must be (32,32,8)")

    sizeof_da = 2
    expected_vec = 16 // sizeof_da
    if k.VEC_A != expected_vec or k.VEC_B != expected_vec:
        errors.append(f"VEC_A/B must be {expected_vec}")
    if k.VEC_C != 4:
        errors.append("VEC_C must be 4")

    lds_depth = 2
    half_bm = k.B_M // lds_depth
    half_bn = k.B_N // lds_depth
    E_M = half_bm // (k.W_M * k.T_M) if (k.W_M * k.T_M) else 0
    E_N = half_bn // (k.W_N * k.T_N) if (k.W_N * k.T_N) else 0
    E_K = k.B_K // k.W_K if k.W_K else 0
    expected = valid_blocks.get(block_key)
    if expected is not None and (k.T_M, k.T_N, E_M, E_N, E_K) != expected:
        errors.append(
            f"E=({E_M},{E_N},{E_K}) inconsistent with the quad_mfma32 tile whitelist"
        )

    agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
    total_agprs = 4 * E_M * E_N * agpr_per_mfma
    vgpr_est = 4 * E_K * (E_M + 2 * E_N) + 80
    if total_agprs > 256:
        errors.append(f"AGPR={total_agprs} must be <= 256")
    if vgpr_est > 256:
        errors.append(f"VGPR_est={vgpr_est} exceeds 256")
    if vgpr_est + total_agprs > 512:
        errors.append(f"VGPR+AGPR={vgpr_est + total_agprs} exceeds 512")

    ab_stage_bytes = 2 * (k.B_M + k.B_N) * k.B_K
    c_stage_bytes = half_bm * half_bn * sizeof_da
    lds_bytes = max(ab_stage_bytes, c_stage_bytes)
    if lds_bytes > _GFX942_LDS_PER_WG_BYTES:
        errors.append(f"LDS={lds_bytes // 1024}KiB exceeds 64KiB")

    if errors:
        raise ValueError(
            f"Invalid gfx942 quad_mfma32 instance '{k.name}':\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return {
        "E_M": E_M,
        "E_N": E_N,
        "E_K": E_K,
        "agprs": total_agprs,
        "vgpr_est": vgpr_est,
        "lds_bytes": lds_bytes,
        "min_k": 2 * k.B_K,
    }


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
