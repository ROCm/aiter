# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 codegen -- emit launchers for gfx942-targeted kid families."""

import os
from pathlib import Path

from codegen.common import register_emit


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
    BIAS_HOST_VALIDATE,
    SPLITK_REDUCE_FAST_ARCHES,
    V3_NVEC_ROWS,
    V2_SUPPORTED_SPLITKS,
    A16W16_TUNE_HOST_EXTRA,
    make_host_decl,
    make_device_decl,
    record_one_instantiation,
    **_unused,
):
    """gfx942 a16w16 splitk launcher emit (fused / non-fused)."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )

    # gfx942 a16w16_traits: 6 params <BLOCK_SIZE, BLOCK, DTYPE, VEC, TILE, WAVE>.
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, fp32_t, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.T_M}, {k.T_N}, 1>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>>;
"""

    # Per-flavor pieces (workspace alloc + reduce dispatch).
    if fused:
        err_label = "a16w16_fused_reduce"
        ws_alloc_extra = """
    int num_flags = batch * num_tiles_m * num_tiles_n;
    size_t total_bytes = ws_bytes + (size_t)num_flags * sizeof(unsigned int);"""
        ws_size_var = "total_bytes"
        flags_block = """
    unsigned int* ptr_flags_ = reinterpret_cast<unsigned int*>(
    static_cast<char*>(ws_cached_ptr) + ws_bytes);
    HIP_CALL(hipMemsetAsync(ptr_flags_, 0, num_flags * sizeof(unsigned int), stream));"""
        kargs_flags_assign = "    kargs.ptr_flags     = ptr_flags_;\n"
        # cooperative_reduce: 1 if all split_k WGs co-resident (split reduce work), else 0.
        cooperative_assign = """
    static thread_local int cu_for_coop = -1;
    if (cu_for_coop < 0) {{
    int dev_c = 0;
    hipDeviceProp_t prop_c{{}};
    if (hipGetDevice(&dev_c) == hipSuccess &&
        hipGetDeviceProperties(&prop_c, dev_c) == hipSuccess) {{
        cu_for_coop = prop_c.multiProcessorCount;
    }}
    if (cu_for_coop <= 0) cu_for_coop = 64;
    }}
    int total_wgs_coop = num_tiles_m * num_tiles_n * batch * split_k;
    kargs.cooperative_reduce = (total_wgs_coop <= cu_for_coop) ? 1 : 0;
"""
        # fused: D_OUT template param so in-kernel reduce casts to Y.dtype() (avoid bf16/fp32 mismatch).
        kernel_fwd_decl = (
            f"template<typename Traits, typename D_OUT>\n"
            f"__global__ void {kernel_func}({kargs_name} kargs);"
        )
        kernel_launch_body = f"""
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
    {kernel_func}<{k.name}_Traits<D_C>, __bf16><<<grid_main, block_main, 0, stream>>>(kargs);
    }}}} else {{{{
    {kernel_func}<{k.name}_Traits<D_C>, float><<<grid_main, block_main, 0, stream>>>(kargs);
    }}}}"""
        reduce_launch = ""  # in-kernel reduce; no separate launch
    else:
        err_label = k.kernel_tag
        ws_alloc_extra = ""
        ws_size_var = "ws_bytes"
        flags_block = ""
        kargs_flags_assign = ""
        cooperative_assign = ""
        # non-fused splitk: D_C only; Y-dtype dispatch inside separate reduce kernel.
        kernel_fwd_decl = (
            f"template<typename Traits{fwd_decl_kargs_tpl}>\n"
            f"__global__ void {kernel_func}({fwd_decl_kargs_fnarg} kargs);"
        )
        # Kargs deduced from kargs fn arg; <Traits> only keeps host/device
        # mangling identical (avoid SA_ vs T0_ substitution mismatch).
        kernel_launch_body = (
            f"\n    {kernel_func}<{k.name}_Traits<D_C>>"
            f"<<<grid_main, block_main, 0, stream>>>(kargs);"
        )
        # V2 essential for N=64+M%row!=0 (V3 misses); baseline 50-80% slower.
        v2_enabled = k.arch_prefix in SPLITK_REDUCE_FAST_ARCHES
        v2_prelude = (
            """
    // V2/V3 fast path: split_k static-unroll, no OOB.
    constexpr int V2_VEC = 8;
    constexpr int V2_BS  = 8;
    const bool v2_align = (N % (V2_VEC * V2_BS) == 0) && (padded_N == N);
    dim3 grid_reduce_v2(v2_align ? (N / (V2_VEC * V2_BS)) : 1, batch * M, 1);
    dim3 block_reduce_v2(V2_BS);

    // V3: multi-row per wg (BLOCK = N_VEC * ROWS_PER_BLOCK = 64, 1 full wave).
    // Dispatch picks the (N_VEC, ROWS) tuple at runtime; supported set is in
    // V3_NVEC_ROWS (gen_instances.py).
    const int v3_n_vec = N / V2_VEC;
"""
            if v2_enabled
            else ""
        )

        def v2_branch(hasbias):
            if not v2_enabled:
                return ""
            hb = "true" if hasbias else "false"
            bias_arg = (
                "reinterpret_cast<const __bf16*>(ptr_bias_), stride_bias_batch_"
                if hasbias
                else "nullptr, 0"
            )
            # V3 branches first; V2 fallback when no V3 (N_VEC, ROWS) tuple matches.
            branches = []
            first = True
            for nvec, rows in V3_NVEC_ROWS:
                block_size = nvec * rows
                for sk in V2_SUPPORTED_SPLITKS:
                    kw = "if" if first else "else if"
                    first = False
                    branches.append(
                        f"""            {kw} (v2_align && v3_n_vec == {nvec} && (M % {rows} == 0) && split_k == {sk}) {{{{{{{{
            dim3 grid_v3(1, M / {rows}, batch);
            dim3 block_v3({block_size});
            splitk_reduce_kernel_v3<{sk}, {nvec}, {rows}, V2_VEC, __bf16, {{hb}}, __bf16>
                <<<grid_v3, block_v3, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    M, N, batch, padded_M, padded_N,
                    {{bias_arg}});
        }}}}}}}}""".format(
                            hb=hb, bias_arg=bias_arg
                        )
                    )
            for sk in V2_SUPPORTED_SPLITKS:
                branches.append(
                    f"""            else if (v2_align && split_k == {sk}) {{{{{{{{
            splitk_reduce_kernel_v2<{sk}, V2_VEC, V2_BS, __bf16, {{hb}}, __bf16>
                <<<grid_reduce_v2, block_reduce_v2, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    M, N, batch, padded_M, padded_N,
                    {{bias_arg}});
        }}}}}}}}""".format(hb=hb, bias_arg=bias_arg)
                )
            return "\n".join(branches) + " else "  # falls through to baseline

        # Baseline reduce call (V2/V3 fall through here; fp32 always lands here).
        def _baseline_call(dtype, hasbias, indent):
            hb = "true" if hasbias else "false"
            bias_args = (
                f"\n{indent}            reinterpret_cast<const {dtype}*>(ptr_bias_),\n"
                f"{indent}            stride_bias_batch_);"
                if hasbias
                else f"\n{indent}            nullptr, 0);"
            )
            return (
                f"{indent}splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, {dtype}, {hb}, {dtype}, true>\n"
                f"{indent}    <<<grid_reduce, block_reduce, 0, stream>>>(\n"
                f"{indent}        reinterpret_cast<const float*>(ptr_workspace_),\n"
                f"{indent}        reinterpret_cast<{dtype}*>(Y.data_ptr()),\n"
                f"{indent}        split_k, M, N, batch, padded_M, padded_N,"
                f"{bias_args}"
            )

        bf16_t = _baseline_call("__bf16", True, "                ")
        bf16_f = _baseline_call("__bf16", False, "                ")
        fp32_t = _baseline_call("float", True, "            ")
        fp32_f = _baseline_call("float", False, "            ")
        reduce_launch = f"""
    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                  batch * M, 1);
    dim3 block_reduce(REDUCE_BS);
{v2_prelude}
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
    if (bias.has_value()) {{{{
{v2_branch(True)}{{{{
{bf16_t}
        }}}}
    }}}} else {{{{
{v2_branch(False)}{{{{
{bf16_f}
        }}}}
    }}}}
    }}}} else {{{{
    // fp32 output: V2 not implemented yet; use baseline.
    if (bias.has_value()) {{{{
{fp32_t}
    }}}} else {{{{
{fp32_f}
    }}}}
    }}}}"""

    INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
{kernel_fwd_decl}
#else
#include "{pipeline_header}"
#endif
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
    "{err_label} main kernel uses fp32 workspace; D_C template param must be fp32_t");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

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
    constexpr bool require_even_loops_dbuf2 = {"true" if k.kernel_tag in ("a16w16_kbuf2v_sk", "a16w16_kbuf2v_bk128_sk") else "false"};
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
    size_t ws_bytes = (size_t)split_k * (size_t)batch
                * (size_t)padded_M * (size_t)padded_N * sizeof(float);{ws_alloc_extra}
    static thread_local void*  ws_cached_ptr   = nullptr;
    static thread_local size_t ws_cached_bytes = 0;
    if (ws_cached_ptr == nullptr || {ws_size_var} > ws_cached_bytes)
    {{
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
    AITER_CHECK(capture_status == hipStreamCaptureStatusNone,
        "{err_label} workspace cache miss inside HIP graph capture is not "
        "supported. Run the launcher once eagerly with the same shape "
        "before capturing the graph.");

    if (ws_cached_ptr != nullptr)
    {{
        HIP_CALL(hipDeviceSynchronize());
        HIP_CALL(hipFree(ws_cached_ptr));
    }}
    const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
    size_t grow_bytes = (({ws_size_var} + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
    HIP_CALL(hipMalloc(&ws_cached_ptr, grow_bytes));
    ws_cached_bytes = grow_bytes;
    }}
    void* ptr_workspace_ = ws_cached_ptr;{flags_block}

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a         = XQ.data_ptr();
    kargs.ptr_b         = WQ.data_ptr();
    kargs.ptr_workspace = ptr_workspace_;
    kargs.ptr_c         = Y.data_ptr();
    kargs.ptr_bias      = ptr_bias_;
{kargs_flags_assign}    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
    kargs.split_k = split_k;
    kargs.stride_a        = K;
    kargs.stride_b        = K;
    kargs.stride_ws       = padded_N;
    kargs.stride_c        = N;
    kargs.stride_a_batch  = M * K;
    kargs.stride_b_batch  = N * K;
    kargs.stride_ws_batch = padded_M * padded_N;
    kargs.stride_c_batch  = M * N;
    kargs.stride_bias_batch = stride_bias_batch_;
{cooperative_assign}
    dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
    dim3 block_main({k.BLOCK_SIZE});

{kernel_launch_body}{reduce_launch}
}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    if fused:
        # fused: instantiate both Y dtypes (bf16, float) so runtime dispatch links.
        for CDtype in k.output_dtypes:
            cg._host_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "host_decl": make_host_decl(k.name, CDtype, A16W16_TUNE_HOST_EXTRA),
                }
            )
            cg._device_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "device_decl": (
                        make_device_decl(
                            k.name, CDtype, kernel_func, kargs_name, ", __bf16"
                        )
                        + make_device_decl(
                            k.name, CDtype, kernel_func, kargs_name, ", float"
                        )
                    ),
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
    instance_impl_preamble,
    instance_impl_host_tu_split,
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

    launch_block = f"""
    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);"""

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
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;
{bias_kargs_block}
    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
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
        device_decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>{kargs_explicit_param}>({kargs_name});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
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
