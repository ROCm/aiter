# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 codegen -- emit launchers for gfx950-targeted kid families.

Free functions taking the parent opus_gemm_codegen instance as first arg.
Self-registers each emit into codegen.common.EMIT_REGISTRY at import time.
"""

import os
from pathlib import Path

from codegen.common import register_emit


def gen_persistent_instance(
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
    **_unused,
):
    """gfx950 a16w16_persistent launcher emit. See gen_instances.opus_gemm_codegen._gen_persistent_instance."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_oob_str = "true" if k.has_oob else "false"

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.T_M}, {k.T_N}, 1>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {has_oob_str},
    {k.cachectl_a},
    {k.cachectl_b}>;
"""

    min_k = 2 * k.B_K
    k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
    AITER_CHECK(batch >= 1, "batch must be >= 1");
"""

    grid_setup = f"""
    constexpr int NUM_CU = 256;
    constexpr int NUM_XCD = 8;
    const int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    const int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    int split_m = std::max(1, (NUM_CU + num_tiles_n - 1) / num_tiles_n);
    while (split_m < num_tiles_m && (num_tiles_m % split_m) != 0) split_m++;
    if (split_m > num_tiles_m) split_m = num_tiles_m;
    const int m_per_wg = num_tiles_m / split_m;
    AITER_CHECK(num_tiles_m % split_m == 0,
        "persistent: num_tiles_m=", num_tiles_m,
        " must be divisible by split_m=", split_m);

    // Pad grid.y so the XCD-local swizzle math stays bijective. See the
    // long comment in opus_gemm_pipeline_a16w16_persistent_gfx950.cuh
    // for why this is needed and why it is free on the large-M shapes
    // the swizzle is tuned for (split_m is already a multiple of
    // NUM_XCD there, so the pad is a no-op). When split_m < NUM_XCD
    // (small-M shapes like M=8192 N=8192 K=256), the pad multiplies
    // grid.y by NUM_XCD/split_m and the kernel's wave-uniform
    // early-return guard drops the over-shoot WGs.
    const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
    const int grid_y_padded = m_grp_per_xcd * NUM_XCD;

    kargs.m_per_wg = m_per_wg;
    kargs.num_tiles_n = num_tiles_n;
    kargs.split_m = split_m;          // un-padded; kernel uses for early-return
    kargs.m_grp_per_xcd = m_grp_per_xcd;

    dim3 grid(num_tiles_n, grid_y_padded, batch);
    dim3 block({k.BLOCK_SIZE});
"""

    preamble = instance_impl_preamble("\n#include <algorithm>")
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
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int /*splitK*/)   // persistent ignores splitK; shares tune-lookup slot signature
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    AITER_CHECK(!bias.has_value(),
        "bias is not supported on a16w16_persistent kid; use a16w16 "
        "split-barrier (kid 4..9) or a16w16_flatmm_splitk (kid 200..299)");

    {kargs_name} kargs{{{{}}}};
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
{grid_setup}
    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_TUNE_HOST_EXTRA)


def gen_scale_instance(
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
    A8W8_SCALE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a8w8_scale launcher emit."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>>;
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
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> x_scale,
    std::optional<aiter_tensor_t> w_scale)
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    using Traits = {k.name}_Traits<D_C>;

    int GROUP_M = {k.GROUP_M};
    int GROUP_N = {k.GROUP_N};
    int GROUP_K = {k.GROUP_K};
    int num_groups_m = M / GROUP_M;
    int num_groups_n = N / GROUP_N;
    int num_groups_k = K / GROUP_K;

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

    kargs.ptr_sfa = x_scale.value().data_ptr();
    kargs.ptr_sfb = w_scale.value().data_ptr();
    kargs.stride_sfa = num_groups_k;
    kargs.stride_sfb = num_groups_k;
    kargs.stride_sfa_batch = num_groups_m * num_groups_k;
    kargs.stride_sfb_batch = num_groups_n * num_groups_k;

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A8W8_SCALE_HOST_EXTRA)


def gen_noscale_instance_gfx950(
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
    BIAS_HOST_VALIDATE,
    A16W16_TUNE_TAGS,
    **_unused,
):
    """gfx950 noscale launcher emit: a16w16 split-barrier (bias-aware double-traits)
    and a8w8 noscale (single traits). a8w8 falls through the else branch."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    is_a16w16_split_barrier = k.kernel_tag == "a16w16"
    is_a16w16_traits_with_tile_wave = (
        is_a16w16_split_barrier  # gfx950 noscale only a16w16 SB
    )
    traits_extra = ""
    if is_a16w16_traits_with_tile_wave:
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

    if k.kernel_tag in A16W16_TUNE_TAGS:
        extra_param = (
            ",\n    std::optional<aiter_tensor_t> bias," "\n    int /*splitK*/"
        )
    else:
        extra_param = ""

    has_oob_str = "true" if k.has_oob else "false"

    if is_a16w16_split_barrier:
        bias_kargs_block = (
            BIAS_HOST_VALIDATE
            + "    kargs.ptr_bias = ptr_bias_;\n"
            + "    kargs.stride_bias_batch = stride_bias_batch_;\n"
        )
    elif k.kernel_tag in A16W16_TUNE_TAGS:
        bias_kargs_block = (
            "    AITER_CHECK(!bias.has_value(),\n"
            '        "bias not supported on this a16w16 kid");\n'
        )
    else:
        bias_kargs_block = ""

    kargs_init_extra = ""

    cachectl_extra = ""
    if is_a16w16_split_barrier and hasattr(k, "cachectl_a") and k.cachectl_a >= 0:
        cachectl_extra = f",\n    {k.cachectl_a}, {k.cachectl_b}"
    traits_alias_tail = f",\n    {has_oob_str}"
    if is_a16w16_split_barrier:
        traits_aliases = f"""
template <typename D_C>
using {k.name}_TraitsNoBias = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
    false,
    D_C{traits_alias_tail}{cachectl_extra}>;
template <typename D_C>
using {k.name}_TraitsBias = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
    true,
    D_C{traits_alias_tail}{cachectl_extra}>;
"""
    else:
        traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra}>;
"""

    if is_a16w16_split_barrier:
        launch_block = f"""
    auto stream = aiter::getCurrentHIPStream();
    if (bias.has_value()) {{{{
        {kernel_func}<{k.name}_TraitsBias<D_C>><<<grid, block, 0, stream>>>(kargs);
    }}}} else {{{{
        {kernel_func}<{k.name}_TraitsNoBias<D_C>><<<grid, block, 0, stream>>>(kargs);
    }}}}"""
    else:
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
{kargs_init_extra}{bias_kargs_block}
    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});
{launch_block}

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    if k.kernel_tag in A16W16_TUNE_TAGS:
        inst_extra_param = ",\n    std::optional<aiter_tensor_t>,\n    int"
    else:
        inst_extra_param = ""

    if is_a16w16_split_barrier:

        def _device_decl(dtype):
            return (
                f"template __global__ void {kernel_func}<\n"
                f"    {k.name}_TraitsNoBias<{dtype}>>({kargs_name});\n"
                f"template __global__ void {kernel_func}<\n"
                f"    {k.name}_TraitsBias<{dtype}>>({kargs_name});\n"
            )

    else:

        def _device_decl(dtype):
            return (
                f"template __global__ void {kernel_func}<\n"
                f"    {k.name}_Traits<{dtype}>{kargs_explicit_param}>({kargs_name});\n"
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
            {"kid_name": k.name, "dtype": CDtype, "device_decl": _device_decl(CDtype)}
        )


def gen_mono_tile_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    **_unused,
):
    """gfx950 a16w16_mono_tile launcher emit."""
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>>;
"""
    min_k = 2 * k.B_K
    k_check = f"""
    int loops_ = K / {k.B_K};
    AITER_CHECK(K % {k.B_K} == 0,
        "mono-tile requires K divisible by B_K={k.B_K}; got K=", K);
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
    AITER_CHECK(batch >= 1, "batch must be >= 1");
    AITER_CHECK(N % {k.B_N} == 0,
        "mono-tile requires N divisible by B_N={k.B_N}; got N=", N);
"""
    INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
// See _gen_noscale_instance for the rationale of the host/device pass split.
#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
template<typename Traits>
__global__ void {kernel_func}({kargs_name} kargs);
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
    int /*splitK*/)
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    AITER_CHECK(!bias.has_value(),
        "bias is not supported on a16w16_mono_tile kid; use a16w16 "
        "split-barrier (kid 4..9) or a16w16_flatmm_splitk (kid 200..299)");

    {kargs_name} kargs{{{{}}}};
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

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    for CDtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{CDtype}>(\n"
            f"    aiter_tensor_t &XQ,\n"
            f"    aiter_tensor_t &WQ,\n"
            f"    aiter_tensor_t &Y,\n"
            f"    std::optional<aiter_tensor_t>,\n"
            f"    int);\n"
        )
        device_decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>>({kargs_name});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
        )


def gen_flatmm_instance(
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
    **_unused,
):
    """gfx950 a16w16_flatmm launcher emit."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_bias_str = "false"

    k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= Traits::prefetch_k_iter,
        "K=", K, " too small for flatmm B_K={k.B_K}, need K >= pfk*B_K = ",
        Traits::prefetch_k_iter * {k.B_K}, " (pfk=", Traits::prefetch_k_iter, ")");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1, "M, N, K must be >= 1");
    AITER_CHECK(batch >= 1, "batch must be >= 1");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
"""

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t, D_C>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {k.WG_PER_CU},
    {has_bias_str}>;
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
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int /*splitK*/)
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(!bias.has_value(),
        "bias is not yet supported on a16w16_flatmm kid; use a16w16 "
        "split-barrier (kid 4..9) or a16w16_flatmm_splitk (kid 200..299)");

    using Traits = {k.name}_Traits<D_C>;
{k_check}
    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.ptr_bias = nullptr;
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

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_TUNE_HOST_EXTRA)


def gen_flatmm_splitk_instance(
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
    BIAS_HOST_VALIDATE,
    **_unused,
):
    """gfx950 a16w16_flatmm_splitk launcher emit (uses ws_handle + reduce kernel call)."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_oob_str = "true" if k.has_oob else "false"
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, fp32_t, fp32_t, {da}>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {k.WG_PER_CU},
    false,
    {has_oob_str}>;
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
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK)
{{{{
    static_assert(std::is_same<D_C, fp32_t>::value,
        "splitk main kernel uses fp32 workspace; D_C template param must be fp32_t "
        "(Y can be bf16 or fp32; reduce kernel handles the cast / passthrough)");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16
                || Y.dtype() == AITER_DTYPE_fp32,
        "flatmm_splitk requires Y dtype bf16 or fp32 "
        "(reduce kernel casts fp32 workspace to D_OUT)");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1 && batch >= 1,
        "M, N, K, batch must be >= 1");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
{BIAS_HOST_VALIDATE}
    using Traits = {k.name}_Traits<D_C>;

    int split_k = (splitK <= 1) ? 1 : splitK;

    int total_iters = (K + {k.B_K} - 1) / {k.B_K};
    constexpr int pfk = Traits::prefetch_k_iter;
    while (split_k > 1) {{{{
        int iters_full = (total_iters + split_k - 1) / split_k;
        int last_loops = total_iters - (split_k - 1) * iters_full;
        if (iters_full >= pfk && last_loops >= pfk) break;
        split_k--;
    }}}}
    AITER_CHECK(total_iters >= pfk,
        "K=", K, " too small for flatmm_splitk B_K={k.B_K}: "
        "need total_iters >= pfk*B_K = ", pfk * {k.B_K},
        " (pfk=", pfk, ")");

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    int padded_M    = num_tiles_m * {k.B_M};
    int padded_N    = num_tiles_n * {k.B_N};

    // Per-stream workspace handle (process-global registry, mutex-protected
    // in opus_gemm.cu). Replaces the prior `static thread_local` cache --
    // under TBO two CPU threads drive two streams concurrently, and each
    // captured graph must bake in its own buffer pointer. Eager: lazy-
    // create. Capture: must be pre-warmed via
    // aiter.opus_gemm_workspace_init() on the capture stream.
    // (opus_splitk_ws_handle is already a complete type at this point via
    // the traits header included at the top of this launcher .cuh.)
    extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);

    auto stream = aiter::getCurrentHIPStream();
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
    const bool capturing = (capture_status != hipStreamCaptureStatusNone);
    auto* ws_handle_ = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

    size_t ws_bytes = (size_t)split_k * (size_t)batch
                    * (size_t)padded_M * (size_t)padded_N * sizeof(float);
    if (ws_handle_->ptr == nullptr || ws_bytes > ws_handle_->bytes)
    {{
        AITER_CHECK(!capturing,
            "splitk workspace grow inside HIP graph capture is not "
            "supported (hipMalloc / hipFree are stream-capture-illegal). "
            "Warm the cache once eagerly with the largest workspace before "
            "capturing. Call aiter.opus_gemm_workspace_init() on the capture "
            "stream first.");

        void* new_ptr = nullptr;
        const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
        size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
        HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
        if (ws_handle_->ptr != nullptr)
        {{
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipFree(ws_handle_->ptr));
        }}
        ws_handle_->ptr = new_ptr;
        ws_handle_->bytes = grow_bytes;
    }}

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a         = XQ.data_ptr();
    kargs.ptr_b         = WQ.data_ptr();
    kargs.ws_handle     = ws_handle_;
    kargs.ptr_c         = Y.data_ptr();
    kargs.ptr_bias      = ptr_bias_;
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
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

    dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
    dim3 block_main({k.BLOCK_SIZE});

    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                      batch * M, 1);
    dim3 block_reduce(REDUCE_BS);

    {kernel_func}<{k.name}_Traits<D_C>><<<grid_main, block_main, 0, stream>>>(kargs);
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
        if (bias.has_value()) {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, __bf16, true, __bf16, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const __bf16*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}} else {{{{
        if (bias.has_value()) {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const float*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}}

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_TUNE_HOST_EXTRA)


# ---------- Self-register at import time ----------
register_emit("gfx950", "a16w16_persistent", gen_persistent_instance)
register_emit("gfx950", "a8w8_scale", gen_scale_instance)
register_emit("gfx950", "a16w16", gen_noscale_instance_gfx950)
register_emit("gfx950", "a8w8", gen_noscale_instance_gfx950)
register_emit("gfx950", "a16w16_mono_tile", gen_mono_tile_instance)
register_emit("gfx950", "a16w16_flatmm", gen_flatmm_instance)
register_emit("gfx950", "a16w16_flatmm_splitk", gen_flatmm_splitk_instance)
