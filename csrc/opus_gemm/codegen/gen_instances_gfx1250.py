# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate gfx1250 OPUS A16W16 launchers."""

import os
from pathlib import Path

from codegen.common import register_arch_map, register_emit, splitk_workspace_type

# ---------------- gfx1250 arch-override maps ----------------

PIPELINE_HEADER_MAP = {
    "a16w16_cluster_tdm_splitk_ws": (
        "gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh"
    ),
    "a16w16_clusterlaunch_tdm_splitk_ws": (
        "gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh"
    ),
    "a16w16_clusterlaunch_tdm_splitk_fuse": (
        "gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh"
    ),
}

TRAITS_HEADER_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_clusterlaunch_tdm_splitk_ws": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
}

KERNEL_FUNC_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gemm_a16w16_cluster_tdm_splitk_ws_kernel_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "gemm_a16w16_clusterlaunch_tdm_splitk_ws_kernel_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "gemm_a16w16_splitk_fuse_kernel_gfx1250",
}

TRAITS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
}

KARGS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_gemm_cluster_tdm_ws_kargs_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "opus_gemm_cluster_tdm_ws_kargs_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "opus_gemm_splitk_fuse_kargs_gfx1250",
}

# fuse workspace storage dtype -> (C type, byte size) for the fuse kernel instantiation.
_FUSE_WS_CTYPE = {"bf16_t": ("__bf16", 2), "fp32_t": ("float", 4)}


def splitk_reduce_extra_device_instantiations():
    # The shared generator emits matched output/bias combinations for both
    # physical workspace types and every split_k specialization. gfx1250 also
    # accepts fp32 bias with bf16 output, so add that mixed combination here.
    out = "// gfx1250 fp32-bias + bf16-output reduce variants\n"
    for has_oob in ("true", "false"):
        for split_k in range(17):
            for workspace_type in ("__bf16", "float"):
                out += (
                    "template __global__ void "
                    "splitk_reduce_kernel_gfx1250<"
                    f"8, 128, __bf16, true, float, {has_oob}, "
                    f"{split_k}, {workspace_type}>(\n"
                    "    const void*, __bf16*, int, int, int, int, int, int,\n"
                    "    const float*, int);\n"
                )
    return out


SPLITK_REDUCE_EXTRA_MAP = {
    "device_instantiations": splitk_reduce_extra_device_instantiations,
}

register_arch_map("gfx1250", "pipeline_header", PIPELINE_HEADER_MAP)
register_arch_map("gfx1250", "traits_header", TRAITS_HEADER_MAP)
register_arch_map("gfx1250", "kernel_func", KERNEL_FUNC_MAP)
register_arch_map("gfx1250", "traits_name", TRAITS_NAME_MAP)
register_arch_map("gfx1250", "kargs_name", KARGS_NAME_MAP)
register_arch_map("gfx1250", "splitk_reduce_extra", SPLITK_REDUCE_EXTRA_MAP)

# tileN = consumers split N (B_N>=32); tileM = consumers split M (B_M>=32).
_LAYOUT_INT = {"tileN": 0, "tileM": 1}


# ---------------- gfx1250 emit ----------------


def gen_cluster_tdm_splitk_ws_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    BIAS_HOST_VALIDATE="",
    **_unused,
):
    """Emit a checked gfx1250 two-stage split-K launcher."""
    workspace_dtype, workspace_ptr_type, workspace_aiter_dtype = splitk_workspace_type(
        k
    )
    # The final #4246 reducer uses the same coalesced VEC=8/BLOCK=128 geometry
    # for either physical workspace type.
    reduce_vec, reduce_bs = 8, 128
    layout_int = _LAYOUT_INT[getattr(k, "ctdm_layout", "tileN")]
    has_oob_str = "true" if k.has_oob else "false"
    enable_bias_str = "true" if getattr(k, "enable_bias", False) else "false"

    # CLUSTER-LAUNCH variant: __cluster_dims__(CWM, CWN, 1) multicast TDM. The
    # plain (no-cluster) variant leaves these empty so it is unchanged.
    is_clusterlaunch = k.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_ws"
    cwm = getattr(k, "cluster_wg_m", 4)
    cwn = getattr(k, "cluster_wg_n", 4)
    # Extra traits template args (CLUSTER_WG_M, CLUSTER_WG_N) appended only for the
    # clusterlaunch tag; the plain base keeps the 11-arg form (defaults apply).
    cluster_traits_args = f",\n    {cwm}, {cwn}" if is_clusterlaunch else ""
    # __cluster_dims__ attribute on the host-side forward-decl stub so the <<<>>>
    # launch sets the cluster geometry (must match the kernel definition).
    cluster_dims_attr = (
        f"__cluster_dims__({cwm}, {cwn}, 1)\n" if is_clusterlaunch else ""
    )
    # Host-pass expansion of __cluster_dims__: the kernel DEFINITION (device TU)
    # gets the cluster_dims attribute via the gfx1250-gated hip_minimal macro, but
    # the fused HOST TU (where the <<<>>> launch lives) includes <hip/hip_runtime.h>
    # (not hip_minimal), so the macro is not in scope there and the launch site
    # would NOT carry the cluster geometry -> WG cluster never forms -> TDM
    # multicast degrades to per-load timeout (correct but ~5x slow). Define it
    # here for the host pass so the forward-decl's attribute actually expands and
    # the launch applies the cluster dims (matches the single-file standalone).
    cluster_dims_host_def = (
        "#ifndef __cluster_dims__\n"
        "#define __cluster_dims__(...) __attribute__((cluster_dims(__VA_ARGS__)))\n"
        "#endif\n"
        if is_clusterlaunch
        else ""
    )
    # A cluster launch grid must contain whole clusters. Round only the physical
    # launch grid up; logical tile counts and workspace strides remain unrounded.
    cluster_grid_roundup = ""
    grid_m_expr, grid_n_expr = "num_tiles_m", "num_tiles_n"
    if is_clusterlaunch:
        cluster_grid_roundup = (
            f"    // Round the physical grid to complete {cwm}x{cwn} clusters.\n"
            f"    // Surplus WGs take the pipeline tile_oob exit; workspace layout\n"
            f"    // continues to use the unrounded logical tile counts.\n"
            f"    int grid_tiles_m = (num_tiles_m + {cwm} - 1) / {cwm} * {cwm};\n"
            f"    int grid_tiles_n = (num_tiles_n + {cwn} - 1) / {cwn} * {cwn};\n"
        )
        grid_m_expr, grid_n_expr = "grid_tiles_m", "grid_tiles_n"

    # gfx1250-specific bias validation (does NOT use the shared BIAS_HOST_VALIDATE,
    # which forces bias.dtype == Y.dtype). The reduce kernel folds bias into its
    # fp32 accumulator before the final cast to Y, regardless of workspace
    # storage, so an fp32 bias is exact for ANY Y dtype (bf16 or fp32). We therefore
    # accept bias.dtype in {{fp32, Y.dtype}} and record bias_is_fp32_ so the reduce
    # launch below can pick the matching D_BIAS template. (Double C++ braces are
    # intentional -- this string is inserted verbatim into the f-string template.)
    gfx1250_bias_validate = """
    const void* ptr_bias_ = nullptr;
    int stride_bias_batch_ = 0;
    bool bias_is_fp32_ = false;
    if (bias.has_value()) {{
        const auto& bt = bias.value();
        AITER_CHECK(bt.is_contiguous(),
            "bias must be contiguous (got non-contiguous tensor)");
        AITER_CHECK(bt.dtype() == AITER_DTYPE_fp32 || bt.dtype() == Y.dtype(),
            "bias dtype must be fp32 or match Y dtype (got bias=",
            AiterDtype_to_str(bt.dtype()),
            " Y=", AiterDtype_to_str(Y.dtype()), ")");
        bias_is_fp32_ = (bt.dtype() == AITER_DTYPE_fp32);
        if (bt.dim() == 1) {{
            AITER_CHECK(bt.size(0) == N,
                "bias 1D length must equal N (got bias.size(0)=", bt.size(0),
                " N=", N, ")");
            stride_bias_batch_ = 0;
        }} else if (bt.dim() == 2) {{
            AITER_CHECK(bt.size(0) == batch && bt.size(1) == N,
                "bias 2D shape must equal [batch, N] (got [", bt.size(0), ", ",
                bt.size(1), "] vs batch=", batch, " N=", N, ")");
            stride_bias_batch_ = N;
        }} else {{
            AITER_CHECK(false, "bias must be 1D [N] or 2D [batch, N]; got dim=",
                bt.dim());
        }}
        ptr_bias_ = bt.data_ptr();
    }}
"""

    num_slots = getattr(k, "num_slots", 3)
    wg_per_cu = getattr(k, "wg_per_cu", 2)
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    {k.B_M}, {k.B_N}, {k.B_K},
    {layout_int},
    {da}, {db}, {workspace_dtype}, fp32_t,
    {enable_bias_str},
    {num_slots}, {wg_per_cu}{cluster_traits_args}>;
"""

    INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "opus_gemm_common.cuh"
#include <optional>
#endif
#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
{cluster_dims_host_def}// Forward declaration for the host-side <<<>>> launch stub. Must match the
// kernel's __launch_bounds__ (and __cluster_dims__ for the clusterlaunch tag, so
// the <<<>>> launch sets the cluster geometry).
template<typename Traits>
__global__ __launch_bounds__(128, 1)
{cluster_dims_attr}void {kernel_func}({kargs_name} kargs);
#else
#include "{pipeline_header}"
#endif
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
// Host launch helper dispatches runtime split_k to the matching compile-time
// reducer specialization. The device definitions remain in the per-arch TU.
#include "gfx1250/splitk_reduce_launch_gfx1250.cuh"

template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &workspace,
    std::optional<aiter_tensor_t> bias,
    int splitK)
{{{{
    static_assert(std::is_same<D_C, fp32_t>::value,
        "cluster_tdm_splitk_ws uses the fp32 launch-dispatch specialization");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16 || Y.dtype() == AITER_DTYPE_fp32,
        "gfx1250 cluster_tdm_splitk_ws requires Y dtype bf16 or fp32");
    // M / N need NOT be multiples of B_M / B_N: the grid is padded to
    // ceil(M/B_M) x ceil(N/B_N) tiles, the main kernel TDM-clamps OOB global
    // reads to the real (M, N) tensor extents (tensor_dim1 = m - tile_row /
    // n - tile_col), partials for padded rows/cols land in the padded typed
    // workspace, and the reduce kernel only iterates m in [0, M) and writes
    // n in [0, N) (HAS_OOB tail). So M=49 transparently runs as a padded
    // M=64 tile, etc.
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K)");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1 && batch >= 1,
        "M, N, K, batch must be >= 1");
    AITER_CHECK(batch == 1,
        "gfx1250 cluster_tdm_splitk_ws supports batch == 1 only; got batch=",
        batch);
{gfx1250_bias_validate}
    using Traits = {k.name}_Traits<D_C>;

    int split_k = (splitK <= 1) ? 1 : splitK;
    int k_steps_tot = (K + {k.B_K} - 1) / {k.B_K};
    // Clamp split_k so there is no empty trailing split -> n_active == split_k,
    // so the reduce can sum all split_k slices (no garbage from unwritten ones).
    while (split_k > 1) {{{{
        int steps_per = (k_steps_tot + split_k - 1) / split_k;
        if ((split_k - 1) * steps_per < k_steps_tot) break;
        split_k--;
    }}}}

    int num_tiles_m = 1 + (M - 1) / {k.B_M};
    int num_tiles_n = 1 + (N - 1) / {k.B_N};
    const size_t padded_M_size = opus_checked_extent_product(
        {{static_cast<size_t>(num_tiles_m), static_cast<size_t>({k.B_M})}},
        "{k.name}");
    const size_t padded_N_size = opus_checked_extent_product(
        {{static_cast<size_t>(num_tiles_n), static_cast<size_t>({k.B_N})}},
        "{k.name}");
    const size_t workspace_slice_numel = opus_checked_extent_product(
        {{padded_M_size, padded_N_size}}, "{k.name}");
    AITER_CHECK(padded_M_size <= static_cast<size_t>(std::numeric_limits<int>::max())
                    && padded_N_size <= static_cast<size_t>(std::numeric_limits<int>::max())
                    && workspace_slice_numel <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "{k.name}: padded workspace extents exceed 32-bit kernel stride limits");
    int padded_M = static_cast<int>(padded_M_size);
    int padded_N = static_cast<int>(padded_N_size);

    // One-batch layout: [split_k, padded_M, padded_N].
    const size_t required_numel = opus_checked_extent_product(
        {{static_cast<size_t>(split_k), workspace_slice_numel}},
        "{k.name}");
    void* workspace_ptr_ = opus_validate_workspace(
        workspace, XQ, {workspace_aiter_dtype}, required_numel, 16, "{k.name}");
    auto stream = aiter::getCurrentHIPStream();

{cluster_grid_roundup}    dim3 grid_main({grid_m_expr}, {grid_n_expr}, split_k);
    dim3 block_main({k.BLOCK_SIZE});

    constexpr int REDUCE_VEC = {reduce_vec};
    constexpr int REDUCE_BS  = {reduce_bs};
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS), M, 1);
    dim3 block_reduce(REDUCE_BS);

    // gfx1250 cluster_tdm_splitk_ws is batch==1 only (the Python layout guard
    // and the 3D grid both assume a single batch). A single main + reduce
    // launch handles the whole gemm -- no host batch loop, no per-batch
    // pointer / bias offsets. The kernels still take stride_*_batch but with
    // batch==1 every batch term collapses (b==0, split_stride==stride_ws_batch).
    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a     = XQ.data_ptr();
    kargs.ptr_b     = WQ.data_ptr();
    kargs.ptr_ws    = workspace_ptr_;
    kargs.ptr_c     = Y.data_ptr();
    kargs.ptr_bias  = ptr_bias_;
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = 1; kargs.split_k = split_k;
    kargs.stride_a        = XQ.stride(1);
    kargs.stride_b        = WQ.stride(1);
    kargs.stride_ws       = padded_N;
    kargs.stride_c        = N;
    kargs.stride_a_batch  = XQ.stride(0);
    kargs.stride_b_batch  = WQ.stride(0);
    kargs.stride_ws_batch = static_cast<int>(workspace_slice_numel);
    kargs.stride_c_batch  = M * N;
    kargs.stride_bias_batch = stride_bias_batch_;

    {kernel_func}<Traits><<<grid_main, block_main, 0, stream>>>(kargs);

    // Reduce reads the bf16 split-K workspace the main kernel wrote (D_WS=__bf16),
    // re-accumulates in fp32, folds bias, casts to Y dtype. split_k is dispatched
    // to a compile-time (unrolled) reduce instance by the launch helper.
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
        __bf16* y_ptr = reinterpret_cast<__bf16*>(Y.data_ptr());
        if (ptr_bias_ && bias_is_fp32_) {{{{
            // fp32 bias + bf16 output: fold the exact fp32 bias in the
            // reduce (D_BIAS=float), then cast the fp32 sum to bf16.
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, float, {has_oob_str}, {workspace_ptr_type}>(
                grid_reduce, block_reduce, stream,
                workspace_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const float*>(ptr_bias_), stride_bias_batch_);
        }}}} else if (ptr_bias_) {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, __bf16, {has_oob_str}, {workspace_ptr_type}>(
                grid_reduce, block_reduce, stream,
                workspace_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const __bf16*>(ptr_bias_), stride_bias_batch_);
        }}}} else {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}, {workspace_ptr_type}>(
                grid_reduce, block_reduce, stream,
                workspace_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
        }}}}
    }}}} else {{{{
        float* y_ptr = reinterpret_cast<float*>(Y.data_ptr());
        if (ptr_bias_) {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}, {workspace_ptr_type}>(
                grid_reduce, block_reduce, stream,
                workspace_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const float*>(ptr_bias_), stride_bias_batch_);
        }}}} else {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}, {workspace_ptr_type}>(
                grid_reduce, block_reduce, stream,
                workspace_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
        }}}}
    }}}}
}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    # The <fp32_t> token is the host launch-dispatch specialization. The physical
    # workspace type is independently embedded in the Traits alias above.
    for CDtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{CDtype}>(\n"
            f"    aiter_tensor_t &XQ,\n"
            f"    aiter_tensor_t &WQ,\n"
            f"    aiter_tensor_t &Y,\n"
            f"    aiter_tensor_t &workspace,\n"
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


def gen_splitk_fuse_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    BIAS_HOST_VALIDATE="",
    **_unused,
):
    """Emit the fused gfx1250 split-K launcher and workspace checks."""
    del BIAS_HOST_VALIDATE
    workspace_dtype, workspace_ptr_type, workspace_aiter_dtype = splitk_workspace_type(
        k
    )
    layout_int = _LAYOUT_INT[getattr(k, "ctdm_layout", "tileN")]
    enable_bias_str = "true" if getattr(k, "enable_bias", False) else "false"
    num_slots = getattr(k, "num_slots", 3)
    wg_per_cu = getattr(k, "wg_per_cu", 2)
    split_k = int(getattr(k, "fuse_split_k", 0))
    # Historical field name retained for compatibility; physically cluster.y
    # groups N-tile peers sharing A.
    n_cluster = int(getattr(k, "fuse_m_cluster", 1))
    if split_k < 2:
        raise ValueError(f"fused instance {k.name} must declare fuse_split_k >= 2")

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    {k.B_M}, {k.B_N}, {k.B_K},
    {layout_int},
    {da}, {db}, {workspace_dtype}, fp32_t,
    {enable_bias_str},
    {num_slots}, {wg_per_cu}>;
"""

    cluster_dims_host_def = (
        "#ifndef __cluster_dims__\n"
        "#define __cluster_dims__(...) __attribute__((cluster_dims(__VA_ARGS__)))\n"
        "#endif\n"
    )

    INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "opus_gemm_common.cuh"
#include <optional>
#endif
#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
{cluster_dims_host_def}// Concrete cluster geometry must be visible on the host launch stub.
template <typename Traits, int SplitK, typename DataWs, int NClusterWg, typename D_OUT>
__global__ __launch_bounds__(128, 1)
__cluster_dims__({split_k}, {n_cluster}, 1)
void {kernel_func}({kargs_name} kargs);
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
    aiter_tensor_t &workspace,
    std::optional<aiter_tensor_t> bias,
    int splitK)
{{{{
    static_assert(std::is_same<D_C, fp32_t>::value,
        "splitk_fuse uses the fp32 workspace-dispatch specialization");
    (void)splitK;  // SplitK is compile-time ({split_k}) for this exact kid.

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(batch == 1,
        "gfx1250 splitk_fuse supports batch == 1 only; got batch=", batch);
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1,
        "splitk_fuse requires positive M, N, and K");
    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16 || Y.dtype() == AITER_DTYPE_fp32,
        "splitk_fuse requires Y dtype bf16 or fp32");
    AITER_CHECK(K % 2 == 0, "K=", K, " must be even");
    AITER_CHECK(N % {k.B_N} == 0,
        "splitk_fuse writes full-N C tiles: N must be a multiple of B_N={k.B_N}; got N=",
        N, ". Ragged M remains supported by the bounded C descriptor.");

    int num_tiles_m = 1 + (M - 1) / {k.B_M};
    int num_tiles_n = N / {k.B_N};
    AITER_CHECK(num_tiles_n % {n_cluster} == 0,
        "splitk_fuse kid n_cluster={n_cluster}: N/B_N=", num_tiles_n,
        " must exactly fill cluster.y");

    int k_steps_tot = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK({split_k} <= k_steps_tot,
        "splitk_fuse kid split_k={split_k} exceeds K-tile count ", k_steps_tot,
        " for K=", K, " and B_K={k.B_K}");

    // #4246 round-1 bias contract: contiguous bf16 [N].
    const void* ptr_bias_ = nullptr;
    int stride_bias_batch_ = 0;
    if (bias.has_value()) {{{{
        const auto& bt = bias.value();
        AITER_CHECK(bt.is_contiguous(), "splitk_fuse bias must be contiguous");
        AITER_CHECK(bt.dtype() == AITER_DTYPE_bf16,
            "splitk_fuse bias must be bf16; got ", AiterDtype_to_str(bt.dtype()));
        AITER_CHECK(bt.dim() == 1 && bt.size(0) == N,
            "splitk_fuse bias must have shape [N]; got dim=", bt.dim());
        ptr_bias_ = bt.data_ptr();
    }}}}

    // Physical layout: [num_tiles_m, num_tiles_n, SplitK-1, B_M, B_N].
    const size_t tile_numel = opus_checked_extent_product(
        {{static_cast<size_t>({k.B_M}), static_cast<size_t>({k.B_N})}},
        "{k.name}");
    const size_t required_numel = opus_checked_extent_product(
        {{static_cast<size_t>(num_tiles_m),
           static_cast<size_t>(num_tiles_n),
           static_cast<size_t>({split_k - 1}),
           tile_numel}},
        "{k.name}");
    void* workspace_ptr_ = opus_validate_workspace(
        workspace, XQ, {workspace_aiter_dtype}, required_numel, 16, "{k.name}");

    using Traits = {k.name}_Traits<D_C>;
    auto stream = aiter::getCurrentHIPStream();

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_ws = workspace_ptr_;
    kargs.ptr_c = Y.data_ptr();
    kargs.ptr_bias = ptr_bias_;
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = 1;
    kargs.split_k = {split_k};
    kargs.stride_a = XQ.stride(1);
    kargs.stride_b = WQ.stride(1);
    kargs.stride_c = N;
    kargs.stride_a_batch = XQ.stride(0);
    kargs.stride_b_batch = WQ.stride(0);
    kargs.stride_c_batch = M * N;
    kargs.stride_bias_batch = stride_bias_batch_;
    kargs.num_tiles_m = num_tiles_m;
    kargs.num_tiles_n = num_tiles_n;

    // cluster = (SplitK, N peers, 1); M tiles occupy grid.z.
    dim3 grid_main({split_k}, num_tiles_n, num_tiles_m);
    dim3 block_main({k.BLOCK_SIZE});
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
        {kernel_func}<Traits, {split_k}, {workspace_ptr_type}, {n_cluster}, __bf16>
            <<<grid_main, block_main, 0, stream>>>(kargs);
    }}}} else {{{{
        {kernel_func}<Traits, {split_k}, {workspace_ptr_type}, {n_cluster}, float>
            <<<grid_main, block_main, 0, stream>>>(kargs);
    }}}}
}}}}
#endif
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    host_decl = (
        f"template void\n"
        f"{k.name}<fp32_t>(\n"
        f"    aiter_tensor_t &XQ,\n"
        f"    aiter_tensor_t &WQ,\n"
        f"    aiter_tensor_t &Y,\n"
        f"    aiter_tensor_t &workspace,\n"
        f"    std::optional<aiter_tensor_t>,\n"
        f"    int);\n"
    )
    cg._host_instantiations.append(
        {"kid_name": k.name, "dtype": "fp32_t", "host_decl": host_decl}
    )
    for d_out in ("__bf16", "float"):
        device_decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<fp32_t>, {split_k}, {workspace_ptr_type}, "
            f"{n_cluster}, {d_out}>({kargs_name});\n"
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": d_out, "device_decl": device_decl}
        )


# ---------- Self-register at import time ----------
register_emit(
    "gfx1250", "a16w16_cluster_tdm_splitk_ws", gen_cluster_tdm_splitk_ws_instance
)
register_emit(
    "gfx1250", "a16w16_clusterlaunch_tdm_splitk_fuse", gen_splitk_fuse_instance
)
# CLUSTER-LAUNCH variant shares the same emit (it branches on k.kernel_tag to add
# __cluster_dims__, physical-grid round-up, and the CLUSTER_WG_M/N traits args).
register_emit(
    "gfx1250", "a16w16_clusterlaunch_tdm_splitk_ws", gen_cluster_tdm_splitk_ws_instance
)
