# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 codegen -- emit launchers for gfx1250-targeted kid families.

Wires the a16w16 cluster/TDM split-K pipeline that reduces via an fp32
WORKSPACE + a separate REDUCE kernel (no atomic_add), mirroring the gfx950
flatmm-splitk launcher (workspace + main kernel + reduce
kernel). The main kernel is always instantiated <fp32_t> (it writes the fp32
workspace); the reduce kernel casts the fp32 partials to the runtime Y dtype
(bf16 / fp32) and folds bias once.

Self-registers each emit into codegen.common.EMIT_REGISTRY at import time.
"""

import os
from pathlib import Path

from codegen.common import register_arch_map, register_emit

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
    # gfx1250 only: fp32 bias with a bf16 output (D_OUT=__bf16, D_BIAS=float).
    # The main kernel writes a bf16 workspace, so an fp32 bias folds in fp32 in
    # the reduce before the cast to bf16. The baseline instantiations cover the
    # matched-dtype cases; this adds the bf16-out + fp32-bias mix. Emitted for
    # every compile-time split_k (0=runtime fallback, 1..16=unrolled) and
    # HAS_OOB, with the bf16 workspace dtype (D_WS=__bf16). Same kernel NAME/ABI.
    out = (
        "// fp32-bias + bf16-out (gfx1250 f32 bias support), per split_k + D_WS=bf16\n"
    )
    for has_oob in ("true", "false"):
        for sk in range(0, 17):
            out += (
                f"template __global__ void splitk_reduce_kernel_gfx1250<8, 128, __bf16, true,  float,  {has_oob}, {sk}, __bf16>(\n"
                "    const void*, __bf16*, int, int, int, int, int, int,\n"
                "    const float*,  int);\n"
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
    """gfx1250 a16w16 TDM split-K (workspace + reduce) launcher emit.

    NO-CLUSTER grid: grid = (M/B_M, N/B_N, split_k); each WG owns one
    B_M x B_N tile (so M %% B_M == 0, N %% B_N == 0). The main kernel writes
    its split's fp32 partial into ws[split, padded_M, padded_N]; the reduce
    kernel sums split_k slices, folds bias, casts to Y dtype. batch handled by
    a per-batch host launch (sequential on stream -> workspace reuse is safe).
    """
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
    # Strict cluster-fill check emitted before the grid launch (the multicast mask
    # names every WG of the cluster -> the grid must fill it exactly).
    cluster_fill_check = ""
    if is_clusterlaunch:
        cluster_fill_check = (
            f"    // CLUSTER-LAUNCH: the multicast mask names EVERY WG of the "
            f"{cwm}x{cwn} cluster,\n"
            f"    // so ceil(M/B_M) and ceil(N/B_N) MUST be multiples of the "
            f"cluster dims\n"
            f"    // (no OOB tail WG, else the multicast + cluster barrier stalls).\n"
            f"    AITER_CHECK(num_tiles_m % {cwm} == 0 && num_tiles_n % {cwn} == 0,\n"
            f'        "gfx1250 clusterlaunch kid {cwm}x{cwn}: ceil(M/B_M)=", '
            f"num_tiles_m,\n"
            f'        " and ceil(N/B_N)=", num_tiles_n,\n'
            f'        " must both fill the cluster (divisible by {cwm}/{cwn})");\n'
        )

    # gfx1250-specific bias validation (does NOT use the shared BIAS_HOST_VALIDATE,
    # which forces bias.dtype == Y.dtype). The main kernel always writes an fp32
    # workspace and the reduce kernel folds bias in fp32 before the final cast to
    # Y, so an fp32 bias is exact for ANY Y dtype (bf16 or fp32). We therefore
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
    {da}, {db}, D_C, fp32_t,
    {enable_bias_str},
    {num_slots}, {wg_per_cu}{cluster_traits_args}>;
"""

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
// Reduce kernel forward-decl + split_k -> compile-time-instance launch
// dispatcher (opus_splitk_reduce_launch_gfx1250). The reduce kernel definition
// lives in gfx1250/splitk_reduce_gfx1250.cuh; explicit instantiations (per
// SPLIT_K + D_WS) live in the dedicated splitk_reduce_gfx1250.device.cu TU.
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
        "cluster_tdm_splitk_ws main kernel writes an fp32 workspace; D_C must "
        "be fp32_t (Y can be bf16 or fp32; the reduce kernel handles the cast)");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16 || Y.dtype() == AITER_DTYPE_fp32,
        "gfx1250 cluster_tdm_splitk_ws requires Y dtype bf16 or fp32");
    // M / N need NOT be multiples of B_M / B_N: the grid is padded to
    // ceil(M/B_M) x ceil(N/B_N) tiles, the main kernel TDM-clamps OOB global
    // reads to the real (M, N) tensor extents (tensor_dim1 = m - tile_row /
    // n - tile_col), partials for padded rows/cols land in the padded fp32
    // workspace, and the reduce kernel only iterates m in [0, M) and writes
    // n in [0, N) (HAS_OOB tail). So M=49 transparently runs as a padded
    // M=64 tile, etc.
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K)");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1 && batch >= 1,
        "M, N, K, batch must be >= 1");
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

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    int padded_M    = num_tiles_m * {k.B_M};
    int padded_N    = num_tiles_n * {k.B_N};

    auto stream = aiter::getCurrentHIPStream();
    void* ws_ptr_ = workspace.data_ptr();

{cluster_fill_check}    dim3 grid_main(num_tiles_m, num_tiles_n, split_k);
    dim3 block_main({k.BLOCK_SIZE});

    // VEC=8 -> each lane owns one dwordx4 of bf16 so the wave stores 512B fully
    // contiguous with no cross-lane shuffle (100% write-transaction efficiency),
    // and the fp32 workspace load drops from a 64B to a 32B lane stride. BLOCK=128
    // (4 waves) is the tuned reduce block; grid.x = ceil(N, VEC*BLOCK) is unchanged
    // vs the old VEC=16/BS=64 (both 1024 N per block).
    constexpr int REDUCE_VEC = 8;
    constexpr int REDUCE_BS  = 128;
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
    kargs.ptr_ws    = workspace.data_ptr();
    kargs.ptr_c     = Y.data_ptr();
    kargs.ptr_bias  = ptr_bias_;
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = 1; kargs.split_k = split_k;
    kargs.stride_a        = XQ.stride(1);
    kargs.stride_b        = WQ.stride(1);
    kargs.stride_ws       = padded_N;
    kargs.stride_c        = N;
    kargs.stride_a_batch  = XQ.stride(0);
    kargs.stride_b_batch  = WQ.stride(0);
    kargs.stride_ws_batch = padded_M * padded_N;
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
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, float, {has_oob_str}, __bf16>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const float*>(ptr_bias_), stride_bias_batch_);
        }}}} else if (ptr_bias_) {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, __bf16, {has_oob_str}, __bf16>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const __bf16*>(ptr_bias_), stride_bias_batch_);
        }}}} else {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}, __bf16>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
        }}}}
    }}}} else {{{{
        float* y_ptr = reinterpret_cast<float*>(Y.data_ptr());
        if (ptr_bias_) {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}, __bf16>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const float*>(ptr_bias_), stride_bias_batch_);
        }}}} else {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}, __bf16>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
        }}}}
    }}}}
}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    # Main kernel: only <fp32_t> is instantiated (writes the fp32 workspace).
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
    """gfx1250 FUSED single-kernel in-cluster split-K reduce launcher emit.

    No reduce kernel: the last split WG folds bias + reduces the partials in-kernel
    (cluster-barrier sync) and writes C directly. The kernel is templated on
    <Traits, SplitK, DataWs, MClusterWg, D_OUT>; SplitK / MClusterWg are compile-
    time (cluster dims), so each kid bakes one (tile, split_k, m_cluster, ws_dtype)
    combo. The launcher (instantiated <fp32_t> for the split-K lookup ABI) picks
    D_OUT = __bf16 / float at runtime from Y.dtype. Requires M %% B_M == 0,
    N %% B_N == 0 (no OOB C-store mask), ceil(M/B_M) %% MClusterWg == 0, and a
    compile-time SplitK with no empty trailing K-slice for the runtime K.
    """
    layout_int = _LAYOUT_INT[getattr(k, "ctdm_layout", "tileN")]
    enable_bias_str = "true" if getattr(k, "enable_bias", False) else "false"
    num_slots = getattr(k, "num_slots", 3)
    wg_per_cu = getattr(k, "wg_per_cu", 2)
    split_k = getattr(k, "fuse_split_k", 2)
    # fuse_m_cluster field holds the cluster's 2nd-dim WG count; for this pipeline
    # it groups N-tile peers (cluster.y, A-multicast), so expose it as n_cluster.
    n_cluster = getattr(k, "fuse_m_cluster", 1)
    ws_dtype = getattr(k, "fuse_ws_dtype", "bf16_t")
    ws_ctype, ws_bytes_elem = _FUSE_WS_CTYPE[ws_dtype]

    # Traits: 11-arg form (default cluster dims; the fuse kernel drives its own
    # __cluster_dims__(SplitK, MClusterWg, 1) and only uses the traits for tile
    # geometry / WindowA/B, not the traits cluster args).
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    {k.B_M}, {k.B_N}, {k.B_K},
    {layout_int},
    {da}, {db}, D_C, fp32_t,
    {enable_bias_str},
    {num_slots}, {wg_per_cu}>;
"""

    # Host expansion of __cluster_dims__ (the fused HOST TU includes hip_runtime.h,
    # not hip_minimal, so the attribute macro is otherwise not in scope -> the
    # launch would not form the cluster -> multicast + cluster barrier stall).
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
#include <optional>
#endif
#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
{cluster_dims_host_def}// Forward decl for the host <<<>>> launch stub. The __cluster_dims__ attribute
// uses this kid's CONCRETE (split_k, m_cluster) -- NOT the template params -- so
// the host launch site actually sets the cluster geometry (a template-parameter
// attribute does not propagate to the launch config; mirrors the ws clusterlaunch
// stub which also bakes concrete cluster dims).
template <typename Traits, int SplitK, typename DataWs, int MClusterWg, typename D_OUT>
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
        "splitk_fuse launcher uses the <fp32_t> split-K lookup ABI (D_C=fp32 traits;"
        " Y dtype is chosen at runtime as D_OUT)");
    (void)splitK;   // SplitK is compile-time ({split_k}); runtime splitK ignored.

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(batch == 1, "splitk_fuse is batch==1 only (got batch=", batch, ")");
    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16 || Y.dtype() == AITER_DTYPE_fp32,
        "splitk_fuse requires Y dtype bf16 or fp32");
    AITER_CHECK(K % 2 == 0, "K=", K, " must be even");
    AITER_CHECK(N % {k.B_N} == 0,
        "splitk_fuse writes full-N C tiles (no N OOB mask): N must be a "
        "multiple of B_N={k.B_N} (got N=", N, "). Ragged M is OK: the last "
        "M-tile's OOB rows fall past the C buffer num_records and are dropped.");

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};   // ceil: last M-tile may be partial (OOB rows dropped by C buffer num_records)
    int num_tiles_n = N / {k.B_N};
    // N-direction cluster (cluster.y groups {n_cluster} N-tile peers, A-multicast):
    // ceil(N/B_N) must be a multiple of the cluster N-peer count (exact fill; an
    // OOB tail WG would still be named in the multicast mask and stall the barrier).
    AITER_CHECK(num_tiles_n % {n_cluster} == 0,
        "splitk_fuse kid n_cluster={n_cluster}: ceil(N/B_N)=", num_tiles_n,
        " must be a multiple of n_cluster (cluster.y N-peer fill)");

    int k_steps_tot = (K + {k.B_K} - 1) / {k.B_K};
    // Balanced K-tile split (see pipeline): every split WG gets >=1 tile as long as
    // split_k <= k_steps_tot, so no WG is empty (the K tail is TDM-clamped, not
    // handled by emptying a WG). Only reject when there are fewer whole B_K tiles
    // than splits.
    AITER_CHECK({split_k} <= k_steps_tot,
        "splitk_fuse kid split_k={split_k} exceeds k_steps_tot=", k_steps_tot,
        " for K=", K, " (more splits than whole B_K tiles -> some WG would be empty);"
        " pick a kid with a smaller split_k for this K");

    // Bias: read as bf16 in-kernel; require bf16 (or absent) for round-1.
    const void* ptr_bias_ = nullptr;
    int stride_bias_batch_ = 0;
    if (bias.has_value()) {{{{
        const auto& bt = bias.value();
        AITER_CHECK(bt.is_contiguous(), "splitk_fuse bias must be contiguous");
        AITER_CHECK(bt.dtype() == AITER_DTYPE_bf16,
            "splitk_fuse bias must be bf16 (got ", AiterDtype_to_str(bt.dtype()), ")");
        if (bt.dim() == 1) {{{{
            AITER_CHECK(bt.size(0) == N, "splitk_fuse 1D bias length must equal N");
            stride_bias_batch_ = 0;
        }}}} else {{{{
            AITER_CHECK(false, "splitk_fuse round-1 supports only 1D [N] bias");
        }}}}
        ptr_bias_ = bt.data_ptr();
    }}}}

    using Traits = {k.name}_Traits<D_C>;

    auto stream = aiter::getCurrentHIPStream();

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a     = XQ.data_ptr();
    kargs.ptr_b     = WQ.data_ptr();
    kargs.ptr_ws    = workspace.data_ptr();
    kargs.ptr_c     = Y.data_ptr();
    kargs.ptr_bias  = ptr_bias_;
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = 1; kargs.split_k = {split_k};
    kargs.stride_a        = XQ.stride(1);
    kargs.stride_b        = WQ.stride(1);
    kargs.stride_c        = N;
    kargs.stride_a_batch  = XQ.stride(0);
    kargs.stride_b_batch  = WQ.stride(0);
    kargs.stride_c_batch  = M * N;
    kargs.stride_bias_batch = stride_bias_batch_;
    kargs.num_tiles_m = num_tiles_m;
    kargs.num_tiles_n = num_tiles_n;

    // N-direction cluster: N-tiles on grid.y so cluster.y groups the {n_cluster}
    // N-peers (A-multicast); M-tiles on grid.z. cluster = ({split_k}, {n_cluster}, 1).
    dim3 grid_main({split_k}, num_tiles_n, num_tiles_m);
    dim3 block_main({k.BLOCK_SIZE});
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
        {kernel_func}<Traits, {split_k}, {ws_ctype}, {n_cluster}, __bf16>
            <<<grid_main, block_main, 0, stream>>>(kargs);
    }}}} else {{{{
        {kernel_func}<Traits, {split_k}, {ws_ctype}, {n_cluster}, float>
            <<<grid_main, block_main, 0, stream>>>(kargs);
    }}}}
}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    # Host launcher: <fp32_t> only (split-K lookup ABI). Device kernel: both D_OUT.
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
            f"    {k.name}_Traits<fp32_t>, {split_k}, {ws_ctype}, {n_cluster}, {d_out}>"
            f"({kargs_name});\n"
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
# __cluster_dims__, the cluster-fill check, and the CLUSTER_WG_M/N traits args).
register_emit(
    "gfx1250", "a16w16_clusterlaunch_tdm_splitk_ws", gen_cluster_tdm_splitk_ws_instance
)
