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
    # 4wave_co has a real pipeline header, but NOTHING in the JIT build may
    # include it: it uses pin builtins a release toolchain does not have, and
    # its device code comes from a pre-built .co instead. Point the entry at the
    # traits header so any generic consumer of this map stays on safe ground --
    # the co emit below never uses it at all.
    "a16w16_4wave_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_4wave_wl_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
}

TRAITS_HEADER_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_clusterlaunch_tdm_splitk_ws": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_4wave_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_4wave_wl_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
}

KERNEL_FUNC_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gemm_a16w16_cluster_tdm_splitk_ws_kernel_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "gemm_a16w16_clusterlaunch_tdm_splitk_ws_kernel_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "gemm_a16w16_splitk_fuse_kernel_gfx1250",
    # Device-side body name. The JIT never instantiates it (the .co does); it is
    # here because gen_instance() resolves this map for every kid, and because
    # build_co.py reads it to spell the stub TU's call.
    "a16w16_4wave_co": "gemm_a16w16_4wave_compute_body_gfx1250",
    "a16w16_4wave_wl_co": "gemm_a16w16_4wave_wl_body_gfx1250",
}

TRAITS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_4wave_co": "opus_a16w16_4wave_compute_traits_gfx1250",
    "a16w16_4wave_wl_co": "opus_a16w16_4wave_wl_traits_gfx1250",
}

KARGS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_gemm_cluster_tdm_ws_kargs_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "opus_gemm_cluster_tdm_ws_kargs_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "opus_gemm_splitk_fuse_kargs_gfx1250",
    "a16w16_4wave_co": "opus_gemm_4wave_compute_kargs_gfx1250",
    "a16w16_4wave_wl_co": "opus_gemm_4wave_compute_kargs_gfx1250",
}


# 4wave_co traits argument list, shared by the JIT launcher emit below and the
# offline stub TU (gen_co/build_co.py imports this). One spelling, so the .co's
# baked traits and the host launcher's tile constants cannot disagree.
def co_traits_args(k):
    d_a, d_b, d_c, d_acc = k.co_dtypes
    return (
        f"{k.BLOCK_SIZE}, {k.B_M}, {k.B_N}, {k.B_K}, {k.num_slots}, "
        f"{d_a}, {d_b}, {d_c}, {d_acc}, "
        f"{k.cluster_wg_m}, {k.cluster_wg_n}"
        # The wave-layout family takes two more: how the 4 waves tile the block.
        + (
            f", {k.co_wave_layout[0]}, {k.co_wave_layout[1]}"
            if k.kernel_tag == "a16w16_4wave_wl_co"
            else ""
        )
    )


# fuse workspace storage dtype -> (C type, byte size) for the fuse kernel instantiation.
_FUSE_WS_CTYPE = {"bf16_t": ("__bf16", 2), "fp32_t": ("float", 4)}


def splitk_reduce_extra_device_instantiations():
    # gfx1250 only: fp32 bias with a bf16 output (D_OUT=__bf16, D_BIAS=float).
    # The main kernel writes the partial workspace, so an fp32 bias folds in fp32 in
    # the reduce before the cast to bf16. The baseline instantiations cover the
    # matched-dtype cases; this adds the bf16-out + fp32-bias mix. Emitted for
    # every compile-time split_k (0=runtime fallback, 1..16=unrolled) and
    # HAS_OOB, and for BOTH partial types -- which one a kid uses is its
    # splitk_workspace_dtype, so both have to exist. Same kernel NAME/ABI.
    out = "// fp32-bias + bf16-out (gfx1250 f32 bias support), per split_k + D_WS\n"
    for d_ws in ("__bf16", "float"):
        for has_oob in ("true", "false"):
            for sk in range(17):
                out += (
                    f"template __global__ void splitk_reduce_kernel_gfx1250<8, 128, __bf16, true,  float,  {has_oob}, {sk}, {d_ws}>(\n"
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
    # Cluster round-up emitted before the grid launch: the runtime rejects a grid
    # that is not a whole number of clusters. The surplus workgroups own no tile and
    # return right after their one cluster-barrier arrival (tile_oob in the pipeline),
    # so any (M, N) is launchable with any cluster dims -- no exact-fill assert.
    cluster_grid_roundup = ""
    grid_m_expr, grid_n_expr = "num_tiles_m", "num_tiles_n"
    if is_clusterlaunch:
        cluster_grid_roundup = (
            f"    // CLUSTER-LAUNCH: the grid must be a whole number of "
            f"{cwm}x{cwn} clusters, so\n"
            f"    // round the tile counts up. The surplus workgroups have no tile and "
            f"leave at\n"
            f"    // the pipeline's tile_oob exit; the workspace strides below stay on "
            f"the\n"
            f"    // UNROUNDED counts, so the reduce kernel is unaffected by the "
            f"padding.\n"
            f"    int grid_tiles_m = (num_tiles_m + {cwm} - 1) / {cwm} * {cwm};\n"
            f"    int grid_tiles_n = (num_tiles_n + {cwn} - 1) / {cwn} * {cwn};\n"
        )
        grid_m_expr, grid_n_expr = "grid_tiles_m", "grid_tiles_n"

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
    // D_C is the split-K PARTIAL type (this kernel stores it, the reduce below
    // reads it and casts to Y), picked per kid by splitk_workspace_dtype. Y is
    // independent and may be bf16 or fp32 either way.
    static_assert(std::is_same<D_C, fp32_t>::value || std::is_same<D_C, bf16_t>::value,
        "cluster_tdm_splitk_ws split-K partial must be fp32_t or bf16_t");

    // The host sizes this buffer from the kid table's splitk_workspace_dtype
    // while D_C was baked in at build time, so a .so that is stale with respect
    // to the table makes the two disagree. Catch it here: unchecked, a narrower
    // buffer than D_C is a GPU page fault with no hint of where it came from.
    AITER_CHECK(workspace.element_size() == sizeof(D_C),
        "split-K workspace is ", workspace.element_size(),
        "-byte but this kernel stores ", sizeof(D_C),
        "-byte partials -- rebuild module_deepgemm_opus after changing "
        "splitk_workspace_dtype");

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

{cluster_grid_roundup}    dim3 grid_main({grid_m_expr}, {grid_n_expr}, split_k);
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

    // Reduce reads the split-K workspace the main kernel wrote. D_C is that
    // partial type for BOTH ends -- the main kernel stores Traits::DataC and
    // this reads the same D_C -- so a per-kid choice cannot desynchronise them,
    // re-accumulates in fp32, folds bias, casts to Y dtype. split_k is dispatched
    // to a compile-time (unrolled) reduce instance by the launch helper.
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
        __bf16* y_ptr = reinterpret_cast<__bf16*>(Y.data_ptr());
        if (ptr_bias_ && bias_is_fp32_) {{{{
            // fp32 bias + bf16 output: fold the exact fp32 bias in the
            // reduce (D_BIAS=float), then cast the fp32 sum to bf16.
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, float, {has_oob_str}, D_C>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const float*>(ptr_bias_), stride_bias_batch_);
        }}}} else if (ptr_bias_) {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, __bf16, {has_oob_str}, D_C>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const __bf16*>(ptr_bias_), stride_bias_batch_);
        }}}} else {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}, D_C>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
        }}}}
    }}}} else {{{{
        float* y_ptr = reinterpret_cast<float*>(Y.data_ptr());
        if (ptr_bias_) {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}, D_C>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                reinterpret_cast<const float*>(ptr_bias_), stride_bias_batch_);
        }}}} else {{{{
            opus_splitk_reduce_launch_gfx1250<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}, D_C>(
                grid_reduce, block_reduce, stream,
                ws_ptr_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
        }}}}
    }}}}
}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    # The kid's template slot is the split-K PARTIAL type (traits D_C), not its
    # output dtype -- the reduce decides the output. Instantiate the partial type
    # the kid declares, since the dispatch table now references it by that type
    # and the host sizes the workspace from the same field. The fuse family keeps
    # output_dtypes: it reduces in-kernel and never lands a partial for us.
    _ws_partial_tags = (
        "a16w16_cluster_tdm_splitk_ws",
        "a16w16_clusterlaunch_tdm_splitk_ws",
    )
    if k.kernel_tag in _ws_partial_tags:
        inst_dtypes = (getattr(k, "splitk_workspace_dtype", "fp32_t"),)
    else:
        inst_dtypes = tuple(k.output_dtypes)
    for CDtype in inst_dtypes:
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
    ws_ctype, _ws_bytes_elem = _FUSE_WS_CTYPE[ws_dtype]

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


def gen_4wave_co_instance(
    cg,
    k,
    traits_header,
    da,
    db,
    traits_name,
    kargs_name,
    **_unused,
):
    """gfx1250 symmetric 4-wave compute launcher emit -- PRE-COMPILED .co kid.

    This is the one emit that does not produce a device kernel. It differs from
    an ordinary kid in three places:

      * the launcher takes the ordinary 5-arg a16w16 signature, NOT the 6-arg
        workspace-carrying one the gfx1250 split-K kids use: this pipeline has
        no split-K, no partial buffer and no reduce kernel, so there is nothing
        to put in a workspace. It therefore dispatches through its own pair of
        tables (opus_a16w16_co_tune_dispatch_gfx1250 by kid,
        opus_a16w16_co_dispatch_gfx1250 by shape) rather than the arch's shared
        ones. A split-K .co variant would move back to the workspace ABI.

      * it includes the TRAITS header only, never the pipeline header. The
        launcher needs nothing but Traits::kBlockM-style constants, and pulling
        the pipeline in would drag TDM builtins and pin builtins into a release
        compile unit that cannot handle them.
      * it appends NOTHING to cg._device_instantiations, so gen_instances.py's
        `{name}_C{dtype}.device.cu` loop skips this kid entirely. That single
        omission IS the compile bypass; everything downstream (manifest, tune
        lookup, dispatch, tuned CSV) is unchanged.

    The device side is loaded at runtime from gen_co/gfx1250/{k.name}.co, whose
    filename and extern "C" symbol both equal the launcher name, so no sidecar
    is needed to connect them. The whole family -- tile, cluster, VGPR budget,
    launch bounds, device flags -- comes from gen_co/co_kernels.json.
    """
    traits_alias = f"""
// Baked as a plain (non-template) alias rather than being parameterised on the
// launcher's D_C, so it can never be instantiated with a configuration the .co
// was not built for.
using {k.name}_Traits = {traits_name}<{co_traits_args(k)}>;
"""

    INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See codegen/gen_instances_gfx1250.py.
//
// Pre-compiled (.co) kid: no device TU is emitted for this launcher. The kernel
// image is built offline by csrc/opus_gemm/gen_co/build_co.py.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
// Traits only -- see the emit docstring. There is deliberately no
// OPUS_FUSED_HOST_TU branch and no pipeline include: this header is identical
// on every pass.
#include "{traits_header}"
{traits_alias}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "gfx1250/opus_co_launch_gfx1250.cuh"

template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK)
{{
    static_assert(std::is_same<D_C, bf16_t>::value,
        "the 4wave_co kernel writes bf16 C directly -- there is no reduce "
        "kernel and no fp32 workspace slot to dispatch through");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    using Traits = {k.name}_Traits;

    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
        "4wave_co writes bf16 C directly (no reduce kernel to cast): Y must be "
        "bf16, got ", AiterDtype_to_str(Y.dtype()));
    AITER_CHECK(!bias.has_value(), "4wave_co does not support bias");
    AITER_CHECK(splitK <= 1,
        "4wave_co has no split-K (got splitK=", splitK, ")");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1 && batch >= 1,
        "M, N, K, batch must be >= 1");
    // No M/N/K alignment guard, and deliberately so. Every tail is handled by
    // the TDM descriptor: tdm::make_descriptor() clamps EVERY dimension with
    // saturating_sub(extent, origin), and the C store window's fastest axis IS
    // N, so the hardware writes min(B_N, n - col) columns of each row. Ragged
    // M and ragged K ride the same mechanism.
    //
    // (An earlier revision asserted N % B_N == 0 on the theory that the
    // epilogue bounded the store by bytes-remaining-in-the-matrix and an N tail
    // would spill into the next row. That is make_gmem's num_records semantics,
    // not the TDM's -- this pipeline builds no gmem descriptor at all.)

    {kargs_name} kargs{{}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.m = M; kargs.n = N; kargs.k = K;
    kargs.stride_a       = XQ.stride(1);
    kargs.stride_b       = WQ.stride(1);
    kargs.stride_c       = N;
    // 64-bit: a batch stride is an ELEMENT count over a whole matrix, so it
    // passes 2^31 at 4 GiB of bf16 and the kernel offsets by it per batch.
    kargs.stride_a_batch = XQ.stride(0);
    kargs.stride_b_batch = WQ.stride(0);
    // kargs carries no batch count (grid.z is it) and no C batch stride (the
    // kernel derives m * stride_c), which is what keeps the struct at 64 B.

    // Round the tile counts up to whole clusters: the runtime rejects a cluster
    // launch whose grid is not a multiple of the cluster dims, so a half-full
    // cluster does not exist and the surplus workgroups WILL be dispatched. The
    // kernel absorbs them -- they fail the tile bound check and leave right
    // after the one cluster-scope barrier they owe their peers, having touched
    // neither LDS nor the TDM. So any (M, N) is launchable with any cluster
    // dims, and there is deliberately no divisibility AITER_CHECK here.
    int grid_m = (M + Traits::kBlockM - 1) / Traits::kBlockM;
    int grid_n = (N + Traits::kBlockN - 1) / Traits::kBlockN;
    grid_m = (grid_m + Traits::kClusterWgM - 1) / Traits::kClusterWgM * Traits::kClusterWgM;
    grid_n = (grid_n + Traits::kClusterWgN - 1) / Traits::kClusterWgN * Traits::kClusterWgN;

    // Resolved (file read + module registration) on first call only: the symbol
    // is this launcher's own name, so one static per launcher is exact.
    static AiterAsmKernelFast& kernel = opus_gfx1250_co::co_kernel("{k.name}");

    // One launch covers the whole batch (grid.z), unlike the ws pipeline's host
    // batch loop: the kernel offsets A/B/C by workgroup_id_z * stride_*_batch.
    opus_co_launch_gfx1250<Traits>(
        kernel,
        dim3(grid_m, grid_n, batch),
        dim3(Traits::BLOCK_SIZE),
        kargs,
        aiter::getCurrentHIPStream());
}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    # Host launcher only. NOTHING is appended to cg._device_instantiations --
    # that omission is the whole bypass (see the docstring).
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
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
        )


# ---------- Self-register at import time ----------
register_emit("gfx1250", "a16w16_4wave_co", gen_4wave_co_instance)
register_emit("gfx1250", "a16w16_4wave_wl_co", gen_4wave_co_instance)
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
