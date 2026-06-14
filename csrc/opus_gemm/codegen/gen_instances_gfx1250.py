# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 codegen -- emit launchers for gfx1250-targeted kid families.

Wires the a16w16 cluster/TDM split-K pipeline that reduces via an fp32
WORKSPACE + a separate REDUCE kernel (no atomic_add), mirroring the gfx950
flatmm-splitk launcher (opus_splitk_ws_get + grow + main kernel + reduce
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
}

TRAITS_HEADER_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
}

KERNEL_FUNC_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gemm_a16w16_cluster_tdm_splitk_ws_kernel_gfx1250",
}

TRAITS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
}

KARGS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_gemm_cluster_tdm_ws_kargs_gfx1250",
}


def splitk_reduce_extra_device_instantiations():
    # gfx1250 only: fp32 bias with a bf16 output (D_OUT=__bf16, D_BIAS=float).
    # The main kernel always writes an fp32 workspace, so an fp32 bias folds
    # exactly in the reduce before the cast to bf16. The baseline instantiations
    # cover the matched-dtype cases; this adds the bf16-out + fp32-bias mix that
    # other arches never request. Same kernel NAME/ABI -> no extra forward decl.
    return (
        "// fp32-bias + bf16-out (gfx1250 f32 bias support)\n"
        "template __global__ void splitk_reduce_kernel_gfx1250<16, 64, __bf16, true,  float,  true>(\n"
        "    const opus_splitk_ws_handle*, __bf16*, int, int, int, int, int, int,\n"
        "    const float*,  int);\n"
        "template __global__ void splitk_reduce_kernel_gfx1250<16, 64, __bf16, true,  float,  false>(\n"
        "    const opus_splitk_ws_handle*, __bf16*, int, int, int, int, int, int,\n"
        "    const float*,  int);\n"
    )


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

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    {k.B_M}, {k.B_N}, {k.B_K},
    {layout_int},
    {da}, {db}, D_C, fp32_t,
    {enable_bias_str}>;
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
// Forward declaration for the host-side <<<>>> launch stub. Must match the
// kernel's __launch_bounds__.
template<typename Traits>
__global__ __launch_bounds__(128, 1)
void {kernel_func}({kargs_name} kargs);
#else
#include "{pipeline_header}"
#endif
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
// Reduce kernel forward declaration (distinct gfx1250 name + ws_handle ABI so
// it never collides with gfx950's identically-signatured splitk_reduce_kernel).
// The definition lives in gfx1250/splitk_reduce_gfx1250.cuh; the explicit
// instantiations live in the dedicated splitk_reduce_gfx1250.device.cu TU.
template<int VEC_, int BLOCK_, typename D_OUT,
         bool HAS_BIAS_, typename D_BIAS_, bool HAS_OOB_>
__global__ void splitk_reduce_kernel_gfx1250(
    const opus_splitk_ws_handle* ws_handle, D_OUT* c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* bias, int stride_bias_batch);

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

    extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);
    auto stream = aiter::getCurrentHIPStream();
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
    const bool capturing = (capture_status != hipStreamCaptureStatusNone);
    auto* ws_handle_ = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

    // Workspace sized for ONE batch's split slices [split_k, padded_M, padded_N].
    size_t ws_bytes = (size_t)split_k * (size_t)padded_M * (size_t)padded_N * sizeof(float);
    if (ws_handle_->ptr == nullptr || ws_bytes > ws_handle_->bytes)
    {{{{
        AITER_CHECK(!capturing,
            "splitk workspace grow inside HIP graph capture is not supported. "
            "Warm the cache once eagerly via aiter.opus_gemm_workspace_init().");
        void* new_ptr = nullptr;
        const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
        size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
        HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
        if (ws_handle_->ptr != nullptr)
        {{{{
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipFree(ws_handle_->ptr));
        }}}}
        ws_handle_->ptr = new_ptr;
        ws_handle_->bytes = grow_bytes;
    }}}}

    const size_t a_batch_bytes = (size_t)M * (size_t)K * sizeof({da});
    const size_t b_batch_bytes = (size_t)N * (size_t)K * sizeof({db});
    const size_t c_elem_bytes  = (Y.dtype() == AITER_DTYPE_bf16) ? sizeof(bf16_t) : sizeof(fp32_t);
    const size_t c_batch_bytes = (size_t)M * (size_t)N * c_elem_bytes;
    const size_t bias_elem_bytes = bias_is_fp32_ ? sizeof(fp32_t) : c_elem_bytes;

    dim3 grid_main(num_tiles_m, num_tiles_n, split_k);
    dim3 block_main({k.BLOCK_SIZE});

    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS), M, 1);
    dim3 block_reduce(REDUCE_BS);

    for (int bb = 0; bb < batch; ++bb)
    {{{{
        {kargs_name} kargs{{{{}}}};
        kargs.ptr_a     = reinterpret_cast<const char*>(XQ.data_ptr()) + (size_t)bb * a_batch_bytes;
        kargs.ptr_b     = reinterpret_cast<const char*>(WQ.data_ptr()) + (size_t)bb * b_batch_bytes;
        kargs.ws_handle = ws_handle_;
        kargs.ptr_c     = reinterpret_cast<char*>(Y.data_ptr()) + (size_t)bb * c_batch_bytes;
        kargs.ptr_bias  = ptr_bias_;
        kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = 1; kargs.split_k = split_k;
        kargs.stride_a        = K;
        kargs.stride_b        = K;
        kargs.stride_ws       = padded_N;
        kargs.stride_c        = N;
        kargs.stride_a_batch  = M * K;
        kargs.stride_b_batch  = N * K;
        kargs.stride_ws_batch = padded_M * padded_N;
        kargs.stride_c_batch  = M * N;
        kargs.stride_bias_batch = stride_bias_batch_;

        {kernel_func}<Traits><<<grid_main, block_main, 0, stream>>>(kargs);

        // Per-batch bias base (reduce is called with batch=1 -> b==0 inside).
        const void* bias_bb = ptr_bias_
            ? (reinterpret_cast<const char*>(ptr_bias_)
               + (size_t)bb * (size_t)stride_bias_batch_ * bias_elem_bytes)
            : nullptr;

        if (Y.dtype() == AITER_DTYPE_bf16) {{{{
            __bf16* y_ptr = reinterpret_cast<__bf16*>(kargs.ptr_c);
            if (bias_bb && bias_is_fp32_) {{{{
                // fp32 bias + bf16 output: fold the exact fp32 bias in the
                // reduce (D_BIAS=float), then cast the fp32 sum to bf16.
                splitk_reduce_kernel_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, float, {has_oob_str}>
                    <<<grid_reduce, block_reduce, 0, stream>>>(
                        ws_handle_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                        reinterpret_cast<const float*>(bias_bb), 0);
            }}}} else if (bias_bb) {{{{
                splitk_reduce_kernel_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, true, __bf16, {has_oob_str}>
                    <<<grid_reduce, block_reduce, 0, stream>>>(
                        ws_handle_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                        reinterpret_cast<const __bf16*>(bias_bb), 0);
            }}}} else {{{{
                splitk_reduce_kernel_gfx1250<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}>
                    <<<grid_reduce, block_reduce, 0, stream>>>(
                        ws_handle_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
            }}}}
        }}}} else {{{{
            float* y_ptr = reinterpret_cast<float*>(kargs.ptr_c);
            if (bias_bb) {{{{
                splitk_reduce_kernel_gfx1250<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}>
                    <<<grid_reduce, block_reduce, 0, stream>>>(
                        ws_handle_, y_ptr, split_k, M, N, 1, padded_M, padded_N,
                        reinterpret_cast<const float*>(bias_bb), 0);
            }}}} else {{{{
                splitk_reduce_kernel_gfx1250<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}>
                    <<<grid_reduce, block_reduce, 0, stream>>>(
                        ws_handle_, y_ptr, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
            }}}}
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


# ---------- Self-register at import time ----------
register_emit(
    "gfx1250", "a16w16_cluster_tdm_splitk_ws", gen_cluster_tdm_splitk_ws_instance
)
