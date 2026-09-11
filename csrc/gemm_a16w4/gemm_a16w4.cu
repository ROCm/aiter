// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host entry point and dispatch for the a16w4 GEMM. The kernels themselves
// live in one translation unit each -- see gemm_a16w4_launch.h for why.

#include "gemm_a16w4.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "gemm_a16w4_launch.h"

#include <array>
#include <string>

namespace {

// ── Runtime architecture gate ──────────────────────────────────────────────
// The device code is #if'd out on non-gfx1201 targets (see
// gfx1201/gemm_a16w4_common_gfx1201.cuh), so on a mixed-arch build the
// kernels exist but trap. This check turns that into a readable error before
// anything is launched.
//
// Cached per device ordinal: hipGetDeviceProperties is far too slow to call
// on the decode path, which is ~30 us end to end.
constexpr int kMaxCachedDevices = 16;

bool device_is_gfx1201()
{
    static std::array<int8_t, kMaxCachedDevices> cache = [] {
        std::array<int8_t, kMaxCachedDevices> a{};
        a.fill(-1);
        return a;
    }();

    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess)
        return false;
    if(dev < 0 || dev >= kMaxCachedDevices)
        return false; // uncached ordinal: refuse rather than guess

    if(cache[dev] < 0)
    {
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
            return false;
        // gcnArchName looks like "gfx1201:sramecc+:xnack-"; match the prefix.
        cache[dev] = std::string(prop.gcnArchName).rfind("gfx1201", 0) == 0 ? 1 : 0;
    }
    return cache[dev] == 1;
}

enum class Path
{
    None,
    Prefill,
    Decode
};

// Smallest M at which the prefill tile is worth using.
//
// Prefill's grid is (M/128, N/512). At M=128 that is grid.x == 1, so on
// N=5120 it launches 10 workgroups against 32 WGPs and leaves most of the
// part idle -- and it costs the SAME as M=256, which is the tell. Decode's
// grid is (ceil(M/16), N/128, SPLIT_K), which at M=128 is 640 workgroups.
//
// MEASURED, gfx1201, fp16, N=K=5120, preallocated buffers:
//
//     M     prefill    decode     winner
//     128   171.9 us   130.9 us   decode   (1.31x)
//     256   172.7 us   267.9 us   prefill  (1.55x)
//     512   305.6 us   554.1 us   prefill
//
// Decode's cost grows linearly in M (BM=16), prefill's is flat until it
// saturates, so the crossover is sharp and sits between 128 and 256. Only
// multiples of 128 ever reach prefill, so this threshold moves exactly one
// bucket -- M=128 -- and leaves every other shape on the path it was
// measured on. Re-measure if the prefill tile or BN changes.
constexpr int64_t kPrefillMinM = 256;

// Decode is tried first only below kPrefillMinM. Above it, prefill wins and
// its constraints are strictly tighter (M % 128 == 0), so a shape it accepts
// is a shape it was tuned for.
Path select_path(int64_t M, int64_t N, int64_t K, bool is_fp16, const char** why)
{
    const int m = (int)M, n = (int)N, k = (int)K;
    const char* prefill_why = nullptr;
    const char* decode_why  = nullptr;

    const bool prefill_ok = is_fp16 ? aiter::a16w4::prefill_fp16_supported(m, n, k, &prefill_why)
                                    : aiter::a16w4::prefill_bf16_supported(m, n, k, &prefill_why);
    const bool decode_ok = is_fp16 ? aiter::a16w4::decode_fp16_supported(m, n, k, &decode_why)
                                   : aiter::a16w4::decode_bf16_supported(m, n, k, &decode_why);

    if(prefill_ok && (M >= kPrefillMinM || !decode_ok))
        return Path::Prefill;
    if(decode_ok)
        return Path::Decode;

    if(why)
        *why = prefill_ok ? prefill_why : (decode_why ? decode_why : "unsupported shape");
    return Path::None;
}

void check_2d(const aiter_tensor_t& t,
              const char* name,
              AiterDtype dtype,
              int64_t d0,
              int64_t d1)
{
    AITER_CHECK(t.is_gpu(), "`", name, "` must be a CUDA/HIP tensor.");
    AITER_CHECK(t.is_contiguous(), "`", name, "` must be contiguous.");
    AITER_CHECK(t.dtype() == dtype,
                "`",
                name,
                "` must be ",
                AiterDtype_to_str(dtype),
                ", got ",
                AiterDtype_to_str(t.dtype()));
    AITER_CHECK(t.dim() == 2, "`", name, "` must be 2-D, got ", t.dim(), "-D.");
    AITER_CHECK(t.size(0) == d0 && t.size(1) == d1,
                "`",
                name,
                "` must be [",
                d0,
                ", ",
                d1,
                "], got [",
                t.size(0),
                ", ",
                t.size(1),
                "].");
}

} // namespace

namespace aiter {

std::string gemm_a16w4_unsupported_reason(int64_t M, int64_t N, int64_t K, bool is_fp16)
{
    const char* why = "unknown";
    if(select_path(M, N, K, is_fp16, &why) != Path::None)
        return "";
    return why ? std::string(why) : std::string("unknown");
}

int64_t gemm_a16w4_workspace_elems(int64_t M, int64_t N, int64_t K, bool is_fp16)
{
    switch(select_path(M, N, K, is_fp16, nullptr))
    {
    case Path::Decode:
        return is_fp16 ? a16w4::decode_fp16_workspace_elems(M, N)
                       : a16w4::decode_bf16_workspace_elems(M, N);
    case Path::Prefill:
    case Path::None:
    default: return 0;
    }
}

void gemm_a16w4(aiter_tensor_t x,
                aiter_tensor_t weight,
                aiter_tensor_t scales,
                aiter_tensor_t zeros,
                aiter_tensor_t out,
                aiter_tensor_t workspace)
{
    AITER_CHECK(device_is_gfx1201(),
                "gemm_a16w4 is gfx1201 (Navi 48) only: it uses the RDNA4 "
                "wmma_*_w32_gfx12 builtins and its tiles are tuned for that "
                "part. There is no fallback path.");

    AITER_CHECK(x.is_gpu(), "`x` must be a CUDA/HIP tensor.");
    AITER_CHECK(x.is_contiguous(), "`x` must be contiguous.");
    AITER_CHECK(x.dim() == 2, "`x` must be 2-D [M, K], got ", x.dim(), "-D.");
    const AiterDtype dtype = x.dtype();
    AITER_CHECK(dtype == AITER_DTYPE_bf16 || dtype == AITER_DTYPE_fp16,
                "`x` must be bf16 or fp16, got ",
                AiterDtype_to_str(dtype));
    const bool is_fp16 = dtype == AITER_DTYPE_fp16;

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    AITER_CHECK(weight.dim() == 2, "`weight` must be 2-D [K/8, N].");
    const int64_t N = weight.size(1);

    const int64_t ngroups = K / a16w4::kGroupSize;
    check_2d(weight, "weight", AITER_DTYPE_i32, K / a16w4::kPackK, N);
    check_2d(scales, "scales", dtype, ngroups, N);
    check_2d(zeros, "zeros", dtype, ngroups, N);
    check_2d(out, "out", dtype, M, N);

    const char* why      = "unknown";
    const Path path      = select_path(M, N, K, is_fp16, &why);
    AITER_CHECK(path != Path::None,
                "gemm_a16w4 does not support M=",
                M,
                " N=",
                N,
                " K=",
                K,
                " (",
                AiterDtype_to_str(dtype),
                "): ",
                why);

    const hipStream_t stream = getCurrentHIPStream();

    if(path == Path::Prefill)
    {
        if(is_fp16)
            a16w4::launch_prefill_fp16(x.data_ptr(),
                                       weight.data_ptr(),
                                       scales.data_ptr(),
                                       zeros.data_ptr(),
                                       out.data_ptr(),
                                       (int)M,
                                       (int)N,
                                       (int)K,
                                       stream);
        else
            a16w4::launch_prefill_bf16(x.data_ptr(),
                                       weight.data_ptr(),
                                       scales.data_ptr(),
                                       zeros.data_ptr(),
                                       out.data_ptr(),
                                       (int)M,
                                       (int)N,
                                       (int)K,
                                       stream);
        return;
    }

    // Decode: split-K writes SPLIT_K fp32 partials, then a second kernel sums
    // them and narrows. The caller owns the scratch so the op signature stays
    // allocation-free -- and so a mismatched allocation fails here rather than
    // corrupting memory.
    const int64_t need = is_fp16 ? a16w4::decode_fp16_workspace_elems(M, N)
                                 : a16w4::decode_bf16_workspace_elems(M, N);
    AITER_CHECK(workspace.is_gpu(), "`workspace` must be a CUDA/HIP tensor.");
    AITER_CHECK(workspace.is_contiguous(), "`workspace` must be contiguous.");
    AITER_CHECK(workspace.dtype() == AITER_DTYPE_fp32,
                "`workspace` must be fp32, got ",
                AiterDtype_to_str(workspace.dtype()));
    AITER_CHECK((int64_t)workspace.numel() >= need,
                "`workspace` needs at least ",
                need,
                " fp32 elements for M=",
                M,
                " N=",
                N,
                ", got ",
                workspace.numel(),
                ". Use aiter.gemm_a16w4_workspace_elems().");

    float* ws = reinterpret_cast<float*>(workspace.data_ptr());
    if(is_fp16)
        a16w4::launch_decode_fp16(x.data_ptr(),
                                  weight.data_ptr(),
                                  scales.data_ptr(),
                                  zeros.data_ptr(),
                                  ws,
                                  out.data_ptr(),
                                  (int)M,
                                  (int)N,
                                  (int)K,
                                  stream);
    else
        a16w4::launch_decode_bf16(x.data_ptr(),
                                  weight.data_ptr(),
                                  scales.data_ptr(),
                                  zeros.data_ptr(),
                                  ws,
                                  out.data_ptr(),
                                  (int)M,
                                  (int)N,
                                  (int)K,
                                  stream);
}

} // namespace aiter
