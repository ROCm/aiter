// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

// Must follow aiter_tensor.h, which provides the HIP error-bridge dependencies.
#include "aiter_ctypes_error.h"
#include "asm_adalngemm_configs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <mutex>
#include <string>
#include <unordered_map>

// Skinny bf16 GEMMs of a DiT's AdaLN modulation linear, one micro-batch of 32 rows (hsa/gfx950/adalngemm):
//   pass 0  forward  y[32, N]  = x[32, K] @ W[N, K]^T + b
//   pass 1  dgrad    dx[32, K] = dy[32, N] @ W[N, K]        (one launch; split-K with a deterministic reduce by the
//                                                             last-arriving workgroup of each column tile: needs an
//                                                             fp32 workspace and int32 counters that start at zero
//                                                             and are left at zero)
//   pass 2  wgrad    dW[N, K]  = dy[32, N]^T @ x[32, K]      (bitwise equal to a single-K-step GEMM: K = 32)
// Rows are exact (pass, N, K). Kernel arguments are raw buffer resources (V#) over each tensor and N.
namespace {

struct Srd
{
    uint32_t w[4];
};
static_assert(sizeof(Srd) == 16, "a buffer resource is 16 bytes");

Srd srd(const aiter_tensor_t* t)
{
    const uint64_t base = reinterpret_cast<uint64_t>(t->ptr);
    Srd s;
    s.w[0] = static_cast<uint32_t>(base);
    s.w[1] = static_cast<uint32_t>((base >> 32) & 0xffff); // stride 0
    s.w[2] = static_cast<uint32_t>(static_cast<int64_t>(t->numel()) * t->element_size());
    s.w[3] = 0x00030000; // raw buffer, 32-bit data format
    return s;
}

const adalngemmConfig* find_row(const std::string& arch, int64_t pass, int64_t N, int64_t K)
{
    static std::mutex mu;
    static std::unordered_map<std::string, const adalngemmConfig*> cache;
    const std::string key =
        arch + ":" + std::to_string(pass) + ":" + std::to_string(N) + "x" + std::to_string(K);
    std::lock_guard<std::mutex> lock(mu);
    auto hit = cache.find(key);
    if(hit != cache.end())
        return hit->second;
    const adalngemmConfig* found = nullptr;
    for(const auto& kv : cfg_adalngemm_bf16)
    {
        const auto& c = kv.second;
        if(c.arch == arch && c.op == pass && c.N == N && c.K == K)
        {
            found = &c;
            break;
        }
    }
    cache.emplace(key, found);
    return found;
}
} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    adaln_gemm_asm,
    (int64_t pass,
     aiter_tensor_t* a,    // pass 0: x [32, K]; 1: dy [32, N]; 2: dy [32, N]
     aiter_tensor_t* b,    // pass 0: W [N, K]; 1: W [N, K]; 2: x [32, K]
     aiter_tensor_t* bias, // pass 0: bias [N] (required); otherwise nullptr
     aiter_tensor_t* out,  // pass 0: y [32, N]; 1: dx [32, K]; 2: dW [N, K]
     aiter_tensor_t* ws,   // pass 1: fp32 workspace (manifest `ws` elements); otherwise nullptr
     aiter_tensor_t* cnt,  // pass 1: int32 counters (manifest `cnt` elements, zero); otherwise nullptr
     hipStream_t stream),
    (pass, a, b, bias, out, ws, cnt, stream))
{
    AITER_CHECK(pass >= 0 && pass <= 2, __func__, " pass must be 0 (forward), 1 (dgrad) or 2 (wgrad)");
    AITER_CHECK(a->dtype() == AITER_DTYPE_bf16 && b->dtype() == AITER_DTYPE_bf16 &&
                    out->dtype() == AITER_DTYPE_bf16,
                __func__,
                " operands and output must be BFloat16");
    AITER_CHECK(a->is_contiguous() && b->is_contiguous() && out->is_contiguous() && a->dim() == 2 &&
                    b->dim() == 2 && out->dim() == 2,
                __func__,
                " operands and output must be contiguous 2D tensors");
    AITER_CHECK(a->device_id == b->device_id && a->device_id == out->device_id,
                __func__,
                " all tensors must be on the same GPU");
    AITER_CHECK(a->size(0) == 32, __func__, " the micro-batch (rows of the 32-row operand) must be 32");
    int64_t N = 0, K = 0;
    if(pass == 0)
    {
        K = a->size(1);
        N = b->size(0);
        AITER_CHECK(b->size(1) == K && out->size(0) == 32 && out->size(1) == N && bias != nullptr &&
                        bias->dtype() == AITER_DTYPE_bf16 && bias->numel() == N,
                    __func__,
                    " forward shapes: x [32, K], W [N, K], bias [N], y [32, N]");
    }
    else if(pass == 1)
    {
        N = a->size(1);
        K = b->size(1);
        AITER_CHECK(b->size(0) == N && out->size(0) == 32 && out->size(1) == K && ws != nullptr && cnt != nullptr &&
                        ws->dtype() == AITER_DTYPE_fp32 && cnt->dtype() == AITER_DTYPE_i32,
                    __func__,
                    " dgrad shapes: dy [32, N], W [N, K], dx [32, K], fp32 workspace, int32 counters");
    }
    else
    {
        N = a->size(1);
        K = b->size(1);
        AITER_CHECK(b->size(0) == 32 && out->size(0) == N && out->size(1) == K,
                    __func__,
                    " wgrad shapes: dy [32, N], x [32, K], dW [N, K]");
    }
    const std::string arch = get_gpu_arch();
    const adalngemmConfig* cfg = find_row(arch, pass, N, K);
    AITER_CHECK(cfg != nullptr,
                __func__,
                " no AdaLN GEMM kernel for pass " + std::to_string(pass) + " N " + std::to_string(N) + " K " +
                    std::to_string(K));
    if(pass == 1)
        AITER_CHECK(ws->numel() >= cfg->ws && cnt->numel() >= cfg->cnt,
                    __func__,
                    " dgrad workspace / counters smaller than the kernel needs");

    unsigned char args[6 * sizeof(Srd) + 4];
    size_t arg_size = 0;
    auto put = [&](const void* p, size_t n) {
        std::memcpy(args + arg_size, p, n);
        arg_size += n;
    };
    const Srd s_a = srd(a), s_b = srd(b), s_out = srd(out);
    if(pass == 0)
    {
        const Srd s_bias = srd(bias);
        put(&s_a, 16), put(&s_b, 16), put(&s_bias, 16), put(&s_out, 16);
    }
    else if(pass == 1)
    {
        const Srd s_ws = srd(ws), s_cnt = srd(cnt);
        put(&s_a, 16), put(&s_b, 16), put(&s_out, 16), put(&s_ws, 16), put(&s_cnt, 16);
    }
    else
    {
        put(&s_a, 16), put(&s_b, 16), put(&s_out, 16);
    }
    const int32_t n32 = static_cast<int32_t>(N);
    put(&n32, 4);

    const HipDeviceGuard device_guard(a->device_id);
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name = cfg->knl_name.c_str();
    const char* co   = cfg->co_name.c_str();
    AiterAsmKernel* impl =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co); });
    impl->launch_kernel({args, &arg_size, cfg->gx, cfg->gy, 1, cfg->block, 1, 1, stream});
}
