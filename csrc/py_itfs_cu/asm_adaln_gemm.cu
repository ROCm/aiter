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
#include <vector>

// Skinny bf16 GEMMs of a DiT's AdaLN modulation linear, one micro-batch of 32 rows (hsa/gfx950/adalngemm):
//   pass 0  forward  y[32, N]  = x[32, K] @ W[N, K]^T + b
//   pass 1  dgrad    dx[32, K] = dy[32, N] @ W[N, K]
//           (both deterministic; they get an fp32 workspace and int32 flags / counters sized by the row, which start
//            at zero and are left at zero)
//   pass 2  wgrad    dW[N, K]  = dy[32, N]^T @ x[32, K]      (bitwise equal to a single-K-step GEMM: K = 32)
// Rows are exact (pass, N, K). A row's `abi` says how its kernel takes its arguments:
//   srd      raw buffer resources (V#) over each tensor, then N
//   tensile  a Tensile (hipBLASLt) GEMM: the row's `karg` is the kernel-argument block for its shape with every
//            pointer slot zero, filled here. A dgrad row writes fp32 partial sums of its K splits into the workspace
//            and a second kernel (`knl2`, `gx2` workgroups, arguments `karg2`) reduces them into dx.
namespace {

// Tensile argument blocks: pointer slots (byte offsets)
constexpr size_t kGemmD = 32, kGemmC = 40, kGemmA = 48, kGemmB = 56, kGemmWs = 64, kGemmFlags = 72, kGemmBias = 152;
constexpr size_t kReduceD = 0, kReduceWs = 8, kReduceC = 16;

std::vector<unsigned char> unhex(const std::string& s)
{
    // "x" + hex digits (the prefix keeps codegen from reading the column as a number)
    std::vector<unsigned char> out;
    for(size_t i = 1; i + 1 < s.size(); i += 2)
        out.push_back(static_cast<unsigned char>(std::stoul(s.substr(i, 2), nullptr, 16)));
    return out;
}

// the parsed templates of a row (rows are static): (GEMM, reduce)
const std::pair<std::vector<unsigned char>, std::vector<unsigned char>>& templates(const adalngemmConfig* cfg)
{
    static std::mutex mu;
    static std::unordered_map<const adalngemmConfig*,
                              std::pair<std::vector<unsigned char>, std::vector<unsigned char>>>
        cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(cfg);
    if(it == cache.end())
        it = cache.emplace(cfg, std::make_pair(unhex(cfg->karg), cfg->op == 1 ? unhex(cfg->karg2)
                                                                              : std::vector<unsigned char>{}))
                 .first;
    return it->second;
}

void put_ptr(std::vector<unsigned char>& args, size_t off, const void* p)
{
    const uint64_t v = reinterpret_cast<uint64_t>(p);
    std::memcpy(args.data() + off, &v, sizeof(v));
}

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
     aiter_tensor_t* ws,   // pass 0 / 1: fp32 workspace (manifest `ws` elements); pass 2: nullptr
     aiter_tensor_t* cnt,  // pass 0 / 1: int32 counters (manifest `cnt` elements, zero); pass 2: nullptr
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
                        bias->dtype() == AITER_DTYPE_bf16 && bias->numel() == N && ws != nullptr && cnt != nullptr &&
                        ws->dtype() == AITER_DTYPE_fp32 && cnt->dtype() == AITER_DTYPE_i32,
                    __func__,
                    " forward shapes: x [32, K], W [N, K], bias [N], y [32, N], fp32 workspace, int32 counters");
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
    if(pass != 2)
        AITER_CHECK(ws->numel() >= cfg->ws && cnt->numel() >= cfg->cnt,
                    __func__,
                    " workspace / counters smaller than the kernel needs");

    const HipDeviceGuard device_guard(a->device_id);
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    auto kernel = [&](const std::string& knl, const std::string& co) {
        return &impl_ptr_map.get_or_create(knl, [&]() { return AiterAsmKernel(knl.c_str(), co.c_str()); });
    };
    if(cfg->abi == "tensile")
    {
        AITER_CHECK(pass != 2, __func__, " the tensile ABI is for the forward and dgrad rows");
        const auto& tpl = templates(cfg);
        std::vector<unsigned char> args = tpl.first;
        size_t arg_size = args.size();
        AITER_CHECK(arg_size > kGemmBias + 8, __func__, " malformed kernel-argument template");
        if(pass == 0)
        {
            put_ptr(args, kGemmD, out->ptr), put_ptr(args, kGemmC, out->ptr);
            put_ptr(args, kGemmA, b->ptr), put_ptr(args, kGemmB, a->ptr);
            put_ptr(args, kGemmWs, ws->ptr), put_ptr(args, kGemmFlags, cnt->ptr);
            put_ptr(args, kGemmBias, bias->ptr);
        }
        else
        {
            put_ptr(args, kGemmD, ws->ptr), put_ptr(args, kGemmC, ws->ptr);
            put_ptr(args, kGemmA, b->ptr), put_ptr(args, kGemmB, a->ptr);
            put_ptr(args, kGemmWs, ws->ptr);
        }
        kernel(cfg->knl_name, cfg->co_name)
            ->launch_kernel({args.data(), &arg_size, cfg->gx, cfg->gy, 1, cfg->block, 1, 1, stream});
        if(pass == 1)
        {
            std::vector<unsigned char> args2 = tpl.second;
            size_t arg2_size = args2.size();
            AITER_CHECK(arg2_size >= kReduceC + 8, __func__, " malformed reduce-argument template");
            put_ptr(args2, kReduceD, out->ptr), put_ptr(args2, kReduceWs, ws->ptr), put_ptr(args2, kReduceC, out->ptr);
            // co2 sits next to the row's own code object (codegen prefixes the directory to co_name only)
            const std::string dir = cfg->co_name.substr(0, cfg->co_name.find_last_of('/') + 1);
            kernel(cfg->knl2, dir + cfg->co2)
                ->launch_kernel({args2.data(), &arg2_size, cfg->gx2, 1, 1, cfg->block, 1, 1, stream});
        }
        return;
    }

    unsigned char args[6 * sizeof(Srd) + 4];
    size_t arg_size = 0;
    auto put = [&](const void* p, size_t n) {
        std::memcpy(args + arg_size, p, n);
        arg_size += n;
    };
    const Srd s_a = srd(a), s_b = srd(b), s_out = srd(out);
    if(pass == 0)
    {
        const Srd s_bias = srd(bias), s_ws = srd(ws), s_cnt = srd(cnt);
        put(&s_a, 16), put(&s_b, 16), put(&s_bias, 16), put(&s_out, 16), put(&s_ws, 16), put(&s_cnt, 16);
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

    kernel(cfg->knl_name, cfg->co_name)->launch_kernel({args, &arg_size, cfg->gx, cfg->gy, 1, cfg->block, 1, 1, stream});
}
