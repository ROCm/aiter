// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

// Must follow aiter_tensor.h, which provides the HIP error-bridge dependencies.
#include "aiter_ctypes_error.h"
#include "asm_tsgemm_configs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <mutex>
#include <string>
#include <unordered_map>

// MX GEMMs on the tilescale layout (aiter/ops/tilescale.py): out[M, N] = A @ B^T (+ bias), bf16 out,
// one E8M0 scale per 32 K in the tilescale slab. Operand formats per row of the manifest
// (hsa/gfx950/tsgemm): a_fmt / b_fmt 4 (E2M1) or 6 (E2M3); FP6 codes are the K128-blocked C0 / C1
// planes in one buffer; FP4 codes are row-major (b_codes 0) or K128-blocked (b_codes 1); b_ilv is
// role B's scale interleave. Rows are selected by (formats, b_codes, b_ilv, bias, M, N, K); `abi`
// selects the kernarg marshalling:
//   0  the FlyDSL MXFP4 GEMM's 160-byte kernarg (A4W4)
//   1  the FlyDSL MXFP6 GEMM's 156-byte kernarg, 172 with bias (A6W6, and A6W4 with B's C1 slot
//      aliasing B: the kernel never reads it)
namespace {
constexpr size_t kTensorAlignment = 16;

struct Memref
{
    void* ptr;
    uint32_t rows, row_bytes, stride, pad;
};
static_assert(sizeof(Memref) == 24, "memref must be 24 bytes");

struct __attribute__((packed)) KernelArgsFly4
{
    Memref A, B, C0;
    void* ptr_SA;
    uint32_t sa_dwords, sa_pad;
    void* ptr_SB;
    uint32_t sb_dwords;
    uint32_t M;
    uint32_t N;
    uint32_t mn_pad;
    Memref C1, C2;
};
static_assert(sizeof(KernelArgsFly4) == 160, "fly4 kernarg must be 160 bytes");
static_assert(offsetof(KernelArgsFly4, ptr_SA) == 72 && offsetof(KernelArgsFly4, M) == 100 &&
                  offsetof(KernelArgsFly4, C1) == 112,
              "fly4 kernarg ABI mismatch");

struct __attribute__((packed)) KernelArgsFly6
{
    Memref A0, A1, B0, B1, C;
    void* ptr_SA;
    uint32_t sa_dwords, sa_pad;
    void* ptr_SB;
    uint32_t sb_dwords;
    uint32_t M;
    uint32_t N;
    uint32_t mn_pad;
    void* ptr_bias;
    uint32_t bias_elems;
};
static_assert(sizeof(KernelArgsFly6) == 172, "fly6 kernarg must be 172 bytes");
static_assert(offsetof(KernelArgsFly6, ptr_SA) == 120 && offsetof(KernelArgsFly6, M) == 148 &&
                  offsetof(KernelArgsFly6, ptr_bias) == 160,
              "fly6 kernarg ABI mismatch");
constexpr size_t kFly6NoBias = 156;

Memref memref(void* p, int64_t rows, int64_t row_bytes)
{
    return Memref{p,
                  static_cast<uint32_t>(rows),
                  static_cast<uint32_t>(row_bytes),
                  static_cast<uint32_t>(row_bytes),
                  0};
}

int64_t code_bytes(int64_t fmt, int64_t rows, int64_t K)
{
    return fmt == 4 ? rows * K / 2 : rows * K * 3 / 4;
}

const tsgemmConfig* find_row(const std::string& arch,
                             int64_t a_fmt,
                             int64_t b_fmt,
                             int64_t b_codes,
                             int64_t b_ilv,
                             bool bias,
                             int64_t M,
                             int64_t N,
                             int64_t K)
{
    static std::mutex mu;
    static std::unordered_map<std::string, const tsgemmConfig*> cache;
    const std::string key = arch + ":" + std::to_string(a_fmt) + "," + std::to_string(b_fmt) + "," +
                            std::to_string(b_codes) + "," + std::to_string(b_ilv) + "," +
                            std::to_string(bias) + "," + std::to_string(M) + "x" +
                            std::to_string(N) + "x" + std::to_string(K);
    std::lock_guard<std::mutex> lock(mu);
    auto hit = cache.find(key);
    if(hit != cache.end())
        return hit->second;
    const tsgemmConfig* found = nullptr;
    for(const auto& kv : cfg_tsgemm_bf16_per1x32)
    {
        const auto& c = kv.second;
        if(c.arch == arch && c.a_fmt == a_fmt && c.b_fmt == b_fmt && c.b_codes == b_codes &&
           c.b_ilv == b_ilv && c.bias == static_cast<int>(bias) && c.M == M && c.N == N &&
           c.K == K)
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
    gemm_tilescale_asm,
    (aiter_tensor_t * A,      // A: tilescale codes of [M, K] (format a_fmt)
     aiter_tensor_t* B,       // B: tilescale codes of [N, K] (format b_fmt, FP4 layout b_codes)
     aiter_tensor_t* A_scale, // A_scale: tilescale E8M0 slab, role A, M*K/32 bytes
     aiter_tensor_t* B_scale, // B_scale: tilescale E8M0 slab, role B (interleave b_ilv), N*K/32 bytes
     aiter_tensor_t* out,     // out: [M, N] bf16 contiguous
     int64_t K,
     aiter_tensor_t* bias, // bias: bf16 [N] contiguous, or nullptr
     int64_t a_fmt,
     int64_t b_fmt,
     int64_t b_codes,
     int64_t b_ilv,
     hipStream_t stream),
    (A, B, A_scale, B_scale, out, K, bias, a_fmt, b_fmt, b_codes, b_ilv, stream))
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, __func__, " only BFloat16 output");
    AITER_CHECK(out->dim() == 2 && out->is_contiguous(),
                __func__,
                " output must be a contiguous 2D tensor");
    AITER_CHECK(A->is_contiguous() && B->is_contiguous() && A_scale->is_contiguous() &&
                    B_scale->is_contiguous(),
                __func__,
                " operands and scales must be contiguous");
    AITER_CHECK(A->device_id == B->device_id && A->device_id == A_scale->device_id &&
                    A->device_id == B_scale->device_id && A->device_id == out->device_id,
                __func__,
                " all tensors must be on the same GPU");
    AITER_CHECK((a_fmt == 4 || a_fmt == 6) && (b_fmt == 4 || b_fmt == 6),
                __func__,
                " formats must be 4 (E2M1) or 6 (E2M3)");
    const int64_t M = out->size(0), N = out->size(1);
    AITER_CHECK(K > 0 && K % 256 == 0 && M % 256 == 0 && N % 256 == 0,
                __func__,
                " M, N, K must be multiples of 256");
    const auto bytes = [](const aiter_tensor_t* t) {
        return static_cast<int64_t>(t->numel()) * t->element_size();
    };
    AITER_CHECK(bytes(A) == code_bytes(a_fmt, M, K) && bytes(B) == code_bytes(b_fmt, N, K),
                __func__,
                " A/B byte sizes do not match their formats");
    AITER_CHECK(bytes(A_scale) == M * K / 32 && bytes(B_scale) == N * K / 32,
                __func__,
                " tilescale scale slabs must be rows*K/32 bytes");
    const auto aligned = [](const aiter_tensor_t* t) {
        return reinterpret_cast<uintptr_t>(t->ptr) % kTensorAlignment == 0;
    };
    AITER_CHECK(aligned(A) && aligned(B) && aligned(A_scale) && aligned(B_scale) && aligned(out),
                __func__,
                " all tensor addresses must be 16-byte aligned");
    if(bias != nullptr)
        AITER_CHECK(bias->dtype() == AITER_DTYPE_bf16 && bias->is_contiguous() &&
                        bias->numel() == N && bias->device_id == out->device_id && aligned(bias),
                    __func__,
                    " bias must be a contiguous, 16-byte aligned bf16 [N] on the output's GPU");

    const std::string arch = get_gpu_arch();
    const tsgemmConfig* cfg =
        find_row(arch, a_fmt, b_fmt, b_codes, b_ilv, bias != nullptr, M, N, K);
    AITER_CHECK(cfg != nullptr && cfg->grid > 0,
                __func__,
                " no tilescale kernel for a" + std::to_string(a_fmt) + "w" +
                    std::to_string(b_fmt) + " " + std::to_string(M) + "x" + std::to_string(N) +
                    "x" + std::to_string(K) + (bias ? " with bias" : ""));

    char* pa = static_cast<char*>(A->ptr);
    char* pb = static_cast<char*>(B->ptr);
    KernelArgsFly4 a4;
    KernelArgsFly6 a6;
    void* args      = nullptr;
    size_t arg_size = 0;
    if(cfg->abi == 0)
    {
        AITER_CHECK(a_fmt == 4 && b_fmt == 4 && bias == nullptr,
                    __func__,
                    " abi 0 rows are A4W4 without bias");
        std::memset(&a4, 0, sizeof(a4));
        a4.A         = memref(pa, M, K / 2);
        a4.B         = memref(pb, N, K / 2);
        a4.C0        = Memref{out->ptr,
                       static_cast<uint32_t>(M),
                       static_cast<uint32_t>(N),
                       static_cast<uint32_t>(N),
                       0};
        a4.C1        = a4.C0;
        a4.C2        = a4.C0;
        a4.ptr_SA    = A_scale->ptr;
        a4.sa_dwords = static_cast<uint32_t>(bytes(A_scale) / 4);
        a4.ptr_SB    = B_scale->ptr;
        a4.sb_dwords = static_cast<uint32_t>(bytes(B_scale) / 4);
        a4.M         = static_cast<uint32_t>(M);
        a4.N         = static_cast<uint32_t>(N);
        args         = &a4;
        arg_size     = sizeof(a4);
    }
    else
    {
        AITER_CHECK(cfg->abi == 1 && a_fmt == 6, __func__, " unknown abi / format pairing");
        std::memset(&a6, 0, sizeof(a6));
        a6.A0 = memref(pa, M, K / 2);
        a6.A1 = memref(pa + M * K / 2, M, K / 4);
        if(b_fmt == 6)
        {
            a6.B0 = memref(pb, N, K / 2);
            a6.B1 = memref(pb + N * K / 2, N, K / 4);
        }
        else
        {
            a6.B0 = memref(pb, N, K / 2);
            a6.B1 = a6.B0; // no C1 plane for an FP4 operand; never read
        }
        a6.C         = Memref{out->ptr,
                      static_cast<uint32_t>(M),
                      static_cast<uint32_t>(N),
                      static_cast<uint32_t>(N),
                      0};
        a6.ptr_SA    = A_scale->ptr;
        a6.sa_dwords = static_cast<uint32_t>(bytes(A_scale) / 4);
        a6.ptr_SB    = B_scale->ptr;
        a6.sb_dwords = static_cast<uint32_t>(bytes(B_scale) / 4);
        a6.M         = static_cast<uint32_t>(M);
        a6.N         = static_cast<uint32_t>(N);
        arg_size     = kFly6NoBias;
        if(bias != nullptr)
        {
            a6.ptr_bias   = bias->ptr;
            a6.bias_elems = static_cast<uint32_t>(N);
            arg_size      = sizeof(a6);
        }
        args = &a6;
    }

    const HipDeviceGuard device_guard(A->device_id);
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name = cfg->knl_name.c_str();
    const char* co   = cfg->co_name.c_str();
    AiterAsmKernel* impl =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co); });
    impl->launch_kernel({args, &arg_size, cfg->grid, 1, 1, 256, 1, 1, stream});
}
