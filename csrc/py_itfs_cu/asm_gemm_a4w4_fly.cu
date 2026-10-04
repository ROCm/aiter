// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

// Must follow aiter_tensor.h, which provides the HIP error-bridge dependencies.
#include "aiter_ctypes_error.h"
#include "asm_f4flygemm_configs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <string>

// A4W4 on assembly ports of FlyDSL's MXFP4 GEMM, 256-wide N tile only (the 192-wide tile variant is
// not supported). One code object per (M, N, K): K, and M/N where FlyDSL folds the tile decode, are
// compile-time, so the launcher selects by exact shape and launches the grid FlyDSL launches
// (manifest `grid`; tiles per workgroup is tiles / grid). Operands: plain MXFP4 rows [rows, K/2]
// and E8M0 scales in FlyDSL's prepacked layout (256-wide N tile, B interleave `b_ilv`).
namespace {
constexpr size_t kTensorAlignment = 16;

// Kernarg layout of FlyDSL's MXFP4 GEMM (160 B): memref-style (ptr, {rows, row bytes, row stride,
// 0}) for A, B and C (C three times), the prepacked scale slabs with their dword counts, then M and
// N.
struct __attribute__((packed)) KernelArgs
{
    void* ptr_A;
    uint32_t a_rows, a_row_bytes, a_stride, a_pad;
    void* ptr_B;
    uint32_t b_rows, b_row_bytes, b_stride, b_pad;
    void* ptr_C0;
    uint32_t c0_rows, c0_cols, c0_stride, c0_pad;
    void* ptr_SA;
    uint32_t sa_dwords, sa_pad;
    void* ptr_SB;
    uint32_t sb_dwords;
    uint32_t M;
    uint32_t N;
    uint32_t mn_pad;
    void* ptr_C1;
    uint32_t c1_rows, c1_cols, c1_stride, c1_pad;
    void* ptr_C2;
    uint32_t c2_rows, c2_cols, c2_stride, c2_pad;
};
static_assert(sizeof(KernelArgs) == 160, "a4w4_fly KernelArgs must be 160 bytes");
static_assert(offsetof(KernelArgs, ptr_B) == 24, "a4w4_fly ptr_B ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_C0) == 48, "a4w4_fly ptr_C ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_SA) == 72, "a4w4_fly ptr_SA ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_SB) == 88, "a4w4_fly ptr_SB ABI mismatch");
static_assert(offsetof(KernelArgs, M) == 100, "a4w4_fly M ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_C1) == 112, "a4w4_fly second C ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_C2) == 136, "a4w4_fly third C ABI mismatch");
} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_a4w4_fly_asm,
    (aiter_tensor_t * A,      // A: MXFP4 rows [M, K/2] uint8
     aiter_tensor_t* B,       // B: MXFP4 rows [N, K/2] uint8
     aiter_tensor_t* A_scale, // A_scale: prepacked E8M0, M*K/32 bytes
     aiter_tensor_t*
         B_scale, // B_scale: prepacked E8M0, N*K/32 bytes (256-wide N tile, b_ilv interleave)
     aiter_tensor_t* out, // Out: [M, N] bf16 contiguous
     int64_t K,
     hipStream_t stream),
    (A, B, A_scale, B_scale, out, K, stream))
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
    const int64_t M = out->size(0), N = out->size(1);
    AITER_CHECK(K > 0 && K % 256 == 0 && M % 256 == 0 && N % 256 == 0,
                __func__,
                " M, N, K must be multiples of 256");
    const auto bytes = [](const aiter_tensor_t* t) {
        return static_cast<int64_t>(t->numel()) * t->element_size();
    };
    AITER_CHECK(bytes(A) == M * K / 2 && bytes(B) == N * K / 2,
                __func__,
                " A/B must be [rows, K/2] MXFP4 rows");
    AITER_CHECK(bytes(A_scale) == M * K / 32 && bytes(B_scale) == N * K / 32,
                __func__,
                " prepacked scales must be rows*K/32 bytes");
    const auto aligned = [](const aiter_tensor_t* t) {
        return reinterpret_cast<uintptr_t>(t->ptr) % kTensorAlignment == 0;
    };
    AITER_CHECK(aligned(A) && aligned(B) && aligned(A_scale) && aligned(B_scale) && aligned(out),
                __func__,
                " all tensor addresses must be 16-byte aligned");

    KernelArgs a;
    std::memset(&a, 0, sizeof(a));
    const uint32_t m = M, n = N, kb = K / 2;
    a.ptr_A         = A->ptr;
    a.a_rows        = m;
    a.a_row_bytes   = kb;
    a.a_stride      = kb;
    a.ptr_B         = B->ptr;
    a.b_rows        = n;
    a.b_row_bytes   = kb;
    a.b_stride      = kb;
    a.ptr_C0        = out->ptr;
    a.c0_rows       = m;
    a.c0_cols       = n;
    a.c0_stride     = n;
    a.ptr_SA        = A_scale->ptr;
    a.sa_dwords     = static_cast<uint32_t>(bytes(A_scale) / 4);
    a.ptr_SB        = B_scale->ptr;
    a.sb_dwords     = static_cast<uint32_t>(bytes(B_scale) / 4);
    a.M             = m;
    a.N             = n;
    a.ptr_C1        = out->ptr;
    a.c1_rows       = m;
    a.c1_cols       = n;
    a.c1_stride     = n;
    a.ptr_C2        = out->ptr;
    a.c2_rows       = m;
    a.c2_cols       = n;
    a.c2_stride     = n;
    size_t arg_size = sizeof(a);

    const HipDeviceGuard device_guard(A->device_id);
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const std::string kname = get_gpu_arch() + "aiter_a4w4_fly_" + std::to_string(M) + "x" +
                              std::to_string(N) + "x" + std::to_string(K);
    CFG* cfgs               = &cfg_f4flygemm_bf16_per1x32Fp4Fp4_fly;
    auto it                 = cfgs->find(kname);
    AITER_CHECK(it != cfgs->end(), __func__, " no a4w4_fly kernel for " + kname);
    const auto& cfg = it->second;
    AITER_CHECK(cfg.M == M && cfg.N == N && cfg.K == K && cfg.grid > 0,
                __func__,
                " manifest mismatch for " + kname);
    const char* name = cfg.knl_name.c_str();
    const char* co   = cfg.co_name.c_str();
    AiterAsmKernel* impl =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co); });
    impl->launch_kernel({&a, &arg_size, cfg.grid, 1, 1, 256, 1, 1, stream});
}
