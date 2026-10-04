// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

// Must follow aiter_tensor.h, which provides the HIP error-bridge dependencies.
#include "aiter_ctypes_error.h"
#include "asm_f6flygemm_configs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <string>

// A6W6 (E2M3 x E2M3) on assembly ports of FlyDSL's MXFP6 GEMM: 256x256 tiles, tiles / grid per
// workgroup. One code object per (M, N, K): shape and tiles per workgroup are compile-time,
// so the launcher selects by exact shape and launches the manifest's grid. Operands: MXFP6 in
// K128-blocked planes, one buffer per operand -- C0 [rows/16, K/128, 16, 64] (bytes 0-15 of each
// packed 24-byte group of 32 codes) then C1 [rows/32, K/128, 32, 32] (bytes 16-23) -- and E8M0
// scales in FlyDSL's prepacked MXFP4 layout at the 256 tile with no B interleave (rows*K/32 bytes).
// Optional bf16 bias[N], added in the store epilogue as fp32(acc) + fp32(bias) with one RNE to bf16
// (A6W6's bias epilogue, bit for bit): a separate code object per shape (manifest `bias`).
namespace {
constexpr size_t kTensorAlignment = 16;

// Kernarg (156 B, from the FlyDSL kernel's metadata, values checked against FlyDSL's own launch):
// memref-style (ptr, {rows, row bytes, row stride, 0}) for the C0 and C1 planes of A and B and for
// C, the prepacked scale slabs with their dword counts, then M and N; the bias variant (172 B)
// appends the bias vector and its element count.
struct __attribute__((packed)) KernelArgs
{
    void* ptr_A0;
    uint32_t a0_rows, a0_row_bytes, a0_stride, a0_pad;
    void* ptr_A1;
    uint32_t a1_rows, a1_row_bytes, a1_stride, a1_pad;
    void* ptr_B0;
    uint32_t b0_rows, b0_row_bytes, b0_stride, b0_pad;
    void* ptr_B1;
    uint32_t b1_rows, b1_row_bytes, b1_stride, b1_pad;
    void* ptr_C;
    uint32_t c_rows, c_cols, c_stride, c_pad;
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
static_assert(sizeof(KernelArgs) == 172, "a6w6_fly KernelArgs must be 172 bytes");
static_assert(offsetof(KernelArgs, ptr_A1) == 24, "a6w6_fly ptr_A1 ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_B0) == 48, "a6w6_fly ptr_B0 ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_B1) == 72, "a6w6_fly ptr_B1 ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_C) == 96, "a6w6_fly ptr_C ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_SA) == 120, "a6w6_fly ptr_SA ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_SB) == 136, "a6w6_fly ptr_SB ABI mismatch");
static_assert(offsetof(KernelArgs, M) == 148, "a6w6_fly M ABI mismatch");
static_assert(offsetof(KernelArgs, ptr_bias) == 160, "a6w6_fly bias ABI mismatch");
constexpr size_t kArgsNoBias = 156;
} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_a6w6_fly_asm,
    (aiter_tensor_t * A,      // A: MXFP6 K128-blocked C0 then C1 planes, M*K*3/4 bytes
     aiter_tensor_t* B,       // B: same for [N, K], N*K*3/4 bytes
     aiter_tensor_t* A_scale, // A_scale: prepacked E8M0, M*K/32 bytes
     aiter_tensor_t*
         B_scale,         // B_scale: prepacked E8M0, N*K/32 bytes (256-wide N tile, no interleave)
     aiter_tensor_t* out, // Out: [M, N] bf16 contiguous
     int64_t K,
     aiter_tensor_t* bias, // bias: bf16 [N] contiguous, or nullptr
     hipStream_t stream),
    (A, B, A_scale, B_scale, out, K, bias, stream))
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
    AITER_CHECK(bytes(A) == M * K * 3 / 4 && bytes(B) == N * K * 3 / 4,
                __func__,
                " A/B must be MXFP6 K128-blocked planes, rows*K*3/4 bytes");
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
    const uint32_t m = M, n = N, k0 = K / 2, k1 = K / 4;
    char* pa        = static_cast<char*>(A->ptr);
    char* pb        = static_cast<char*>(B->ptr);
    a.ptr_A0        = pa;
    a.a0_rows       = m;
    a.a0_row_bytes  = k0;
    a.a0_stride     = k0;
    a.ptr_A1        = pa + M * K / 2;
    a.a1_rows       = m;
    a.a1_row_bytes  = k1;
    a.a1_stride     = k1;
    a.ptr_B0        = pb;
    a.b0_rows       = n;
    a.b0_row_bytes  = k0;
    a.b0_stride     = k0;
    a.ptr_B1        = pb + N * K / 2;
    a.b1_rows       = n;
    a.b1_row_bytes  = k1;
    a.b1_stride     = k1;
    a.ptr_C         = out->ptr;
    a.c_rows        = m;
    a.c_cols        = n;
    a.c_stride      = n;
    a.ptr_SA        = A_scale->ptr;
    a.sa_dwords     = static_cast<uint32_t>(bytes(A_scale) / 4);
    a.ptr_SB        = B_scale->ptr;
    a.sb_dwords     = static_cast<uint32_t>(bytes(B_scale) / 4);
    a.M             = m;
    a.N             = n;
    size_t arg_size = kArgsNoBias;
    if(bias != nullptr)
    {
        AITER_CHECK(bias->dtype() == AITER_DTYPE_bf16 && bias->is_contiguous() &&
                        bias->numel() == N && bias->device_id == out->device_id && aligned(bias),
                    __func__,
                    " bias must be a contiguous, 16-byte aligned bf16 [N] on the output's GPU");
        a.ptr_bias   = bias->ptr;
        a.bias_elems = n;
        arg_size     = sizeof(a);
    }

    const HipDeviceGuard device_guard(A->device_id);
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const std::string kname = get_gpu_arch() + "aiter_a6w6_fly_" + (bias ? "bias_" : "") +
                              std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    CFG* cfgs               = &cfg_f6flygemm_bf16_per1x32Fp6Fp6_fly;
    auto it                 = cfgs->find(kname);
    AITER_CHECK(it != cfgs->end(), __func__, " no a6w6_fly kernel for " + kname);
    const auto& cfg = it->second;
    AITER_CHECK(cfg.M == M && cfg.N == N && cfg.K == K && cfg.bias == (bias != nullptr) &&
                    cfg.grid > 0,
                __func__,
                " manifest mismatch for " + kname);
    const char* name = cfg.knl_name.c_str();
    const char* co   = cfg.co_name.c_str();
    AiterAsmKernel* impl =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co); });
    impl->launch_kernel({&a, &arg_size, cfg.grid, 1, 1, 256, 1, 1, stream});
}
