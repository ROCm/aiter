// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Skinny GEMM with BF16 activations and MXFP8 weights for decode on gfx942:
// out = A @ dequant(B, B_scale).T, with A [M, K] bf16, B [N, K] fp8 bytes,
// B_scale [N, K / 32] with one UE8M0 exponent per 32 weights along K, and out
// [M, N] bf16, for M from 1 to 64.
//
// gfx942 has no MXFP8 MFMA, and its fp8 conversion reads e4m3fnuz. The kernels
// read the weight bytes and the scale exponents as stored, convert them to fp32
// in registers (v_cvt_pk_f32_fp8), multiply by 2^(e - 127) and pack them to BF16,
// which is exact, and accumulate in fp32 with v_mfma_f32_16x16x16_bf16. Each
// lane's 32-element K chunk is one scale group. The activations are never
// quantized. gemm_a16w8_mxfp8_prepare_weight() in aiter/ops/gemm_op_a16w8_mxfp8.py
// converts an OCP checkpoint (e4m3fn + UE8M0) to what the kernels read
// (e4m3fnuz bits of the same values / 2, exponent + 1); both steps are exact.
//
// A kernel is named by its tile: tr 16-row tiles (M <= 16 * tr), tn 16-column
// tiles per workgroup (N % (16 * tn) == 0), nw waves that split K inside the
// workgroup and reduce through LDS. The "sk" kernels also split K over
// splitK workgroups (grid.y): each writes an fp32 partial tile to `workspace`
// and bumps a per-tile arrival counter in `counters`; the last arrival sums the
// partials in split order (deterministic), stores BF16 and resets the counter
// to 0, so neither buffer needs clearing between calls. Calls that share the
// buffers must be stream-ordered; the Python op keeps one pair per (device,
// stream).
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_a16w8gemm_configs.hpp"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstring>
#include <string>

namespace {

// Kernel arguments of every a16w8gemm_bf16_mxfp8 code object: one 16-byte slot
// per argument (pointer + p2, uint32 + p3). The kernels build their buffer
// descriptors from the pointers and sizes; every buffer access is range-checked
// against the tensor's byte size, which is how rows past M are dropped (Y holds
// exactly M * N * 2 bytes). The split-K kernels append their two buffers and the
// split count.
struct __attribute__((packed)) KernelArgs
{
    const void* ptr_X; // A [M, K] bf16
    p2 _p0;
    const void* ptr_W; // B [N, K] fp8
    p2 _p1;
    const void* ptr_S; // B_scale [N, K / 32] uint8
    p2 _p2;
    void* ptr_Y; // out [M, N] bf16
    p2 _p3;
    unsigned int dim_M;
    p3 _p4;
    unsigned int dim_K;
    p3 _p5;
    unsigned int dim_N;
    p3 _p6;
    unsigned int dim_G; // K / 32
    p3 _p7;
    unsigned int dim_T; // K steps (4 scale groups each) per wave
    p3 _p8;
    void* ptr_WS; // split-K: fp32 partial tiles
    p2 _p9;
    void* ptr_C; // split-K: int32 arrival counters
    p2 _p10;
    unsigned int dim_KS; // split-K: number of splits
    p3 _p11;
};
static_assert(offsetof(KernelArgs, dim_M) == 64, "kernarg ABI of the code objects");
static_assert(offsetof(KernelArgs, ptr_WS) == 144, "kernarg ABI of the code objects");
static_assert(sizeof(KernelArgs) == 192, "kernarg ABI of the code objects");
constexpr size_t kArgsPlain  = 144;
constexpr size_t kArgsSplitK = 192;
constexpr int kMaxSplitK     = 8; // the split-K epilogue sums at most 8 partials

const a16w8gemmConfig& find_kernel(const std::string& arch_id, const char* kernelName)
{
    for(const auto& el : cfg_a16w8gemm_mxfp8)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        if(el.second.knl_name == kernelName)
            return el.second;
    }
    AITER_CHECK(false,
                "gemm_a16w8_mxfp8_asm: no kernel ",
                kernelName,
                " for arch ",
                arch_id,
                ". The MXFP8 kernels are built for gfx942.");
    __builtin_unreachable();
}

} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(gemm_a16w8_mxfp8_asm,
                                    (aiter_tensor_t* A,         // [M, K] bf16
                                     aiter_tensor_t* B,         // [N, K] fp8 (kernel layout)
                                     aiter_tensor_t* B_scale,   // [N, K / 32] uint8 / e8m0
                                     aiter_tensor_t* out,       // [M, N] bf16
                                     aiter_tensor_t* workspace, // split-K partials (bytes)
                                     aiter_tensor_t* counters,  // split-K arrival counters, int32
                                     const char* kernelName,
                                     int splitK,
                                     hipStream_t stream),
                                    (A, B, B_scale, out, workspace, counters, kernelName, splitK, stream))
{
    const char* fn = "gemm_a16w8_mxfp8_asm";
    AITER_CHECK(A->dtype() == AITER_DTYPE_bf16, fn, ": A must be bf16, got ", AiterDtype_to_str(A->dtype()));
    AITER_CHECK(B->element_size() == 1, fn, ": B must hold 1-byte fp8 weights");
    AITER_CHECK(B_scale->element_size() == 1, fn, ": B_scale must hold 1-byte UE8M0 exponents");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, fn, ": out must be bf16, got ", AiterDtype_to_str(out->dtype()));
    AITER_CHECK(A->dim() == 2 && B->dim() == 2 && B_scale->dim() == 2 && out->dim() == 2,
                fn, ": A, B, B_scale and out must be 2-D");
    AITER_CHECK(kernelName != nullptr && kernelName[0] != '\0', fn, ": kernelName is required");

    const int64_t m = A->size(0);
    const int64_t k = A->size(1);
    const int64_t n = B->size(0);
    if(m == 0)
        return;
    AITER_CHECK(k >= 32 && k % 32 == 0, fn, ": K must be a positive multiple of 32, got ", k);
    const int64_t g = k / 32;

    // The kernels address every tensor with fixed row strides.
    AITER_CHECK(A->stride(1) == 1 && A->stride(0) == k, fn, ": A must be contiguous");
    AITER_CHECK(B->size(1) == k && B->stride(1) == 1 && B->stride(0) == k,
                fn, ": B must be a contiguous [N, ", k, "] tensor");
    AITER_CHECK(B_scale->size(0) == n && B_scale->size(1) == g && B_scale->is_contiguous(),
                fn, ": B_scale must be a contiguous [", n, ", ", g, "] tensor");
    AITER_CHECK(out->size(0) == m && out->size(1) == n && out->stride(1) == 1 && out->stride(0) == n,
                fn, ": out must be a contiguous [", m, ", ", n, "] tensor");
    const int dev = A->device_id;
    AITER_CHECK(B->device_id == dev && B_scale->device_id == dev && out->device_id == dev,
                fn, ": all tensors must be on the same GPU");
    // The kernels compute 32-bit buffer sizes and offsets from 48-bit bases.
    AITER_CHECK(n * k < (int64_t(1) << 31) && m * k * 2 < (int64_t(1) << 31) && m * n * 2 < (int64_t(1) << 31),
                fn, ": tensors must be smaller than 2 GiB");
    AITER_CHECK(reinterpret_cast<uint64_t>(A->ptr) < (uint64_t(1) << 48) &&
                    reinterpret_cast<uint64_t>(B->ptr) < (uint64_t(1) << 48) &&
                    reinterpret_cast<uint64_t>(B_scale->ptr) < (uint64_t(1) << 48) &&
                    reinterpret_cast<uint64_t>(out->ptr) < (uint64_t(1) << 48),
                fn, ": addresses must fit in 48 bits");

    const HipDeviceGuard device_guard(dev);
    const std::string arch_id = get_gpu_arch();
    const a16w8gemmConfig& cfg = find_kernel(arch_id, kernelName);

    AITER_CHECK(m <= 16 * cfg.tr, fn, ": ", kernelName, " takes at most ", 16 * cfg.tr, " rows, got M=", m);
    AITER_CHECK(n % (16 * cfg.tn) == 0, fn, ": ", kernelName, " needs N % ", 16 * cfg.tn, " == 0, got N=", n);
    const int ks = cfg.sk ? splitK : 1;
    AITER_CHECK(cfg.sk ? (ks >= 1 && ks <= kMaxSplitK) : (splitK <= 1),
                fn, ": ", kernelName, (cfg.sk ? " takes splitK 1 to 8" : " does not split K"), ", got ", splitK);

    const int grid = static_cast<int>(n / (16 * cfg.tn));
    KernelArgs args;
    std::memset(&args, 0, sizeof(args));
    args.ptr_X  = A->ptr;
    args.ptr_W  = B->ptr;
    args.ptr_S  = B_scale->ptr;
    args.ptr_Y  = out->ptr;
    args.dim_M  = static_cast<unsigned int>(m);
    args.dim_K  = static_cast<unsigned int>(k);
    args.dim_N  = static_cast<unsigned int>(n);
    args.dim_G  = static_cast<unsigned int>(g);
    args.dim_T  = static_cast<unsigned int>(((g + 3) / 4 + cfg.nw * ks - 1) / (cfg.nw * ks));
    size_t arg_size = kArgsPlain;
    if(cfg.sk)
    {
        const int64_t tiles    = int64_t(grid) * cfg.tr * cfg.tn;
        const int64_t ws_bytes = int64_t(ks) * tiles * 1024; // one 16 x 16 fp32 tile per split
        AITER_CHECK(workspace != nullptr && counters != nullptr, fn, ": split-K kernels need workspace and counters");
        AITER_CHECK(workspace->numel() * workspace->element_size() >= static_cast<size_t>(ws_bytes),
                    fn, ": workspace holds ", workspace->numel() * workspace->element_size(), " bytes, ", ws_bytes, " needed");
        AITER_CHECK(counters->dtype() == AITER_DTYPE_i32 && counters->numel() >= static_cast<size_t>(tiles),
                    fn, ": counters must be int32 with at least ", tiles, " zeroed entries");
        AITER_CHECK(workspace->device_id == dev && counters->device_id == dev, fn, ": workspace and counters must be on the GPU of A");
        args.ptr_WS = workspace->ptr;
        args.ptr_C  = counters->ptr;
        args.dim_KS = static_cast<unsigned int>(ks);
        arg_size    = kArgsSplitK;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             grid,        // gdx: N / (16 * tn)
                             ks,          // gdy: K splits
                             1,           // gdz
                             cfg.threads, // bdx: 64 * nw
                             1,
                             1,
                             stream});
}
