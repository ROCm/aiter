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
//   0  the A4W4 160-byte kernarg (KernelArgsTs4)
//   1  the A6W6 156-byte kernarg, 172 with bias (KernelArgsTs6): A6W6, and A6W4 with B's C1 slot
//      aliasing B: the kernel never reads it
//   2  abi 1 with K in the dword after N: the shape-generic kernels (M = N = K = 0 in the manifest), which
//      serve every M, N (multiples of 256) and every K in their K-loop class -- K/128 = kcls (mod 12) and
//      K >= kmin -- one tile per workgroup (grid = tiles).
//   3  the K-generic A4W4 kernel: the first 112 bytes of abi 0 with K in the dword after N; serves every
//      M, N (multiples of 256) and every K that is a multiple of 512 and >= kmin (kcls 12: any K-loop
//      class), B scales at interleave 4; one tile per workgroup.
//   4  abi 0 plus the attention softmax_d epilogue (148 bytes): C1 carries O ([M, N] bf16, out's layout)
//      and C2's pointer / first dword δ ([B, N/128, S] fp32, from this GEMM's first sequence position) and
//      its remaining elements. Rows are sbhd (row = s * B + b); the kernel adds, per row and 128-column
//      head, the fp32 sum of bf16(out) * O over the head into δ, which the caller zeroes. Selected by the
//      manifest's epi_b / epi_s (B, S; 0 for every other row).
//   5  two-segment A6W6 with a gated-residual epilogue (gemm_tilescale_cat_asm, 304 bytes): abi 1 without
//      bias (segment 0: A x B over K) + 4 bytes of padding, then segment 1's operands (A2's codes, B2's C0
//      and C1 planes, A2's and B2's scale slabs), X, G, bias and H as (pointer, element count, pad), and the
//      gate's row stride in elements. One fp32 accumulator runs over segment 0's K, then segment 1's K2
//      (manifest column k2); then a = bf16(acc), h32 = a + bias (fp32), H = bf16(h32) and
//      out = bf16(fma(G[row % epi_b], h32, X)). epi_b is the gate's row count (rows are sbhd, row = s * B + b).
namespace {
constexpr size_t kTensorAlignment = 16;

struct Memref
{
    void* ptr;
    uint32_t rows, row_bytes, stride, pad;
};
static_assert(sizeof(Memref) == 24, "memref must be 24 bytes");

struct __attribute__((packed)) KernelArgsTs4
{
    Memref A, B, C0;
    void* ptr_SA;
    uint32_t sa_dwords, sa_pad;
    void* ptr_SB;
    uint32_t sb_dwords;
    uint32_t M;
    uint32_t N;
    uint32_t K; // abi 3 only (0 for abi 0)
    Memref C1, C2;
};
static_assert(sizeof(KernelArgsTs4) == 160, "ts4 kernarg must be 160 bytes");
static_assert(offsetof(KernelArgsTs4, ptr_SA) == 72 && offsetof(KernelArgsTs4, M) == 100 &&
                  offsetof(KernelArgsTs4, C1) == 112,
              "ts4 kernarg ABI mismatch");

struct __attribute__((packed)) KernelArgsTs6
{
    Memref A0, A1, B0, B1, C;
    void* ptr_SA;
    uint32_t sa_dwords, sa_pad;
    void* ptr_SB;
    uint32_t sb_dwords;
    uint32_t M;
    uint32_t N;
    uint32_t K; // abi 2 only (0 for abi 1)
    void* ptr_bias;
    uint32_t bias_elems;
};
static_assert(sizeof(KernelArgsTs6) == 172, "ts6 kernarg must be 172 bytes");
static_assert(offsetof(KernelArgsTs6, ptr_SA) == 120 && offsetof(KernelArgsTs6, M) == 148 &&
                  offsetof(KernelArgsTs6, ptr_bias) == 160,
              "ts6 kernarg ABI mismatch");
constexpr size_t kTs6NoBias = 156;

struct __attribute__((packed)) Ptr1d
{
    void* ptr;
    uint32_t elems, pad;
};
static_assert(sizeof(Ptr1d) == 16, "1-d tensor slot must be 16 bytes");

struct __attribute__((packed)) KernelArgsCat6
{
    Memref A0, A1, B0, B1, C;
    void* ptr_SA;
    uint32_t sa_dwords, sa_pad;
    void* ptr_SB;
    uint32_t sb_dwords;
    uint32_t M;
    uint32_t N;
    uint32_t pad0;
    Ptr1d A2, B20, B21, SA2, SB2, X, G, Bias;
    void* ptr_H;
    uint32_t h_elems;
    int32_t ldg;
};
static_assert(sizeof(KernelArgsCat6) == 304, "cat6 kernarg must be 304 bytes");
static_assert(offsetof(KernelArgsCat6, M) == 148 && offsetof(KernelArgsCat6, A2) == 160 &&
                  offsetof(KernelArgsCat6, X) == 240 && offsetof(KernelArgsCat6, ptr_H) == 288 &&
                  offsetof(KernelArgsCat6, ldg) == 300,
              "cat6 kernarg ABI mismatch");

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
                             int64_t K,
                             int64_t epi_b,
                             int64_t epi_s)
{
    static std::mutex mu;
    static std::unordered_map<std::string, const tsgemmConfig*> cache;
    const std::string key = arch + ":" + std::to_string(a_fmt) + "," + std::to_string(b_fmt) + "," +
                            std::to_string(b_codes) + "," + std::to_string(b_ilv) + "," +
                            std::to_string(bias) + "," + std::to_string(M) + "x" +
                            std::to_string(N) + "x" + std::to_string(K) + ":" +
                            std::to_string(epi_b) + "x" + std::to_string(epi_s);
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
           c.K == K && c.epi_b == epi_b && c.epi_s == epi_s && c.k2 == 0)
        {
            found = &c;
            break;
        }
    }
    // No exact row: the shape-generic kernel of this K-loop class, if any.
    for(const auto& kv : cfg_tsgemm_bf16_per1x32)
    {
        if(found != nullptr || epi_b != 0)
            break;
        const auto& c = kv.second;
        if(c.arch == arch && (c.abi == 2 || c.abi == 3) && c.a_fmt == a_fmt &&
           c.b_fmt == b_fmt && c.b_codes == b_codes && c.b_ilv == b_ilv &&
           c.bias == static_cast<int>(bias) && K % 512 == 0 &&
           (c.kcls == 12 || (K / 128) % 12 == c.kcls) && K >= c.kmin)
            found = &c;
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
     aiter_tensor_t* B_c1, // optional: an FP6 B's C1 plane in its own buffer (B then holds C0 only)
     aiter_tensor_t* epi_o,     // optional: O for the softmax_d epilogue ([M, N] bf16, abi 4)
     aiter_tensor_t* epi_delta, // with epi_o: δ [B, N/128, S] fp32, zeroed by the caller
     int64_t epi_s0,            // the sequence position of this GEMM's first row
     hipStream_t stream),
    (A, B, A_scale, B_scale, out, K, bias, a_fmt, b_fmt, b_codes, b_ilv, B_c1, epi_o, epi_delta, epi_s0,
     stream))
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
    if(B_c1 != nullptr)
        AITER_CHECK(b_fmt == 6 && B_c1->is_contiguous() && B_c1->device_id == B->device_id &&
                        bytes(B) == N * K / 2 && bytes(B_c1) == N * K / 4,
                    __func__,
                    " B_c1: an FP6 B's C1 plane (N*K/4 bytes) with B its C0 plane (N*K/2 bytes)");
    AITER_CHECK(bytes(A) == code_bytes(a_fmt, M, K) &&
                    (B_c1 != nullptr || bytes(B) == code_bytes(b_fmt, N, K)),
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

    AITER_CHECK((epi_o == nullptr) == (epi_delta == nullptr), __func__, " epi_o and epi_delta go together");
    int64_t epi_b = 0, epi_s = 0;
    if(epi_o != nullptr)
    {
        AITER_CHECK(epi_o->dtype() == AITER_DTYPE_bf16 && epi_o->is_contiguous() && epi_o->dim() == 2 &&
                        epi_o->size(0) == M && epi_o->size(1) == N && aligned(epi_o) &&
                        epi_o->device_id == out->device_id,
                    __func__,
                    " epi_o must be a contiguous, 16-byte aligned bf16 [M, N] on the output's GPU");
        AITER_CHECK(epi_delta->dtype() == AITER_DTYPE_fp32 && epi_delta->is_contiguous() &&
                        epi_delta->dim() == 3 && epi_delta->size(1) * 128 == N &&
                        epi_delta->device_id == out->device_id,
                    __func__,
                    " epi_delta must be a contiguous fp32 [B, N/128, S] on the output's GPU");
        epi_b = epi_delta->size(0);
        epi_s = epi_delta->size(2);
        AITER_CHECK(M % epi_b == 0 && epi_s0 >= 0 && epi_s0 + M / epi_b <= epi_s,
                    __func__,
                    " the rows must be whole sequence positions of epi_delta");
    }
    const std::string arch = get_gpu_arch();
    const tsgemmConfig* cfg =
        find_row(arch, a_fmt, b_fmt, b_codes, b_ilv, bias != nullptr, M, N, K, epi_b, epi_s);
    const bool generic = cfg != nullptr && (cfg->abi == 2 || cfg->abi == 3);
    AITER_CHECK(cfg != nullptr && (cfg->grid > 0 || generic),
                __func__,
                " no tilescale kernel for a" + std::to_string(a_fmt) + "w" +
                    std::to_string(b_fmt) + " " + std::to_string(M) + "x" + std::to_string(N) +
                    "x" + std::to_string(K) + (bias ? " with bias" : "") +
                    (epi_b ? " with softmax_d B " + std::to_string(epi_b) + " S " + std::to_string(epi_s)
                           : ""));

    char* pa = static_cast<char*>(A->ptr);
    char* pb = static_cast<char*>(B->ptr);
    KernelArgsTs4 a4;
    KernelArgsTs6 a6;
    void* args      = nullptr;
    size_t arg_size = 0;
    if(cfg->abi == 0 || cfg->abi == 3 || cfg->abi == 4)
    {
        AITER_CHECK(a_fmt == 4 && b_fmt == 4 && bias == nullptr,
                    __func__,
                    " abi 0 / 3 / 4 rows are A4W4 without bias");
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
        a4.K         = cfg->abi == 3 ? static_cast<uint32_t>(K) : 0u;
        args         = &a4;
        arg_size     = cfg->abi == 3 ? offsetof(KernelArgsTs4, C1) : sizeof(a4);
        if(cfg->abi == 4)
        {
            a4.C1         = Memref{epi_o->ptr,
                           static_cast<uint32_t>(M),
                           static_cast<uint32_t>(N),
                           static_cast<uint32_t>(N),
                           0};
            a4.C2.ptr     = static_cast<float*>(epi_delta->ptr) + epi_s0;
            a4.C2.rows    = static_cast<uint32_t>(epi_delta->numel() - epi_s0);
            arg_size      = offsetof(KernelArgsTs4, C2) + 12;
        }
    }
    else
    {
        AITER_CHECK((cfg->abi == 1 || cfg->abi == 2) && a_fmt == 6,
                    __func__,
                    " unknown abi / format pairing");
        std::memset(&a6, 0, sizeof(a6));
        a6.A0 = memref(pa, M, K / 2);
        a6.A1 = memref(pa + M * K / 2, M, K / 4);
        if(b_fmt == 6)
        {
            a6.B0 = memref(pb, N, K / 2);
            a6.B1 = memref(B_c1 != nullptr ? static_cast<char*>(B_c1->ptr) : pb + N * K / 2, N, K / 4);
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
        a6.K         = cfg->abi == 2 ? static_cast<uint32_t>(K) : 0u;
        arg_size     = kTs6NoBias + (cfg->abi == 2 ? 4 : 0);
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
    const int grid = generic ? static_cast<int>((M / 256) * (N / 256)) : cfg->grid;
    impl->launch_kernel({args, &arg_size, grid, 1, 1, 256, 1, 1, stream});
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_tilescale_cat_asm,
    (aiter_tensor_t * A,       // segment 0: A6 tilescale codes of [M, K] (C0 then C1)
     aiter_tensor_t* B,        // segment 0: A6 tilescale codes of [N, K], or its C0 plane with B_c1
     aiter_tensor_t* A_scale,  // segment 0: tilescale slab, role A, M*K/32 bytes
     aiter_tensor_t* B_scale,  // segment 0: tilescale slab, role B (interleave 0), N*K/32 bytes
     aiter_tensor_t* A2,       // segment 1: the same over K2
     aiter_tensor_t* B2,
     aiter_tensor_t* A2_scale,
     aiter_tensor_t* B2_scale,
     aiter_tensor_t* out,      // [M, N] bf16 contiguous
     aiter_tensor_t* h,        // [M, N] bf16 contiguous
     int64_t K,
     int64_t K2,
     aiter_tensor_t* bias,     // bf16 [N] contiguous
     aiter_tensor_t* x,        // bf16 [M, N] contiguous
     aiter_tensor_t* gate,     // bf16 [Bt, N], unit column stride, any row stride
     aiter_tensor_t* B_c1,     // optional: B's C1 plane in its own buffer
     aiter_tensor_t* B2_c1,    // optional: B2's C1 plane in its own buffer
     hipStream_t stream),
    (A, B, A_scale, B_scale, A2, B2, A2_scale, B2_scale, out, h, K, K2, bias, x, gate, B_c1, B2_c1, stream))
{
    const auto bytes = [](const aiter_tensor_t* t) {
        return static_cast<int64_t>(t->numel()) * t->element_size();
    };
    const auto aligned = [](const aiter_tensor_t* t) {
        return reinterpret_cast<uintptr_t>(t->ptr) % kTensorAlignment == 0;
    };
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16 && out->dim() == 2 && out->is_contiguous(),
                __func__,
                " out must be a contiguous 2D bf16 tensor");
    const int64_t M = out->size(0), N = out->size(1);
    AITER_CHECK(K > 0 && K2 > 0 && K % 256 == 0 && K2 % 256 == 0 && M % 256 == 0 && N % 256 == 0,
                __func__,
                " M, N, K, K2 must be multiples of 256");
    const int dev = out->device_id;
    for(const aiter_tensor_t* t : {h, x})
        AITER_CHECK(t->dtype() == AITER_DTYPE_bf16 && t->dim() == 2 && t->size(0) == M &&
                        t->size(1) == N && t->is_contiguous() && aligned(t) && t->device_id == dev,
                    __func__,
                    " h and x must be contiguous, 16-byte aligned bf16 [M, N] on the output's GPU");
    AITER_CHECK(aligned(out), __func__, " out must be 16-byte aligned");
    // out and h are written tile by tile while x is still being read, so none of the three may overlap
    const auto overlap = [&](const aiter_tensor_t* p, const aiter_tensor_t* q) {
        const auto a0 = reinterpret_cast<uintptr_t>(p->ptr), b0 = reinterpret_cast<uintptr_t>(q->ptr);
        return a0 < b0 + static_cast<uintptr_t>(bytes(q)) && b0 < a0 + static_cast<uintptr_t>(bytes(p));
    };
    AITER_CHECK(!overlap(out, h) && !overlap(out, x) && !overlap(h, x),
                __func__,
                " out, h and x must not overlap");
    AITER_CHECK(bias->dtype() == AITER_DTYPE_bf16 && bias->is_contiguous() && bias->numel() == N &&
                    aligned(bias) && bias->device_id == dev,
                __func__,
                " bias must be a contiguous, 16-byte aligned bf16 [N] on the output's GPU");
    AITER_CHECK(gate->dtype() == AITER_DTYPE_bf16 && gate->dim() == 2 && gate->size(1) == N &&
                    gate->stride(1) == 1 && gate->stride(0) >= N && gate->stride(0) % 8 == 0 &&
                    aligned(gate) && gate->device_id == dev,
                __func__,
                " gate must be bf16 [B, N] with unit column stride, a row stride >= N and a multiple of 8, "
                "16-byte aligned, on the output's GPU");
    const int64_t Bt = gate->size(0), ldg = gate->stride(0);
    AITER_CHECK(Bt > 0 && M % Bt == 0 && ldg < (int64_t{1} << 26), __func__, " gate rows must divide M");

    // per segment: codes, optional separate C1 plane, scale slabs
    char* pb[2];
    char* pb1[2];
    const aiter_tensor_t* seg[2][5] = {{A, B, A_scale, B_scale, B_c1}, {A2, B2, A2_scale, B2_scale, B2_c1}};
    const int64_t ks[2] = {K, K2};
    for(int s = 0; s < 2; ++s)
    {
        const aiter_tensor_t *a = seg[s][0], *b = seg[s][1], *sa = seg[s][2], *sb = seg[s][3],
                             *c1 = seg[s][4];
        const int64_t k = ks[s];
        for(const aiter_tensor_t* t : {a, b, sa, sb})
            AITER_CHECK(t->is_contiguous() && aligned(t) && t->device_id == dev,
                        __func__,
                        " operands and scales must be contiguous, 16-byte aligned, on the output's GPU");
        AITER_CHECK(bytes(a) == M * k * 3 / 4 && bytes(sa) == M * k / 32 && bytes(sb) == N * k / 32,
                    __func__,
                    " A / scale byte sizes do not match M, N, K (segment " + std::to_string(s) + ")");
        pb[s] = static_cast<char*>(b->ptr);
        if(c1 != nullptr)
        {
            AITER_CHECK(c1->is_contiguous() && aligned(c1) && c1->device_id == dev &&
                            bytes(b) == N * k / 2 && bytes(c1) == N * k / 4,
                        __func__,
                        " B_c1: B's C1 plane (N*K/4 bytes) with B its C0 plane (N*K/2 bytes)");
            pb1[s] = static_cast<char*>(c1->ptr);
        }
        else
        {
            AITER_CHECK(bytes(b) == N * k * 3 / 4, __func__, " B byte size does not match N, K");
            pb1[s] = pb[s] + N * k / 2;
        }
    }

    const std::string arch = get_gpu_arch();
    const tsgemmConfig* cfg = nullptr;
    for(const auto& kv : cfg_tsgemm_bf16_per1x32)
    {
        const auto& c = kv.second;
        if(c.arch == arch && c.abi == 5 && c.a_fmt == 6 && c.b_fmt == 6 && c.M == M && c.N == N &&
           c.K == K && c.k2 == K2 && c.epi_b == Bt)
        {
            cfg = &c;
            break;
        }
    }
    AITER_CHECK(cfg != nullptr,
                __func__,
                " no two-segment tilescale kernel for " + std::to_string(M) + "x" + std::to_string(N) +
                    "x(" + std::to_string(K) + "+" + std::to_string(K2) + ") with " +
                    std::to_string(Bt) + " gate rows");

    KernelArgsCat6 a;
    std::memset(&a, 0, sizeof(a));
    char* pa = static_cast<char*>(A->ptr);
    a.A0         = memref(pa, M, K / 2);
    a.A1         = memref(pa + M * K / 2, M, K / 4);
    a.B0         = memref(pb[0], N, K / 2);
    a.B1         = memref(pb1[0], N, K / 4);
    a.C          = Memref{out->ptr,
                 static_cast<uint32_t>(M),
                 static_cast<uint32_t>(N),
                 static_cast<uint32_t>(N),
                 0};
    a.ptr_SA     = A_scale->ptr;
    a.sa_dwords  = static_cast<uint32_t>(bytes(A_scale) / 4);
    a.ptr_SB     = B_scale->ptr;
    a.sb_dwords  = static_cast<uint32_t>(bytes(B_scale) / 4);
    a.M          = static_cast<uint32_t>(M);
    a.N          = static_cast<uint32_t>(N);
    const auto p1 = [](void* p, int64_t n) { return Ptr1d{p, static_cast<uint32_t>(n), 0}; };
    a.A2         = p1(A2->ptr, bytes(A2));
    a.B20        = p1(pb[1], N * K2 / 2);
    a.B21        = p1(pb1[1], N * K2 / 4);
    a.SA2        = p1(A2_scale->ptr, bytes(A2_scale) / 4);
    a.SB2        = p1(B2_scale->ptr, bytes(B2_scale) / 4);
    a.X          = p1(x->ptr, M * N);
    a.G          = p1(gate->ptr, (Bt - 1) * ldg + N);
    a.Bias       = p1(bias->ptr, N);
    a.ptr_H      = h->ptr;
    a.h_elems    = static_cast<uint32_t>(M * N);
    a.ldg        = static_cast<int32_t>(ldg);
    size_t arg_size = sizeof(a);

    const HipDeviceGuard device_guard(dev);
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name = cfg->knl_name.c_str();
    const char* co   = cfg->co_name.c_str();
    AiterAsmKernel* impl =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co); });
    impl->launch_kernel({&a, &arg_size, cfg->grid, 1, 1, 256, 1, 1, stream});
}
