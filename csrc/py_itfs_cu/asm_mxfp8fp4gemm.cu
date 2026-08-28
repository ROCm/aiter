// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ============================================================================
// gfx1250 F8GEMM ASM Support Matrix
// ----------------------------------------------------------------------------
//  OUTTYPE | A_PRESHUFFLE | B_INTYPE |   M    |   N    |   K
// ---------+--------------+----------+--------+--------+--------
//  BF16    |      0       |  MXFP8   | %1==0  | %16==0 | %128==0
//  BF16    |      0       |  MXFP4   | %1==0  | %16==0 | %128==0
//  BF16    |      1       |  MXFP8   | %2==0  | %16==0 | %128==0
//  BF16    |      1       |  MXFP4   | %2==0  | %16==0 | %128==0
// ----------------------------------------------------------------------------
// Notes:
//  - Currently only support BF16 output.
//  - B_PRESHUFFLE is always 1 (B is always pre-shuffled).
//  - A_PRESHUFFLE=1 tightens the M constraint from %1==0 to %2==0.
//  - K is always a multiple of 128.
// ============================================================================
//
// gfx1250 MXFP8 x {MXFP8, MXFP4} GEMM ASM dispatch (preload SGPR mode).
// A (activation) is always MXFP8 (e4m3, 1 byte/elem); B (weight) is either
// MXFP8 (a8w8) or MXFP4 (a8w4, e2m1, 2 elems/byte). Both operands carry OCP
// micro-scaling block scales (e8m0, one per 32 K-elements).
//
// Two entrypoints:
//   - mxfp8_mxfp8_gemm_asm: D[M,N] bf16 = A[M,K] mxfp8 * B[N,K] mxfp8   (a8w8)
//   - mxfp8_mxfp4_gemm_asm: D[M,N] bf16 = A[M,K] mxfp8 * B[N,K/2] mxfp4 (a8w4)
//
// KernelArgs is the packed preload layout the POC silicon host ships (76B):
// 5 pointers (MEM-first), then 9 tight 4B scalars. The persistent + cluster
// shaders do their own tile scheduling, so unlike f4gemm there are no
// log2_grid kernargs -- the host only supplies M/N/K/batch and launches on a
// fixed cluster grid.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_mxfp8fp4gemm_configs.hpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <hip/hip_runtime.h>

constexpr int MX_SCALE_BLOCK = 32;

constexpr int F8GEMM_N_ALIGN      = 16;
constexpr int F8GEMM_K_ALIGN      = 128;
constexpr int F8GEMM_M_ALIGN_APRE = 2;

// Preload-mode KernelArgs (4B-tight, MEM-first). Offsets in comments are the
// kernarg byte offsets the preload-aware shader s_load's from.
struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;             // s[2:3]   off 0x00
    void* ptr_A;             // s[4:5]   off 0x08
    void* ptr_B;             // s[6:7]   off 0x10
    void* ptr_ScaleA;        // s[8:9]   off 0x18
    void* ptr_ScaleB;        // s[10:11] off 0x20
    unsigned int stride_C;   // s12      off 0x28  (bytes)
    unsigned int stride_A;   // s13      off 0x2c  (bytes)
    unsigned int stride_B;   // s14      off 0x30  (bytes)
    unsigned int ScaleA_K;   // s15      off 0x34  (= K/32)
    unsigned int ScaleB_K;   // s16      off 0x38  (= K/32)
    unsigned int M;          // s17      off 0x3c
    unsigned int N;          // s18      off 0x40
    unsigned int K;          // s19      off 0x44
    unsigned int batch_size; // s20      off 0x48
};
static_assert(sizeof(KernelArgs) == 76, "mxfp8fp4 preload KernelArgs must be 76B");

// Pick the best registered kernel variant for (M,N,K) given the B dtype and
// a_preshuffle.
static std::tuple<std::string, int> get_heuristic_kernel(int M,
                                                         int N,
                                                         int K,
                                                         std::string arch_id,
                                                         const std::string& b_intype,
                                                         const std::string& outtype,
                                                         int a_preshuffle,
                                                         CFG* cfgs)
{
    // Tile choice is a plain size rule, not a round/efficiency search: a tiny M
    // wastes most of a 256-tall tile's rows, so M<=64 takes the 64x512 variant;
    // any larger M takes 256x256 (which also fills a full persistent 256-TG wave).
    const int want_tile_m = (M <= 64) ? 64 : 256;
    const int want_tile_n = (M <= 64) ? 512 : 256;

    std::string selectedKernelName = "";
    std::string fallbackKernelName = ""; // any valid variant if the wanted tile is absent

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.b_intype != b_intype || cfg.a_preshuffle != a_preshuffle)
            continue;

        if(cfg.outtype != outtype)
            continue;

        const int m_align = a_preshuffle ? F8GEMM_M_ALIGN_APRE : 1;
        if((M % m_align) != 0 || (N % F8GEMM_N_ALIGN) != 0 || (K % F8GEMM_K_ALIGN) != 0)
            continue;

        // Remember the first valid variant so an odd (b_intype,outtype) combo that
        // only ships one tile still resolves.
        if(fallbackKernelName.empty())
            fallbackKernelName = el.first;

        if(cfg.tile_m == want_tile_m && cfg.tile_n == want_tile_n)
        {
            selectedKernelName = el.first;
            break;
        }
    }

    if(selectedKernelName.empty())
        selectedKernelName = fallbackKernelName;

    AITER_CHECK(selectedKernelName != "",
                __func__,
                ": cannot get heuristic kernel for b_intype=",
                b_intype,
                ", a_preshuffle=",
                a_preshuffle,
                ", M=",
                M,
                ", N=",
                N,
                ", K=",
                K,
                " (require N%16==0, K%128==0, and M%2==0 when a_preshuffle=1)");
    return std::make_tuple(selectedKernelName, 1);
}

// Shared dispatch body for both a8w8 (B=mxfp8) and a8w4 (B=mxfp4).
static void mxfp8fp4_launch(aiter_tensor_t* A,
                            aiter_tensor_t* B,
                            aiter_tensor_t* ScaleA,
                            aiter_tensor_t* ScaleB,
                            aiter_tensor_t* out,
                            const char* kernelName,
                            const std::string& b_intype,
                            int a_preshuffle,
                            hipStream_t stream)
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, __func__, " only supports BFloat16 output");
    const char* out_type = "bf16";
    AITER_CHECK(
        b_intype == "mxfp8" || b_intype == "mxfp4", __func__, " unsupported b_intype ", b_intype);
    AITER_CHECK(a_preshuffle == 0 || a_preshuffle == 1, __func__, " a_preshuffle must be 0 or 1");

    int Mdim = A->size(0);
    int Ndim = B->size(0);
    int Kdim = A->size(1); // A is mxfp8: 1 byte/elem, so col count == K

    AITER_CHECK(Kdim % F8GEMM_K_ALIGN == 0,
                __func__,
                " K must be divisible by ",
                F8GEMM_K_ALIGN,
                " (got K=",
                Kdim,
                ")");

    // Strides in bytes. A is fp8 (1 byte); B fp8 (1 byte) or fp4 (0.5 byte);
    // D is bf16 (2 bytes). Scales are e8m0, one per 32-K block.
    unsigned int stride_a = static_cast<unsigned int>(Kdim);
    unsigned int stride_b = (b_intype == "mxfp4") ? static_cast<unsigned int>(Kdim / 2)
                                                  : static_cast<unsigned int>(Kdim);
    unsigned int stride_d = static_cast<unsigned int>(Ndim) * 2;
    unsigned int scale_k  = static_cast<unsigned int>(Kdim / MX_SCALE_BLOCK);

    KernelArgs args{};
    args.ptr_D      = out->ptr;
    args.ptr_A      = A->ptr;
    args.ptr_B      = B->ptr;
    args.ptr_ScaleA = ScaleA->ptr;
    args.ptr_ScaleB = ScaleB->ptr;
    args.stride_C   = stride_d;
    args.stride_A   = stride_a;
    args.stride_B   = stride_b;
    args.ScaleA_K   = scale_k;
    args.ScaleB_K   = scale_k;
    args.M          = Mdim;
    args.N          = Ndim;
    args.K          = Kdim;
    args.batch_size = 1;
    size_t arg_size = sizeof(KernelArgs);

    const HipDeviceGuard device_guard(A->device_id);

    static CFG* config_map = &cfg_mxfp8fp4gemm;
    AITER_CHECK(!config_map->empty(),
                __func__,
                " no kernel registered for mxfp8fp4gemm; check AITER_GPU_ARCHS=gfx1250");

    std::string arch_id      = get_gpu_arch();
    std::string selectedName = (kernelName && kernelName[0] != '\0') ? (arch_id + kernelName) : "";

    const int intype_id = (b_intype == "mxfp4") ? 1 : 0; // else mxfp8
    using DictKey       = std::tuple<int, int, int, int, int>; // M,N,K,intype_id,apre
    struct DictHash
    {
        size_t operator()(const DictKey& k) const
        {
            const auto& [m, n, kk, it, ap] = k;
            size_t h                       = 1469598103934665603ull;
            for(int v : {m, n, kk, it, ap})
                h = (h ^ static_cast<size_t>(static_cast<unsigned>(v))) * 1099511628211ull;
            return h;
        }
    };
    static SynchronizedCache<DictKey, std::string, DictHash> heuristic_kernel_dict;

    if(selectedName.empty())
    {
        selectedName = heuristic_kernel_dict.get_or_create(
            DictKey(Mdim, Ndim, Kdim, intype_id, a_preshuffle), [&]() {
                auto [name, _] = get_heuristic_kernel(
                    Mdim, Ndim, Kdim, arch_id, b_intype, out_type, a_preshuffle, config_map);
                return name;
            });
    }

    auto it = config_map->find(selectedName);
    AITER_CHECK(
        it != config_map->end(), __func__, " kernel not in cfg_mxfp8fp4gemm: ", selectedName);

    const auto& cfg = it->second;
    // Guard the explicit-kernelName path. outtype MUST match: a mismatched .co
    // keeps the same kernarg size (HIP won't catch it) but sizes stride_d / the
    // output buffer for a different element width -> device-side OOB write.
    AITER_CHECK(cfg.b_intype == b_intype && cfg.a_preshuffle == a_preshuffle &&
                    cfg.outtype == out_type,
                __func__,
                " selected kernel ",
                selectedName,
                " mismatches requested b_intype/a_preshuffle/outtype (got outtype=",
                cfg.outtype,
                ", requested ",
                out_type,
                ")");

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        cfg.knl_name, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    // ----- Launch geometry: cluster + persistent -----
    // Every f8gemm .co is a persistent shader, so the launch is fixed-size and
    // independent of M/N/K: exactly WG_MAX threadgroups, laid out 1D along X with
    // Y carrying only the cluster_y rows. The tile-walk swizzle (GRID_X/GRID_Y) is
    // baked into the .co at assemble time, which asserts
    // (GRID_X*CLUSTER_X) * (GRID_Y*CLUSTER_Y) == WG_MAX -- so the host only has to
    // ship the right *total* threadgroup count, not the same grid shape.
    const int cluster_x = cfg.cluster_x > 0 ? cfg.cluster_x : 1; // compile-time per .co
    const int cluster_y = cfg.cluster_y > 0 ? cfg.cluster_y : 1;

    constexpr int WG_MAX = 256; // must match the .co's WG_MAX

    const int cluster_size = cluster_x * cluster_y;
    AITER_CHECK((WG_MAX % cluster_size) == 0,
                __func__,
                " persistent WG_MAX=",
                WG_MAX,
                " not divisible by cluster_x*cluster_y=",
                cluster_size);

    // HIP gridDim must be a multiple of clusterDim per axis.
    const int gdx = (WG_MAX / cluster_size) * cluster_x;
    const int gdy = 1 * cluster_y;
    const int gdz = 1;

    const int bdx = 128; // 4 waves * 32 threads on gfx1250

    impl_ptr->launch_kernel(
        {&args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, stream, cluster_x, cluster_y, 1});
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp8_mxfp8_gemm_asm,
    (aiter_tensor_t * A,     // A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,      // B:[N, K]   mxfp8 e4m3 (always preshuffled)
     aiter_tensor_t* ScaleA, // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB, // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,    // Out:[M, N] bf16
     const char* kernelName,
     int a_preshuffle,
     hipStream_t stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, stream))
{
    mxfp8fp4_launch(A, B, ScaleA, ScaleB, out, kernelName, "mxfp8", a_preshuffle, stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp8_mxfp4_gemm_asm,
    (aiter_tensor_t * A,     // A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,      // B:[N, K/2] mxfp4 e2m1 (always preshuffled)
     aiter_tensor_t* ScaleA, // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB, // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,    // Out:[M, N] bf16
     const char* kernelName,
     int a_preshuffle,
     hipStream_t stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, stream))
{
    mxfp8fp4_launch(A, B, ScaleA, ScaleB, out, kernelName, "mxfp4", a_preshuffle, stream);
}
