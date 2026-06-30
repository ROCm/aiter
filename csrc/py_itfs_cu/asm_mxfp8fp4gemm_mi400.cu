// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 MXFP8 x {MXFP8, MXFP4} GEMM ASM dispatch (kernarg preload mode).
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
// fixed cluster grid (see f8gemm POC mxfp8fp4gemm.cpp).
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_mxfp8fp4gemm_configs.hpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <hip/hip_runtime.h>

// B (weight) dtype selector (mirrors the POC b_type: 0=fp8, 1=fp4).
constexpr int B_DTYPE_FP8     = 0;
constexpr int B_DTYPE_FP4     = 1;
constexpr int MX_SCALE_BLOCK  = 32; // OCP MX block size (e8m0, one scale / 32 K)
constexpr int K_ALIGN         = 128; // POC requires K % 128 == 0

// Packed preload KernelArgs (4B-tight, MEM-first). Offsets in comments are the
// kernarg byte offsets the preload-aware shader s_load's from. Must stay
// bit-identical to the POC host struct (sizeof == 76).
struct __attribute__((packed)) KernelArgs
{
    void*        ptr_D;        // s[2:3]   off 0x00
    void*        ptr_A;        // s[4:5]   off 0x08
    void*        ptr_B;        // s[6:7]   off 0x10
    void*        ptr_ScaleA;   // s[8:9]   off 0x18
    void*        ptr_ScaleB;   // s[10:11] off 0x20
    unsigned int stride_C;     // s12      off 0x28  (bytes)
    unsigned int stride_A;     // s13      off 0x2c  (bytes)
    unsigned int stride_B;     // s14      off 0x30  (bytes)
    unsigned int ScaleA_K;     // s15      off 0x34  (= K/32)
    unsigned int ScaleB_K;     // s16      off 0x38  (= K/32)
    unsigned int M;            // s17      off 0x3c
    unsigned int N;            // s18      off 0x40
    unsigned int K;            // s19      off 0x44
    unsigned int batch_size;   // s20      off 0x48
};
static_assert(sizeof(KernelArgs) == 76, "mxfp8fp4 preload KernelArgs must be 76B");

// Pick the best registered kernel variant for (M,N,K) given the B dtype and
// a_preshuffle.
static std::tuple<std::string, int> get_heuristic_kernel(
    int M, int N, int K, std::string arch_id, int b_intype, int a_preshuffle, CFG* cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu        = dev_prop.multiProcessorCount;
    uint32_t empty_cu      = num_cu;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0f;
    std::string selectedKernelName = "";

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.b_intype != b_intype || cfg.a_preshuffle != a_preshuffle)
            continue;

        int cl_x = cfg.cluster_x > 0 ? cfg.cluster_x : 1;
        int cl_y = cfg.cluster_y > 0 ? cfg.cluster_y : 1;

        // N is cluster-tiled with no partial-tile masking, so it must tile
        // exactly. M only needs to fill tile_m when it is clustered
        // (cluster_y > 1); a cluster_y == 1 tile lets the persistent scheduler
        // run a partial trailing M-tile bounded by the M kernarg (scale padded
        // to 32 rows), so small M selects the 64mx1_128nx4 tile.
        if((N % cfg.tile_n) != 0)
            continue;
        if(cl_y > 1 && (M % cfg.tile_m) != 0)
            continue;

        int tg_num_M = (M + cfg.tile_m - 1) / cfg.tile_m; // tiles in M (gdy)
        int tg_num_N = (N + cfg.tile_n - 1) / cfg.tile_n; // tiles in N (gdx)

        if((cl_x > 1 && (tg_num_N % cl_x) != 0) || (cl_y > 1 && (tg_num_M % cl_y) != 0))
            continue;

        uint32_t tg_num      = tg_num_M * tg_num_N;
        uint32_t local_round = (tg_num + num_cu - 1) / num_cu;

        float local_compute2mem_effi =
            (float)(cfg.tile_m * cfg.tile_n) / (cfg.tile_m + cfg.tile_n);

        bool is_earlier_round        = (local_round < round);
        bool is_same_round           = (local_round == round);
        bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
        bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);

        if(is_earlier_round ||
           (is_same_round && (has_sufficient_empty_cu || has_better_efficiency)))
        {
            round              = local_round;
            empty_cu           = local_round * num_cu - tg_num;
            compute2mem_effi   = local_compute2mem_effi;
            selectedKernelName = el.first;
        }
    }

    AITER_CHECK(selectedKernelName != "",
                __func__,
                ": cannot get heuristic kernel for b_intype=", b_intype,
                ", a_preshuffle=", a_preshuffle,
                ", M=", M, ", N=", N, ", K=", K,
                " (persistent/cluster tiles require N%tile_n==0 && M%tile_m==0 and "
                "cluster dims dividing the tile grid)");
    return std::make_tuple(selectedKernelName, 1);
}

// Shared dispatch body for both a8w8 (B=mxfp8) and a8w4 (B=mxfp4).
static void mxfp8fp4_mi400_launch(aiter_tensor_t* A,
                                  aiter_tensor_t* B,
                                  aiter_tensor_t* ScaleA,
                                  aiter_tensor_t* ScaleB,
                                  aiter_tensor_t* out,
                                  const char*     kernelName,
                                  int             b_intype,
                                  int             a_preshuffle,
                                  hipStream_t     stream)
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                __func__, " only supports BFloat16 output");
    AITER_CHECK(b_intype == B_DTYPE_FP8 || b_intype == B_DTYPE_FP4,
                __func__, " unsupported b_intype ", b_intype);
    AITER_CHECK(a_preshuffle == 0 || a_preshuffle == 1,
                __func__, " a_preshuffle must be 0 or 1");

    int Mdim = A->size(0);
    int Ndim = B->size(0);
    int Kdim = A->size(1); // A is mxfp8: 1 byte/elem, so col count == K

    AITER_CHECK(Kdim % K_ALIGN == 0,
                __func__, " K must be divisible by ", K_ALIGN, " (got K=", Kdim, ")");

    // Strides in bytes. A is fp8 (1 byte); B fp8 (1 byte) or fp4 (0.5 byte);
    // D is bf16 (2 bytes). Scales are e8m0, one per 32-K block.
    unsigned int stride_a = static_cast<unsigned int>(Kdim);
    unsigned int stride_b = (b_intype == B_DTYPE_FP4)
                                ? static_cast<unsigned int>(Kdim / 2)
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

    std::string arch_id = get_gpu_arch();
    std::string selectedName =
        (kernelName && kernelName[0] != '\0') ? (arch_id + kernelName) : "";

    using DictKey = std::tuple<int, int, int, int, int>; // M,N,K,b_intype,apre
    struct DictHash
    {
        size_t operator()(const DictKey& k) const
        {
            const auto& [m, n, kk, it, ap] = k;
            return std::hash<int>()(m) ^ std::hash<int>()(n) ^ std::hash<int>()(kk) ^
                   std::hash<int>()(it) ^ std::hash<int>()(ap);
        }
    };
    static SynchronizedCache<DictKey, std::string, DictHash> heuristic_kernel_dict;

    if(selectedName.empty())
    {
        selectedName = heuristic_kernel_dict.get_or_create(
            DictKey(Mdim, Ndim, Kdim, b_intype, a_preshuffle), [&]() {
                auto [name, _] = get_heuristic_kernel(
                    Mdim, Ndim, Kdim, arch_id, b_intype, a_preshuffle, config_map);
                return name;
            });
    }

    auto it = config_map->find(selectedName);
    AITER_CHECK(it != config_map->end(),
                __func__, " kernel not in cfg_mxfp8fp4gemm: ", selectedName);

    const auto& cfg = it->second;
    AITER_CHECK(cfg.b_intype == b_intype && cfg.a_preshuffle == a_preshuffle,
                __func__, " selected kernel ", selectedName,
                " mismatches requested b_intype/a_preshuffle");

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        cfg.knl_name, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    // ----- Launch geometry: cluster + persistent (see POC host) -----
    const int SUBM      = cfg.tile_m;
    const int SUBN      = cfg.tile_n;
    const int cluster_x = cfg.cluster_x > 0 ? cfg.cluster_x : 1;
    const int cluster_y = cfg.cluster_y > 0 ? cfg.cluster_y : 1;
    const int persistent = cfg.persistent;
    const int wg_max     = cfg.wg_max > 0 ? cfg.wg_max : 256;

    // Logical tile grid (also the launch grid in non-persistent mode).
    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;
    int gdz = 1;

    // Cluster dims must evenly tile the work grid (compile-time per .co).
    if(cluster_x > 1 || cluster_y > 1)
    {
        AITER_CHECK(gdx >= cluster_x && (gdx % cluster_x) == 0,
                    __func__, " cluster_x=", cluster_x, " requires N tiles (", gdx,
                    ") to be a multiple of it; N=", Ndim, " must be a multiple of ",
                    SUBN * cluster_x);
        AITER_CHECK(gdy >= cluster_y && (gdy % cluster_y) == 0,
                    __func__, " cluster_y=", cluster_y, " requires M tiles (", gdy,
                    ") to be a multiple of it; M=", Mdim, " must be a multiple of ",
                    SUBM * cluster_y);
    }

    if(persistent)
    {
        // 1D persistent launch of wg_max threadgroups along X; Y carries only the
        // cluster_y rows. GRID_X = wg_max / (cluster_x*cluster_y), GRID_Y = 1.
        const int cluster_size = cluster_x * cluster_y;
        AITER_CHECK(cluster_size > 0 && (wg_max % cluster_size) == 0,
                    __func__, " persistent wg_max=", wg_max,
                    " not divisible by cluster_x*cluster_y=", cluster_size);
        const int grid_x = wg_max / cluster_size;
        gdx = grid_x * cluster_x;
        gdy = 1 * cluster_y;
        gdz = 1;
    }

    const int bdx = 128; // 4 waves * 32 threads on gfx1250

    impl_ptr->launch_kernel(
        {&args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, stream, cluster_x, cluster_y, 1});
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp8_mxfp8_gemm_asm,
    (aiter_tensor_t* A,        // A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,        // B:[N, K]   mxfp8 e4m3 (always preshuffled)
     aiter_tensor_t* ScaleA,   // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB,   // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,      // Out:[M, N] bf16
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, stream))
{
    mxfp8fp4_mi400_launch(A, B, ScaleA, ScaleB, out,
                          kernelName, B_DTYPE_FP8, a_preshuffle, stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp8_mxfp4_gemm_asm,
    (aiter_tensor_t* A,        // A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,        // B:[N, K/2] mxfp4 e2m1 (always preshuffled)
     aiter_tensor_t* ScaleA,   // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB,   // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,      // Out:[M, N] bf16
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, stream))
{
    mxfp8fp4_mi400_launch(A, B, ScaleA, ScaleB, out,
                          kernelName, B_DTYPE_FP4, a_preshuffle, stream);
}
