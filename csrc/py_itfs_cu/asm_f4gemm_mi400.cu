// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 F4GEMM ASM dispatch (preload SGPR mode).
// Two entrypoints:
//   - mxfp4_gemm_asm: D[M,N] bf16 = A[M,K/2] mxfp4 * B[N,K/2] mxfp4 (e8m0 scales)
//   - nvfp4_gemm_asm: D[M,N] bf16 = A[M,K/2] nvfp4 * B[N,K/2] nvfp4 (e4m3 scales + GlobalScale)
//
// KernelArgs uses the ROCm kernarg-preload layout (sgpr_mode==1): pointers
// first (dw 0..9, MEM-first), then 4B-tight scalars. Bytes shipped to HW:
//   MXFP4: 80B (struct minus the 2 trailing persistent log2 dwords)
//   NVFP4: 88B (full struct incl. GlobalScaleA/B + trailing log2)
//
// Launch is cluster- and persistent-aware:
//   - cluster_x/cluster_y are compile-time per .co and read from the CSV; a
//     dim > 1 launches via hipDrvLaunchKernelEx with a cluster-dim attribute.
//   - persistent dispatch is hardcoded on (persistent_tg=256, grid_y=4; both
//     runtime-only knobs that don't affect the .co). The host ships
//     log2(gridX)/log2(gridY) so the persistent shader can walk the cluster
//     grid. NVFP4 carries them at dw20/21, MXFP4 reuses dw18/19.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_f4gemm_configs.hpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <hip/hip_runtime.h>

constexpr int F4_INTYPE_MXFP4 = 7;
constexpr int F4_INTYPE_NVFP4 = 8;
constexpr int MXFP4_SCALE_BLOCK = 32;
constexpr int NVFP4_SCALE_BLOCK = 16;

// Preload-mode KernelArgs (4B-tight, MEM-first). Offsets in comments are the
// kernarg byte offsets the preload-aware shader s_load's from.
struct __attribute__((packed)) KernelArgs
{
    void*        ptr_D;            // dw 0..1   (off 0x00)
    void*        ptr_A;           // dw 2..3   (off 0x08)
    void*        ptr_B;           // dw 4..5   (off 0x10)
    void*        ptr_ScaleA;       // dw 6..7   (off 0x18)
    void*        ptr_ScaleB;       // dw 8..9   (off 0x20)
    unsigned int strideD0;         // dw 10     (off 0x28)
    unsigned int strideA0;         // dw 11     (off 0x2C)
    unsigned int strideB0;        // dw 12     (off 0x30)
    unsigned int ScaleA_stride0;   // dw 13     (off 0x34)
    unsigned int ScaleB_stride0;  // dw 14     (off 0x38)
    unsigned int M;                // dw 15     (off 0x3C)
    unsigned int N;                // dw 16     (off 0x40)
    unsigned int K;                // dw 17     (off 0x44)
    float        GlobalScaleA;     // dw 18     (off 0x48) NVFP4 only
    float        GlobalScaleB;     // dw 19     (off 0x4C) NVFP4 only
    unsigned int log2_grid_x;      // dw 20     persistent only (unused -> 0)
    unsigned int log2_grid_y;      // dw 21     persistent only (unused -> 0)
};
// 5 ptrs (40B) + 8 scalars (32B) + GlobalScaleA/B (8B) + 2 log2 (8B) = 88B.
static_assert(sizeof(KernelArgs) == 88, "f4gemm preload KernelArgs must be 88B");

static std::tuple<std::string, int> get_heuristic_kernel(
    int M, int N, int K, std::string arch_id, int intype, int a_preshuffle, CFG* cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu        = dev_prop.multiProcessorCount;
    uint32_t empty_cu      = num_cu;
    uint32_t tg_num        = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0f;
    std::string selectedKernelName = "";

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.intype != intype || cfg.a_preshuffle != a_preshuffle)
            continue;
        // Persistent/cluster shaders don't mask partial tiles, so the problem
        // must tile both dims exactly.
        if((N % cfg.tile_n) != 0 || (M % cfg.tile_m) != 0)
            continue;

        int tg_num_M         = (M + cfg.tile_m - 1) / cfg.tile_m;  // tiles in M (gdy)
        int tg_num_N         = (N + cfg.tile_n - 1) / cfg.tile_n;  // tiles in N (gdx)

        // Cluster dims (compile-time per .co, declared in the CSV) must evenly
        // divide the tile grid; otherwise this variant can't run this shape.
        int cl_x = cfg.cluster_x > 0 ? cfg.cluster_x : 1;
        int cl_y = cfg.cluster_y > 0 ? cfg.cluster_y : 1;
        if((cl_x > 1 && (tg_num_N % cl_x) != 0) || (cl_y > 1 && (tg_num_M % cl_y) != 0))
            continue;

        tg_num               = tg_num_M * tg_num_N;
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
                ": cannot get heuristic kernel for intype=",
                intype,
                ", a_preshuffle=",
                a_preshuffle,
                ", M=",
                M,
                ", N=",
                N,
                ", K=",
                K);
    return std::make_tuple(selectedKernelName, 1);
}

// Shared dispatch body. NVFP4 ships the full preload struct (88B) with real
// GlobalScale floats; MXFP4 leaves GlobalScale unset and ships 80B (dropping
// the trailing persistent log2 dword pair).
static void f4gemm_mi400_launch(aiter_tensor_t* A,
                                aiter_tensor_t* B,
                                aiter_tensor_t* ScaleA,
                                aiter_tensor_t* ScaleB,
                                aiter_tensor_t* out,
                                const char*     kernelName,
                                int             intype,
                                int             a_preshuffle,
                                float           GlobalScaleA,
                                float           GlobalScaleB,
                                hipStream_t     stream)
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                __func__,
                " only supports BFloat16 output");
    AITER_CHECK(intype == F4_INTYPE_MXFP4 || intype == F4_INTYPE_NVFP4,
                __func__,
                " unsupported intype ",
                intype);
    AITER_CHECK(a_preshuffle == 0 || a_preshuffle == 1,
                __func__,
                " a_preshuffle must be 0 or 1");

    int Mdim = A->size(0);
    int Ndim = B->size(0);
    int Kdim = A->size(1) * 2; // packed fp4: stored dim = K/2 bytes

    int scale_block = (intype == F4_INTYPE_NVFP4) ? NVFP4_SCALE_BLOCK : MXFP4_SCALE_BLOCK;
    AITER_CHECK(Kdim % scale_block == 0,
                __func__,
                " K must be divisible by scale block size (",
                scale_block,
                ")");

    // Strides in bytes.
    unsigned int stride_a = static_cast<unsigned int>(Kdim / 2);     // fp4 packed
    unsigned int stride_b = static_cast<unsigned int>(Kdim / 2);     // fp4 packed
    unsigned int stride_d = static_cast<unsigned int>(Ndim) * 2;     // bf16
    unsigned int stride_sa = static_cast<unsigned int>(Kdim / scale_block);
    unsigned int stride_sb = static_cast<unsigned int>(Kdim / scale_block);

    KernelArgs args{};                       // zero-init; log2_grid_x/y set below if persistent
    args.ptr_D           = out->ptr;
    args.ptr_A           = A->ptr;
    args.ptr_B           = B->ptr;
    args.ptr_ScaleA      = ScaleA->ptr;
    args.ptr_ScaleB      = ScaleB->ptr;
    args.strideD0        = stride_d;
    args.strideA0        = stride_a;
    args.strideB0        = stride_b;
    args.ScaleA_stride0  = stride_sa;
    args.ScaleB_stride0  = stride_sb;
    args.M               = Mdim;
    args.N               = Ndim;
    args.K               = Kdim;
    if(intype == F4_INTYPE_NVFP4)
    {
        args.GlobalScaleA = GlobalScaleA;
        args.GlobalScaleB = GlobalScaleB;
    }

    // Bytes shipped to HW:
    //   NVFP4: full struct (5 ptrs + 8 scalars + GlobalScaleA/B + 2 log2) = 88B
    //   MXFP4: drop the 2 trailing persistent log2 dwords                 = 80B
    size_t arg_size = (intype == F4_INTYPE_NVFP4)
                          ? sizeof(KernelArgs)
                          : (sizeof(KernelArgs) - 2 * sizeof(unsigned int));

    const HipDeviceGuard device_guard(A->device_id);

    static CFG* config_map = &cfg_f4gemm;
    AITER_CHECK(!config_map->empty(),
                __func__,
                " no kernel registered for f4gemm; check AITER_GPU_ARCHS=gfx1250");

    std::string arch_id = get_gpu_arch();
    std::string selectedName =
        (kernelName && kernelName[0] != '\0') ? (arch_id + kernelName) : "";

    using DictKey = std::tuple<int, int, int, int, int>; // M,N,K,intype,apre
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
            DictKey(Mdim, Ndim, Kdim, intype, a_preshuffle), [&]() {
                auto [name, _] = get_heuristic_kernel(
                    Mdim, Ndim, Kdim, arch_id, intype, a_preshuffle, config_map);
                return name;
            });
    }

    auto it = config_map->find(selectedName);
    AITER_CHECK(it != config_map->end(),
                __func__,
                " kernel not in cfg_f4gemm: ",
                selectedName);

    const auto& cfg     = it->second;
    AITER_CHECK(cfg.intype == intype && cfg.a_preshuffle == a_preshuffle,
                __func__,
                " selected kernel ",
                selectedName,
                " mismatches requested intype/a_preshuffle");

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        cfg.knl_name, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    // ----- Launch geometry: cluster + persistent -----
    const int SUBM      = cfg.tile_m;
    const int SUBN      = cfg.tile_n;
    const int cluster_x = cfg.cluster_x > 0 ? cfg.cluster_x : 1;  // compile-time per .co (CSV)
    const int cluster_y = cfg.cluster_y > 0 ? cfg.cluster_y : 1;

    // Persistent dispatch is hardcoded on: all f4gemm_mi400 .co are persistent
    // shaders. persistent_tg / grid_y are runtime-only knobs (don't affect the
    // .co), so they're fixed here at the default; gridX is derived.
    constexpr int PERSISTENT    = 1;
    constexpr int PERSISTENT_TG = 256; // total threadgroups (pow2 * cluster count)
    constexpr int PERSISTENT_GY = 4;   // cluster-grid Y dim (M dir); gridX derived

    const int tiles_x = Ndim / SUBN;   // N-direction output tiles (exact: see heuristic)
    const int tiles_y = Mdim / SUBM;   // M-direction output tiles (exact: see heuristic)

    // Cluster dims must evenly tile the work grid (cluster is compile-time per .co).
    if(cluster_x > 1 || cluster_y > 1)
    {
        AITER_CHECK(tiles_x >= cluster_x && (tiles_x % cluster_x) == 0,
                    __func__, " cluster_x=", cluster_x, " requires N tiles (", tiles_x,
                    ") to be a multiple of it; N=", Ndim, " must be a multiple of ",
                    SUBN * cluster_x);
        AITER_CHECK(tiles_y >= cluster_y && (tiles_y % cluster_y) == 0,
                    __func__, " cluster_y=", cluster_y, " requires M tiles (", tiles_y,
                    ") to be a multiple of it; M=", Mdim, " must be a multiple of ",
                    SUBM * cluster_y);
    }

    int          gdx         = tiles_x;
    int          gdy         = tiles_y;
    int          gdz         = 1;
    unsigned int log2_grid_x = 0;
    unsigned int log2_grid_y = 0;

    if(PERSISTENT)
    {
        AITER_CHECK((PERSISTENT_TG % (cluster_x * cluster_y)) == 0,
                    __func__, " persistent_tg=", PERSISTENT_TG,
                    " not divisible by cluster_x*cluster_y=", cluster_x * cluster_y);
        const int clusters = PERSISTENT_TG / (cluster_x * cluster_y);
        AITER_CHECK(PERSISTENT_GY != 0 && (clusters % PERSISTENT_GY) == 0,
                    __func__, " grid_y=", PERSISTENT_GY,
                    " must be a nonzero divisor of cluster count ", clusters);
        const int gridY = PERSISTENT_GY;
        const int gridX = clusters / gridY;
        // grid_flat advance = 1 << (log2_grid_x + log2_grid_y) must equal the
        // cluster count, so cluster count and both grid dims must be power-of-two.
        AITER_CHECK((clusters & (clusters - 1)) == 0,
                    __func__, " persistent cluster count ", clusters, " must be power-of-two");
        AITER_CHECK((gridX & (gridX - 1)) == 0 && (gridY & (gridY - 1)) == 0,
                    __func__, " persistent gridX=", gridX, " gridY=", gridY,
                    " must each be power-of-two");

        // HIP gridDim must be a multiple of clusterDim per axis: the cluster grid
        // scaled by the cluster dims.
        gdx = gridX * cluster_x;
        gdy = gridY * cluster_y;
        gdz = 1;

        for(int g = gridX; g > 1; g >>= 1)
            log2_grid_x++;
        for(int g = gridY; g > 1; g >>= 1)
            log2_grid_y++;

        // Persistent shader reads log2(gridX)/log2(gridY). NVFP4 ships them at
        // dw20/21; MXFP4 has no GlobalScale so the shader reads them from the
        // GlobalScale slots (dw18/19).
        if(intype == F4_INTYPE_NVFP4)
        {
            args.log2_grid_x = log2_grid_x;
            args.log2_grid_y = log2_grid_y;
        }
        else
        {
            std::memcpy(&args.GlobalScaleA, &log2_grid_x, sizeof(unsigned int));
            std::memcpy(&args.GlobalScaleB, &log2_grid_y, sizeof(unsigned int));
        }
    }

    const int bdx = 128; // 4 wave * 32 thread on gfx1250

    impl_ptr->launch_kernel(
        {&args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, stream, cluster_x, cluster_y, 1});
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp4_gemm_asm,
    (aiter_tensor_t* A,        // A:[M, K/2] fp4x2 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,        // B:[N, K/2] fp4x2 (always preshuffled)
     aiter_tensor_t* ScaleA,   // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB,   // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,      // Out:[M, N] bf16
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, stream))
{
    f4gemm_mi400_launch(A, B, ScaleA, ScaleB, out,
                        kernelName, F4_INTYPE_MXFP4, a_preshuffle,
                        0.0f, 0.0f, stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    nvfp4_gemm_asm,
    (aiter_tensor_t* A,        // A:[M, K/2] fp4x2 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,        // B:[N, K/2] fp4x2 (always preshuffled)
     aiter_tensor_t* ScaleA,   // ScaleA:[M, K/32] e4m3 (shuffled)
     aiter_tensor_t* ScaleB,   // ScaleB:[N, K/32] e4m3 (shuffled)
     float           GlobalScaleA,
     float           GlobalScaleB,
     aiter_tensor_t* out,      // Out:[M, N] bf16
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, GlobalScaleA, GlobalScaleB,
     out, kernelName, a_preshuffle, stream))
{
    f4gemm_mi400_launch(A, B, ScaleA, ScaleB, out,
                        kernelName, F4_INTYPE_NVFP4, a_preshuffle,
                        GlobalScaleA, GlobalScaleB, stream);
}
